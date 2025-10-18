# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import contextlib
import functools
import logging
import multiprocessing.resource_sharer as mp_resource_sharer
import os
import signal
import socket
import sys
import tempfile
import time
import traceback
from typing import Dict, Mapping, Optional

import torch
import torch.multiprocessing as mp

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

from ..shared_utils.health_check import GPUHealthCheck, NicHealthCheck
from .config import FaultToleranceConfig
from .data import (
    AuthkeyMsg,
    ErrorMsg,
    HeartbeatMsg,
    InitMsg,
    OkMsg,
    SectionAction,
    SectionMsg,
    UpdateConfigMsg,
)
from .rank_monitor_state_machine import RankMonitorStateMachine
from .utils import is_process_alive, read_obj_from_ipc_stream, write_obj_to_ipc_stream


class RankMonitorLogger(logging.Logger):
    """Logger used in a rank monitor process"""

    def __init__(
        self,
        name="RankMonServer",
        level=logging.INFO,
        connected_rank=None,
        is_restarter_logger=False,
    ):
        super().__init__(name, level)
        self.is_restarter_logger = is_restarter_logger
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self._setup_logger()
        self._create_restarter_sublogger()
        self.set_connected_rank(connected_rank)

    def set_connected_rank(self, rank: Optional[int]):
        self.rank = rank
        rank_str = f"{self.rank}" if self.rank is not None else "<none>"
        name_fmt = f"{self.name}{self.pid}@{self.hostname}:{rank_str}"
        for handler in self.handlers:
            formatter = logging.Formatter(f"[%(asctime)s] [%(levelname)s] [{name_fmt}] %(message)s")
            handler.setFormatter(formatter)

    def log_for_restarter(self, message, *args, **kwargs):
        self.restarter_logger.log(logging.DEBUG, message, *args, **kwargs)

    def _setup_logger(self):
        self.setLevel(self.level)
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(self.level)
        self.addHandler(ch)
        self.propagate = False

    def _create_restarter_sublogger(self):
        self.restarter_logger = logging.getLogger(f"{self.name}.Restarter")
        self.restarter_logger.setLevel(
            logging.DEBUG if self.is_restarter_logger else logging.CRITICAL
        )
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(asctime)s] [ALWAYS] [%(name)s] %(message)s")
        stdout_handler.setFormatter(formatter)
        self.restarter_logger.addHandler(stdout_handler)
        self.restarter_logger.propagate = False
        return self.restarter_logger

    def log_restarter_event(self, message, *args, **kwargs):
        """
        Log a restart event that should always be visible, but only if restarter logging is enabled.
        Args:
            is_restarter_logger: Whether restarter logging is enabled
            message: The message to log
            *args, **kwargs: Additional arguments for logging
        """
        if self.is_restarter_logger:
            self.log_for_restarter(message, *args, **kwargs)


class RankMonitorServer:
    """
    RankMonitorServer, running in a separate process, is responsible for monitoring the ranks.
    RankMonitorClient is intialized in each rank and is used to communicate with the RankMonitorServer.
    """

    """
    How much time to wait for the rank monitor process to initialize.
    Default 3 minutes should be enough, even on slowest machines.
    """
    RANK_MONITOR_INIT_TIMEOUT = 180.0

    def __init__(
        self,
        cfg: FaultToleranceConfig,
        ipc_socket_path: str,
        rank_monitor_ready_event,
        logger: logging.Logger,
        is_restarter_logger: bool,
    ):
        """
        Initializes the RankMonitorServer object.

        Args:
            cfg (FaultToleranceConfig): The configuration object for fault tolerance.
            ipc_socket_path (str): Path of the IPC socket connecting this monitor with its rank
            rank_monitor_ready_event (mp.Event): The event indicating that the rank monitor is ready.
            is_restarter_logger (bool): True if this monitor writes state transition logs
            logger (logging.Logger): The logger object for logging.

        """
        self.cfg = cfg
        self.ipc_socket_path = ipc_socket_path
        self.rank_info = None
        self.start_time = None
        self.last_hb_time = None
        self.out_of_section_time = None
        self.open_sections: Mapping[str, float] = dict()  # section name -> start time
        self.is_closing = None
        self.server = None
        self.connection_lock = asyncio.Lock()
        self.rank_monitor_ready_event = rank_monitor_ready_event
        self.logger = logger
        self.rmlogger = RankMonitorLogger(
            level=logger.level, is_restarter_logger=is_restarter_logger
        )
        self.state_machine = RankMonitorStateMachine(self.rmlogger)
        self._periodic_restart_task = None
        self._restart_check_log_count = 0  # Counter for "Started periodic restart check" logs
        self.health_checker = GPUHealthCheck(
            interval=self.cfg.node_health_check_interval, on_failure=self._handle_unhealthy_node
        )
        self.current_writer = None
        self.launcher_ipc_socket_path = (
            f"{tempfile.gettempdir()}/_ft_launcher{os.getpid()}_to_rmon.socket"
        )
        self.launcher_server = None

        if self.cfg.enable_nic_monitor:
            self.logger.info("Enable NIC health monitoring.")
            self.nic_health_checker = NicHealthCheck(
                interval=self.cfg.node_health_check_interval,
                pci_topo_file=self.cfg.pci_topo_file,
                link_down_path_template=self.cfg.link_down_path_template,
                on_failure=self._handle_unhealthy_nic,
            )
        else:
            self.nic_health_checker = None

    def start_periodic_restart_check(self):
        if self._periodic_restart_task and not self._periodic_restart_task.done():
            self.logger.warning("Periodic restart check is already running.")
            return
        self._periodic_restart_task = asyncio.get_running_loop().create_task(
            self._periodic_restart_check()
        )
        # Only log for local rank 0 to reduce log spam, and limit to 2 times (first 2 minutes)
        if (
            self.rank_info is None or self.rank_info.local_rank == 0
        ) and self._restart_check_log_count < 2:
            self.logger.info("Started periodic restart check.")
            self._restart_check_log_count += 1

    async def stop_periodic_restart_check(self):
        if self._periodic_restart_task:
            self._periodic_restart_task.cancel()
            try:
                await self._periodic_restart_task
            except asyncio.CancelledError:
                self.logger.debug("Periodic restart check task cancelled.")
            self._periodic_restart_task = None
            # Reset counter when restart completes successfully
            self._restart_check_log_count = 0

    def _shutdown_rank(self):
        # First sends SIGCONT to wake up the process, then "rank_termination_signal" to terminate it
        try:
            sig = self.cfg.rank_termination_signal
            rank_pid = self.rank_info.pid
            os.kill(rank_pid, signal.SIGCONT)
            os.kill(rank_pid, sig)
            self.logger.debug(
                f"Rank monitor sent SIGCONT then {sig.name} to the rank (PID={rank_pid})"
            )
        except Exception as e:
            self.logger.error(
                f"Rank monitor could not send {sig.name} to the rank (PID={rank_pid}). Exception={e}"
            )
            # do nothing, most probably rank process is terminated anyway

    def _shutdown_rank_if_alive(self):
        if self.rank_info is None:
            return

        try:
            if is_process_alive(self.rank_info.pid):
                self._shutdown_rank()
            else:
                self.logger.debug(
                    f"_shutdown_rank_if_alive: rank process {self.rank_info.pid} is dead."
                )
        except Exception:
            self.logger.error("_shutdown_rank_if_alive exception! ", exc_info=True)

    async def _handle_unhealthy_node(self):
        self.logger.debug("In RankMonitorServer._handle_unhealthy_node.")
        self._shutdown_rank_if_alive()

    async def _handle_unhealthy_nic(self):
        self.logger.debug("In RankMonitorServer._handle_unhealthy_nic.")
        self._shutdown_rank_if_alive()

    async def _handle_timeout(self):
        self.logger.debug("In RankMonitorServer._handle_timeout.")
        self._shutdown_rank_if_alive()

    async def _handle_authkey_msg(self, msg, writer):
        # resource sharer needs to be restarted to use updated authkey
        # after we stop it, it will be started automatically.
        # TODO: are there any side effects?
        mp_resource_sharer.stop()
        # this process authkey need to be set to the same authkey as in client
        # otherwise we wont be able to receive pickled Manager, Tensors etc.
        mp.current_process().authkey = msg.authkey
        await write_obj_to_ipc_stream(OkMsg(), writer)

    async def _handle_update_config_msg(self, msg, writer):
        # Reseived when new timeouts are computed by the client
        if msg.hb_timeouts is not None:
            self.cfg.initial_rank_heartbeat_timeout = msg.hb_timeouts.initial
            self.cfg.rank_heartbeat_timeout = msg.hb_timeouts.subsequent
            self.logger.debug(f"Updated heartbeat timeouts: {msg.hb_timeouts}")
        if msg.section_timeouts is not None:
            self.cfg.rank_section_timeouts = msg.section_timeouts.section
            self.cfg.rank_out_of_section_timeout = msg.section_timeouts.out_of_section
            self.logger.debug(f"Updated section timeouts: {msg.section_timeouts}")
        await write_obj_to_ipc_stream(OkMsg(), writer)

    async def _handle_init_msg(self, msg, writer):
        self.rank_info = msg.rank_info
        self.start_time = time.monotonic()
        self.out_of_section_time = time.monotonic()
        self.open_sections.clear()
        self.last_hb_time = None
        # Update NIC health checker on the rank to monitor.
        if self.nic_health_checker is not None:
            self.nic_health_checker.set_nic_device(local_rank=self.rank_info.local_rank)
        self.rmlogger.set_connected_rank(msg.rank_info.global_rank)
        await write_obj_to_ipc_stream(OkMsg(cfg=self.cfg), writer)

    async def _handle_heartbeat_msg(self, msg, writer):
        if self._periodic_restart_task is not None:
            await self.stop_periodic_restart_check()
        self.state_machine.handle_heartbeat_msg()
        self.last_hb_time = time.monotonic()
        assert not msg.state, "state in heartbeat is not supported in this version"

        # Only send response if skip_section_response is disabled
        # When enabled, client doesn't expect response (unidirectional)
        if not self.cfg.skip_section_response:
            await write_obj_to_ipc_stream(OkMsg(), writer)

    async def _handle_section_msg(self, msg: SectionMsg, writer):
        if self._periodic_restart_task is not None:
            await self.stop_periodic_restart_check()
        self.state_machine.handle_section_msg()
        resp = ErrorMsg()
        current_time = time.monotonic()
        if msg.action is SectionAction.OPEN:
            if msg.section not in self.open_sections:
                self.open_sections[msg.section] = current_time
                self.out_of_section_time = None
                resp = OkMsg()
            else:
                # Log error but don't send response when skip_section_response is enabled
                error_msg = f"Section '{msg.section}' is already open."
                if self.cfg.skip_section_response:
                    self.logger.error(f"Section error (not sent to client): {error_msg}")
                else:
                    resp = ErrorMsg(cause=error_msg)
        elif msg.action is SectionAction.CLOSE:
            if msg.section in self.open_sections:
                del self.open_sections[msg.section]
                if not self.open_sections:
                    self.out_of_section_time = current_time
                resp = OkMsg()
            else:
                # Log error but don't send response when skip_section_response is enabled
                error_msg = f"Section '{msg.section}' is not open."
                if self.cfg.skip_section_response:
                    self.logger.error(f"Section error (not sent to client): {error_msg}")
                else:
                    resp = ErrorMsg(cause=error_msg)
        elif msg.action is SectionAction.CLOSE_ALL:
            self.open_sections.clear()
            self.out_of_section_time = current_time
            resp = OkMsg()
        else:
            raise AssertionError(f"Unknown SectionAction: {msg.action}")

        # Only send response if skip_section_response is disabled
        # When enabled, client doesn't expect ANY response (true unidirectional)
        if not self.cfg.skip_section_response:
            await write_obj_to_ipc_stream(resp, writer)

    def _handle_ipc_connection_lost(self):
        self.state_machine.handle_ipc_connection_lost()
        if self.state_machine.is_restarting():
            self.start_periodic_restart_check()
        self.rank_info = None
        self.start_time = None
        self.last_hb_time = None
        self.out_of_section_time = None
        if self.open_sections:
            open_section_names = list(self.open_sections.keys())
            self.logger.warning(
                f"Section(s) {open_section_names} were still open. you can use`.end_all_sections` to avoid this warning"
            )
            self.open_sections.clear()
        self.rmlogger.set_connected_rank(None)
        if self.connection_lock.locked():
            self.connection_lock.release()

    async def _handle_launcher_ipc_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle IPC connection from the launcher."""
        try:
            while True:
                msg = await read_obj_from_ipc_stream(reader)
                if msg == "close_worker_ipc_connection":
                    self.logger.debug(
                        "Received request from launcher to close worker IPC connection"
                    )
                    await self.close_current_connection()
                else:
                    self.logger.warning(f"Received unknown message from launcher: {msg}")
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError, EOFError):
            # Valid stream close exceptions.
            pass
        except Exception as e:
            self.logger.error(f"Error handling launcher IPC connection: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _handle_ipc_connection(self, reader, writer):
        if not self.is_closing:
            try:
                if self.connection_lock.locked():
                    self.logger.warning(
                        "Got a new connection while previous is still active. Need to wait..."
                    )
                await asyncio.wait_for(
                    self.connection_lock.acquire(),
                    RankMonitorServer.RANK_MONITOR_INIT_TIMEOUT,
                )
                self.current_writer = writer
                while msg := await read_obj_from_ipc_stream(reader):
                    if isinstance(msg, AuthkeyMsg):
                        await self._handle_authkey_msg(msg, writer)
                    elif isinstance(msg, InitMsg):
                        await self._handle_init_msg(msg, writer)
                    elif isinstance(msg, HeartbeatMsg):
                        await self._handle_heartbeat_msg(msg, writer)
                    elif isinstance(msg, UpdateConfigMsg):
                        await self._handle_update_config_msg(msg, writer)
                    elif isinstance(msg, SectionMsg):
                        await self._handle_section_msg(msg, writer)
                    else:
                        raise AssertionError(f"Unknown msg type: {type(msg)}")
            except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError, EOFError):
                # Valid stream close exceptions.
                pass
            except Exception:
                self.logger.error(f"Exception in _handle_ipc_connection: {traceback.format_exc()}")
            finally:
                self.current_writer = None
                self._handle_ipc_connection_lost()
                assert not self.connection_lock.locked()
        else:
            self.logger.warning("Ignored incoming connection: service is shutting down.")

        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()
        self.logger.debug("Leaving _handle_ipc_connection. ")

    def _handle_signal(self, sig):
        self.state_machine.handle_signal()
        self.logger.debug(f"Received signal: {sig}")
        if not self.is_closing:
            self.is_closing = True
            self.server.close()
            self.launcher_server.close()

    def _is_hb_timeout_elapsed(self, curr_time) -> bool:
        is_elapsed = False
        if self.last_hb_time is None:
            # has not got any heartbeats yet
            time_since_start = curr_time - self.start_time
            timeout = self.cfg.initial_rank_heartbeat_timeout
            is_elapsed = timeout is not None and time_since_start > timeout
            if is_elapsed:
                self.logger.warning(f"Did not get initial heartbeat. Waited {timeout:.2f} seconds.")
        else:
            # did get some heartbeats
            time_since_last_hb = curr_time - self.last_hb_time
            timeout = self.cfg.rank_heartbeat_timeout
            is_elapsed = timeout is not None and time_since_last_hb > timeout
            if is_elapsed:
                self.logger.warning(
                    f"Did not get subsequent heartbeat. " f"Waited {timeout:.2f} seconds."
                )
        return is_elapsed

    def _is_section_timeout_elapsed(self, curr_time) -> bool:
        is_elapsed = False
        # If any sections are open, check their timeouts,
        # otherwise check the timeout for "out of section"
        for section, section_start_time in self.open_sections.items():
            elapsed = curr_time - section_start_time
            timeout = self.cfg.rank_section_timeouts[section]
            is_elapsed = timeout is not None and elapsed > timeout
            if is_elapsed:
                self.logger.warning(
                    f"Section '{section}' has been open for {elapsed:.2f} seconds. "
                    f"Timeout is {timeout:.2f} seconds."
                )
                is_elapsed = True
        if not self.open_sections:
            elapsed = curr_time - self.out_of_section_time
            timeout = self.cfg.rank_out_of_section_timeout
            is_elapsed = timeout is not None and elapsed > timeout
            if is_elapsed:
                self.logger.warning(
                    f"Was out of section for {elapsed:.2f} seconds. "
                    f"Timeout is {timeout:.2f} seconds."
                )
        return is_elapsed

    async def _periodic_rank_check(self):
        while True:
            if self.rank_info is not None:
                curr_time = time.monotonic()
                hb_timed_out = self._is_hb_timeout_elapsed(curr_time)
                section_timed_out = self._is_section_timeout_elapsed(curr_time)
                if hb_timed_out or section_timed_out:
                    await self._handle_timeout()
            await asyncio.sleep(self.cfg.workload_check_interval)

    async def _periodic_restart_check(self):
        await asyncio.sleep(self.cfg.restart_check_interval)
        while True:
            self.state_machine.periodic_restart_check()
            await asyncio.sleep(self.cfg.restart_check_interval)

    async def _periodic_node_health_check(self):
        await self.health_checker.async_check()

    async def _periodic_nic_health_check(self):
        await self.nic_health_checker.async_check()

    async def _rank_monitor_loop(self):
        # Handle usual termination signals
        for sig_to_handle in [
            signal.SIGTERM,
            signal.SIGINT,
            signal.SIGUSR1,
            signal.SIGUSR2,
        ]:
            asyncio.get_event_loop().add_signal_handler(
                sig_to_handle,
                functools.partial(self._handle_signal, sig_to_handle),
            )

        # Run local server
        if os.path.exists(self.ipc_socket_path):
            os.unlink(self.ipc_socket_path)
        self.server = await asyncio.start_unix_server(
            self._handle_ipc_connection,
            self.ipc_socket_path,
        )

        # Start launcher IPC server
        if os.path.exists(self.launcher_ipc_socket_path):
            os.unlink(self.launcher_ipc_socket_path)
        self.launcher_server = await asyncio.start_unix_server(
            self._handle_launcher_ipc_connection,
            self.launcher_ipc_socket_path,
        )

        # Periodic checks
        asyncio.get_running_loop().create_task(self._periodic_rank_check())

        # Periodic node health check
        asyncio.get_running_loop().create_task(self._periodic_node_health_check())

        # Periodic nic health check
        if self.nic_health_checker is not None:
            asyncio.get_running_loop().create_task(self._periodic_nic_health_check())

        self.rank_monitor_ready_event.set()

        try:
            async with self.server, self.launcher_server:
                await asyncio.gather(
                    self.server.serve_forever(), self.launcher_server.serve_forever()
                )
        except asyncio.exceptions.CancelledError:
            self.logger.debug("server.serve_forever() cancelled.")
        finally:
            with contextlib.suppress(Exception):
                os.unlink(self.ipc_socket_path)
                os.unlink(self.launcher_ipc_socket_path)

    @staticmethod
    def run(
        cfg: FaultToleranceConfig,
        ipc_socket_path: str,
        rank_monitor_ready_event,
        is_restarter_logger: bool,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        # Set environment variables if provided
        if env is not None:
            for key, value in env.items():
                os.environ[key] = value

        # Set up the nvrx logger for subprocess
        # Use force_reset=True to ensure fresh logger setup with correct rank info
        from nvidia_resiliency_ext.shared_utils.log_manager import setup_logger

        try:
            setup_logger(force_reset=True, node_local_tmp_prefix="rankmonsvr")
            logger = logging.getLogger(LogConfig.name)

            logger.debug(f"Starting RankMonitorServer... PID={os.getpid()}")
            inst = RankMonitorServer(
                cfg,
                ipc_socket_path,
                rank_monitor_ready_event,
                logger,
                is_restarter_logger,
            )
            asyncio.run(inst._rank_monitor_loop())
            logger.debug("Leaving RankMonitorServer process")
        except asyncio.exceptions.CancelledError:
            logger.debug("asyncio.exceptions.CancelledError in RankMonitorServer.run")
            pass  # Ignore the exception
        except Exception as e:
            logger.error(f"Exception caught in RankMonitorServer.run: {e}")
            pass  # Ignore the exception

    @staticmethod
    def run_in_subprocess(
        cfg,
        ipc_socket_path: str,
        is_restarter_logger: bool = False,
        mp_ctx=torch.multiprocessing,
        env=None,
    ):
        rank_monitor_ready_event = mp_ctx.Event()

        rank_monitor_process_kwargs = {
            "cfg": cfg,
            "ipc_socket_path": ipc_socket_path,
            "rank_monitor_ready_event": rank_monitor_ready_event,
            "is_restarter_logger": is_restarter_logger,
            "env": env,
        }

        rank_monitor_process = mp_ctx.Process(
            target=RankMonitorServer.run, kwargs=rank_monitor_process_kwargs
        )

        rank_monitor_process.daemon = True
        rank_monitor_process.start()

        if not rank_monitor_ready_event.wait(timeout=RankMonitorServer.RANK_MONITOR_INIT_TIMEOUT):
            raise RuntimeError(
                f"Could not start rank monitor. Waited {RankMonitorServer.RANK_MONITOR_INIT_TIMEOUT} sec."
            )

        return rank_monitor_process

    async def close_current_connection(self):
        if self.current_writer is not None:
            self.current_writer.close()
            await self.current_writer.wait_closed()
            self.logger.debug("Closed current IPC connection")
        else:
            self.logger.debug("No active connection to close")
