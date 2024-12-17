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
import tempfile
import time
from typing import Any, Mapping, Optional

import torch
import torch.multiprocessing as mp

from .config import FaultToleranceConfig
from .data import AuthkeyMsg, HeartbeatMsg, InitMsg, OkMsg, UpdateConfigMsg
from .utils import (
    create_logger,
    is_process_alive,
    read_obj_from_ipc_stream,
    write_obj_to_ipc_stream,
)


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
        parent_rank: int,
        rank_monitor_ready_event,
        logger: logging.Logger,
    ):
        """
        Initializes the RankMonitorServer object.

        Args:
            cfg (FaultToleranceConfig): The configuration object for fault tolerance.
            parent_rank (int): which rank is being monitored by this RankMonitorServer instance.
            rank_monitor_ready_event (mp.Event): The event indicating that the rank monitor is ready.
            logger (Logger.Logger): The logger object for logging.

        """
        self.cfg = cfg
        self.parent_rank = parent_rank
        self.ipc_socket_path = RankMonitorServer.get_ipc_socket_path(parent_rank)
        self.state_dict_for_chkpt: Mapping[str, Any] = dict()
        self.rank_info = None
        self.start_time = None
        self.last_hb_time = None
        self.is_closing = None
        self.server = None
        self.connection_lock = asyncio.Lock()
        self.rank_monitor_ready_event = rank_monitor_ready_event
        self.logger = logger

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

    async def _handle_hb_timeout(self):
        try:
            self.logger.debug("In RankMonitorServer._handle_hb_timeout.")

            if is_process_alive(self.rank_info.pid):
                self._shutdown_rank()
            else:
                self.logger.debug("_handle_hb_timeout: rank process is dead.")
        except Exception:
            self.logger.error("_handle_hb_timeout exception! ", exc_info=True)

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
        self.cfg.initial_rank_heartbeat_timeout = msg.new_initial_heartbeat_timeout
        self.cfg.rank_heartbeat_timeout = msg.new_heartbeat_timeout
        await write_obj_to_ipc_stream(OkMsg(), writer)
        self.logger.debug(
            f"Updated timeouts. "
            f"Initial={msg.new_initial_heartbeat_timeout:.2f} "
            f"Subsequent={msg.new_heartbeat_timeout:.2f}"
        )

    async def _handle_init_msg(self, msg, writer):
        self.rank_info = msg.rank_info
        self.start_time = time.monotonic()
        self.last_hb_time = None
        await write_obj_to_ipc_stream(OkMsg(cfg=self.cfg), writer)

    async def _handle_heartbeat_msg(self, msg, writer):
        self.last_hb_time = time.monotonic()
        assert not msg.state_dict_for_chkpt, "state in heartbeat is not supported in this version"
        await write_obj_to_ipc_stream(OkMsg(), writer)

    def _handle_ipc_connection_lost(self):
        self.rank_info = None
        self.start_time = None
        self.last_hb_time = None
        if self.connection_lock.locked():
            self.connection_lock.release()

    async def _handle_ipc_connection(self, reader, writer):
        if not self.is_closing:
            try:
                if self.connection_lock.locked():
                    self.logger.warn(
                        "Got a new connection while previous is still active. Need to wait..."
                    )
                await asyncio.wait_for(
                    self.connection_lock.acquire(),
                    RankMonitorServer.RANK_MONITOR_INIT_TIMEOUT,
                )
                while msg := await read_obj_from_ipc_stream(reader):
                    if isinstance(msg, AuthkeyMsg):
                        await self._handle_authkey_msg(msg, writer)
                    elif isinstance(msg, InitMsg):
                        await self._handle_init_msg(msg, writer)
                    elif isinstance(msg, HeartbeatMsg):
                        await self._handle_heartbeat_msg(msg, writer)
                    elif isinstance(msg, UpdateConfigMsg):
                        await self._handle_update_config_msg(msg, writer)
                    else:
                        raise AssertionError(f"Unknown msg type: {type(msg)}")
            except Exception as e:
                self.logger.error(f"Exception in _handle_ipc_connection: {e}")
            self._handle_ipc_connection_lost()
            assert not self.connection_lock.locked()
        else:
            self.logger.warn("Ignored incoming connection: service is shutting down.")
        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()
        self.logger.debug("Leaving _handle_ipc_connection. ")

    def _handle_signal(self, sig):
        self.logger.debug(f"Received signal: {sig}")
        if not self.is_closing:
            self.is_closing = True
            self.server.close()

    async def _periodic_rank_check(self):
        while True:
            if self.rank_info is not None:
                curr_time = time.monotonic()
                if self.last_hb_time is None:
                    time_since_start = curr_time - self.start_time
                    if self.cfg.initial_rank_heartbeat_timeout is not None and (
                        time_since_start > self.cfg.initial_rank_heartbeat_timeout
                    ):
                        self.logger.warn(
                            f"Did not get initial heartbeat. "
                            f"Waited {self.cfg.initial_rank_heartbeat_timeout:.2f} seconds."
                        )
                        await self._handle_hb_timeout()
                elif self.cfg.rank_heartbeat_timeout is not None and (
                    (curr_time - self.last_hb_time) > self.cfg.rank_heartbeat_timeout
                ):
                    self.logger.warn(
                        f"Did not get subsequent heartbeat. "
                        f"Waited {self.cfg.rank_heartbeat_timeout:.2f} seconds."
                    )
                    await self._handle_hb_timeout()
            await asyncio.sleep(self.cfg.workload_check_interval)

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

        # Periodic checks
        asyncio.get_running_loop().create_task(self._periodic_rank_check())

        self.rank_monitor_ready_event.set()

        try:
            async with self.server:
                await self.server.serve_forever()
                await self.server.wait_closed()
        except asyncio.exceptions.CancelledError:
            self.logger.debug("server.serve_forever() cancelled.")

    @staticmethod
    def run(
        cfg: FaultToleranceConfig,
        parent_rank: int,
        rank_monitor_ready_event,
    ) -> None:
        try:
            logger = create_logger(
                "RankMonitorServer",
                cfg.log_level,
                parent_rank,
            )
            logger.debug(f"Starting RankMonitorServer... PID={os.getpid()}")
            inst = RankMonitorServer(
                cfg,
                parent_rank,
                rank_monitor_ready_event,
                logger,
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
    def get_ipc_socket_path(parent_rank=None):
        parent_rank = parent_rank if parent_rank is not None else os.environ["RANK"]
        return f'{tempfile.gettempdir()}/fault_tol_rmon_{parent_rank}.sock'

    @staticmethod
    def run_in_subprocess(cfg, parent_rank: Optional[int] = None, mp_ctx=torch.multiprocessing):
        rank_monitor_ready_event = mp_ctx.Event()

        rank_monitor_process_kwargs = {
            "cfg": cfg,
            "parent_rank": parent_rank,
            "rank_monitor_ready_event": rank_monitor_ready_event,
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
