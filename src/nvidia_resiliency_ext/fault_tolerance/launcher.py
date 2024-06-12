#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# SPDX-License-Identifier: BSD-3-Clause
# Modifications made by NVIDIA
# - Added rank monitor setup
# - Changed shutdown logic
# - security fix for watchdog_file_path

# fmt: off
import collections
import contextlib
import importlib.metadata as metadata
import json
import logging
import os
import signal
import socket
import sys
import tempfile
import time
import uuid
from argparse import REMAINDER, ArgumentParser
from dataclasses import dataclass, field
from string import Template
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import torch
from torch.distributed.argparse_util import check_env, env

from nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat import events, metrics, timer
from nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.agent.server.api import (
    RunResult,
    SimpleElasticAgent,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
)
from nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.events.api import (
    EventMetadataValue,
)
from nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.metrics.api import prof, put_metric
from nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.multiprocessing import (
    DefaultLogsSpecs,
    LogsSpecs,
    PContext,
    SignalException,
    Std,
    start_processes,
)
from nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.multiprocessing.errors import (
    ChildFailedError,
    record,
)
from nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.rendezvous import (
    RendezvousParameters,
)
from nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.rendezvous import (
    registry as rdzv_registry,
)
from nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.rendezvous.api import (
    RendezvousGracefulExitError,
)
from nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.rendezvous.utils import (
    _matches_machine_hostname,
    _parse_rendezvous_config,
    parse_rendezvous_endpoint,
)
from nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.utils import macros
from nvidia_resiliency_ext.fault_tolerance.config import FaultToleranceConfig
from nvidia_resiliency_ext.fault_tolerance.data import (
    FT_LAUNCHER_IPC_SOCKET_ENV_VAR,
    FT_RANK_MONITOR_IPC_SOCKET_ENV_VAR,
)
from nvidia_resiliency_ext.fault_tolerance.rank_monitor_server import RankMonitorServer
from nvidia_resiliency_ext.fault_tolerance.utils import patched_method, terminate_mp_processes

logging.basicConfig(
    level=os.getenv('FT_LAUNCHER_LOGLEVEL', 'INFO'),
    format=f"[%(asctime)s] [%(levelname)s] [ft_launcher{os.getpid()}@{socket.gethostname()}] %(message)s",
)

logger = logging.getLogger(__name__)

TORCHELASTIC_ENABLE_FILE_TIMER = "TORCHELASTIC_ENABLE_FILE_TIMER"
TORCHELASTIC_TIMER_FILE = "TORCHELASTIC_TIMER_FILE"

FT_LAUNCHER_IPC_SOCKET = f"{tempfile.gettempdir()}/_ft_launcher{os.getpid()}.socket"


def _register_ft_rdzv_handler():

    from ._ft_rendezvous import FtRendezvousHandler, create_handler
    from ._torch_elastic_compat.rendezvous import rendezvous_handler_registry
    from ._torch_elastic_compat.rendezvous.c10d_rendezvous_backend import create_backend

    def _create_ft_rdzv_handler(params: RendezvousParameters) -> FtRendezvousHandler:
        backend, store = create_backend(params)
        return create_handler(store, backend, params)

    del rendezvous_handler_registry._registry['c10d']  # FIXME: ugly hack to swap the c10d handler
    rendezvous_handler_registry.register("c10d", _create_ft_rdzv_handler)


class UnhealthyNodeException(Exception):
    """Exception raised when a node in a cluster is found to be unhealthy."""

    def __init__(self, message="A node in the cluster is unhealthy"):
        self.message = message
        super().__init__(self.message)


# LocalElasticAgent source
# https://github.com/pytorch/pytorch/blob/release/2.3/torch/distributed/elastic/agent/server/local_elastic_agent.py


class LocalElasticAgent(SimpleElasticAgent):
    """An implementation of :py:class:`torchelastic.agent.server.ElasticAgent` that handles host-local workers.

    This agent is deployed per host and is configured to spawn ``n`` workers.
    When using GPUs, ``n`` maps to the number of GPUs available on the host.

    The local agent does not communicate to other local agents deployed on
    other hosts, even if the workers may communicate inter-host. The worker id
    is interpreted to be a local process. The agent starts and stops all worker
    processes as a single unit.


    The worker function and argument passed to the worker function must be
    python multiprocessing compatible. To pass multiprocessing data structures
    to the workers you may create the data structure in the same multiprocessing
    context as the specified ``start_method`` and pass it as a function argument.

    The ``exit_barrier_timeout`` specifies the amount of time (in seconds) to wait
    for other agents to finish. This acts as a safety net to handle cases where
    workers finish at different times, to prevent agents from viewing workers
    that finished early as a scale-down event. It is strongly advised that the
    user code deal with ensuring that workers are terminated in a synchronous
    manner rather than relying on the exit_barrier_timeout.

    A named pipe based watchdog can be enabled in ```LocalElasticAgent``` if an
    environment variable ``TORCHELASTIC_ENABLE_FILE_TIMER`` with value 1 has
    been defined in the ```LocalElasticAgent``` process.
    Optionally, another environment variable ```TORCHELASTIC_TIMER_FILE```
    can be set with a unique file name for the named pipe. If the environment
    variable ```TORCHELASTIC_TIMER_FILE``` is not set, ```LocalElasticAgent```
    will internally create a unique file name and set it to the environment
    variable ```TORCHELASTIC_TIMER_FILE```, and this environment variable will
    be propagated to the worker processes to allow them to connect to the same
    named pipe that ```LocalElasticAgent``` uses.

    Logs are written to the specified log directory. Each log line will be by default
    prefixed by ``[${role_name}${local_rank}]:`` (e.g. ``[trainer0]: foobar``).
    Log prefixes can be customized by passing a `template string
    <https://docs.python.org/3/library/string.html#template-strings>`_ as the
    ``log_line_prefix_template`` argument.
    The following macros (identifiers) are substituted at runtime:
    ``${role_name}, ${local_rank}, ${rank}``. For example, to prefix each log line with
    global rank instead of the local rank, set ``log_line_prefix_template = "[${rank}]:``.


    Example launching function

    ::

        def trainer(args) -> str:
            return "do train"

        def main():
            start_method="spawn"
            shared_queue= multiprocessing.get_context(start_method).Queue()
            spec = WorkerSpec(
                        role="trainer",
                        local_world_size=nproc_per_process,
                        entrypoint=trainer,
                        args=("foobar",),
                        ...<OTHER_PARAMS...>)
            agent = LocalElasticAgent(spec, start_method)
            results = agent.run()

            if results.is_failed():
                print("trainer failed")
            else:
                print(f"rank 0 return value: {results.return_values[0]}")
                # prints -> rank 0 return value: do train

    Example launching binary

    ::

        def main():
            spec = WorkerSpec(
                        role="trainer",
                        local_world_size=nproc_per_process,
                        entrypoint="/usr/local/bin/trainer",
                        args=("--trainer-args", "foobar"),
                        ...<OTHER_PARAMS...>)
            agent = LocalElasticAgent(spec)
            results = agent.run()

            if not results.is_failed():
                print("binary launches do not have return values")

    """

    def __init__(
        self,
        spec: WorkerSpec,
        fault_tol_cfg: FaultToleranceConfig,
        logs_specs: LogsSpecs,
        start_method="spawn",
        exit_barrier_timeout: float = 300,
        log_line_prefix_template: Optional[str] = None,
        term_timeout: float = 1800,
        workers_stop_timeout: float = 30, 
        restart_policy: str = "any-failed",
        is_store_host: bool = False,
    ):
        super().__init__(spec, exit_barrier_timeout)
        self._start_method = start_method
        self._pcontext: Optional[PContext] = None
        self._rdzv_handler = spec.rdzv_handler
        self._log_line_prefix_template = log_line_prefix_template
        self._worker_watchdog: Optional[timer.FileTimerServer] = None
        self._logs_specs = logs_specs
        self._term_timeout = term_timeout
        self._workers_stop_timeout = workers_stop_timeout
        self._is_store_host = is_store_host
        self._local_rank_to_rmon: Dict[int, Any] = dict()
        self._ft_cfg = fault_tol_cfg
        self._children_pgids: Set[int] = set()
        self._restart_policy = restart_policy

    DEFAULT_ROLE = "default"  # FIXME

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.metrics.prof`.
    @prof
    def run(self, role: str = DEFAULT_ROLE) -> RunResult:
        start_time = time.monotonic()
        shutdown_called: bool = False
        try:
            result = self._invoke_run(role)
            self._total_execution_time = int(time.monotonic() - start_time)
            self._record_metrics(result)
            self._record_worker_events(result)
            return result
        except RendezvousGracefulExitError as e:
            logger.info("Rendezvous gracefully exited: %s", e)
        except SignalException as e:
            logger.warning("Received %s death signal, shutting down workers, timeout %s sec.", e.sigval, self._term_timeout)
            self._shutdown(e.sigval, timeout=self._term_timeout)
            shutdown_called = True
            raise
        finally:
            if not shutdown_called:
                self._shutdown()
            # record the execution time in case there were any exceptions during run.
            self._total_execution_time = int(time.monotonic() - start_time)

    def _invoke_run(self, role: str = DEFAULT_ROLE) -> RunResult:
        if self._restart_policy == 'any-failed':
            return self._invoke_run_with_any_failed_policy(role)
        elif self._restart_policy == 'min-healthy':
            return self._invoke_run_with_min_healthy_policy(role)
        else:
            raise AssertionError(
                f"Unexpected restart-policy: {self._restart_policy}."
                " check CLI help for --restart-policy allowed options"
            )

    def _invoke_run_with_any_failed_policy(self, role: str = DEFAULT_ROLE) -> RunResult:
        # NOTE: currently only works for a single role

        spec = self._worker_group.spec
        role = spec.role

        logger.info("[%s] starting workers for entrypoint: %s", role, spec.get_entrypoint_name())

        self._initialize_workers(self._worker_group)
        monitor_interval = spec.monitor_interval
        rdzv_handler = spec.rdzv_handler

        while True:
            assert self._worker_group.state != WorkerState.INIT
            time.sleep(monitor_interval)
            run_result = self._monitor_workers(self._worker_group)
            state = run_result.state
            self._worker_group.state = state

            put_metric(f"workers.{role}.remaining_restarts", self._remaining_restarts)
            put_metric(f"workers.{role}.{state.name.lower()}", 1)

            if state == WorkerState.SUCCEEDED:
                logger.info(
                    "[%s] worker group successfully finished."
                    " Waiting %s seconds for other agents to finish.",
                    role,
                    self._exit_barrier_timeout,
                )
                self._exit_barrier()
                return run_result
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
                if self._remaining_restarts > 0:
                    logger.info(
                        "[%s] Worker group %s. "
                        "%s/%s attempts left;"
                        " will restart worker group",
                        role,
                        state.name,
                        self._remaining_restarts,
                        spec.max_restarts,
                    )
                    self._remaining_restarts -= 1
                    self._restart_workers(self._worker_group)
                else:
                    self._stop_workers(self._worker_group)
                    self._worker_group.state = WorkerState.FAILED
                    # to preserve torchrun's behaviour, should not return WorkerState.UNHEALTHY.
                    # we use WorkerState.UNHEALTHY to denote a worker group that is still
                    # running but has some failed workers. torchrun does not use WorkerState.UNHEALTHY
                    run_result = self._monitor_workers(self._worker_group)
                    return run_result
            elif state == WorkerState.HEALTHY:
                # membership changes do not count as retries
                num_nodes_waiting = rdzv_handler.num_nodes_waiting()
                group_rank = self._worker_group.group_rank
                if num_nodes_waiting > 0:
                    logger.info(
                        "[%s] Detected %s "
                        "new nodes from group_rank=%s; "
                        "will restart worker group",
                        role,
                        num_nodes_waiting,
                        group_rank,
                    )
                    self._restart_workers(self._worker_group)
            else:
                raise Exception(f"[{role}] Worker group in {state.name} state")

    def _invoke_run_with_min_healthy_policy(self, role: str = DEFAULT_ROLE) -> RunResult:
        # NOTE: currently only works for a single role

        spec = self._worker_group.spec
        role = spec.role

        logger.info(
            f"[{role}] starting workers for entrypoint: {spec.get_entrypoint_name()} with min-healthy policy"
        )

        self._initialize_workers(self._worker_group)
        monitor_interval = spec.monitor_interval
        rdzv_handler = spec.rdzv_handler
        min_nodes = rdzv_handler._settings.min_nodes
        group_rank = self._worker_group.group_rank

        while True:
            assert self._worker_group.state != WorkerState.INIT
            time.sleep(monitor_interval)
            run_result = self._monitor_workers(self._worker_group)
            state = run_result.state
            self._worker_group.state = state

            put_metric(f"workers.{role}.remaining_restarts", self._remaining_restarts)
            put_metric(f"workers.{role}.{state.name.lower()}", 1)

            if state in {
                WorkerState.SUCCEEDED,
                WorkerState.FAILED,
                WorkerState.UNHEALTHY,
                WorkerState.HEALTHY,
            }:
                state_in_rdzv = rdzv_handler.try_set_worker_state(state)
                if state_in_rdzv != state:
                    assert (
                        state_in_rdzv == WorkerState.UNKNOWN
                    ), f"Could not set worker group state {state=} {state_in_rdzv=}"
                    # state in the rdzv is UNKNOWN if this node was marked as dead by other participants
            else:
                raise RuntimeError(f"[{role}] Worker group in unexpected state: {state.name}")

            # count the number of worker groups in each state
            all_worker_states = rdzv_handler.get_worker_states()
            worker_state_cnt = collections.Counter(all_worker_states.values())
            num_succ = worker_state_cnt[WorkerState.SUCCEEDED]
            num_healthy = worker_state_cnt[WorkerState.HEALTHY]

            logger.debug(
                "[%s] group_rank=%s worker_state_cnt=%s", role, group_rank, worker_state_cnt
            )

            # check if the current run (rendezvous) ended successfully:
            # at least "min_nodes" worker groups should succeed to consider a run successful.
            # NOTE: if the run was successful all agents exit with WorkerState.SUCCEEDED despite
            # their actual run result
            if num_healthy == 0:
                if num_succ >= min_nodes:
                    logger.info(
                        f"[{role}] {group_rank=} {state.name=} detected that the run ended successfuly: {worker_state_cnt=}"
                    )
                    # ensure this node workers are terminated
                    self._stop_workers(self._worker_group)
                    # WAR: return values are not meaningful in this case,
                    # but some dummy values are required for the event logging
                    dummy_ret_vals = {w.global_rank: None for w in self._worker_group.workers}
                    return RunResult(state=WorkerState.SUCCEEDED, return_values=dummy_ret_vals)

            # check if the current run can end successfully
            max_possible_succ = num_succ + num_healthy
            can_continue = max_possible_succ >= min_nodes

            if can_continue:
                # upscaling should be disabled in min-healthy mode
                num_nodes_waiting = rdzv_handler.num_nodes_waiting()
                if num_nodes_waiting > 0:
                    raise RuntimeError(
                        f"Detected {num_nodes_waiting} nodes in the waiting list. This should not happen with min-healthy mode."
                    )
            else:
                # we can't have min_nodes successful worker groups.
                # NOTE: this worker group still might be successful/healthy.
                # all we can do is to restart if possible or shutdown otherwise
                # NOTE: we use the rdzv round to count the restarts, so restarts are counted
                # globally (while in any-failed mode each worker counts its own restarts)
                logger.warning(
                    f"[{role}] {group_rank=} {state.name=} detected that the current run failed: {worker_state_cnt=}"
                )
                self._remaining_restarts = spec.max_restarts - self._rdzv_handler.round()
                if self._remaining_restarts > 0:
                    logger.info(
                        f"{self._remaining_restarts}/{spec.max_restarts} restart attempts left; restarting worker group...",
                    )
                    self._remaining_restarts -= 1
                    self._restart_workers(self._worker_group)
                else:
                    # try to stop the workers and return the updated run result
                    logger.info(f"0/{spec.max_restarts} restart attempts left. Exiting...")
                    self._stop_workers(self._worker_group)
                    run_result = self._monitor_workers(self._worker_group)
                    return run_result

    def get_rank_mon_socket_path(self, local_rank):
        return f"{tempfile.gettempdir()}/_ft_launcher{os.getpid()}_rmon{local_rank}.socket"

    def setup_rank_monitors(self, envs: Dict[int, Dict[str, str]]) -> None:
        spawn_mp_ctx = torch.multiprocessing.get_context("spawn")
        for worker_env in envs.values():
            # Start rank monitors if not already started
            # Each rank (re)connects to its rank monitor when it starts
            # Monitor of the local rank0 on the store hosting node is the restarter logger
            local_rank = int(worker_env['LOCAL_RANK'])
            is_restarter_logger = self._is_store_host and local_rank == 0
            rmon_ipc_socket = worker_env[FT_RANK_MONITOR_IPC_SOCKET_ENV_VAR]
            if local_rank not in self._local_rank_to_rmon:
                self._local_rank_to_rmon[local_rank] = RankMonitorServer.run_in_subprocess(
                    cfg=self._ft_cfg,
                    ipc_socket_path=rmon_ipc_socket,
                    is_restarter_logger=is_restarter_logger,
                    mp_ctx=spawn_mp_ctx,
                )

    def shutdown_rank_monitors(self):
        for local_rank, rmon_proc in self._local_rank_to_rmon.items():
            with contextlib.suppress(Exception):
                rmon_proc.terminate()
            with contextlib.suppress(Exception):
                rmon_proc.join()
            with contextlib.suppress(Exception):
                os.unlink(self.get_rank_mon_socket_path(local_rank))

    def _setup_local_watchdog(self, envs: Dict[int, Dict[str, str]]) -> None:
        enable_watchdog_env_name = TORCHELASTIC_ENABLE_FILE_TIMER
        watchdog_enabled = os.getenv(enable_watchdog_env_name)
        watchdog_file_env_name = TORCHELASTIC_TIMER_FILE
        watchdog_file_path = os.getenv(watchdog_file_env_name)
        if watchdog_enabled is not None and str(watchdog_enabled) == "1":
            if watchdog_file_path is None:
                watchdog_file_path = os.path.join(
                    tempfile.gettempdir(), f"watchdog_timer_{uuid.uuid4()}"
                )
            logger.info("Starting a FileTimerServer with %s ...", watchdog_file_path)
            self._worker_watchdog = timer.FileTimerServer(
                file_path=watchdog_file_path,
                max_interval=0.1,
                daemon=True,
                log_event=self._log_watchdog_event,
            )
            self._worker_watchdog.start()
            logger.info("FileTimerServer started")
        else:
            logger.info(
                "Environment variable '%s' not found. Do not start FileTimerServer.",
                enable_watchdog_env_name,
            )
        # Propagate the watchdog file env to worker processes
        if watchdog_file_path is not None:
            for worker_env in envs.values():
                worker_env[watchdog_file_env_name] = watchdog_file_path

    def _get_fq_hostname(self) -> str:
        return socket.getfqdn(socket.gethostname())

    def _log_watchdog_event(
        self,
        name: str,
        request: Optional[timer.FileTimerRequest],
    ) -> None:
        wg = self._worker_group
        spec = wg.spec
        md = {"watchdog_event": name}
        if request is not None:
            md["worker_pid"] = str(request.worker_pid)
            md["scope_id"] = request.scope_id
            md["expiration_time"] = str(request.expiration_time)
            md["signal"] = str(request.signal)
        md_str = json.dumps(md)
        state = "RUNNING"
        metadata: Dict[str, EventMetadataValue] = {
            "run_id": spec.rdzv_handler.get_run_id(),
            "global_rank": None,
            "group_rank": wg.group_rank,
            "worker_id": None,
            "role": spec.role,
            "hostname": self._get_fq_hostname(),
            "state": state,
            "total_run_time": self._total_execution_time,
            "rdzv_backend": spec.rdzv_handler.get_backend(),
            "raw_error": None,
            "metadata": md_str,
            "agent_restarts": spec.max_restarts - self._remaining_restarts,
        }
        # Note: The 'metadata' field of the Event is converted to a TorchelasticStatusLogEntry later.
        #       The 'name' field of the Event is NOT used in the TorchelasticStatusLogEntry.
        event = events.Event(name=name, source=events.EventSource.AGENT, metadata=metadata)
        events.record(event)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _stop_workers(self, worker_group: WorkerGroup) -> None:
        logger.info(f"Stopping workers... Timeout = {self._workers_stop_timeout} sec.")
        self._shutdown(timeout=self._workers_stop_timeout)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        spec = worker_group.spec
        store = worker_group.store
        assert store is not None
        master_addr, master_port = super()._get_master_addr_port(store)
        restart_count = spec.max_restarts - self._remaining_restarts

        use_agent_store = spec.rdzv_handler.get_backend() == "static"

        args: Dict[int, Tuple] = {}
        envs: Dict[int, Dict[str, str]] = {}
        log_line_prefixes: Optional[Dict[int, str]] = {} if self._log_line_prefix_template else None
        for worker in worker_group.workers:
            local_rank = worker.local_rank
            worker_env = {
                "LOCAL_RANK": str(local_rank),
                "RANK": str(worker.global_rank),
                "GROUP_RANK": str(worker_group.group_rank),
                "ROLE_RANK": str(worker.role_rank),
                "ROLE_NAME": spec.role,
                "LOCAL_WORLD_SIZE": str(spec.local_world_size),
                "WORLD_SIZE": str(worker.world_size),
                "GROUP_WORLD_SIZE": str(worker_group.group_world_size),
                "ROLE_WORLD_SIZE": str(worker.role_world_size),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
                "TORCHELASTIC_RESTART_COUNT": str(restart_count),
                "TORCHELASTIC_MAX_RESTARTS": str(spec.max_restarts),
                "TORCHELASTIC_RUN_ID": spec.rdzv_handler.get_run_id(),
                "TORCHELASTIC_USE_AGENT_STORE": str(use_agent_store),
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": os.getenv(
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING", str(1)
                ),
                FT_LAUNCHER_IPC_SOCKET_ENV_VAR: FT_LAUNCHER_IPC_SOCKET,
                FT_RANK_MONITOR_IPC_SOCKET_ENV_VAR: self.get_rank_mon_socket_path(local_rank),
            }
            if "OMP_NUM_THREADS" in os.environ:
                worker_env["OMP_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]

            if self._log_line_prefix_template:
                log_line_prefix = Template(self._log_line_prefix_template).safe_substitute(
                    role_name=spec.role,
                    rank=worker.global_rank,
                    local_rank=local_rank,
                )
                log_line_prefixes[local_rank] = log_line_prefix

            envs[local_rank] = worker_env
            worker_args = list(spec.args)
            worker_args = macros.substitute(worker_args, str(local_rank))
            args[local_rank] = tuple(worker_args)

        self._setup_local_watchdog(envs=envs)

        self.setup_rank_monitors(envs=envs)

        assert spec.entrypoint is not None
        assert self._logs_specs is not None
        self._pcontext = start_processes(
            name=spec.role,
            entrypoint=spec.entrypoint,
            args=args,
            envs=envs,
            logs_specs=self._logs_specs,
            log_line_prefixes=log_line_prefixes,
            start_method=self._start_method,
        )

        self._children_pgids = {os.getpgid(p) for p in self._pcontext.pids().values()}

        return self._pcontext.pids()

    def _shutdown(self, death_sig: signal.Signals = signal.SIGTERM, timeout: int = 30) -> None:
        if self._worker_watchdog is not None:
            self._worker_watchdog.stop()
            self._worker_watchdog = None
        if self._pcontext:
            self._pcontext.close(death_sig, timeout=timeout)
            # Remove multiprocessing leftovers
            # PID=1 become a parent if the original parent died
            terminate_mp_processes(allowed_ppids={1}, allowed_pgids=self._children_pgids)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        role = worker_group.spec.role
        worker_pids = {w.id for w in worker_group.workers}
        assert self._pcontext is not None
        pc_pids = set(self._pcontext.pids().values())
        if worker_pids != pc_pids:
            logger.error(
                "[%s] worker pids do not match process_context pids." " Expected: %s, actual: %s",
                role,
                worker_pids,
                pc_pids,
            )
            return RunResult(state=WorkerState.UNKNOWN)

        with patched_method(self._pcontext, 'close', lambda *args, **kwargs: None):
            # do not terminate healthy workers if some failed
            # we will terminate them when leaving the rendezvous round
            result = self._pcontext.wait(0)

        if result:

            if result.is_failed():
                # map local rank failure to global rank
                worker_failures = {}
                for local_rank, failure in result.failures.items():
                    worker = worker_group.workers[local_rank]
                    worker_failures[worker.global_rank] = failure
                all_failed = len(result.failures) == len(worker_pids)
                if all_failed:
                    run_result = RunResult(
                        state=WorkerState.FAILED,
                        failures=worker_failures,
                    )
                else:
                    # some workers have failed, while others are still running or succeeded
                    # this worker group is going to finish in WorkerState.FAILED state or stay UNHEALTHY.
                    run_result = RunResult(
                        state=WorkerState.UNHEALTHY,
                        failures=worker_failures,
                    )
                return run_result
            else:
                # copy ret_val_queue into a map with a global ranks
                workers_ret_vals = {}
                for local_rank, ret_val in result.return_values.items():
                    worker = worker_group.workers[local_rank]
                    workers_ret_vals[worker.global_rank] = ret_val
                return RunResult(
                    state=WorkerState.SUCCEEDED,
                    return_values=workers_ret_vals,
                )
        else:
            return RunResult(state=WorkerState.HEALTHY)

    def any_rank_failed(self) -> bool:
        result = None
        if self._pcontext is not None:
            result = self._pcontext.wait(0)
        return result is not None and result.is_failed()

    def clean_rdzv_shutdown(self, close):
        # Try to exit rendezvous gracefully, with the store host leaving last.
        # If the store host exits early, other nodes might try to use the store
        # and fail due to connection errors.
        # Marking the rdzv as closed causes spare nodes to exit the rdzv with RendezvousGracefulExitError.
        # and prevents all other agents from using the rdzv, so it should be closed only when leaving
        # the workload
        try:
            rdzv_handler = self._rdzv_handler
            rdzv_handler._stop_heartbeats()  # stops rdzv backround thread
            rdzv_handler.remove_this_node()
            if close:
                rdzv_handler.set_closed()
            if self._is_store_host:
                timeout = self._exit_barrier_timeout
                pooling_interval = 1.0
                time_left = timeout
                while rdzv_handler.num_nodes() > 0 and time_left > 0:
                    time.sleep(pooling_interval)
                    time_left -= pooling_interval
                if rdzv_handler.num_nodes() > 0:
                    logger.warning(
                        f"Some nodes did not leave the rendezvous on time ({timeout=} seconds). "
                        "Exiting agent anyway, but this might result in rdzv failures."
                    )
                else:
                    logger.info(
                        "Rendezvous don't have any nodes. Leaving the store hosting process..."
                    )
        except Exception as e:
            logger.warning(f"Error while trying to gracefully exit the rendezvous: {e}")
            pass  # continue, we are exiting the process anyway


# Source
# https://github.com/pytorch/pytorch/blob/release/2.3/torch/distributed/launcher/api.py


@dataclass
class LaunchConfig:
    """
    Creates a rendezvous config.

    Args:
        min_nodes: Minimum amount of nodes that the user function will
                        be launched on. Elastic agent ensures that the user
                        function start only when the min_nodes amount enters
                        the rendezvous.
        max_nodes: Maximum amount of nodes that the user function
                        will be launched on.
        nproc_per_node: On each node the elastic agent will launch
                            this amount of workers that will execute user
                            defined function.
        rdzv_backend: rdzv_backend to use in the rendezvous (zeus-adapter, etcd).
        rdzv_endpoint: The endpoint of the rdzv sync. storage.
        rdzv_configs: Key, value pair that specifies rendezvous specific configuration.
        rdzv_timeout: Legacy argument that specifies timeout for the rendezvous. It is going
            to be removed in future versions, see the note below. The default timeout is 900 seconds.
        run_id: The unique run id of the job (if not passed a unique one will be
                deduced from run environment - flow workflow id in flow - or auto generated).
        role: User defined role of the worker (defaults to "trainer").
        max_restarts: The maximum amount of restarts that elastic agent will conduct
                    on workers before failure.
        restart_policy: Determines when worker groups are restarted e.g. if any worker group failed,
                    or if number of healthy worker groups falls below min-nodes etc. see also the CLI arg
        monitor_interval: The interval in seconds that is used by the elastic_agent
                        as a period of monitoring workers.
        start_method: The method is used by the elastic agent to start the
                    workers (spawn, fork, forkserver).
        metrics_cfg: configuration to initialize metrics.
        local_addr: address of the local node if any. If not set, a lookup on the local
                machine's FQDN will be performed.
        local_ranks_filter: ranks for which to show logs in console. If not set, show from all.
    ..note:
        `rdzv_timeout` is a legacy argument that will be removed in future.
        Set the timeout via `rdzv_configs['timeout']`

    """

    min_nodes: int
    max_nodes: int
    nproc_per_node: int
    fault_tol_cfg: FaultToleranceConfig
    logs_specs: Optional[LogsSpecs] = None
    run_id: str = ""
    role: str = "default_role"
    rdzv_endpoint: str = ""
    rdzv_backend: str = "etcd"
    rdzv_configs: Dict[str, Any] = field(default_factory=dict)
    rdzv_timeout: int = -1
    max_restarts: int = 3
    restart_policy: str = "any-failed"
    term_timeout: float = 1800
    workers_stop_timeout: float = 30
    monitor_interval: float = 30
    start_method: str = "spawn"
    log_line_prefix_template: Optional[str] = None
    metrics_cfg: Dict[str, str] = field(default_factory=dict)
    local_addr: Optional[str] = None

    def __post_init__(self):
        default_timeout = 900
        if self.rdzv_timeout != -1:
            self.rdzv_configs["timeout"] = self.rdzv_timeout
        elif "timeout" not in self.rdzv_configs:
            self.rdzv_configs["timeout"] = default_timeout
        # bump Torch Elastic default timeouts, so it will work with large scale workloads
        if "join_timeout" not in self.rdzv_configs:
            self.rdzv_configs["join_timeout"] = 900  # default is 600 seconds
        if "close_timeout" not in self.rdzv_configs:
            self.rdzv_configs["close_timeout"] = 600  # default is 30 seconds
        if "read_timeout" not in self.rdzv_configs:
            self.rdzv_configs["read_timeout"] = 600  # default is 60 seconds

        # Post-processing to enable refactoring to introduce logs_specs due to non-torchrun API usage
        if self.logs_specs is None:
            self.logs_specs = DefaultLogsSpecs()


class elastic_launch:
    """
    Launches an torchelastic agent on the container that invoked the entrypoint.

        1. Pass the ``entrypoint`` arguments as non ``kwargs`` (e.g. no named parameters)/
           ``entrypoint`` can be a function or a command.
        2. The return value is a map of each worker's output mapped
           by their respective global rank.

    Usage

    ::

    def worker_fn(foo):
        # ...

    def main():
        # entrypoint is a function.
        outputs = elastic_launch(LaunchConfig, worker_fn)(foo)
        # return rank 0's output
        return outputs[0]

        # entrypoint is a command and ``script.py`` is the python module.
        outputs = elastic_launch(LaunchConfig, "script.py")(args)
        outputs = elastic_launch(LaunchConfig, "python")("script.py")
    """

    def __init__(
        self,
        config: LaunchConfig,
        entrypoint: Union[Callable, str, None],
    ):
        self._config = config
        self._entrypoint = entrypoint

    def __call__(self, *args):
        return launch_agent(self._config, self._entrypoint, list(args))


def _get_entrypoint_name(entrypoint: Union[Callable, str, None], args: List[Any]) -> str:
    """Retrieve entrypoint name with the rule:
    1. If entrypoint is a function, use ``entrypoint.__qualname__``.
    2. If entrypoint is a string, check its value:
        2.1 if entrypoint equals to ``sys.executable`` (like "python"), use the first element from ``args``
            which does not start with hifen letter (for example, "-u" will be skipped).
        2.2 otherwise, use ``entrypoint`` value.
    3. Otherwise, return empty string.
    """
    if isinstance(entrypoint, Callable):  # type: ignore[arg-type]
        return entrypoint.__name__  # type: ignore[union-attr]
    elif isinstance(entrypoint, str):
        if entrypoint == sys.executable:
            return next((arg for arg in args if arg[0] != "-"), "")
        else:
            return entrypoint
    else:
        return ""


def _get_addr_and_port(
    rdzv_parameters: RendezvousParameters,
) -> Tuple[Optional[str], Optional[int]]:
    if rdzv_parameters.backend != "static":
        return (None, None)
    endpoint = rdzv_parameters.endpoint
    endpoint = endpoint.strip()
    if not endpoint:
        raise ValueError(
            "Endpoint is missing in endpoint. Try to add --master-addr and --master-port"
        )
    master_addr, master_port = parse_rendezvous_endpoint(endpoint, default_port=-1)
    if master_port == -1:
        raise ValueError(f"port is missing in endpoint: {endpoint}. Try to specify --master-port")
    return (master_addr, master_port)


def _is_store_host(params: RendezvousParameters) -> bool:
    """Returns true if this agent is hosting the TCP store"""
    host, _ = parse_rendezvous_endpoint(params.endpoint, default_port=0)
    cfg_is_host = params.get_as_bool("is_host")
    if cfg_is_host is not None:
        return bool(cfg_is_host)
    return _matches_machine_hostname(host)


def launch_agent(
    config: LaunchConfig,
    entrypoint: Union[Callable, str, None],
    args: List[Any],
) -> Dict[int, Any]:
    if not config.run_id:
        run_id = str(uuid.uuid4().int)
        logger.warning("config has no run_id, generated a random run_id: %s", run_id)
        config.run_id = run_id

    entrypoint_name = _get_entrypoint_name(entrypoint, args)

    # with min-healthy restarting policy, if the rendezvous is completed (workers are running),
    # we dont want to replace missing/dead nodes with spares nor to upscale the rendezvous with new arrivals
    config.rdzv_configs['upscaling_enabled'] = config.restart_policy != "min-healthy"

    logger.info(
        "Starting elastic_operator with launch configs:\n"
        "  entrypoint       : %(entrypoint)s\n"
        "  min_nodes        : %(min_nodes)s\n"
        "  max_nodes        : %(max_nodes)s\n"
        "  nproc_per_node   : %(nproc_per_node)s\n"
        "  run_id           : %(run_id)s\n"
        "  rdzv_backend     : %(rdzv_backend)s\n"
        "  rdzv_endpoint    : %(rdzv_endpoint)s\n"
        "  rdzv_configs     : %(rdzv_configs)s\n"
        "  max_restarts     : %(max_restarts)s\n"
        "  restart_policy   : %(restart_policy)s\n"
        "  monitor_interval : %(monitor_interval)s\n"
        "  log_dir          : %(log_dir)s\n"
        "  metrics_cfg      : %(metrics_cfg)s\n",
        {
            "entrypoint": entrypoint_name,
            "min_nodes": config.min_nodes,
            "max_nodes": config.max_nodes,
            "nproc_per_node": config.nproc_per_node,
            "run_id": config.run_id,
            "rdzv_backend": config.rdzv_backend,
            "rdzv_endpoint": config.rdzv_endpoint,
            "rdzv_configs": config.rdzv_configs,
            "max_restarts": config.max_restarts,
            "restart_policy": config.restart_policy,
            "monitor_interval": config.monitor_interval,
            "log_dir": config.logs_specs.root_log_dir,  # type: ignore[union-attr]
            "metrics_cfg": config.metrics_cfg,
        },
    )
    rdzv_parameters = RendezvousParameters(
        backend=config.rdzv_backend,
        endpoint=config.rdzv_endpoint,
        run_id=config.run_id,
        min_nodes=config.min_nodes,
        max_nodes=config.max_nodes,
        local_addr=config.local_addr,
        **config.rdzv_configs,
    )

    master_addr, master_port = _get_addr_and_port(rdzv_parameters)

    spec = WorkerSpec(
        role=config.role,
        local_world_size=config.nproc_per_node,
        entrypoint=entrypoint,
        args=tuple(args),
        rdzv_handler=rdzv_registry.get_rendezvous_handler(rdzv_parameters),
        max_restarts=config.max_restarts,
        monitor_interval=config.monitor_interval,
        master_addr=master_addr,
        master_port=master_port,
        local_addr=config.local_addr,
    )

    agent = LocalElasticAgent(
        spec=spec,
        fault_tol_cfg=config.fault_tol_cfg,
        logs_specs=config.logs_specs,  # type: ignore[arg-type]
        start_method=config.start_method,
        log_line_prefix_template=config.log_line_prefix_template,
        term_timeout=config.term_timeout,
        workers_stop_timeout=config.workers_stop_timeout,
        restart_policy=config.restart_policy,
        is_store_host=_is_store_host(rdzv_parameters),
    )

    shutdown_rdzv = True
    try:
        metrics.initialize_metrics(metrics.MetricsConfig(config.metrics_cfg))

        result = agent.run()

        # records that agent.run() has succeeded NOT that workers have succeeded
        events.record(agent.get_event_succeeded())

        if result is None:
            logger.info("Agent .run() result is None. Agent was waiting at the rendezvous.")
            return None

        if result.is_failed():
            # ChildFailedError is treated specially by @record
            # if the error files for the failed children exist
            # @record will copy the first error (root cause)
            # to the error file of the launcher process.
            raise ChildFailedError(
                name=entrypoint_name,
                failures=result.failures,
            )

        logger.info(f"Agent .run() is OK. No failures in the result. {result=}")
        return result.return_values
    except UnhealthyNodeException as e:
        # do not shutdown rendezvous when an unhealthy node is leaving
        shutdown_rdzv = False
        logger.error(f"Agent .run() raised UnhealthyNodeException: {e}")
        events.record(agent.get_event_failed())
    except ChildFailedError:
        raise
    except SignalException as e:
        # when the agent dies with a signal do NOT shutdown the rdzv_handler
        # since this closes the rendezvous on this rdzv_id permanently and
        # prevents any additional scaling events
        shutdown_rdzv = False
        logger.error(f"Agent .run() raised SignalException: {e}")
        events.record(agent.get_event_failed())
        if agent.any_rank_failed():
            logger.warning("Some ranks exited with non-zero. Re-raising SignalException.")
            raise
        else:
            logger.info("All ranks exited gracefully. Launcher exiting without an error.")
    except Exception as e:
        logger.error(f"Agent .run() raised exception, {e=}")
        events.record(agent.get_event_failed())
        raise
    finally:
        agent.clean_rdzv_shutdown(close=shutdown_rdzv)
        agent.shutdown_rank_monitors()
        with contextlib.suppress(Exception):
            os.unlink(FT_LAUNCHER_IPC_SOCKET)


# Source
# https://github.com/pytorch/pytorch/blob/release/2.3/torch/distributed/run.py

"""
Superset of ``torch.distributed.launch``.

``torchrun`` provides a superset of the functionality as ``torch.distributed.launch``
with the following additional functionalities:

1. Worker failures are handled gracefully by restarting all workers.

2. Worker ``RANK`` and ``WORLD_SIZE`` are assigned automatically.

3. Number of nodes is allowed to change between minimum and maximum sizes (elasticity).

.. note:: ``torchrun`` is a python
          `console script <https://packaging.python.org/en/latest/specifications/entry-points/#use-for-scripts>`_
          to the main module
          `torch.distributed.run <https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py>`_
          declared in the ``entry_points`` configuration in
          `setup.py <https://github.com/pytorch/pytorch/blob/master/setup.py>`_.
          It is equivalent to invoking ``python -m torch.distributed.run``.


Transitioning from torch.distributed.launch to torchrun
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


``torchrun`` supports the same arguments as ``torch.distributed.launch`` **except**
for ``--use-env`` which is now deprecated. To migrate from ``torch.distributed.launch``
to ``torchrun`` follow these steps:

1.  If your training script is already reading ``local_rank`` from the ``LOCAL_RANK`` environment variable.
    Then you need simply omit the ``--use-env`` flag, e.g.:

    +--------------------------------------------------------------------+--------------------------------------------+
    |         ``torch.distributed.launch``                               |                ``torchrun``                |
    +====================================================================+============================================+
    |                                                                    |                                            |
    | .. code-block:: shell-session                                      | .. code-block:: shell-session              |
    |                                                                    |                                            |
    |    $ python -m torch.distributed.launch --use-env train_script.py  |    $ torchrun train_script.py              |
    |                                                                    |                                            |
    +--------------------------------------------------------------------+--------------------------------------------+

2.  If your training script reads local rank from a ``--local-rank`` cmd argument.
    Change your training script to read from the ``LOCAL_RANK`` environment variable as
    demonstrated by the following code snippet:

    +-------------------------------------------------------+----------------------------------------------------+
    |         ``torch.distributed.launch``                  |                    ``torchrun``                    |
    +=======================================================+====================================================+
    |                                                       |                                                    |
    | .. code-block:: python                                | .. code-block:: python                             |
    |                                                       |                                                    |
    |                                                       |                                                    |
    |    import argparse                                    |     import os                                      |
    |    parser = argparse.ArgumentParser()                 |     local_rank = int(os.environ["LOCAL_RANK"])     |
    |    parser.add_argument("--local-rank", type=int)      |                                                    |
    |    args = parser.parse_args()                         |                                                    |
    |                                                       |                                                    |
    |    local_rank = args.local_rank                       |                                                    |
    |                                                       |                                                    |
    +-------------------------------------------------------+----------------------------------------------------+

The aformentioned changes suffice to migrate from ``torch.distributed.launch`` to ``torchrun``.
To take advantage of new features such as elasticity, fault-tolerance, and error reporting of ``torchrun``
please refer to:

* :ref:`elastic_train_script` for more information on authoring training scripts that are ``torchrun`` compliant.
* the rest of this page for more information on the features of ``torchrun``.


Usage
--------

Single-node multi-worker
++++++++++++++++++++++++++++++

::

    torchrun
        --standalone
        --nnodes=1
        --nproc-per-node=$NUM_TRAINERS
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

Stacked single-node multi-worker
+++++++++++++++++++++++++++++++++++

To run multiple instances (separate jobs) of single-node, multi-worker on the
same host, we need to make sure that each instance (job) is
setup on different ports to avoid port conflicts (or worse, two jobs being merged
as a single job). To do this you have to run with ``--rdzv-backend=c10d``
and specify a different port by setting ``--rdzv-endpoint=localhost:$PORT_k``.
For ``--nodes=1``, its often convenient to let ``torchrun`` pick a free random
port automatically instead of manually assigning different ports for each run.

::

    torchrun
        --rdzv-backend=c10d
        --rdzv-endpoint=localhost:0
        --nnodes=1
        --nproc-per-node=$NUM_TRAINERS
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)


Fault tolerant (fixed sized number of workers, no elasticity, tolerates 3 failures)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::

    torchrun
        --nnodes=$NUM_NODES
        --nproc-per-node=$NUM_TRAINERS
        --max-restarts=3
        --rdzv-id=$JOB_ID
        --rdzv-backend=c10d
        --rdzv-endpoint=$HOST_NODE_ADDR
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

``HOST_NODE_ADDR``, in form <host>[:<port>] (e.g. node1.example.com:29400), specifies the node and
the port on which the C10d rendezvous backend should be instantiated and hosted. It can be any
node in your training cluster, but ideally you should pick a node that has a high bandwidth.

.. note::
   If no port number is specified ``HOST_NODE_ADDR`` defaults to 29400.

Elastic (``min=1``, ``max=4``, tolerates up to 3 membership changes or failures)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::

    torchrun
        --nnodes=1:4
        --nproc-per-node=$NUM_TRAINERS
        --max-restarts=3
        --rdzv-id=$JOB_ID
        --rdzv-backend=c10d
        --rdzv-endpoint=$HOST_NODE_ADDR
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

``HOST_NODE_ADDR``, in form <host>[:<port>] (e.g. node1.example.com:29400), specifies the node and
the port on which the C10d rendezvous backend should be instantiated and hosted. It can be any
node in your training cluster, but ideally you should pick a node that has a high bandwidth.

.. note::
   If no port number is specified ``HOST_NODE_ADDR`` defaults to 29400.

Note on rendezvous backend
------------------------------

For multi-node training you need to specify:

1. ``--rdzv-id``: A unique job id (shared by all nodes participating in the job)
2. ``--rdzv-backend``: An implementation of
   :py:class:`torch.distributed.elastic.rendezvous.RendezvousHandler`
3. ``--rdzv-endpoint``: The endpoint where the rendezvous backend is running; usually in form
   ``host:port``.

Currently ``c10d`` (recommended), ``etcd-v2``, and ``etcd`` (legacy)  rendezvous backends are
supported out of the box. To use ``etcd-v2`` or ``etcd``, setup an etcd server with the ``v2`` api
enabled (e.g. ``--enable-v2``).

.. warning::
   ``etcd-v2`` and ``etcd`` rendezvous use etcd API v2. You MUST enable the v2 API on the etcd
   server. Our tests use etcd v3.4.3.

.. warning::
   For etcd-based rendezvous we recommend using ``etcd-v2`` over ``etcd`` which is functionally
   equivalent, but uses a revised implementation. ``etcd`` is in maintenance mode and will be
   removed in a future version.

Definitions
--------------

1. ``Node`` - A physical instance or a container; maps to the unit that the job manager works with.

2. ``Worker`` - A worker in the context of distributed training.

3. ``WorkerGroup`` - The set of workers that execute the same function (e.g. trainers).

4. ``LocalWorkerGroup`` - A subset of the workers in the worker group running on the same node.

5. ``RANK`` - The rank of the worker within a worker group.

6. ``WORLD_SIZE`` - The total number of workers in a worker group.

7. ``LOCAL_RANK`` - The rank of the worker within a local worker group.

8. ``LOCAL_WORLD_SIZE`` - The size of the local worker group.

9. ``rdzv_id`` - A user-defined id that uniquely identifies the worker group for a job. This id is
   used by each node to join as a member of a particular worker group.

9. ``rdzv_backend`` - The backend of the rendezvous (e.g. ``c10d``). This is typically a strongly
   consistent key-value store.

10. ``rdzv_endpoint`` - The rendezvous backend endpoint; usually in form ``<host>:<port>``.

A ``Node`` runs ``LOCAL_WORLD_SIZE`` workers which comprise a ``LocalWorkerGroup``. The union of
all ``LocalWorkerGroups`` in the nodes in the job comprise the ``WorkerGroup``.

Environment Variables
----------------------

The following environment variables are made available to you in your script:

1. ``LOCAL_RANK`` -  The local rank.

2. ``RANK`` -  The global rank.

3. ``GROUP_RANK`` - The rank of the worker group. A number between 0 and ``max_nnodes``. When
   running a single worker group per node, this is the rank of the node.

4. ``ROLE_RANK`` -  The rank of the worker across all the workers that have the same role. The role
   of the worker is specified in the ``WorkerSpec``.

5. ``LOCAL_WORLD_SIZE`` - The local world size (e.g. number of workers running locally); equals to
   ``--nproc-per-node`` specified on ``torchrun``.

6. ``WORLD_SIZE`` - The world size (total number of workers in the job).

7. ``ROLE_WORLD_SIZE`` - The total number of workers that was launched with the same role specified
   in ``WorkerSpec``.

8. ``MASTER_ADDR`` - The FQDN of the host that is running worker with rank 0; used to initialize
   the Torch Distributed backend.

9. ``MASTER_PORT`` - The port on the ``MASTER_ADDR`` that can be used to host the C10d TCP store.

10. ``TORCHELASTIC_RESTART_COUNT`` - The number of worker group restarts so far.

11. ``TORCHELASTIC_MAX_RESTARTS`` - The configured maximum number of restarts.

12. ``TORCHELASTIC_RUN_ID`` - Equal to the rendezvous ``run_id`` (e.g. unique job id).

13. ``PYTHON_EXEC`` - System executable override. If provided, the python user script will
    use the value of ``PYTHON_EXEC`` as executable. The `sys.executable` is used by default.

Deployment
------------

1. (Not needed for the C10d backend) Start the rendezvous backend server and get the endpoint (to be
   passed as ``--rdzv-endpoint`` to the launcher script)

2. Single-node multi-worker: Start the launcher on the host to start the agent process which
   creates and monitors a local worker group.

3. Multi-node multi-worker: Start the launcher with the same arguments on all the nodes
   participating in training.

When using a job/cluster manager the entry point command to the multi-node job should be this
launcher.

Failure Modes
---------------

1. Worker failure: For a training job with ``n`` workers, if ``k<=n`` workers fail all workers
   are stopped and restarted up to ``max_restarts``.

2. Agent failure: An agent failure results in a local worker group failure. It is up to the job
   manager to fail the entire job (gang semantics) or attempt to replace the node. Both behaviors
   are supported by the agent.

3. Node failure: Same as agent failure.

Membership Changes
--------------------

1. Node departure (scale-down): The agent is notified of the departure, all existing workers are
   stopped, a new ``WorkerGroup`` is formed, and all workers are started with a new ``RANK`` and
   ``WORLD_SIZE``.

2. Node arrival (scale-up): The new node is admitted to the job, all existing workers are stopped,
   a new ``WorkerGroup`` is formed, and all workers are started with a new ``RANK`` and
   ``WORLD_SIZE``.

Important Notices
--------------------

1. This utility and multi-process distributed (single-node or
   multi-node) GPU training currently only achieves the best performance using
   the NCCL distributed backend. Thus NCCL backend is the recommended backend to
   use for GPU training.

2. The environment variables necessary to initialize a Torch process group are provided to you by
   this module, no need for you to pass ``RANK`` manually.  To initialize a process group in your
   training script, simply run:

::

 >>> # xdoctest: +SKIP("stub")
 >>> import torch.distributed as dist
 >>> dist.init_process_group(backend="gloo|nccl")

3. In your training program, you can either use regular distributed functions
   or use :func:`torch.nn.parallel.DistributedDataParallel` module. If your
   training program uses GPUs for training and you would like to use
   :func:`torch.nn.parallel.DistributedDataParallel` module,
   here is how to configure it.

::

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)

Please ensure that ``device_ids`` argument is set to be the only GPU device id
that your code will be operating on. This is generally the local rank of the
process. In other words, the ``device_ids`` needs to be ``[int(os.environ("LOCAL_RANK"))]``,
and ``output_device`` needs to be ``int(os.environ("LOCAL_RANK"))`` in order to use this
utility


4. On failures or membership changes ALL surviving workers are killed immediately. Make sure to
   checkpoint your progress. The frequency of checkpoints should depend on your job's tolerance
   for lost work.

5. This module only supports homogeneous ``LOCAL_WORLD_SIZE``. That is, it is assumed that all
   nodes run the same number of local workers (per role).

6. ``RANK`` is NOT stable. Between restarts, the local workers on a node can be assigned a
   different range of ranks than before. NEVER hard code any assumptions about the stable-ness of
   ranks or some correlation between ``RANK`` and ``LOCAL_RANK``.

7. When using elasticity (``min_size!=max_size``) DO NOT hard code assumptions about
   ``WORLD_SIZE`` as the world size can change as nodes are allowed to leave and join.

8. It is recommended for your script to have the following structure:

::

  def main():
    load_checkpoint(checkpoint_path)
    initialize()
    train()

  def train():
    for batch in iter(dataset):
      train_step(batch)

      if should_checkpoint:
        save_checkpoint(checkpoint_path)

9. (Recommended) On worker errors, this tool will summarize the details of the error
   (e.g. time, rank, host, pid, traceback, etc). On each node, the first error (by timestamp)
   is heuristically reported as the "Root Cause" error. To get tracebacks as part of this
   error summary print out, you must decorate your main entrypoint function in your
   training script as shown in the example below. If not decorated, then the summary
   will not include the traceback of the exception and will only contain the exitcode.
   For details on torchelastic error handling see: https://pytorch.org/docs/stable/elastic/errors.html

::

  from torch.distributed.elastic.multiprocessing.errors import record

  @record
  def main():
      # do train
      pass

  if __name__ == "__main__":
      main()

"""


def get_args_parser() -> ArgumentParser:
    """Parse the command line options."""
    parser = ArgumentParser(description="Torch Distributed Elastic Training Launcher")

    #
    # Worker/node size related arguments.
    #

    parser.add_argument(
        "--nnodes",
        action=env,
        type=str,
        default="1:1",
        help="Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>.",
    )
    parser.add_argument(
        "--nproc-per-node",
        "--nproc_per_node",
        action=env,
        type=str,
        default="1",
        help="Number of workers per node; supported values: [auto, cpu, gpu, int].",
    )

    #
    # Rendezvous related arguments
    #

    parser.add_argument(
        "--rdzv-backend",
        "--rdzv_backend",
        action=env,
        type=str,
        default="c10d",
        help="Rendezvous backend. Currently only c10d is supported.",
    )
    parser.add_argument(
        "--rdzv-endpoint",
        "--rdzv_endpoint",
        action=env,
        type=str,
        default="",
        help="Rendezvous backend endpoint; usually in form <host>:<port>.",
    )
    parser.add_argument(
        "--rdzv-id",
        "--rdzv_id",
        action=env,
        type=str,
        default="none",
        help="User-defined group id.",
    )
    parser.add_argument(
        "--rdzv-conf",
        "--rdzv_conf",
        action=env,
        type=str,
        default="",
        help="Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).",
    )
    parser.add_argument(
        "--standalone",
        action=check_env,
        help="Start a local standalone rendezvous backend that is represented by a C10d TCP store "
        "on a free port. Useful when launching single-node, multi-worker job. If specified "
        "--rdzv-backend, --rdzv-endpoint, --rdzv-id are auto-assigned and any explicitly set values "
        "are ignored.",
    )

    #
    # User-code launch related arguments.
    #

    parser.add_argument(
        "--max-restarts",
        "--max_restarts",
        action=env,
        type=int,
        default=0,
        help="Maximum number of worker group restarts before failing.",
    )
    parser.add_argument(
        "--term-timeout",
        "--term_timeout",
        action=env,
        type=float,
        default=1800,
        help="Interval, in seconds, between initial SIGTERM and rank termination with SIGKILL, when the launcher forwards a received signal to ranks.",
    )
    parser.add_argument(
        "--workers-stop-timeout",
        "--workers_stop_timeout",
        action=env,
        type=float,
        default=30,
        help="Interval, in seconds, between initial SIGTERM and rank termination with SIGKILL, when the launcher stops its ranks in order to restart them.",
    )
    parser.add_argument(
        "--monitor-interval",
        "--monitor_interval",
        action=env,
        type=float,
        default=5,
        help="Interval, in seconds, to monitor the state of workers.",
    )
    parser.add_argument(
        "--start-method",
        "--start_method",
        action=env,
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method to use when creating workers.",
    )
    parser.add_argument(
        "--role",
        action=env,
        type=str,
        default="default",
        help="User-defined role for the workers.",
    )
    parser.add_argument(
        "-m",
        "--module",
        action=check_env,
        help="Change each process to interpret the launch script as a Python module, executing "
        "with the same behavior as 'python -m'.",
    )
    parser.add_argument(
        "--no-python",
        "--no_python",
        action=check_env,
        help="Skip prepending the training script with 'python' - just execute it directly. Useful "
        "when the script is not a Python script.",
    )

    parser.add_argument(
        "--run-path",
        "--run_path",
        action=check_env,
        help="Run the training script with runpy.run_path in the same interpreter."
        " Script must be provided as an abs path (e.g. /abs/path/script.py)."
        " Takes precedence over --no-python.",
    )
    parser.add_argument(
        "--log-dir",
        "--log_dir",
        action=env,
        type=str,
        default=None,
        help="Base directory to use for log files (e.g. /var/log/torch/elastic). The same "
        "directory is re-used for multiple runs (a unique job-level sub-directory is created with "
        "rdzv_id as the prefix).",
    )
    parser.add_argument(
        "-r",
        "--redirects",
        action=env,
        type=str,
        default="0",
        help="Redirect std streams into a log file in the log directory (e.g. [-r 3] redirects "
        "both stdout+stderr for all workers, [-r 0:1,1:2] redirects stdout for local rank 0 and "
        "stderr for local rank 1).",
    )
    parser.add_argument(
        "-t",
        "--tee",
        action=env,
        type=str,
        default="0",
        help="Tee std streams into a log file and also to console (see --redirects for format).",
    )

    parser.add_argument(
        "--local-ranks-filter",
        "--local_ranks_filter",
        action=env,
        type=str,
        default="",
        help="Only show logs from specified ranks in console (e.g. [--local_ranks_filter=0,1,2] will "
        "only show logs from rank 0, 1 and 2). This will only apply to stdout and stderr, not to"
        "log files saved via --redirect or --tee",
    )

    #
    # Backwards compatible parameters with caffe2.distributed.launch.
    #

    parser.add_argument(
        "--node-rank",
        "--node_rank",
        type=int,
        action=env,
        default=0,
        help="Rank of the node for multi-node distributed training.",
    )
    parser.add_argument(
        "--master-addr",
        "--master_addr",
        default="127.0.0.1",
        type=str,
        action=env,
        help="Address of the master node (rank 0) that only used for static rendezvous. It should "
        "be either the IP address or the hostname of rank 0. For single node multi-proc training "
        "the --master-addr can simply be 127.0.0.1; IPv6 should have the pattern "
        "`[0:0:0:0:0:0:0:1]`.",
    )
    parser.add_argument(
        "--master-port",
        "--master_port",
        default=29500,
        type=int,
        action=env,
        help="Port on the master node (rank 0) to be used for communication during distributed "
        "training. It is only used for static rendezvous.",
    )
    parser.add_argument(
        "--local-addr",
        "--local_addr",
        default=None,
        type=str,
        action=env,
        help="Address of the local node. If specified, will use the given address for connection. "
        "Else, will look up the local node address instead. Else, it will be default to local "
        "machine's FQDN.",
    )

    parser.add_argument(
        "--logs-specs",
        "--logs_specs",
        default=None,
        type=str,
        help="torchrun.logs_specs group entrypoint name, value must be type of LogsSpecs. "
        "Can be used to override custom logging behavior.",
    )

    #
    # Fault tolerance related items
    #
    parser.add_argument(
        "--fault-tol-cfg-path",
        "--fault-tol-cfg-path",
        default=None,
        type=str,
        action=env,
        help="Path to a YAML file that contains Fault Tolerance pkg config (`fault_tolerance` section)."
        " NOTE: config items from the file can be overwritten by `--ft-param-*` args.",
    )

    parser.add_argument(
        "--ignore-missing-fault-tol-cfg",
        "--ignore-missing-fault-tol-cfg",
        action='store_true',
        help="Do not raise an error if there is no Fault Tolerance pkg config provided, just use default settings.",
    )

    parser.add_argument(
        "--ft-param-workload_check_interval",
        "--ft-param-workload_check_interval",
        type=float,
        default=None,
        help="Part of Fault Tolerance pkg config (workload_check_interval).",
    )

    parser.add_argument(
        "--ft-param-initial_rank_heartbeat_timeout",
        "--ft-param-initial_rank_heartbeat_timeout",
        type=float,
        default=None,
        help="Part of Fault Tolerance pkg config (initial_rank_heartbeat_timeout).",
    )

    parser.add_argument(
        "--ft-param-rank_heartbeat_timeout",
        "--ft-param-rank_heartbeat_timeout",
        type=float,
        default=None,
        help="Part of Fault Tolerance pkg config (rank_heartbeat_timeout).",
    )

    parser.add_argument(
        "--ft-param-node_health_check_interval",
        "--ft-param-node_health_check_interval",
        type=float,
        default=None,
        help="Part of Fault Tolerance pkg config (node_health_check_interval).",
    )

    parser.add_argument(
        "--ft-param-safety_factor",
        "--ft-param-safety_factor",
        type=float,
        default=None,
        help="Part of Fault Tolerance pkg config (safety_factor).",
    )

    parser.add_argument(
        "--ft-param-rank_termination_signal",
        "--ft-param-rank_termination_signal",
        type=str,
        default=None,
        help="Part of Fault Tolerance pkg config (rank_termination_signal).",
    )

    parser.add_argument(
        "--ft-param-log_level",
        "--ft-param-log_level",
        type=str,
        default=None,
        help="Part of Fault Tolerance pkg config (log_level).",
    )

    parser.add_argument(
        "--ft-param-rank_out_of_section_timeout",
        "--ft-param-rank_out_of_section_timeout",
        type=float,
        default=None,
        help="Part of Fault Tolerance pkg config (rank_out_of_section_timeout).",
    )

    parser.add_argument(
        "--ft-param-rank_section_timeouts",
        "--ft-param-rank_section_timeouts",
        type=str,
        default=None,
        help="Part of Fault Tolerance pkg config (rank_section_timeouts). "
        "Expected format: name1:value1,name2:value2,... Use 'null'|'none'|'' for None",
    )

    parser.add_argument(
        "--ft-param-restart_check_interval",
        "--ft-param-restart_check_interval",
        type=float,
        default=None,
        help="Part of Fault Tolerance pkg config (restart_check_interval).",
    )

    parser.add_argument(
        "--restart-policy",
        "--restart_policy",
        type=str,
        choices=['any-failed', 'min-healthy'],
        default='any-failed',
        help="Worker groups restarting policy. Options: "
        "'any-failed' restart if any worker group fails (torchrun's default); "
        "'min-healthy' restart if number of healthy worker groups falls below <minimum_nodes>",
    )

    parser.add_argument(
        "--ft_param_enable_nic_monitor",
        "--ft-param-enable-nic-monitor",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
        help="Enable or Disable NIC health monitoring in training.",
    )

    parser.add_argument(
        "--ft_param_pci_topo_file",
        "--ft-param-pci-topo-file",
        type=str,
        default=None,
        help="PCI topology file that describes GPU and NIC topology.",
    )

    parser.add_argument(
        "--ft_param_link_down_path_template",
        "--ft-param-link-down-path-template",
        type=str,
        default=None,
        help="Part of Fault Tolerance pkg config (link_down_path_template). "
        "Template path to check if a NIC link is down.",
    )

    #
    # Positional arguments.
    #

    parser.add_argument(
        "training_script",
        type=str,
        help="Full path to the (single GPU) training program/script to be launched in parallel, "
        "followed by all the arguments for the training script.",
    )

    # Rest from the training program.
    parser.add_argument("training_script_args", nargs=REMAINDER)

    return parser


def parse_args(args):
    parser = get_args_parser()
    return parser.parse_args(args)


def parse_min_max_nnodes(nnodes: str):
    arr = nnodes.split(":")

    if len(arr) == 1:
        min_nodes = max_nodes = int(arr[0])
    elif len(arr) == 2:
        min_nodes = int(arr[0])
        max_nodes = int(arr[1])
    else:
        raise RuntimeError(f'nnodes={nnodes} is not in "MIN:MAX" format')  # noqa: E231

    return min_nodes, max_nodes


def determine_local_world_size(nproc_per_node: str):
    try:
        logger.info("Using nproc_per_node=%s.", nproc_per_node)
        return int(nproc_per_node)
    except ValueError as e:
        if nproc_per_node == "cpu":
            num_proc = os.cpu_count()
            device_type = "cpu"
        elif nproc_per_node == "gpu":
            if not torch.cuda.is_available():
                raise ValueError("Cuda is not available.") from e
            device_type = "gpu"
            num_proc = torch.cuda.device_count()
        else:
            raise ValueError(f"Unsupported nproc_per_node value: {nproc_per_node}") from e

        logger.info(
            "Using nproc_per_node=%s," " setting to %s since the instance " "has %s %s",
            nproc_per_node,
            num_proc,
            os.cpu_count(),
            device_type,
        )
        return num_proc


def get_rdzv_endpoint(args):
    if args.rdzv_backend == "static" and not args.rdzv_endpoint:
        return f"{args.master_addr}:{args.master_port}"  # noqa: E231
    return args.rdzv_endpoint


def get_use_env(args) -> bool:
    """
    Retrieve ``use_env`` from the args.
    ``use_env`` is a legacy argument, if ``use_env`` is False, the
    ``--node-rank`` argument will be transferred to all worker processes.
    ``use_env`` is only used by the ``torch.distributed.launch`` and will
    be deprecated in future releases.
    """
    if not hasattr(args, "use_env"):
        return True
    return args.use_env


def _get_logs_specs_class(logs_specs_name: Optional[str]) -> Type[LogsSpecs]:
    """
    Attemps to load `torchrun.logs_spec` entrypoint with key of `logs_specs_name` param.
    Provides plugin mechanism to provide custom implementation of LogsSpecs.

    Returns `DefaultLogsSpecs` when logs_spec_name is None.
    Raises ValueError when entrypoint for `logs_spec_name` can't be found in entrypoints.
    """
    logs_specs_cls = None
    if logs_specs_name is not None:
        eps = metadata.entry_points()
        if hasattr(eps, "select"):  # >= 3.10
            group = eps.select(group="torchrun.logs_specs")
            if group.select(name=logs_specs_name):
                logs_specs_cls = group[logs_specs_name].load()

        elif specs := eps.get("torchrun.logs_specs"):  # < 3.10
            if entrypoint_list := [ep for ep in specs if ep.name == logs_specs_name]:
                logs_specs_cls = entrypoint_list[0].load()

        if logs_specs_cls is None:
            raise ValueError(
                f"Could not find entrypoint under 'torchrun.logs_specs[{logs_specs_name}]' key"
            )

        logger.info("Using logs_spec '%s' mapped to %s", logs_specs_name, str(logs_specs_cls))
    else:
        logs_specs_cls = DefaultLogsSpecs

    return logs_specs_cls


def config_from_args(args) -> Tuple[LaunchConfig, Union[Callable, str], List[str]]:
    # If ``args`` not passed, defaults to ``sys.argv[:1]``
    min_nodes, max_nodes = parse_min_max_nnodes(args.nnodes)
    assert 0 < min_nodes <= max_nodes
    assert args.max_restarts >= 0

    if hasattr(args, "master_addr") and args.rdzv_backend != "static" and not args.rdzv_endpoint:
        logger.warning(
            "master_addr is only used for static rdzv_backend and when rdzv_endpoint "
            "is not specified."
        )

    nproc_per_node = determine_local_world_size(args.nproc_per_node)
    if "OMP_NUM_THREADS" not in os.environ and nproc_per_node > 1:
        omp_num_threads = 1
        logger.warning(
            "\n*****************************************\n"
            "Setting OMP_NUM_THREADS environment variable for each process to be "
            "%s in default, to avoid your system being overloaded, "
            "please further tune the variable for optimal performance in "
            "your application as needed. \n"
            "*****************************************",
            omp_num_threads,
        )
        # This env variable will be passed down to the subprocesses
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    log_line_prefix_template = os.getenv("TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE")

    rdzv_configs = _parse_rendezvous_config(args.rdzv_conf)

    if args.rdzv_backend == "static":
        rdzv_configs["rank"] = args.node_rank

    rdzv_endpoint = get_rdzv_endpoint(args)

    if args.rdzv_backend.lower() != 'c10d':
        raise ValueError(
            f"Current ft_launcher version supports only rdzv_backend=c10d. Got {args.rdzv_backend}"
        )

    try:
        fault_tol_cfg = FaultToleranceConfig.from_args(
            args,
            cfg_file_arg="fault_tol_cfg_path",
            ft_args_prefix="ft_param_",
        )
    except ValueError:
        if args.ignore_missing_fault_tol_cfg:
            logger.warning(
                f"Could not load FT config from '{args.fault_tol_cfg_path}' or read from CLI args. Will use default FT settings."
            )
            fault_tol_cfg = FaultToleranceConfig()
        else:
            raise ValueError("Fault Tolerance configuration not provided.")

    ranks: Optional[Set[int]] = None
    if args.local_ranks_filter:
        try:
            ranks = set(map(int, args.local_ranks_filter.split(",")))
            assert ranks
        except Exception as e:
            raise Exception(
                "--local_ranks_filter must be a comma-separated list of integers e.g. --local_ranks_filter=0,1,2"
            ) from e

    logs_specs_cls: Type[LogsSpecs] = _get_logs_specs_class(args.logs_specs)
    logs_specs = logs_specs_cls(
        log_dir=args.log_dir,
        redirects=Std.from_str(args.redirects),
        tee=Std.from_str(args.tee),
        local_ranks_filter=ranks,
    )

    config = LaunchConfig(
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        nproc_per_node=nproc_per_node,
        run_id=args.rdzv_id,
        role=args.role,
        rdzv_endpoint=rdzv_endpoint,
        rdzv_backend=args.rdzv_backend,
        rdzv_configs=rdzv_configs,
        max_restarts=args.max_restarts,
        restart_policy=args.restart_policy,
        term_timeout=args.term_timeout,
        workers_stop_timeout=args.workers_stop_timeout,
        monitor_interval=args.monitor_interval,
        start_method=args.start_method,
        log_line_prefix_template=log_line_prefix_template,
        local_addr=args.local_addr,
        logs_specs=logs_specs,
        fault_tol_cfg=fault_tol_cfg,
    )

    with_python = not args.no_python
    cmd: Union[Callable, str]
    cmd_args = []
    use_env = get_use_env(args)
    if args.run_path:
        cmd = run_script_path
        cmd_args.append(args.training_script)
    else:
        if with_python:
            cmd = os.getenv("PYTHON_EXEC", sys.executable)
            cmd_args.append("-u")
            if args.module:
                cmd_args.append("-m")
            cmd_args.append(args.training_script)
        else:
            if args.module:
                raise ValueError(
                    "Don't use both the '--no-python' flag"
                    " and the '--module' flag at the same time."
                )
            cmd = args.training_script
    if not use_env:
        cmd_args.append(f"--local-rank={macros.local_rank}")
    cmd_args.extend(args.training_script_args)

    return config, cmd, cmd_args


def run_script_path(training_script: str, *training_script_args: str):
    """
    Run the provided `training_script` from within this interpreter.

    Usage: `script_as_function("/abs/path/to/script.py", "--arg1", "val1")`
    """
    import runpy
    import sys

    sys.argv = [training_script] + [*training_script_args]
    runpy.run_path(sys.argv[0], run_name="__main__")


def run(args):
    if args.standalone:
        args.rdzv_backend = "c10d"
        args.rdzv_endpoint = "localhost:0"
        args.rdzv_id = str(uuid.uuid4())
        logger.info(
            "\n**************************************\n"
            "Rendezvous info:\n"
            "--rdzv-backend=%s "
            "--rdzv-endpoint=%s "
            "--rdzv-id=%s\n"
            "**************************************\n",
            args.rdzv_backend,
            args.rdzv_endpoint,
            args.rdzv_id,
        )

    _register_ft_rdzv_handler()
    config, cmd, cmd_args = config_from_args(args)
    elastic_launch(
        config=config,
        entrypoint=cmd,
    )(*cmd_args)


@record
def main(args=None):
    args = parse_args(args)
    try:
        run(args)
    except ChildFailedError as e:
        logger.error(
            f"Some rank(s) exited with non-zero exit code: {e.failures}. Agent's exit code = 1"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Agent run ended with exception, {e=}. Agent's exit code = 1")
        sys.exit(1)
    logger.info("Agent exits with exit code = 0.")
    sys.exit(0)


if __name__ == "__main__":
    main()
