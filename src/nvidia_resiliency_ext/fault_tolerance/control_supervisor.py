# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Control-host lifecycle, with an optional node-agent for colocated deployment.

Return the generation's exit status to the caller for current-array cleanup.
The supervisor does not cancel queued singleton successors.
"""

import argparse
import contextlib
import ctypes
import logging
import os
import signal
import subprocess
import time

import psutil

from ..shared_utils.log_manager import LogConfig, setup_logger
from .attribution_manager import DEFAULT_ATTRIBUTION_PORT, AttributionConfig, AttributionManager
from .child_process import spawn_child, wait_or_kill_child
from .cli_args import default_rendezvous_timeouts
from .config import FaultToleranceConfig
from .control_logging import ControlLogCapture
from .log_funnel import (
    LogFunnelPorts,
    _start_grpc_log_servers,
    _validate_managed_attribution_listen_port_not_in_log_funnel,
    stop_grpc_log_servers,
)

logger = logging.getLogger(LogConfig.name)

DESCENDANT_CLEANUP_TIMEOUT = 2.0
NODE_AGENT_CLEANUP_GRACE_SECONDS = 15.0
_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37


def _rdzv_config(args):
    from torch.distributed.elastic.rendezvous.utils import _parse_rendezvous_config

    return _parse_rendezvous_config(args.rdzv_conf)


def _node_agent_args(args, endpoint):
    values = vars(args).copy()
    configs = _rdzv_config(args)
    configs["is_host"] = "false"
    values.update(
        rdzv_endpoint=endpoint,
        rdzv_conf=",".join(f"{key}={value}" for key, value in configs.items()),
        _is_control_host_agent=True,
    )
    return argparse.Namespace(**values)


def _job_controller_args(args):
    from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint

    from .launcher import get_rdzv_endpoint

    values = vars(args).copy()
    configs = _rdzv_config(args)
    default_rendezvous_timeouts(configs)
    # c10d accepts a host without a port, defaulting to 29400. Resolve it
    # before passing the endpoint to the controller and optional node-agent.
    host, port = parse_rendezvous_endpoint(get_rdzv_endpoint(args), default_port=29400)
    if port == 0:
        raise ValueError(
            "The control supervisor requires a fixed rendezvous port; use --standalone for a single-node run."
        )
    values.update(
        rdzv_conf=",".join(f"{key}={value}" for key, value in configs.items()),
        rdzv_endpoint=f"[{host}]:{port}" if ":" in host else f"{host}:{port}",
    )
    return argparse.Namespace(**values)


class _ShutdownRequested(Exception):
    pass


def _set_subreaper(enabled):
    """Adopt orphaned workers even when they create their own sessions (Linux)."""
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(previous), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER failed")
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, int(enabled), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")
    return bool(previous.value)


class ControlSupervisor:
    def __init__(self, args):
        self.args = args
        self.capture = None
        self.grpc_processes = []
        self.attribution = None
        self.job_controller = None
        self.node_agent = None
        self.signum = None
        self.stopping = False

    def _raise_if_shutdown_requested(self):
        if self.signum is not None:
            raise _ShutdownRequested()

    def _auxiliary_processes(self):
        procs = list(self.grpc_processes)
        if self.attribution is not None and self.attribution.process is not None:
            procs.append(self.attribution.process)
        return procs

    def _failure_exit_code(self):
        children = [("node-agent", self.node_agent), ("job-controller", self.job_controller)]
        children.extend(("auxiliary service", proc) for proc in self._auxiliary_processes())
        for name, proc in children:
            if proc is None:
                continue
            code = proc.poll()
            if code is None or (name != "auxiliary service" and code == 0):
                continue
            logger.error("%s PID=%s exited unexpectedly: returncode=%s", name, proc.pid, code)
            # A persistent auxiliary service exiting zero is still unavailable.
            return 128 - code if code < 0 else code or 1
        return None

    def _start(self):
        args = self.args
        ft_cfg = FaultToleranceConfig.from_args(args)
        if args.ft_nvrx_logfile and args.ft_enable_log_server is None:
            args.ft_enable_log_server = True
        self.capture = ControlLogCapture(args.ft_nvrx_logfile)
        self.capture.redirect_supervisor()
        setup_logger()
        logger.info("Starting control-host supervisor; combined process log=%s", self.capture.path)

        ports = None
        if args.ft_enable_log_server and (args.ft_per_cycle_applog_prefix or args.ft_nvrx_logfile):
            ports = LogFunnelPorts.from_launcher_args(args)
            self.grpc_processes = _start_grpc_log_servers(args, ports, capture=self.capture)
            if not self.grpc_processes:
                raise RuntimeError("failed to start gRPC log aggregation")

        if ft_cfg.attribution_endpoint:
            if not args.ft_per_cycle_applog_prefix:
                raise ValueError("attribution requires --ft-per-cycle-applog-prefix")
            cfg = AttributionConfig.from_args(args, args.ft_per_cycle_applog_prefix, ft_cfg)
            if cfg.is_managed:
                if ports is not None:
                    _validate_managed_attribution_listen_port_not_in_log_funnel(
                        DEFAULT_ATTRIBUTION_PORT, ports
                    )
                self.attribution = AttributionManager(cfg, is_store_host=True)
                self.attribution.start_if_needed(
                    capture=self.capture,
                    raise_if_shutdown_requested=self._raise_if_shutdown_requested,
                )

        self._raise_if_shutdown_requested()
        control_args = _job_controller_args(args)
        self.job_controller = spawn_child(
            "nvidia_resiliency_ext.fault_tolerance.job_controller",
            control_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.capture.attach(self.job_controller.stdout)
        if getattr(args, "ft_control_only", False):
            logger.info("External control host: no local node-agent is started")
            return
        agent_args = _node_agent_args(args, control_args.rdzv_endpoint)
        # Do not inherit the supervisor's control-log pipe: the agent retains the
        # existing compute log path and original console fallback.
        self.node_agent = spawn_child(
            "nvidia_resiliency_ext.fault_tolerance.node_agent",
            agent_args,
            stdout=self.capture.stdout_fd,
            stderr=self.capture.stderr_fd,
            start_new_session=True,
        )
        logger.info("node-agent started: PID=%s", self.node_agent.pid)

    def _monitor(self):
        while True:
            self._raise_if_shutdown_requested()
            failure = self._failure_exit_code()
            if failure is not None:
                return failure
            codes = [self.job_controller.poll()]
            if self.node_agent is not None:
                codes.append(self.node_agent.poll())
            if all(code == 0 for code in codes):
                return 0
            time.sleep(0.1)

    def _stop_node_agent(self, *, terminate=True):
        if self.node_agent is None:
            return
        # Workers create their own sessions. The supervisor is a subreaper, so
        # descendants orphaned by a crash are adopted here, not lost to PID 1.
        if self.node_agent.poll() is None:
            if terminate:
                with contextlib.suppress(ProcessLookupError):
                    self.node_agent.terminate()
            # term_timeout belongs to the workers. Allow the agent time afterward
            # to stop monitors and drain logs before stopping the aggregators.
            wait_or_kill_child(
                self.node_agent,
                float(self.args.term_timeout) + NODE_AGENT_CLEANUP_GRACE_SECONDS,
                process_group=True,
            )
        deadline = time.monotonic() + DESCENDANT_CLEANUP_TIMEOUT
        descendants = self._agent_descendants(deadline)
        for proc in descendants:
            if time.monotonic() >= deadline:
                logger.error("Descendant cleanup deadline reached before all signals were sent")
                break
            with contextlib.suppress(psutil.NoSuchProcess):
                proc.kill()
        if descendants:
            _, alive = psutil.wait_procs(descendants, timeout=max(0.0, deadline - time.monotonic()))
            if alive:
                logger.error("Descendants survived cleanup: PIDs=%s", [proc.pid for proc in alive])

    def _agent_descendants(self, deadline):
        # One local PID/parent snapshot, followed by an in-memory walk. Calling
        # children(recursive=True) separately for each orphan rescans all PIDs.
        by_parent = {}
        for proc in psutil.process_iter(["pid", "ppid"]):
            if time.monotonic() >= deadline:
                logger.error(
                    "Process discovery deadline reached; descendant snapshot is incomplete"
                )
                break
            by_parent.setdefault(proc.info["ppid"], []).append(proc)
        protected = {p.pid for p in self._auxiliary_processes()}
        if self.job_controller is not None:
            protected.add(self.job_controller.pid)
        pending = [os.getpid()]
        if self.node_agent.poll() is None:
            pending.append(self.node_agent.pid)
        seen = set(protected)
        descendants = {}
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            for proc in by_parent.get(pid, ()):
                if proc.pid not in seen:
                    if proc.pid != self.node_agent.pid:
                        descendants[proc.pid] = proc
                    pending.append(proc.pid)
        return list(descendants.values())

    def _cleanup(self):
        # Stop the critical children together. Attribution outlives its controller
        # client, and gRPC servers outlive the agent's final log drain.
        for proc in (self.node_agent, self.job_controller):
            if proc is not None and proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()

        operations = [lambda: self._stop_node_agent(terminate=False)]
        if self.job_controller is not None:

            def stop_control():
                if self.job_controller.poll() is None:
                    wait_or_kill_child(self.job_controller, float(self.args.term_timeout))

            operations.append(stop_control)
        if self.attribution is not None:
            operations.append(self.attribution.stop)
        if self.grpc_processes:
            operations.append(
                lambda: stop_grpc_log_servers(
                    self.grpc_processes, float(self.args.ft_log_server_graceful_shutdown_timeout)
                )
            )
        for operation in operations:
            try:
                operation()
            except Exception:
                logger.exception("Supervisor cleanup failed")

    def run(self):
        # Match the catchable termination signals handled by the worker-launching
        # agent so targeting ft_launcher still cleans up its child processes.
        previous = {
            sig: signal.getsignal(sig)
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)
        }
        for sig in previous:
            signal.signal(sig, lambda signum, _frame: setattr(self, "signum", signum))
        code = 1
        previous_subreaper = None
        try:
            previous_subreaper = _set_subreaper(True)
            self._start()
            code = self._monitor()
        except _ShutdownRequested:
            code = 128 + self.signum
            logger.info("Supervisor received signal %s", self.signum)
        except Exception:
            code = self._failure_exit_code() or 1
            logger.exception("Control-host supervisor failed")
        finally:
            # Disable future restart policies before sending any shutdown signals.
            self.stopping = True
            try:
                self._cleanup()
                logger.info("Control-host supervisor final exit code=%s", code)
            finally:
                try:
                    if self.capture is not None:
                        self.capture.close()
                except Exception:
                    # Cleanup logging must not replace the generation's exit code.
                    logger.exception("Failed to finish the combined control log")
                finally:
                    for sig, handler in previous.items():
                        signal.signal(sig, handler)
                    if previous_subreaper is not None:
                        with contextlib.suppress(OSError):
                            _set_subreaper(previous_subreaper)
        return code


def run_control_host(args):
    return ControlSupervisor(args).run()
