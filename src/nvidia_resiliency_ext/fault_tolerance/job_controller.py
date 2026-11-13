# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVRx job-wide rendezvous and control process.

``job-controller`` runs under the control supervisor and owns the TCPStore,
rendezvous state, attribution client, and cycle-info reporter. The supervisor
owns all sibling service processes.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Optional

from torch.distributed import TCPStore
from torch.distributed.elastic.rendezvous.api import (
    RendezvousClosedError,
    RendezvousGracefulExitError,
    RendezvousParameters,
)
from torch.distributed.elastic.rendezvous.utils import (
    _parse_rendezvous_config,
    parse_rendezvous_endpoint,
)

from nvidia_resiliency_ext.fault_tolerance.attribution_manager import AttributionConfig
from nvidia_resiliency_ext.fault_tolerance.child_process import read_child_args
from nvidia_resiliency_ext.fault_tolerance.cli_args import parse_min_max_nnodes
from nvidia_resiliency_ext.fault_tolerance.config import FaultToleranceConfig
from nvidia_resiliency_ext.fault_tolerance.cycle_info_writer import CycleInfoReporter
from nvidia_resiliency_ext.fault_tolerance.ft_rendezvous_barrier import (
    _get_replacement_group_size,
    _NodeDescGenerator,
    _RendezvousBarrierState,
)
from nvidia_resiliency_ext.fault_tolerance.rdzv_utils import (
    rdzv_config_get_as_bool,
    rdzv_config_get_as_float,
)
from nvidia_resiliency_ext.fault_tolerance.utils import (
    DEFAULT_NO_RESTART_EXIT_CODE,
    RDZV_NO_RESTART_REASONS,
    RDZV_SHUTDOWN_REASON_GRACEFUL,
)
from nvidia_resiliency_ext.shared_utils import telemetry
from nvidia_resiliency_ext.shared_utils.health_check import AttributionService
from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig, setup_logger

logger = logging.getLogger(LogConfig.name)


@dataclass
class ControllerResources:
    # Despite its legacy name, AttributionService is a client with a poller thread.
    # ControlSupervisor owns the managed attribution server process.
    attribution_service: Optional[AttributionService] = None
    cycle_info_reporter: Optional[CycleInfoReporter] = None


def _validate_args(args: argparse.Namespace, ft_cfg: FaultToleranceConfig) -> None:
    if args.rdzv_backend.lower() != "c10d":
        raise ValueError(
            f"job-controller supports only --rdzv-backend=c10d, got {args.rdzv_backend}"
        )
    host, port = parse_rendezvous_endpoint(args.rdzv_endpoint, default_port=-1)
    if not host or port == -1:
        raise ValueError("--rdzv-endpoint must include host and port, e.g. control-host:29500")

    if ft_cfg.attribution_endpoint and not args.ft_per_cycle_applog_prefix:
        raise ValueError(
            "job-controller requires --ft-per-cycle-applog-prefix when attribution is enabled."
        )

    if args.ft_per_cycle_applog_prefix and not os.path.isabs(args.ft_per_cycle_applog_prefix):
        raise ValueError(
            "--ft-per-cycle-applog-prefix must be an absolute path, "
            f"got {args.ft_per_cycle_applog_prefix!r}"
        )
    if args.ft_nvrx_logfile and not os.path.isabs(args.ft_nvrx_logfile):
        raise ValueError(
            "--ft-nvrx-logfile must be an absolute path, " f"got {args.ft_nvrx_logfile!r}"
        )


def _create_tcp_store(args: argparse.Namespace, read_timeout: float) -> TCPStore:
    if read_timeout <= 0:
        raise ValueError("The read timeout must be positive.")
    configs = _parse_rendezvous_config(args.rdzv_conf)
    use_libuv = rdzv_config_get_as_bool(configs, "use_libuv", True)
    host, port = parse_rendezvous_endpoint(args.rdzv_endpoint, default_port=-1)
    logger.info("Starting control TCPStore on %s:%s", host, port)
    return TCPStore(
        host_name=host,
        port=port,
        is_master=True,
        timeout=timedelta(seconds=read_timeout),
        wait_for_workers=False,
        multi_tenant=True,
        use_libuv=use_libuv,
    )


def _stop_controller_resources(resources: ControllerResources) -> None:
    with contextlib.suppress(Exception):
        if resources.cycle_info_reporter is not None:
            resources.cycle_info_reporter.shutdown()
    with contextlib.suppress(Exception):
        if resources.attribution_service is not None:
            resources.attribution_service.stop_poller()


def _start_controller_resources(
    args: argparse.Namespace,
    ft_cfg: FaultToleranceConfig,
    *,
    node_id: str,
) -> ControllerResources:
    resources = ControllerResources()
    try:
        if ft_cfg.cycle_info_dir:
            identity = {}
            if getattr(args, "ft_control_only", False):
                # Preserve the external controller's explicit training job ID
                # and attempt-0 namespace, independent of the CPU allocation.
                identity = {"cycle_info_job_id": args.ft_cycle_info_job_id, "attempt_index": 0}
            resources.cycle_info_reporter = CycleInfoReporter(
                ft_cfg.cycle_info_dir,
                cycle_log_prefix=args.ft_per_cycle_applog_prefix,
                **identity,
            )
        if ft_cfg.attribution_endpoint:
            cfg = AttributionConfig.from_args(args, args.ft_per_cycle_applog_prefix, ft_cfg)
            endpoint = cfg.client_endpoint
            resources.attribution_service = AttributionService(
                endpoint=endpoint.endpoint, enforce_stop=endpoint.enforce_stop
            )
            resources.attribution_service.start_poller(node_id=node_id)
        return resources
    except Exception:
        _stop_controller_resources(resources)
        raise


def _run_control_rendezvous_loop(
    args: argparse.Namespace,
    ft_cfg: FaultToleranceConfig,
    store: TCPStore,
    services: ControllerResources,
    stop_event: threading.Event,
) -> int:
    min_nodes, max_nodes = parse_min_max_nnodes(args.nnodes)
    rdzv_configs = _parse_rendezvous_config(args.rdzv_conf)
    join_timeout = rdzv_config_get_as_float(
        rdzv_configs, "join_timeout", rdzv_config_get_as_float(rdzv_configs, "timeout", 600.0)
    )
    last_call_timeout = rdzv_config_get_as_float(rdzv_configs, "last_call_timeout", 1.0)
    segment_check_interval = rdzv_config_get_as_float(rdzv_configs, "segment_check_interval", 5.0)
    params = RendezvousParameters(
        backend="c10d",
        endpoint=args.rdzv_endpoint,
        run_id=args.rdzv_id,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        **rdzv_configs,
    )
    replacement_group_size = _get_replacement_group_size(params)
    if replacement_group_size is not None and replacement_group_size < 1:
        raise ValueError("replacement_group_size must be a positive integer")

    state = _RendezvousBarrierState(
        store=store,
        run_id=args.rdzv_id,
        is_store_host=True,
        join_timeout_seconds=join_timeout,
        last_call_timeout_seconds=last_call_timeout,
        # Match the agents' effective --ft-segment setting, including None.
        segment=ft_cfg.segment,
        replacement_group_size=replacement_group_size,
        cycle_log_prefix=args.ft_per_cycle_applog_prefix,
    )
    node = _NodeDescGenerator().generate(args.local_addr)

    logger.info(
        "job-controller rendezvous loop started: run_id=%s nnodes=%s endpoint=%s",
        args.rdzv_id,
        args.nnodes,
        args.rdzv_endpoint,
    )
    if services.cycle_info_reporter is not None:
        state._cycle_info_reporter = services.cycle_info_reporter
    state._attribution_service = services.attribution_service

    while not stop_event.is_set():
        try:
            closed_round = state.close_current_round_as_host(
                node,
                min_nodes,
                max_nodes,
                segment_check_interval=segment_check_interval,
                stop_event=stop_event,
            )
        except (RendezvousGracefulExitError, RendezvousClosedError) as exc:
            logger.info("job-controller rendezvous loop exiting: %s", exc)
            break

        logger.info("job-controller closed rendezvous round %s", closed_round)

    state._request_terminal_attribution_for_submitted_cycle()

    reason = state.get_shutdown_reason()
    if reason in RDZV_NO_RESTART_REASONS:
        # Stop InJob recovery in this generation. This exit code does not cancel
        # queued singleton successors; the caller handles current-array cleanup.
        code = getattr(args, "ft_no_restart_exit_code", None)
        return DEFAULT_NO_RESTART_EXIT_CODE if code is None else int(code)
    if reason == RDZV_SHUTDOWN_REASON_GRACEFUL:
        # Preserve the existing rendezvous completion semantics.
        return 0
    logger.error("job-controller stopped without successful completion: %s", reason)
    return 1


# Peers read the recorded shutdown reason from the TCPStore to decide their exit code,
# and this process owns that store: it dies when job-controller exits. Leaving too fast
# downgrades a no-restart stop to a generic failure exit on nodes that had not read it
# yet. The embedded launcher holds the same window open in launch_agent().
_STORE_READ_GRACE_SECONDS = 3.0


def _wait_for_store_read_grace(since: float) -> None:
    """Keep the TCPStore alive until _STORE_READ_GRACE_SECONDS after the loop ended.

    Teardown of the controller resources already consumes part of that window, so only
    the remainder is waited out and a slow teardown costs nothing extra.
    """
    remaining = _STORE_READ_GRACE_SECONDS - (time.monotonic() - since)
    if remaining <= 0:
        return
    logger.info(
        "job-controller waiting %.1fs before exit so other nodes can read final " "TCPStore state",
        remaining,
    )
    time.sleep(remaining)


def run(args: argparse.Namespace) -> int:
    setup_logger()
    ft_cfg = FaultToleranceConfig.from_args(args)
    _validate_args(args, ft_cfg)
    rdzv_configs = _parse_rendezvous_config(args.rdzv_conf)
    read_timeout = rdzv_config_get_as_float(rdzv_configs, "read_timeout", 60.0)

    stop_event = threading.Event()
    received_signal = None

    def _handle_signal(signum: int, _frame: Any) -> None:
        nonlocal received_signal
        if received_signal is None:
            received_signal = signum
        logger.info("job-controller received %s; shutting down", signal.Signals(signum).name)
        stop_event.set()

    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in previous_handlers:
        signal.signal(sig, _handle_signal)

    services = ControllerResources()
    tel_handle = None
    try:
        node_id = socket.getfqdn(socket.gethostname())
        # The controller is exec'd independently of the node-agent. Its provider
        # consumes inherited OTEL_RESOURCE_ATTRIBUTES and owns its process identity.
        tel_handle = telemetry.setup_telemetry(
            "nvrx.job_controller",
            f"job-controller-{node_id}",
            {"nv.nvrx.ftl.node": node_id},
        )
        store = _create_tcp_store(args, read_timeout)
        services = _start_controller_resources(args, ft_cfg, node_id=node_id)
        result = _run_control_rendezvous_loop(args, ft_cfg, store, services, stop_event)
    finally:
        rendezvous_loop_ended = time.monotonic()
        _stop_controller_resources(services)
        # Stop the poller before flushing its spans. Export failures must not
        # replace the controller's exit status or prevent handler restoration.
        if tel_handle is not None:
            with contextlib.suppress(Exception):
                telemetry.shutdown(tel_handle)
        with contextlib.suppress(Exception):
            _wait_for_store_read_grace(rendezvous_loop_ended)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return 128 + received_signal if received_signal is not None else result


def main() -> None:
    parsed_args = read_child_args()
    if parsed_args is None:
        raise RuntimeError("job-controller must be started by ft_launcher")
    try:
        result = run(parsed_args)
    except Exception as exc:
        logger.error("job-controller exited with exception: %s", exc, exc_info=True)
        sys.exit(1)
    sys.exit(result)


if __name__ == "__main__":
    main()
