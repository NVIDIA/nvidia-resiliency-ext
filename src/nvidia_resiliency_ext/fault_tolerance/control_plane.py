# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External NVRx control-node process.

``nvrx-control`` owns long-lived store-host responsibilities for deployments where
compute ``ft_launcher`` processes should only connect as TCPStore clients.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import signal
import sys
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Optional

from torch.distributed import TCPStore
from torch.distributed.elastic.rendezvous.api import (
    RendezvousClosedError,
    RendezvousGracefulExitError,
)
from torch.distributed.elastic.rendezvous.utils import (
    _parse_rendezvous_config,
    parse_rendezvous_endpoint,
)

from nvidia_resiliency_ext.fault_tolerance.attribution_manager import (
    DEFAULT_ATTRIBUTION_PORT,
    AttributionConfig,
    AttributionManager,
)
from nvidia_resiliency_ext.fault_tolerance.cli_args import (
    add_control_services_args,
    add_ft_config_file_args,
    add_log_funnel_args,
    add_rendezvous_args,
    str_to_bool,
)
from nvidia_resiliency_ext.fault_tolerance.config import FaultToleranceConfig
from nvidia_resiliency_ext.fault_tolerance.cycle_info_writer import (
    CycleInfoReporter,
    cycle_log_file,
)
from nvidia_resiliency_ext.fault_tolerance.ft_rendezvous_barrier import (
    _NodeDescGenerator,
    _RendezvousBarrierState,
)
from nvidia_resiliency_ext.fault_tolerance.launcher import (
    LogFunnelPorts,
    _resolve_grpc_log_server_log_prefix,
    _start_grpc_log_servers,
    _validate_managed_attribution_listen_port_not_in_log_funnel,
    parse_min_max_nnodes,
    stop_grpc_log_servers,
)
from nvidia_resiliency_ext.fault_tolerance.rdzv_utils import (
    rdzv_config_get_as_float,
    rdzv_config_get_as_int,
)
from nvidia_resiliency_ext.shared_utils.health_check import AttributionService
from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig, setup_logger

logger = logging.getLogger(LogConfig.name)


@dataclass
class ControlServices:
    grpc_processes: list[Any] = field(default_factory=list)
    attribution_manager: Optional[AttributionManager] = None
    attribution_service: Optional[AttributionService] = None
    cycle_info_reporter: Optional[CycleInfoReporter] = None


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nvrx-control",
        description="Run the external NVRx control-node process.",
    )
    add_rendezvous_args(
        parser,
        nnodes_required=True,
        nnodes_help="Training node range MIN[:MAX].",
        endpoint_required=True,
        include_local_addr=True,
        include_ft_segment=True,
    )
    add_ft_config_file_args(parser, aliases=("--ft-cfg-path", "--ft-cfg_path"))
    add_log_funnel_args(parser, enable_default=True, enable_type=str_to_bool)
    add_control_services_args(parser)
    return parser


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    return get_args_parser().parse_args(args)


def _validate_args(args: argparse.Namespace, ft_cfg: FaultToleranceConfig) -> None:
    if args.rdzv_backend.lower() != "c10d":
        raise ValueError(f"nvrx-control supports only --rdzv-backend=c10d, got {args.rdzv_backend}")
    host, port = parse_rendezvous_endpoint(args.rdzv_endpoint, default_port=-1)
    if not host or port == -1:
        raise ValueError("--rdzv-endpoint must include host and port, e.g. control-host:29500")

    if args.ft_enable_log_server and _resolve_grpc_log_server_log_prefix(args) is None:
        raise ValueError(
            "nvrx-control requires one of --ft-log-server-log-prefix, "
            "--ft-nvrx-logfile, or --ft-per-cycle-applog-prefix when log funneling is enabled."
        )

    if (args.ft_attribution_endpoint or ft_cfg.attribution_endpoint) and (
        not args.ft_per_cycle_applog_prefix
    ):
        raise ValueError(
            "nvrx-control requires --ft-per-cycle-applog-prefix when attribution is enabled."
        )

    if args.ft_per_cycle_applog_prefix and not os.path.isabs(args.ft_per_cycle_applog_prefix):
        raise ValueError(
            "--ft-per-cycle-applog-prefix must be an absolute path, "
            f"got {args.ft_per_cycle_applog_prefix!r}"
        )
    if ft_cfg.cycle_info_dir and not args.ft_cycle_info_job_id:
        raise ValueError("nvrx-control requires --ft-cycle-info-job-id with --ft-cycle-info-dir.")
    if args.ft_nvrx_logfile and not os.path.isabs(args.ft_nvrx_logfile):
        raise ValueError(
            "--ft-nvrx-logfile must be an absolute path, " f"got {args.ft_nvrx_logfile!r}"
        )


def _create_tcp_store(args: argparse.Namespace, read_timeout: float) -> TCPStore:
    host, port = parse_rendezvous_endpoint(args.rdzv_endpoint, default_port=-1)
    logger.info("Starting control TCPStore on %s:%s", host, port)
    return TCPStore(
        host_name=host,
        port=port,
        is_master=True,
        timeout=timedelta(seconds=read_timeout),
        wait_for_workers=False,
        multi_tenant=True,
    )


def _start_control_services(
    args: argparse.Namespace,
    ft_cfg: FaultToleranceConfig,
) -> ControlServices:
    services = ControlServices()
    try:
        if ft_cfg.cycle_info_dir:
            services.cycle_info_reporter = CycleInfoReporter(
                ft_cfg.cycle_info_dir,
                cycle_log_prefix=args.ft_per_cycle_applog_prefix,
                cycle_info_job_id=args.ft_cycle_info_job_id,
                attempt_index=0,
            )

        applog_prefix = args.ft_per_cycle_applog_prefix
        log_funnel_ports = None
        grpc_log_prefix = None
        if args.ft_enable_log_server:
            grpc_log_prefix = _resolve_grpc_log_server_log_prefix(args)
            assert grpc_log_prefix is not None
            log_funnel_ports = LogFunnelPorts.from_launcher_args(args)

        if args.ft_attribution_endpoint or ft_cfg.attribution_endpoint:
            assert applog_prefix is not None
            attribution_cfg = AttributionConfig.from_args(args, applog_prefix, ft_cfg)
            if log_funnel_ports is not None and attribution_cfg.is_managed:
                _validate_managed_attribution_listen_port_not_in_log_funnel(
                    DEFAULT_ATTRIBUTION_PORT, log_funnel_ports
                )
            services.attribution_manager = AttributionManager(attribution_cfg, is_store_host=True)
            attribution_endpoint = services.attribution_manager.start_if_needed()
            if attribution_endpoint is not None:
                services.attribution_service = AttributionService(
                    endpoint=attribution_endpoint.endpoint
                )

        if args.ft_enable_log_server:
            assert log_funnel_ports is not None
            assert grpc_log_prefix is not None
            services.grpc_processes = _start_grpc_log_servers(
                args, grpc_log_prefix, log_funnel_ports
            )
            if not services.grpc_processes:
                raise RuntimeError("failed to start gRPC log funnel process(es)")

        return services
    except Exception:
        with contextlib.suppress(Exception):
            if services.grpc_processes:
                stop_grpc_log_servers(
                    services.grpc_processes,
                    float(args.ft_log_server_graceful_shutdown_timeout),
                )
        with contextlib.suppress(Exception):
            if services.attribution_manager is not None:
                services.attribution_manager.stop()
        with contextlib.suppress(Exception):
            if services.cycle_info_reporter is not None:
                services.cycle_info_reporter.shutdown()
        raise


def _run_control_rendezvous_loop(
    args: argparse.Namespace,
    ft_cfg: FaultToleranceConfig,
    store: TCPStore,
    services: ControlServices,
    stop_event: threading.Event,
) -> None:
    min_nodes, max_nodes = parse_min_max_nnodes(args.nnodes)
    rdzv_configs = _parse_rendezvous_config(args.rdzv_conf)
    join_timeout = rdzv_config_get_as_float(
        rdzv_configs, "join_timeout", rdzv_config_get_as_float(rdzv_configs, "timeout", 600.0)
    )
    segment_check_interval = rdzv_config_get_as_float(rdzv_configs, "segment_check_interval", 5.0)
    segment = ft_cfg.segment
    if segment is None:
        segment = rdzv_config_get_as_int(rdzv_configs, "segment", 0) or None

    state = _RendezvousBarrierState(
        store=store,
        run_id=args.rdzv_id,
        is_store_host=True,
        join_timeout_seconds=join_timeout,
        segment=segment,
    )
    node = _NodeDescGenerator().generate(args.local_addr)

    logger.info(
        "nvrx-control rendezvous loop started: run_id=%s nnodes=%s endpoint=%s",
        args.rdzv_id,
        args.nnodes,
        args.rdzv_endpoint,
    )
    if services.cycle_info_reporter is not None:
        state._cycle_info_reporter = services.cycle_info_reporter

    while not stop_event.is_set():
        if services.attribution_service is not None:
            services.attribution_service()
        try:
            closed_round = state.close_current_round_as_host(
                node,
                min_nodes,
                max_nodes,
                segment_check_interval=segment_check_interval,
                stop_event=stop_event,
            )
        except (RendezvousGracefulExitError, RendezvousClosedError) as exc:
            logger.info("nvrx-control rendezvous loop exiting: %s", exc)
            return

        logger.info("nvrx-control closed rendezvous round %s", closed_round)
        if services.attribution_service is not None and args.ft_per_cycle_applog_prefix:
            services.attribution_service._submit_log(
                cycle_log_file(args.ft_per_cycle_applog_prefix, closed_round)
            )


def run(args: argparse.Namespace) -> None:
    setup_logger(node_local_tmp_prefix="nvrxcontrol")
    ft_cfg = FaultToleranceConfig.from_args(args)
    _validate_args(args, ft_cfg)
    rdzv_configs = _parse_rendezvous_config(args.rdzv_conf)
    read_timeout = rdzv_config_get_as_float(rdzv_configs, "read_timeout", 60.0)

    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: Any) -> None:
        logger.info("nvrx-control received %s; shutting down", signal.Signals(signum).name)
        stop_event.set()

    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in previous_handlers:
        signal.signal(sig, _handle_signal)

    services = ControlServices()
    try:
        store = _create_tcp_store(args, read_timeout)
        services = _start_control_services(args, ft_cfg)
        _run_control_rendezvous_loop(args, ft_cfg, store, services, stop_event)
    finally:
        with contextlib.suppress(Exception):
            if services.grpc_processes:
                stop_grpc_log_servers(
                    services.grpc_processes,
                    float(args.ft_log_server_graceful_shutdown_timeout),
                )
        with contextlib.suppress(Exception):
            if services.attribution_manager is not None:
                services.attribution_manager.stop()
        with contextlib.suppress(Exception):
            if services.cycle_info_reporter is not None:
                services.cycle_info_reporter.shutdown()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main(args: Optional[list[str]] = None) -> None:
    parsed_args = parse_args(args)
    try:
        run(parsed_args)
    except Exception as exc:
        logger.error("nvrx-control exited with exception: %s", exc, exc_info=True)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
