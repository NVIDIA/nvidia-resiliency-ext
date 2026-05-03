# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI argument registration for fault-tolerance entrypoints."""

from __future__ import annotations

import argparse
from typing import Any, Callable, Optional, Sequence

from nvidia_resiliency_ext.fault_tolerance.attribution_manager import (
    DEFAULT_ATTRIBUTION_STARTUP_TIMEOUT,
)


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def _add_argument(
    parser: argparse.ArgumentParser,
    *names: str,
    action: Optional[Callable[..., Any]] = None,
    **kwargs: Any,
) -> None:
    if action is not None:
        kwargs["action"] = action
    parser.add_argument(*names, **kwargs)


def _add_nnodes_arg(
    parser: argparse.ArgumentParser,
    *,
    action: Optional[Callable[..., Any]] = None,
    default: Optional[str] = "1:1",
    required: bool = False,
    help: str = "Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>.",
) -> None:
    kwargs: dict[str, Any] = {
        "type": str,
        "help": help,
    }
    if required:
        kwargs["required"] = True
    else:
        kwargs["default"] = default
    _add_argument(parser, "--nnodes", action=action, **kwargs)


def _add_rendezvous_core_args(
    parser: argparse.ArgumentParser,
    *,
    action: Optional[Callable[..., Any]] = None,
    endpoint_default: Optional[str] = "",
    endpoint_required: bool = False,
) -> None:
    _add_argument(
        parser,
        "--rdzv-backend",
        "--rdzv_backend",
        action=action,
        type=str,
        default="c10d",
        help="Rendezvous backend. Currently only c10d is supported.",
    )
    endpoint_kwargs: dict[str, Any] = {
        "type": str,
        "help": "Rendezvous backend endpoint; usually in form <host>:<port>.",
    }
    if endpoint_required:
        endpoint_kwargs["required"] = True
    else:
        endpoint_kwargs["default"] = endpoint_default
    _add_argument(
        parser,
        "--rdzv-endpoint",
        "--rdzv_endpoint",
        action=action,
        **endpoint_kwargs,
    )
    _add_argument(
        parser,
        "--rdzv-id",
        "--rdzv_id",
        action=action,
        type=str,
        default="none",
        help="User-defined group id.",
    )
    _add_argument(
        parser,
        "--rdzv-conf",
        "--rdzv_conf",
        action=action,
        type=str,
        default="",
        help="Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).",
    )


def _add_local_addr_arg(
    parser: argparse.ArgumentParser,
    *,
    action: Optional[Callable[..., Any]] = None,
) -> None:
    _add_argument(
        parser,
        "--local-addr",
        "--local_addr",
        action=action,
        default=None,
        type=str,
        help="Address of the local node. If specified, will use the given address for connection. "
        "Else, will look up the local node address instead. Else, it will be default to local "
        "machine's FQDN.",
    )


def _add_ft_segment_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ft-segment",
        "--ft_segment",
        type=int,
        default=None,
        dest="ft_segment",
        help="Controls hot spare node behavior and segment-aware rank assignment. "
        "Default: None (simple hot spare mode for H100 and non-NVSwitch systems, no ClusterUUID required). "
        "When set to N: Enables segment-aware mode for NVSwitch systems (DGX H200, HGX B200). "
        "Specifies minimum nodes per NVLink domain (identified by GPU ClusterUUID via nvidia-smi). "
        "Domains with fewer than N nodes are excluded. From valid domains, complete segments are selected. "
        "min_nodes must be divisible by segment. "
        "Note: segment=None (default) is suitable for H100; segment=1 is similar but requires ClusterUUID.",
    )


def add_rendezvous_args(
    parser: argparse.ArgumentParser,
    *,
    action: Optional[Callable[..., Any]] = None,
    nnodes_default: Optional[str] = "1:1",
    nnodes_required: bool = False,
    nnodes_help: str = "Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>.",
    endpoint_default: Optional[str] = "",
    endpoint_required: bool = False,
    include_local_addr: bool = False,
    include_ft_segment: bool = False,
) -> None:
    _add_nnodes_arg(
        parser,
        action=action,
        default=nnodes_default,
        required=nnodes_required,
        help=nnodes_help,
    )
    _add_rendezvous_core_args(
        parser,
        action=action,
        endpoint_default=endpoint_default,
        endpoint_required=endpoint_required,
    )
    if include_local_addr:
        _add_local_addr_arg(parser, action=action)
    if include_ft_segment:
        _add_ft_segment_arg(parser)


def add_log_funnel_args(
    parser: argparse.ArgumentParser,
    *,
    action: Optional[Callable[..., Any]] = None,
    enable_default: Optional[bool] = None,
    enable_type: Callable[[Any], bool] = str_to_bool,
) -> None:
    if enable_default is None:
        enable_default_help = (
            "False unless auto-enabled by another log-routing option such as --ft-nvrx-logfile"
        )
    else:
        enable_default_help = str(enable_default)

    _add_argument(
        parser,
        "--ft-per-cycle-applog-prefix",
        "--ft_per_cycle_applog_prefix",
        action=action,
        type=str,
        default=None,
        dest="ft_per_cycle_applog_prefix",
        help="Prefix for per-cycle application log files (must be absolute path, e.g. /lustre/logs/job_12345.log). "
        "Creates training worker logs per cycle: /lustre/logs/job_12345_cycle0.log, job_12345_cycle1.log, etc. "
        "All ranks/nodes capture logs with automatic rank prefixes (like 'srun -l'). "
        "Without --ft-enable-log-server: Each node writes directly to Lustre with O_APPEND (safe concurrent writes). "
        "With --ft-enable-log-server (recommended): All nodes stream logs to gRPC server on rank 0, which becomes the single Lustre writer. "
        "gRPC mode eliminates Lustre lock contention and scales to 1000+ nodes. "
        "Note: NVRx launcher logs go to stdout/stderr by default unless --ft-nvrx-logfile is specified.",
    )
    _add_argument(
        parser,
        "--ft-nvrx-logfile",
        "--ft_nvrx_logfile",
        action=action,
        type=str,
        default=None,
        dest="ft_nvrx_logfile",
        help="Optional: Path for NVRx process logs (must be absolute path if specified). "
        "ft_launcher routes these logs through the same infrastructure as worker logs. "
        "nvrx-control also uses this path as a fallback source for gRPC server diagnostic logs. "
        "Example: --ft-nvrx-logfile /lustre/logs/launcher.log",
    )
    _add_argument(
        parser,
        "--ft-enable-log-server",
        "--ft_enable_log_server",
        action=action,
        type=enable_type,
        default=enable_default,
        dest="ft_enable_log_server",
        help="Enable gRPC-based log funneling to a single Lustre file. "
        "When enabled with --ft-per-cycle-applog-prefix, NVRx automatically: "
        "(1) Starts gRPC log server(s) on the TCPStore host, "
        "(2) Configures all nodes as gRPC clients to a first-level aggregator port, "
        "(3) Root server writes to Lustre (single writer). "
        "With --ft-log-aggregator-count=N>1, N leaf processes plus one root use ports P..P+N (see --ft-log-server-port). "
        "Use --ft-log-aggregator-count=1 for a single server on P only. "
        "Benefits: No Lustre lock contention, scales to 1500+ nodes. "
        f"Requires: grpcio. Default: {enable_default_help}.",
    )
    _add_argument(
        parser,
        "--ft-log-server-port",
        "--ft_log_server_port",
        action=action,
        type=int,
        default=50051,
        dest="ft_log_server_port",
        help="Port for gRPC log funnel server (only used if --ft-enable-log-server is enabled). "
        "Default: 50051",
    )
    _add_argument(
        parser,
        "--ft-log-aggregator-count",
        "--ft_log_aggregator_count",
        action=action,
        type=int,
        default=0,
        dest="ft_log_aggregator_count",
        help="Number of first-level gRPC log aggregators. "
        "0 (default): auto - single root server for up to 1536 nodes, "
        "adds leaf servers beyond that (one per 1536 nodes). "
        "1: single root server (legacy). "
        "N>1: N leaf servers on ports [P, P+1, ..., P+N-1] and one root on P+N "
        "(P = --ft-log-server-port). "
        "Open TCP ports P through P+N on the TCP store host. "
        "Each node picks a leaf from SLURM array/procid when set, else from hostname.",
    )
    _add_argument(
        parser,
        "--ft-log-leaf-max-queue-chunks",
        "--ft_log_leaf_max_queue_chunks",
        action=action,
        type=int,
        default=-1,
        dest="ft_log_leaf_max_queue_chunks",
        help="Max queued log chunks per leaf (only when N>1). "
        "Default -1: auto scale with fan-in, max(16384, min(1000000, per_leaf*256)).",
    )
    _add_argument(
        parser,
        "--ft-log-server-log-prefix",
        "--ft_log_server_log_prefix",
        action=action,
        type=str,
        default=None,
        dest="ft_log_server_log_prefix",
        help="Prefix for gRPC log server process logs (only used if --ft-enable-log-server is enabled). "
        "Root and leaf process logs are written to '<prefix>_root.log' and "
        "'<prefix>_leaf_<N>.log'. If not specified, derives from --ft-nvrx-logfile first, "
        "then --ft-per-cycle-applog-prefix by stripping a trailing '.log' and appending '_grpc'.",
    )
    _add_argument(
        parser,
        "--ft-log-server-graceful-shutdown-timeout",
        "--ft_log_server_graceful_shutdown_timeout",
        action=action,
        type=float,
        default=60.0,
        dest="ft_log_server_graceful_shutdown_timeout",
        help="Maximum seconds to wait for clients during gRPC server graceful shutdown (only used if --ft-enable-log-server is enabled). "
        "When store host exits, server will continue accepting logs from other ranks for up to this timeout "
        "or until all clients disconnect, whichever comes first. This ensures other nodes can flush their final logs. "
        "When --ft-log-aggregator-count>1, each leaf uses this timeout; the root log server uses 2x so leaves can finish "
        "waiting for downstream clients and drain to root first. "
        "Default: 60.0",
    )


def _add_ft_config_file_arg(
    parser: argparse.ArgumentParser,
    *,
    aliases: Sequence[str],
    action: Optional[Callable[..., Any]] = None,
) -> None:
    _add_argument(
        parser,
        *aliases,
        action=action,
        default=None,
        type=str,
        dest="ft_cfg_path",
        help="Path to a YAML file that contains Fault Tolerance pkg config (`fault_tolerance` section)."
        " NOTE: config items from the file can be overwritten by `--ft-param-*` args.",
    )


def add_ft_config_file_args(
    parser: argparse.ArgumentParser,
    *,
    aliases: Sequence[str],
    action: Optional[Callable[..., Any]] = None,
) -> None:
    _add_ft_config_file_arg(parser, aliases=aliases, action=action)


def _add_attribution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ft-attribution-endpoint",
        "--ft_attribution_endpoint",
        type=str,
        default=None,
        dest="ft_attribution_endpoint",
        help=(
            "Endpoint for the attribution service. Default: disabled. "
            "Set to localhost to let the TCPStore host process manage nvrx-attrsvc. "
            "Use an explicit endpoint such as http://host:port, grpc://host:port, "
            "or unix:///path/to/socket for an externally managed attribution service."
        ),
    )
    parser.add_argument(
        "--ft-attribution-startup-timeout",
        "--ft_attribution_startup_timeout",
        type=float,
        default=DEFAULT_ATTRIBUTION_STARTUP_TIMEOUT,
        dest="ft_attribution_startup_timeout",
        help=(
            "Seconds to wait for launcher-managed attribution service /healthz readiness. "
            f"Default: {DEFAULT_ATTRIBUTION_STARTUP_TIMEOUT}."
        ),
    )
    parser.add_argument(
        "--ft-attribution-llm-api-key-file",
        "--ft_attribution_llm_api_key_file",
        type=str,
        default=None,
        dest="ft_attribution_llm_api_key_file",
        help=(
            "Path to the LLM API key file for launcher-managed attribution service. "
            "If unset, LLM_API_KEY_FILE from the launcher environment is used."
        ),
    )
    parser.add_argument(
        "--ft-attribution-llm-base-url",
        "--ft_attribution_llm_base_url",
        type=str,
        default=None,
        dest="ft_attribution_llm_base_url",
        help="LLM base URL for launcher-managed attribution service.",
    )
    parser.add_argument(
        "--ft-attribution-llm-model",
        "--ft_attribution_llm_model",
        type=str,
        default=None,
        dest="ft_attribution_llm_model",
        help="LLM model identifier for launcher-managed attribution service.",
    )
    parser.add_argument(
        "--ft-attribution-analysis-backend",
        "--ft_attribution_analysis_backend",
        type=str,
        default=None,
        dest="ft_attribution_analysis_backend",
        choices=("mcp", "lib"),
        help="Analysis backend for launcher-managed attribution service: mcp or lib.",
    )
    parser.add_argument(
        "--ft-attribution-compute-timeout",
        "--ft_attribution_compute_timeout",
        type=float,
        default=None,
        dest="ft_attribution_compute_timeout",
        help="Analysis compute timeout in seconds for launcher-managed attribution service.",
    )
    parser.add_argument(
        "--ft-attribution-log-level",
        "--ft_attribution_log_level",
        type=str.upper,
        default=None,
        dest="ft_attribution_log_level",
        choices=("DEBUG", "INFO", "WARNING"),
        help="Log level for launcher-managed attribution service.",
    )


def add_attribution_args(parser: argparse.ArgumentParser) -> None:
    _add_attribution_args(parser)


def _add_cycle_info_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ft-cycle-info-dir",
        "--ft_cycle_info_dir",
        type=str,
        default=None,
        dest="ft_cycle_info_dir",
        help="Full path to NVRx cycle info directory (e.g. <base>/nvrx/). The rendezvous "
        "host writes cycle_info.<job_id>.<attempt>.<cycle> and symlink "
        "cycle_info.<job_id>.current there. Workload receives current cycle file path via "
        "NVRX_CURRENT_CYCLE_INFO env.",
    )


def add_cycle_info_args(parser: argparse.ArgumentParser) -> None:
    _add_cycle_info_arg(parser)


def add_control_services_args(parser: argparse.ArgumentParser) -> None:
    add_attribution_args(parser)
    add_cycle_info_args(parser)
    parser.add_argument(
        "--ft-cycle-info-job-id",
        "--ft_cycle_info_job_id",
        type=str,
        default=None,
        dest="ft_cycle_info_job_id",
        help="Job id used by nvrx-control when writing cycle_info files. Use the same "
        "effective job id compute launchers use for NVRX_CURRENT_CYCLE_INFO "
        "(SLURM_ARRAY_JOB_ID when present, otherwise SLURM_JOB_ID).",
    )
