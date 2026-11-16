# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared startup, port layout, and shutdown for the existing log funnel."""

import contextlib
import logging
import os
import subprocess  # nosec B404
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, List, Set

from ..shared_utils.log_manager import LogConfig
from .child_process import wait_or_kill_child
from .cli_args import parse_min_max_nnodes
from .control_logging import agent_log_path
from .utils import get_log_aggregator_shard_index

logger = logging.getLogger(LogConfig.name)


def _resolve_grpc_log_allowed_roots(args: Any) -> List[str]:
    """
    Directories the gRPC log servers are allowed to write into.

    The servers bind an unauthenticated port and write to a path chosen by each client, so
    they are given exactly the directories that gRPC clients actually target: the per-cycle
    application log prefix and the derived node-agent log. Per-cycle files (``<prefix>_cycleN.log``)
    stay in the prefix's directory, so the dirname covers every cycle.

    The servers' own logs go through stdout/stderr pipes into the supervisor's
    local control log, never through a ``LogChunk``.

    Each directory is created before being resolved. ``realpath`` of a path that does not
    exist yet returns the path unchanged, so a directory that only later appears as (or
    under) a symlink would resolve differently at write time and the job's own logs would
    be rejected. Creating it first pins the resolution; the log server would create it on
    the first chunk regardless.

    Returns:
        Deduplicated resolved directories; empty only if no log paths are configured.
    """
    candidates = [
        getattr(args, 'ft_per_cycle_applog_prefix', None),
        agent_log_path(getattr(args, 'ft_nvrx_logfile', None)),
    ]
    roots: List[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        expanded = os.path.abspath(os.path.expanduser(candidate))
        dirname = os.path.dirname(expanded) or os.sep
        # Best effort: if creation fails the server will report the real error on first write.
        with contextlib.suppress(OSError):
            os.makedirs(dirname, exist_ok=True)
        root = os.path.realpath(dirname)
        if root not in roots:
            roots.append(root)
    return roots


_LOG_FUNNEL_NODES_PER_LEAF = 1536  # max nodes per single log server (≈6K GPU at 4 GPUs/node)


@dataclass(frozen=True)
class LogFunnelPorts:
    """Port assignments for the gRPC log funnel (single root vs N leaves + root).

    Single-level (``first_level_count == 1``): one root listens on ``base_port`` and
    clients connect there.

    Two-level (``first_level_count > 1``): ``first_level_count`` leaves listen on
    ``base_port .. base_port + N - 1``; the root (sole Lustre writer) listens on
    ``base_port + N``. Each launcher picks a leaf using ``get_log_aggregator_shard_index(N)``
    (SLURM array / ``SLURM_PROCID`` when present, else stable hostname hashing).
    """

    base_port: int
    first_level_count: int

    def __post_init__(self) -> None:
        if not 1 <= self.base_port <= 65535:
            raise ValueError(f"base_port must be a valid TCP port, got {self.base_port}")
        if self.first_level_count < 1:
            raise ValueError("first_level_count must be >= 1")
        if self.first_level_count > 1:
            root_port = self.base_port + self.first_level_count
            if root_port > 65535:
                raise ValueError(
                    f"root_listen_port (base_port + first_level_count) = {root_port} exceeds "
                    f"65535; reduce --ft-log-server-port (base {self.base_port}) or "
                    f"--ft-log-aggregator-count ({self.first_level_count})"
                )

    @classmethod
    def from_launcher_args(cls, args: Any) -> "LogFunnelPorts":
        base = int(getattr(args, "ft_log_server_port", 50051))
        n = int(getattr(args, "ft_log_aggregator_count", 0))
        if n == 0:
            _, max_nodes = parse_min_max_nnodes(getattr(args, "nnodes", "1"))
            n = max(1, (max_nodes + _LOG_FUNNEL_NODES_PER_LEAF - 1) // _LOG_FUNNEL_NODES_PER_LEAF)
            if n > 16:
                logger.warning(
                    f"Auto log aggregator count is {n} (max_nodes={max_nodes}); "
                    f"head node will run {n} leaf processes (~{n * 50} MB extra RSS). "
                    f"Override with --ft-log-aggregator-count if needed."
                )
        if n < 1:
            raise ValueError(f"ft_log_aggregator_count must be >= 1, got {n}")
        return cls(base_port=base, first_level_count=n)

    @property
    def uses_two_level(self) -> bool:
        return self.first_level_count > 1

    @property
    def root_listen_port(self) -> int:
        """Port bound by the process that writes to Lustre."""
        if self.uses_two_level:
            return self.base_port + self.first_level_count
        return self.base_port

    def leaf_listen_port(self, leaf_index: int) -> int:
        """Port for leaf ``leaf_index`` in ``[0, first_level_count)`` (two-level only)."""
        if not self.uses_two_level:
            raise ValueError("leaf_listen_port is only defined when first_level_count > 1")
        if not 0 <= leaf_index < self.first_level_count:
            raise ValueError(
                f"leaf_index must be in [0, {self.first_level_count}), got {leaf_index}"
            )
        return self.base_port + leaf_index

    def client_connect_port(self) -> int:
        """Aggregator port this launcher should use (leaf shard or single root)."""
        if not self.uses_two_level:
            return self.base_port
        shard = get_log_aggregator_shard_index(self.first_level_count)
        return self.leaf_listen_port(shard)

    def listen_ports(self) -> Set[int]:
        """All TCP ports bound by the log funnel on the TCPStore host."""
        if not self.uses_two_level:
            return {self.base_port}
        return set(range(self.base_port, self.base_port + self.first_level_count + 1))


def _validate_managed_attribution_listen_port_not_in_log_funnel(
    managed_port: int, funnel_ports: LogFunnelPorts
) -> None:
    log_ports = funnel_ports.listen_ports()
    if managed_port in log_ports:
        sorted_ports = sorted(log_ports)
        raise ValueError(
            "launcher-managed attribution service port overlaps the gRPC log funnel port range: "
            f"managed_attrsvc_listen_port={managed_port}, log_funnel_ports={sorted_ports}. "
            "Use --ft-attribution-endpoint with an external service or use "
            "--ft-log-server-port to choose a non-overlapping log funnel port."
        )


# Listen on all interfaces so peer training nodes can reach the log funnel on the TCP store host.
_GRPC_LOG_FUNNEL_BIND_HOST = "0.0.0.0"  # nosec B104


def _start_grpc_log_servers(
    args: Any, funnel_ports: LogFunnelPorts, *, capture
) -> List[subprocess.Popen]:
    """
    On TCP store host: start root log server (Lustre writer), and optionally N leaf
    aggregators that forward to the root. In two-level mode the root is given
    ``2 * graceful_shutdown_timeout`` so it outlasts leaf drain after SIGTERM.

    Returns:
        Non-empty list of Popen [root] or [root, leaf0, ...] on success, [] on failure.
    """
    graceful_shutdown_timeout = getattr(args, 'ft_log_server_graceful_shutdown_timeout', 60.0)
    _, max_nodes = parse_min_max_nnodes(args.nnodes)

    procs: List[subprocess.Popen] = []
    try:
        # Clients pick the destination path for every log chunk, and these servers listen
        # on all interfaces without auth, so confine writes to the configured log
        # directories. Raising here is caught below and reported as a startup failure.
        allowed_roots = _resolve_grpc_log_allowed_roots(args)
        if not allowed_roots:
            raise ValueError(
                "cannot determine allowed log write roots for the gRPC log server(s); "
                "expected --ft-per-cycle-applog-prefix (or --ft-nvrx-logfile) to be set"
            )
        allowed_root_args: List[str] = []
        for root in allowed_roots:
            allowed_root_args += ['--allowed-root', root]
        logger.info(f"gRPC log server(s) confined to allowed roots: {allowed_roots}")

        if not funnel_ports.uses_two_level:
            max_workers = min(4096, max(100, max_nodes + 10))
            p = subprocess.Popen(  # nosec B603
                [
                    sys.executable,
                    '-m',
                    'nvidia_resiliency_ext.shared_utils.grpc_log_server',
                    '--host',
                    _GRPC_LOG_FUNNEL_BIND_HOST,
                    '--port',
                    str(funnel_ports.base_port),
                    '--max-workers',
                    str(max_workers),
                    '--graceful-shutdown-timeout',
                    str(graceful_shutdown_timeout),
                    *allowed_root_args,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
            )
            procs.append(p)
            capture.attach(p.stdout)
            logger.info(
                f"gRPC log server started: PID={p.pid}, port={funnel_ports.base_port}, "
                f"max_workers={max_workers} (for {max_nodes} nodes)"
            )
            return procs

        root_port = funnel_ports.root_listen_port
        n = funnel_ports.first_level_count
        root_workers = max(n + 10, 32)
        per_leaf = max(100, (max_nodes + n - 1) // n + 100)
        raw_leaf_chunks = int(getattr(args, 'ft_log_leaf_max_queue_chunks', -1))
        if raw_leaf_chunks < 0:
            max_queue = min(1_000_000, max(16_384, per_leaf * 256))
        else:
            max_queue = max(64, raw_leaf_chunks)

        # Root must outlive leaves: leaves may wait the full leaf grace window for downstream
        # clients before draining their queues to root; root's own grace must still be open then.
        root_graceful_shutdown_timeout = graceful_shutdown_timeout * 2.0

        root_p = subprocess.Popen(  # nosec B603
            [
                sys.executable,
                '-m',
                'nvidia_resiliency_ext.shared_utils.grpc_log_server',
                '--host',
                _GRPC_LOG_FUNNEL_BIND_HOST,
                '--port',
                str(root_port),
                '--max-workers',
                str(root_workers),
                '--graceful-shutdown-timeout',
                str(root_graceful_shutdown_timeout),
                *allowed_root_args,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
        )
        procs.append(root_p)
        capture.attach(root_p.stdout)
        logger.info(
            f"gRPC root log server: PID={root_p.pid}, port={root_port}, max_workers={root_workers}, "
            f"graceful_shutdown_timeout_s={root_graceful_shutdown_timeout} (2× leaf {graceful_shutdown_timeout}s)"
        )

        upstream = f'localhost:{root_port}'
        for i in range(n):
            leaf_port = funnel_ports.leaf_listen_port(i)
            leaf_cmd = [
                sys.executable,
                '-m',
                'nvidia_resiliency_ext.shared_utils.grpc_log_leaf_server',
                '--host',
                _GRPC_LOG_FUNNEL_BIND_HOST,
                '--port',
                str(leaf_port),
                '--upstream',
                upstream,
                '--max-workers',
                str(per_leaf),
                '--max-queue-chunks',
                str(max_queue),
                '--graceful-shutdown-timeout',
                str(graceful_shutdown_timeout),
                *allowed_root_args,
            ]
            lp = subprocess.Popen(  # nosec B603
                leaf_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
            )
            procs.append(lp)
            capture.attach(lp.stdout)
            logger.info(
                f"gRPC leaf log server {i}: PID={lp.pid}, port={leaf_port}, "
                f"max_workers={per_leaf}, max_queue_chunks={max_queue}, upstream={upstream}"
            )

        return procs
    except Exception as e:
        logger.error(f"Failed to start gRPC log server(s): {e}", exc_info=True)
        for p in procs:
            with contextlib.suppress(Exception):
                wait_or_kill_child(p, 0.0)
        return []


def _grpc_child_wait_timeout_after_terminate(graceful_shutdown_timeout: float) -> float:
    """Seconds to wait after SIGTERM before SIGKILL for a gRPC funnel child.

    Matches each server's graceful client-wait, ``server.stop(grace=5)``, and
    ``wait_for_termination`` budget (see ``grpc_log_server`` / ``grpc_log_leaf_server``).
    """
    return float(graceful_shutdown_timeout) + 5.0 + 65.0


def _wait_grpc_subprocess_after_terminate(p: subprocess.Popen, wait_timeout: float) -> None:
    """Wait for one gRPC funnel child after SIGTERM; force-kill if still alive past timeout."""
    wait_or_kill_child(p, wait_timeout)


def stop_grpc_log_servers(
    processes: List[subprocess.Popen],
    graceful_shutdown_timeout: float,
) -> None:
    if not processes:
        return

    uses_two_level = len(processes) > 1
    single_wait = _grpc_child_wait_timeout_after_terminate(graceful_shutdown_timeout)
    if uses_two_level:
        root_grace = graceful_shutdown_timeout * 2.0
        leaf_wait = _grpc_child_wait_timeout_after_terminate(graceful_shutdown_timeout)
        root_wait = _grpc_child_wait_timeout_after_terminate(root_grace)
        wait_msg = (
            f"leaf child wait up to {leaf_wait:.0f}s, root child up to {root_wait:.0f}s "
            f"(ft_log_server_graceful_shutdown_timeout={graceful_shutdown_timeout}s, root uses 2×)"
        )
    else:
        leaf_wait = root_wait = single_wait
        wait_msg = (
            f"per-process shutdown wait up to {single_wait:.0f}s "
            f"(ft_log_server_graceful_shutdown_timeout={graceful_shutdown_timeout}s)"
        )

    logger.info(
        "Sending SIGTERM to gRPC log funnel process(es) (root last in wait order). "
        f"{wait_msg}. "
        "Check gRPC root and leaf server logs for Shutdown snapshot lines: "
        "in_memory_pending_bytes, chunk_queue_depth, still_connected_streams, etc."
    )
    # Terminate leaves first so they drain to root before root exits.
    for p in reversed(processes):
        with contextlib.suppress(Exception):
            p.terminate()

    # All processes got SIGTERM together; wait in parallel so wall time is roughly one timeout.
    procs_to_wait = list(reversed(processes))
    root_proc = processes[0]
    with ThreadPoolExecutor(max_workers=max(1, len(procs_to_wait))) as pool:
        pending_futures = [
            pool.submit(
                _wait_grpc_subprocess_after_terminate,
                p,
                root_wait if p is root_proc else leaf_wait,
            )
            for p in procs_to_wait
        ]
        for fut in as_completed(pending_futures):
            with contextlib.suppress(Exception):
                fut.result()

    for p in processes:
        rc = getattr(p, "returncode", None)
        logger.info(f"gRPC log subprocess PID={p.pid} finished with returncode={rc}")
    processes.clear()
