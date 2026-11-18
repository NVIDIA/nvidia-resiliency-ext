# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for the cycle records: the messages of nvrx_interface.proto.

The proto messages are the records themselves. The store host builds one `NVRxCycleInfo` per
cycle, when the rendezvous round that starts it closes, and an updated copy when the
cycle ends. Each launcher builds one `NodeCycleEndReport` per cycle, when the cycle ends
for it; end reports are local facts, so which node detected the end first is concluded
from the end reports of all launchers (`JobCycleEndReport`, see the aggregator).

The messages are kept in their wire and file format: ISO 8601 UTC times and
SLURM-compressed node and rank lists. The helpers here build them, check received ones,
and read back what the CycleAnalyzer computes with.

Messages are mutable: never modify one after publishing it; copy it instead.
"""

import json
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING, List, Mapping, Optional, Sequence

from nvidia_resiliency_ext.shared_utils.proto import nvrx_interface_pb2 as pb

from ..cycle_info_writer import utc_iso
from ..utils import (
    hostnames_to_slurm_nodelist,
    parse_bool_env,
    range_str_to_ranks,
    ranks_to_range_str,
    slurm_nodelist_to_hostnames,
    slurm_sort_addrs,
)

if TYPE_CHECKING:
    # Annotations only: the CycleAnalyzer process imports this module, and data.py
    # imports torch.
    from torch.distributed.elastic.agent.server.api import Worker
    from torch.distributed.elastic.multiprocessing.errors import ProcessFailure

    from ..data import RankTimeout

# Bound each end report, so a workload with many or long section names, or many failed
# workers, cannot grow it.
MAX_TIMEOUT_SECTIONS = 4
MAX_SECTION_NAME_LEN = 64
# Workers listed in NodeCycleEndReport.exit_details; the rest are counted, not listed.
MAX_EXIT_DETAILS_WORKERS = 16


def now_ms() -> int:
    """The current time in milliseconds since the Unix epoch (UTC), as the messages keep it."""
    return time.time_ns() // 1_000_000


def iso_to_seconds(value: str) -> Optional[float]:
    """An ISO 8601 UTC time of a message in seconds since the epoch; None for "" (not set).

    Raises:
        ValueError: `value` is not an ISO 8601 time.
    """
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _ranks_in_host_order(nodes: Sequence[str], ranks: Sequence[int]) -> List[int]:
    """`ranks`, reordered like the compressed node list (sorted hosts), so the i-th rank
    belongs to the i-th expanded host. Launchers of one host keep their order."""
    if len(nodes) != len(ranks):
        return list(ranks)
    host_order = {node: i for i, node in enumerate(slurm_sort_addrs(list(dict.fromkeys(nodes))))}
    return [rank for _, rank in sorted(zip(nodes, ranks), key=lambda pair: host_order[pair[0]])]


def build_cycle_info(
    *,
    job_id: str,
    attempt_index: int,
    cycle_number: int,
    started_at_ms: int,
    cycle_log_file: str = "",
    active_nodes: Sequence[str] = (),
    active_ranks: Sequence[int] = (),
    standby_nodes: Sequence[str] = (),
) -> pb.NVRxCycleInfo:
    """The `NVRxCycleInfo` of a cycle that started at `started_at_ms` (milliseconds since the
    epoch), as the store host assigned it.

    `active_nodes` and `active_ranks` are aligned: the i-th rank is the i-th node's. The
    node lists are compressed into SLURM node lists (short host names, each once, sorted),
    and the ranks into a range string in that host order. `timestamp` is the start time.
    """
    return pb.NVRxCycleInfo(
        job_id=job_id,
        attempt_index=attempt_index,
        cycle_number=cycle_number,
        cycle_start_time=utc_iso(started_at_ms / 1000),
        cycle_log_file=cycle_log_file,
        active_nodes=hostnames_to_slurm_nodelist(list(active_nodes)),
        standby_nodes=hostnames_to_slurm_nodelist(list(standby_nodes)),
        active_ranks=ranks_to_range_str(_ranks_in_host_order(active_nodes, active_ranks)),
        timestamp=started_at_ms,
    )


def ended_cycle_info(info: pb.NVRxCycleInfo, ended_at_ms: int) -> pb.NVRxCycleInfo:
    """A copy of `info` for the cycle that ended at `ended_at_ms` (milliseconds since the
    epoch): `cycle_end_time` set, and `timestamp` the end time."""
    ended = pb.NVRxCycleInfo()
    ended.CopyFrom(info)
    ended.cycle_end_time = utc_iso(ended_at_ms / 1000)
    ended.timestamp = ended_at_ms
    return ended


def active_hosts(info: pb.NVRxCycleInfo) -> List[str]:
    """The active nodes of `info`, expanded: short host names, each once, sorted."""
    return slurm_nodelist_to_hostnames(info.active_nodes)


def active_launchers(info: pb.NVRxCycleInfo) -> int:
    """How many launchers are active in `info`'s cycle: one per active rank. Several can
    share a host, which active_nodes lists once."""
    return len(range_str_to_ranks(info.active_ranks))


# Set for simulations that run several launchers on one host (see
# launcher._validate_slurm_single_launcher_per_node).
MULTI_LAUNCHERS_PER_NODE_ENV = "NVRX_ENABLE_MULTI_LAUNCHERS_PER_NODE"


def launcher_node_id(host: str, pid: Optional[int] = None) -> str:
    """The node_id of a launcher on `host`: the host name, or "<host>_<pid>" (by default
    this process's pid) when several launchers may share a host
    (NVRX_ENABLE_MULTI_LAUNCHERS_PER_NODE)."""
    if parse_bool_env(MULTI_LAUNCHERS_PER_NODE_ENV):
        return f"{host}_{os.getpid() if pid is None else pid}"
    return host


def build_node_cycle_end_report(
    job_id: str,
    attempt_index: int,
    cycle: int,
    node_id: str,
    end_reason: int,
    workers: Sequence["Worker"],
    failures: Optional[Mapping[int, "ProcessFailure"]],
    detected_timestamp: int,
    timeouts: Optional[Mapping[int, "RankTimeout"]] = None,
) -> pb.NodeCycleEndReport:
    """Describe how the cycle ended on this launcher, when it observed the end.

    `node_id` identifies the launcher (launcher_node_id). `end_reason` is a pb.EndReason value, and `detected_timestamp` in milliseconds since the
    epoch (now_ms()). Workers in `failures` (keyed by global rank) exited; those also in
    `timeouts` (keyed by global rank) were stopped by their rank monitor, the others failed
    on their own. Each is listed in `exit_details`, as JSON: its global and local rank,
    pid, how it ended ("failed" or "stopped_by_rank_monitor"), its exit code, and for a
    rank monitor stop the timeout kind and sections. A WorkerFailed end becomes
    WorkerTerminatedByRankMonitor when any local worker was stopped by its rank monitor.
    """
    failures = failures or {}
    timeouts = timeouts or {}
    exited = []
    stopped_by_rank_monitor = False
    for worker in workers:
        failure = failures.get(worker.global_rank)
        if failure is None:
            continue
        timeout = timeouts.get(worker.global_rank)
        entry = {
            "rank": worker.global_rank,
            "local_rank": worker.local_rank,
            "pid": worker.id,
            "end": "stopped_by_rank_monitor" if timeout is not None else "failed",
            "exitcode": failure.exitcode,
        }
        if timeout is not None:
            stopped_by_rank_monitor = True
            entry["timeout"] = timeout.kind.value
            if timeout.sections:
                entry["sections"] = [
                    name[:MAX_SECTION_NAME_LEN] for name in timeout.sections[:MAX_TIMEOUT_SECTIONS]
                ]
        exited.append(entry)
    if len(exited) > MAX_EXIT_DETAILS_WORKERS:
        omitted = len(exited) - MAX_EXIT_DETAILS_WORKERS
        exited = exited[:MAX_EXIT_DETAILS_WORKERS] + [{"omitted": omitted}]
    if end_reason == pb.CycleEndReasonWorkerFailed and stopped_by_rank_monitor:
        end_reason = pb.CycleEndReasonWorkerTerminatedByRankMonitor
    return pb.NodeCycleEndReport(
        job_id=job_id,
        attempt_index=attempt_index,
        cycle_number=cycle,
        node_id=node_id,
        end_reason=end_reason,
        detected_timestamp=detected_timestamp,
        exit_details=json.dumps(exited, separators=(",", ":")) if exited else "",
    )


def check_cycle_info(info: pb.NVRxCycleInfo) -> None:
    """Check a received `NVRxCycleInfo`: its times and node and rank lists must parse.

    Raises:
        ValueError: A time or a list is malformed, or the start time is missing.
    """
    if iso_to_seconds(info.cycle_start_time) is None:
        raise ValueError(f"cycle info of cycle {info.cycle_number} has no start time")
    iso_to_seconds(info.cycle_end_time)
    slurm_nodelist_to_hostnames(info.active_nodes)
    slurm_nodelist_to_hostnames(info.standby_nodes)
    range_str_to_ranks(info.active_ranks)


def check_node_cycle_end_report(report: pb.NodeCycleEndReport) -> None:
    """Check a received `NodeCycleEndReport`.

    Raises:
        ValueError: The end reason is unknown to this version, or the detection time is
            missing (0).
    """
    if report.end_reason not in pb.EndReason.values():
        raise ValueError(f"unknown end reason {report.end_reason} in an end report")
    if not report.detected_timestamp:
        raise ValueError(f"end report of cycle {report.cycle_number} has no detection time")
