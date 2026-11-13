# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The records the job-wide cycle info is built from.

The store host records one `CycleInfo` per cycle, when the rendezvous round that starts
it closes, and updates it when the cycle ends. Each launcher records one
`NodeCycleEndReport` per cycle, when the cycle
ends for it; end reports are local facts, so the cycle end and which node caused a failure
are concluded from the end reports of all launchers.

Serialize enum members with ``.value``: for ``(str, Enum)`` mixins, ``str()`` and
f-strings render the member name on newer Python versions.
"""

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

if TYPE_CHECKING:
    # Annotations only: rank monitor messages import this module, so keep it light.
    from torch.distributed.elastic.agent.server.api import Worker
    from torch.distributed.elastic.multiprocessing.errors import ProcessFailure


@dataclass(frozen=True)
class CycleInfo:
    """A cycle as the store host assigned it; mirrors CycleInfo of nvrx_interface.proto.

    The store host publishes it when the rendezvous round that starts the cycle closes,
    then again with `ended_at` set when the cycle ends. `started_at` and `ended_at` are
    time.time(), seconds since the Unix epoch in UTC (ISO 8601 strings in the proto);
    `timestamp` is in milliseconds since the epoch (now_ms()). Comparable across nodes only
    up to their clock skew, as `NodeCycleEndReport.detected_timestamp`.
    """

    job_id: str
    attempt_index: int
    cycle_number: int
    started_at: float  # when the store host closed the round (cycle_start_time)
    ended_at: Optional[float] = None  # when the cycle ended (cycle_end_time); None until then
    cycle_log_file: str = ""  # empty when per-cycle logs are not configured
    # Host names of the active nodes and their group ranks, aligned. Compressed on the
    # wire (SLURM node range format): after transport, the nodes are short host names,
    # each once and sorted, and the ranks are in that host order.
    active_nodes: tuple[str, ...] = ()
    active_ranks: tuple[int, ...] = ()
    standby_nodes: tuple[str, ...] = ()  # hot spares: joined the round, run no workers
    # When this record was created, on the node that created it; milliseconds since the epoch.
    timestamp: int = 0


def now_ms() -> int:
    """The current time in milliseconds since the Unix epoch (UTC), as the records keep it."""
    return time.time_ns() // 1_000_000


class TimeoutKind(str, Enum):
    HEARTBEAT = "heartbeat"  # initial or subsequent heartbeat
    SECTION = "section"
    OUT_OF_SECTION = "out_of_section"


@dataclass(frozen=True)
class RankTimeout:
    kind: TimeoutKind
    sections: tuple[str, ...] = ()  # set only for SECTION


class EndReason(str, Enum):
    """How the cycle ended on a launcher; mirrors the EndReason of nvrx_interface.proto."""

    UNSPECIFIED = "unspecified"
    COMPLETED = "completed"
    FAILED = "failed"  # a local worker exited on its own
    TERMINATED_DUE_TO_PEER = "terminated_due_to_peer"  # a peer opened the next round
    # A local worker was stopped by its rank monitor on a timeout.
    TERMINATED_BY_RANK_MONITOR = "terminated_by_rank_monitor"
    TERMINATED_BY_ATTRIBUTION = "terminated_by_attribution"  # an attribution stop


@dataclass(frozen=True)
class NodeCycleEndReport:
    """How a cycle ended on one launcher; mirrors NodeCycleEndReport of nvrx_interface.proto.

    Recorded by the launcher's main thread when it observes the end; its publisher thread
    converts it to the proto message.
    """

    job_id: str
    attempt_index: int
    cycle_number: int
    node_name: str
    end_reason: EndReason
    # When the launcher observed the end, in milliseconds since the Unix epoch in UTC
    # (now_ms()). Comparable across nodes only up to their clock skew.
    detected_timestamp: int
    # JSON list, one entry per local worker that failed or was stopped by its rank monitor
    # (see build_node_cycle_end_report); "" when none did.
    exit_details: str = ""


# Bound each record, so a workload with many or long section names, or many failed workers,
# cannot grow it.
MAX_TIMEOUT_SECTIONS = 4
MAX_SECTION_NAME_LEN = 64
# Workers listed in NodeCycleEndReport.exit_details; the rest are counted, not listed.
MAX_EXIT_DETAILS_WORKERS = 16


# global start


class JobEndReason(str, Enum):
    """How the cycle ended for the job, as concluded from all launchers' end reports."""

    COMPLETED = "completed"
    # A worker failed on its own or was stopped by its rank monitor on a timeout; the
    # launcher's end report tells which.
    RANK_FAILURE = "rank_failure"
    TERMINATED = "terminated"  # a no-restart decision
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CycleSummary:
    # aggregated end reports for the whole job
    cycle: int
    outcomes: tuple[NodeCycleEndReport, ...]  # one per reporting launcher, de-duplicated
    # Active launchers of the cycle start; None when the start was not received.
    launchers_expected: Optional[int] = None
    # TODO: group ranks instead of node names for missing and first_detected, once end
    # reports carry their launcher's group rank; node names repeat when launchers share a
    # host.
    missing: tuple[str, ...] = ()  # nodes of active launchers that did not report
    # Filled by optional aggregation steps; None means "not computed".
    end_reason: Optional[JobEndReason] = None
    first_detected: tuple[str, ...] = ()  # nodes that detected an issue first
    # When the cycle ended, in milliseconds since the Unix epoch in UTC, like
    # NodeCycleEndReport.detected_timestamp.
    ended_at: Optional[int] = None


# global end


def build_node_cycle_end_report(
    job_id: str,
    attempt_index: int,
    cycle: int,
    node: str,
    end_reason: EndReason,
    workers: Sequence["Worker"],
    failures: Optional[Mapping[int, "ProcessFailure"]],
    detected_timestamp: int,
    timeouts: Optional[Mapping[int, RankTimeout]] = None,
) -> NodeCycleEndReport:
    """Describe how the cycle ended on this launcher, when it observed the end.

    Workers in `failures` (keyed by global rank) exited; those also in `timeouts` (keyed
    by global rank) were stopped by their rank monitor, the others failed on their own.
    Each is listed in `exit_details`, as JSON: its global and local rank, pid, how it
    ended ("failed" or "stopped_by_rank_monitor"), its exit code, and for a rank monitor
    stop the timeout kind and sections. A FAILED end becomes TERMINATED_BY_RANK_MONITOR
    when any local worker was stopped by its rank monitor.
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
    if end_reason is EndReason.FAILED and stopped_by_rank_monitor:
        end_reason = EndReason.TERMINATED_BY_RANK_MONITOR
    return NodeCycleEndReport(
        job_id=job_id,
        attempt_index=attempt_index,
        cycle_number=cycle,
        node_name=node,
        end_reason=end_reason,
        detected_timestamp=detected_timestamp,
        exit_details=json.dumps(exited, separators=(",", ":")) if exited else "",
    )
