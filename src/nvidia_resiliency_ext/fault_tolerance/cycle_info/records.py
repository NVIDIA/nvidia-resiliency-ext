# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The records the job-wide cycle info is built from.

The store host records one `CycleStartInfo` per cycle, when the rendezvous round that
starts it closes. Each launcher records one `LocalCycleOutcome` per cycle, when the cycle
ends for it; outcomes are local facts, so the cycle end and which node caused a failure
are concluded from the outcomes of all launchers.

Serialize enum members with ``.value``: for ``(str, Enum)`` mixins, ``str()`` and
f-strings render the member name on newer Python versions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

if TYPE_CHECKING:
    # Annotations only: rank monitor messages import this module, so keep it light.
    from torch.distributed.elastic.agent.server.api import Worker
    from torch.distributed.elastic.multiprocessing.errors import ProcessFailure


@dataclass(frozen=True)
class CycleStartInfo:
    """How the rendezvous started a cycle, as the store host assigned it."""

    cycle: int
    # When the store host closed the round: time.time(), seconds since the Unix epoch in
    # UTC. Comparable across nodes only up to their clock skew, as
    # `LocalCycleOutcome.detected_at`.
    started_at: float
    active_nodes: tuple[str, ...] = ()
    standby_nodes: tuple[str, ...] = ()  # hot spares: joined the round, run no workers
    # Group ranks of the active nodes, aligned with `active_nodes`.
    active_ranks: tuple[int, ...] = ()
    cycle_log_file: str = ""  # empty when per-cycle logs are not configured


class CycleEndReason(str, Enum):
    """Why the cycle ended on this launcher"""

    COMPLETED = "completed"
    FAILED = "failed"  # a local worker failed or was stopped by its rank monitor
    PEER_RESTART = "peer_restart"  # a peer opened the next rendezvous round
    TERMINATED = "terminated"  # a no-restart decision (attribution, progress, or a peer)


class RankEndReason(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"  # exited with a nonzero code or a signal on its own
    STOPPED_BY_LAUNCHER = "stopped_by_launcher"
    # Exited after its rank monitor detected a timeout and signalled it. The monitor records
    # the timeout only once it signals the worker, so a worker that exits on its own after
    # detection (e.g. during a timeout dump capture) is reported as FAILED.
    STOPPED_BY_RMON = "stopped_by_rank_monitor"


class TimeoutKind(str, Enum):
    HEARTBEAT = "heartbeat"  # initial or subsequent heartbeat
    SECTION = "section"
    OUT_OF_SECTION = "out_of_section"


@dataclass(frozen=True)
class RankTimeout:
    kind: TimeoutKind
    sections: tuple[str, ...] = ()  # set only for SECTION


@dataclass(frozen=True)
class RankOutcome:
    global_rank: int
    local_rank: int
    pid: int
    end: RankEndReason
    exitcode: Optional[int] = None  # negative for a signal, as reported by torchelastic
    # How the worker ended beyond its exit code, e.g. an exception summary. Not captured
    # yet: always empty for now.
    exitdetails: str = ""
    timeout_kind: Optional[TimeoutKind] = None  # set only when end is STOPPED_BY_RMON
    # Workload section names; several open sections can time out at once.
    timeout_sections: tuple[str, ...] = ()


@dataclass(frozen=True)
class LocalCycleOutcome:
    cycle: int
    node: str
    node_end: CycleEndReason
    # When the launcher observed the end: time.time(), seconds since the Unix epoch in UTC.
    # Comparable across nodes only up to their clock skew (NTP/chrony, typically
    # milliseconds); NVRx does not synchronize or correct node clocks.
    detected_at: float
    ranks: tuple[RankOutcome, ...]

    @property
    def launcher(self) -> tuple[str, tuple[int, ...]]:
        """Identifies the reporting launcher: its node and its workers' global ranks, as
        hostnames can repeat when launchers share a host."""
        return self.node, tuple(rank.global_rank for rank in self.ranks)


# Bound each record when converting it to a wire format, so a workload with many or long
# section names, or long exit details, cannot grow it.
MAX_TIMEOUT_SECTIONS = 4
MAX_SECTION_NAME_LEN = 64
MAX_EXIT_DETAILS_LEN = 256


# global start


class JobEndReason(str, Enum):
    """How the cycle ended for the job, as concluded from all launchers' outcomes."""

    COMPLETED = "completed"
    # A worker failed on its own or was stopped by its rank monitor on a timeout; the
    # rank's record (RankOutcome) tells which, with the timeout's kind and sections.
    RANK_FAILURE = "rank_failure"
    TERMINATED = "terminated"  # a no-restart decision
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CycleSummary:
    # aggregated outcomes for the whole job
    cycle: int
    outcomes: tuple[LocalCycleOutcome, ...]  # one per reporting launcher, de-duplicated
    # Active launchers of the cycle start; None when the start was not received.
    launchers_expected: Optional[int] = None
    # TODO: group ranks instead of node names for missing and first_detected, once outcomes
    # carry their launcher's group rank; node names repeat when launchers share a host.
    missing: tuple[str, ...] = ()  # nodes of active launchers that did not report
    # Filled by optional aggregation steps; None means "not computed".
    end_reason: Optional[JobEndReason] = None
    first_detected: tuple[str, ...] = ()  # nodes that detected an issue first
    # When the cycle ended, in UTC epoch seconds like LocalCycleOutcome.detected_at.
    ended_at: Optional[float] = None


# global end


def build_local_cycle_outcome(
    cycle: int,
    node: str,
    node_end_reason: CycleEndReason,
    workers: Sequence["Worker"],
    failures: Optional[Mapping[int, "ProcessFailure"]],
    detected_at: float,
    timeouts: Optional[Mapping[int, RankTimeout]] = None,
) -> LocalCycleOutcome:
    """Classify each local worker at the moment the launcher observed the cycle end.

    Workers in `failures` (keyed by global rank) exited; those also in `timeouts` (keyed
    by global rank) were stopped by their rank monitor, the others failed on their own.
    When the cycle completed, the other workers completed too; otherwise the launcher
    stops them.
    """
    failures = failures or {}
    timeouts = timeouts or {}
    ranks = []
    for worker in workers:
        failure = failures.get(worker.global_rank)
        timeout = timeouts.get(worker.global_rank) if failure is not None else None
        if timeout is not None:
            end, exitcode = RankEndReason.STOPPED_BY_RMON, failure.exitcode
        elif failure is not None:
            end, exitcode = RankEndReason.FAILED, failure.exitcode
        elif node_end_reason is CycleEndReason.COMPLETED:
            end, exitcode = RankEndReason.COMPLETED, 0
        else:
            end, exitcode = RankEndReason.STOPPED_BY_LAUNCHER, None
        ranks.append(
            RankOutcome(
                global_rank=worker.global_rank,
                local_rank=worker.local_rank,
                pid=worker.id,
                end=end,
                exitcode=exitcode,
                exitdetails="",
                timeout_kind=timeout.kind if timeout is not None else None,
                timeout_sections=timeout.sections if timeout is not None else (),
            )
        )
    return LocalCycleOutcome(
        cycle=cycle,
        node=node,
        node_end=node_end_reason,
        detected_at=detected_at,
        ranks=tuple(ranks),
    )
