# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How a cycle ended on one launcher, as that launcher observed it.

Each launcher records one `LocalCycleOutcome` per cycle. The records are local facts;
deciding which node caused a job-wide failure needs the records of all launchers.

Serialize enum members with ``.value``: for ``(str, Enum)`` mixins, ``str()`` and
f-strings render the member name on newer Python versions.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

if TYPE_CHECKING:
    # Annotations only: rank monitor messages import this module, so keep it light.
    from torch.distributed.elastic.agent.server.api import Worker
    from torch.distributed.elastic.multiprocessing.errors import ProcessFailure


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


# global start


class JobEndReason(str, Enum):
    """How the cycle ended for the job, as concluded from all launchers' outcomes."""

    COMPLETED = "completed"
    RANK_FAILURE = "rank_failure"  # a worker failed on its own
    TIMEOUT_FAILURE = "timeout_failure"  # a rank monitor stopped a hung worker
    TERMINATED = "terminated"  # a no-restart decision
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CycleSummary:
    # aggregated outcomes for the whole job
    cycle: int
    outcomes: tuple[LocalCycleOutcome, ...]  # one per reporting launcher, de-duplicated
    missing: tuple[
        int, ...
    ] = ()  # group ranks of active launchers that did not report - may end up not needing that
    # Filled by optional aggregation steps; None means "not computed".
    end_reason: Optional[JobEndReason] = None
    first_detected: tuple[int, ...] = ()  # group ranks that detected an issue first


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


# Wire format: one JSON object per line, so records appended to one store key can be split
# on newlines (JSON escapes newlines inside strings). Ranks are positional lists to keep
# records small; bump WIRE_VERSION when changing the layout.
WIRE_VERSION = 2
# Bound each record so a workload with many or long section names, or long exit details,
# cannot grow it.
MAX_TIMEOUT_SECTIONS = 4
MAX_SECTION_NAME_LEN = 64
MAX_EXIT_DETAILS_LEN = 256


def encode_outcome(outcome: LocalCycleOutcome) -> bytes:
    """Serialize `outcome` as one newline-terminated JSON record."""
    record = {
        "v": WIRE_VERSION,
        "c": outcome.cycle,
        "n": outcome.node,
        "e": outcome.node_end.value,
        "t": round(outcome.detected_at, 3),
        "r": [
            [
                rank.global_rank,
                rank.local_rank,
                rank.pid,
                rank.end.value,
                rank.exitcode,
                rank.exitdetails[:MAX_EXIT_DETAILS_LEN],
                rank.timeout_kind.value if rank.timeout_kind is not None else None,
                [
                    name[:MAX_SECTION_NAME_LEN]
                    for name in rank.timeout_sections[:MAX_TIMEOUT_SECTIONS]
                ],
            ]
            for rank in outcome.ranks
        ],
    }
    return json.dumps(record, separators=(",", ":")).encode() + b"\n"


def decode_outcome(line: bytes) -> LocalCycleOutcome:
    """Parse one record written by `encode_outcome`.

    Raises:
        ValueError: The record is malformed or has an unsupported version.
    """
    try:
        record = json.loads(line)
        if record["v"] != WIRE_VERSION:
            raise ValueError(f"unsupported cycle outcome version {record['v']!r}")
        ranks = tuple(
            RankOutcome(
                global_rank=global_rank,
                local_rank=local_rank,
                pid=pid,
                end=RankEndReason(end),
                exitcode=exitcode,
                exitdetails=exitdetails,
                timeout_kind=TimeoutKind(kind) if kind is not None else None,
                timeout_sections=tuple(sections),
            )
            for global_rank, local_rank, pid, end, exitcode, exitdetails, kind, sections in record[
                "r"
            ]
        )
        return LocalCycleOutcome(
            cycle=record["c"],
            node=record["n"],
            node_end=CycleEndReason(record["e"]),
            detected_at=record["t"],
            ranks=ranks,
        )
    except (KeyError, TypeError) as e:
        raise ValueError(f"malformed cycle outcome record: {e!r}") from e
