# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How a cycle ended on one launcher, as that launcher observed it.

Each launcher records one `LocalCycleOutcome` per cycle. The records are local facts;
deciding which node caused a job-wide failure needs the records of all launchers.

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


class CycleEndReason(str, Enum):
    """Why the cycle ended on this launcher"""

    COMPLETED = "completed"
    FAILED = "failed"  # a local worker failed or was stopped by its rank monitor
    PEER_RESTART = "peer_restart"  # a peer opened the next rendezvous round
    TERMINATED = "terminated"  # a no-restart decision (attribution, progress, or a peer)


class RankEndReason(str, Enum):
    """How one local worker ended."""

    COMPLETED = "completed"
    FAILED = "failed"  # exited with a nonzero code or a signal on its own
    STOPPED_BY_LAUNCHER = "stopped_by_launcher"
    # Exited after its rank monitor detected a timeout. The monitor records the timeout
    # before signaling, so a worker that died on its own just after detection (or during
    # a timeout dump capture) is reported here too: the monitor saw the hang first.
    STOPPED_BY_RMON = "stopped_by_rank_monitor"


class TimeoutKind(str, Enum):
    """Which rank monitor timeout stopped the worker."""

    HEARTBEAT = "heartbeat"  # initial or subsequent heartbeat
    SECTION = "section"
    OUT_OF_SECTION = "out_of_section"


@dataclass(frozen=True)
class RankTimeout:
    """The rank monitor timeout that stopped a worker, as reported to the launcher."""

    kind: TimeoutKind
    sections: tuple[str, ...] = ()  # set only for SECTION


@dataclass(frozen=True)
class RankOutcome:
    global_rank: int
    local_rank: int
    pid: int
    end: RankEndReason
    exitcode: Optional[int] = None  # negative for a signal, as reported by torchelastic
    timeout_kind: Optional[TimeoutKind] = None  # set only when end is STOPPED_BY_RMON
    # Workload section names; several open sections can time out at once.
    timeout_sections: tuple[str, ...] = ()


@dataclass(frozen=True)
class LocalCycleOutcome:
    cycle: int
    node: str
    node_end: CycleEndReason
    detected_at: float  # time.time() when the launcher observed the end
    ranks: tuple[RankOutcome, ...]


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
