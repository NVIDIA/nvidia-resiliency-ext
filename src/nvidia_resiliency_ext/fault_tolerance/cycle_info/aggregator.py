# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job-wide conclusions from the launchers' end reports of one cycle.

Pure functions of the records: no I/O and no clock, so the CycleAnalyzer decides when to
aggregate and where the result goes.
"""

from collections import Counter
from typing import Optional, Sequence

from .records import CycleInfo, CycleSummary, EndReason, JobEndReason, NodeCycleEndReport

# Launchers whose workers failed or were stopped on a timeout.
_RANK_FAILURES = (EndReason.FAILED, EndReason.TERMINATED_BY_RANK_MONITOR)
# Launchers stopped by a no-restart decision.
_NO_RESTART_STOPS = (EndReason.TERMINATED_BY_ATTRIBUTION,)
# Launchers that saw the cycle end themselves, rather than following a peer's restart.
_OBSERVED_ENDS = _RANK_FAILURES + _NO_RESTART_STOPS


def _short_host(name: str) -> str:
    """The host name without its domain, as the cycle info's node lists carry it."""
    return name.split(".")[0]


class CycleOutcomeAggregator:
    """Builds the CycleSummary of a cycle from its cycle info and the launchers' end reports.

    Stateless; override a step to change one conclusion.
    """

    def aggregate(
        self,
        cycle: int,
        start: Optional[CycleInfo],
        outcomes: Sequence[NodeCycleEndReport],
    ) -> CycleSummary:
        outcomes = self.deduplicate(outcomes)
        first = self.first_detected(outcomes)
        return CycleSummary(
            cycle=cycle,
            outcomes=outcomes,
            launchers_expected=self.launchers_expected(start),
            missing=self.missing_nodes(start, outcomes),
            end_reason=self.end_reason(outcomes, first),
            first_detected=tuple(outcome.node_name for outcome in first),
            ended_at=self.ended_at(outcomes, first),
        )

    @staticmethod
    def deduplicate(outcomes: Sequence[NodeCycleEndReport]) -> tuple[NodeCycleEndReport, ...]:
        """One end report per launcher, the first received.

        TODO: launchers are told apart by node name only, so launchers sharing a host
        count as one.
        """
        unique = {}
        for outcome in outcomes:
            unique.setdefault(outcome.node_name, outcome)
        return tuple(unique.values())

    @staticmethod
    def launchers_expected(start: Optional[CycleInfo]) -> Optional[int]:
        """Active launchers of the cycle, one per active group rank; None without the start."""
        if start is None:
            return None
        return len(start.active_ranks) or len(start.active_nodes)

    @staticmethod
    def missing_nodes(
        start: Optional[CycleInfo], outcomes: Sequence[NodeCycleEndReport]
    ) -> tuple[str, ...]:
        """Active hosts of the cycle with no end report; empty when the start is unknown.

        Compared by short host name: the cycle info's node lists carry each host once,
        without its domain.
        """
        if start is None:
            return ()
        reported = Counter(_short_host(outcome.node_name) for outcome in outcomes)
        missing = Counter(_short_host(node) for node in start.active_nodes) - reported
        return tuple(sorted(missing.elements()))

    @staticmethod
    def first_detected(outcomes: Sequence[NodeCycleEndReport]) -> tuple[NodeCycleEndReport, ...]:
        """The end reports of the launchers that observed the end first.

        Only launchers that saw a failure or a no-restart decision themselves count, not
        those that followed a peer's restart. Several when they share the earliest
        detection time; comparable only up to the nodes' clock skew.
        """
        observed = [outcome for outcome in outcomes if outcome.end_reason in _OBSERVED_ENDS]
        if not observed:
            return ()
        earliest = min(outcome.detected_timestamp for outcome in observed)
        return tuple(outcome for outcome in observed if outcome.detected_timestamp == earliest)

    @staticmethod
    def end_reason(
        outcomes: Sequence[NodeCycleEndReport], first: Sequence[NodeCycleEndReport]
    ) -> JobEndReason:
        """Why the cycle ended for the job, from how it ended for the launchers that observed
        it first. The details of each launcher's workers are in its end report."""
        if outcomes and all(o.end_reason is EndReason.COMPLETED for o in outcomes):
            return JobEndReason.COMPLETED
        if any(outcome.end_reason in _RANK_FAILURES for outcome in first):
            return JobEndReason.RANK_FAILURE
        if any(outcome.end_reason in _NO_RESTART_STOPS for outcome in first):
            return JobEndReason.TERMINATED
        return JobEndReason.UNKNOWN

    @staticmethod
    def ended_at(
        outcomes: Sequence[NodeCycleEndReport], first: Sequence[NodeCycleEndReport]
    ) -> Optional[int]:
        """When the cycle ended, in milliseconds since the epoch: when it was first observed
        to end.

        A completed cycle ends when its last launcher completed. Without a launcher that
        observed the end, the earliest detection of any launcher; None without reports.
        """
        if not outcomes:
            return None
        if all(outcome.end_reason is EndReason.COMPLETED for outcome in outcomes):
            return max(outcome.detected_timestamp for outcome in outcomes)
        if first:
            return first[0].detected_timestamp
        return min(outcome.detected_timestamp for outcome in outcomes)
