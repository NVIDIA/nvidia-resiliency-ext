# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job-wide conclusions from the launchers' outcomes of one cycle.

Pure functions of the records: no I/O and no clock, so the CycleAnalyzer decides when to
aggregate and where the result goes.
"""

from collections import Counter
from typing import Optional, Sequence

from .records import CycleEndReason, CycleStartInfo, CycleSummary, JobEndReason, LocalCycleOutcome

# Launchers that saw the cycle end themselves, rather than following a peer's restart.
_OBSERVED_ENDS = (CycleEndReason.FAILED, CycleEndReason.TERMINATED)


class CycleOutcomeAggregator:
    """Builds the CycleSummary of a cycle from its start and the launchers' outcomes.

    Stateless; override a step to change one conclusion.
    """

    def aggregate(
        self,
        cycle: int,
        start: Optional[CycleStartInfo],
        outcomes: Sequence[LocalCycleOutcome],
    ) -> CycleSummary:
        outcomes = self.deduplicate(outcomes)
        first = self.first_detected(outcomes)
        return CycleSummary(
            cycle=cycle,
            outcomes=outcomes,
            launchers_expected=len(start.active_nodes) if start is not None else None,
            missing=self.missing_nodes(start, outcomes),
            end_reason=self.end_reason(outcomes, first),
            first_detected=tuple(outcome.node for outcome in first),
            ended_at=self.ended_at(outcomes, first),
        )

    @staticmethod
    def deduplicate(outcomes: Sequence[LocalCycleOutcome]) -> tuple[LocalCycleOutcome, ...]:
        """One outcome per launcher (LocalCycleOutcome.launcher), the first received."""
        unique = {}
        for outcome in outcomes:
            unique.setdefault(outcome.launcher, outcome)
        return tuple(unique.values())

    @staticmethod
    def missing_nodes(
        start: Optional[CycleStartInfo], outcomes: Sequence[LocalCycleOutcome]
    ) -> tuple[str, ...]:
        """Active nodes of the cycle with no outcome; empty when the start is unknown.

        Counted per node, so with several launchers on one host each missing one counts.
        """
        if start is None:
            return ()
        missing = Counter(start.active_nodes) - Counter(outcome.node for outcome in outcomes)
        return tuple(sorted(missing.elements()))

    @staticmethod
    def first_detected(outcomes: Sequence[LocalCycleOutcome]) -> tuple[LocalCycleOutcome, ...]:
        """The outcomes of the launchers that observed the end first.

        Only launchers that saw a failure or a no-restart decision themselves count, not
        those that followed a peer's restart. Several when they share the earliest
        detection time; comparable only up to the nodes' clock skew.
        """
        observed = [outcome for outcome in outcomes if outcome.node_end in _OBSERVED_ENDS]
        if not observed:
            return ()
        earliest = min(outcome.detected_at for outcome in observed)
        return tuple(outcome for outcome in observed if outcome.detected_at == earliest)

    @staticmethod
    def end_reason(
        outcomes: Sequence[LocalCycleOutcome], first: Sequence[LocalCycleOutcome]
    ) -> JobEndReason:
        """Why the cycle ended for the job, from how it ended for the launchers that observed
        it first. How each rank ended, e.g. stopped by its rank monitor on a timeout, is in
        the launchers' outcomes."""
        if outcomes and all(o.node_end is CycleEndReason.COMPLETED for o in outcomes):
            return JobEndReason.COMPLETED
        if any(outcome.node_end is CycleEndReason.FAILED for outcome in first):
            return JobEndReason.RANK_FAILURE
        if any(outcome.node_end is CycleEndReason.TERMINATED for outcome in first):
            return JobEndReason.TERMINATED
        return JobEndReason.UNKNOWN

    @staticmethod
    def ended_at(
        outcomes: Sequence[LocalCycleOutcome], first: Sequence[LocalCycleOutcome]
    ) -> Optional[float]:
        """When the cycle ended: when it was first observed to end.

        A completed cycle ends when its last launcher completed. Without a launcher that
        observed the end, the earliest detection of any launcher; None without outcomes.
        """
        if not outcomes:
            return None
        if all(outcome.node_end is CycleEndReason.COMPLETED for outcome in outcomes):
            return max(outcome.detected_at for outcome in outcomes)
        if first:
            return first[0].detected_at
        return min(outcome.detected_at for outcome in outcomes)
