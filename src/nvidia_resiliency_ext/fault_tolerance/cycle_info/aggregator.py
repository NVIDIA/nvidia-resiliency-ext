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


class CycleEndReportAggregator:
    """Builds the CycleSummary of a cycle from its cycle info and the launchers' end reports.

    Stateless; override a step to change one conclusion.
    """

    def aggregate(
        self,
        cycle: int,
        start: Optional[CycleInfo],
        end_reports: Sequence[NodeCycleEndReport],
    ) -> CycleSummary:
        end_reports = self.deduplicate(end_reports)
        first = self.first_detected(end_reports)
        return CycleSummary(
            cycle=cycle,
            end_reports=end_reports,
            launchers_expected=self.launchers_expected(start),
            missing=self.missing_nodes(start, end_reports),
            end_reason=self.end_reason(end_reports, first),
            first_detected=tuple(report.node_name for report in first),
            ended_at=self.ended_at(end_reports, first),
        )

    @staticmethod
    def deduplicate(
        end_reports: Sequence[NodeCycleEndReport],
    ) -> tuple[NodeCycleEndReport, ...]:
        """One end report per launcher, the first received.

        TODO: launchers are told apart by node name only, so launchers sharing a host
        count as one.
        """
        unique = {}
        for report in end_reports:
            unique.setdefault(report.node_name, report)
        return tuple(unique.values())

    @staticmethod
    def launchers_expected(start: Optional[CycleInfo]) -> Optional[int]:
        """Active launchers of the cycle, one per active group rank; None without the start."""
        if start is None:
            return None
        return len(start.active_ranks) or len(start.active_nodes)

    @staticmethod
    def missing_nodes(
        start: Optional[CycleInfo], end_reports: Sequence[NodeCycleEndReport]
    ) -> tuple[str, ...]:
        """Active hosts of the cycle with no end report; empty when the start is unknown.

        Compared by short host name: the cycle info's node lists carry each host once,
        without its domain.
        """
        if start is None:
            return ()
        reported = Counter(_short_host(report.node_name) for report in end_reports)
        missing = Counter(_short_host(node) for node in start.active_nodes) - reported
        return tuple(sorted(missing.elements()))

    @staticmethod
    def first_detected(
        end_reports: Sequence[NodeCycleEndReport],
    ) -> tuple[NodeCycleEndReport, ...]:
        """The end reports of the launchers that observed the end first.

        Only launchers that saw a failure or a no-restart decision themselves count, not
        those that followed a peer's restart. Several when they share the earliest
        detection time; comparable only up to the nodes' clock skew.
        """
        observed = [report for report in end_reports if report.end_reason in _OBSERVED_ENDS]
        if not observed:
            return ()
        earliest = min(report.detected_timestamp for report in observed)
        return tuple(report for report in observed if report.detected_timestamp == earliest)

    @staticmethod
    def end_reason(
        end_reports: Sequence[NodeCycleEndReport], first: Sequence[NodeCycleEndReport]
    ) -> JobEndReason:
        """Why the cycle ended for the job, from how it ended for the launchers that observed
        it first. The details of each launcher's workers are in its end report."""
        if end_reports and all(r.end_reason is EndReason.COMPLETED for r in end_reports):
            return JobEndReason.COMPLETED
        if any(report.end_reason in _RANK_FAILURES for report in first):
            return JobEndReason.RANK_FAILURE
        if any(report.end_reason in _NO_RESTART_STOPS for report in first):
            return JobEndReason.TERMINATED
        return JobEndReason.UNKNOWN

    @staticmethod
    def ended_at(
        end_reports: Sequence[NodeCycleEndReport], first: Sequence[NodeCycleEndReport]
    ) -> Optional[int]:
        """When the cycle ended, in milliseconds since the epoch: when it was first observed
        to end.

        A completed cycle ends when its last launcher completed. Without a launcher that
        observed the end, the earliest detection of any launcher; None without reports.
        """
        if not end_reports:
            return None
        if all(report.end_reason is EndReason.COMPLETED for report in end_reports):
            return max(report.detected_timestamp for report in end_reports)
        if first:
            return first[0].detected_timestamp
        return min(report.detected_timestamp for report in end_reports)
