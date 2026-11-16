# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job-wide conclusions from the launchers' end reports of one cycle.

Pure functions of the records: no I/O and no clock, so the CycleAnalyzer decides when to
aggregate and where the result goes.
"""

from collections import Counter
from typing import Optional, Sequence

from nvidia_resiliency_ext.shared_utils.proto import nvrx_interface_pb2 as pb

from ..cycle_info_writer import utc_iso_ms
from .messages import active_hosts

# Launchers that saw the cycle end themselves, rather than following a peer's restart:
# their workers failed or were stopped on a timeout, or a no-restart decision stopped them.
_OBSERVED_ENDS = (
    pb.CycleEndReasonWorkerFailed,
    pb.CycleEndReasonWorkerTerminatedByRankMonitor,
    pb.CycleEndReasonWorkerTerminatedByAttribution,
)


def _short_host(name: str) -> str:
    """The host name without its domain, as the cycle info's node lists carry it."""
    return name.split(".")[0]


class CycleEndReportAggregator:
    """Builds the JobCycleEndReport of a cycle from the launchers' end reports.

    Stateless; override a step to change one conclusion. `missing_nodes` also needs the
    cycle info: the CycleAnalyzer uses it to tell when every active launcher reported.
    """

    def aggregate(self, end_reports: Sequence[pb.NodeCycleEndReport]) -> pb.JobCycleEndReport:
        """How the cycle ended for the job, from the end reports received for it.

        Lists the end reports of the launchers that observed the end themselves, sorted by
        when they detected it; those that only followed a peer's restart are left out.
        """
        end_reports = self.deduplicate(end_reports)
        first = self.first_detected(end_reports)
        first_detected_timestamp = self.first_detected_timestamp(end_reports, first)
        return pb.JobCycleEndReport(
            end_reason=self.end_reason(end_reports, first),
            first_detected_timestamp=(
                utc_iso_ms(first_detected_timestamp) if first_detected_timestamp is not None else ""
            ),
            first_detected_nodes=[report.node_name for report in first],
            launcher_end_reports=self.listed_end_reports(end_reports),
        )

    @staticmethod
    def deduplicate(
        end_reports: Sequence[pb.NodeCycleEndReport],
    ) -> tuple[pb.NodeCycleEndReport, ...]:
        """One end report per launcher, the first received.

        TODO: launchers are told apart by node name only, so launchers sharing a host
        count as one.
        """
        unique = {}
        for report in end_reports:
            unique.setdefault(report.node_name, report)
        return tuple(unique.values())

    @staticmethod
    def missing_nodes(
        start: Optional[pb.CycleInfo], end_reports: Sequence[pb.NodeCycleEndReport]
    ) -> tuple[str, ...]:
        """Active hosts of the cycle with no end report; empty when the start is unknown.

        Compared by short host name: the cycle info's node lists carry each host once,
        without its domain.
        """
        if start is None:
            return ()
        reported = Counter(_short_host(report.node_name) for report in end_reports)
        missing = Counter(_short_host(node) for node in active_hosts(start)) - reported
        return tuple(sorted(missing.elements()))

    @staticmethod
    def first_detected(
        end_reports: Sequence[pb.NodeCycleEndReport],
    ) -> tuple[pb.NodeCycleEndReport, ...]:
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
        end_reports: Sequence[pb.NodeCycleEndReport], first: Sequence[pb.NodeCycleEndReport]
    ) -> int:
        """Why the cycle ended for the job, as a pb.EndReason value: WorkerCompleted when
        every launcher completed, else the end reason of the first launcher that observed
        the end, else Unspecified. The details of each launcher's workers are in its end
        report."""
        if end_reports and all(
            r.end_reason == pb.CycleEndReasonWorkerCompleted for r in end_reports
        ):
            return pb.CycleEndReasonWorkerCompleted
        if first:
            return first[0].end_reason
        return pb.CycleEndReasonUnspecified

    @staticmethod
    def listed_end_reports(
        end_reports: Sequence[pb.NodeCycleEndReport],
    ) -> list[pb.NodeCycleEndReport]:
        """The end reports listed in the JobCycleEndReport: all but those of launchers that
        only followed a peer's restart, sorted by detection time (then node name)."""
        return sorted(
            (
                report
                for report in end_reports
                if report.end_reason != pb.CycleEndReasonWorkerTerminatedDueToPeer
            ),
            key=lambda report: (report.detected_timestamp, report.node_name),
        )

    @staticmethod
    def first_detected_timestamp(
        end_reports: Sequence[pb.NodeCycleEndReport], first: Sequence[pb.NodeCycleEndReport]
    ) -> Optional[int]:
        """When a launcher first detected the end of the cycle, in milliseconds since the
        epoch: the detection time of the launchers that observed the end first.

        Without a launcher that observed the end itself (e.g. a completed cycle), the
        earliest detection of any launcher; None without reports.
        """
        if not end_reports:
            return None
        if first:
            return first[0].detected_timestamp
        return min(report.detected_timestamp for report in end_reports)
