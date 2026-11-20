# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job-wide conclusions from the launchers' end reports of one cycle.

Pure functions of the records: no I/O and no clock, so the CycleAnalyzer decides when to
aggregate and where the result goes.
"""

from typing import Optional, Sequence

from nvidia_resiliency_ext.shared_utils.proto import nvrx_interface_pb2 as pb

from ..cycle_info_writer import utc_iso_ms
from ..utils import hostnames_to_slurm_nodelist
from .messages import active_launchers

# Launchers that saw the cycle end themselves, rather than following a peer's restart:
# their workers failed or were stopped on a timeout, or a no-restart decision (attribution,
# or no training progress across restarts) stopped them.
_OBSERVED_ENDS = (
    pb.CycleEndReasonWorkerFailed,
    pb.CycleEndReasonWorkerTerminatedByRankMonitor,
    pb.CycleEndReasonWorkerTerminatedByAttribution,
    pb.CycleEndReasonWorkerTerminatedDueToNoProgress,
)


class CycleEndReportAggregator:
    """Builds the JobCycleEndReport of a cycle from the launchers' end reports.

    Stateless; override a step to change one conclusion. `is_complete` and
    `missing_launchers` also need the cycle info: the CycleAnalyzer uses them to tell when
    every active launcher reported.
    """

    def aggregate(
        self, end_reports: Sequence[pb.NodeCycleEndReport]
    ) -> Optional[pb.JobCycleEndReport]:
        """How the cycle ended for the job, from the end reports received for it.

        Lists the end reports of the launchers that observed the end themselves, sorted by
        when they detected it; those that only followed a peer's restart are left out.
        Without end reports there is nothing to conclude: None.
        """
        if not end_reports:
            return None
        end_reports = self.deduplicate(end_reports)
        first = self.first_detected(end_reports)
        first_detected_timestamp = self.first_detected_timestamp(end_reports, first)
        return pb.JobCycleEndReport(
            end_reason=self.end_reason(end_reports, first),
            first_detected_timestamp=(
                utc_iso_ms(first_detected_timestamp) if first_detected_timestamp is not None else ""
            ),
            first_detected_nodes=hostnames_to_slurm_nodelist([report.node_id for report in first]),
            launcher_end_reports=self.listed_end_reports(end_reports),
        )

    @staticmethod
    def deduplicate(
        end_reports: Sequence[pb.NodeCycleEndReport],
    ) -> tuple[pb.NodeCycleEndReport, ...]:
        """One end report per launcher (node_id), the first received."""
        unique = {}
        for report in end_reports:
            unique.setdefault(report.node_id, report)
        return tuple(unique.values())

    @classmethod
    def is_complete(
        cls, start: Optional[pb.NVRxCycleInfo], end_reports: Sequence[pb.NodeCycleEndReport]
    ) -> bool:
        """Whether every active launcher of the cycle reported: as many launchers (distinct
        node_ids) as active launchers. False when the start is unknown."""
        return start is not None and not cls.missing_launchers(start, end_reports)

    @staticmethod
    def missing_launchers(
        start: Optional[pb.NVRxCycleInfo], end_reports: Sequence[pb.NodeCycleEndReport]
    ) -> int:
        """How many active launchers of the cycle sent no end report: active launchers
        (active_launchers) less those that reported (distinct node_ids); 0 when the start is
        unknown."""
        if start is None:
            return 0
        reported = len({report.node_id for report in end_reports})
        return max(0, active_launchers(start) - reported)

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
        only followed a peer's restart, sorted by detection time (then node_id)."""
        return sorted(
            (
                report
                for report in end_reports
                if report.end_reason != pb.CycleEndReasonWorkerTerminatedDueToPeer
            ),
            key=lambda report: (report.detected_timestamp, report.node_id),
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
