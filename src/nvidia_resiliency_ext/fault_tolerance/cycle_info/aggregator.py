# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job-wide conclusions from the launchers' end reports of one cycle.

No I/O and no clock, so the CycleAnalyzer decides when to aggregate and where the result
goes.
"""

import heapq
import itertools
from typing import List, Optional, Set

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

# By default, at most this many end reports are listed in a cycle's JobCycleEndReport: the
# earliest (CycleAnalyzer's max_listed_end_reports).
MAX_LISTED_END_REPORTS = 256


class CycleEndReports:
    """The end reports received for one cycle, kept as they arrive.

    Keeps what the job-wide end report needs, without keeping every report:
    - the launchers (node_id) that reported: duplicates are dropped, missing ones counted;
    - whether every launcher that reported completed;
    - the earliest detection by a launcher that observed the end itself (_OBSERVED_ENDS),
      the launchers tied at it, and its end reason; and the earliest detection of any
      report, for cycles where no launcher observed the end (e.g. completed);
    - the `max_listed` earliest reports, except those of launchers that only followed a
      peer's restart: a max-heap by detection time, whose root is the latest kept.
    Adding a report is O(log max_listed); the kept reports are sorted once, when listed.
    """

    def __init__(self, max_listed: int = MAX_LISTED_END_REPORTS):
        self.max_listed = max_listed
        self.node_ids: Set[str] = set()
        self.completed_success = True
        self.earliest_timestamp: Optional[int] = None
        self.first_detected_timestamp: Optional[int] = None
        self.first_detected_nodes: List[str] = []
        self.first_detected_end_reason = pb.CycleEndReasonUnspecified
        # (-detected_timestamp, -arrival, report): heapq is a min-heap, so negated, the root
        # is the latest report kept. The arrival order breaks ties (the earlier is kept)
        # without comparing the messages.
        self._heap: List[tuple] = []
        self._arrival = itertools.count()

    def __len__(self) -> int:
        """How many launchers reported."""
        return len(self.node_ids)

    def add(self, report: pb.NodeCycleEndReport) -> bool:
        """Count `report`; False, with nothing changed, if its launcher already reported
        (the first report is kept)."""
        if report.node_id in self.node_ids:
            return False
        self.node_ids.add(report.node_id)
        timestamp = report.detected_timestamp
        if report.end_reason != pb.CycleEndReasonWorkerCompleted:
            self.completed_success = False
        if self.earliest_timestamp is None or timestamp < self.earliest_timestamp:
            self.earliest_timestamp = timestamp
        if report.end_reason in _OBSERVED_ENDS:
            if self.first_detected_timestamp is None or timestamp < self.first_detected_timestamp:
                self.first_detected_timestamp = timestamp
                self.first_detected_nodes = []
                self.first_detected_end_reason = report.end_reason
            if timestamp == self.first_detected_timestamp:
                self.first_detected_nodes.append(report.node_id)
        if report.end_reason != pb.CycleEndReasonWorkerTerminatedDueToPeer:
            entry = (-timestamp, -next(self._arrival), report)
            if len(self._heap) < self.max_listed:
                heapq.heappush(self._heap, entry)
            elif entry > self._heap[0]:  # earlier than the latest kept: replace it
                heapq.heapreplace(self._heap, entry)
        return True

    def listed(self) -> List[pb.NodeCycleEndReport]:
        """The kept reports, sorted by detection time (then node_id)."""
        return sorted(
            (report for _, _, report in self._heap),
            key=lambda report: (report.detected_timestamp, report.node_id),
        )


class CycleEndReportAggregator:
    """Builds the JobCycleEndReport of a cycle from its CycleEndReports.

    Stateless. `is_complete` and `missing_launchers` also need the cycle info: the
    CycleAnalyzer uses them to tell when every active launcher reported.
    """

    def aggregate(self, end_reports: Optional[CycleEndReports]) -> Optional[pb.JobCycleEndReport]:
        """How the cycle ended for the job, from the end reports received for it.

        end_reason: WorkerCompleted when every launcher that reported completed, else the
        end reason of the first launcher that observed the end, else Unspecified. Only the
        reports received are considered: launchers missing from them are not counted (see
        the comment of JobCycleEndReport.end_reason). Without end reports there is nothing
        to conclude: None.
        """
        if not end_reports:
            return None
        if end_reports.completed_success:
            end_reason = pb.CycleEndReasonWorkerCompleted
        else:
            end_reason = end_reports.first_detected_end_reason
        # Without a launcher that observed the end itself (e.g. a completed cycle), the
        # earliest detection of any launcher.
        first_detected_timestamp = end_reports.first_detected_timestamp
        if first_detected_timestamp is None:
            first_detected_timestamp = end_reports.earliest_timestamp
        return pb.JobCycleEndReport(
            end_reason=end_reason,
            first_detected_timestamp=(
                utc_iso_ms(first_detected_timestamp) if first_detected_timestamp is not None else ""
            ),
            first_detected_nodes=hostnames_to_slurm_nodelist(end_reports.first_detected_nodes),
            launcher_end_reports=end_reports.listed(),
        )

    @classmethod
    def is_complete(
        cls, start: Optional[pb.NVRxCycleInfo], end_reports: Optional[CycleEndReports]
    ) -> bool:
        """Whether every active launcher of the cycle reported. False when the start is
        unknown."""
        return start is not None and not cls.missing_launchers(start, end_reports)

    @staticmethod
    def missing_launchers(
        start: Optional[pb.NVRxCycleInfo], end_reports: Optional[CycleEndReports]
    ) -> int:
        """How many active launchers of the cycle sent no end report: active launchers
        (active_launchers) less those that reported; 0 when the start is unknown."""
        if start is None:
            return 0
        return max(0, active_launchers(start) - len(end_reports or ()))
