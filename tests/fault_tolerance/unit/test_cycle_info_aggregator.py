# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nvidia_resiliency_ext.fault_tolerance.cycle_info.aggregator import (
    MAX_LISTED_END_REPORTS,
    CycleEndReportAggregator,
    CycleEndReports,
)
from nvidia_resiliency_ext.fault_tolerance.cycle_info.messages import build_cycle_info
from nvidia_resiliency_ext.shared_utils.proto import nvrx_interface_pb2 as pb

START = build_cycle_info(
    job_id="job1",
    attempt_index=0,
    cycle_number=3,
    started_at_ms=1_000,
    active_nodes=("n0", "n1", "n2"),
    active_ranks=(0, 1, 2),
)


def _report(node, end_reason, detected_timestamp):
    return pb.NodeCycleEndReport(
        job_id="job1",
        cycle_number=3,
        node_id=node,
        end_reason=end_reason,
        detected_timestamp=detected_timestamp,
    )


def _failed(node, detected_timestamp):
    return _report(node, pb.CycleEndReasonWorkerFailed, detected_timestamp)


def _peer(node, detected_timestamp):
    return _report(node, pb.CycleEndReasonWorkerTerminatedDueToPeer, detected_timestamp)


def _completed(node, detected_timestamp):
    return _report(node, pb.CycleEndReasonWorkerCompleted, detected_timestamp)


def _end_reports(reports, max_listed=MAX_LISTED_END_REPORTS):
    """The CycleEndReports of `reports`, added in order."""
    end_reports = CycleEndReports(max_listed)
    for report in reports:
        end_reports.add(report)
    return end_reports


def aggregate(reports, max_listed=MAX_LISTED_END_REPORTS):
    return CycleEndReportAggregator().aggregate(_end_reports(reports, max_listed))


def test_rank_failure_first_detected_by_the_failing_launcher():
    end_report = aggregate([_peer("n0", 12_000), _failed("n1", 10_000)])

    assert end_report.end_reason == pb.CycleEndReasonWorkerFailed
    assert end_report.first_detected_nodes == "n[1]"  # node range format, as active_nodes
    assert end_report.first_detected_timestamp == "1970-01-01T00:00:10.000Z"


def test_rank_monitor_stop_first_is_the_job_reason():
    end_report = aggregate(
        [
            _report("n0", pb.CycleEndReasonWorkerTerminatedByRankMonitor, 10_000),
            _failed("n1", 11_000),  # a later, consequent failure
        ]
    )
    assert end_report.end_reason == pb.CycleEndReasonWorkerTerminatedByRankMonitor
    assert end_report.first_detected_nodes == "n[0]"


@pytest.mark.parametrize(
    "end_reason",
    [
        pb.CycleEndReasonWorkerTerminatedByAttribution,
        pb.CycleEndReasonWorkerTerminatedDueToNoProgress,
    ],
)
def test_no_restart_decision_first_is_the_job_reason(end_reason):
    """A launcher stopped by a no-restart decision observed the end itself: it is listed,
    and its decision is the job's end reason."""
    report = _report("n0", end_reason, 10_000)
    end_report = aggregate([report, _peer("n1", 11_000)])

    assert end_report.end_reason == end_reason
    assert end_report.first_detected_nodes == "n[0]"
    assert list(end_report.launcher_end_reports) == [report]


def test_an_attribution_stop_of_the_job_controller_is_the_job_reason():
    """Under supervised control the job-controller latches the STOP and no launcher does:
    every launcher reports it as the cause of its stop, so the job's end keeps it."""
    reports = [
        _report(node, pb.CycleEndReasonWorkerTerminatedByAttribution, at)
        for node, at in (("n1", 10_100), ("n0", 10_000), ("n2", 10_200))
    ]
    end_report = aggregate(reports)

    assert end_report.end_reason == pb.CycleEndReasonWorkerTerminatedByAttribution
    assert end_report.first_detected_nodes == "n[0]"  # the first launcher to see the stop
    assert [r.node_id for r in end_report.launcher_end_reports] == ["n0", "n1", "n2"]


def test_job_reason_follows_the_first_detector():
    end_report = aggregate(
        [
            _failed("n0", 10_000),
            _report("n1", pb.CycleEndReasonWorkerTerminatedByAttribution, 11_000),
        ]
    )
    assert end_report.end_reason == pb.CycleEndReasonWorkerFailed


def test_completed_cycle_is_first_detected_by_the_first_launcher_that_completed():
    end_report = aggregate([_completed(n, 10_002 - i) for i, n in enumerate(("n0", "n1", "n2"))])
    assert end_report.end_reason == pb.CycleEndReasonWorkerCompleted
    assert end_report.first_detected_timestamp == "1970-01-01T00:00:10.000Z"
    assert end_report.first_detected_nodes == ""


def test_only_peer_restarts_leave_the_reason_unspecified():
    end_report = aggregate([_peer("n0", 12_000), _peer("n1", 11_000)])
    assert end_report.end_reason == pb.CycleEndReasonUnspecified
    assert end_report.first_detected_nodes == ""
    # The earliest detection of any launcher.
    assert end_report.first_detected_timestamp == "1970-01-01T00:00:11.000Z"


def test_unspecified_end_is_unspecified():
    end_report = aggregate([_report("n0", pb.CycleEndReasonUnspecified, 10_000)])
    assert end_report.end_reason == pb.CycleEndReasonUnspecified


def test_launchers_that_observed_the_end_are_listed_by_detection_time():
    late = _report("n2", pb.CycleEndReasonWorkerTerminatedByRankMonitor, 20_500)
    early = _failed("n0", 10_250)
    peer = _peer("n1", 11_000)

    end_report = aggregate([late, peer, early])

    # Peers left out; the listed reports are copies of the received ones.
    assert list(end_report.launcher_end_reports) == [early, late]


def test_no_reports_give_no_end_report():
    assert aggregate([]) is None
    assert CycleEndReportAggregator().aggregate(None) is None


def test_only_the_earliest_reports_are_listed_sorted():
    reports = [_failed(f"n{i}", 10_000 + (i * 7) % 10) for i in range(10)]  # arrival != time
    end_report = aggregate(reports + [_peer("p", 1)], max_listed=3)

    # The 3 earliest of the launchers that did not follow a peer, earliest first.
    expected = sorted(reports, key=lambda r: (r.detected_timestamp, r.node_id))[:3]
    assert list(end_report.launcher_end_reports) == expected


def test_a_capped_list_still_counts_every_launcher():
    reports = [_failed(f"n{i}", 10_000 + i) for i in range(5)]
    end_reports = _end_reports(reports, max_listed=2)

    assert len(end_reports) == 5
    assert len(end_reports.listed()) == 2


def test_every_launcher_tied_first_is_named_even_beyond_the_listed():
    end_report = aggregate([_failed(f"n{i}", 10_000) for i in range(5)], max_listed=2)

    assert end_report.first_detected_nodes == "n[0-4]"
    assert len(end_report.launcher_end_reports) == 2


def test_an_earlier_detection_replaces_the_first_detected():
    end_report = aggregate(
        [
            _failed("n0", 10_000),
            _failed("n1", 10_000),
            _report("n2", pb.CycleEndReasonWorkerTerminatedByRankMonitor, 9_000),
        ]
    )
    assert end_report.first_detected_nodes == "n[2]"
    assert end_report.end_reason == pb.CycleEndReasonWorkerTerminatedByRankMonitor


def test_ties_in_detection_time_are_all_first():
    end_report = aggregate([_failed("n0", 10_000), _failed("n1", 10_000)])
    assert end_report.first_detected_nodes == "n[0-1]"
    assert end_report.first_detected_timestamp == "1970-01-01T00:00:10.000Z"


def test_first_detected_nodes_are_compressed_like_the_active_nodes():
    """Short host names in SLURM node range format; launchers of one host count once."""
    reports = [
        _failed("cpu-0013.cm.cluster_123", 10_000),
        _failed("cpu-0013.cm.cluster_456", 10_000),
        _failed("cpu-0014.cm.cluster", 10_000),
        _failed("cpu-0016.cm.cluster", 10_000),
    ]
    assert aggregate(reports).first_detected_nodes == "cpu-[0013-0014,0016]"


def test_duplicate_reports_keep_the_first():
    first, second = _failed("n0", 10_000), _failed("n0", 9_000)
    end_reports = CycleEndReports()
    assert end_reports.add(first)
    assert not end_reports.add(second)
    assert len(end_reports) == 1
    assert list(aggregate([first, second]).launcher_end_reports) == [first]


def test_launchers_sharing_a_host_are_told_apart_by_node_id():
    assert len(_end_reports([_peer("n0_100", 12_000), _peer("n0_101", 12_000)])) == 2
    assert len(_end_reports([_peer("n0", 12_000), _peer("n0", 12_000)])) == 1


def missing_launchers(start, reports):
    return CycleEndReportAggregator.missing_launchers(start, _end_reports(reports))


def is_complete(start, reports):
    return CycleEndReportAggregator.is_complete(start, _end_reports(reports))


# Three launchers on one host, as in a simulation (NVRX_ENABLE_MULTI_LAUNCHERS_PER_NODE).
SHARED_HOST_START = build_cycle_info(
    job_id="job1",
    attempt_index=0,
    cycle_number=3,
    started_at_ms=1_000,
    active_nodes=("n0", "n0", "n0"),
    active_ranks=(0, 1, 2),
)


def test_a_cycle_is_complete_when_every_launcher_reported():
    reports = [_peer("n0", 12_000), _failed("n1", 10_000)]
    assert (missing_launchers(START, reports), is_complete(START, reports)) == (1, False)
    reports.append(_peer("n2.cluster", 12_000))
    assert (missing_launchers(START, reports), is_complete(START, reports)) == (0, True)
    assert not is_complete(None, reports)


def test_launchers_sharing_a_host_are_counted():
    reports = [_failed("n0_100", 10_000), _peer("n0.cluster_101", 12_000)]
    # Two of the host's three launchers reported.
    assert missing_launchers(SHARED_HOST_START, reports) == 1
    assert not is_complete(SHARED_HOST_START, reports)
    reports.append(_peer("n0_102", 12_000))
    assert is_complete(SHARED_HOST_START, reports)


def test_a_duplicate_does_not_count_twice():
    reports = [_peer("n0_100", 12_000), _peer("n0_100", 12_000)]
    assert missing_launchers(SHARED_HOST_START, reports) == 2


def test_missing_launchers_are_unknown_without_the_start():
    assert missing_launchers(None, [_failed("n0", 10_000)]) == 0
    assert missing_launchers(START, []) == 3
    assert CycleEndReportAggregator.missing_launchers(START, None) == 3
