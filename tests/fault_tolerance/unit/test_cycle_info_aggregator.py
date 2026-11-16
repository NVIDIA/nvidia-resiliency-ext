# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvidia_resiliency_ext.fault_tolerance.cycle_info.aggregator import CycleEndReportAggregator
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
        node_name=node,
        end_reason=end_reason,
        detected_timestamp=detected_timestamp,
    )


def _failed(node, detected_timestamp):
    return _report(node, pb.CycleEndReasonWorkerFailed, detected_timestamp)


def _peer(node, detected_timestamp):
    return _report(node, pb.CycleEndReasonWorkerTerminatedDueToPeer, detected_timestamp)


def _completed(node, detected_timestamp):
    return _report(node, pb.CycleEndReasonWorkerCompleted, detected_timestamp)


aggregate = CycleEndReportAggregator().aggregate


def test_rank_failure_first_detected_by_the_failing_launcher():
    end_report = aggregate([_peer("n0", 12_000), _failed("n1", 10_000)])

    assert end_report.end_reason == pb.CycleEndReasonWorkerFailed
    assert list(end_report.first_detected_nodes) == ["n1"]
    assert end_report.first_detected_timestamp == "1970-01-01T00:00:10.000Z"


def test_rank_monitor_stop_first_is_the_job_reason():
    end_report = aggregate(
        [
            _report("n0", pb.CycleEndReasonWorkerTerminatedByRankMonitor, 10_000),
            _failed("n1", 11_000),  # a later, consequent failure
        ]
    )
    assert end_report.end_reason == pb.CycleEndReasonWorkerTerminatedByRankMonitor
    assert list(end_report.first_detected_nodes) == ["n0"]


def test_no_restart_decision_first_is_the_job_reason():
    report = _report("n0", pb.CycleEndReasonWorkerTerminatedByAttribution, 10_000)
    end_report = aggregate([report, _peer("n1", 11_000)])

    assert end_report.end_reason == pb.CycleEndReasonWorkerTerminatedByAttribution
    assert list(end_report.first_detected_nodes) == ["n0"]


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
    assert list(end_report.first_detected_nodes) == []


def test_only_peer_restarts_leave_the_reason_unspecified():
    end_report = aggregate([_peer("n0", 12_000), _peer("n1", 11_000)])
    assert end_report.end_reason == pb.CycleEndReasonUnspecified
    assert list(end_report.first_detected_nodes) == []
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


def test_no_reports():
    end_report = aggregate([])
    assert end_report.end_reason == pb.CycleEndReasonUnspecified
    assert end_report.first_detected_timestamp == ""
    assert list(end_report.launcher_end_reports) == []


def test_ties_in_detection_time_are_all_first():
    end_report = aggregate([_failed("n0", 10_000), _failed("n1", 10_000)])
    assert list(end_report.first_detected_nodes) == ["n0", "n1"]
    assert end_report.first_detected_timestamp == "1970-01-01T00:00:10.000Z"


deduplicate = CycleEndReportAggregator.deduplicate


def test_duplicate_reports_keep_the_first():
    first, second = _failed("n0", 10_000), _failed("n0", 9_000)
    assert deduplicate([first, second]) == (first,)
    assert list(aggregate([first, second]).launcher_end_reports) == [first]


def test_launchers_sharing_a_host_count_as_one():
    """Known limitation: end reports are told apart by node name only."""
    assert len(deduplicate([_peer("n0", 12_000), _peer("n0", 12_000)])) == 1


missing_nodes = CycleEndReportAggregator.missing_nodes


def test_missing_nodes_are_the_active_nodes_without_a_report():
    assert missing_nodes(START, [_peer("n0", 12_000), _failed("n1", 10_000)]) == ("n2",)
    assert missing_nodes(START, []) == ("n0", "n1", "n2")


def test_missing_nodes_are_matched_by_short_host_name():
    reports = [_failed("n0.cluster", 10_000), _peer("n2.cluster", 11_000)]
    assert missing_nodes(START, reports) == ("n1",)


def test_missing_nodes_are_unknown_without_the_start():
    assert missing_nodes(None, [_failed("n0", 10_000)]) == ()
