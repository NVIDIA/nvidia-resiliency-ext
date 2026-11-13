# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvidia_resiliency_ext.fault_tolerance.cycle_info.aggregator import CycleOutcomeAggregator
from nvidia_resiliency_ext.fault_tolerance.cycle_info.records import (
    CycleInfo,
    EndReason,
    JobEndReason,
    NodeCycleEndReport,
)

# Node lists as they arrive over the wire: short host names, each once, sorted.
START = CycleInfo(
    job_id="job1",
    attempt_index=0,
    cycle_number=3,
    started_at=1.0,
    active_nodes=("n0", "n1", "n2"),
    active_ranks=(0, 1, 2),
)


def _report(node, end_reason, detected_timestamp):
    return NodeCycleEndReport("job1", 0, 3, node, end_reason, detected_timestamp)


def _failed(node, detected_timestamp):
    return _report(node, EndReason.FAILED, detected_timestamp)


def _peer(node, detected_timestamp):
    return _report(node, EndReason.TERMINATED_DUE_TO_PEER, detected_timestamp)


def _completed(node, detected_timestamp):
    return _report(node, EndReason.COMPLETED, detected_timestamp)


aggregate = CycleOutcomeAggregator().aggregate


def test_rank_failure_first_detected_by_the_failing_launcher():
    summary = aggregate(3, START, [_peer("n0", 12_000), _failed("n1", 10_000)])

    assert summary.end_reason is JobEndReason.RANK_FAILURE
    assert summary.first_detected == ("n1",)
    assert summary.ended_at == 10_000
    assert summary.missing == ("n2",)
    assert summary.launchers_expected == 3


def test_rank_monitor_stop_is_a_rank_failure():
    summary = aggregate(
        3,
        START,
        [
            _report("n0", EndReason.TERMINATED_BY_RANK_MONITOR, 10_000),
            _failed("n1", 11_000),  # a later, consequent failure
        ],
    )
    assert summary.end_reason is JobEndReason.RANK_FAILURE
    assert summary.first_detected == ("n0",)


def test_no_restart_decision_is_terminated():
    report = _report("n0", EndReason.TERMINATED_BY_ATTRIBUTION, 10_000)
    summary = aggregate(3, START, [report, _peer("n1", 11_000)])

    assert summary.end_reason is JobEndReason.TERMINATED
    assert summary.first_detected == ("n0",)


def test_job_reason_follows_the_first_detector():
    summary = aggregate(
        3,
        START,
        [_failed("n0", 10_000), _report("n1", EndReason.TERMINATED_BY_ATTRIBUTION, 11_000)],
    )
    assert summary.end_reason is JobEndReason.RANK_FAILURE


def test_completed_cycle_ends_when_the_last_launcher_completed():
    summary = aggregate(
        3, START, [_completed(n, 10_000 + i) for i, n in enumerate(START.active_nodes)]
    )
    assert summary.end_reason is JobEndReason.COMPLETED
    assert summary.ended_at == 10_002
    assert summary.first_detected == ()
    assert summary.missing == ()


def test_only_peer_restarts_leave_the_reason_unknown():
    summary = aggregate(3, START, [_peer("n0", 12_000), _peer("n1", 11_000)])
    assert summary.end_reason is JobEndReason.UNKNOWN
    assert summary.first_detected == ()
    assert summary.ended_at == 11_000  # the earliest detection of any launcher


def test_unspecified_end_is_unknown():
    summary = aggregate(3, START, [_report("n0", EndReason.UNSPECIFIED, 10_000)])
    assert summary.end_reason is JobEndReason.UNKNOWN


def test_duplicate_reports_keep_the_first():
    first, second = _failed("n0", 10_000), _failed("n0", 9_000)
    summary = aggregate(3, START, [first, second])
    assert summary.outcomes == (first,)


def test_launchers_sharing_a_host_count_as_one():
    """Known limitation: end reports are told apart by node name only."""
    summary = aggregate(3, START, [_peer("n0", 12_000), _peer("n0", 12_000)])
    assert len(summary.outcomes) == 1


def test_missing_nodes_are_matched_by_short_host_name():
    summary = aggregate(3, START, [_failed("n0.cluster", 10_000), _peer("n2.cluster", 11_000)])
    assert summary.missing == ("n1",)


def test_expected_launchers_fall_back_to_the_nodes_without_ranks():
    start = CycleInfo("job1", 0, 3, 1.0, active_nodes=("n0", "n1"))
    assert aggregate(3, start, []).launchers_expected == 2


def test_missing_and_expected_are_unknown_without_the_start():
    summary = aggregate(3, None, [_failed("n0", 10_000)])
    assert (summary.missing, summary.launchers_expected) == ((), None)


def test_no_reports():
    summary = aggregate(3, START, [])
    assert summary.end_reason is JobEndReason.UNKNOWN
    assert summary.ended_at is None
    assert summary.missing == START.active_nodes


def test_ties_in_detection_time_are_all_first():
    summary = aggregate(3, START, [_failed("n0", 10_000), _failed("n1", 10_000)])
    assert summary.first_detected == ("n0", "n1")
    assert summary.ended_at == 10_000
