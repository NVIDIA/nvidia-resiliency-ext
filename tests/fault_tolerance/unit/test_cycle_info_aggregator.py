# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvidia_resiliency_ext.fault_tolerance.cycle_info.aggregator import CycleOutcomeAggregator
from nvidia_resiliency_ext.fault_tolerance.cycle_info.records import (
    CycleEndReason,
    CycleStartInfo,
    JobEndReason,
    LocalCycleOutcome,
    RankEndReason,
    RankOutcome,
    TimeoutKind,
)

START = CycleStartInfo(3, 1.0, ("n0", "n1", "n2"), (), (0, 1, 2))


def _outcome(node, end, detected_at, rank_end=RankEndReason.STOPPED_BY_LAUNCHER, rank=0):
    kind = TimeoutKind.SECTION if rank_end is RankEndReason.STOPPED_BY_RMON else None
    return LocalCycleOutcome(
        3, node, end, detected_at, (RankOutcome(rank, 0, 100, rank_end, timeout_kind=kind),)
    )


def _failed(node, detected_at, rank_end=RankEndReason.FAILED, rank=0):
    return _outcome(node, CycleEndReason.FAILED, detected_at, rank_end, rank)


def _peer(node, detected_at, rank=0):
    return _outcome(node, CycleEndReason.PEER_RESTART, detected_at, rank=rank)


def _completed(node, detected_at, rank=0):
    return _outcome(node, CycleEndReason.COMPLETED, detected_at, RankEndReason.COMPLETED, rank)


aggregate = CycleOutcomeAggregator().aggregate


def test_rank_failure_first_detected_by_the_failing_launcher():
    summary = aggregate(3, START, [_peer("n0", 12.0, 0), _failed("n1", 10.0, rank=1)])

    assert summary.end_reason is JobEndReason.RANK_FAILURE
    assert summary.first_detected == ("n1",)
    assert summary.ended_at == 10.0
    assert summary.missing == ("n2",)
    assert summary.launchers_expected == 3


def test_rank_stopped_by_its_rank_monitor_is_a_rank_failure_with_the_timeout_in_its_record():
    summary = aggregate(
        3,
        START,
        [
            _failed("n0", 10.0, RankEndReason.STOPPED_BY_RMON),
            _failed("n1", 11.0, rank=1),  # a later, consequent failure
        ],
    )
    assert summary.end_reason is JobEndReason.RANK_FAILURE
    assert summary.first_detected == ("n0",)
    first_rank = summary.outcomes[0].ranks[0]
    assert (first_rank.end, first_rank.timeout_kind) == (
        RankEndReason.STOPPED_BY_RMON,
        TimeoutKind.SECTION,
    )


def test_no_restart_decision_after_a_failure_is_terminated():
    """The job-wide reason follows how the cycle ended for the first detector."""
    outcome = _outcome("n0", CycleEndReason.TERMINATED, 10.0, RankEndReason.FAILED)
    summary = aggregate(3, START, [outcome])
    assert summary.end_reason is JobEndReason.TERMINATED
    assert summary.outcomes[0].ranks[0].end is RankEndReason.FAILED


def test_completed_cycle_ends_when_the_last_launcher_completed():
    summary = aggregate(
        3, START, [_completed(n, 10.0 + i, i) for i, n in enumerate(START.active_nodes)]
    )
    assert summary.end_reason is JobEndReason.COMPLETED
    assert summary.ended_at == 12.0
    assert summary.first_detected == ()
    assert summary.missing == ()


def test_no_restart_decision_without_failures_is_terminated():
    outcome = LocalCycleOutcome(3, "n0", CycleEndReason.TERMINATED, 10.0, ())
    assert aggregate(3, START, [outcome]).end_reason is JobEndReason.TERMINATED


def test_only_peer_restarts_leave_the_reason_unknown():
    summary = aggregate(3, START, [_peer("n0", 12.0), _peer("n1", 11.0, rank=1)])
    assert summary.end_reason is JobEndReason.UNKNOWN
    assert summary.first_detected == ()
    assert summary.ended_at == 11.0  # the earliest detection of any launcher


def test_duplicate_outcomes_are_counted_once():
    outcome = _failed("n0", 10.0)
    summary = aggregate(3, START, [outcome, outcome])
    assert summary.outcomes == (outcome,)


def test_launchers_sharing_a_host_are_told_apart_by_their_ranks():
    start = CycleStartInfo(3, 1.0, ("host", "host"), (), (0, 1))
    summary = aggregate(3, start, [_peer("host", 12.0, rank=0), _peer("host", 12.0, rank=1)])
    assert len(summary.outcomes) == 2
    assert summary.missing == ()


def test_missing_and_expected_are_unknown_without_the_start():
    summary = aggregate(3, None, [_failed("n0", 10.0)])
    assert (summary.missing, summary.launchers_expected) == ((), None)


def test_no_outcomes():
    summary = aggregate(3, START, [])
    assert summary.end_reason is JobEndReason.UNKNOWN
    assert summary.ended_at is None
    assert summary.missing == START.active_nodes


def test_ties_in_detection_time_are_all_first():
    summary = aggregate(3, START, [_failed("n0", 10.0), _failed("n1", 10.0, rank=1)])
    assert summary.first_detected == ("n0", "n1")
