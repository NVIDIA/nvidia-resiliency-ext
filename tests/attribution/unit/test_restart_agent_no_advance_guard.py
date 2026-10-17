"""Regression: the no-progress guard must not scale with history depth.

Job 4234179 (a rollback run) reached STOP on its fourth cycle with
``consecutive_no_advance_attempts=3`` against a budget of 3, while that same
job's cycle 1 had reported progress advancing. The count was measuring the
current attempt against every prior rather than a run of stalled attempts.
"""

from nvidia_resiliency_ext.attribution.restart_agent.l3.history import (
    _advanced_over,
    _consecutive_no_advance_attempts,
)
from nvidia_resiliency_ext.attribution.restart_agent.models import (
    AttemptFailureFacts,
    AttemptFailureFactsSource,
    AttemptProgressSummary,
    AttemptRecord,
)


def _progress(step):
    return AttemptProgressSummary(
        training_progress="observed",
        first_completed_step=step,
        last_completed_step=step,
        progress_marker_count=1,
    )


def _facts():
    return AttemptFailureFacts(
        source=AttemptFailureFactsSource.L0_DETERMINISTIC,
        root_fingerprint="observed:runtimeerror:test",
        root_fingerprint_source="test_fixture",
        fault_outcome="terminal",
        primary_line=1,
    )


def _record(cycle_id, step):
    return AttemptRecord(
        job_id="j",
        cycle_id=cycle_id,
        progress=_progress(step),
        deterministic=_facts(),
    )


def test_advanced_over_reads_its_immediate_predecessor():
    assert _advanced_over(_record(2, 300), _record(1, 200)) is True
    assert _advanced_over(_record(2, 200), _record(1, 200)) is False
    assert _advanced_over(_record(2, 100), _record(1, 200)) is False


def test_an_advancing_attempt_contributes_nothing():
    current = _record(4, 500)
    priors = [_record(index, 100 * index) for index in range(4)]
    assert _consecutive_no_advance_attempts(current, priors) == 0


def test_one_stalled_attempt_counts_once_regardless_of_history_depth():
    # The bug: every comparison is current-vs-a-prior, so one attempt that
    # fails to beat its own best turned them all SAME at once and the count
    # equalled the number of priors.
    priors = [_record(index, 100 * (index + 1)) for index in range(6)]
    current = _record(6, 600)  # same as the last prior, advanced over the rest
    assert _consecutive_no_advance_attempts(current, priors) == 1


def test_a_rollback_is_not_counted_as_a_stalled_run():
    # 4234179's shape: steady progress, then a resume from an earlier
    # checkpoint. The current attempt sits below several predecessors by
    # design, which must not read as that many stalled attempts.
    priors = [_record(0, 100), _record(1, 200), _record(2, 300)]
    current = _record(3, 150)  # rolled back below cycles 1 and 2
    assert _consecutive_no_advance_attempts(current, priors) == 1


def test_a_genuine_stalled_run_is_still_counted():
    priors = [_record(0, 100), _record(1, 200), _record(2, 200), _record(3, 200)]
    current = _record(4, 200)
    # cycles 2, 3 and the current attempt each failed to beat their predecessor.
    assert _consecutive_no_advance_attempts(current, priors) == 3


def test_the_run_stops_at_the_last_attempt_that_advanced():
    priors = [_record(0, 100), _record(1, 100), _record(2, 300), _record(3, 300)]
    current = _record(4, 300)
    # cycle 2 advanced, so only cycle 3 and the current attempt count.
    assert _consecutive_no_advance_attempts(current, priors) == 2


def test_no_priors_means_nothing_to_stall_against():
    assert _consecutive_no_advance_attempts(_record(0, 100), []) == 0


def test_a_budget_of_three_survives_a_deep_history_with_one_stall():
    # The live failure: a job deep enough to have 3+ priors must not exhaust a
    # budget of 3 the first time a single cycle fails to advance.
    priors = [_record(index, 100 * (index + 1)) for index in range(10)]
    current = _record(10, 1000)
    assert _consecutive_no_advance_attempts(current, priors) < 3
