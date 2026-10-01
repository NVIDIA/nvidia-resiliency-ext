"""Regression: the unknown-progress guard must not scale with history depth."""

from nvidia_resiliency_ext.attribution.restart_agent.l3.history import (
    _consecutive_unknown_progress_attempts,
    _has_progress_signal,
)
from nvidia_resiliency_ext.attribution.restart_agent.models import (
    AttemptFailureFacts,
    AttemptFailureFactsSource,
    AttemptProgressSummary,
    AttemptRecord,
)


def _progress(step=None, *, unknown=False):
    if unknown:
        return AttemptProgressSummary()  # both dimensions default to "unknown"
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


def _record(cycle_id, step=None, *, unknown=False):
    return AttemptRecord(
        job_id="j",
        cycle_id=cycle_id,
        progress=_progress(step, unknown=unknown),
        deterministic=_facts(),
    )


def test_signal_present_when_either_dimension_is_known():
    assert _has_progress_signal(_progress(10)) is True
    assert _has_progress_signal(_progress(unknown=True)) is False


def test_advancing_attempt_contributes_nothing():
    current = _record(4, step=500)
    priors = [_record(i, step=100 * i) for i in range(4)]
    assert _consecutive_unknown_progress_attempts(current, priors) == 0


def test_one_unverifiable_attempt_counts_once_regardless_of_history_depth():
    # The bug: every comparison went UNKNOWN because the *current* attempt had no
    # signal, so the count equalled the number of priors and a job exhausted a
    # budget of 3 on its first unverifiable cycle.
    current = _record(4, unknown=True)
    priors = [_record(i, step=100 * i) for i in range(4)]
    assert _consecutive_unknown_progress_attempts(current, priors) == 1


def test_streak_grows_only_with_consecutive_unverifiable_attempts():
    current = _record(4, unknown=True)
    priors = [
        _record(0, step=0),
        _record(1, step=100),
        _record(2, unknown=True),
        _record(3, unknown=True),
    ]
    assert _consecutive_unknown_progress_attempts(current, priors) == 3


def test_streak_stops_at_the_most_recent_attempt_with_a_signal():
    current = _record(4, unknown=True)
    priors = [
        _record(0, unknown=True),
        _record(1, unknown=True),
        _record(2, unknown=True),
        _record(3, step=300),
    ]
    assert _consecutive_unknown_progress_attempts(current, priors) == 1


def test_no_priors_counts_only_the_current_attempt():
    assert _consecutive_unknown_progress_attempts(_record(0, unknown=True), []) == 1
    assert _consecutive_unknown_progress_attempts(_record(0, step=1), []) == 0


# ─── through the real evaluation path, so the old behaviour is exercised ───


def _prior_view(records):
    from nvidia_resiliency_ext.attribution.restart_agent.models import PriorAttemptView

    return PriorAttemptView(available=True, availability_reason="ready", records=tuple(records))


def test_evaluate_job_progress_does_not_scale_the_guard_with_history_depth():
    from nvidia_resiliency_ext.attribution.restart_agent.l3.history import _evaluate_job_progress

    # Four advancing attempts, then one whose progress cannot be read. Before the
    # fix this reported 4 consecutive unknown-progress attempts and exhausted a
    # budget of 3 on the first unverifiable cycle.
    priors = [_record(i, step=100 * i) for i in range(4)]
    history = _evaluate_job_progress(_record(4, unknown=True), _prior_view(priors))

    assert history.same_job_attempts == 4
    assert history.consecutive_unknown_progress_attempts == 1


def test_evaluate_job_progress_still_counts_a_real_streak():
    from nvidia_resiliency_ext.attribution.restart_agent.l3.history import _evaluate_job_progress

    priors = [_record(0, step=0), _record(1, unknown=True), _record(2, unknown=True)]
    history = _evaluate_job_progress(_record(3, unknown=True), _prior_view(priors))

    assert history.consecutive_unknown_progress_attempts == 3
