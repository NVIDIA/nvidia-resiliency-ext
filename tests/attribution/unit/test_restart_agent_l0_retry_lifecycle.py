# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L0 operation-retry lifecycle classification and primary-selection behavior."""

from nvidia_resiliency_ext.attribution.restart_agent.current_failure_facts import (
    build_attempt_failure_facts,
)
from nvidia_resiliency_ext.attribution.restart_agent.l0 import build_l0_bundle
from nvidia_resiliency_ext.attribution.restart_agent.l0.codec import read_l0_bundle, write_l0_bundle
from nvidia_resiliency_ext.attribution.restart_agent.l0.decision import build_decision_evidence
from nvidia_resiliency_ext.attribution.restart_agent.l0.retry_lifecycle import (
    classify_retry_lifecycle,
)
from nvidia_resiliency_ext.attribution.restart_agent.models import (
    AttemptFailureFactsSource,
    FaultOutcome,
    RetryLifecycleState,
)


def _pending_read_warning() -> str:
    return (
        "88: WARNING:dataset.reader:Attempt 1/4 to read data item failed with "
        'exception "[Errno 2] No such file or directory: \'/data/shard.bin\'"; '
        "going to sleep for 10 seconds and then re-try..."
    )


def test_retry_lifecycle_parses_pending_numbered_attempt():
    lifecycle = classify_retry_lifecycle(_pending_read_warning())

    assert lifecycle is not None
    assert lifecycle.state == RetryLifecycleState.PENDING
    assert lifecycle.attempt == 1
    assert lifecycle.max_attempts == 4


def test_retry_lifecycle_parses_final_attempt_as_exhausted():
    lifecycle = classify_retry_lifecycle(
        "Attempt 4/4 to read data item failed with FileNotFoundError"
    )

    assert lifecycle is not None
    assert lifecycle.state == RetryLifecycleState.EXHAUSTED
    assert lifecycle.attempt == 4
    assert lifecycle.max_attempts == 4


def test_retry_lifecycle_parses_explicit_success_and_exhaustion():
    succeeded = classify_retry_lifecycle("Data read successfully retried")
    exhausted = classify_retry_lifecycle("All retries failed; giving up")

    assert succeeded is not None
    assert succeeded.state == RetryLifecycleState.SUCCEEDED
    assert exhausted is not None
    assert exhausted.state == RetryLifecycleState.EXHAUSTED


def test_retry_lifecycle_requires_observed_transition_not_hypothetical_retry():
    assert classify_retry_lifecycle("A retry may recover this failure") is None


def test_l0_retains_pending_retry_but_does_not_select_it_as_primary(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "0: iteration 640001 completed\n"
        f"{_pending_read_warning()}\n"
        "0: destroy_process_group() called during shutdown\n",
        encoding="utf-8",
    )

    bundle = build_l0_bundle(log_path)

    retry_match = next(
        match
        for match in bundle.registry_matches
        if match.registry_id == "artifact_or_path_not_found"
    )
    assert retry_match.fault_outcome == FaultOutcome.RETRY_PENDING.value
    assert retry_match.retry_lifecycle is not None
    assert retry_match.retry_lifecycle.state == RetryLifecycleState.PENDING
    assert bundle.deterministic_primary_candidate is None
    assert bundle.selection_summary["primary_selection_basis"] == "not_available"
    decision_evidence = build_decision_evidence(bundle)
    current_facts = build_attempt_failure_facts(
        None,
        decision_evidence,
        source=AttemptFailureFactsSource.L0_DETERMINISTIC,
    )
    assert current_facts.root_fingerprint is None
    assert current_facts.history_identity_ready is False


def test_l0_pending_retry_does_not_displace_independent_terminal_failure(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "0: iteration 640001 completed\n"
        f"{_pending_read_warning()}\n"
        "7: RuntimeError: TCPStore connection reset by peer\n",
        encoding="utf-8",
    )

    bundle = build_l0_bundle(log_path)

    assert bundle.deterministic_primary_candidate is not None
    assert bundle.deterministic_primary_candidate.line == 3
    assert bundle.deterministic_primary_candidate.retry_lifecycle is None


def test_l0_explicitly_exhausted_retry_can_be_primary(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "88: Attempt 4/4 to read data item failed with FileNotFoundError: "
        "No such file or directory: '/data/shard.bin'; giving up\n",
        encoding="utf-8",
    )

    bundle = build_l0_bundle(log_path)

    primary = bundle.deterministic_primary_candidate
    assert primary is not None
    assert primary.line == 1
    assert primary.retry_lifecycle is not None
    assert primary.retry_lifecycle.state == RetryLifecycleState.EXHAUSTED
    assert primary.fault_outcome == FaultOutcome.TERMINAL.value


def test_l0_retry_lifecycle_round_trips_through_replay_artifact(tmp_path):
    log_path = tmp_path / "job.log"
    bundle_path = tmp_path / "l0_bundle.json"
    log_path.write_text(_pending_read_warning() + "\n", encoding="utf-8")
    bundle = build_l0_bundle(log_path)

    write_l0_bundle(bundle_path, bundle)
    replayed = read_l0_bundle(bundle_path, expected_log_path=str(log_path))

    retry_match = next(
        match
        for match in replayed.registry_matches
        if match.registry_id == "artifact_or_path_not_found"
    )
    assert retry_match.retry_lifecycle is not None
    assert retry_match.retry_lifecycle.state == RetryLifecycleState.PENDING
    assert retry_match.retry_lifecycle.attempt == 1
    assert retry_match.retry_lifecycle.max_attempts == 4
