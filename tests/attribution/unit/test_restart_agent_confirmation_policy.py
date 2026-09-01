# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L4 confirmation rules, job guards, and immutable restart-context tests."""

import pytest

from nvidia_resiliency_ext.attribution.restart_agent import RestartAgent, RestartAgentRuntime
from nvidia_resiliency_ext.attribution.restart_agent.config import parse_restart_agent_config
from nvidia_resiliency_ext.attribution.restart_agent.l4 import (
    L4CyclePolicyInput,
    L4PolicyInput,
    evaluate_cycle_policy,
    evaluate_policy,
)
from nvidia_resiliency_ext.attribution.restart_agent.models import (
    AffectedEntity,
    AffectedEntityKind,
    AttemptFailureFacts,
    AttemptFailureFactsSource,
    CudaOomNoRetryConfig,
    CycleHistoryComparison,
    Decision,
    DecisionBasis,
    FailureClassifier,
    FailureEvidence,
    FaultOutcome,
    HistoryMatchScope,
    HistoryProgressComparison,
    HistoryProgressRelation,
    HistorySummary,
    JobProgressHistory,
    ModelRecoveryAssessment,
    PolicyContextConfig,
    PortBindConfirmationRetryConfig,
    RejectedIterationRetryThenSkipConfig,
    RetryPolicyConfig,
    RetryPolicyRule,
    RouteHistorySummary,
)


def _artifact_entity(
    identity: str = "/checkpoints/model#checkpoint_iteration=42",
) -> AffectedEntity:
    return AffectedEntity(
        kind=AffectedEntityKind.ARTIFACT,
        identity=identity,
        fingerprint=f"affected_entity:artifact:{identity}",
        evidence_line=2,
    )


def _checkpoint_primary() -> FailureEvidence:
    return FailureEvidence(
        failure_class="checkpoint_decode",
        signature="UnicodeDecodeError",
        root_fingerprint="observed:unicode_decode:tensor_to_object",
        fault_outcome="terminal",
    )


def _rejected_iteration_facts(
    *,
    observer_ranks=("180",),
    unattributed_count=0,
) -> AttemptFailureFacts:
    return AttemptFailureFacts(
        source=AttemptFailureFactsSource.L0_DETERMINISTIC,
        root_fingerprint="observed:runtimeerror:unexpected_result_inf",
        root_fingerprint_source="observed_exception",
        fault_outcome="terminal",
        primary_line=100,
        failure_iteration=670314,
        classifiers=(FailureClassifier.REJECTED_NONFINITE_ITERATION.value,),
        root_observer_ranks=observer_ranks,
        unattributed_root_occurrence_count=unattributed_count,
    )


def _rejected_iteration_primary() -> FailureEvidence:
    return FailureEvidence(
        failure_class="observed_exception",
        signature="RuntimeError:",
        root_fingerprint="observed:runtimeerror:unexpected_result_inf",
        root_fingerprint_source="observed_exception",
        fault_outcome="terminal",
        line=100,
        failure_iteration=670314,
    )


def _cuda_oom_facts(*, fault_outcome=FaultOutcome.TERMINAL.value) -> AttemptFailureFacts:
    return AttemptFailureFacts(
        source=AttemptFailureFactsSource.L0_DETERMINISTIC,
        root_fingerprint="cuda_oom:allocation_failure",
        root_fingerprint_source="cuda_oom",
        fault_outcome=fault_outcome,
        primary_line=100,
        classifiers=(FailureClassifier.CUDA_OOM.value,),
    )


def _cuda_oom_primary(*, fault_outcome=FaultOutcome.TERMINAL.value) -> FailureEvidence:
    return FailureEvidence(
        failure_class="cuda_oom",
        signature="CUDA error: out of memory",
        root_fingerprint="cuda_oom:allocation_failure",
        root_fingerprint_source="cuda_oom",
        fault_outcome=fault_outcome,
        line=100,
        registry_id="cuda_oom",
    )


def _port_bind_facts(*, fault_outcome=FaultOutcome.TERMINAL.value) -> AttemptFailureFacts:
    return AttemptFailureFacts(
        source=AttemptFailureFactsSource.L0_DETERMINISTIC,
        root_fingerprint="observed:oserror:create_sockets:address_already_in_use",
        root_fingerprint_source="observed_exception",
        fault_outcome=fault_outcome,
        primary_line=100,
        classifiers=(FailureClassifier.PORT_BIND_CONFLICT.value,),
    )


def _port_bind_primary(*, fault_outcome=FaultOutcome.TERMINAL.value) -> FailureEvidence:
    return FailureEvidence(
        failure_class="observed_exception",
        signature="OSError:",
        root_fingerprint="observed:oserror:create_sockets:address_already_in_use",
        root_fingerprint_source="observed_exception",
        fault_outcome=fault_outcome,
        line=100,
    )


def test_cuda_oom_policy_context_stops_without_history():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_cuda_oom_primary(),
            current_failure_facts=_cuda_oom_facts(),
            history=HistorySummary(available=False),
        )
    ).retry_policy

    assert outcome.base_rule == RetryPolicyRule.GENERAL_RETRY.value
    assert outcome.effective_policy is not None
    assert outcome.effective_policy.rule == RetryPolicyRule.CUDA_OOM_NO_RETRY.value
    assert outcome.effective_policy.allowed_retries == 0
    assert outcome.selected_policy_ledger is None
    assert outcome.decision == Decision.STOP.value
    assert outcome.decision_basis == DecisionBasis.POLICY_CONTEXT_NO_RETRY.value
    assert outcome.applied_policy_context == {
        "policy_context_id": "cuda_oom_no_retry",
        "matched": True,
        "current_signature": {
            "classifiers": [FailureClassifier.CUDA_OOM.value],
            "failure_class": "cuda_oom",
            "fault_outcome": FaultOutcome.TERMINAL.value,
        },
        "retry_policy": outcome.effective_policy.to_payload(),
    }


@pytest.mark.parametrize(
    "fault_outcome",
    (
        FaultOutcome.RECOVERED.value,
        FaultOutcome.PROGRESSED_AFTER.value,
        FaultOutcome.RETRY_PENDING.value,
    ),
)
def test_cuda_oom_policy_context_excludes_nonterminal_oom(fault_outcome):
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_cuda_oom_primary(fault_outcome=fault_outcome),
            current_failure_facts=_cuda_oom_facts(fault_outcome=fault_outcome),
            history=HistorySummary(available=False),
        )
    ).retry_policy

    assert outcome.applied_policy_context is None
    assert outcome.effective_policy is not None
    assert outcome.effective_policy.rule == RetryPolicyRule.GENERAL_RETRY.value
    assert outcome.decision == Decision.RESTART.value


def test_retry_pending_primary_cannot_qualify_immediate_workload_stop():
    primary = FailureEvidence(
        failure_class="artifact_or_path_not_found",
        signature="FileNotFoundError",
        root_fingerprint="artifact_or_path_not_found:data_shard",
        fault_outcome=FaultOutcome.RETRY_PENDING.value,
    )
    assessment = ModelRecoveryAssessment.from_mapping(
        {
            "failure_domain": {
                "value": "workload",
                "status": "established_by_current_log",
                "confidence": 95,
            },
            "retry_outlook_without_workload_change": {
                "value": "cannot_recover",
                "status": "established_by_current_log",
                "confidence": 95,
            },
            "rationale": "The selected failure was classified as a workload defect.",
        }
    )

    outcome = evaluate_policy(
        L4PolicyInput(
            primary=primary,
            history=HistorySummary(available=False),
            model_recovery_assessment=assessment,
        )
    ).retry_policy

    assert outcome.current_evidence_qualified is False
    assert outcome.effective_policy is not None
    assert outcome.effective_policy.rule == RetryPolicyRule.WORKLOAD_CONFIRMATION_RETRY.value
    assert outcome.decision == Decision.RESTART.value


def test_cuda_oom_policy_context_accepts_l2_grounded_failure_class():
    facts = AttemptFailureFacts(
        source=AttemptFailureFactsSource.L2_GROUNDED,
        root_fingerprint="cuda_oom:allocation_failure",
        root_fingerprint_source="l0_registry",
        fault_outcome=FaultOutcome.TERMINAL.value,
        primary_line=101,
        classifiers=(),
    )

    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_cuda_oom_primary(),
            current_failure_facts=facts,
            history=HistorySummary(available=False),
        )
    ).retry_policy

    assert outcome.effective_policy is not None
    assert outcome.effective_policy.rule == RetryPolicyRule.CUDA_OOM_NO_RETRY.value
    assert outcome.decision == Decision.STOP.value


def test_cuda_oom_policy_context_can_be_disabled():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_cuda_oom_primary(),
            current_failure_facts=_cuda_oom_facts(),
            history=HistorySummary(available=False),
            policy_contexts=PolicyContextConfig(
                cuda_oom_no_retry=CudaOomNoRetryConfig(enabled=False)
            ),
        )
    ).retry_policy

    assert outcome.applied_policy_context is None
    assert outcome.effective_policy is not None
    assert outcome.effective_policy.rule == RetryPolicyRule.GENERAL_RETRY.value
    assert outcome.decision == Decision.RESTART.value


def test_cuda_oom_runtime_applies_no_retry_context(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            (
                "0: iteration 40/100 | consumed samples: 4096 |",
                "7: [rank7]: torch.AcceleratorError: CUDA error: out of memory",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    run = RestartAgentRuntime(RestartAgent()).analyze(
        {
            "schema_version": "restart_agent_request.v1",
            "log_path": str(log_path),
            "job_id": "job-oom",
            "cycle_id": 0,
        }
    )

    assert run.result.decision == Decision.STOP.value
    assert run.result.decision_basis == DecisionBasis.POLICY_CONTEXT_NO_RETRY.value
    assert run.result.retry_policy["effective_policy"]["rule"] == (
        RetryPolicyRule.CUDA_OOM_NO_RETRY.value
    )
    assert run.result.retry_policy["applied_policy_context"]["current_signature"] == {
        "classifiers": [FailureClassifier.CUDA_OOM.value],
        "failure_class": "cuda_oom",
        "fault_outcome": FaultOutcome.TERMINAL.value,
    }


def test_port_bind_policy_context_selects_one_confirmation_retry():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_port_bind_primary(),
            current_failure_facts=_port_bind_facts(),
            history=HistorySummary(available=False),
        )
    ).retry_policy

    assert outcome.base_rule == RetryPolicyRule.GENERAL_RETRY.value
    assert outcome.effective_policy is not None
    assert outcome.effective_policy.rule == RetryPolicyRule.PORT_BIND_CONFIRMATION_RETRY.value
    assert outcome.effective_policy.history_match_scope == HistoryMatchScope.ROOT_ONLY.value
    assert outcome.effective_policy.allowed_retries == 1
    assert outcome.decision == Decision.RESTART.value
    assert outcome.decision_basis == DecisionBasis.POLICY_CONTEXT_RETRY_AVAILABLE.value
    assert outcome.applied_policy_context == {
        "policy_context_id": "port_bind_confirmation_retry",
        "matched": True,
        "current_signature": {
            "classifiers": [FailureClassifier.PORT_BIND_CONFLICT.value],
            "failure_class": "observed_exception",
            "fault_outcome": FaultOutcome.TERMINAL.value,
        },
        "retry_policy": outcome.effective_policy.to_payload(),
    }


@pytest.mark.parametrize(
    "fault_outcome",
    (
        FaultOutcome.RECOVERED.value,
        FaultOutcome.PROGRESSED_AFTER.value,
        FaultOutcome.RETRY_PENDING.value,
    ),
)
def test_port_bind_policy_context_excludes_nonterminal_failure(fault_outcome):
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_port_bind_primary(fault_outcome=fault_outcome),
            current_failure_facts=_port_bind_facts(fault_outcome=fault_outcome),
            history=HistorySummary(available=False),
        )
    ).retry_policy

    assert outcome.applied_policy_context is None
    assert outcome.effective_policy is not None
    assert outcome.effective_policy.rule == RetryPolicyRule.GENERAL_RETRY.value


def test_port_bind_policy_context_can_be_disabled():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_port_bind_primary(),
            current_failure_facts=_port_bind_facts(),
            history=HistorySummary(available=False),
            policy_contexts=PolicyContextConfig(
                port_bind_confirmation_retry=PortBindConfirmationRetryConfig(enabled=False)
            ),
        )
    ).retry_policy

    assert outcome.applied_policy_context is None
    assert outcome.effective_policy is not None
    assert outcome.effective_policy.rule == RetryPolicyRule.GENERAL_RETRY.value


def test_port_bind_policy_runs_restart_stop_sequence(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            (
                "0: loading distributed checkpoint at iteration 600000",
                "0: successfully loaded checkpoint at iteration 600000",
                "0: OSError: [Errno 98] Address already in use",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    runtime = RestartAgentRuntime(RestartAgent())

    outcomes = [
        runtime.analyze(
            {
                "schema_version": "restart_agent_request.v1",
                "log_path": str(log_path),
                "job_id": "job-port-bind",
                "cycle_id": cycle_id,
            }
        ).result
        for cycle_id in range(2)
    ]

    assert [item.decision for item in outcomes] == [
        Decision.RESTART.value,
        Decision.STOP.value,
    ]
    for result in outcomes:
        assert result.retry_policy["effective_policy"]["rule"] == (
            RetryPolicyRule.PORT_BIND_CONFIRMATION_RETRY.value
        )
    assert outcomes[0].retry_policy["selected_policy_ledger"]["matching_prior_attempts"] == 0
    assert outcomes[1].retry_policy["selected_policy_ledger"]["matching_prior_attempts"] == 1


def _signature_comparison(
    *,
    relation=HistoryProgressRelation.SAME.value,
    same_iteration=True,
    same_observer_count=True,
    same_unattributed_count=True,
) -> HistoryProgressComparison:
    return HistoryProgressComparison(
        prior_cycle_id=1,
        relation=relation,
        prior_fault_outcome="terminal",
        same_failure_iteration=same_iteration,
        same_root_observer_count=same_observer_count,
        same_unattributed_root_occurrence_count=same_unattributed_count,
    )


def test_rejected_iteration_selects_policy_context_without_history():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_rejected_iteration_primary(),
            current_failure_facts=_rejected_iteration_facts(),
            history=HistorySummary(available=False),
        )
    ).retry_policy

    assert outcome.base_rule == RetryPolicyRule.GENERAL_RETRY.value
    assert outcome.effective_policy is not None
    assert outcome.effective_policy.source == "policy_context"
    assert outcome.effective_policy.rule == (
        RetryPolicyRule.REJECTED_ITERATION_RETRY_THEN_SKIP.value
    )
    assert outcome.effective_policy.allowed_retries == 2
    assert outcome.applied_policy_context is not None
    assert outcome.selected_policy_ledger is not None
    assert outcome.selected_policy_ledger.applicable is False
    assert outcome.selected_policy_ledger.inapplicable_reason == "history_unavailable"
    assert outcome.decision == Decision.RESTART.value
    assert outcome.decision_basis == DecisionBasis.POLICY_CONTEXT_RETRY_AVAILABLE.value


def test_rejected_iteration_history_mismatch_does_not_demote_policy_context():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_rejected_iteration_primary(),
            current_failure_facts=_rejected_iteration_facts(),
            history=HistorySummary(
                available=True,
                comparisons=(_signature_comparison(same_iteration=False),),
                consecutive_same_root_no_advance_attempts=1,
            ),
        )
    ).retry_policy

    assert outcome.effective_policy is not None
    assert outcome.effective_policy.rule == (
        RetryPolicyRule.REJECTED_ITERATION_RETRY_THEN_SKIP.value
    )
    assert outcome.selected_policy_ledger is not None
    assert outcome.selected_policy_ledger.matching_prior_attempts == 0
    assert outcome.decision == Decision.RESTART.value


@pytest.mark.parametrize(
    ("matching_prior_attempts", "expected_decision", "expected_exhausted"),
    (
        (1, Decision.RESTART.value, False),
        (2, Decision.STOP.value, True),
    ),
)
def test_rejected_iteration_history_consumes_context_budget(
    matching_prior_attempts,
    expected_decision,
    expected_exhausted,
):
    comparisons = tuple(_signature_comparison() for _index in range(matching_prior_attempts))
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_rejected_iteration_primary(),
            current_failure_facts=_rejected_iteration_facts(),
            history=HistorySummary(
                available=True,
                comparisons=comparisons,
                consecutive_same_root_no_advance_attempts=matching_prior_attempts,
            ),
        )
    ).retry_policy

    assert outcome.selected_policy_ledger is not None
    assert outcome.selected_policy_ledger.matching_prior_attempts == (matching_prior_attempts)
    assert outcome.selected_policy_ledger.exhausted is expected_exhausted
    assert outcome.decision == expected_decision


def test_rejected_iteration_policy_context_requires_one_observer():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_rejected_iteration_primary(),
            current_failure_facts=_rejected_iteration_facts(observer_ranks=("180", "181")),
            history=HistorySummary(available=True),
        )
    ).retry_policy

    assert outcome.applied_policy_context is None
    assert outcome.effective_policy is not None
    assert outcome.effective_policy.rule == RetryPolicyRule.GENERAL_RETRY.value


def test_rejected_iteration_policy_context_can_be_disabled():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_rejected_iteration_primary(),
            current_failure_facts=_rejected_iteration_facts(),
            history=HistorySummary(available=True),
            policy_contexts=PolicyContextConfig(
                rejected_iteration_retry_then_skip=(
                    RejectedIterationRetryThenSkipConfig(enabled=False)
                )
            ),
        )
    ).retry_policy

    assert outcome.applied_policy_context is None
    assert outcome.effective_policy is not None
    assert outcome.effective_policy.rule == RetryPolicyRule.GENERAL_RETRY.value


def test_rejected_iteration_policy_context_is_configurable():
    config = parse_restart_agent_config(
        {
            "schema_version": "restart_agent_config.v1",
            "config_id": "rejected-iteration-policy",
            "config_version": 1,
            "enrichment": {"enabled": False},
            "policy_contexts": {
                "rejected_iteration_retry_then_skip": {
                    "enabled": True,
                    "allowed_retries": 4,
                }
            },
            "model_routes": [],
        },
        environ={},
    )

    context = config.policy_contexts.rejected_iteration_retry_then_skip
    assert context.enabled is True
    assert context.allowed_retries == 4
    assert config.effective_config["policy_contexts"] == {
        "cuda_oom_no_retry": {
            "enabled": True,
        },
        "port_bind_confirmation_retry": {
            "enabled": True,
            "allowed_retries": 1,
        },
        "rejected_iteration_retry_then_skip": {
            "enabled": True,
            "allowed_retries": 4,
        },
    }


def test_port_bind_policy_context_is_configurable():
    config = parse_restart_agent_config(
        {
            "schema_version": "restart_agent_config.v1",
            "config_id": "port-bind-policy",
            "config_version": 1,
            "enrichment": {"enabled": False},
            "policy_contexts": {
                "port_bind_confirmation_retry": {
                    "enabled": True,
                    "allowed_retries": 3,
                }
            },
            "model_routes": [],
        },
        environ={},
    )

    context = config.policy_contexts.port_bind_confirmation_retry
    assert context.enabled is True
    assert context.allowed_retries == 3


def test_rejected_iteration_policy_runs_restart_restart_stop_sequence(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            (
                "0: iteration 670310/794728 | consumed samples: 2059192320 |",
                (
                    "180: [rank180]: RuntimeError: Rank 180, iteration 670314: "
                    "Unexpected result inf (message='found Inf in local grad norm')"
                ),
            )
        ),
        encoding="utf-8",
    )
    runtime = RestartAgentRuntime(RestartAgent())

    outcomes = []
    for cycle_id in range(3):
        run = runtime.analyze(
            {
                "schema_version": "restart_agent_request.v1",
                "log_path": str(log_path),
                "job_id": "job-1",
                "cycle_id": cycle_id,
            }
        )
        outcomes.append(run.result)

    assert [item.decision for item in outcomes] == [
        Decision.RESTART.value,
        Decision.RESTART.value,
        Decision.STOP.value,
    ]
    for result in outcomes:
        assert result.retry_policy["effective_policy"]["rule"] == (
            RetryPolicyRule.REJECTED_ITERATION_RETRY_THEN_SKIP.value
        )
    assert outcomes[0].retry_policy["selected_policy_ledger"]["matching_prior_attempts"] == 0
    assert outcomes[1].retry_policy["selected_policy_ledger"]["matching_prior_attempts"] == 1
    assert outcomes[2].retry_policy["selected_policy_ledger"]["matching_prior_attempts"] == 2


def test_exact_artifact_selects_concrete_confirmation_retry():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_checkpoint_primary(),
            history=HistorySummary(available=True),
            current_affected_entity=_artifact_entity(),
        )
    ).retry_policy

    assert outcome.base_rule == RetryPolicyRule.CONCRETE_CONFIRMATION_RETRY.value
    assert outcome.decision == Decision.RESTART.value
    assert outcome.decision_basis == (DecisionBasis.CONCRETE_CONFIRMATION_RETRY_AVAILABLE.value)
    assert outcome.selected_policy_ledger is not None
    assert outcome.selected_policy_ledger.allowed_retries == 1
    assert outcome.selected_policy_ledger.history_match_scope == (
        HistoryMatchScope.ROOT_AND_ENTITY.value
    )


def test_l4_prefers_route_primary_and_uses_only_primary_track_history():
    facts = AttemptFailureFacts(
        source=AttemptFailureFactsSource.L2_GROUNDED,
        root_fingerprint="route-primary",
        root_fingerprint_source="test",
        fault_outcome=FaultOutcome.TERMINAL.value,
        primary_line=2,
    )
    outcome = evaluate_cycle_policy(
        L4CyclePolicyInput(
            deterministic_primary=_checkpoint_primary(),
            deterministic_observation=None,
            deterministic_facts=_rejected_iteration_facts(),
            route_id="gpt",
            grounded_primary=_checkpoint_primary(),
            primary_facts=facts,
            history=CycleHistoryComparison(
                job_progress=JobProgressHistory(available=True),
                deterministic=HistorySummary(
                    available=True,
                    consecutive_same_root_no_advance_attempts=3,
                ),
                routes=(
                    RouteHistorySummary(
                        route_id="gpt",
                        primary=HistorySummary(available=True),
                    ),
                ),
            ),
        )
    )

    assert outcome.path_selection is not None
    assert outcome.path_selection.path == "primary"
    assert outcome.selected_history is not None
    assert outcome.selected_history.consecutive_same_root_no_advance_attempts == 0
    assert outcome.retry_policy.decision == Decision.RESTART.value


def test_l4_observation_fallback_does_not_inherit_primary_recovery_semantics():
    observation = FailureEvidence(
        failure_class="tcpstore_connection_loss",
        signature="connection reset by peer",
        root_fingerprint=None,
        observation_fingerprint="observation:tcpstore-loss",
        observation_fingerprint_source="test",
        fault_outcome=FaultOutcome.TERMINAL.value,
        line=3,
    )
    observation_facts = AttemptFailureFacts(
        source=AttemptFailureFactsSource.L2_GROUNDED,
        identity_kind="observation_only",
        root_fingerprint=None,
        root_fingerprint_source=None,
        observation_fingerprint="observation:tcpstore-loss",
        observation_fingerprint_source="test",
        fault_outcome=FaultOutcome.TERMINAL.value,
        selected_observation_line=3,
    )
    assessment = ModelRecoveryAssessment.from_mapping(
        {
            "failure_domain": {
                "value": "workload",
                "status": "established_by_current_log",
                "confidence": 95,
            },
            "retry_outlook_without_workload_change": {
                "value": "cannot_recover",
                "status": "established_by_current_log",
                "confidence": 95,
            },
            "rationale": "This assessment described the ungrounded primary.",
        }
    )
    outcome = evaluate_cycle_policy(
        L4CyclePolicyInput(
            deterministic_primary=None,
            deterministic_observation=None,
            deterministic_facts=AttemptFailureFacts(
                source=AttemptFailureFactsSource.L0_DETERMINISTIC,
                identity_kind="none",
                root_fingerprint=None,
                root_fingerprint_source=None,
                fault_outcome=None,
            ),
            route_id="gpt",
            grounded_observation=observation,
            observation_facts=observation_facts,
            model_recovery_assessment=assessment,
            l1_primary_declared=True,
            history=CycleHistoryComparison(
                job_progress=JobProgressHistory(available=True),
                deterministic=HistorySummary(available=False),
                routes=(
                    RouteHistorySummary(
                        route_id="gpt",
                        observation=HistorySummary(
                            available=False,
                            observation_history_available=True,
                        ),
                    ),
                ),
            ),
        )
    )

    assert outcome.path_selection is not None
    assert outcome.path_selection.path == "observation"
    assert outcome.retry_policy.failure_domain is None
    assert outcome.retry_policy.base_rule == RetryPolicyRule.GENERAL_RETRY.value
    assert outcome.retry_policy.decision == Decision.RESTART.value


def test_confirmation_retry_stops_on_first_exact_no_progress_recurrence():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_checkpoint_primary(),
            history=HistorySummary(
                available=True,
                consecutive_same_root_no_advance_attempts=1,
                consecutive_same_root_and_entity_no_advance_attempts=1,
            ),
            current_affected_entity=_artifact_entity(),
        )
    ).retry_policy

    assert outcome.base_rule == RetryPolicyRule.CONCRETE_CONFIRMATION_RETRY.value
    assert outcome.selected_policy_ledger is not None
    assert outcome.selected_policy_ledger.exhausted is True
    assert outcome.exhausted_by == ("selected_policy_ledger",)
    assert outcome.decision == Decision.STOP.value


def test_confirmation_retry_does_not_consume_on_different_artifact():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_checkpoint_primary(),
            history=HistorySummary(
                available=True,
                consecutive_same_root_no_advance_attempts=1,
                consecutive_same_root_and_entity_no_advance_attempts=0,
            ),
            current_affected_entity=_artifact_entity("/checkpoints/model#checkpoint_iteration=43"),
        )
    ).retry_policy

    assert outcome.selected_policy_ledger is not None
    assert outcome.selected_policy_ledger.exhausted is False
    assert outcome.decision == Decision.RESTART.value


def test_observed_advance_protects_confirmation_retry():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_checkpoint_primary(),
            history=HistorySummary(
                available=True,
                consecutive_same_root_and_entity_no_advance_attempts=1,
                advanced_beyond_all_same_entity_comparable_attempts=True,
            ),
            current_affected_entity=_artifact_entity(),
        )
    ).retry_policy

    assert outcome.selected_policy_ledger is not None
    assert outcome.selected_policy_ledger.exhausted is False
    assert outcome.decision == Decision.RESTART.value
    assert outcome.decision_basis == DecisionBasis.OBSERVED_ADVANCE.value


def test_grounded_workload_without_entity_uses_workload_confirmation():
    from nvidia_resiliency_ext.attribution.restart_agent.models import (
        AssessmentStatus,
        FailureDomain,
        FailureDomainAssessment,
        ModelRecoveryAssessment,
        RetryOutlookAssessment,
        RetryOutlookWithoutWorkloadChange,
    )

    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_checkpoint_primary(),
            history=HistorySummary(available=True),
            model_recovery_assessment=ModelRecoveryAssessment(
                failure_domain=FailureDomainAssessment(
                    value=FailureDomain.WORKLOAD,
                    status=AssessmentStatus.SUPPORTED_BUT_UNCONFIRMED,
                    confidence=80,
                ),
                retry_outlook_without_workload_change=RetryOutlookAssessment(
                    value=RetryOutlookWithoutWorkloadChange.MAY_RECOVER,
                    status=AssessmentStatus.SUPPORTED_BUT_UNCONFIRMED,
                    confidence=70,
                ),
                rationale="The workload owns the failing operation.",
            ),
        )
    ).retry_policy

    assert outcome.base_rule == RetryPolicyRule.WORKLOAD_CONFIRMATION_RETRY.value
    assert outcome.selected_policy_ledger is not None
    assert outcome.selected_policy_ledger.history_match_scope == (HistoryMatchScope.ROOT_ONLY.value)


def test_job_no_progress_guard_applies_without_primary():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=None,
            history=HistorySummary(
                available=False,
                availability_reason="missing_root_fingerprint",
                job_history_available=True,
                job_history_availability_reason="ready",
                consecutive_same_job_no_advance_attempts=3,
            ),
        )
    ).retry_policy

    assert outcome.base_rule is None
    assert outcome.decision == Decision.STOP.value
    assert outcome.decision_basis == (DecisionBasis.JOB_NO_PROGRESS_BUDGET_EXHAUSTED.value)


def test_unknown_progress_has_its_own_bounded_guard():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=None,
            history=HistorySummary(
                available=False,
                availability_reason="missing_root_fingerprint",
                job_history_available=True,
                job_history_availability_reason="ready",
                consecutive_same_job_unknown_progress_attempts=3,
            ),
        )
    ).retry_policy

    assert outcome.decision == Decision.STOP.value
    assert outcome.decision_basis == (DecisionBasis.PROGRESS_UNVERIFIABLE_BUDGET_EXHAUSTED.value)


def test_configured_zero_retry_exhausts_policy_without_claiming_unrecoverable():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_checkpoint_primary(),
            history=HistorySummary(available=False),
            current_affected_entity=_artifact_entity(),
            retry_policy=RetryPolicyConfig(
                concrete_confirmation_retry_allowed_retries=0,
            ),
        )
    ).retry_policy

    assert outcome.base_rule == RetryPolicyRule.CONCRETE_CONFIRMATION_RETRY.value
    assert outcome.selected_policy_ledger is not None
    assert outcome.selected_policy_ledger.exhausted is True
    assert outcome.retry_budget_exhausted is True
    assert outcome.decision == Decision.STOP.value
    assert outcome.decision_basis == DecisionBasis.RETRY_BUDGET_EXHAUSTED.value


def test_cluster_execution_context_is_not_configurable():
    with pytest.raises(ValueError, match="unsupported fields: cluster_execution_context"):
        parse_restart_agent_config(
            {
                "schema_version": "restart_agent_config.v1",
                "config_id": "invalid-restart-guarantee-override",
                "config_version": 1,
                "enrichment": {"enabled": True},
                "cluster_execution_context": {
                    "allocation_model": "heterogeneous_node_pool",
                },
                "model_routes": [{"route_id": "model-a", "model": "provider/model-a"}],
            },
            environ={"LLM_API_KEY_FILE": "/tmp/test-key"},
        )


def test_removed_recovery_capability_contract_is_rejected():
    with pytest.raises(ValueError, match="unsupported fields: declared_recovery_capabilities"):
        parse_restart_agent_config(
            {
                "schema_version": "restart_agent_config.v1",
                "config_id": "removed-capability-contract",
                "config_version": 1,
                "enrichment": {"enabled": False},
                "declared_recovery_capabilities": [],
                "model_routes": [],
            },
            environ={},
        )
