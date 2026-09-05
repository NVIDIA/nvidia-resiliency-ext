# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the l1_category_confirmed_restart policy_context (two-way L1 category override).

Fires only when:
- The context is enabled in PolicyContextConfig.
- Primary is grounded.
- L1 category picker chose a RESTART-labeled taxonomy entry.
- Base rule concluded workload_unrecoverable.
- History shows no prior attempt with the same root_fingerprint.

Emits workload_confirmation_retry with history_match_scope=ROOT_ONLY. On
recurrence, the first-occurrence guard suppresses the override and base_rule
+ history exhaustion take over.
"""

from nvidia_resiliency_ext.attribution.restart_agent.l4 import L4PolicyInput, evaluate_policy
from nvidia_resiliency_ext.attribution.restart_agent.models import (
    AssessmentStatus,
    AttemptFailureFacts,
    AttemptFailureFactsSource,
    CudaOomNoRetryConfig,
    Decision,
    DecisionBasis,
    FailureClassifier,
    FailureDomain,
    FailureDomainAssessment,
    FailureEvidence,
    FaultOutcome,
    HistoryMatchScope,
    HistorySummary,
    L1CategoryConfirmedRestartConfig,
    ModelRecoveryAssessment,
    PolicyContextConfig,
    RetryOutlookAssessment,
    RetryOutlookWithoutWorkloadChange,
    RetryPolicyRule,
)

L1_CATEGORY_CONFIRMED_RESTART = "l1_category_confirmed_restart"

# Cat 14 in our taxonomy is a RESTART-labeled category
# ("Post-checkpoint progress-log assertion"). Cat 24 CPU OOM is also RESTART.
# Cat 25 CUDA OOM is STOP. We use these for the tests below.
RESTART_CATEGORY_ID = 14
STOP_CATEGORY_ID = 25


def _grad_inf_primary() -> FailureEvidence:
    """A workload_unrecoverable-shaped primary — the gemini b1-037 pattern."""
    return FailureEvidence(
        failure_class="assertion_error",
        signature="AssertionError: progress log missing entry",
        root_fingerprint="assertion:progress_log_missing",
        root_fingerprint_source="assertion",
        fault_outcome=FaultOutcome.TERMINAL.value,
        line=200,
        registry_id="assertion",
    )


def _grad_inf_facts() -> AttemptFailureFacts:
    return AttemptFailureFacts(
        source=AttemptFailureFactsSource.L2_GROUNDED,
        root_fingerprint="assertion:progress_log_missing",
        root_fingerprint_source="assertion",
        fault_outcome=FaultOutcome.TERMINAL.value,
        primary_line=200,
        classifiers=(),
    )


def _workload_unrecoverable_recovery() -> ModelRecoveryAssessment:
    """Assessment that forces base_rule to workload_unrecoverable via _immediate_stop_qualified."""
    return ModelRecoveryAssessment(
        failure_domain=FailureDomainAssessment(
            value=FailureDomain.WORKLOAD,
            status=AssessmentStatus.ESTABLISHED_BY_CURRENT_LOG,
            confidence=90,
        ),
        retry_outlook_without_workload_change=RetryOutlookAssessment(
            value=RetryOutlookWithoutWorkloadChange.CANNOT_RECOVER,
            status=AssessmentStatus.ESTABLISHED_BY_CURRENT_LOG,
            confidence=90,
        ),
        rationale="terminal workload failure",
    )


def _category_selection(category_id: int, category_confidence: int = 90) -> dict:
    return {
        "category_id": category_id,
        "category_confidence": category_confidence,
        "category_rationale": "test",
    }


def _enabled_policy_contexts() -> PolicyContextConfig:
    return PolicyContextConfig(
        l1_category_confirmed_restart=L1CategoryConfirmedRestartConfig(
            enabled=True, allowed_retries=1
        ),
    )


def _empty_history() -> HistorySummary:
    return HistorySummary(available=True, matching_root_attempts=0)


def _recurrence_history() -> HistorySummary:
    return HistorySummary(available=True, matching_root_attempts=1)


# ---------------------------------------------------------------------------
# Positive path: first occurrence with RESTART category flips STOP to RESTART
# ---------------------------------------------------------------------------


def test_restart_context_fires_on_first_occurrence_with_restart_category():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_grad_inf_primary(),
            current_failure_facts=_grad_inf_facts(),
            history=_empty_history(),
            model_recovery_assessment=_workload_unrecoverable_recovery(),
            policy_contexts=_enabled_policy_contexts(),
            l1_category_selection=_category_selection(RESTART_CATEGORY_ID),
        )
    ).retry_policy

    assert outcome.decision == Decision.RESTART.value
    assert outcome.effective_policy.policy_context_id == L1_CATEGORY_CONFIRMED_RESTART
    assert outcome.effective_policy.rule == RetryPolicyRule.WORKLOAD_CONFIRMATION_RETRY.value
    assert outcome.effective_policy.history_match_scope == HistoryMatchScope.ROOT_ONLY.value
    assert outcome.effective_policy.allowed_retries == 1


# ---------------------------------------------------------------------------
# Negative paths (context does not fire)
# ---------------------------------------------------------------------------


def test_restart_context_no_op_when_disabled():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_grad_inf_primary(),
            current_failure_facts=_grad_inf_facts(),
            history=_empty_history(),
            model_recovery_assessment=_workload_unrecoverable_recovery(),
            policy_contexts=PolicyContextConfig(),  # disabled by default
            l1_category_selection=_category_selection(RESTART_CATEGORY_ID),
        )
    ).retry_policy

    assert outcome.decision == Decision.STOP.value
    assert outcome.decision_basis == DecisionBasis.WORKLOAD_UNRECOVERABLE.value
    assert outcome.effective_policy.policy_context_id is None


def test_restart_context_no_op_on_recurrence():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_grad_inf_primary(),
            current_failure_facts=_grad_inf_facts(),
            history=_recurrence_history(),
            model_recovery_assessment=_workload_unrecoverable_recovery(),
            policy_contexts=_enabled_policy_contexts(),
            l1_category_selection=_category_selection(RESTART_CATEGORY_ID),
        )
    ).retry_policy

    assert outcome.decision == Decision.STOP.value
    assert outcome.decision_basis == DecisionBasis.WORKLOAD_UNRECOVERABLE.value
    assert outcome.effective_policy.policy_context_id is None


def test_restart_context_no_op_for_stop_category():
    """Cat 25 CUDA OOM is a STOP category. The RESTART context must not touch it."""
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_grad_inf_primary(),
            current_failure_facts=_grad_inf_facts(),
            history=_empty_history(),
            model_recovery_assessment=_workload_unrecoverable_recovery(),
            policy_contexts=_enabled_policy_contexts(),
            l1_category_selection=_category_selection(STOP_CATEGORY_ID),
        )
    ).retry_policy

    # STOP-category context takes precedence and fires with allowed_retries=0.
    assert outcome.decision == Decision.STOP.value


def test_restart_context_no_op_when_base_rule_is_not_workload_unrecoverable():
    """If base_rule is already a RESTART rule, the override should not fire."""
    # Weaker assessment: workload/supported_but_unconfirmed produces
    # workload_confirmation_retry, not workload_unrecoverable.
    weak = ModelRecoveryAssessment(
        failure_domain=FailureDomainAssessment(
            value=FailureDomain.WORKLOAD,
            status=AssessmentStatus.SUPPORTED_BUT_UNCONFIRMED,
            confidence=50,
        ),
        retry_outlook_without_workload_change=RetryOutlookAssessment(
            value=RetryOutlookWithoutWorkloadChange.MAY_RECOVER,
            status=AssessmentStatus.SUPPORTED_BUT_UNCONFIRMED,
            confidence=50,
        ),
        rationale="uncertain workload failure",
    )
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_grad_inf_primary(),
            current_failure_facts=_grad_inf_facts(),
            history=_empty_history(),
            model_recovery_assessment=weak,
            policy_contexts=_enabled_policy_contexts(),
            l1_category_selection=_category_selection(RESTART_CATEGORY_ID),
        )
    ).retry_policy

    assert outcome.decision == Decision.RESTART.value
    # Base rule handled it; our RESTART context did not.
    assert outcome.effective_policy.policy_context_id is None


def test_restart_context_no_op_when_category_id_missing():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_grad_inf_primary(),
            current_failure_facts=_grad_inf_facts(),
            history=_empty_history(),
            model_recovery_assessment=_workload_unrecoverable_recovery(),
            policy_contexts=_enabled_policy_contexts(),
            l1_category_selection={"category_id": 0, "category_confidence": 0},
        )
    ).retry_policy

    assert outcome.decision == Decision.STOP.value
    assert outcome.effective_policy.policy_context_id is None


def test_restart_context_yields_precedence_to_cuda_oom_stop_context():
    """cuda_oom_no_retry fires first; our RESTART context never gets a chance."""
    cuda_oom_facts = AttemptFailureFacts(
        source=AttemptFailureFactsSource.L0_DETERMINISTIC,
        root_fingerprint="cuda_oom:allocation_failure",
        root_fingerprint_source="cuda_oom",
        fault_outcome=FaultOutcome.TERMINAL.value,
        primary_line=100,
        classifiers=(FailureClassifier.CUDA_OOM.value,),
    )
    cuda_oom_primary = FailureEvidence(
        failure_class="cuda_oom",
        signature="CUDA error: out of memory",
        root_fingerprint="cuda_oom:allocation_failure",
        root_fingerprint_source="cuda_oom",
        fault_outcome=FaultOutcome.TERMINAL.value,
        line=100,
        registry_id="cuda_oom",
    )
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=cuda_oom_primary,
            current_failure_facts=cuda_oom_facts,
            history=_empty_history(),
            model_recovery_assessment=_workload_unrecoverable_recovery(),
            policy_contexts=PolicyContextConfig(
                cuda_oom_no_retry=CudaOomNoRetryConfig(enabled=True),
                l1_category_confirmed_restart=L1CategoryConfirmedRestartConfig(
                    enabled=True, allowed_retries=1
                ),
            ),
            l1_category_selection=_category_selection(RESTART_CATEGORY_ID),
        )
    ).retry_policy

    # cuda_oom_no_retry wins by classifier-context precedence.
    assert outcome.decision == Decision.STOP.value
    assert outcome.effective_policy.policy_context_id == "cuda_oom_no_retry"
