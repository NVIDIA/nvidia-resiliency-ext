# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declared workload recovery capability configuration and L4 behavior."""

import pytest

from nvidia_resiliency_ext.attribution.restart_agent import RestartAgent, RestartAgentRequest
from nvidia_resiliency_ext.attribution.restart_agent.config import parse_restart_agent_config
from nvidia_resiliency_ext.attribution.restart_agent.l4 import L4PolicyInput, evaluate_policy
from nvidia_resiliency_ext.attribution.restart_agent.models import (
    AffectedEntity,
    AffectedEntityKind,
    AssessmentStatus,
    Decision,
    DecisionBasis,
    DeclaredRecoveryCapability,
    FailureDomain,
    FailureDomainAssessment,
    FailureEvidence,
    HistoryMatchScope,
    HistorySummary,
    ModelRecoveryAssessment,
    RecoveryBehavior,
    RecoveryCapabilityId,
    RetryOutlookAssessment,
    RetryOutlookWithoutWorkloadChange,
    RetryPolicyRule,
)


def _bad_token_capability(*, allowed_retries: int = 2) -> DeclaredRecoveryCapability:
    return DeclaredRecoveryCapability(
        capability_id=RecoveryCapabilityId.BAD_TOKEN_RETRY_THEN_SKIP,
        behavior=RecoveryBehavior.RETRY_THEN_SKIP,
        applies_to=("bad_token_or_window",),
        required_entity_kind=AffectedEntityKind.DATA_POSITION,
        history_match_scope=HistoryMatchScope.ROOT_AND_ENTITY,
        allowed_retries=allowed_retries,
    )


def _data_position_entity(identity: str = "token:42") -> AffectedEntity:
    return AffectedEntity(
        kind=AffectedEntityKind.DATA_POSITION,
        identity=identity,
        fingerprint=f"affected_entity:data_position:{identity}",
        evidence_line=2,
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


def _bad_token_primary() -> FailureEvidence:
    return FailureEvidence(
        failure_class="bad_token_or_window",
        signature="bad token detected",
        root_fingerprint="observed:data:bad_token",
        fault_outcome="terminal",
        registry_id="bad_token_or_window",
        recovery_behavior=RecoveryBehavior.RETRY_THEN_SKIP.value,
    )


def _established_unrecoverable_assessment() -> ModelRecoveryAssessment:
    return ModelRecoveryAssessment(
        failure_domain=FailureDomainAssessment(
            value=FailureDomain.WORKLOAD,
            status=AssessmentStatus.ESTABLISHED_BY_CURRENT_LOG,
            confidence=95,
        ),
        retry_outlook_without_workload_change=RetryOutlookAssessment(
            value=RetryOutlookWithoutWorkloadChange.CANNOT_RECOVER,
            status=AssessmentStatus.ESTABLISHED_BY_CURRENT_LOG,
            confidence=95,
        ),
        rationale="The unchanged workload cannot process this token.",
    )


def test_bad_token_capability_precedes_generic_workload_unrecoverable_rule():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_bad_token_primary(),
            history=HistorySummary(available=True),
            current_affected_entity=_data_position_entity(),
            model_recovery_assessment=_established_unrecoverable_assessment(),
            assessment_grounded=True,
            declared_recovery_capabilities=(_bad_token_capability(),),
        )
    ).retry_policy

    assert outcome.rule == RetryPolicyRule.WORKLOAD_MANAGED_RECOVERY.value
    assert outcome.general_root_ceiling.allowed_retries == 3
    assert outcome.selected_rule_budget is not None
    assert outcome.selected_rule_budget.allowed_retries == 2
    assert outcome.selected_rule_budget.history_match_scope == (
        HistoryMatchScope.ROOT_AND_ENTITY.value
    )
    assert outcome.decision == Decision.RESTART.value
    assert outcome.decision_basis == DecisionBasis.WORKLOAD_MANAGED_RECOVERY_AVAILABLE.value
    assert outcome.current_evidence_qualified is True
    assert outcome.declared_recovery_capability_ids == (
        RecoveryCapabilityId.BAD_TOKEN_RETRY_THEN_SKIP.value,
    )
    assert outcome.applied_recovery_capability == {
        "capability_id": RecoveryCapabilityId.BAD_TOKEN_RETRY_THEN_SKIP.value,
        "behavior": RecoveryBehavior.RETRY_THEN_SKIP.value,
        "applies_to": ["bad_token_or_window"],
        "required_entity_kind": AffectedEntityKind.DATA_POSITION.value,
        "history_match_scope": HistoryMatchScope.ROOT_AND_ENTITY.value,
        "allowed_retries": 2,
    }


def test_exact_artifact_selects_confirmation_retry():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=FailureEvidence(
                failure_class="checkpoint_decode",
                signature="UnicodeDecodeError",
                root_fingerprint="observed:unicode_decode:tensor_to_object",
                fault_outcome="terminal",
            ),
            history=HistorySummary(available=True),
            current_affected_entity=_artifact_entity(),
        )
    ).retry_policy

    assert outcome.rule == RetryPolicyRule.CONFIRMATION_RETRY.value
    assert outcome.decision == Decision.RESTART.value
    assert outcome.decision_basis == DecisionBasis.CONFIRMATION_RETRY_AVAILABLE.value
    assert outcome.selected_rule_budget is not None
    assert outcome.selected_rule_budget.allowed_retries == 1
    assert outcome.selected_rule_budget.history_match_scope == (
        HistoryMatchScope.ROOT_AND_ENTITY.value
    )


def test_confirmation_retry_stops_on_first_exact_no_progress_recurrence():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=FailureEvidence(
                failure_class="checkpoint_decode",
                signature="UnicodeDecodeError",
                root_fingerprint="observed:unicode_decode:tensor_to_object",
                fault_outcome="terminal",
            ),
            history=HistorySummary(
                available=True,
                consecutive_same_root_no_advance_attempts=1,
                consecutive_same_root_and_entity_no_advance_attempts=1,
            ),
            current_affected_entity=_artifact_entity(),
        )
    ).retry_policy

    assert outcome.rule == RetryPolicyRule.CONFIRMATION_RETRY.value
    assert outcome.general_root_ceiling.exhausted is False
    assert outcome.selected_rule_budget is not None
    assert outcome.selected_rule_budget.exhausted is True
    assert outcome.exhausted_by == ("selected_rule_budget",)
    assert outcome.decision == Decision.STOP.value


def test_confirmation_retry_does_not_consume_on_different_entity():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=FailureEvidence(
                failure_class="checkpoint_decode",
                signature="UnicodeDecodeError",
                root_fingerprint="observed:unicode_decode:tensor_to_object",
                fault_outcome="terminal",
            ),
            history=HistorySummary(
                available=True,
                consecutive_same_root_no_advance_attempts=1,
                consecutive_same_root_and_entity_no_advance_attempts=0,
            ),
            current_affected_entity=_artifact_entity("/checkpoints/model#checkpoint_iteration=43"),
        )
    ).retry_policy

    assert outcome.rule == RetryPolicyRule.CONFIRMATION_RETRY.value
    assert outcome.selected_rule_budget is not None
    assert outcome.selected_rule_budget.exhausted is False
    assert outcome.decision == Decision.RESTART.value


def test_observed_advance_protects_confirmation_retry():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=FailureEvidence(
                failure_class="checkpoint_decode",
                signature="UnicodeDecodeError",
                root_fingerprint="observed:unicode_decode:tensor_to_object",
                fault_outcome="terminal",
            ),
            history=HistorySummary(
                available=True,
                consecutive_same_root_and_entity_no_advance_attempts=1,
                advanced_beyond_all_same_entity_comparable_attempts=True,
            ),
            current_affected_entity=_artifact_entity(),
        )
    ).retry_policy

    assert outcome.selected_rule_budget is not None
    assert outcome.selected_rule_budget.exhausted is False
    assert outcome.decision == Decision.RESTART.value
    assert outcome.decision_basis == DecisionBasis.OBSERVED_ADVANCE.value


def test_bad_token_capability_stops_after_its_declared_retry_budget():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_bad_token_primary(),
            history=HistorySummary(
                available=True,
                consecutive_same_root_and_entity_no_advance_attempts=2,
            ),
            current_affected_entity=_data_position_entity(),
            declared_recovery_capabilities=(_bad_token_capability(),),
        )
    ).retry_policy

    assert outcome.rule == RetryPolicyRule.WORKLOAD_MANAGED_RECOVERY.value
    assert outcome.retry_budget_exhausted is True
    assert outcome.exhausted_by == ("selected_rule_budget",)
    assert outcome.decision == Decision.STOP.value
    assert outcome.decision_basis == DecisionBasis.RETRY_BUDGET_EXHAUSTED.value


def test_bad_token_capability_allows_first_matching_recurrence():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_bad_token_primary(),
            history=HistorySummary(
                available=True,
                consecutive_same_root_and_entity_no_advance_attempts=1,
            ),
            current_affected_entity=_data_position_entity(),
            declared_recovery_capabilities=(_bad_token_capability(),),
        )
    ).retry_policy

    assert outcome.rule == RetryPolicyRule.WORKLOAD_MANAGED_RECOVERY.value
    assert outcome.retry_budget_exhausted is False
    assert outcome.decision == Decision.RESTART.value


def test_observed_advance_protects_exhausted_bad_token_capability_budget():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_bad_token_primary(),
            history=HistorySummary(
                available=True,
                consecutive_same_root_and_entity_no_advance_attempts=2,
                advanced_beyond_all_same_entity_comparable_attempts=True,
            ),
            current_affected_entity=_data_position_entity(),
            declared_recovery_capabilities=(_bad_token_capability(),),
        )
    ).retry_policy

    assert outcome.rule == RetryPolicyRule.WORKLOAD_MANAGED_RECOVERY.value
    assert outcome.retry_budget_exhausted is False
    assert outcome.decision == Decision.RESTART.value
    assert outcome.decision_basis == DecisionBasis.OBSERVED_ADVANCE.value


def test_capability_does_not_apply_without_grounded_bad_token_behavior():
    primary = FailureEvidence(
        failure_class="bad_token_or_window",
        signature="numeric instability",
        root_fingerprint="observed:data:numeric_instability",
        fault_outcome="terminal",
        registry_id="model_selected",
        recovery_behavior=RecoveryBehavior.NONE.value,
    )

    outcome = evaluate_policy(
        L4PolicyInput(
            primary=primary,
            history=HistorySummary(available=False),
            declared_recovery_capabilities=(_bad_token_capability(),),
        )
    ).retry_policy

    assert outcome.rule == RetryPolicyRule.GENERAL_RETRY.value
    assert outcome.applied_recovery_capability is None


def test_bad_token_capability_requires_a_data_position_entity():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_bad_token_primary(),
            history=HistorySummary(
                available=True,
                consecutive_same_root_no_advance_attempts=3,
            ),
            declared_recovery_capabilities=(_bad_token_capability(),),
        )
    ).retry_policy

    assert outcome.rule == RetryPolicyRule.GENERAL_RETRY.value
    assert outcome.applied_recovery_capability is None
    assert outcome.general_root_ceiling.history_match_scope == (HistoryMatchScope.ROOT_ONLY.value)
    assert outcome.selected_rule_budget is None


def test_restart_agent_threads_declared_capability_to_l4(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "iteration 418 completed\nbad token detected token_id=42\n",
        encoding="utf-8",
    )

    run = RestartAgent(declared_recovery_capabilities=(_bad_token_capability(),)).run(
        RestartAgentRequest(log_path=str(log_path))
    )
    result = run.result

    assert result.decision == Decision.RESTART.value
    assert result.retry_policy["rule"] == RetryPolicyRule.WORKLOAD_MANAGED_RECOVERY.value
    assert result.retry_policy["applied_recovery_capability"]["capability_id"] == (
        RecoveryCapabilityId.BAD_TOKEN_RETRY_THEN_SKIP.value
    )
    assert result.retry_policy["selected_rule_budget"]["history_match_scope"] == (
        HistoryMatchScope.ROOT_AND_ENTITY.value
    )
    assert result.retry_policy["current_affected_entity"]["kind"] == (
        AffectedEntityKind.DATA_POSITION.value
    )
    assert result.retry_policy["current_affected_entity"]["identity"] == "token:42"
    assert run.trace["layers"]["L0"]["affected_entity"] == (
        result.retry_policy["current_affected_entity"]
    )
    assert run.trace["layers"]["L3"]["current_affected_entity"] == (
        result.retry_policy["current_affected_entity"]
    )
    assert run.trace["layers"]["L4"]["selected_rule_budget"]["history_match_scope"] == (
        HistoryMatchScope.ROOT_AND_ENTITY.value
    )


def test_general_root_ceiling_still_exhausts_when_entity_budget_does_not():
    outcome = evaluate_policy(
        L4PolicyInput(
            primary=_bad_token_primary(),
            history=HistorySummary(
                available=True,
                consecutive_same_root_no_advance_attempts=3,
                consecutive_same_root_and_entity_no_advance_attempts=0,
            ),
            current_affected_entity=_data_position_entity(),
            declared_recovery_capabilities=(_bad_token_capability(),),
        )
    ).retry_policy

    assert outcome.rule == RetryPolicyRule.WORKLOAD_MANAGED_RECOVERY.value
    assert outcome.general_root_ceiling.exhausted is True
    assert outcome.selected_rule_budget is not None
    assert outcome.selected_rule_budget.exhausted is False
    assert outcome.exhausted_by == ("general_root_ceiling",)
    assert outcome.decision == Decision.STOP.value


def test_restart_agent_config_declares_bad_token_retry_then_skip():
    config = parse_restart_agent_config(
        {
            "schema_version": "restart_agent_config.v1",
            "config_id": "bad-token-managed-recovery",
            "config_version": 1,
            "enrichment": {"enabled": True},
            "declared_recovery_capabilities": [
                {
                    "capability_id": "bad_token_retry_then_skip",
                    "behavior": "retry_then_skip",
                    "applies_to": ["bad_token_or_window"],
                    "required_entity_kind": "data_position",
                    "history_match_scope": "root_and_entity",
                    "allowed_retries": 2,
                }
            ],
            "model_routes": [{"route_id": "model-a", "model": "provider/model-a"}],
        },
        environ={"LLM_API_KEY_FILE": "/tmp/test-key"},
    )

    assert config.declared_recovery_capabilities == (_bad_token_capability(),)
    assert config.effective_config["declared_recovery_capabilities"] == [
        {
            "capability_id": "bad_token_retry_then_skip",
            "behavior": "retry_then_skip",
            "applies_to": ["bad_token_or_window"],
            "required_entity_kind": "data_position",
            "history_match_scope": "root_and_entity",
            "allowed_retries": 2,
        }
    ]


def test_restart_environment_guarantees_are_not_configurable():
    with pytest.raises(ValueError, match="unsupported fields: restart_environment_context"):
        parse_restart_agent_config(
            {
                "schema_version": "restart_agent_config.v1",
                "config_id": "invalid-restart-guarantee-override",
                "config_version": 1,
                "enrichment": {"enabled": True},
                "restart_environment_context": {
                    "hardware_allocation_may_change": False,
                },
                "model_routes": [{"route_id": "model-a", "model": "provider/model-a"}],
            },
            environ={"LLM_API_KEY_FILE": "/tmp/test-key"},
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("behavior", "fallback", "behavior is unsupported"),
        ("applies_to", ["numeric_instability"], "applies_to must be"),
        ("allowed_retries", 0, "allowed_retries must be greater than zero"),
    ],
)
def test_restart_agent_config_rejects_invalid_bad_token_capability(
    field,
    value,
    message,
):
    capability = {
        "capability_id": "bad_token_retry_then_skip",
        "behavior": "retry_then_skip",
        "applies_to": ["bad_token_or_window"],
        "required_entity_kind": "data_position",
        "history_match_scope": "root_and_entity",
        "allowed_retries": 2,
    }
    capability[field] = value

    with pytest.raises(ValueError, match=message):
        parse_restart_agent_config(
            {
                "schema_version": "restart_agent_config.v1",
                "config_id": "invalid-capability",
                "config_version": 1,
                "enrichment": {"enabled": True},
                "declared_recovery_capabilities": [capability],
                "model_routes": [{"route_id": "model-a", "model": "provider/model-a"}],
            },
            environ={"LLM_API_KEY_FILE": "/tmp/test-key"},
        )


def test_restart_agent_config_rejects_capability_budget_above_general_ceiling():
    with pytest.raises(
        ValueError,
        match="allowed_retries must not exceed.*general_retry_allowed_retries",
    ):
        parse_restart_agent_config(
            {
                "schema_version": "restart_agent_config.v1",
                "config_id": "invalid-capability-budget",
                "config_version": 1,
                "retry_policy": {
                    "bounded_retry_allowed_retries": 1,
                    "general_retry_allowed_retries": 1,
                },
                "enrichment": {"enabled": False},
                "declared_recovery_capabilities": [
                    {
                        "capability_id": "bad_token_retry_then_skip",
                        "behavior": "retry_then_skip",
                        "applies_to": ["bad_token_or_window"],
                        "required_entity_kind": "data_position",
                        "history_match_scope": "root_and_entity",
                        "allowed_retries": 2,
                    }
                ],
                "model_routes": [],
            },
            environ={},
        )
