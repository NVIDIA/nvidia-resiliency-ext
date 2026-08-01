# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic retry-rule and concurrent retry-ledger policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..models import (
    AffectedEntity,
    AssessmentStatus,
    Decision,
    DecisionBasis,
    DeclaredRecoveryCapability,
    FailureDomain,
    FailureEvidence,
    HistoryMatchScope,
    HistorySummary,
    ModelRecoveryAssessment,
    RetryOutlookWithoutWorkloadChange,
    RetryPolicyConfig,
    RetryPolicyRule,
)

GENERAL_ROOT_CEILING_ID = "general_root_ceiling"
SELECTED_RULE_BUDGET_ID = "selected_rule_budget"


@dataclass(frozen=True)
class L4PolicyInput:
    primary: FailureEvidence | None
    history: HistorySummary
    current_affected_entity: AffectedEntity | None = None
    model_recovery_assessment: ModelRecoveryAssessment | None = None
    assessment_grounded: bool = False
    retry_policy: RetryPolicyConfig = RetryPolicyConfig()
    declared_recovery_capabilities: tuple[DeclaredRecoveryCapability, ...] = ()

    def __post_init__(self) -> None:
        if self.model_recovery_assessment is not None and not isinstance(
            self.model_recovery_assessment,
            ModelRecoveryAssessment,
        ):
            raise TypeError("L4 model_recovery_assessment must be typed")
        if self.current_affected_entity is not None and not isinstance(
            self.current_affected_entity,
            AffectedEntity,
        ):
            raise TypeError("L4 current_affected_entity must be typed")
        if not isinstance(self.retry_policy, RetryPolicyConfig):
            raise TypeError("L4 retry_policy must be typed")
        if any(
            not isinstance(capability, DeclaredRecoveryCapability)
            for capability in self.declared_recovery_capabilities
        ):
            raise TypeError("L4 declared_recovery_capabilities must be typed")


@dataclass(frozen=True)
class RetryLedgerEvaluation:
    ledger_id: str
    applicable: bool
    rule: str
    history_match_scope: str
    allowed_retries: int | None
    matching_prior_failures: int = 0
    observed_advance: bool = False
    exhausted: bool = False
    inapplicable_reason: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "ledger_id": self.ledger_id,
            "applicable": self.applicable,
            "rule": self.rule,
            "history_match_scope": self.history_match_scope,
            "allowed_retries": self.allowed_retries,
            "matching_prior_failures": self.matching_prior_failures,
            "observed_advance": self.observed_advance,
            "exhausted": self.exhausted,
            "inapplicable_reason": self.inapplicable_reason,
        }


@dataclass(frozen=True)
class RetryPolicyEvaluation:
    policy_version: str
    rule: str | None
    retry_budget_exhausted: bool
    exhausted_by: tuple[str, ...]
    general_root_ceiling: RetryLedgerEvaluation
    selected_rule_budget: RetryLedgerEvaluation | None
    decision: str
    decision_basis: str
    failure_domain: str | None = None
    failure_domain_status: str | None = None
    failure_domain_confidence: int | None = None
    retry_outlook_without_workload_change: str | None = None
    retry_outlook_status: str | None = None
    retry_outlook_confidence: int | None = None
    recovery_assessment_policy_grounded: bool = False
    current_evidence_qualified: bool = False
    current_affected_entity: Mapping[str, Any] | None = None
    match_requirements: Mapping[str, str] | None = None
    declared_recovery_capability_ids: tuple[str, ...] = ()
    applied_recovery_capability: Mapping[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "rule": self.rule,
            "decision": self.decision,
            "decision_basis": self.decision_basis,
            "retry_budget_exhausted": self.retry_budget_exhausted,
            "exhausted_by": list(self.exhausted_by),
            "general_root_ceiling": self.general_root_ceiling.to_payload(),
            "selected_rule_budget": (
                self.selected_rule_budget.to_payload()
                if self.selected_rule_budget is not None
                else None
            ),
            "failure_domain": self.failure_domain,
            "failure_domain_status": self.failure_domain_status,
            "failure_domain_confidence": self.failure_domain_confidence,
            "retry_outlook_without_workload_change": (self.retry_outlook_without_workload_change),
            "retry_outlook_status": self.retry_outlook_status,
            "retry_outlook_confidence": self.retry_outlook_confidence,
            "recovery_assessment_policy_grounded": (self.recovery_assessment_policy_grounded),
            "current_evidence_qualified": self.current_evidence_qualified,
            "current_affected_entity": (
                dict(self.current_affected_entity)
                if self.current_affected_entity is not None
                else None
            ),
            "match_requirements": dict(self.match_requirements or {}),
            "declared_recovery_capability_ids": list(self.declared_recovery_capability_ids),
            "applied_recovery_capability": (
                dict(self.applied_recovery_capability)
                if self.applied_recovery_capability is not None
                else None
            ),
        }


@dataclass(frozen=True)
class L4PolicyOutcome:
    primary: FailureEvidence | None
    retry_policy: RetryPolicyEvaluation


def evaluate_policy(policy_input: L4PolicyInput) -> L4PolicyOutcome:
    configured = policy_input.retry_policy
    assessment = policy_input.model_recovery_assessment
    domain = assessment.failure_domain.value.value if assessment is not None else None
    domain_status = assessment.failure_domain.status.value if assessment is not None else None
    domain_confidence = assessment.failure_domain.confidence if assessment is not None else None
    outlook = (
        assessment.retry_outlook_without_workload_change.value.value
        if assessment is not None
        else None
    )
    outlook_status = (
        assessment.retry_outlook_without_workload_change.status.value
        if assessment is not None
        else None
    )
    outlook_confidence = (
        assessment.retry_outlook_without_workload_change.confidence
        if assessment is not None
        else None
    )
    current_evidence_qualified = _current_evidence_qualifies_for_immediate_stop(
        primary=policy_input.primary,
        assessment_grounded=policy_input.assessment_grounded,
        domain=domain,
        domain_status=domain_status,
        outlook=outlook,
        outlook_status=outlook_status,
    )
    matching_capability = _matching_recovery_capability(
        policy_input.primary,
        policy_input.declared_recovery_capabilities,
    )
    applied_capability = (
        matching_capability
        if _capability_entity_matches(
            matching_capability,
            policy_input.current_affected_entity,
        )
        else None
    )
    rule = _select_rule(
        primary=policy_input.primary,
        current_affected_entity=policy_input.current_affected_entity,
        matching_capability=matching_capability,
        applied_capability=applied_capability,
        assessment_grounded=policy_input.assessment_grounded,
        outlook=outlook,
        outlook_status=outlook_status,
        current_evidence_qualified=current_evidence_qualified,
    )

    general_root_ceiling = _general_root_ceiling(
        primary=policy_input.primary,
        rule=rule,
        history=policy_input.history,
        allowed_retries=configured.general_retry_allowed_retries,
    )
    selected_rule_budget = _selected_rule_budget(
        rule=rule,
        applied_capability=applied_capability,
        history=policy_input.history,
        confirmation_retries=configured.confirmation_retry_allowed_retries,
        bounded_retries=configured.bounded_retry_allowed_retries,
        general_retries=configured.general_retry_allowed_retries,
    )
    exhausted_by = tuple(
        ledger.ledger_id
        for ledger in (general_root_ceiling, selected_rule_budget)
        if ledger is not None and ledger.exhausted
    )
    retry_budget_exhausted = bool(exhausted_by)
    observed_advance = any(
        ledger is not None and ledger.applicable and ledger.observed_advance
        for ledger in (general_root_ceiling, selected_rule_budget)
    )
    decision, basis = _decision(
        primary=policy_input.primary,
        rule=rule,
        retry_budget_exhausted=retry_budget_exhausted,
        observed_advance=observed_advance,
    )
    selected_scope = (
        selected_rule_budget.history_match_scope
        if selected_rule_budget is not None
        else HistoryMatchScope.ROOT_ONLY.value
    )
    return L4PolicyOutcome(
        primary=policy_input.primary,
        retry_policy=RetryPolicyEvaluation(
            policy_version=configured.policy_version,
            rule=rule,
            retry_budget_exhausted=retry_budget_exhausted,
            exhausted_by=exhausted_by,
            general_root_ceiling=general_root_ceiling,
            selected_rule_budget=selected_rule_budget,
            decision=decision,
            decision_basis=basis,
            failure_domain=domain,
            failure_domain_status=domain_status,
            failure_domain_confidence=domain_confidence,
            retry_outlook_without_workload_change=outlook,
            retry_outlook_status=outlook_status,
            retry_outlook_confidence=outlook_confidence,
            recovery_assessment_policy_grounded=policy_input.assessment_grounded,
            current_evidence_qualified=current_evidence_qualified,
            current_affected_entity=(
                policy_input.current_affected_entity.to_payload()
                if policy_input.current_affected_entity is not None
                else None
            ),
            match_requirements={
                "job_id": "exact",
                "root_fingerprint": ("exact" if policy_input.primary is not None else "missing"),
                "affected_entity": (
                    "exact"
                    if selected_scope == HistoryMatchScope.ROOT_AND_ENTITY.value
                    else "not_required"
                ),
                "progress": "no_observed_advance",
            },
            declared_recovery_capability_ids=tuple(
                capability.capability_id.value
                for capability in policy_input.declared_recovery_capabilities
            ),
            applied_recovery_capability=(
                applied_capability.to_payload() if applied_capability is not None else None
            ),
        ),
    )


def _current_evidence_qualifies_for_immediate_stop(
    *,
    primary: FailureEvidence | None,
    assessment_grounded: bool,
    domain: str | None,
    domain_status: str | None,
    outlook: str | None,
    outlook_status: str | None,
) -> bool:
    return bool(
        primary is not None
        and assessment_grounded
        and domain == FailureDomain.WORKLOAD.value
        and domain_status == AssessmentStatus.ESTABLISHED_BY_CURRENT_LOG.value
        and outlook == RetryOutlookWithoutWorkloadChange.CANNOT_RECOVER.value
        and outlook_status == AssessmentStatus.ESTABLISHED_BY_CURRENT_LOG.value
    )


def _select_rule(
    *,
    primary: FailureEvidence | None,
    current_affected_entity: AffectedEntity | None,
    matching_capability: DeclaredRecoveryCapability | None,
    applied_capability: DeclaredRecoveryCapability | None,
    assessment_grounded: bool,
    outlook: str | None,
    outlook_status: str | None,
    current_evidence_qualified: bool,
) -> str | None:
    if primary is None:
        return None
    if applied_capability is not None:
        return RetryPolicyRule.WORKLOAD_MANAGED_RECOVERY.value
    if matching_capability is not None:
        return RetryPolicyRule.GENERAL_RETRY.value
    if current_evidence_qualified:
        return RetryPolicyRule.WORKLOAD_UNRECOVERABLE.value
    if current_affected_entity is not None:
        return RetryPolicyRule.CONFIRMATION_RETRY.value
    if (
        assessment_grounded
        and outlook == RetryOutlookWithoutWorkloadChange.MAY_RECOVER.value
        and outlook_status
        in {
            AssessmentStatus.ESTABLISHED_BY_CURRENT_LOG.value,
            AssessmentStatus.SUPPORTED_BUT_UNCONFIRMED.value,
        }
    ):
        return RetryPolicyRule.BOUNDED_RETRY.value
    return RetryPolicyRule.GENERAL_RETRY.value


def _general_root_ceiling(
    *,
    primary: FailureEvidence | None,
    rule: str | None,
    history: HistorySummary,
    allowed_retries: int,
) -> RetryLedgerEvaluation:
    if primary is None:
        return _inapplicable_ledger(
            ledger_id=GENERAL_ROOT_CEILING_ID,
            rule=RetryPolicyRule.GENERAL_RETRY.value,
            scope=HistoryMatchScope.ROOT_ONLY,
            reason="missing_primary",
        )
    if rule == RetryPolicyRule.WORKLOAD_UNRECOVERABLE.value:
        return _inapplicable_ledger(
            ledger_id=GENERAL_ROOT_CEILING_ID,
            rule=RetryPolicyRule.GENERAL_RETRY.value,
            scope=HistoryMatchScope.ROOT_ONLY,
            reason="immediate_unrecoverable",
        )
    return _evaluate_ledger(
        ledger_id=GENERAL_ROOT_CEILING_ID,
        rule=RetryPolicyRule.GENERAL_RETRY.value,
        scope=HistoryMatchScope.ROOT_ONLY,
        allowed_retries=allowed_retries,
        history=history,
    )


def _selected_rule_budget(
    *,
    rule: str | None,
    applied_capability: DeclaredRecoveryCapability | None,
    history: HistorySummary,
    confirmation_retries: int,
    bounded_retries: int,
    general_retries: int,
) -> RetryLedgerEvaluation | None:
    if rule == RetryPolicyRule.WORKLOAD_MANAGED_RECOVERY.value:
        assert applied_capability is not None
        allowed_retries = applied_capability.allowed_retries
        scope = applied_capability.history_match_scope
    elif rule == RetryPolicyRule.CONFIRMATION_RETRY.value:
        allowed_retries = confirmation_retries
        scope = HistoryMatchScope.ROOT_AND_ENTITY
    elif rule == RetryPolicyRule.BOUNDED_RETRY.value:
        allowed_retries = bounded_retries
        scope = HistoryMatchScope.ROOT_ONLY
    else:
        return None
    if allowed_retries >= general_retries:
        return None
    return _evaluate_ledger(
        ledger_id=SELECTED_RULE_BUDGET_ID,
        rule=rule,
        scope=scope,
        allowed_retries=allowed_retries,
        history=history,
    )


def _evaluate_ledger(
    *,
    ledger_id: str,
    rule: str,
    scope: HistoryMatchScope,
    allowed_retries: int,
    history: HistorySummary,
) -> RetryLedgerEvaluation:
    if not history.available:
        return _inapplicable_ledger(
            ledger_id=ledger_id,
            rule=rule,
            scope=scope,
            reason="history_unavailable",
            allowed_retries=allowed_retries,
        )
    matching_prior_failures, observed_advance = _history_measurements(history, scope)
    return RetryLedgerEvaluation(
        ledger_id=ledger_id,
        applicable=True,
        rule=rule,
        history_match_scope=scope.value,
        allowed_retries=allowed_retries,
        matching_prior_failures=matching_prior_failures,
        observed_advance=observed_advance,
        exhausted=(not observed_advance and matching_prior_failures >= allowed_retries),
    )


def _inapplicable_ledger(
    *,
    ledger_id: str,
    rule: str,
    scope: HistoryMatchScope,
    reason: str,
    allowed_retries: int | None = None,
) -> RetryLedgerEvaluation:
    return RetryLedgerEvaluation(
        ledger_id=ledger_id,
        applicable=False,
        rule=rule,
        history_match_scope=scope.value,
        allowed_retries=allowed_retries,
        inapplicable_reason=reason,
    )


def _matching_recovery_capability(
    primary: FailureEvidence | None,
    capabilities: tuple[DeclaredRecoveryCapability, ...],
) -> DeclaredRecoveryCapability | None:
    if primary is None:
        return None
    primary_classifiers = {primary.registry_id, primary.failure_class}
    for capability in capabilities:
        if primary.recovery_behavior != capability.behavior.value:
            continue
        if primary_classifiers.intersection(capability.applies_to):
            return capability
    return None


def _capability_entity_matches(
    capability: DeclaredRecoveryCapability | None,
    affected_entity: AffectedEntity | None,
) -> bool:
    return bool(
        capability is not None
        and affected_entity is not None
        and affected_entity.kind == capability.required_entity_kind
    )


def _history_measurements(
    history: HistorySummary,
    scope: HistoryMatchScope,
) -> tuple[int, bool]:
    if scope == HistoryMatchScope.ROOT_AND_ENTITY:
        return (
            history.consecutive_same_root_and_entity_no_advance_attempts,
            history.advanced_beyond_all_same_entity_comparable_attempts,
        )
    return (
        history.consecutive_same_root_no_advance_attempts,
        history.advanced_beyond_all_comparable_attempts,
    )


def _decision(
    *,
    primary: FailureEvidence | None,
    rule: str | None,
    retry_budget_exhausted: bool,
    observed_advance: bool,
) -> tuple[str, str]:
    if primary is None:
        return Decision.RESTART.value, DecisionBasis.NO_PRIMARY_FAILURE.value
    if rule == RetryPolicyRule.WORKLOAD_UNRECOVERABLE.value:
        return Decision.STOP.value, DecisionBasis.WORKLOAD_UNRECOVERABLE.value
    if retry_budget_exhausted:
        return Decision.STOP.value, DecisionBasis.RETRY_BUDGET_EXHAUSTED.value
    if observed_advance:
        return Decision.RESTART.value, DecisionBasis.OBSERVED_ADVANCE.value
    if rule == RetryPolicyRule.WORKLOAD_MANAGED_RECOVERY.value:
        return (
            Decision.RESTART.value,
            DecisionBasis.WORKLOAD_MANAGED_RECOVERY_AVAILABLE.value,
        )
    if rule == RetryPolicyRule.BOUNDED_RETRY.value:
        return Decision.RESTART.value, DecisionBasis.RETRY_RECOVERY_AVAILABLE.value
    if rule == RetryPolicyRule.CONFIRMATION_RETRY.value:
        return Decision.RESTART.value, DecisionBasis.CONFIRMATION_RETRY_AVAILABLE.value
    return Decision.RESTART.value, DecisionBasis.GENERAL_RETRY_AVAILABLE.value
