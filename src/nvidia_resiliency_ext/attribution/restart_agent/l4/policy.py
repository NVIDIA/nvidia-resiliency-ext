# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic generic retry-budget policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..models import (
    AssessmentStatus,
    Decision,
    DecisionBasis,
    FailureDomain,
    FailureEvidence,
    HistorySummary,
    ModelRecoveryAssessment,
    RetryOutlookWithoutWorkloadChange,
    RetryPolicyConfig,
    RetryPolicyRule,
)

TIME_LIMIT_FINE_CLASS = "time_limit"


@dataclass(frozen=True)
class L4PolicyInput:
    primary: FailureEvidence | None
    history: HistorySummary
    model_recovery_assessment: ModelRecoveryAssessment | None = None
    assessment_grounded: bool = False
    retry_policy: RetryPolicyConfig = RetryPolicyConfig()

    def __post_init__(self) -> None:
        if self.model_recovery_assessment is not None and not isinstance(
            self.model_recovery_assessment,
            ModelRecoveryAssessment,
        ):
            raise TypeError("L4 model_recovery_assessment must be typed")
        if not isinstance(self.retry_policy, RetryPolicyConfig):
            raise TypeError("L4 retry_policy must be typed")


@dataclass(frozen=True)
class RetryPolicyEvaluation:
    policy_version: str
    rule: str
    allowed_retries: int
    matching_prior_failures: int
    retry_budget_exhausted: bool
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
    observed_advance: bool = False
    match_requirements: Mapping[str, str] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "rule": self.rule,
            "allowed_retries": self.allowed_retries,
            "matching_prior_failures": self.matching_prior_failures,
            "retry_budget_exhausted": self.retry_budget_exhausted,
            "decision": self.decision,
            "decision_basis": self.decision_basis,
            "failure_domain": self.failure_domain,
            "failure_domain_status": self.failure_domain_status,
            "failure_domain_confidence": self.failure_domain_confidence,
            "retry_outlook_without_workload_change": (self.retry_outlook_without_workload_change),
            "retry_outlook_status": self.retry_outlook_status,
            "retry_outlook_confidence": self.retry_outlook_confidence,
            "recovery_assessment_policy_grounded": (self.recovery_assessment_policy_grounded),
            "current_evidence_qualified": self.current_evidence_qualified,
            "observed_advance": self.observed_advance,
            "match_requirements": dict(self.match_requirements or {}),
        }


@dataclass(frozen=True)
class L4PolicyOutcome:
    primary: FailureEvidence | None
    retry_policy: RetryPolicyEvaluation


def evaluate_policy(policy_input: L4PolicyInput) -> L4PolicyOutcome:
    configured = policy_input.retry_policy
    policy_version = configured.policy_version
    bounded_retries = configured.bounded_retry_allowed_retries
    general_retries = configured.general_retry_allowed_retries

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
    rule, allowed_retries = _select_rule(
        primary=policy_input.primary,
        assessment_grounded=policy_input.assessment_grounded,
        outlook=outlook,
        outlook_status=outlook_status,
        current_evidence_qualified=current_evidence_qualified,
        bounded_retries=bounded_retries,
        general_retries=general_retries,
    )

    matching_prior_failures = policy_input.history.consecutive_same_root_no_advance_attempts
    observed_advance = policy_input.history.advanced_beyond_all_comparable_attempts
    exhausted = current_evidence_qualified or (
        policy_input.primary is not None
        and policy_input.primary.registry_id != TIME_LIMIT_FINE_CLASS
        and not observed_advance
        and matching_prior_failures >= allowed_retries
    )
    decision, basis = _decision(
        primary=policy_input.primary,
        rule=rule,
        exhausted=exhausted,
        observed_advance=observed_advance,
    )
    return L4PolicyOutcome(
        primary=policy_input.primary,
        retry_policy=RetryPolicyEvaluation(
            policy_version=policy_version,
            rule=rule,
            allowed_retries=allowed_retries,
            matching_prior_failures=matching_prior_failures,
            retry_budget_exhausted=exhausted,
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
            observed_advance=observed_advance,
            match_requirements={
                "job_id": "exact",
                "root_fingerprint": "exact",
                "progress": "no_observed_advance",
            },
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
    assessment_grounded: bool,
    outlook: str | None,
    outlook_status: str | None,
    current_evidence_qualified: bool,
    bounded_retries: int,
    general_retries: int,
) -> tuple[str, int]:
    if primary is None:
        return RetryPolicyRule.NO_PRIMARY.value, general_retries
    if primary.registry_id == TIME_LIMIT_FINE_CLASS:
        return RetryPolicyRule.TIME_LIMIT.value, general_retries
    if current_evidence_qualified:
        return RetryPolicyRule.WORKLOAD_UNRECOVERABLE.value, 0
    if (
        assessment_grounded
        and outlook == RetryOutlookWithoutWorkloadChange.MAY_RECOVER.value
        and outlook_status
        in {
            AssessmentStatus.ESTABLISHED_BY_CURRENT_LOG.value,
            AssessmentStatus.SUPPORTED_BUT_UNCONFIRMED.value,
        }
    ):
        return RetryPolicyRule.BOUNDED_RETRY.value, bounded_retries
    return RetryPolicyRule.GENERAL_RETRY.value, general_retries


def _decision(
    *,
    primary: FailureEvidence | None,
    rule: str,
    exhausted: bool,
    observed_advance: bool,
) -> tuple[str, str]:
    if primary is None:
        return Decision.RESTART.value, DecisionBasis.NO_PRIMARY_FAILURE.value
    if rule == RetryPolicyRule.TIME_LIMIT.value:
        return Decision.RESTART.value, DecisionBasis.TIME_LIMIT.value
    if rule == RetryPolicyRule.WORKLOAD_UNRECOVERABLE.value:
        return Decision.STOP.value, DecisionBasis.WORKLOAD_UNRECOVERABLE.value
    if observed_advance:
        return Decision.RESTART.value, DecisionBasis.OBSERVED_ADVANCE.value
    if exhausted:
        return Decision.STOP.value, DecisionBasis.RETRY_BUDGET_EXHAUSTED.value
    if rule == RetryPolicyRule.BOUNDED_RETRY.value:
        return Decision.RESTART.value, DecisionBasis.RETRY_RECOVERY_AVAILABLE.value
    return Decision.RESTART.value, DecisionBasis.GENERAL_RETRY_AVAILABLE.value
