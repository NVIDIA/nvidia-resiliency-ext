# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic retry-rule selection and concurrent retry ledgers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..models import (
    CUDA_OOM_NO_RETRY_CONTEXT_ID,
    L1_CATEGORY_CONFIRMED_RESTART_CONTEXT_ID,
    PORT_BIND_CONFIRMATION_RETRY_CONTEXT_ID,
    REJECTED_ITERATION_RETRY_THEN_SKIP_CONTEXT_ID,
    AffectedEntity,
    AssessmentStatus,
    AttemptFailureFacts,
    CycleHistoryComparison,
    Decision,
    DecisionBasis,
    FailureClassifier,
    FailureDomain,
    FailureEvidence,
    FaultOutcome,
    HistoryMatchScope,
    HistoryProgressRelation,
    HistorySummary,
    ModelRecoveryAssessment,
    PolicyContextConfig,
    RetryOutlookWithoutWorkloadChange,
    RetryPolicyConfig,
    RetryPolicyRule,
)

GENERAL_ROOT_CEILING_ID = "general_root_ceiling"
SELECTED_POLICY_LEDGER_ID = "selected_policy_ledger"
JOB_NO_PROGRESS_GUARD_ID = "job_no_progress_guard"
JOB_UNKNOWN_PROGRESS_GUARD_ID = "job_unknown_progress_guard"


@dataclass(frozen=True)
class L4PolicyInput:
    primary: FailureEvidence | None
    history: HistorySummary
    selected_observed_failure: FailureEvidence | None = None
    current_failure_facts: AttemptFailureFacts | None = None
    current_affected_entity: AffectedEntity | None = None
    model_recovery_assessment: ModelRecoveryAssessment | None = None
    retry_policy: RetryPolicyConfig = RetryPolicyConfig()
    policy_contexts: PolicyContextConfig = PolicyContextConfig()
    # Opt-in L1 signal: the model's category selection from the 38-taxonomy in
    # l1/categories.py. When the picked category has decision=STOP and the
    # primary is grounded, it drives a policy_context override. None means
    # "no signal" and reverts to the base rule cascade; there is no default.
    l1_category_selection: Mapping[str, Any] | None = None

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
        if self.current_failure_facts is not None and not isinstance(
            self.current_failure_facts,
            AttemptFailureFacts,
        ):
            raise TypeError("L4 current_failure_facts must be typed")
        if not isinstance(self.retry_policy, RetryPolicyConfig):
            raise TypeError("L4 retry_policy must be typed")
        if not isinstance(self.policy_contexts, PolicyContextConfig):
            raise TypeError("L4 policy_contexts must be typed")


@dataclass(frozen=True)
class EffectiveRetryPolicy:
    source: str
    rule: str
    history_match_scope: str | None
    allowed_retries: int
    policy_context_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "rule": self.rule,
            "history_match_scope": self.history_match_scope,
            "allowed_retries": self.allowed_retries,
            "policy_context_id": self.policy_context_id,
        }


@dataclass(frozen=True)
class RetryLedgerEvaluation:
    ledger_id: str
    applicable: bool
    rule: str
    history_match_scope: str
    allowed_retries: int | None
    matching_prior_attempts: int = 0
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
            "matching_prior_attempts": self.matching_prior_attempts,
            "observed_advance": self.observed_advance,
            "exhausted": self.exhausted,
            "inapplicable_reason": self.inapplicable_reason,
        }


@dataclass(frozen=True)
class RetryPolicyEvaluation:
    policy_version: str
    base_rule: str | None
    effective_policy: EffectiveRetryPolicy | None
    applied_policy_context: Mapping[str, Any] | None
    retry_budget_exhausted: bool
    exhausted_by: tuple[str, ...]
    general_root_ceiling: RetryLedgerEvaluation
    selected_policy_ledger: RetryLedgerEvaluation | None
    job_no_progress_guard: RetryLedgerEvaluation
    job_unknown_progress_guard: RetryLedgerEvaluation
    decision: str
    decision_basis: str
    failure_domain: str | None = None
    failure_domain_status: str | None = None
    failure_domain_confidence: int | None = None
    retry_outlook_without_workload_change: str | None = None
    retry_outlook_status: str | None = None
    retry_outlook_confidence: int | None = None
    current_evidence_qualified: bool = False
    current_affected_entity: Mapping[str, Any] | None = None
    match_requirements: Mapping[str, str] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "base_rule": self.base_rule,
            "effective_policy": (
                self.effective_policy.to_payload() if self.effective_policy is not None else None
            ),
            "applied_policy_context": (
                dict(self.applied_policy_context)
                if self.applied_policy_context is not None
                else None
            ),
            "decision": self.decision,
            "decision_basis": self.decision_basis,
            "retry_budget_exhausted": self.retry_budget_exhausted,
            "exhausted_by": list(self.exhausted_by),
            "general_root_ceiling": self.general_root_ceiling.to_payload(),
            "selected_policy_ledger": (
                self.selected_policy_ledger.to_payload()
                if self.selected_policy_ledger is not None
                else None
            ),
            "job_no_progress_guard": self.job_no_progress_guard.to_payload(),
            "job_unknown_progress_guard": self.job_unknown_progress_guard.to_payload(),
            "failure_domain": self.failure_domain,
            "failure_domain_status": self.failure_domain_status,
            "failure_domain_confidence": self.failure_domain_confidence,
            "retry_outlook_without_workload_change": (self.retry_outlook_without_workload_change),
            "retry_outlook_status": self.retry_outlook_status,
            "retry_outlook_confidence": self.retry_outlook_confidence,
            "current_evidence_qualified": self.current_evidence_qualified,
            "current_affected_entity": (
                dict(self.current_affected_entity)
                if self.current_affected_entity is not None
                else None
            ),
            "match_requirements": dict(self.match_requirements or {}),
        }


@dataclass(frozen=True)
class L4PolicyOutcome:
    primary: FailureEvidence | None
    selected_observed_failure: FailureEvidence | None
    retry_policy: RetryPolicyEvaluation
    selected_failure_facts: AttemptFailureFacts | None = None
    selected_history: HistorySummary | None = None
    path_selection: "L4PathSelection | None" = None


@dataclass(frozen=True)
class L4PathSelection:
    """The one evidence/history track selected for policy evaluation."""

    path: str
    route_id: str | None
    reason: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "route_id": self.route_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class L4CyclePolicyInput:
    """All independent current-cycle paths available to L4."""

    deterministic_primary: FailureEvidence | None
    deterministic_observation: FailureEvidence | None
    deterministic_facts: AttemptFailureFacts
    history: CycleHistoryComparison
    route_id: str | None = None
    grounded_primary: FailureEvidence | None = None
    grounded_observation: FailureEvidence | None = None
    primary_facts: AttemptFailureFacts | None = None
    observation_facts: AttemptFailureFacts | None = None
    model_recovery_assessment: ModelRecoveryAssessment | None = None
    l1_primary_declared: bool = False
    retry_policy: RetryPolicyConfig = RetryPolicyConfig()
    policy_contexts: PolicyContextConfig = PolicyContextConfig()
    # L1 category selection - opt-in evidence signal for policy_context match.
    # None means the model didn't provide it (or L1 didn't run at all).
    l1_category_selection: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _PolicyContextMatch:
    effective_policy: EffectiveRetryPolicy
    payload: Mapping[str, Any]


def evaluate_policy(policy_input: L4PolicyInput) -> L4PolicyOutcome:
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

    current_evidence_qualified = _immediate_stop_qualified(
        primary=policy_input.primary,
        domain=domain,
        domain_status=domain_status,
        outlook=outlook,
        outlook_status=outlook_status,
    )
    base_rule = _select_base_rule(
        primary=policy_input.primary,
        selected_observed_failure=policy_input.selected_observed_failure,
        current_affected_entity=policy_input.current_affected_entity,
        domain=domain,
        domain_status=domain_status,
        current_evidence_qualified=current_evidence_qualified,
    )
    base_effective_policy = _effective_policy(
        base_rule,
        policy_input.retry_policy,
        observation_only=(
            policy_input.primary is None and policy_input.selected_observed_failure is not None
        ),
    )
    policy_context_match = _match_policy_context(
        primary=policy_input.primary,
        current_facts=policy_input.current_failure_facts,
        configured=policy_input.policy_contexts,
        l1_category_selection=policy_input.l1_category_selection,
        base_rule=base_rule,
        history=policy_input.history,
    )
    effective_policy = (
        policy_context_match.effective_policy
        if policy_context_match is not None
        else base_effective_policy
    )

    general_root_ceiling = _general_root_ceiling(
        primary=policy_input.primary,
        base_rule=base_rule,
        history=policy_input.history,
        allowed_retries=policy_input.retry_policy.general_retry_allowed_retries,
    )
    selected_policy_ledger = _selected_policy_ledger(
        effective_policy=effective_policy,
        history=policy_input.history,
        general_retries=policy_input.retry_policy.general_retry_allowed_retries,
    )
    job_no_progress_guard = _job_guard(
        ledger_id=JOB_NO_PROGRESS_GUARD_ID,
        scope=HistoryMatchScope.SAME_JOB_NO_PROGRESS,
        history=policy_input.history,
        allowed_retries=policy_input.retry_policy.job_no_progress_allowed_retries,
    )
    job_unknown_progress_guard = _job_guard(
        ledger_id=JOB_UNKNOWN_PROGRESS_GUARD_ID,
        scope=HistoryMatchScope.SAME_JOB_UNKNOWN_PROGRESS,
        history=policy_input.history,
        allowed_retries=policy_input.retry_policy.job_unknown_progress_allowed_retries,
    )

    ledgers = (
        selected_policy_ledger,
        general_root_ceiling,
        job_no_progress_guard,
        job_unknown_progress_guard,
    )
    exhausted_by = tuple(
        ledger.ledger_id for ledger in ledgers if ledger is not None and ledger.exhausted
    )
    observed_advance = any(
        ledger is not None and ledger.applicable and ledger.observed_advance for ledger in ledgers
    )
    decision, basis = _decision(
        primary=policy_input.primary,
        selected_observed_failure=policy_input.selected_observed_failure,
        effective_policy=effective_policy,
        exhausted_by=exhausted_by,
        observed_advance=observed_advance,
    )
    selected_scope = effective_policy.history_match_scope if effective_policy is not None else None
    return L4PolicyOutcome(
        primary=policy_input.primary,
        selected_observed_failure=policy_input.selected_observed_failure,
        selected_failure_facts=policy_input.current_failure_facts,
        selected_history=policy_input.history,
        retry_policy=RetryPolicyEvaluation(
            policy_version=policy_input.retry_policy.policy_version,
            base_rule=base_rule,
            effective_policy=effective_policy,
            applied_policy_context=(
                policy_context_match.payload if policy_context_match is not None else None
            ),
            retry_budget_exhausted=bool(exhausted_by),
            exhausted_by=exhausted_by,
            general_root_ceiling=general_root_ceiling,
            selected_policy_ledger=selected_policy_ledger,
            job_no_progress_guard=job_no_progress_guard,
            job_unknown_progress_guard=job_unknown_progress_guard,
            decision=decision,
            decision_basis=basis,
            failure_domain=domain,
            failure_domain_status=domain_status,
            failure_domain_confidence=domain_confidence,
            retry_outlook_without_workload_change=outlook,
            retry_outlook_status=outlook_status,
            retry_outlook_confidence=outlook_confidence,
            current_evidence_qualified=current_evidence_qualified,
            current_affected_entity=(
                policy_input.current_affected_entity.to_payload()
                if policy_input.current_affected_entity is not None
                else None
            ),
            match_requirements={
                "job_id": "exact",
                "root_fingerprint": ("exact" if policy_input.primary is not None else "missing"),
                "selected_observation": (
                    "grounded" if policy_input.selected_observed_failure is not None else "missing"
                ),
                "affected_entity": (
                    "exact"
                    if selected_scope == HistoryMatchScope.ROOT_AND_ENTITY.value
                    else "not_required"
                ),
                "rejected_iteration_signature": (
                    "exact"
                    if selected_scope == HistoryMatchScope.REJECTED_ITERATION_SIGNATURE.value
                    else "not_required"
                ),
                "cuda_oom_signature": (
                    "exact"
                    if effective_policy is not None
                    and effective_policy.policy_context_id == CUDA_OOM_NO_RETRY_CONTEXT_ID
                    else "not_required"
                ),
                "port_bind_conflict_signature": (
                    "exact"
                    if effective_policy is not None
                    and effective_policy.policy_context_id
                    == PORT_BIND_CONFIRMATION_RETRY_CONTEXT_ID
                    else "not_required"
                ),
                "progress": "ledger_specific",
            },
        ),
    )


def evaluate_cycle_policy(policy_input: L4CyclePolicyInput) -> L4PolicyOutcome:
    """Select one eligible track, then apply the deterministic retry policy."""

    route_history = (
        policy_input.history.route(policy_input.route_id)
        if policy_input.route_id is not None
        else None
    )
    if (
        policy_input.grounded_primary is not None
        and policy_input.primary_facts is not None
        and route_history is not None
        and route_history.primary is not None
    ):
        primary = policy_input.grounded_primary
        observation = None
        facts = policy_input.primary_facts
        history = route_history.primary
        assessment = policy_input.model_recovery_assessment
        selection = L4PathSelection(
            path="primary",
            route_id=policy_input.route_id,
            reason="grounded_primary_available",
        )
    elif (
        policy_input.grounded_observation is not None
        and policy_input.observation_facts is not None
        and route_history is not None
        and route_history.observation is not None
    ):
        primary = None
        observation = policy_input.grounded_observation
        facts = policy_input.observation_facts
        history = route_history.observation
        assessment = (
            None if policy_input.l1_primary_declared else policy_input.model_recovery_assessment
        )
        selection = L4PathSelection(
            path="observation",
            route_id=policy_input.route_id,
            reason=(
                "grounded_observation_after_primary_unavailable"
                if policy_input.l1_primary_declared
                else "grounded_observation_without_primary"
            ),
        )
    elif policy_input.deterministic_facts.history_identity_ready:
        primary = (
            policy_input.deterministic_primary
            if policy_input.deterministic_facts.identity_kind == "root"
            else None
        )
        observation = (
            policy_input.deterministic_observation
            if policy_input.deterministic_facts.identity_kind == "observation_only"
            else None
        )
        facts = policy_input.deterministic_facts
        history = policy_input.history.deterministic
        assessment = None
        selection = L4PathSelection(
            path="deterministic",
            route_id=None,
            reason="deterministic_selected_after_enriched_unavailable",
        )
    else:
        primary = None
        observation = None
        facts = policy_input.deterministic_facts
        history = policy_input.history.deterministic
        assessment = None
        selection = L4PathSelection(
            path="none",
            route_id=None,
            reason="no_usable_failure_identity",
        )

    outcome = evaluate_policy(
        L4PolicyInput(
            primary=primary,
            selected_observed_failure=observation,
            history=history,
            current_failure_facts=facts,
            current_affected_entity=facts.affected_entity,
            model_recovery_assessment=assessment,
            retry_policy=policy_input.retry_policy,
            policy_contexts=policy_input.policy_contexts,
            l1_category_selection=policy_input.l1_category_selection,
        )
    )
    return L4PolicyOutcome(
        primary=outcome.primary,
        selected_observed_failure=outcome.selected_observed_failure,
        retry_policy=outcome.retry_policy,
        selected_failure_facts=facts,
        selected_history=history,
        path_selection=selection,
    )


def _immediate_stop_qualified(
    *,
    primary: FailureEvidence | None,
    domain: str | None,
    domain_status: str | None,
    outlook: str | None,
    outlook_status: str | None,
) -> bool:
    return bool(
        primary is not None
        and primary.fault_outcome in {FaultOutcome.TERMINAL.value, FaultOutcome.UNRESOLVED.value}
        and domain == FailureDomain.WORKLOAD.value
        and domain_status == AssessmentStatus.ESTABLISHED_BY_CURRENT_LOG.value
        and outlook == RetryOutlookWithoutWorkloadChange.CANNOT_RECOVER.value
        and outlook_status == AssessmentStatus.ESTABLISHED_BY_CURRENT_LOG.value
    )


def _select_base_rule(
    *,
    primary: FailureEvidence | None,
    selected_observed_failure: FailureEvidence | None,
    current_affected_entity: AffectedEntity | None,
    domain: str | None,
    domain_status: str | None,
    current_evidence_qualified: bool,
) -> str | None:
    if primary is None:
        return (
            RetryPolicyRule.GENERAL_RETRY.value if selected_observed_failure is not None else None
        )
    if current_evidence_qualified:
        return RetryPolicyRule.WORKLOAD_UNRECOVERABLE.value
    if primary.root_fingerprint and current_affected_entity is not None:
        return RetryPolicyRule.CONCRETE_CONFIRMATION_RETRY.value
    if domain == FailureDomain.WORKLOAD.value and domain_status in {
        AssessmentStatus.ESTABLISHED_BY_CURRENT_LOG.value,
        AssessmentStatus.SUPPORTED_BUT_UNCONFIRMED.value,
    }:
        return RetryPolicyRule.WORKLOAD_CONFIRMATION_RETRY.value
    return RetryPolicyRule.GENERAL_RETRY.value


def _effective_policy(
    base_rule: str | None,
    configured: RetryPolicyConfig,
    *,
    observation_only: bool = False,
) -> EffectiveRetryPolicy | None:
    if base_rule is None:
        return None
    if base_rule == RetryPolicyRule.WORKLOAD_UNRECOVERABLE.value:
        scope = None
        allowed_retries = 0
    elif base_rule == RetryPolicyRule.CONCRETE_CONFIRMATION_RETRY.value:
        scope = HistoryMatchScope.ROOT_AND_ENTITY.value
        allowed_retries = configured.concrete_confirmation_retry_allowed_retries
    elif base_rule == RetryPolicyRule.WORKLOAD_CONFIRMATION_RETRY.value:
        scope = HistoryMatchScope.ROOT_ONLY.value
        allowed_retries = configured.workload_confirmation_retry_allowed_retries
    else:
        scope = (
            HistoryMatchScope.SAME_JOB_NO_PROGRESS.value
            if observation_only
            else HistoryMatchScope.ROOT_ONLY.value
        )
        allowed_retries = configured.general_retry_allowed_retries
    return EffectiveRetryPolicy(
        source="base_rule",
        rule=base_rule,
        history_match_scope=scope,
        allowed_retries=allowed_retries,
    )


L1_CATEGORY_CONFIRMED_STOP_CONTEXT_ID = "l1_category_confirmed_stop"


def _match_l1_category_context(
    *,
    primary: FailureEvidence | None,
    l1_category_selection: Mapping[str, Any] | None,
) -> _PolicyContextMatch | None:
    """Match the L1 category taxonomy as a policy_context.

    The LLM classifies into the 38-entry taxonomy at l1/categories.py; when it
    picks a category whose taxonomy-declared decision is STOP and the primary
    is grounded, override with a zero-retry policy (which behaves the same as
    cuda_oom_no_retry for the ledger and produces STOP on first occurrence).

    Deliberate constraints:
    - Requires a grounded primary. Category can't drive policy for a
      no-primary case; that path stays with the existing base rules.
    - Only STOP-labeled categories fire. RESTART-labeled categories fall
      through to the base rule, since general_retry already models them.
    - Precedence: this context is checked LAST, so any deterministic
      classifier context (cuda_oom_no_retry, port_bind_confirmation_retry,
      rejected_iteration_retry_then_skip) that matched earlier wins.
    - No confidence gate. Empirical calibration on the 79-case corpus showed
      the threshold was doing zero filtering work on qwen397b/gemini/nemotron
      and negative work on gpt (blocking 2 correct STOPs at conf 76-78).
      The category confidence field is still emitted for transparency and
      future calibration analysis, but the gate is 'is a STOP category picked
      with a grounded primary', not 'is confidence above N'.
    """

    from ..l1.categories import category_by_id  # local import to keep L4 loose-coupled

    if primary is None or not isinstance(l1_category_selection, Mapping):
        return None
    cid = l1_category_selection.get("category_id")
    if not isinstance(cid, int) or isinstance(cid, bool) or cid <= 0:
        return None
    category = category_by_id(cid)
    if category is None or category.decision != "STOP":
        return None
    conf = l1_category_selection.get("category_confidence")
    reported_confidence = conf if isinstance(conf, int) and not isinstance(conf, bool) else None
    effective = EffectiveRetryPolicy(
        source="policy_context",
        rule=RetryPolicyRule.WORKLOAD_UNRECOVERABLE.value,
        history_match_scope=None,
        allowed_retries=0,
        policy_context_id=L1_CATEGORY_CONFIRMED_STOP_CONTEXT_ID,
    )
    return _PolicyContextMatch(
        effective_policy=effective,
        payload={
            "policy_context_id": L1_CATEGORY_CONFIRMED_STOP_CONTEXT_ID,
            "matched": True,
            "current_signature": {
                "l1_category_id": cid,
                "l1_category_name": category.name,
                "l1_category_confidence_reported": reported_confidence,
            },
            "retry_policy": effective.to_payload(),
        },
    )


def _match_l1_category_restart_context(
    *,
    primary: FailureEvidence | None,
    l1_category_selection: Mapping[str, Any] | None,
    base_rule: str | None,
    history: HistorySummary | None,
    configured: Any,
) -> _PolicyContextMatch | None:
    """Match category-driven RESTART override on FIRST occurrence.

    Fires when all of the following hold:
    - The category context is enabled.
    - Primary is grounded.
    - Category picker chose a RESTART-labeled taxonomy entry.
    - Base rule concluded workload_unrecoverable (a STOP eligible for override).
    - History shows zero prior attempts with the same root_fingerprint.

    Emits workload_confirmation_retry with history_match_scope=ROOT_ONLY and
    a tight allowed_retries budget. On any subsequent cycle with the same
    root, the history guard suppresses the override; base_rule +
    exhaustion take over and STOP.

    Design rationale:
    - Only overrides workload_unrecoverable, not other STOP-yielding rules,
      to keep the override surface minimal.
    - Only fires on first occurrence: PR #400's history ledger handles
      recurrences; we do not compete with it.
    - No confidence gate: the category picker's decision label is treated
      as authoritative, matching the STOP branch above.
    """

    from ..l1.categories import category_by_id  # local import to keep L4 loose-coupled

    if not getattr(configured, "enabled", False):
        return None
    if primary is None or not isinstance(l1_category_selection, Mapping):
        return None
    if base_rule != RetryPolicyRule.WORKLOAD_UNRECOVERABLE.value:
        return None
    cid = l1_category_selection.get("category_id")
    if not isinstance(cid, int) or isinstance(cid, bool) or cid <= 0:
        return None
    category = category_by_id(cid)
    if category is None or category.decision != "RESTART":
        return None
    # First-occurrence guard: refuse to override once we have any recurrence.
    if history is None or history.matching_root_attempts > 0:
        return None
    conf = l1_category_selection.get("category_confidence")
    reported_confidence = conf if isinstance(conf, int) and not isinstance(conf, bool) else None
    effective = EffectiveRetryPolicy(
        source="policy_context",
        rule=RetryPolicyRule.WORKLOAD_CONFIRMATION_RETRY.value,
        history_match_scope=HistoryMatchScope.ROOT_ONLY.value,
        allowed_retries=int(getattr(configured, "allowed_retries", 1)),
        policy_context_id=L1_CATEGORY_CONFIRMED_RESTART_CONTEXT_ID,
    )
    return _PolicyContextMatch(
        effective_policy=effective,
        payload={
            "policy_context_id": L1_CATEGORY_CONFIRMED_RESTART_CONTEXT_ID,
            "matched": True,
            "current_signature": {
                "l1_category_id": cid,
                "l1_category_name": category.name,
                "l1_category_confidence_reported": reported_confidence,
                "overridden_base_rule": base_rule,
            },
            "retry_policy": effective.to_payload(),
        },
    )


def _match_policy_context(
    *,
    primary: FailureEvidence | None,
    current_facts: AttemptFailureFacts | None,
    configured: PolicyContextConfig,
    l1_category_selection: Mapping[str, Any] | None = None,
    base_rule: str | None = None,
    history: HistorySummary | None = None,
) -> _PolicyContextMatch | None:
    cuda_oom = configured.cuda_oom_no_retry
    if (
        cuda_oom.enabled
        and primary is not None
        and current_facts is not None
        and (
            FailureClassifier.CUDA_OOM.value in current_facts.classifiers
            or primary.failure_class == "cuda_oom"
        )
        and current_facts.fault_outcome
        in {FaultOutcome.TERMINAL.value, FaultOutcome.UNRESOLVED.value}
    ):
        effective = EffectiveRetryPolicy(
            source="policy_context",
            rule=RetryPolicyRule.CUDA_OOM_NO_RETRY.value,
            history_match_scope=None,
            allowed_retries=0,
            policy_context_id=CUDA_OOM_NO_RETRY_CONTEXT_ID,
        )
        return _PolicyContextMatch(
            effective_policy=effective,
            payload={
                "policy_context_id": CUDA_OOM_NO_RETRY_CONTEXT_ID,
                "matched": True,
                "current_signature": {
                    "classifiers": list(current_facts.classifiers),
                    "failure_class": primary.failure_class,
                    "fault_outcome": current_facts.fault_outcome,
                },
                "retry_policy": effective.to_payload(),
            },
        )

    port_bind = configured.port_bind_confirmation_retry
    if (
        port_bind.enabled
        and primary is not None
        and current_facts is not None
        and FailureClassifier.PORT_BIND_CONFLICT.value in current_facts.classifiers
        and current_facts.root_fingerprint is not None
        and current_facts.fault_outcome
        in {FaultOutcome.TERMINAL.value, FaultOutcome.UNRESOLVED.value}
    ):
        effective = EffectiveRetryPolicy(
            source="policy_context",
            rule=RetryPolicyRule.PORT_BIND_CONFIRMATION_RETRY.value,
            history_match_scope=HistoryMatchScope.ROOT_ONLY.value,
            allowed_retries=port_bind.allowed_retries,
            policy_context_id=PORT_BIND_CONFIRMATION_RETRY_CONTEXT_ID,
        )
        return _PolicyContextMatch(
            effective_policy=effective,
            payload={
                "policy_context_id": PORT_BIND_CONFIRMATION_RETRY_CONTEXT_ID,
                "matched": True,
                "current_signature": {
                    "classifiers": list(current_facts.classifiers),
                    "failure_class": primary.failure_class,
                    "fault_outcome": current_facts.fault_outcome,
                },
                "retry_policy": effective.to_payload(),
            },
        )

    # rejected_iteration_retry_then_skip: check signature; if it matches, return
    # match. If any precondition fails, fall through (do NOT return None early,
    # since the L1 category context below is the next thing to try).
    context = configured.rejected_iteration_retry_then_skip
    rej_iter_match = _try_rejected_iteration_context(
        context=context, primary=primary, current_facts=current_facts
    )
    if rej_iter_match is not None:
        return rej_iter_match

    # L1 category-driven STOP context runs after all deterministic classifier
    # contexts so any deterministic classifier above takes precedence. Callers
    # pass l1_category_selection via L4PolicyInput.
    stop_match = _match_l1_category_context(
        primary=primary,
        l1_category_selection=l1_category_selection,
    )
    if stop_match is not None:
        return stop_match

    # L1 category-driven RESTART context runs LAST. Only fires when the
    # optional l1_category_confirmed_restart context is enabled and the
    # base_rule would have concluded workload_unrecoverable. History guard
    # ensures we only override on FIRST occurrence.
    return _match_l1_category_restart_context(
        primary=primary,
        l1_category_selection=l1_category_selection,
        base_rule=base_rule,
        history=history,
        configured=configured.l1_category_confirmed_restart,
    )


def _try_rejected_iteration_context(
    *,
    context: Any,
    primary: FailureEvidence | None,
    current_facts: AttemptFailureFacts | None,
) -> _PolicyContextMatch | None:
    """Return a rejected_iteration_retry_then_skip match, or None to fall through."""

    if not context.enabled or primary is None or current_facts is None:
        return None
    if not current_facts.root_fingerprint or current_facts.failure_iteration is None:
        return None
    if FailureClassifier.REJECTED_NONFINITE_ITERATION.value not in current_facts.classifiers:
        return None
    observer_ranks = current_facts.root_observer_ranks
    if observer_ranks is None or len(observer_ranks) != 1:
        return None
    if current_facts.unattributed_root_occurrence_count != 0:
        return None

    effective = EffectiveRetryPolicy(
        source="policy_context",
        rule=RetryPolicyRule.REJECTED_ITERATION_RETRY_THEN_SKIP.value,
        history_match_scope=HistoryMatchScope.REJECTED_ITERATION_SIGNATURE.value,
        allowed_retries=context.allowed_retries,
        policy_context_id=REJECTED_ITERATION_RETRY_THEN_SKIP_CONTEXT_ID,
    )
    return _PolicyContextMatch(
        effective_policy=effective,
        payload={
            "policy_context_id": REJECTED_ITERATION_RETRY_THEN_SKIP_CONTEXT_ID,
            "matched": True,
            "current_signature": {
                "classifiers": list(current_facts.classifiers),
                "failure_iteration": current_facts.failure_iteration,
                "root_observer_count": len(observer_ranks),
                "unattributed_root_occurrence_count": (
                    current_facts.unattributed_root_occurrence_count
                ),
            },
            "retry_policy": effective.to_payload(),
        },
    )


def _general_root_ceiling(
    *,
    primary: FailureEvidence | None,
    base_rule: str | None,
    history: HistorySummary,
    allowed_retries: int,
) -> RetryLedgerEvaluation:
    if primary is None:
        return _inapplicable_ledger(
            ledger_id=GENERAL_ROOT_CEILING_ID,
            rule=RetryPolicyRule.GENERAL_RETRY.value,
            scope=HistoryMatchScope.ROOT_ONLY,
            reason="missing_primary",
            allowed_retries=allowed_retries,
        )
    if base_rule == RetryPolicyRule.WORKLOAD_UNRECOVERABLE.value:
        return _inapplicable_ledger(
            ledger_id=GENERAL_ROOT_CEILING_ID,
            rule=RetryPolicyRule.GENERAL_RETRY.value,
            scope=HistoryMatchScope.ROOT_ONLY,
            reason="immediate_unrecoverable",
            allowed_retries=allowed_retries,
        )
    return _evaluate_root_ledger(
        ledger_id=GENERAL_ROOT_CEILING_ID,
        rule=RetryPolicyRule.GENERAL_RETRY.value,
        scope=HistoryMatchScope.ROOT_ONLY,
        allowed_retries=allowed_retries,
        history=history,
    )


def _selected_policy_ledger(
    *,
    effective_policy: EffectiveRetryPolicy | None,
    history: HistorySummary,
    general_retries: int,
) -> RetryLedgerEvaluation | None:
    if effective_policy is None:
        return None
    if effective_policy.rule in {
        RetryPolicyRule.WORKLOAD_UNRECOVERABLE.value,
        RetryPolicyRule.CUDA_OOM_NO_RETRY.value,
    }:
        return None
    if effective_policy.rule == RetryPolicyRule.GENERAL_RETRY.value:
        if effective_policy.history_match_scope != HistoryMatchScope.SAME_JOB_NO_PROGRESS.value:
            return None
        return _job_guard(
            ledger_id=SELECTED_POLICY_LEDGER_ID,
            scope=HistoryMatchScope.SAME_JOB_NO_PROGRESS,
            history=history,
            allowed_retries=effective_policy.allowed_retries,
            rule=effective_policy.rule,
        )
    if effective_policy.rule == RetryPolicyRule.REJECTED_ITERATION_RETRY_THEN_SKIP.value:
        return _evaluate_rejected_iteration_ledger(
            effective_policy=effective_policy,
            history=history,
        )
    if effective_policy.allowed_retries >= general_retries:
        return None
    assert effective_policy.history_match_scope is not None
    return _evaluate_root_ledger(
        ledger_id=SELECTED_POLICY_LEDGER_ID,
        rule=effective_policy.rule,
        scope=HistoryMatchScope(effective_policy.history_match_scope),
        allowed_retries=effective_policy.allowed_retries,
        history=history,
    )


def _evaluate_rejected_iteration_ledger(
    *,
    effective_policy: EffectiveRetryPolicy,
    history: HistorySummary,
) -> RetryLedgerEvaluation:
    scope = HistoryMatchScope.REJECTED_ITERATION_SIGNATURE
    allowed_retries = effective_policy.allowed_retries
    if allowed_retries == 0:
        return RetryLedgerEvaluation(
            ledger_id=SELECTED_POLICY_LEDGER_ID,
            applicable=True,
            rule=effective_policy.rule,
            history_match_scope=scope.value,
            allowed_retries=0,
            exhausted=True,
        )
    if not history.available:
        return _inapplicable_ledger(
            ledger_id=SELECTED_POLICY_LEDGER_ID,
            rule=effective_policy.rule,
            scope=scope,
            reason="history_unavailable",
            allowed_retries=allowed_retries,
        )

    matching = 0
    observed_advance = False
    no_advance_boundary = history.consecutive_same_root_no_advance_attempts
    for comparison in reversed(history.comparisons):
        if not (
            comparison.same_failure_iteration
            and comparison.same_root_observer_count
            and comparison.same_unattributed_root_occurrence_count
        ):
            break
        if comparison.prior_fault_outcome not in {
            FaultOutcome.TERMINAL.value,
            FaultOutcome.UNRESOLVED.value,
        }:
            break
        if comparison.relation == HistoryProgressRelation.ADVANCED.value:
            observed_advance = matching == 0
            break
        if comparison.relation not in {
            HistoryProgressRelation.SAME.value,
            HistoryProgressRelation.REGRESSED.value,
        }:
            break
        if matching >= no_advance_boundary:
            break
        matching += 1

    return RetryLedgerEvaluation(
        ledger_id=SELECTED_POLICY_LEDGER_ID,
        applicable=True,
        rule=effective_policy.rule,
        history_match_scope=scope.value,
        allowed_retries=allowed_retries,
        matching_prior_attempts=matching,
        observed_advance=observed_advance,
        exhausted=(not observed_advance and matching >= allowed_retries),
    )


def _evaluate_root_ledger(
    *,
    ledger_id: str,
    rule: str,
    scope: HistoryMatchScope,
    allowed_retries: int,
    history: HistorySummary,
) -> RetryLedgerEvaluation:
    if allowed_retries == 0:
        return RetryLedgerEvaluation(
            ledger_id=ledger_id,
            applicable=True,
            rule=rule,
            history_match_scope=scope.value,
            allowed_retries=0,
            matching_prior_attempts=0,
            exhausted=True,
        )
    if not history.available:
        return _inapplicable_ledger(
            ledger_id=ledger_id,
            rule=rule,
            scope=scope,
            reason="history_unavailable",
            allowed_retries=allowed_retries,
        )
    if scope == HistoryMatchScope.ROOT_AND_ENTITY:
        matching = history.consecutive_same_root_and_entity_no_advance_attempts
        observed_advance = history.advanced_beyond_all_same_entity_comparable_attempts
    else:
        matching = history.consecutive_same_root_no_advance_attempts
        observed_advance = history.advanced_beyond_all_comparable_attempts
    return RetryLedgerEvaluation(
        ledger_id=ledger_id,
        applicable=True,
        rule=rule,
        history_match_scope=scope.value,
        allowed_retries=allowed_retries,
        matching_prior_attempts=matching,
        observed_advance=observed_advance,
        exhausted=(not observed_advance and matching >= allowed_retries),
    )


def _job_guard(
    *,
    ledger_id: str,
    scope: HistoryMatchScope,
    history: HistorySummary,
    allowed_retries: int,
    rule: str | None = None,
) -> RetryLedgerEvaluation:
    if not history.job_history_available:
        return _inapplicable_ledger(
            ledger_id=ledger_id,
            rule=rule or ledger_id,
            scope=scope,
            reason=history.job_history_availability_reason,
            allowed_retries=allowed_retries,
        )
    matching = (
        history.consecutive_same_job_no_advance_attempts
        if scope == HistoryMatchScope.SAME_JOB_NO_PROGRESS
        else history.consecutive_same_job_unknown_progress_attempts
    )
    observed_advance = history.job_progress_advanced
    return RetryLedgerEvaluation(
        ledger_id=ledger_id,
        applicable=True,
        rule=rule or ledger_id,
        history_match_scope=scope.value,
        allowed_retries=allowed_retries,
        matching_prior_attempts=matching,
        observed_advance=observed_advance,
        exhausted=(not observed_advance and matching >= allowed_retries),
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


def _decision(
    *,
    primary: FailureEvidence | None,
    selected_observed_failure: FailureEvidence | None,
    effective_policy: EffectiveRetryPolicy | None,
    exhausted_by: tuple[str, ...],
    observed_advance: bool,
) -> tuple[str, str]:
    if effective_policy is not None and effective_policy.allowed_retries == 0:
        if effective_policy.rule == RetryPolicyRule.WORKLOAD_UNRECOVERABLE.value:
            return Decision.STOP.value, DecisionBasis.WORKLOAD_UNRECOVERABLE.value
        if effective_policy.source == "policy_context":
            return Decision.STOP.value, DecisionBasis.POLICY_CONTEXT_NO_RETRY.value
        return Decision.STOP.value, DecisionBasis.RETRY_BUDGET_EXHAUSTED.value
    if observed_advance:
        return Decision.RESTART.value, DecisionBasis.OBSERVED_ADVANCE.value
    if JOB_NO_PROGRESS_GUARD_ID in exhausted_by:
        return Decision.STOP.value, DecisionBasis.JOB_NO_PROGRESS_BUDGET_EXHAUSTED.value
    if JOB_UNKNOWN_PROGRESS_GUARD_ID in exhausted_by:
        return (
            Decision.STOP.value,
            DecisionBasis.PROGRESS_UNVERIFIABLE_BUDGET_EXHAUSTED.value,
        )
    if exhausted_by:
        return Decision.STOP.value, DecisionBasis.RETRY_BUDGET_EXHAUSTED.value
    if primary is None and selected_observed_failure is None:
        return Decision.RESTART.value, DecisionBasis.NO_PRIMARY_FAILURE.value
    if effective_policy is None:
        return Decision.RESTART.value, DecisionBasis.GENERAL_RETRY_AVAILABLE.value
    if effective_policy.rule == RetryPolicyRule.CONCRETE_CONFIRMATION_RETRY.value:
        return (
            Decision.RESTART.value,
            DecisionBasis.CONCRETE_CONFIRMATION_RETRY_AVAILABLE.value,
        )
    if effective_policy.rule == RetryPolicyRule.WORKLOAD_CONFIRMATION_RETRY.value:
        return (
            Decision.RESTART.value,
            DecisionBasis.WORKLOAD_CONFIRMATION_RETRY_AVAILABLE.value,
        )
    if effective_policy.source == "policy_context":
        return Decision.RESTART.value, DecisionBasis.POLICY_CONTEXT_RETRY_AVAILABLE.value
    return Decision.RESTART.value, DecisionBasis.GENERAL_RETRY_AVAILABLE.value
