# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed L2-L4 decision execution and external result assembly."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from .attempt_records import AttemptRecordAssembler
from .causality import build_result_cascades
from .l1 import L1EvidenceResult, execution_status
from .l2 import L2Result, build_attempt_failure_facts
from .l3 import DETERMINISTIC_FACT_SELECTOR, HistoryEvaluationInput, evaluate_history
from .l4 import L4PolicyInput, RetryPolicyEvaluation, evaluate_policy
from .models import (
    AnalysisExecutionContext,
    AnalysisResult,
    AttemptFailureFacts,
    AttemptFailureFactsSource,
    AttemptRecord,
    CausalRole,
    CoverageStatus,
    Decision,
    DecisionBasis,
    DecisionEvidence,
    FailureEvidence,
    HistorySummary,
    L0Bundle,
    PolicyClass,
    RetryPolicyConfig,
)
from .runtime import SYSTEM_CLOCK, Clock


@dataclass(frozen=True)
class DecisionOutcome:
    """Typed result of failure-fact selection, history, and policy."""

    result: AnalysisResult
    primary: FailureEvidence | None
    l2_primary: FailureEvidence | None
    l2_audit: Mapping[str, Any]
    history: HistorySummary
    retry_policy: RetryPolicyEvaluation
    attempt_record: AttemptRecord | None
    selected_failure_facts: AttemptFailureFacts
    l3_wall_clock_s: float
    l4_wall_clock_s: float


@dataclass(frozen=True)
class _ResolvedFailure:
    audit: Mapping[str, Any]
    l2_primary: FailureEvidence | None
    primary: FailureEvidence | None
    enriched_failure_facts: AttemptFailureFacts | None


def build_decision_outcome(
    *,
    bundle: L0Bundle,
    decision_evidence: DecisionEvidence,
    execution_context: AnalysisExecutionContext,
    l1_configured: bool,
    l1_result: L1EvidenceResult,
    l1_output_health: Mapping[str, Any],
    l2_result: L2Result,
    candidate_kind: str,
    l1_pending: bool,
    route_id: str = "single",
    clock: Clock = SYSTEM_CLOCK,
) -> DecisionOutcome:
    resolved = _resolve_attempt_failure_facts(
        decision_evidence=decision_evidence,
        l2_result=l2_result,
    )

    deterministic_facts = build_attempt_failure_facts(
        decision_evidence.deterministic_primary_candidate,
        decision_evidence,
        source=AttemptFailureFactsSource.L0_DETERMINISTIC,
    )
    attempt_record = _attempt_record(
        bundle=bundle,
        decision_evidence=decision_evidence,
        execution_context=execution_context,
        enriched_failure_facts=resolved.enriched_failure_facts if l2_result.used else None,
        route_id=route_id,
    )
    fact_selector = (
        route_id
        if l2_result.used and resolved.enriched_failure_facts is not None
        else DETERMINISTIC_FACT_SELECTOR
    )
    selected_failure_facts = resolved.enriched_failure_facts or deterministic_facts

    l3_started = clock.monotonic()
    if attempt_record is None:
        history = HistorySummary(
            available=False,
            availability_reason=execution_context.prior_attempts.availability_reason,
        )
    else:
        history = evaluate_history(
            HistoryEvaluationInput(
                current_record=attempt_record,
                fact_selector=fact_selector,
                prior_attempts=execution_context.prior_attempts,
            )
        )
    l3_wall_clock_s = round(clock.monotonic() - l3_started, 3)

    l4_started = clock.monotonic()
    l4_outcome = evaluate_policy(
        L4PolicyInput(
            primary=resolved.primary,
            history=history,
            model_recovery_assessment=l2_result.model_recovery_assessment,
            assessment_grounded=l2_result.recovery_assessment_policy_grounded,
            retry_policy=RetryPolicyConfig.from_mapping(execution_context.retry_policy),
        )
    )
    primary = l4_outcome.primary
    retry_policy = l4_outcome.retry_policy
    result_provenance = _candidate_provenance(
        primary=primary,
        retry_policy=retry_policy,
        l1_configured=l1_configured,
        l1_result=l1_result,
        l1_output_health=l1_output_health,
        l2_audit=resolved.audit,
        history=history,
        candidate_kind=candidate_kind,
        l1_pending=l1_pending,
    )
    result = _assemble_analysis_result(
        bundle=bundle,
        primary=primary,
        retry_policy=retry_policy,
        result_provenance=result_provenance,
        l1_result=l1_result,
        l1_output_health=l1_output_health,
        l2_audit=resolved.audit,
        history=history,
        l1_configured=l1_configured,
        candidate_kind=candidate_kind,
    )
    l4_wall_clock_s = round(clock.monotonic() - l4_started, 3)
    return DecisionOutcome(
        result=result,
        primary=primary,
        l2_primary=resolved.l2_primary,
        l2_audit=resolved.audit,
        attempt_record=attempt_record,
        selected_failure_facts=selected_failure_facts,
        history=history,
        retry_policy=retry_policy,
        l3_wall_clock_s=l3_wall_clock_s,
        l4_wall_clock_s=l4_wall_clock_s,
    )


def _resolve_attempt_failure_facts(
    *,
    decision_evidence: DecisionEvidence,
    l2_result: L2Result,
) -> _ResolvedFailure:
    audit = l2_result.to_payload()
    l2_primary = l2_result.primary if l2_result.used else None
    primary = (
        l2_primary if l2_primary is not None else decision_evidence.deterministic_primary_candidate
    )
    if not audit.get("used"):
        primary = _downgrade_l0_semantic_stop(primary, audit)
    return _ResolvedFailure(
        audit=audit,
        l2_primary=l2_primary,
        primary=primary,
        enriched_failure_facts=l2_result.enriched_failure_facts,
    )


def _attempt_record(
    *,
    bundle: L0Bundle,
    decision_evidence: DecisionEvidence,
    execution_context: AnalysisExecutionContext,
    enriched_failure_facts: AttemptFailureFacts | None,
    route_id: str,
) -> AttemptRecord | None:
    if execution_context.job_id is None or execution_context.cycle_id is None:
        return None
    assembler = AttemptRecordAssembler()
    record = assembler.initial_record(
        job_id=execution_context.job_id,
        cycle_id=execution_context.cycle_id,
        bundle=bundle,
        decision_evidence=decision_evidence,
    )
    if enriched_failure_facts is not None:
        record = assembler.with_enriched(
            record,
            route_id=route_id,
            facts=enriched_failure_facts,
        )
    return record


def _candidate_provenance(
    *,
    primary: FailureEvidence | None,
    retry_policy: RetryPolicyEvaluation,
    l1_configured: bool,
    l1_result: L1EvidenceResult,
    l1_output_health: Mapping[str, Any],
    l2_audit: Mapping[str, Any],
    history: HistorySummary,
    candidate_kind: str,
    l1_pending: bool,
) -> dict[str, Any]:
    provenance = _result_provenance(
        primary=primary,
        decision_basis=retry_policy.decision_basis,
        l1_configured=l1_configured,
        l1_result=l1_result,
        l1_output_health=l1_output_health,
        l2_audit=l2_audit,
        history=history,
        retry_policy=retry_policy,
    )
    provenance["candidate_kind"] = candidate_kind
    if not l1_pending:
        return provenance
    provenance["model_contribution"] = "pending_not_used"
    provenance["l1_execution_status"] = "in_flight"
    provenance["l1_execution_issues"] = []
    provenance["notes"] = [*list(provenance.get("notes") or []), "l1_pending"]
    if (
        provenance.get("result_quality") == "normal"
        and provenance.get("history_contribution") != "recurrence_applied"
    ):
        provenance["result_quality"] = "degraded"
        provenance["nvrx_use"] = "eligible_degraded"
    return provenance


def _assemble_analysis_result(
    *,
    bundle: L0Bundle,
    primary: FailureEvidence | None,
    retry_policy: RetryPolicyEvaluation,
    result_provenance: Mapping[str, Any],
    l1_result: L1EvidenceResult,
    l1_output_health: Mapping[str, Any],
    l2_audit: Mapping[str, Any],
    history: HistorySummary,
    l1_configured: bool,
    candidate_kind: str,
) -> AnalysisResult:
    if primary is None and not l1_output_health["usable"] and l1_result.model:
        malformed_provenance = _result_provenance(
            primary=None,
            decision_basis=DecisionBasis.MALFORMED_MODEL_OUTPUT.value,
            l1_configured=l1_configured,
            l1_result=l1_result,
            l1_output_health=l1_output_health,
            l2_audit=l2_audit,
            history=history,
            retry_policy=retry_policy,
        )
        malformed_provenance["candidate_kind"] = candidate_kind
        return AnalysisResult(
            decision=Decision.RESTART.value,
            decision_basis=DecisionBasis.MALFORMED_MODEL_OUTPUT.value,
            retry_policy=retry_policy.to_payload(),
            failure_domain=retry_policy.failure_domain,
            result_provenance=malformed_provenance,
            primary_failure=None,
            secondary_failures=(),
            cascades=(),
            evidence_coverage=_coverage_with_history(bundle.evidence_coverage, history.available),
            evidence=(),
            justification="L1 model evidence was malformed and no L0 primary was available.",
        )
    return AnalysisResult(
        decision=retry_policy.decision,
        decision_basis=retry_policy.decision_basis,
        retry_policy=retry_policy.to_payload(),
        failure_domain=retry_policy.failure_domain,
        result_provenance=result_provenance,
        primary_failure=primary.to_failure_payload() if primary is not None else None,
        root_cause_assessment=_assessment(l2_audit, "root_cause_assessment"),
        model_recovery_assessment=_assessment(l2_audit, "model_recovery_assessment"),
        secondary_failures=_secondary_failures(
            bundle,
            primary,
            l2_audit=l2_audit,
            neutralize_semantic_policy_labels=not bool(l2_audit.get("used")),
        ),
        cascades=build_result_cascades(bundle, primary, l2_audit),
        evidence_coverage=_coverage_with_history(bundle.evidence_coverage, history.available),
        evidence=_evidence(primary, l2_audit),
        justification=_justification(primary, retry_policy.decision_basis, l2_audit),
    )


def _secondary_failures(
    bundle: L0Bundle,
    primary: FailureEvidence | None,
    *,
    l2_audit: Mapping[str, Any],
    neutralize_semantic_policy_labels: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    if l2_audit.get("used"):
        related = l2_audit.get("audited_related_failures")
        if isinstance(related, (list, tuple)):
            return tuple(
                item
                for item in related
                if isinstance(item, Mapping)
                and item.get("causal_role")
                not in {CausalRole.CASCADE.value, CausalRole.TEARDOWN.value}
            )
        return ()

    primary_line = primary.line if primary else None
    primary_fingerprint = primary.root_fingerprint if primary else None
    primary_episode_chain_lines = _primary_episode_chain_lines(bundle, primary_line)
    secondary = []
    for match in bundle.registry_matches:
        if match.line == primary_line and match.root_fingerprint == primary_fingerprint:
            continue
        if match.policy_class == "cascade":
            continue
        if match.line in primary_episode_chain_lines:
            continue
        if (
            neutralize_semantic_policy_labels
            and match.policy_class == PolicyClass.USER_FAILURE.value
        ):
            match = _downgraded_l0_failure(match)
        secondary.append(match.to_failure_payload())
        if len(secondary) >= 5:
            break
    return tuple(secondary)


def _primary_episode_chain_lines(bundle: L0Bundle, primary_line: int | None) -> set[int]:
    if primary_line is None:
        return set()
    for episode in bundle.failure_episodes:
        episode_lines = {
            *episode.exception_chain_lines,
            *(confirmation.line for confirmation in episode.cause_confirmations),
        }
        if episode.terminal_exception_line is not None:
            episode_lines.add(episode.terminal_exception_line)
        if primary_line in episode_lines:
            return {line for line in episode_lines if line is not None}
    return set()


def _assessment(
    l2_audit: Mapping[str, Any],
    field: str,
) -> Mapping[str, Any] | None:
    if not l2_audit.get("used"):
        return None
    if field == "model_recovery_assessment" and not l2_audit.get("recovery_assessment_used"):
        return None
    value = l2_audit.get(field)
    return dict(value) if isinstance(value, Mapping) else None


def _coverage_with_history(coverage: Mapping[str, str], available: bool) -> dict[str, str]:
    result = dict(coverage)
    result["history"] = (
        CoverageStatus.FOUND.value if available else CoverageStatus.NOT_AVAILABLE.value
    )
    return result


def _evidence(
    primary: FailureEvidence | None,
    l2_audit: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    if l2_audit.get("used"):
        grounded = l2_audit.get("grounded_evidence")
        if isinstance(grounded, (list, tuple)):
            return tuple(item for item in grounded if isinstance(item, Mapping))
    if primary is None or primary.line is None or primary.quote is None:
        return ()
    return (
        {
            "line": primary.line,
            "quote": primary.quote,
            "supports": ["primary_failure"],
        },
    )


def _justification(
    primary: FailureEvidence | None,
    decision_basis: str,
    l2_audit: Mapping[str, Any],
) -> str:
    if primary is None:
        return "No actionable failure signature was found in the available log."
    if l2_audit.get("used"):
        recovery = l2_audit.get("model_recovery_assessment")
        if isinstance(recovery, Mapping):
            rationale = recovery.get("rationale")
            if isinstance(rationale, str) and rationale.strip():
                return rationale
        root_cause = l2_audit.get("root_cause_assessment")
        if isinstance(root_cause, Mapping):
            summary = root_cause.get("summary")
            if isinstance(summary, str) and summary.strip():
                return summary
    if l2_audit.get("l0_policy_downgraded"):
        return (
            f"L0 matched line {primary.line} as {primary.fine_class}, but semantic "
            "user-failure STOP requires L1 or history; policy basis is "
            f"{decision_basis}."
        )
    return (
        f"Line {primary.line} matched {primary.fine_class} with policy class "
        f"{primary.policy_class}; policy basis is {decision_basis}."
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _downgrade_l0_semantic_stop(
    primary: FailureEvidence | None,
    l2_audit: dict[str, Any],
) -> FailureEvidence | None:
    if primary is None or primary.policy_class != PolicyClass.USER_FAILURE.value:
        return primary
    l2_audit["l0_policy_downgraded"] = True
    l2_audit["l0_policy_class"] = primary.policy_class
    l2_audit["l0_policy_downgrade_reason"] = "semantic_stop_requires_l1_or_history"
    return _downgraded_l0_failure(primary)


def _downgraded_l0_failure(primary: FailureEvidence) -> FailureEvidence:
    return replace(
        primary,
        policy_class=PolicyClass.AMBIGUOUS.value,
    )


def _result_provenance(
    *,
    primary: FailureEvidence | None,
    decision_basis: str,
    l1_configured: bool,
    l1_result: L1EvidenceResult,
    l1_output_health: Mapping[str, Any],
    l2_audit: Mapping[str, Any],
    history: HistorySummary,
    retry_policy: RetryPolicyEvaluation,
) -> dict[str, Any]:
    model_contribution = _model_contribution(
        l1_configured=l1_configured,
        l1_result=l1_result,
        l1_output_health=l1_output_health,
        l2_audit=l2_audit,
    )
    history_contribution = _history_contribution(history, retry_policy)
    evidence_source = _evidence_source(
        primary=primary,
        decision_basis=decision_basis,
        model_used=bool(l2_audit.get("used")),
        history_contribution=history_contribution,
    )
    result_quality = _result_quality(
        primary=primary,
        evidence_source=evidence_source,
        model_contribution=model_contribution,
        l2_audit=l2_audit,
    )
    notes = _result_provenance_notes(
        model_contribution=model_contribution,
        l1_result=l1_result,
        l1_output_health=l1_output_health,
        l2_audit=l2_audit,
    )
    l1_execution = execution_status(
        configured=l1_configured,
        result=l1_result,
        health=l1_output_health,
    )
    return {
        "evidence_source": evidence_source,
        "model_contribution": model_contribution,
        "history_contribution": history_contribution,
        "result_quality": result_quality,
        "nvrx_use": _nvrx_use(result_quality),
        "l1_execution_status": l1_execution["status"],
        "l1_execution_issues": l1_execution["issues"],
        "notes": notes,
    }


def _evidence_source(
    *,
    primary: FailureEvidence | None,
    decision_basis: str,
    model_used: bool,
    history_contribution: str,
) -> str:
    if decision_basis == DecisionBasis.LOG_UNAVAILABLE.value:
        return "fallback_log_unavailable"
    if primary is None and decision_basis == DecisionBasis.MALFORMED_MODEL_OUTPUT.value:
        return "fallback_malformed_model_output"
    if history_contribution in {"retry_budget_exhausted", "observed_advance"}:
        return "history_over_l1" if model_used else "history_over_l0"
    if model_used:
        return "l1_model_audited"
    if primary is not None:
        return "l0_deterministic"
    return "fallback_default_not_proven_user"


def _model_contribution(
    *,
    l1_configured: bool,
    l1_result: L1EvidenceResult,
    l1_output_health: Mapping[str, Any],
    l2_audit: Mapping[str, Any],
) -> str:
    if l2_audit.get("used"):
        if l2_audit.get("audit_status") == "findings":
            return "attempted_used_with_findings"
        if l2_audit.get("audit_status") == "resolved":
            return "attempted_used_with_resolved_findings"
        return "attempted_used"
    if not l1_configured:
        return "not_enabled"
    if not l1_result.model_calls and not l1_result.model:
        return "not_needed_l0"
    if l1_result.anomalies.get("provider_timeout"):
        return "attempted_not_used_timeout"
    if l1_result.anomalies.get("model_output_truncated"):
        return "attempted_not_used_truncated"
    if l1_result.anomalies.get("provider_error"):
        return "attempted_not_used_provider_error"
    if l1_output_health.get("status") in {"contract_invalid", "malformed"}:
        return "attempted_not_used_malformed"
    return "not_needed_l0"


def _history_contribution(
    history: HistorySummary,
    retry_policy: RetryPolicyEvaluation,
) -> str:
    if not history.available:
        return "not_available"
    if retry_policy.decision_basis == DecisionBasis.RETRY_BUDGET_EXHAUSTED.value:
        return "retry_budget_exhausted"
    if retry_policy.observed_advance:
        return "observed_advance"
    return "checked_no_effect"


def _result_quality(
    *,
    primary: FailureEvidence | None,
    evidence_source: str,
    model_contribution: str,
    l2_audit: Mapping[str, Any],
) -> str:
    if evidence_source in {
        "fallback_log_unavailable",
        "fallback_malformed_model_output",
    }:
        return "fallback_only"
    if primary is None:
        return "fallback_only"
    if model_contribution.startswith("attempted_not_used") or (
        model_contribution == "attempted_used_with_findings"
        and _l2_material_finding_count(l2_audit) > 0
    ):
        return "degraded"
    return "normal"


def _l2_material_finding_count(audit: Mapping[str, Any]) -> int:
    return sum(
        1
        for finding in audit.get("findings") or []
        if isinstance(finding, Mapping) and finding.get("policy_material") is True
    )


def _nvrx_use(result_quality: str) -> str:
    if result_quality == "fallback_only":
        return "fallback_to_nvrx_default"
    if result_quality == "degraded":
        return "eligible_degraded"
    return "eligible"


def _result_provenance_notes(
    *,
    model_contribution: str,
    l1_result: L1EvidenceResult,
    l1_output_health: Mapping[str, Any],
    l2_audit: Mapping[str, Any],
) -> list[str]:
    notes: list[str] = []
    if model_contribution.startswith("attempted_not_used"):
        notes.append(model_contribution.removeprefix("attempted_not_used_"))
    if model_contribution == "attempted_used_with_findings":
        notes.append(
            "l2_audit_material_findings"
            if _l2_material_finding_count(l2_audit)
            else "l2_audit_non_material_findings"
        )
    if model_contribution == "attempted_used_with_resolved_findings":
        notes.append("l2_audit_findings_resolved")
    if l2_audit.get("l0_policy_downgraded"):
        notes.append("l0_semantic_stop_downgraded")
    if l1_result.unsupported_tool_requests:
        notes.append("unsupported_tool_request")
    if l1_result.anomalies.get("forced_final_evidence_call"):
        notes.append("forced_final_evidence_call")
    if l1_result.anomalies.get("contract_repair_requested"):
        notes.append("contract_repair_requested")
    if l1_output_health.get("status") == "contract_invalid":
        notes.append("l1_contract_invalid")
    return notes
