# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed L2-L4 decision execution and external result assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .attempt_records import AttemptRecordAssembler
from .causality import build_result_cascades
from .current_failure_facts import build_attempt_failure_facts
from .l1 import L1EvidenceResult, l1_contract_advisories
from .l2 import L2Result
from .l3 import evaluate_cycle_history
from .l4 import L4CyclePolicyInput, L4PathSelection, RetryPolicyEvaluation, evaluate_cycle_policy
from .models import (
    AnalysisExecutionContext,
    AnalysisResult,
    AttemptFailureFacts,
    AttemptFailureFactsSource,
    AttemptRecord,
    CausalRole,
    CoverageStatus,
    CycleHistoryComparison,
    Decision,
    DecisionBasis,
    DecisionEvidence,
    FailureEvidence,
    HistorySummary,
    JobProgressHistory,
    L0Bundle,
    ModelRecoveryAssessment,
    RetryPolicyConfig,
    RouteHistorySummary,
)
from .runtime import SYSTEM_CLOCK, Clock


@dataclass(frozen=True)
class DecisionOutcome:
    """Typed result of failure-fact selection, history, and policy."""

    result: AnalysisResult
    primary: FailureEvidence | None
    l2_primary: FailureEvidence | None
    selected_observed_failure: FailureEvidence | None
    l2_audit: Mapping[str, Any]
    history: HistorySummary
    cycle_history: CycleHistoryComparison
    path_selection: L4PathSelection
    retry_policy: RetryPolicyEvaluation
    attempt_record: AttemptRecord | None
    deterministic_failure_facts: AttemptFailureFacts
    selected_failure_facts: AttemptFailureFacts
    l3_wall_clock_s: float
    l4_wall_clock_s: float


@dataclass(frozen=True)
class _ResolvedFailure:
    audit: Mapping[str, Any]
    l2_primary: FailureEvidence | None
    l2_observation: FailureEvidence | None
    primary_failure_facts: AttemptFailureFacts | None
    observation_failure_facts: AttemptFailureFacts | None


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
    resolved = _resolve_attempt_failure_facts(l2_result)

    deterministic_facts = build_attempt_failure_facts(
        decision_evidence.deterministic_primary_candidate,
        decision_evidence,
        source=AttemptFailureFactsSource.L0_DETERMINISTIC,
        selected_observation=decision_evidence.selected_observed_failure,
    )
    attempt_record = _attempt_record(
        bundle=bundle,
        decision_evidence=decision_evidence,
        execution_context=execution_context,
        primary_failure_facts=resolved.primary_failure_facts,
        observation_failure_facts=resolved.observation_failure_facts,
        route_id=route_id,
    )

    l3_started = clock.monotonic()
    if attempt_record is None:
        cycle_history = _unavailable_cycle_history(
            availability_reason=execution_context.prior_attempts.availability_reason,
            route_id=route_id,
            primary_available=resolved.primary_failure_facts is not None,
            observation_available=resolved.observation_failure_facts is not None,
        )
    else:
        cycle_history = evaluate_cycle_history(
            current_record=attempt_record,
            prior_attempts=execution_context.prior_attempts,
        )
    l3_wall_clock_s = round(clock.monotonic() - l3_started, 3)

    l4_started = clock.monotonic()
    l4_outcome = evaluate_cycle_policy(
        L4CyclePolicyInput(
            deterministic_primary=decision_evidence.deterministic_primary_candidate,
            deterministic_observation=decision_evidence.selected_observed_failure,
            deterministic_facts=deterministic_facts,
            history=cycle_history,
            route_id=route_id,
            grounded_primary=resolved.l2_primary,
            grounded_observation=resolved.l2_observation,
            primary_facts=resolved.primary_failure_facts,
            observation_facts=resolved.observation_failure_facts,
            model_recovery_assessment=_typed_l1_recovery_assessment(
                l1_result,
                selected_evidence_grounded=l2_result.used,
            ),
            l1_primary_declared=_l1_primary_declared(l1_result),
            retry_policy=RetryPolicyConfig.from_mapping(execution_context.retry_policy),
            policy_contexts=execution_context.policy_contexts,
            l1_category_selection=l1_result.category_selection(),
        )
    )
    assert l4_outcome.selected_failure_facts is not None
    assert l4_outcome.selected_history is not None
    assert l4_outcome.path_selection is not None
    primary = l4_outcome.primary
    selected_observed_failure = l4_outcome.selected_observed_failure
    selected_failure_facts = l4_outcome.selected_failure_facts
    history = l4_outcome.selected_history
    path_selection = l4_outcome.path_selection
    retry_policy = l4_outcome.retry_policy
    result_provenance = _candidate_provenance(
        primary=primary,
        selected_observed_failure=selected_observed_failure,
        retry_policy=retry_policy,
        l1_configured=l1_configured,
        l1_result=l1_result,
        l1_output_health=l1_output_health,
        l2_audit=resolved.audit,
        history=history,
        candidate_kind=candidate_kind,
        l1_pending=l1_pending,
        path_selection=path_selection,
    )
    result = _assemble_analysis_result(
        bundle=bundle,
        primary=primary,
        selected_observed_failure=selected_observed_failure,
        retry_policy=retry_policy,
        result_provenance=result_provenance,
        l1_result=l1_result,
        l1_output_health=l1_output_health,
        l2_audit=resolved.audit,
        l2_primary=resolved.l2_primary,
        l2_observation=resolved.l2_observation,
        history=history,
        l1_configured=l1_configured,
        candidate_kind=candidate_kind,
    )
    l4_wall_clock_s = round(clock.monotonic() - l4_started, 3)
    return DecisionOutcome(
        result=result,
        primary=primary,
        l2_primary=resolved.l2_primary,
        selected_observed_failure=selected_observed_failure,
        l2_audit=resolved.audit,
        attempt_record=attempt_record,
        selected_failure_facts=selected_failure_facts,
        history=history,
        cycle_history=cycle_history,
        path_selection=path_selection,
        retry_policy=retry_policy,
        deterministic_failure_facts=deterministic_facts,
        l3_wall_clock_s=l3_wall_clock_s,
        l4_wall_clock_s=l4_wall_clock_s,
    )


def _resolve_attempt_failure_facts(l2_result: L2Result) -> _ResolvedFailure:
    audit = l2_result.to_payload()
    return _ResolvedFailure(
        audit=audit,
        l2_primary=l2_result.primary,
        l2_observation=l2_result.selected_observed_failure,
        primary_failure_facts=l2_result.primary_failure_facts,
        observation_failure_facts=l2_result.observation_failure_facts,
    )


def _attempt_record(
    *,
    bundle: L0Bundle,
    decision_evidence: DecisionEvidence,
    execution_context: AnalysisExecutionContext,
    primary_failure_facts: AttemptFailureFacts | None,
    observation_failure_facts: AttemptFailureFacts | None,
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
    if primary_failure_facts is not None or observation_failure_facts is not None:
        record = assembler.with_enriched(
            record,
            route_id=route_id,
            primary=primary_failure_facts,
            observation=observation_failure_facts,
        )
    return record


def _unavailable_cycle_history(
    *,
    availability_reason: str,
    route_id: str,
    primary_available: bool,
    observation_available: bool,
) -> CycleHistoryComparison:
    unavailable = HistorySummary(
        available=False,
        availability_reason=availability_reason,
        job_history_available=False,
        job_history_availability_reason=availability_reason,
    )
    route = RouteHistorySummary(
        route_id=route_id,
        primary=unavailable if primary_available else None,
        observation=unavailable if observation_available else None,
    )
    return CycleHistoryComparison(
        job_progress=JobProgressHistory(
            available=False,
            availability_reason=availability_reason,
        ),
        deterministic=unavailable,
        routes=(route,),
    )


def _candidate_provenance(
    *,
    primary: FailureEvidence | None,
    selected_observed_failure: FailureEvidence | None,
    retry_policy: RetryPolicyEvaluation,
    l1_configured: bool,
    l1_result: L1EvidenceResult,
    l1_output_health: Mapping[str, Any],
    l2_audit: Mapping[str, Any],
    history: HistorySummary,
    candidate_kind: str,
    l1_pending: bool,
    path_selection: L4PathSelection,
) -> dict[str, Any]:
    provenance = _result_provenance(
        primary=primary,
        selected_observed_failure=selected_observed_failure,
        decision_basis=retry_policy.decision_basis,
        l1_configured=l1_configured,
        l1_result=l1_result,
        l1_output_health=l1_output_health,
        l2_audit=l2_audit,
        history=history,
        retry_policy=retry_policy,
    )
    provenance["candidate_kind"] = candidate_kind
    provenance["selected_evidence_path"] = path_selection.path
    provenance["selected_route_id"] = path_selection.route_id
    provenance["path_selection_reason"] = path_selection.reason
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
    selected_observed_failure: FailureEvidence | None,
    retry_policy: RetryPolicyEvaluation,
    result_provenance: Mapping[str, Any],
    l1_result: L1EvidenceResult,
    l1_output_health: Mapping[str, Any],
    l2_audit: Mapping[str, Any],
    l2_primary: FailureEvidence | None,
    l2_observation: FailureEvidence | None,
    history: HistorySummary,
    l1_configured: bool,
    candidate_kind: str,
) -> AnalysisResult:
    if (
        primary is None
        and selected_observed_failure is None
        and not l1_output_health["usable"]
        and l1_result.model
    ):
        malformed_provenance = _result_provenance(
            primary=None,
            selected_observed_failure=None,
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
            l1_assessment=_l1_assessment(l1_result),
            l2_grounding=_public_l2_grounding(
                l2_audit, primary=l2_primary, selected_observed_failure=l2_observation
            ),
            primary_failure=None,
            observed_failures=(),
            selected_observed_failure=None,
            secondary_failures=(),
            cascades=(),
            evidence_coverage=_coverage_with_history(bundle.evidence_coverage, history.available),
            justification="L1 model evidence was malformed and no L0 primary was available.",
        )
    return AnalysisResult(
        decision=retry_policy.decision,
        decision_basis=retry_policy.decision_basis,
        retry_policy=retry_policy.to_payload(),
        failure_domain=retry_policy.failure_domain,
        result_provenance=result_provenance,
        l1_assessment=_l1_assessment(l1_result),
        l2_grounding=_public_l2_grounding(
            l2_audit,
            primary=l2_primary,
            selected_observed_failure=l2_observation,
        ),
        primary_failure=primary.to_failure_payload() if primary is not None else None,
        observed_failures=(
            ((l2_observation or selected_observed_failure).to_failure_payload(),)
            if l2_observation is not None or selected_observed_failure is not None
            else ()
        ),
        selected_observed_failure=(
            selected_observed_failure.to_failure_payload()
            if selected_observed_failure is not None
            else None
        ),
        secondary_failures=_secondary_failures(
            bundle,
            primary,
            l2_audit=l2_audit,
        ),
        cascades=build_result_cascades(bundle, primary, l2_audit),
        evidence_coverage=_coverage_with_history(bundle.evidence_coverage, history.available),
        justification=_justification(
            primary,
            selected_observed_failure,
            retry_policy.decision_basis,
            l2_audit,
        ),
    )


def _secondary_failures(
    bundle: L0Bundle,
    primary: FailureEvidence | None,
    *,
    l2_audit: Mapping[str, Any],
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
        if match.causal_role in {CausalRole.CASCADE.value, CausalRole.TEARDOWN.value}:
            continue
        if match.line in primary_episode_chain_lines:
            continue
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


def _l1_assessment(l1_result: L1EvidenceResult) -> Mapping[str, Any] | None:
    """Return the exact parsed L1 semantic object without downstream edits."""

    value = l1_result.semantic_payload
    return dict(value) if isinstance(value, Mapping) else None


def _typed_l1_recovery_assessment(
    l1_result: L1EvidenceResult,
    *,
    selected_evidence_grounded: bool,
) -> ModelRecoveryAssessment | None:
    """Return exact typed L1 recovery semantics after mechanical primary grounding."""

    if not selected_evidence_grounded or not isinstance(l1_result.semantic_payload, Mapping):
        return None
    value = l1_result.semantic_payload.get("model_recovery_assessment")
    return ModelRecoveryAssessment.from_mapping(value) if isinstance(value, Mapping) else None


def _l1_primary_declared(l1_result: L1EvidenceResult) -> bool:
    payload = l1_result.semantic_payload
    return isinstance(payload, Mapping) and isinstance(payload.get("primary_failure"), Mapping)


def _public_l2_grounding(
    l2_audit: Mapping[str, Any],
    *,
    primary: FailureEvidence | None,
    selected_observed_failure: FailureEvidence | None,
) -> Mapping[str, Any]:
    """Project compact L2 output without repeating or rewriting L1 semantics."""

    used = bool(l2_audit.get("used"))
    related = l2_audit.get("audited_related_failures")
    grounded_evidence = l2_audit.get("grounded_evidence")
    adjustments = l2_audit.get("grounding_adjustments")
    findings = l2_audit.get("findings")
    enriched_tracks = dict(l2_audit.get("enriched_failure_tracks") or {})
    primary_track = enriched_tracks.get("primary")
    observation_track = enriched_tracks.get("observation")
    observation_grounding = l2_audit.get("observation_grounding")
    observation_identity_grounding = (
        observation_grounding.get("failure_identity_grounding")
        if isinstance(observation_grounding, Mapping)
        else None
    )
    if isinstance(observation_grounding, Mapping):
        primary_findings = [dict(item) for item in findings or () if isinstance(item, Mapping)]
        observation_findings = [
            dict(item)
            for item in observation_grounding.get("findings") or ()
            if isinstance(item, Mapping)
        ]
    else:
        primary_findings = []
        observation_findings = [dict(item) for item in findings or () if isinstance(item, Mapping)]
    all_findings = [
        {**item, "track": track}
        for track, track_items in (
            ("primary", primary_findings),
            ("observation", observation_findings),
        )
        for item in track_items
    ]
    return {
        "used": used,
        "track_grounding": dict(l2_audit.get("track_grounding") or {}),
        "enriched_failure_tracks": enriched_tracks,
        "grounding_status": str(l2_audit.get("grounding_status") or "not_run"),
        "audit_status": str(l2_audit.get("audit_status") or "not_run"),
        "not_run_reason": l2_audit.get("not_run_reason"),
        "grounded_primary_failure": (
            primary.to_failure_payload() if used and primary is not None else None
        ),
        "grounded_observed_failures": (
            [selected_observed_failure.to_failure_payload()]
            if used and selected_observed_failure is not None
            else []
        ),
        "grounded_selected_observation": (
            selected_observed_failure.to_failure_payload()
            if used and selected_observed_failure is not None
            else None
        ),
        "grounded_related_failures": [
            dict(item) for item in related or () if isinstance(item, Mapping)
        ],
        "grounded_evidence": [
            dict(item) for item in grounded_evidence or () if isinstance(item, Mapping)
        ],
        "audit_influence": "observational_only",
        "grounded_failure_identities": {
            "primary": _grounded_failure_identity_projection(
                l2_audit.get("failure_identity_grounding"),
                published=primary_track is not None,
            ),
            "observation": _grounded_failure_identity_projection(
                observation_identity_grounding,
                published=observation_track is not None,
            ),
        },
        "affected_entity_selection": (
            dict(l2_audit["affected_entity_selection"])
            if primary_track is not None
            and isinstance(l2_audit.get("affected_entity_selection"), Mapping)
            else None
        ),
        "history_identities": {
            "primary": _history_identity_projection(primary_track),
            "observation": _history_identity_projection(observation_track),
        },
        "grounding_adjustments": [
            dict(item) for item in adjustments or () if isinstance(item, Mapping)
        ],
        "track_findings": {
            "primary": primary_findings,
            "observation": observation_findings,
        },
        "findings": all_findings,
    }


def _grounded_failure_identity_projection(
    value: Any,
    *,
    published: bool,
) -> Mapping[str, Any] | None:
    if not published or not isinstance(value, Mapping):
        return None
    return {
        str(field): dict(grounding)
        for field, grounding in value.items()
        if isinstance(grounding, Mapping)
    }


def _history_identity_projection(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        "ready": bool(value.get("history_identity_ready")),
        "identity_kind": value.get("identity_kind"),
        "anchor_line": value.get("identity_anchor_line"),
        "anchor_reason": value.get("identity_anchor_reason"),
        "root_fingerprint": value.get("root_fingerprint"),
        "root_fingerprint_source": value.get("root_fingerprint_source"),
        "observation_fingerprint": value.get("observation_fingerprint"),
        "observation_fingerprint_source": value.get("observation_fingerprint_source"),
    }


def _coverage_with_history(coverage: Mapping[str, str], available: bool) -> dict[str, str]:
    result = dict(coverage)
    result["history"] = (
        CoverageStatus.FOUND.value if available else CoverageStatus.NOT_AVAILABLE.value
    )
    return result


def _justification(
    primary: FailureEvidence | None,
    selected_observed_failure: FailureEvidence | None,
    decision_basis: str,
    l2_audit: Mapping[str, Any],
) -> str:
    if primary is None:
        if selected_observed_failure is not None:
            return (
                f"Line {selected_observed_failure.line} is a grounded terminal failure "
                f"surface without an identified primary; L4 policy basis is {decision_basis}."
            )
        return "No actionable failure signature was found in the available log."
    if l2_audit.get("used"):
        return (
            f"Line {primary.line} is the L2-grounded primary failure; "
            f"L4 policy basis is {decision_basis}."
        )
    return (
        f"Line {primary.line} matched failure class {primary.failure_class}; "
        f"policy basis is {decision_basis}."
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


def _result_provenance(
    *,
    primary: FailureEvidence | None,
    selected_observed_failure: FailureEvidence | None,
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
        selected_observed_failure=selected_observed_failure,
        decision_basis=decision_basis,
        model_used=bool(l2_audit.get("used")),
        history_contribution=history_contribution,
    )
    result_quality = _result_quality(
        primary=primary,
        selected_observed_failure=selected_observed_failure,
        evidence_source=evidence_source,
        model_contribution=model_contribution,
    )
    notes = _result_provenance_notes(
        model_contribution=model_contribution,
        l1_result=l1_result,
        l1_output_health=l1_output_health,
    )
    l1_execution = l1_output_health.get("execution_assessment") or {}
    l1_result_quality = l1_execution.get("result_quality")
    if l1_result_quality == "not_applicable":
        l1_execution_status = "not_run"
    elif l1_result_quality == "usable":
        l1_execution_status = "ok"
    elif l1_result_quality == "degraded":
        l1_execution_status = "degraded"
    else:
        l1_execution_status = "failed"
    return {
        "evidence_source": evidence_source,
        "model_contribution": model_contribution,
        "history_contribution": history_contribution,
        "result_quality": result_quality,
        "nvrx_use": _nvrx_use(result_quality),
        "l1_execution_status": l1_execution_status,
        "l1_execution_issues": list(l1_execution.get("reason_codes") or []),
        "notes": notes,
    }


def _evidence_source(
    *,
    primary: FailureEvidence | None,
    selected_observed_failure: FailureEvidence | None,
    decision_basis: str,
    model_used: bool,
    history_contribution: str,
) -> str:
    if decision_basis == DecisionBasis.LOG_UNAVAILABLE.value:
        return "log_unavailable"
    if primary is None and decision_basis == DecisionBasis.MALFORMED_MODEL_OUTPUT.value:
        return "malformed_model_output"
    if history_contribution in {"retry_budget_exhausted", "observed_advance"}:
        return "history_over_l1" if model_used else "history_over_l0"
    if model_used:
        return (
            "l1_model_grounded_observation"
            if primary is None and selected_observed_failure is not None
            else "l1_model_grounded"
        )
    if primary is not None:
        return "l0_deterministic"
    if selected_observed_failure is not None:
        return "l0_observation_only"
    return "no_primary"


def _model_contribution(
    *,
    l1_configured: bool,
    l1_result: L1EvidenceResult,
    l1_output_health: Mapping[str, Any],
    l2_audit: Mapping[str, Any],
) -> str:
    if l2_audit.get("used"):
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
    if l1_output_health.get("status") == "contract_invalid":
        return "attempted_not_used_contract_invalid"
    if l1_output_health.get("status") == "malformed":
        return "attempted_not_used_malformed"
    if l2_audit.get("grounding_status") == "unavailable":
        return "attempted_not_used_ungrounded"
    return "not_needed_l0"


def _history_contribution(
    history: HistorySummary,
    retry_policy: RetryPolicyEvaluation,
) -> str:
    if not history.available and not history.job_history_available:
        return "not_available"
    if retry_policy.retry_budget_exhausted:
        return "retry_budget_exhausted"
    if any(
        ledger is not None and ledger.applicable and ledger.observed_advance
        for ledger in (
            retry_policy.general_root_ceiling,
            retry_policy.selected_policy_ledger,
            retry_policy.job_no_progress_guard,
            retry_policy.job_unknown_progress_guard,
        )
    ):
        return "observed_advance"
    return "checked_no_effect"


def _result_quality(
    *,
    primary: FailureEvidence | None,
    selected_observed_failure: FailureEvidence | None,
    evidence_source: str,
    model_contribution: str,
) -> str:
    if evidence_source in {
        "log_unavailable",
        "malformed_model_output",
    }:
        return "unusable"
    if primary is None and selected_observed_failure is None:
        return "unusable"
    if model_contribution.startswith("attempted_not_used"):
        return "degraded"
    return "normal"


def _nvrx_use(result_quality: str) -> str:
    if result_quality == "unusable":
        return "fallback_to_nvrx_default"
    if result_quality == "degraded":
        return "eligible_degraded"
    return "eligible"


def _result_provenance_notes(
    *,
    model_contribution: str,
    l1_result: L1EvidenceResult,
    l1_output_health: Mapping[str, Any],
) -> list[str]:
    notes: list[str] = []
    if model_contribution.startswith("attempted_not_used"):
        notes.append(model_contribution.removeprefix("attempted_not_used_"))
    if l1_result.unsupported_tool_requests:
        notes.append("unsupported_tool_request")
    if l1_result.anomalies.get("final_evidence_turn"):
        notes.append(str(l1_result.anomalies.get("final_evidence_reason") or "final_evidence_turn"))
    if l1_output_health.get("status") == "contract_invalid":
        notes.append("l1_contract_invalid")
    notes.extend(
        str(item.get("code")) for item in l1_contract_advisories(l1_result) if item.get("code")
    )
    return notes
