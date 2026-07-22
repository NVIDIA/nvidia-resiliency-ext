# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace construction and stage observability for restart-agent runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping

from ..l1.contracts import L1EvidenceResult
from ..l4.policy import RetryPolicyEvaluation
from ..models import (
    AnalysisExecutionContext,
    AnalysisResult,
    AttemptFailureFacts,
    AttemptFailureFactsSource,
    DecisionCandidate,
    DecisionCandidateKind,
    DecisionEvidence,
    FailureEvidence,
    HistorySummary,
    L0Bundle,
    L0ModelFacingView,
)


def _assessment(audit: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    value = audit.get(name)
    return dict(value) if isinstance(value, Mapping) else None


def _l2_material_finding_count(audit: Mapping[str, Any]) -> int:
    return sum(
        1
        for finding in audit.get("findings") or []
        if isinstance(finding, Mapping) and bool(finding.get("policy_material", True))
    )


@dataclass(frozen=True)
class DecisionTraceInputs:
    """Typed inputs required to render one completed decision trace."""

    execution_context: AnalysisExecutionContext
    result: AnalysisResult
    bundle: L0Bundle
    decision_evidence: DecisionEvidence
    primary: FailureEvidence | None
    l2_primary: FailureEvidence | None
    total_wall_clock_s: float
    l0_wall_clock_s: float
    l0a_wall_clock_s: float
    decision_evidence_wall_clock_s: float
    l0b_wall_clock_s: float
    model_view: L0ModelFacingView | None
    l1_result: L1EvidenceResult
    l1_output_health: Mapping[str, Any]
    l1_wall_clock_s: float
    l2_audit: Mapping[str, Any]
    selected_failure_facts: AttemptFailureFacts
    history: Any
    retry_policy: RetryPolicyEvaluation
    l0_reused: bool
    l2_wall_clock_s: float
    l3_wall_clock_s: float
    l4_wall_clock_s: float
    fallback_candidate: DecisionCandidate | None
    selected_candidate_kind: str
    fallback_callback_error: str | None
    fallback_callback_wall_clock_s: float
    analysis_timeout_seconds: float


@dataclass(frozen=True)
class _TraceMetrics:
    l1_model_wall_clock_s: float
    failed_model_calls: tuple[Mapping[str, Any], ...]
    retried_model_calls: tuple[Mapping[str, Any], ...]
    l1_tool_wall_clock_s: float
    token_usage: Mapping[str, Any]
    token_limit: Mapping[str, Any]
    l0_root_fingerprint: Any
    l0_root_fingerprint_source: Any
    l2_root_fingerprint: Any
    l2_root_fingerprint_source: Any
    l2_matches_l0_fingerprint: bool | None

    @classmethod
    def from_inputs(cls, inputs: DecisionTraceInputs) -> "_TraceMetrics":
        failed_model_calls = tuple(
            call for call in inputs.l1_result.model_calls if call.get("success") is False
        )
        l0_identity = inputs.decision_evidence.canonical_observed_identity
        l0_root_fingerprint = l0_identity.get("root_fingerprint")
        l2_root_fingerprint = inputs.l2_audit.get("stable_root_fingerprint")
        return cls(
            l1_model_wall_clock_s=round(
                sum(float(call.get("latency_s") or 0.0) for call in inputs.l1_result.model_calls),
                3,
            ),
            failed_model_calls=failed_model_calls,
            retried_model_calls=tuple(
                call for call in failed_model_calls if call.get("retry_scheduled") is True
            ),
            l1_tool_wall_clock_s=round(
                sum(float(call.get("latency_ms") or 0.0) for call in inputs.l1_result.tool_calls)
                / 1000,
                3,
            ),
            token_usage=_l1_token_usage(inputs.l1_result.model_calls),
            token_limit=l1_token_limit_summary(inputs.l1_result.model_calls),
            l0_root_fingerprint=l0_root_fingerprint,
            l0_root_fingerprint_source=l0_identity.get("root_fingerprint_source"),
            l2_root_fingerprint=l2_root_fingerprint,
            l2_root_fingerprint_source=inputs.l2_audit.get("root_fingerprint_source"),
            l2_matches_l0_fingerprint=(
                l2_root_fingerprint == l0_root_fingerprint
                if l2_root_fingerprint and l0_root_fingerprint
                else None
            ),
        )


def build_decision_trace(inputs: DecisionTraceInputs) -> dict[str, Any]:
    metrics = _TraceMetrics.from_inputs(inputs)
    return {
        "context_mode": _context_mode(inputs.primary, inputs.l1_result, inputs.l2_audit),
        "log_path": inputs.execution_context.log_path,
        "request": inputs.execution_context.request.to_payload(),
        "job_id": inputs.execution_context.job_id,
        "cycle_id": inputs.execution_context.cycle_id,
        "result_provenance": dict(inputs.result.result_provenance),
        "external_output": inputs.result.to_payload(),
        "decision_candidates": _decision_candidates_trace(
            result=inputs.result,
            history=inputs.history,
            fallback_candidate=inputs.fallback_candidate,
            selected_candidate_kind=inputs.selected_candidate_kind,
            l1_output_health=inputs.l1_output_health,
            total_wall_clock_s=inputs.total_wall_clock_s,
            l1_configured=bool(inputs.l1_result.model or inputs.fallback_candidate is not None),
        ),
        "timing": _timing_trace(inputs, metrics),
        "latency_measurement": {
            "mode": "terminal_request_to_result",
            "terminal_total_wall_clock_s": inputs.total_wall_clock_s,
            "post_progressive_end_wall_clock_s": None,
            "progressive_decision_window_hit": None,
            "production_gate_measured": False,
        },
        "analysis_deadline": {
            "timeout_seconds": inputs.analysis_timeout_seconds,
            "deadline_exceeded": bool(inputs.l1_result.anomalies.get("deadline_exceeded")),
            "remaining_at_return_s": max(
                0.0, inputs.analysis_timeout_seconds - inputs.total_wall_clock_s
            ),
        },
        "token_usage": metrics.token_usage,
        "token_limit": metrics.token_limit,
        "layers": _layers_trace(inputs, metrics),
        "l0_summary": _l0_summary_trace(inputs),
        "decision_evidence": inputs.decision_evidence.to_payload(),
        "selected_failure_facts": inputs.selected_failure_facts.to_payload(),
        "l0_model_view": (
            inputs.model_view.to_payload() if inputs.model_view is not None else None
        ),
        "l2_grounded_semantics": _l2_grounded_semantics(inputs.l2_primary, inputs.l2_audit),
        "l2_audit": _l2_audit_trace(inputs.l2_audit, inputs.l1_result),
        "l1": inputs.l1_result.to_trace(),
        "selection_summary": dict(inputs.bundle.selection_summary),
        "l3_history": history_trace(inputs.history),
        "l4_policy": {
            "decision": inputs.result.decision,
            "decision_basis": inputs.result.decision_basis,
            "retry_policy": inputs.retry_policy.to_payload(),
            "history_inputs": history_trace(inputs.history),
            "result_provenance": dict(inputs.result.result_provenance),
        },
        "anomalies": _anomalies_trace(inputs, metrics),
    }


def _timing_trace(inputs: DecisionTraceInputs, metrics: _TraceMetrics) -> dict[str, Any]:
    return {
        "total_wall_clock_s": inputs.total_wall_clock_s,
        "l0_wall_clock_s": inputs.l0_wall_clock_s,
        "l0a_wall_clock_s": inputs.l0a_wall_clock_s,
        "decision_evidence_wall_clock_s": inputs.decision_evidence_wall_clock_s,
        "l0b_wall_clock_s": inputs.l0b_wall_clock_s,
        "l1_wall_clock_s": round(inputs.l1_wall_clock_s, 3),
        "l1_model_call_wall_clock_s": metrics.l1_model_wall_clock_s,
        "l1_tool_wall_clock_s": metrics.l1_tool_wall_clock_s,
        "l2_wall_clock_s": inputs.l2_wall_clock_s,
        "l3_wall_clock_s": inputs.l3_wall_clock_s,
        "l4_wall_clock_s": inputs.l4_wall_clock_s,
        "l1_model_calls": len(inputs.l1_result.model_calls),
        "l1_failed_model_calls": len(metrics.failed_model_calls),
        "l1_retried_model_calls": len(metrics.retried_model_calls),
        "l1_tool_calls": len(inputs.l1_result.tool_calls),
        "l0_reused": inputs.l0_reused,
        "fallback_callback_wall_clock_s": inputs.fallback_callback_wall_clock_s,
    }


def _layers_trace(inputs: DecisionTraceInputs, metrics: _TraceMetrics) -> dict[str, Any]:
    return {
        "L0": _l0_layer_trace(inputs, metrics),
        "L1": _l1_layer_trace(inputs, metrics),
        "L2": _l2_layer_trace(inputs, metrics),
        "L3": _l3_layer_trace(inputs),
        "L4": _l4_layer_trace(inputs),
    }


def _l0_layer_trace(inputs: DecisionTraceInputs, metrics: _TraceMetrics) -> dict[str, Any]:
    bundle = inputs.bundle
    identity_ready = bool(metrics.l0_root_fingerprint)
    return {
        "name": "evidence_assembly_and_projection",
        "wall_clock_s": inputs.l0_wall_clock_s,
        "reused": inputs.l0_reused,
        "model_calls": 0,
        "tool_calls": 0,
        "line_count": bundle.line_count,
        "byte_size": bundle.byte_size,
        "context_windows": len(bundle.context_windows),
        "candidate_anchors": len(bundle.candidate_anchors),
        "failure_episodes": len(bundle.failure_episodes),
        "distributed_failure_incidents": len(bundle.distributed_failure_incidents),
        "seconds_from_last_progress_to_terminal_incident": bundle.run_progress_summary.seconds_from_last_progress_to_terminal_incident,
        "configured_terminal_timeout_seconds": bundle.run_progress_summary.configured_terminal_timeout_seconds,
        "terminal_detection_lag_seconds": bundle.run_progress_summary.terminal_detection_lag_seconds,
        "occurrence_groups": len(bundle.occurrence_groups),
        "root_fingerprint_owner": "L0",
        "root_fingerprint": metrics.l0_root_fingerprint,
        "root_fingerprint_source": metrics.l0_root_fingerprint_source,
        "root_fingerprint_available": identity_ready,
        "history_identity_ready": identity_ready,
        "sub_stages": {
            "L0A": {
                "name": "complete_evidence_assembly",
                "wall_clock_s": inputs.l0a_wall_clock_s,
                "status": "reused" if inputs.l0_reused else "completed",
                "line_count": bundle.line_count,
                "byte_size": bundle.byte_size,
                "context_windows": len(bundle.context_windows),
                "candidate_anchors": len(bundle.candidate_anchors),
                "failure_episodes": len(bundle.failure_episodes),
                "distributed_failure_incidents": len(bundle.distributed_failure_incidents),
                "occurrence_groups": len(bundle.occurrence_groups),
            },
            "DecisionEvidence": {
                "name": "canonical_decision_evidence_selection",
                "wall_clock_s": inputs.decision_evidence_wall_clock_s,
                "status": "completed",
                "schema_version": inputs.decision_evidence.schema_version,
                "primary_available": (
                    inputs.decision_evidence.deterministic_primary_candidate is not None
                ),
                "identity_available": bool(
                    inputs.decision_evidence.canonical_observed_identity.get("available")
                ),
                "root_fingerprint_owner": "L0",
                "root_fingerprint": metrics.l0_root_fingerprint,
                "root_fingerprint_source": metrics.l0_root_fingerprint_source,
                "root_fingerprint_available": identity_ready,
                "history_identity_ready": identity_ready,
                "referenced_source_lines": len(
                    inputs.decision_evidence.selected_evidence_references.get("source_lines", ())
                ),
            },
            "L0B": {
                "name": "initial_model_evidence_view",
                "wall_clock_s": inputs.l0b_wall_clock_s,
                "status": "completed" if inputs.model_view is not None else "not_run",
                **(
                    dict(inputs.model_view.projection_metrics)
                    if inputs.model_view is not None
                    else {}
                ),
            },
        },
    }


def _l1_layer_trace(inputs: DecisionTraceInputs, metrics: _TraceMetrics) -> dict[str, Any]:
    return {
        "name": "semantic_analysis",
        "wall_clock_s": round(inputs.l1_wall_clock_s, 3),
        "model_call_wall_clock_s": metrics.l1_model_wall_clock_s,
        "tool_wall_clock_s": metrics.l1_tool_wall_clock_s,
        "model_calls": len(inputs.l1_result.model_calls),
        "failed_model_calls": len(metrics.failed_model_calls),
        "retried_model_calls": len(metrics.retried_model_calls),
        "tool_calls": len(inputs.l1_result.tool_calls),
        "provider_response_returned": bool(inputs.l1_result.model_calls),
        "response_parsed": bool(inputs.l1_result.success),
        "output_status": inputs.l1_output_health.get("status"),
        "output_usable": bool(inputs.l1_output_health.get("usable")),
        "output_errors": list(inputs.l1_output_health.get("errors") or []),
        "execution_status": inputs.result.result_provenance.get("l1_execution_status"),
        "execution_issues": list(inputs.result.result_provenance.get("l1_execution_issues") or []),
        "prompt_tokens": metrics.token_usage["prompt_tokens"],
        "completion_tokens": metrics.token_usage["completion_tokens"],
        "total_tokens": metrics.token_usage["total_tokens"],
    }


def _l2_layer_trace(inputs: DecisionTraceInputs, metrics: _TraceMetrics) -> dict[str, Any]:
    audit = inputs.l2_audit
    citations = audit.get("citation_audits") or []
    identity_ready = bool(metrics.l2_root_fingerprint)
    return {
        "name": "evidence_grounding_and_identity",
        "wall_clock_s": inputs.l2_wall_clock_s,
        "grounding_status": audit.get("grounding_status"),
        "grounding_method": audit.get("grounding_method"),
        "enriched_failure_facts_available": bool(
            inputs.selected_failure_facts.source == AttemptFailureFactsSource.L2_GROUNDED
        ),
        "audit_status": audit.get("audit_status"),
        "primary_available": bool(audit.get("primary_used")),
        "recovery_assessment_available": bool(audit.get("recovery_assessment_used")),
        "recovery_assessment_policy_grounded": bool(
            audit.get("recovery_assessment_policy_grounded")
        ),
        "related_failures_audited": len(audit.get("audited_related_failure_roles") or []),
        "finding_count": sum(len(items) for items in (audit.get("field_findings") or {}).values()),
        "material_finding_count": _l2_material_finding_count(audit),
        "finding_severity_counts": _l2_finding_severity_counts(audit),
        "citation_count": len(citations),
        "nearby_resolved_count": sum(
            1 for item in citations if item.get("status") == "nearby_resolved"
        ),
        "rendered_exact_count": sum(
            1 for item in citations if item.get("status") == "rendered_exact"
        ),
        "abbreviated_exact_count": sum(
            1 for item in citations if item.get("status") == "abbreviated_exact"
        ),
        "grounding_adjustment_count": len(audit.get("grounding_adjustments") or []),
        "root_fingerprint_owner": "L2",
        "root_fingerprint": metrics.l2_root_fingerprint,
        "root_fingerprint_source": metrics.l2_root_fingerprint_source,
        "root_fingerprint_available": identity_ready,
        "history_identity_ready": identity_ready,
        "matches_l0_root_fingerprint": metrics.l2_matches_l0_fingerprint,
    }


def _l3_layer_trace(inputs: DecisionTraceInputs) -> dict[str, Any]:
    history = inputs.history
    return {
        "name": "history_enrichment",
        "wall_clock_s": inputs.l3_wall_clock_s,
        "selected_failure_facts_source": inputs.selected_failure_facts.source.value,
        "history_identity_ready": inputs.selected_failure_facts.history_identity_ready,
        "history_available": bool(getattr(history, "available", False)),
        "same_job_attempts": int(getattr(history, "same_job_attempts", 0) or 0),
        "matching_root_attempts": int(getattr(history, "matching_root_attempts", 0) or 0),
        "observed_advance_attempts": int(getattr(history, "observed_advance_attempts", 0) or 0),
        "no_observed_advance_attempts": int(
            getattr(history, "no_observed_advance_attempts", 0) or 0
        ),
        "unknown_progress_attempts": int(getattr(history, "unknown_progress_attempts", 0) or 0),
        "exact_failure_position_attempts": int(
            getattr(history, "exact_failure_position_attempts", 0) or 0
        ),
        "same_rank_iteration_attempts": int(
            getattr(history, "same_rank_iteration_attempts", 0) or 0
        ),
        "same_data_position_attempts": int(getattr(history, "same_data_position_attempts", 0) or 0),
        "same_artifact_attempts": int(getattr(history, "same_artifact_attempts", 0) or 0),
        "consecutive_same_root_no_advance_attempts": int(
            getattr(history, "consecutive_same_root_no_advance_attempts", 0) or 0
        ),
        "advanced_beyond_all_comparable_attempts": bool(
            getattr(history, "advanced_beyond_all_comparable_attempts", False)
        ),
    }


def _l4_layer_trace(inputs: DecisionTraceInputs) -> dict[str, Any]:
    return {
        "name": "policy_decision",
        "wall_clock_s": inputs.l4_wall_clock_s,
        "decision": inputs.result.decision,
        "decision_basis": inputs.result.decision_basis,
        "rule": inputs.retry_policy.rule,
        "allowed_retries": inputs.retry_policy.allowed_retries,
        "matching_prior_failures": inputs.retry_policy.matching_prior_failures,
        "retry_budget_exhausted": inputs.retry_policy.retry_budget_exhausted,
        "recovery_assessment_policy_grounded": (
            inputs.retry_policy.recovery_assessment_policy_grounded
        ),
        "current_evidence_qualified": inputs.retry_policy.current_evidence_qualified,
        "observed_advance": inputs.retry_policy.observed_advance,
        "result_quality": inputs.result.result_provenance.get("result_quality"),
    }


def _l0_summary_trace(inputs: DecisionTraceInputs) -> dict[str, Any]:
    bundle = inputs.bundle
    return {
        "line_count": bundle.line_count,
        "byte_size": bundle.byte_size,
        "deterministic_primary_candidate": (
            bundle.deterministic_primary_candidate.to_failure_payload()
            if bundle.deterministic_primary_candidate is not None
            else None
        ),
        "occurrence_groups": len(bundle.occurrence_groups),
        "registry_matches": len(bundle.registry_matches),
        "context_windows": len(bundle.context_windows),
        "candidate_anchors": len(bundle.candidate_anchors),
        "distributed_failure_incidents": len(bundle.distributed_failure_incidents),
        "first_terminal_incident_line": (bundle.run_progress_summary.first_terminal_incident_line),
        "seconds_from_last_progress_to_terminal_incident": (
            bundle.run_progress_summary.seconds_from_last_progress_to_terminal_incident
        ),
        "terminal_detection_lag_seconds": (
            bundle.run_progress_summary.terminal_detection_lag_seconds
        ),
        "model_view_projection": (
            dict(inputs.model_view.projection_metrics) if inputs.model_view is not None else None
        ),
    }


def _anomalies_trace(inputs: DecisionTraceInputs, metrics: _TraceMetrics) -> dict[str, Any]:
    return {
        **dict(inputs.bundle.anomalies),
        "provider_error": bool(inputs.l1_result.anomalies.get("provider_error")),
        "provider_error_type": inputs.l1_result.anomalies.get("provider_error_type"),
        "provider_timeout": bool(inputs.l1_result.anomalies.get("provider_timeout")),
        "context_window_exceeded": bool(inputs.l1_result.anomalies.get("context_window_exceeded")),
        "provider_retries": len(metrics.retried_model_calls),
        "token_limit_hit": metrics.token_limit["hit"],
        "contract_repair_requested": bool(
            inputs.l1_result.anomalies.get("contract_repair_requested")
        ),
        "l1_contract_invalid": inputs.l1_output_health.get("status") == "contract_invalid",
        "malformed_model_evidence": inputs.l1_output_health.get("status") == "malformed",
        "model_output_truncated": bool(inputs.l1_result.anomalies.get("model_output_truncated")),
        "model_prohibited_fields_ignored": bool(inputs.l2_audit.get("ignored_prohibited_fields")),
        "l0_policy_downgraded": bool(inputs.l2_audit.get("l0_policy_downgraded")),
        "unsupported_tool_request_seen": bool(inputs.l1_result.unsupported_tool_requests),
        "forced_final_evidence_call": bool(
            inputs.l1_result.anomalies.get("forced_final_evidence_call")
        ),
        "fallback_callback_error": inputs.fallback_callback_error,
    }


def _decision_candidates_trace(
    *,
    result: AnalysisResult,
    history: Any,
    fallback_candidate: DecisionCandidate | None,
    selected_candidate_kind: str,
    l1_output_health: Mapping[str, Any],
    total_wall_clock_s: float,
    l1_configured: bool,
) -> dict[str, Any]:
    fallback_payload = fallback_candidate.to_payload() if fallback_candidate is not None else None
    if fallback_payload is None and selected_candidate_kind == (
        DecisionCandidateKind.DETERMINISTIC_FALLBACK.value
    ):
        fallback_payload = DecisionCandidate(
            candidate_kind=DecisionCandidateKind.DETERMINISTIC_FALLBACK.value,
            result=result,
            ready_wall_clock_s=total_wall_clock_s,
            l1_execution_status=str(
                result.result_provenance.get("l1_execution_status") or "not_run"
            ),
            history_summary=history_trace(history),
        ).to_payload()

    enriched_ready = selected_candidate_kind == DecisionCandidateKind.L1_ENRICHED.value
    enriched_payload = None
    if enriched_ready:
        enriched_payload = DecisionCandidate(
            candidate_kind=DecisionCandidateKind.L1_ENRICHED.value,
            result=result,
            ready_wall_clock_s=total_wall_clock_s,
            l1_execution_status=str(result.result_provenance.get("l1_execution_status") or "ok"),
            history_summary=history_trace(history),
        ).to_payload()

    selected_ready_wall_clock_s = total_wall_clock_s
    if fallback_candidate is not None and not enriched_ready:
        selected_ready_wall_clock_s = fallback_candidate.ready_wall_clock_s
    best_available = DecisionCandidate(
        candidate_kind=selected_candidate_kind,
        result=result,
        ready_wall_clock_s=selected_ready_wall_clock_s,
        l1_execution_status=str(result.result_provenance.get("l1_execution_status") or "not_run"),
        history_summary=history_trace(history),
    ).to_payload()

    if enriched_ready:
        selection_reason = "structurally_usable_l1_with_l2_grounding_audit"
    elif not l1_configured:
        selection_reason = "l1_not_configured"
    elif l1_output_health.get("usable"):
        selection_reason = "l1_returned_no_audited_primary"
    else:
        selection_reason = f"l1_not_usable:{l1_output_health.get('status')}"

    return {
        "analysis_state": "completed",
        "fallback_ready": fallback_payload is not None,
        "enriched_ready": enriched_ready,
        "best_available_kind": selected_candidate_kind,
        "best_available": best_available,
        "selected": selected_candidate_kind,
        "selection_reason": selection_reason,
        "deterministic_fallback": fallback_payload,
        "l1_enriched": enriched_payload,
    }


def _l2_audit_trace(
    audit: Mapping[str, Any],
    l1_result: L1EvidenceResult,
) -> dict[str, Any]:
    result = dict(audit)
    grounding_adjustments = [
        item for item in audit.get("grounding_adjustments") or [] if isinstance(item, Mapping)
    ]
    adjusted_fields = {str(item.get("field")) for item in grounding_adjustments}
    field_findings = audit.get("field_findings") or {}
    if not isinstance(field_findings, Mapping):
        field_findings = {}
    field_finding_codes = audit.get("field_finding_codes") or {}
    if not isinstance(field_finding_codes, Mapping):
        field_finding_codes = {}

    primary_status = "not_evaluated"
    if l1_result.success:
        primary_status = "available" if audit.get("primary_used") else "not_evaluated"
        if any(field.startswith("primary_failure.") for field in adjusted_fields):
            primary_status = "resolved"
        if field_findings.get("primary_failure") and audit.get("primary_used"):
            primary_status = "findings"

    recovery_status = "not_evaluated"
    if audit.get("primary_used"):
        recovery_status = "available" if audit.get("recovery_assessment_used") else "not_evaluated"
        if any("recovery" in field for field in adjusted_fields):
            recovery_status = "resolved"
        if field_findings.get("model_recovery_assessment"):
            recovery_status = "findings"

    related_findings = field_findings.get("related_failures") or []
    audited_related_count = len(audit.get("audited_related_failure_roles") or [])
    related_status = "not_evaluated"
    if audit.get("primary_used"):
        related_status = "audited"
        if related_findings:
            related_status = "findings"

    result["field_audits"] = {
        "primary_failure": {
            "status": primary_status,
            "findings": list(field_findings.get("primary_failure") or []),
            "finding_classes": list(field_finding_codes.get("primary_failure") or []),
        },
        "root_cause_assessment": {
            "status": (
                "findings"
                if field_findings.get("root_cause_assessment")
                else "available" if audit.get("primary_used") else "not_evaluated"
            ),
            "findings": list(field_findings.get("root_cause_assessment") or []),
            "finding_classes": list(field_finding_codes.get("root_cause_assessment") or []),
        },
        "model_recovery_assessment": {
            "status": recovery_status,
            "findings": list(field_findings.get("model_recovery_assessment") or []),
            "finding_classes": list(field_finding_codes.get("model_recovery_assessment") or []),
        },
        "related_failures": {
            "status": related_status,
            "audited_count": audited_related_count,
            "finding_count": len(related_findings),
            "findings": list(related_findings),
            "finding_classes": list(field_finding_codes.get("related_failures") or []),
        },
    }
    return result


def _l2_finding_severity_counts(audit: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in audit.get("findings") or []:
        if not isinstance(finding, Mapping):
            continue
        severity = str(finding.get("severity") or "credibility")
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def _l2_grounded_semantics(
    primary: FailureEvidence | None,
    audit: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not audit.get("used") or primary is None:
        return None
    return {
        "grounding_status": audit.get("grounding_status"),
        "grounding_method": audit.get("grounding_method"),
        "audit_status": audit.get("audit_status"),
        "recovery_assessment_policy_grounded": bool(
            audit.get("recovery_assessment_policy_grounded")
        ),
        "primary_failure": primary.to_failure_payload(),
        "root_cause_assessment": _assessment(audit, "root_cause_assessment"),
        "model_recovery_assessment": _assessment(audit, "model_recovery_assessment"),
        "recovery_field_audits": list(audit.get("recovery_field_audits") or []),
        "related_failures": list(audit.get("audited_related_failure_roles") or []),
        "secondary_failures": list(audit.get("audited_related_failures") or []),
        "evidence": list(audit.get("grounded_evidence") or []),
    }


def history_trace(history: HistorySummary) -> dict[str, Any]:
    fields = (
        "available",
        "availability_reason",
        "same_job_attempts",
        "matching_root_attempts",
        "observed_advance_attempts",
        "same_progress_attempts",
        "regressed_progress_attempts",
        "unknown_progress_attempts",
        "no_observed_advance_attempts",
        "matching_root_attempts_with_observed_training_progress",
        "matching_root_attempts_before_observed_training_progress",
        "matching_root_attempts_with_unknown_training_progress",
        "exact_failure_position_attempts",
        "same_rank_iteration_attempts",
        "same_data_position_attempts",
        "same_artifact_attempts",
        "consecutive_same_root_no_advance_attempts",
        "advanced_beyond_all_comparable_attempts",
        "cross_node_recurrence",
        "same_node_recurrence",
        "same_gpu_recurrence",
        "same_rank_only_recurrence",
        "rank_to_gpu_mapping_available",
    )
    payload = {field: getattr(history, field, None) for field in fields}
    payload["comparisons"] = [
        asdict(item) if is_dataclass(item) else item for item in getattr(history, "comparisons", ())
    ]
    return payload


def l1_token_limit_summary(model_calls: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    hit_calls = [
        {
            "model_turn": call.get("model_turn"),
            "attempt": call.get("attempt"),
            "finish_reason": call.get("finish_reason"),
            "completion_tokens": _usage_int(call.get("usage"), "completion_tokens"),
            "reasoning_tokens": _nested_usage_int(
                call.get("usage"),
                "completion_tokens_details",
                "reasoning_tokens",
            ),
            "max_retries": call.get("max_retries"),
            "limit_kind": (
                "context_window"
                if call.get("error_type") == "context_window_exceeded"
                else "output"
            ),
        }
        for call in model_calls
        if call.get("finish_reason") == "length"
        or call.get("error_type") == "context_window_exceeded"
    ]
    return {
        "hit": bool(hit_calls),
        "hit_count": len(hit_calls),
        "hit_calls": hit_calls,
        "meaning": (
            "At least one L1 call exhausted output tokens or was rejected because "
            "input plus requested output exceeded the model context window."
        ),
    }


def _l1_token_usage(model_calls: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    cached_prompt_tokens = 0
    reasoning_tokens = 0
    calls_with_usage = 0

    for call in model_calls:
        usage = call.get("usage")
        if not isinstance(usage, Mapping):
            continue
        calls_with_usage += 1
        call_prompt_tokens = _usage_int(usage, "prompt_tokens")
        call_completion_tokens = _usage_int(usage, "completion_tokens")
        call_total_tokens = _usage_int(usage, "total_tokens")
        prompt_tokens += call_prompt_tokens
        completion_tokens += call_completion_tokens
        total_tokens += call_total_tokens

        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, Mapping):
            cached_prompt_tokens += _usage_int(prompt_details, "cached_tokens")
        completion_details = usage.get("completion_tokens_details")
        if isinstance(completion_details, Mapping):
            reasoning_tokens += _usage_int(completion_details, "reasoning_tokens")

    return {
        "model_calls_with_usage": calls_with_usage,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def _usage_int(usage: Mapping[str, Any], key: str) -> int:
    if not isinstance(usage, Mapping):
        return 0
    try:
        return int(usage.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _nested_usage_int(usage: Any, parent_key: str, key: str) -> int:
    if not isinstance(usage, Mapping):
        return 0
    child = usage.get(parent_key)
    if not isinstance(child, Mapping):
        return 0
    return _usage_int(child, key)


def _context_mode(
    primary: FailureEvidence | None,
    l1_result: L1EvidenceResult,
    l2_audit: Mapping[str, Any],
) -> str:
    if l2_audit.get("used"):
        return "tool_loop"
    if l1_result.model:
        return "fallback_precurated"
    return "fallback_precurated"


def build_log_unavailable_trace(
    result: AnalysisResult,
    execution_context: AnalysisExecutionContext | None,
    reason: str,
    *,
    total_wall_clock_s: float,
) -> dict[str, Any]:
    return {
        "context_mode": "log_unavailable",
        "request": execution_context.request.to_payload() if execution_context else None,
        "log_path": execution_context.log_path if execution_context else None,
        "job_id": execution_context.job_id if execution_context else None,
        "cycle_id": execution_context.cycle_id if execution_context else None,
        "result_provenance": dict(result.result_provenance),
        "external_output": result.to_payload(),
        "timing": {
            "total_wall_clock_s": total_wall_clock_s,
            "l0_wall_clock_s": 0.0,
            "l0a_wall_clock_s": 0.0,
            "l0b_wall_clock_s": 0.0,
            "l1_wall_clock_s": 0.0,
            "l1_model_call_wall_clock_s": 0.0,
            "l1_tool_wall_clock_s": 0.0,
            "l2_wall_clock_s": 0.0,
            "l3_wall_clock_s": 0.0,
            "l4_wall_clock_s": 0.0,
            "l1_model_calls": 0,
            "l1_tool_calls": 0,
        },
        "latency_measurement": {
            "mode": "terminal_request_to_result",
            "terminal_total_wall_clock_s": total_wall_clock_s,
            "post_progressive_end_wall_clock_s": None,
            "progressive_decision_window_hit": None,
            "production_gate_measured": False,
        },
        "layers": {
            "L0": {
                "name": "evidence_assembly_and_projection",
                "wall_clock_s": 0.0,
                "status": "log_unavailable",
                "sub_stages": {
                    "L0A": {
                        "name": "complete_evidence_assembly",
                        "wall_clock_s": 0.0,
                        "status": "log_unavailable",
                    },
                    "L0B": {
                        "name": "initial_model_evidence_view",
                        "wall_clock_s": 0.0,
                        "status": "not_run",
                    },
                },
            },
            "L1": {
                "name": "semantic_analysis",
                "wall_clock_s": 0.0,
                "status": "not_run",
            },
            "L2": {
                "name": "evidence_grounding_and_identity",
                "wall_clock_s": 0.0,
                "grounding_status": "not_run",
                "audit_status": "not_run",
                "not_run_reason": "log_unavailable",
            },
            "L3": {
                "name": "history_enrichment",
                "wall_clock_s": 0.0,
                "selected_failure_facts_source": None,
                "history_identity_ready": False,
                "history_available": False,
                "same_job_attempts": 0,
                "matching_root_attempts": 0,
                "observed_advance_attempts": 0,
                "no_observed_advance_attempts": 0,
                "unknown_progress_attempts": 0,
            },
            "L4": {
                "name": "policy_decision",
                "wall_clock_s": 0.0,
                "decision": result.decision,
                "decision_basis": result.decision_basis,
                "result_quality": result.result_provenance.get("result_quality"),
            },
        },
        "selected_failure_facts": None,
        "l2_grounded_semantics": None,
        "l2_audit": {
            "used": False,
            "grounding_status": "not_run",
            "audit_status": "not_run",
            "not_run_reason": "log_unavailable",
        },
        "l3_history": {"available": False},
        "l0_model_view": None,
        "l4_policy": {
            "decision": result.decision,
            "decision_basis": result.decision_basis,
            "retry_policy": dict(result.retry_policy),
            "history_inputs": {"available": False},
            "result_provenance": dict(result.result_provenance),
        },
        "anomalies": {"log_unavailable": True, "reason": reason},
    }
