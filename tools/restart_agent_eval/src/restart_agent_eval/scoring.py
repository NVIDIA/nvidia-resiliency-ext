# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure scoring, KPI, and diagnostic calculations for one-log reviews."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .gold import GoldSchemaError, validate_gold_source, validate_scored_gold_label

__all__ = (
    "ToolEfficiencyInput",
    "distributed_incident_summaries",
    "l0_bundle_kpis",
    "l1_execution_status",
    "l1_kpis",
    "l2_kpis",
    "l3_kpis",
    "l4_kpis",
    "line_numbering_summary",
    "model_call_summary",
    "model_selection_signals",
    "path_redaction_audit",
    "read_gold_label",
    "score_against_gold",
    "score_l0_against_gold",
    "score_l0b_against_gold",
    "score_l2_audit",
    "score_path_effect",
    "score_semantic_view",
    "semantic_safety_summary",
    "tool_efficiency_summary",
)

GENERIC_PATH_COMPONENTS = frozenset(
    {
        "data",
        "dataset",
        "datasets",
        "input",
        "inputs",
        "log",
        "logs",
        "output",
        "outputs",
        "private",
        "src",
        "tmp",
        "users",
        "var",
    }
)
MIN_DISTINCTIVE_PATH_COMPONENT_LENGTH = 5
OPAQUE_TRANSCRIPT_FIELDS = frozenset(
    {
        "call_id",
        "id",
        "provider_specific_fields",
        "thought_signature",
        "tool_call_id",
    }
)


def read_gold_label(path: Path, *, source_log: Path | None = None) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"gold label is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"gold label must be an object: {path}")
    if value.get("schema_version") != "restart_agent_eval.v1":
        raise SystemExit(f"gold label schema_version is invalid: {path}")
    try:
        validate_scored_gold_label(value)
        if source_log is not None:
            validate_gold_source(value, source_log)
    except GoldSchemaError as exc:
        raise SystemExit(f"gold label schema is invalid: {path}: {exc}") from exc
    return value


def score_against_gold(
    analysis: dict[str, Any],
    gold: dict[str, Any] | None,
    *,
    l0_bundle: Any = None,
    l0_model_view: Any = None,
    l1_evidence: Any = None,
    l2_grounded_semantics: Any = None,
    l2_audit: Any = None,
    include_l1: bool = True,
    fallback_analysis: Any = None,
    enriched_analysis: Any = None,
) -> dict[str, Any] | None:
    if gold is None:
        return None
    l1_view = l1_evidence if isinstance(l1_evidence, dict) else {}
    l2_view = l2_grounded_semantics if isinstance(l2_grounded_semantics, dict) else {}
    l2_audit = l2_audit if isinstance(l2_audit, dict) else {}
    l0_bundle = l0_bundle if isinstance(l0_bundle, dict) else {}
    l0_model_view = l0_model_view if isinstance(l0_model_view, dict) else {}
    l1_score = score_semantic_view(l1_view, gold, include_action=False) if include_l1 else None
    l2_score = score_semantic_view(l2_view, gold, include_action=False) if l2_view else None
    l4_score = score_semantic_view(analysis, gold, include_action=True)
    fallback_score = (
        score_semantic_view(fallback_analysis, gold, include_action=True)
        if isinstance(fallback_analysis, dict) and fallback_analysis
        else None
    )
    enriched_score = (
        score_semantic_view(enriched_analysis, gold, include_action=True)
        if isinstance(enriched_analysis, dict) and enriched_analysis
        else None
    )
    return {
        "case_id": gold.get("case_id"),
        "label_version": gold.get("label_version"),
        "l0a": score_l0_against_gold(l0_bundle, gold),
        "l0b": score_l0b_against_gold(l0_bundle, l0_model_view, gold),
        "l1": l1_score,
        "l2": {
            "audit_status": l2_audit.get("audit_status"),
            "audit_correct": score_l2_audit(l2_audit, gold),
            **_score_l2_history_identity(l2_audit, l0_bundle, gold),
            "reference_audit_effect": _reference_audit_effect(l1_score or {}, l2_score),
            "grounded_semantic_projection": l2_score,
        },
        "l4": l4_score,
        "fallback_l4": fallback_score,
        "enriched_l4": enriched_score,
        "l4_path_comparison": {
            "action_effect": score_path_effect(
                fallback_score,
                enriched_score,
                "action_correct",
            ),
            "policy_action_effect": score_path_effect(
                fallback_score,
                enriched_score,
                "policy_action_pass",
            ),
        },
        "calibration_score": None,
        "calibration_note": (
            "Confidence is recorded per case; calibration requires a labeled corpus."
        ),
    }


def l1_execution_status(
    *,
    l1_layer: dict[str, Any],
    model_call_summary: dict[str, Any],
    route_execution_status: str | None = None,
) -> tuple[str, list[str]]:
    if route_execution_status == "deadline_exceeded":
        return "deadline_exceeded", ["analysis_deadline_exceeded"]
    product_status = l1_layer.get("execution_status")
    raw_product_issues = l1_layer.get("execution_issues") or []
    product_issues = (
        [str(value) for value in raw_product_issues] if isinstance(raw_product_issues, list) else []
    )
    if product_status:
        return str(product_status), product_issues
    if not l1_layer.get("output_usable"):
        return "failed", [str(l1_layer.get("output_status") or "unusable")]

    issues: list[str] = []
    if _int_or_zero(model_call_summary.get("failed_calls")):
        issues.append("model_call_failed")
    if _int_or_zero(model_call_summary.get("retried_calls")):
        issues.append("retry_used")
    if _int_or_zero(model_call_summary.get("timeout_calls")):
        issues.append("provider_timeout")
    if _int_or_zero(model_call_summary.get("provider_error_count")):
        issues.append("provider_error")
    return ("degraded", issues) if issues else ("ok", [])


def l2_kpis(
    *,
    l2_audit: dict[str, Any],
    l0_primary: dict[str, Any],
    timing: dict[str, Any],
) -> dict[str, Any]:
    field_findings = l2_audit.get("field_findings") or {}
    citations = l2_audit.get("citation_audits") or []
    findings = l2_audit.get("findings") or []
    if not isinstance(field_findings, dict):
        field_findings = {}
    if not isinstance(citations, list):
        citations = []
    else:
        citations = [item for item in citations if isinstance(item, dict)]
    if not isinstance(findings, list):
        findings = []
    else:
        findings = [item for item in findings if isinstance(item, dict)]
    finding_count = sum(len(items) for items in field_findings.values() if isinstance(items, list))
    severity_counts: dict[str, int] = {}
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        severity = str(finding.get("severity") or "credibility")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    root_fingerprint = l2_audit.get("stable_root_fingerprint")
    l0_root_fingerprint = l0_primary.get("root_fingerprint")
    matches_l0_root_fingerprint = (
        root_fingerprint == l0_root_fingerprint
        if root_fingerprint and l0_root_fingerprint
        else None
    )
    return {
        "wall_clock_s": timing.get("l2_wall_clock_s"),
        "grounding_status": l2_audit.get("grounding_status"),
        "grounding_method": l2_audit.get("grounding_method"),
        "audit_status": l2_audit.get("audit_status"),
        "audit_ran": bool(l2_audit.get("used")),
        "primary_available": bool(l2_audit.get("primary_used")),
        "recovery_assessment_available": bool(l2_audit.get("recovery_assessment_used")),
        "related_failures_audited": len(l2_audit.get("audited_related_failure_roles") or []),
        "failure_domain_supporting_lines": l2_audit.get("failure_domain_supporting_lines") or [],
        "retry_outlook_supporting_lines": l2_audit.get("retry_outlook_supporting_lines") or [],
        "unresolved_recovery_supporting_lines": l2_audit.get("recovery_unresolved_supporting_lines")
        or [],
        "finding_count": finding_count,
        "material_finding_count": (
            sum(
                1
                for finding in findings
                if isinstance(finding, dict) and finding.get("policy_material") is True
            )
            if findings
            else finding_count
        ),
        "finding_severity_counts": (
            severity_counts
            if findings
            else ({"unspecified": finding_count} if finding_count else {})
        ),
        "field_findings": field_findings,
        "citation_count": len(citations),
        "exact_citation_count": sum(1 for item in citations if item.get("status") == "exact"),
        "rendered_exact_citation_count": sum(
            1 for item in citations if item.get("status") == "rendered_exact"
        ),
        "abbreviated_exact_citation_count": sum(
            1 for item in citations if item.get("status") == "abbreviated_exact"
        ),
        "nearby_resolved_count": sum(
            1 for item in citations if item.get("status") == "nearby_resolved"
        ),
        "ungrounded_citation_count": sum(
            1
            for item in citations
            if item.get("status") in {"ungrounded", "ambiguous_nearby_match"}
        ),
        "grounding_adjustment_count": len(l2_audit.get("grounding_adjustments") or []),
        "recovery_audit_observation_count": len(l2_audit.get("recovery_field_audits") or []),
        "recovery_field_audits": l2_audit.get("recovery_field_audits") or [],
        "operation_artifact_audit_observations": (
            l2_audit.get("operation_artifact_audit_observations") or []
        ),
        "root_fingerprint_owner": "L2",
        "root_fingerprint": root_fingerprint,
        "root_fingerprint_source": l2_audit.get("root_fingerprint_source"),
        "root_fingerprint_available": bool(root_fingerprint),
        "history_identity_ready": bool(root_fingerprint),
        "matches_l0_root_fingerprint": matches_l0_root_fingerprint,
        "stable_identity_anchor_line": l2_audit.get("stable_identity_anchor_line"),
        "stable_identity_anchor_reason": l2_audit.get("stable_identity_anchor_reason"),
    }


def l3_kpis(
    *,
    analyzer_trace: dict[str, Any],
    timing: dict[str, Any],
) -> dict[str, Any]:
    history = analyzer_trace.get("l3_history") or {}
    current = analyzer_trace.get("current_failure_facts") or {}
    if not isinstance(history, dict):
        history = {}
    if not isinstance(current, dict):
        current = {}
    return {
        "wall_clock_s": timing.get("l3_wall_clock_s"),
        "current_failure_facts_source": current.get("source"),
        "history_identity_ready": current.get("history_identity_ready"),
        "current_root_fingerprint": current.get("root_fingerprint"),
        "history_available": history.get("available"),
        "same_job_attempts": history.get("same_job_attempts"),
        "matching_root_attempts": history.get("matching_root_attempts"),
        "observed_advance_attempts": history.get("observed_advance_attempts"),
        "no_observed_advance_attempts": history.get("no_observed_advance_attempts"),
        "unknown_progress_attempts": history.get("unknown_progress_attempts"),
        "exact_failure_position_attempts": history.get("exact_failure_position_attempts"),
        "same_data_position_attempts": history.get("same_data_position_attempts"),
        "same_artifact_attempts": history.get("same_artifact_attempts"),
        "consecutive_same_root_no_advance_attempts": history.get(
            "consecutive_same_root_no_advance_attempts"
        ),
        "advanced_beyond_all_comparable_attempts": history.get(
            "advanced_beyond_all_comparable_attempts"
        ),
        "retry_then_skip_history": history.get("retry_then_skip_history"),
        "comparisons": history.get("comparisons") or [],
    }


def l4_kpis(
    *,
    analysis: dict[str, Any],
    analyzer_trace: dict[str, Any],
    timing: dict[str, Any],
) -> dict[str, Any]:
    l4_policy = analyzer_trace.get("l4_policy") or {}
    latency = analyzer_trace.get("latency_measurement") or {}
    if not isinstance(l4_policy, dict):
        l4_policy = {}
    if not isinstance(latency, dict):
        latency = {}
    retry_policy = analysis.get("retry_policy") or l4_policy.get("retry_policy") or {}
    provenance = analysis.get("result_provenance") or {}
    if not isinstance(retry_policy, dict):
        retry_policy = {}
    if not isinstance(provenance, dict):
        provenance = {}
    downstream_roles = [
        {
            "causal_role": item.get("causal_role"),
            "first_line": item.get("first_line"),
            "last_line": item.get("last_line"),
            "count": item.get("count"),
        }
        for item in analysis.get("cascades") or []
        if isinstance(item, dict)
    ]
    return {
        "wall_clock_s": timing.get("l4_wall_clock_s"),
        "decision": analysis.get("decision"),
        "decision_basis": analysis.get("decision_basis"),
        "policy_version": retry_policy.get("policy_version"),
        "rule": retry_policy.get("rule"),
        "allowed_retries": retry_policy.get("allowed_retries"),
        "matching_prior_failures": retry_policy.get("matching_prior_failures"),
        "retry_budget_exhausted": retry_policy.get("retry_budget_exhausted"),
        "recovery_assessment_policy_grounded": retry_policy.get(
            "recovery_assessment_policy_grounded"
        ),
        "current_evidence_qualified": retry_policy.get("current_evidence_qualified"),
        "observed_advance": retry_policy.get("observed_advance"),
        "failure_domain": retry_policy.get("failure_domain"),
        "failure_domain_status": retry_policy.get("failure_domain_status"),
        "failure_domain_confidence": retry_policy.get("failure_domain_confidence"),
        "retry_outlook_without_workload_change": retry_policy.get(
            "retry_outlook_without_workload_change"
        ),
        "retry_outlook_status": retry_policy.get("retry_outlook_status"),
        "retry_outlook_confidence": retry_policy.get("retry_outlook_confidence"),
        "match_requirements": retry_policy.get("match_requirements") or {},
        "evidence_source": provenance.get("evidence_source"),
        "model_contribution": provenance.get("model_contribution"),
        "result_quality": provenance.get("result_quality"),
        "nvrx_use": provenance.get("nvrx_use"),
        "downstream_roles": downstream_roles,
        "latency_mode": latency.get("mode") or "terminal_request_to_result",
        "terminal_total_wall_clock_s": latency.get("terminal_total_wall_clock_s")
        or timing.get("total_wall_clock_s"),
        "post_progressive_end_wall_clock_s": latency.get("post_progressive_end_wall_clock_s"),
        "progressive_decision_window_hit": latency.get("progressive_decision_window_hit"),
        "production_gate_measured": bool(latency.get("production_gate_measured")),
    }


def l0_bundle_kpis(
    *,
    analysis: dict[str, Any],
    l0_bundle: dict[str, Any],
    l0_model_view: dict[str, Any],
    timing: dict[str, Any],
    tool_efficiency: dict[str, Any],
) -> dict[str, Any]:
    context = _build_l0_kpi_context(
        analysis=analysis,
        l0_bundle=l0_bundle,
        l0_model_view=l0_model_view,
        timing=timing,
        tool_efficiency=tool_efficiency,
    )
    return {
        **_l0a_bundle_kpis(context),
        **_l0b_projection_kpis(context),
        **_l0_cross_stage_tool_diagnostics(context),
        "l0_reused": bool(timing.get("l0_reused")),
        "review_mode_note": (
            "selected_primary_* fields describe analyzer-selected primary coverage. "
            "A scored eval case with a human label should add gold_root_* coverage."
        ),
    }


@dataclass(frozen=True)
class _L0KpiContext:
    bundle: dict[str, Any]
    model_view: dict[str, Any]
    timing: dict[str, Any]
    tool_efficiency: dict[str, Any]
    primary_line: int | None
    primary_excerpt: bool | None
    primary_bundle: bool | None
    primary_windows: list[str]
    primary_anchor_sources: list[str]
    progress: dict[str, Any]
    l0_primary: dict[str, Any]
    selection: dict[str, Any]
    coverage: dict[str, Any]
    projection: dict[str, Any]
    setup_markers: list[dict[str, Any]]
    candidate_anchors: list[dict[str, Any]]
    context_windows: list[dict[str, Any]]
    occurrence_groups: list[dict[str, Any]]
    failure_episodes: list[dict[str, Any]]
    distributed_incidents: list[dict[str, Any]]
    path_access_facts: list[dict[str, Any]]
    path_namespace_summary: dict[str, Any]
    later_progress_observations: list[dict[str, Any]]


def _build_l0_kpi_context(
    *,
    analysis: dict[str, Any],
    l0_bundle: dict[str, Any],
    l0_model_view: dict[str, Any],
    timing: dict[str, Any],
    tool_efficiency: dict[str, Any],
) -> _L0KpiContext:
    primary = analysis.get("primary_failure") or {}
    if not isinstance(primary, dict):
        primary = {}
    progress = l0_bundle.get("run_progress_summary") or {}
    if not isinstance(progress, dict):
        progress = {}
    l0_primary = l0_bundle.get("deterministic_primary_candidate") or {}
    if not isinstance(l0_primary, dict):
        l0_primary = {}
    selection = l0_bundle.get("selection_summary") or {}
    if not isinstance(selection, dict):
        selection = {}
    coverage = l0_bundle.get("evidence_coverage") or {}
    if not isinstance(coverage, dict):
        coverage = {}
    progress_facts = l0_bundle.get("progress") or {}
    if not isinstance(progress_facts, dict):
        progress_facts = {}
    projection = l0_model_view.get("projection_metrics") or {}
    if not isinstance(projection, dict):
        projection = {}
    setup_markers = _list_of_dicts(progress_facts.get("setup_markers"))
    primary_line = _int_or_none(primary.get("line"))
    primary_excerpt = _line_in_context_windows(primary_line, l0_bundle)
    primary_bundle = _line_in_l0_bundle(primary_line, l0_bundle, primary_excerpt)
    candidate_anchors = _list_of_dicts(l0_bundle.get("candidate_anchors"))
    path_namespace_summary = l0_bundle.get("path_namespace_summary") or {}
    if not isinstance(path_namespace_summary, dict):
        path_namespace_summary = {}
    return _L0KpiContext(
        bundle=l0_bundle,
        model_view=l0_model_view,
        timing=timing,
        tool_efficiency=tool_efficiency,
        primary_line=primary_line,
        primary_excerpt=primary_excerpt,
        primary_bundle=primary_bundle,
        primary_windows=_context_window_ids_for_line(primary_line, l0_bundle),
        primary_anchor_sources=_candidate_anchor_sources_for_line(primary_line, l0_bundle),
        progress=progress,
        l0_primary=l0_primary,
        selection=selection,
        coverage=coverage,
        projection=projection,
        setup_markers=setup_markers,
        candidate_anchors=candidate_anchors,
        context_windows=_list_of_dicts(l0_bundle.get("context_windows")),
        occurrence_groups=_list_of_dicts(l0_bundle.get("occurrence_groups")),
        failure_episodes=_list_of_dicts(l0_bundle.get("failure_episodes")),
        distributed_incidents=distributed_incident_summaries(l0_bundle),
        path_access_facts=_list_of_dicts(l0_bundle.get("path_access_facts")),
        path_namespace_summary=path_namespace_summary,
        later_progress_observations=_list_of_dicts(
            l0_bundle.get("later_progress_after_fault_observations")
        ),
    )


def _l0a_bundle_kpis(context: _L0KpiContext) -> dict[str, Any]:
    progress = context.progress
    progress_after = progress.get("progress_after_failure_episode")
    terminal_episode_count = sum(
        1 for episode in context.failure_episodes if episode.get("status") == "terminal"
    )
    top_anchor_progress_after_count = sum(
        1
        for anchor in context.candidate_anchors
        if anchor.get("later_observed_progress_line") is not None
    )
    recovered_or_progressed_anchor_count = sum(
        1
        for anchor in context.candidate_anchors
        if anchor.get("later_observation_proves_recovery")
        or (
            anchor.get("later_observed_progress_line") is not None
            and "failure_episode" not in (anchor.get("sources") or [])
        )
    )
    candidate_anchors_without_excerpt = sum(
        1 for anchor in context.candidate_anchors if not anchor.get("context_window_ids")
    )
    return {
        "l0_wall_clock_s": context.timing.get("l0_wall_clock_s"),
        "l0a_wall_clock_s": context.timing.get("l0a_wall_clock_s"),
        "decision_evidence_wall_clock_s": context.timing.get("decision_evidence_wall_clock_s"),
        "line_count": context.bundle.get("line_count"),
        "byte_size": context.bundle.get("byte_size"),
        "context_window_count": len(context.context_windows),
        "candidate_anchor_count": len(context.candidate_anchors),
        "occurrence_group_count": len(context.occurrence_groups),
        "failure_episode_count": len(context.failure_episodes),
        "terminal_failure_episode_count": terminal_episode_count,
        "distributed_failure_incident_count": len(context.distributed_incidents),
        "distributed_failure_incidents": context.distributed_incidents,
        "path_access_fact_count": len(context.path_access_facts),
        "path_namespaces_by_role": context.path_namespace_summary.get("namespaces_by_role") or {},
        "cross_namespace_paths_observed": context.path_namespace_summary.get(
            "cross_namespace_paths_observed"
        ),
        "failed_vs_configured_write_mismatch": context.path_namespace_summary.get(
            "failed_vs_configured_write_mismatch"
        ),
        "path_ownership_verified": context.path_namespace_summary.get("ownership_verified"),
        "operation_artifact_comparisons": _list_of_dicts(
            context.bundle.get("operation_artifact_comparisons")
        ),
        "selected_primary_line": context.primary_line,
        "root_fingerprint_owner": "L0",
        "root_fingerprint": context.l0_primary.get("root_fingerprint"),
        "root_fingerprint_source": context.l0_primary.get("root_fingerprint_source"),
        "root_fingerprint_available": bool(context.l0_primary.get("root_fingerprint")),
        "history_identity_ready": bool(context.l0_primary.get("root_fingerprint")),
        "selected_primary_in_bundle": _yes_no_unknown(context.primary_bundle),
        "selected_primary_context_window_ids": context.primary_windows,
        "selected_primary_anchor_sources": context.primary_anchor_sources,
        "progress_after_fault_known": progress_after is not None,
        "progress_after_fault": progress_after,
        "first_terminal_incident_line": progress.get("first_terminal_incident_line"),
        "first_terminal_incident_timestamp": progress.get("first_terminal_incident_timestamp"),
        "configured_terminal_timeout_seconds": progress.get("configured_terminal_timeout_seconds"),
        "seconds_from_last_progress_to_terminal_incident": progress.get(
            "seconds_from_last_progress_to_terminal_incident"
        ),
        "terminal_detection_lag_seconds": progress.get("terminal_detection_lag_seconds"),
        "successful_runtime_seconds": progress.get("successful_runtime_seconds"),
        "first_iteration": progress.get("first_iteration"),
        "last_iteration": progress.get("last_iteration"),
        "iteration_delta": progress.get("iteration_delta"),
        "last_checkpoint_iteration": progress.get("last_checkpoint_iteration"),
        "iterations_since_checkpoint": progress.get("iterations_since_checkpoint"),
        "checkpoint_load_iteration": progress.get("checkpoint_load_iteration"),
        "latest_observed_failure_iteration": progress.get("latest_observed_failure_iteration"),
        "latest_observed_failure_iteration_line": progress.get(
            "latest_observed_failure_iteration_line"
        ),
        "observed_iterations_after_checkpoint_load": progress.get(
            "observed_iterations_after_checkpoint_load"
        ),
        "observed_failure_phase": context.l0_primary.get("phase"),
        "later_progress_after_fault_observation_count": len(context.later_progress_observations),
        "later_progress_after_fault_event_count": sum(
            _int_or_zero(item.get("event_count")) for item in context.later_progress_observations
        ),
        "setup_marker_count": len(context.setup_markers),
        "setup_marker_types": [marker.get("marker_type") for marker in context.setup_markers],
        "setup_marker_lines": [marker.get("line") for marker in context.setup_markers],
        "candidate_anchors_without_excerpt": candidate_anchors_without_excerpt,
        "top_anchor_progress_after_count": top_anchor_progress_after_count,
        "recovered_or_progressed_top_anchor_count": (recovered_or_progressed_anchor_count),
        "selection_summary": {
            "high_signal_lines": context.selection.get("high_signal_lines"),
            "sampled_candidate_lines": context.selection.get("sampled_candidate_lines"),
            "dropped_noise_lines": context.selection.get("dropped_noise_lines"),
            "caps_hit": context.selection.get("caps_hit") or [],
            "primary_after_context_available": context.selection.get(
                "primary_after_context_available"
            ),
        },
        "evidence_coverage": context.coverage,
    }


def _l0b_projection_kpis(context: _L0KpiContext) -> dict[str, Any]:
    return {
        "l0b_wall_clock_s": context.timing.get("l0b_wall_clock_s"),
        "l0b_schema_version": context.model_view.get("schema_version"),
        "restart_environment_context": context.model_view.get("restart_environment_context") or {},
        "l0b_projection_metrics": context.projection,
        "selected_primary_in_excerpt": _yes_no_unknown(context.primary_excerpt),
    }


def _l0_cross_stage_tool_diagnostics(context: _L0KpiContext) -> dict[str, Any]:
    tool_calls = _int_or_zero(context.tool_efficiency.get("calls"))
    useful_tool_proxy = "n/a"
    if tool_calls:
        useful_tool_proxy = (
            "yes"
            if context.tool_efficiency.get("final_context_impact")
            in {"decision_critical_primary", "decision_relevant_support"}
            else "no"
        )
    return {
        "tool_calls_needed": tool_calls,
        "tool_calls_useful_proxy": useful_tool_proxy,
        "tool_calls_useful_basis": "final_context_impact",
        "tool_calls_added_new_prompt_lines": _int_or_zero(
            context.tool_efficiency.get("new_prompt_excerpt_line_count")
        ),
        "duplicate_tool_calls": context.tool_efficiency.get("duplicate_prompt_context_calls"),
        "no_new_prompt_line_tool_calls": context.tool_efficiency.get("no_new_prompt_line_calls"),
    }


def _line_in_l0_bundle(
    line_no: int | None,
    l0_bundle: dict[str, Any],
    in_context_window: bool | None,
) -> bool | None:
    if line_no is None:
        return None
    if in_context_window:
        return True
    for anchor in _list_of_dicts(l0_bundle.get("candidate_anchors")):
        if _int_or_none(anchor.get("line")) == line_no:
            return True
    for match in _list_of_dicts(l0_bundle.get("registry_matches")):
        if _int_or_none(match.get("line")) == line_no:
            return True
    deterministic_primary_candidate = l0_bundle.get("deterministic_primary_candidate")
    if isinstance(deterministic_primary_candidate, dict):
        if _int_or_none(deterministic_primary_candidate.get("line")) == line_no:
            return True
    for episode in _list_of_dicts(l0_bundle.get("failure_episodes")):
        if _line_in_failure_episode(line_no, episode):
            return True
    return False


def _line_in_failure_episode(line_no: int, episode: dict[str, Any]) -> bool:
    for key in (
        "first_exception_line",
        "terminal_exception_line",
        "first_teardown_line",
        "first_process_termination_line",
        "first_scheduler_cancel_line",
    ):
        if _int_or_none(episode.get(key)) == line_no:
            return True
    start = _int_or_none(episode.get("start_line"))
    end = _int_or_none(episode.get("end_line"))
    return start is not None and end is not None and start <= line_no <= end


def _line_in_context_windows(
    line_no: int | None,
    l0_bundle: dict[str, Any],
) -> bool | None:
    if line_no is None:
        return None
    return bool(_context_window_ids_for_line(line_no, l0_bundle))


def _context_window_ids_for_line(
    line_no: int | None,
    l0_bundle: dict[str, Any],
) -> list[Any]:
    if line_no is None:
        return []
    result: list[Any] = []
    for window in _list_of_dicts(l0_bundle.get("context_windows")):
        if _context_window_contains_line(window, line_no):
            result.append(window.get("window_id"))
    return result


def _context_window_contains_line(window: dict[str, Any], line_no: int) -> bool:
    start = _int_or_none(window.get("start_line"))
    end = _int_or_none(window.get("end_line"))
    if start is not None and end is not None and start <= line_no <= end:
        return True
    for entry in window.get("lines") or []:
        if isinstance(entry, dict) and _int_or_none(entry.get("line")) == line_no:
            return True
    return False


def _candidate_anchor_sources_for_line(
    line_no: int | None,
    l0_bundle: dict[str, Any],
) -> list[Any]:
    if line_no is None:
        return []
    sources: list[Any] = []
    for anchor in _list_of_dicts(l0_bundle.get("candidate_anchors")):
        if _int_or_none(anchor.get("line")) == line_no:
            values = anchor.get("sources") or []
            if isinstance(values, list):
                sources.extend(values)
    return sources


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def distributed_incident_summaries(
    l0_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    fields = (
        "incident_id",
        "incident_kind",
        "incident_type",
        "status",
        "first_observed_line",
        "last_observed_line",
        "primary_observed_line",
        "sample_lines",
        "event_count",
        "unique_operation_count",
        "operation_types",
        "operation_signatures",
        "observed_rank_count",
        "rank_spread",
        "process_group_types",
        "phase",
        "configured_timeout_seconds",
        "last_progress_line",
        "last_progress_timestamp",
        "first_detection_timestamp",
        "seconds_since_last_progress",
        "detection_lag_seconds",
        "history_fingerprint",
        "history_fingerprint_source",
        "root_cause_status",
        "interpretation",
    )
    return [
        {field: incident.get(field) for field in fields}
        for incident in _list_of_dicts(l0_bundle.get("distributed_failure_incidents"))
    ]


def _yes_no_unknown(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def semantic_safety_summary(
    *, l2_audit: dict[str, Any], l4_policy: dict[str, Any]
) -> dict[str, Any]:
    grounding_adjustments = l2_audit.get("grounding_adjustments") or []
    if not isinstance(grounding_adjustments, list):
        grounding_adjustments = []
    semantic_normalizations = [
        item
        for item in grounding_adjustments
        if isinstance(item, dict)
        and str(item.get("field") or "")
        in {
            "primary_failure.fine_class",
            "primary_failure.policy_class",
            "primary_failure.root_fingerprint",
            "primary_failure.signature",
        }
    ]
    recovery_field_audits = [
        item
        for item in l2_audit.get("recovery_field_audits") or []
        if isinstance(item, dict)
        and str(item.get("field") or "").startswith("model_recovery_assessment.")
    ]
    retry_policy = l4_policy.get("retry_policy") or {}
    model_failure_domain = l2_audit.get("model_failure_domain")
    model_failure_domain_status = l2_audit.get("model_failure_domain_status")
    model_failure_domain_confidence = l2_audit.get("model_failure_domain_confidence")
    model_retry_outlook = l2_audit.get("model_retry_outlook_without_workload_change")
    model_retry_outlook_status = l2_audit.get("model_retry_outlook_status")
    model_retry_outlook_confidence = l2_audit.get("model_retry_outlook_confidence")
    model_root_cause_status = l2_audit.get("model_root_cause_status")
    if not l2_audit or (
        not l2_audit.get("model") and not l2_audit.get("used") and not semantic_normalizations
    ):
        semantic_safety = "not_applicable"
    elif semantic_normalizations:
        semantic_safety = "stable_identity_adjusted"
    elif recovery_field_audits:
        semantic_safety = "recovery_audit_observation"
    elif l2_audit:
        semantic_safety = "ok"
    else:
        semantic_safety = "not_reported"

    return {
        "semantic_safety": semantic_safety,
        "model_failure_domain": model_failure_domain,
        "model_failure_domain_status": model_failure_domain_status,
        "model_failure_domain_confidence": model_failure_domain_confidence,
        "model_retry_outlook_without_workload_change": model_retry_outlook,
        "model_retry_outlook_status": model_retry_outlook_status,
        "model_retry_outlook_confidence": model_retry_outlook_confidence,
        "model_root_cause_status": model_root_cause_status,
        "l2_recovery_suggestion_applied": any(
            bool(item.get("applied")) for item in recovery_field_audits
        ),
        "retry_policy_rule": retry_policy.get("rule"),
        "retry_budget_exhausted": retry_policy.get("retry_budget_exhausted"),
        "normalization_count": len(grounding_adjustments),
        "semantic_normalization_count": len(semantic_normalizations),
        "recovery_audit_observation_count": len(recovery_field_audits),
        "recovery_field_audits": recovery_field_audits,
        "normalizations": grounding_adjustments,
    }


def _prompt_excerpt_line_numbers(interaction_transcript: list[Any]) -> set[int]:
    for event in interaction_transcript:
        if not isinstance(event, dict):
            continue
        bundle = event.get("bundle")
        if not isinstance(bundle, dict):
            continue
        result: set[int] = set()
        for window in bundle.get("context_windows") or []:
            if not isinstance(window, dict):
                continue
            for line in window.get("lines") or []:
                if isinstance(line, dict):
                    line_no = _int_or_none(line.get("line"))
                    if line_no is not None:
                        result.add(line_no)
        return result
    return set()


def _tool_result_events_by_id(
    interaction_transcript: list[Any],
) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for event in interaction_transcript:
        if not isinstance(event, dict):
            continue
        tool_call_id = event.get("tool_call_id")
        if tool_call_id is not None and "result" in event:
            result[tool_call_id] = event
    return result


def _tool_result_line_numbers(result: Any) -> list[int]:
    line_numbers: list[int] = []

    def add_line(value: Any) -> None:
        line_no = _int_or_none(value)
        if line_no is not None:
            line_numbers.append(line_no)

    if not isinstance(result, dict):
        return line_numbers
    for key in ("matches", "lines", "head", "tail"):
        entries = result.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                add_line(entry.get("line"))
    return line_numbers


def _tool_optimization_note(
    *,
    calls: int,
    duplicate_prompt_context_calls: int,
    no_new_prompt_line_calls: int,
) -> str:
    if calls == 0:
        return "no_tool_calls"
    if duplicate_prompt_context_calls == calls:
        return "all_tool_calls_retrieved_only_prompt_excerpt_lines"
    if no_new_prompt_line_calls:
        return "some_tool_calls_added_no_new_prompt_excerpt_lines"
    return "tool_calls_added_new_prompt_excerpt_lines_or_non_line_context"


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _int_or_zero(value: Any) -> int:
    return _int_or_none(value) or 0


def _float_or_zero(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)] + "..."


def score_path_effect(
    fallback_score: dict[str, Any] | None,
    enriched_score: dict[str, Any] | None,
    field: str,
) -> str:
    if fallback_score is None or enriched_score is None:
        return "not_available"
    fallback_value = fallback_score.get(field)
    enriched_value = enriched_score.get(field)
    if fallback_value is None or enriched_value is None:
        return "unscored"
    if fallback_value is False and enriched_value is True:
        return "improved"
    if fallback_value is True and enriched_value is False:
        return "regressed"
    return "unchanged_correct" if fallback_value is True else "unchanged_incorrect"


def score_l0_against_gold(l0_bundle: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any] | None:
    expectation = gold.get("l0_expectation") or {}
    if not isinstance(expectation, dict) or not expectation:
        return None
    setup = _score_l0_setup_and_coverage(l0_bundle, expectation)
    primary = _score_l0_primary(l0_bundle, gold, expectation)
    progress = _score_l0_progress(
        l0_bundle,
        expectation,
        line_tolerance=primary["line_tolerance"],
        selected_primary=primary["selected_primary"],
    )
    artifact_checks = _score_l0_operation_artifacts(l0_bundle, expectation)
    result = {
        **setup,
        **{key: value for key, value in primary.items() if key != "selected_primary"},
        **progress,
        "operation_artifact_comparison_checks": artifact_checks,
    }
    scored_checks = (
        "primary_evidence_coverage",
        "selected_primary_accuracy",
        "root_fingerprint_accuracy",
        "progress_line_recall",
        "checkpoint_line_recall",
        "primary_phase_correct",
        "checkpoint_load_iteration_correct",
        "progress_after_failure_correct",
        "cascade_line_recall",
    )
    result["overall_pass"] = bool(
        result["setup_marker_types_correct"]
        and result["setup_marker_count_correct"]
        and all(result["coverage_checks"].values())
        and all(item["passed"] for item in artifact_checks)
        and all(result[name] is not False for name in scored_checks)
    )
    return result


def _score_l0_setup_and_coverage(
    bundle: dict[str, Any],
    expectation: dict[str, Any],
) -> dict[str, Any]:
    progress = bundle.get("progress") or {}
    progress = progress if isinstance(progress, dict) else {}
    markers = [item for item in progress.get("setup_markers") or [] if isinstance(item, dict)]
    observed_types = [str(item.get("marker_type")) for item in markers]
    required_types = [str(value) for value in expectation.get("required_setup_marker_types") or []]
    minimum_count = int(expectation.get("minimum_setup_marker_count") or 0)
    coverage = bundle.get("evidence_coverage") or {}
    coverage = coverage if isinstance(coverage, dict) else {}
    required_coverage = expectation.get("required_coverage") or {}
    required_coverage = required_coverage if isinstance(required_coverage, dict) else {}
    return {
        "required_setup_marker_types": required_types,
        "observed_setup_marker_types": observed_types,
        "minimum_setup_marker_count": minimum_count,
        "observed_setup_marker_count": len(markers),
        "setup_marker_types_correct": all(value in observed_types for value in required_types),
        "setup_marker_count_correct": len(markers) >= minimum_count,
        "coverage_checks": {
            str(key): coverage.get(str(key)) == value for key, value in required_coverage.items()
        },
    }


def _score_l0_primary(
    bundle: dict[str, Any],
    gold: dict[str, Any],
    expectation: dict[str, Any],
) -> dict[str, Any]:
    primary_expectation = gold.get("primary_anchor_expectation") or {}
    primary_expectation = primary_expectation if isinstance(primary_expectation, dict) else {}
    accepted_lines = [
        int(value)
        for value in (
            expectation.get("accepted_primary_lines")
            or primary_expectation.get("accepted_lines")
            or []
        )
    ]
    tolerance = int(
        expectation.get("line_tolerance")
        if expectation.get("line_tolerance") is not None
        else primary_expectation.get("tolerance_lines") or 0
    )
    candidate_lines = _l0_candidate_evidence_lines(bundle)
    selected = bundle.get("deterministic_primary_candidate") or {}
    selected = selected if isinstance(selected, dict) else {}
    selected_line = _int_or_none(selected.get("line"))
    observed_fingerprint = selected.get("root_fingerprint")
    accepted_fingerprints = [
        str(value) for value in expectation.get("accepted_root_fingerprints") or []
    ]
    return {
        "accepted_primary_lines": accepted_lines,
        "line_tolerance": tolerance,
        "candidate_evidence_line_count": len(candidate_lines),
        "primary_evidence_coverage": _line_set_matches(candidate_lines, accepted_lines, tolerance),
        "selected_primary_line": selected_line,
        "selected_primary_accuracy": _line_matches(selected_line, accepted_lines, tolerance),
        "observed_root_fingerprint": observed_fingerprint,
        "accepted_root_fingerprints": accepted_fingerprints,
        "root_fingerprint_accuracy": (
            str(observed_fingerprint) in set(accepted_fingerprints)
            if accepted_fingerprints and observed_fingerprint
            else False if accepted_fingerprints else None
        ),
        "selected_primary": selected,
    }


def _score_l0_progress(
    bundle: dict[str, Any],
    expectation: dict[str, Any],
    *,
    line_tolerance: int,
    selected_primary: dict[str, Any],
) -> dict[str, Any]:
    progress = bundle.get("progress") or {}
    progress = progress if isinstance(progress, dict) else {}
    run_progress = bundle.get("run_progress_summary") or {}
    run_progress = run_progress if isinstance(run_progress, dict) else {}
    markers = [item for item in progress.get("setup_markers") or [] if isinstance(item, dict)]
    required_progress = [int(value) for value in expectation.get("required_progress_lines") or []]
    observed_progress = _explicit_line_values(progress)
    observed_progress.update(_explicit_line_values(run_progress))
    required_checkpoints = [
        int(value) for value in expectation.get("required_checkpoint_lines") or []
    ]
    checkpoint_lines = {
        _int_or_none(marker.get("line"))
        for marker in markers
        if "checkpoint" in str(marker.get("marker_type") or "").casefold()
    }
    checkpoint_lines.discard(None)
    checkpoint_lines.update(
        line
        for key, line in _named_line_values(run_progress).items()
        if "checkpoint" in key.casefold()
    )
    required_cascades = [int(value) for value in expectation.get("required_cascade_lines") or []]
    expected_phase = expectation.get("expected_primary_phase")
    expected_checkpoint_iteration = expectation.get("expected_checkpoint_load_iteration")
    expected_progress_after = expectation.get("expected_progress_after_failure_episode")
    return {
        "required_progress_lines": required_progress,
        "progress_line_recall": _required_line_recall(
            observed_progress, required_progress, line_tolerance
        ),
        "required_checkpoint_lines": required_checkpoints,
        "checkpoint_line_recall": _required_line_recall(
            checkpoint_lines, required_checkpoints, line_tolerance
        ),
        "expected_primary_phase": expected_phase,
        "primary_phase_correct": (
            selected_primary.get("phase") == expected_phase if expected_phase is not None else None
        ),
        "expected_checkpoint_load_iteration": expected_checkpoint_iteration,
        "checkpoint_load_iteration_correct": (
            _int_or_none(run_progress.get("checkpoint_load_iteration"))
            == _int_or_none(expected_checkpoint_iteration)
            if expected_checkpoint_iteration is not None
            else None
        ),
        "expected_progress_after_failure_episode": expected_progress_after,
        "progress_after_failure_correct": (
            bool(run_progress.get("progress_after_failure_episode"))
            == bool(expected_progress_after)
            if expected_progress_after is not None
            else None
        ),
        "required_cascade_lines": required_cascades,
        "cascade_line_recall": _required_line_recall(
            _explicit_line_values(bundle.get("cascades") or []),
            required_cascades,
            line_tolerance,
        ),
    }


def _score_l0_operation_artifacts(
    bundle: dict[str, Any],
    expectation: dict[str, Any],
) -> list[dict[str, Any]]:
    observed = [
        item
        for item in bundle.get("operation_artifact_comparisons") or []
        if isinstance(item, dict)
    ]
    checks = []
    for required in expectation.get("required_operation_artifact_comparisons") or []:
        if not isinstance(required, dict):
            continue
        match = next(
            (
                item
                for item in observed
                if item.get("operation") == required.get("operation")
                and (
                    required.get("logical_artifact_id") is None
                    or item.get("logical_artifact_id") == required.get("logical_artifact_id")
                )
                and (
                    required.get("physical_unit_id") is None
                    or item.get("physical_unit_id") == required.get("physical_unit_id")
                )
            ),
            None,
        )
        minimum_success_count = int(required.get("minimum_success_count") or 0)
        checks.append(
            {
                "operation": required.get("operation"),
                "logical_artifact_id": required.get("logical_artifact_id"),
                "physical_unit_id": required.get("physical_unit_id"),
                "comparison_level": required.get("comparison_level"),
                "minimum_success_count": minimum_success_count,
                "observed": match,
                "passed": bool(
                    match is not None
                    and _int_or_zero(match.get("success_count")) >= minimum_success_count
                    and (
                        required.get("current_outcome") is None
                        or match.get("current_outcome") == required.get("current_outcome")
                    )
                    and (
                        required.get("comparison_level") is None
                        or match.get("comparison_level") == required.get("comparison_level")
                    )
                ),
            }
        )
    return checks


def score_l0b_against_gold(
    l0_bundle: dict[str, Any],
    l0_model_view: dict[str, Any],
    gold: dict[str, Any],
) -> dict[str, Any] | None:
    expectation = gold.get("l0b_expectation") or {}
    if not isinstance(expectation, dict) or not expectation:
        return None

    evidence_bundle = l0_model_view.get("evidence_bundle") or {}
    if not isinstance(evidence_bundle, dict):
        evidence_bundle = {}
    line_tolerance = int(expectation.get("line_tolerance") or 0)
    required_lines = [int(value) for value in expectation.get("required_evidence_lines") or []]
    visible_lines = _explicit_line_values(evidence_bundle.get("context_windows") or [])
    required_line_recall = _required_line_recall(
        visible_lines,
        required_lines,
        line_tolerance,
    )

    accepted_primary_lines = [
        int(value)
        for value in (
            expectation.get("accepted_primary_lines")
            or (gold.get("l0_expectation") or {}).get("accepted_primary_lines")
            or (gold.get("primary_anchor_expectation") or {}).get("accepted_lines")
            or []
        )
    ]
    primary_retained = _line_set_matches(
        visible_lines,
        accepted_primary_lines,
        line_tolerance,
    )
    l0a_primary_covered = _line_set_matches(
        _l0_candidate_evidence_lines(l0_bundle),
        accepted_primary_lines,
        line_tolerance,
    )
    if l0a_primary_covered is not True:
        primary_retained = None

    required_references = expectation.get("required_reference_ids") or {}
    if not isinstance(required_references, dict):
        required_references = {}
    observed_references = _l0b_reference_ids(evidence_bundle)
    reference_checks = {
        str(kind): set(str(value) for value in values or []).issubset(
            observed_references.get(str(kind), set())
        )
        for kind, values in required_references.items()
    }
    projection = l0_model_view.get("projection_metrics") or {}
    if not isinstance(projection, dict):
        projection = {}
    integrity = projection.get("projection_integrity") or {}
    if not isinstance(integrity, dict):
        integrity = {}
    integrity_ok = integrity.get("status") == "ok" if integrity else None
    return {
        "required_evidence_lines": required_lines,
        "visible_context_line_count": len(visible_lines),
        "required_evidence_line_recall": required_line_recall,
        "accepted_primary_lines": accepted_primary_lines,
        "primary_retained_from_l0a": primary_retained,
        "required_reference_checks": reference_checks,
        "projection_integrity": integrity,
        "projection_integrity_ok": integrity_ok,
        "overall_pass": (
            required_line_recall is not False
            and primary_retained is not False
            and all(reference_checks.values())
            and integrity_ok is not False
        ),
    }


def _l0_candidate_evidence_lines(l0_bundle: dict[str, Any]) -> set[int]:
    lines: set[int] = set()
    for name in (
        "candidate_anchors",
        "context_windows",
        "registry_matches",
        "occurrence_groups",
        "failure_episodes",
        "distributed_failure_incidents",
    ):
        lines.update(_explicit_line_values(l0_bundle.get(name) or []))
    return lines


def _l0b_reference_ids(evidence_bundle: dict[str, Any]) -> dict[str, set[str]]:
    collections = {
        "context_window_ids": ("context_windows", "window_id"),
        "candidate_anchor_ids": ("candidate_anchors", "anchor_id"),
        "occurrence_group_ids": ("occurrence_groups", "occurrence_group_id"),
        "failure_episode_ids": ("failure_episodes", "episode_id"),
        "distributed_incident_ids": (
            "distributed_failure_incidents",
            "incident_id",
        ),
    }
    result: dict[str, set[str]] = {}
    for output_name, (collection_name, id_name) in collections.items():
        result[output_name] = {
            str(item[id_name])
            for item in evidence_bundle.get(collection_name) or []
            if isinstance(item, dict) and item.get(id_name) is not None
        }
    return result


def _explicit_line_values(value: Any) -> set[int]:
    lines: set[int] = set()
    if isinstance(value, list):
        for item in value:
            lines.update(_explicit_line_values(item))
        return lines
    if not isinstance(value, dict):
        return lines
    for key, item in value.items():
        key_text = str(key)
        if key_text == "line" or key_text.endswith("_line"):
            line = _int_or_none(item)
            if line is not None:
                lines.add(line)
        elif key_text.endswith("_lines") and isinstance(item, list):
            lines.update(line for line in (_int_or_none(entry) for entry in item) if line)
        elif isinstance(item, (dict, list)):
            lines.update(_explicit_line_values(item))
    return lines


def _named_line_values(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): line
        for key, item in value.items()
        if (str(key) == "line" or str(key).endswith("_line"))
        and (line := _int_or_none(item)) is not None
    }


def _required_line_recall(
    observed_lines: set[int],
    required_lines: list[int],
    tolerance: int,
) -> bool | None:
    if not required_lines:
        return None
    return all(
        any(abs(observed - required) <= tolerance for observed in observed_lines)
        for required in required_lines
    )


def _line_set_matches(
    observed_lines: set[int],
    expected_lines: list[int],
    tolerance: int,
) -> bool | None:
    if not expected_lines:
        return None
    return any(
        abs(observed - expected) <= tolerance
        for observed in observed_lines
        for expected in expected_lines
    )


def _line_matches(
    observed_line: int | None,
    expected_lines: list[int],
    tolerance: int,
) -> bool | None:
    if not expected_lines:
        return None
    if observed_line is None:
        return False
    return any(abs(observed_line - expected) <= tolerance for expected in expected_lines)


def score_semantic_view(
    view: dict[str, Any],
    gold: dict[str, Any],
    *,
    include_action: bool,
) -> dict[str, Any]:
    root = _score_root_cause(view, gold)
    recovery = _score_recovery_assessment(view, gold)
    policy = _score_policy(view, gold, include_action=include_action)
    downstream = _score_downstream_roles(view, gold)
    unsupported = _unsupported_claims(root.searchable_text, gold)

    core_checks = [root.correct]
    if recovery.correct is not None:
        core_checks.append(recovery.correct)
    core_semantic_pass = all(core_checks) and not unsupported
    checks = list(core_checks)
    if downstream.correct is not None:
        checks.append(downstream.correct)
    policy_action_checks: list[bool] = []
    if include_action:
        policy_action_checks.append(bool(policy.action_correct))
        for value in (
            policy.retry_rule_correct,
            policy.allowed_retries_correct,
            policy.retry_exhaustion_correct,
        ):
            if value is not None:
                policy_action_checks.append(value)
        checks.extend(policy_action_checks)
    return {
        "primary_anchor_correct": root.primary_anchor_correct,
        "root_cause_correct": root.correct,
        "root_cause_concept_hits": root.concept_hits,
        "root_cause_uncertainty_preserved": root.uncertainty_preserved,
        "root_cause_operation_correct": root.operation_correct,
        "root_cause_mechanism_contradiction": root.mechanism_contradiction,
        "recovery_assessment_correct": recovery.correct,
        "recovery_field_results": recovery.field_results,
        "action_correct": policy.action_correct,
        "retry_rule_correct": policy.retry_rule_correct,
        "allowed_retries_correct": policy.allowed_retries_correct,
        "retry_exhaustion_correct": policy.retry_exhaustion_correct,
        "cascade_correct": downstream.correct,
        "cascade_line_correct": downstream.cascade_line_correct,
        "teardown_role_correct": downstream.teardown_role_correct,
        "related_failure_recall": downstream.correct,
        "unsupported_claims": unsupported,
        "model_recovery_confidence": recovery.confidence,
        "core_semantic_pass": core_semantic_pass,
        "policy_action_pass": (all(policy_action_checks) if policy_action_checks else None),
        "overall_semantic_pass": all(checks) and not unsupported,
    }


@dataclass(frozen=True)
class _RootCauseScore:
    primary_anchor_correct: bool
    correct: bool
    concept_hits: list[bool]
    uncertainty_preserved: bool
    operation_correct: bool | None
    mechanism_contradiction: bool
    searchable_text: str


@dataclass(frozen=True)
class _RecoveryScore:
    correct: bool | None
    field_results: dict[str, bool | None]
    confidence: Any


@dataclass(frozen=True)
class _PolicyScore:
    action_correct: bool | None
    retry_rule_correct: bool | None
    allowed_retries_correct: bool | None
    retry_exhaustion_correct: bool | None


@dataclass(frozen=True)
class _DownstreamScore:
    correct: bool | None
    cascade_line_correct: bool | None
    teardown_role_correct: bool | None


def _score_root_cause(view: dict[str, Any], gold: dict[str, Any]) -> _RootCauseScore:
    primary = view.get("primary_failure") or {}
    primary = primary if isinstance(primary, dict) else {}
    expectation = gold.get("primary_anchor_expectation") or {}
    accepted_lines = [int(line) for line in expectation.get("accepted_lines") or []]
    tolerance = int(expectation.get("tolerance_lines") or 0)
    primary_line = _int_or_none(primary.get("line"))
    primary_correct = bool(
        primary_line is not None
        and any(abs(primary_line - expected) <= tolerance for expected in accepted_lines)
    )
    root_cause = view.get("root_cause_assessment") or {}
    root_cause = root_cause if isinstance(root_cause, dict) else {}
    recovery = view.get("model_recovery_assessment") or {}
    recovery = recovery if isinstance(recovery, dict) else {}
    failure_identity = primary.get("failure_identity") or {}
    failure_identity = failure_identity if isinstance(failure_identity, dict) else {}
    searchable = " ".join(
        str(value)
        for value in (
            primary.get("fine_class"),
            primary.get("signature"),
            failure_identity.get("operation"),
            failure_identity.get("mechanism"),
            failure_identity.get("component"),
            root_cause.get("summary"),
            root_cause.get("plausible_causes"),
            root_cause.get("missing_evidence"),
            recovery.get("rationale"),
            view.get("justification"),
        )
        if value is not None
    ).casefold()
    root_expectation = gold.get("root_cause_expectation") or {}
    concept_hits = [
        any(str(term).casefold() in searchable for term in group)
        for group in root_expectation.get("required_concept_groups") or []
        if isinstance(group, list)
    ]
    uncertainty_terms = [
        str(term).casefold() for term in root_expectation.get("uncertainty_terms_any") or []
    ]
    uncertainty_preserved = bool(root_cause.get("missing_evidence") or []) or any(
        term in searchable for term in uncertainty_terms
    )
    if not root_expectation.get("require_uncertainty_preserved"):
        uncertainty_preserved = True
    accepted_operations = {
        _normalized_identity_value(value)
        for value in root_expectation.get("accepted_operations") or []
    }
    observed_operation = _normalized_identity_value(failure_identity.get("operation"))
    operation_correct = observed_operation in accepted_operations if accepted_operations else None
    mechanism_terms = set(_normalized_identity_value(failure_identity.get("mechanism")).split("_"))
    rejected_mechanism_terms = {
        _normalized_identity_value(value)
        for value in root_expectation.get("rejected_mechanism_terms") or []
    }
    mechanism_contradiction = bool(mechanism_terms.intersection(rejected_mechanism_terms))
    semantic_checks = [primary_correct, all(concept_hits), uncertainty_preserved]
    if operation_correct is not None:
        semantic_checks.append(operation_correct)
    semantic_checks.append(not mechanism_contradiction)
    return _RootCauseScore(
        primary_correct,
        all(semantic_checks),
        concept_hits,
        uncertainty_preserved,
        operation_correct,
        mechanism_contradiction,
        searchable,
    )


def _score_recovery_assessment(view: dict[str, Any], gold: dict[str, Any]) -> _RecoveryScore:
    assessment = view.get("model_recovery_assessment") or {}
    assessment = assessment if isinstance(assessment, dict) else {}
    expectation = gold.get("recovery_assessment_expectation") or {}
    field_results: dict[str, bool | None] = {}
    for name in (
        "failure_domain",
        "failure_domain_status",
        "retry_outlook_without_workload_change",
        "retry_outlook_status",
    ):
        expected = expectation.get(name)
        accepted = (
            {str(value) for value in expected}
            if isinstance(expected, list)
            else {str(expected)} if expected is not None else set()
        )
        actual = _recovery_claim_field(assessment, name)
        field_results[name] = str(actual) in accepted if accepted else None
    scored = [value for value in field_results.values() if value is not None]
    return _RecoveryScore(
        all(scored) if scored else None,
        field_results,
        {
            "failure_domain": _recovery_claim_field(
                assessment,
                "failure_domain_confidence",
            ),
            "retry_outlook": _recovery_claim_field(
                assessment,
                "retry_outlook_confidence",
            ),
        },
    )


def _recovery_claim_field(assessment: dict[str, Any], field: str) -> Any:
    if field.startswith("failure_domain"):
        claim = assessment.get("failure_domain") or {}
    else:
        claim = assessment.get("retry_outlook_without_workload_change") or {}
    if not isinstance(claim, dict):
        return None
    if field.endswith("_status"):
        return claim.get("status")
    if field.endswith("_confidence"):
        return claim.get("confidence")
    return claim.get("value")


def _score_policy(
    view: dict[str, Any],
    gold: dict[str, Any],
    *,
    include_action: bool,
) -> _PolicyScore:
    retry_policy = view.get("retry_policy") or {}
    retry_policy = retry_policy if isinstance(retry_policy, dict) else {}
    expectation = gold.get("retry_policy_expectation") or {}
    accepted_rules = {str(value) for value in expectation.get("accepted_rules") or []}
    expected_retries = expectation.get("allowed_retries")
    expected_exhausted = expectation.get("retry_budget_exhausted")
    accepted_actions = {
        str(value) for value in (gold.get("action_expectation") or {}).get("accepted") or []
    }
    return _PolicyScore(
        (str(view.get("decision") or "") in accepted_actions if include_action else None),
        str(retry_policy.get("rule")) in accepted_rules if accepted_rules else None,
        (
            _int_or_none(retry_policy.get("allowed_retries")) == _int_or_none(expected_retries)
            if expected_retries is not None
            else None
        ),
        (
            bool(retry_policy.get("retry_budget_exhausted")) == bool(expected_exhausted)
            if expected_exhausted is not None
            else None
        ),
    )


def _score_downstream_roles(view: dict[str, Any], gold: dict[str, Any]) -> _DownstreamScore:
    expectation = gold.get("cascade_expectation") or {}
    cascades = [item for item in view.get("cascades") or [] if isinstance(item, dict)]
    related = [
        item
        for item in [
            *(view.get("related_failures") or []),
            *(view.get("secondary_failures") or []),
        ]
        if isinstance(item, dict)
    ]
    cascade_lines = [int(line) for line in expectation.get("expected_lines") or []]
    teardown_lines = [int(line) for line in expectation.get("teardown_lines") or []]
    expected_groups = [
        item for item in expectation.get("expected_groups") or [] if isinstance(item, dict)
    ]
    cascade_correct = _downstream_lines_match(
        cascade_lines,
        cascades,
        related,
        require_teardown=False,
    )
    teardown_correct = _downstream_lines_match(
        teardown_lines,
        cascades,
        related,
        require_teardown=True,
    )
    group_correct = _downstream_groups_match(expected_groups, cascades)
    checks = [
        value for value in (cascade_correct, teardown_correct, group_correct) if value is not None
    ]
    return _DownstreamScore(all(checks) if checks else None, cascade_correct, teardown_correct)


def _downstream_groups_match(
    expected_groups: list[dict[str, Any]],
    cascades: list[dict[str, Any]],
) -> bool | None:
    if not expected_groups:
        return None
    return all(
        any(
            item.get("causal_role") == expected.get("causal_role")
            and _int_or_none(item.get("first_line")) == _int_or_none(expected.get("first_line"))
            and _int_or_zero(item.get("count")) >= _int_or_zero(expected.get("minimum_count"))
            and len(item.get("rank_spread") or [])
            >= _int_or_zero(expected.get("minimum_rank_count"))
            for item in cascades
        )
        for expected in expected_groups
    )


def _downstream_lines_match(
    expected_lines: list[int],
    cascades: list[dict[str, Any]],
    related: list[dict[str, Any]],
    *,
    require_teardown: bool,
) -> bool | None:
    if not expected_lines:
        return None
    allowed_roles = {"teardown"} if require_teardown else {"cascade", "teardown"}
    return all(
        any(
            _int_or_zero(item.get("first_line")) <= expected <= _int_or_zero(item.get("last_line"))
            and (not require_teardown or item.get("causal_role") == "teardown")
            for item in cascades
        )
        or any(
            _int_or_none(item.get("line")) == expected and item.get("causal_role") in allowed_roles
            for item in related
        )
        for expected in expected_lines
    )


def _unsupported_claims(searchable: str, gold: dict[str, Any]) -> list[str]:
    result = []
    for claim in gold.get("unsupported_claims") or []:
        if not isinstance(claim, dict):
            continue
        patterns = [str(value).casefold() for value in claim.get("text_patterns") or []]
        if any(pattern in searchable for pattern in patterns):
            result.append(str(claim.get("id") or "unsupported_claim"))
    return result


def _reference_audit_effect(
    l1_score: dict[str, Any],
    l2_score: dict[str, Any] | None,
) -> str:
    if l2_score is None:
        return "not_comparable"
    fields = ("primary_anchor_correct", "root_cause_correct", "policy_correct")
    before = all(bool(l1_score.get(field)) for field in fields) and not bool(
        l1_score.get("unsupported_claims")
    )
    after = all(bool(l2_score.get(field)) for field in fields) and not bool(
        l2_score.get("unsupported_claims")
    )
    if before == after:
        return "neutral"
    return "helped" if after else "harmed"


def score_l2_audit(
    l2_audit: dict[str, Any],
    gold: dict[str, Any],
) -> bool | None:
    expectations = gold.get("l2_audit_expectation") or []
    if not expectations:
        return None
    field_audits = l2_audit.get("field_audits") or {}
    grounding_adjustments = [
        item for item in l2_audit.get("grounding_adjustments") or [] if isinstance(item, dict)
    ]
    checks: list[bool] = []
    for expectation in expectations:
        if not isinstance(expectation, dict):
            continue
        field = str(expectation.get("field") or "")
        expected = str(expectation.get("expected") or "")
        actual_audit = field_audits.get(field) or {}
        check = actual_audit.get("status") == expected
        if expected == "resolved" and "normalized_value" in expectation:
            check = check and any(
                item.get("field") == field and item.get("to") == expectation.get("normalized_value")
                for item in grounding_adjustments
            )
        reason_class = expectation.get("reason_class")
        if reason_class:
            check = check and str(reason_class) in {
                str(value) for value in actual_audit.get("finding_classes") or []
            }
        checks.append(check)
    return all(checks) if checks else None


def _score_l2_history_identity(
    l2_audit: dict[str, Any],
    l0_bundle: dict[str, Any],
    gold: dict[str, Any],
) -> dict[str, Any]:
    expectation = gold.get("history_identity_expectation") or {}
    if not isinstance(expectation, dict):
        expectation = {}
    observed = l2_audit.get("stable_root_fingerprint")
    expected_anchor = _int_or_none(expectation.get("canonical_anchor_line"))
    observed_anchor = _int_or_none(l2_audit.get("stable_identity_anchor_line"))
    anchor_correct = observed_anchor == expected_anchor if expected_anchor is not None else None
    expected_operation = _normalized_identity_value(expectation.get("operation"))
    observed_operations = {
        _normalized_identity_value(item.get("operation"))
        for item in l0_bundle.get("operation_artifact_comparisons") or []
        if isinstance(item, dict) and item.get("operation")
    }
    operation_correct = expected_operation in observed_operations if expected_operation else None
    expected_mechanism_terms = set(
        _normalized_identity_value(expectation.get("mechanism")).split("_")
    ) - {""}
    observed_mechanism_terms = set(_normalized_identity_value(observed).split("_")) - {""}
    mechanism_correct = (
        expected_mechanism_terms.issubset(observed_mechanism_terms)
        if expected_mechanism_terms
        else None
    )
    checks = [
        value
        for value in (anchor_correct, operation_correct, mechanism_correct)
        if value is not None
    ]
    return {
        "observed_root_fingerprint": observed,
        "canonical_anchor_correct": anchor_correct,
        "operation_correct": operation_correct,
        "mechanism_correct": mechanism_correct,
        "history_identity_correct": all(checks) if checks else None,
        "expected_cross_route_identity_count": expectation.get(
            "expected_cross_route_identity_count"
        ),
    }


def _normalized_identity_value(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")
    aliases = {
        "checkpointing": "checkpoint",
        "pos": "position",
        "unexpected": "mismatch",
    }
    return "_".join(aliases.get(token, token) for token in normalized.split("_") if token)


def path_redaction_audit(source_log: Path, interaction_transcript: Any) -> dict[str, Any]:
    if not isinstance(interaction_transcript, list):
        interaction_transcript = []
    source_log = source_log.resolve()
    full_path = str(source_log)
    tokens = {full_path, source_log.name}
    tokens.update(
        part
        for part in source_log.parts[-5:-1]
        if part not in {"/", ""}
        and part.casefold() not in GENERIC_PATH_COMPONENTS
        and len(part) >= MIN_DISTINCTIVE_PATH_COMPONENT_LENGTH
    )
    tokens = {token for token in tokens if token}
    transcript_text = "\n".join(_redaction_visible_text(interaction_transcript))
    candidates = sorted(
        token
        for token in tokens
        if _contains_path_token(transcript_text, token, full_path=token == full_path)
    )
    try:
        source_text = source_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        source_text = ""
    source_content_overlap = sorted(
        token
        for token in candidates
        if _contains_path_token(source_text, token, full_path=token == full_path)
    )
    leaked = sorted(set(candidates).difference(source_content_overlap))
    return {
        "checked_model_interaction_transcript": True,
        "transcript_source_path_candidates": candidates,
        "source_content_overlap_tokens": source_content_overlap,
        "source_path_tokens_found": leaked,
        "passed": not leaked,
    }


def _redaction_visible_text(value: Any, *, field_name: str | None = None) -> list[str]:
    if field_name in OPAQUE_TRANSCRIPT_FIELDS:
        return []
    if isinstance(value, dict):
        text: list[str] = []
        for key, item in value.items():
            text.extend(_redaction_visible_text(item, field_name=str(key)))
        return text
    if isinstance(value, list):
        text = []
        for item in value:
            text.extend(_redaction_visible_text(item))
        return text
    return [value] if isinstance(value, str) else []


def _contains_path_token(text: str, token: str, *, full_path: bool) -> bool:
    if full_path:
        return token in text
    boundary = r"[A-Za-z0-9_]"
    return (
        re.search(
            rf"(?<!{boundary}){re.escape(token)}(?!{boundary})",
            text,
        )
        is not None
    )


def line_numbering_summary(source_log: Path) -> dict[str, Any]:
    try:
        data = source_log.read_bytes()
    except OSError as exc:
        return {"error": str(exc)}
    return {
        "lf_count": data.count(b"\n"),
        "cr_count": data.count(b"\r"),
        "crlf_count": data.count(b"\r\n"),
        "splitlines_count": len(data.decode("utf-8", errors="replace").splitlines()),
    }


def model_call_summary(model_calls: Any) -> dict[str, Any]:
    stats = _ModelCallStats()
    for call in model_calls if isinstance(model_calls, list) else []:
        if isinstance(call, dict):
            stats.observe(call)
    return stats.to_payload()


@dataclass
class _ModelCallStats:
    calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    retried_calls: int = 0
    timeout_calls: int = 0
    http_error_calls: int = 0
    context_window_error_calls: int = 0
    retry_after_s_total: float = 0.0
    latency_s_total: float = 0.0
    failed_latency_s_total: float = 0.0
    provider_timing_call_count: int = 0
    model_turns: set[int] = field(default_factory=set)
    finish_reasons: dict[str, int] = field(default_factory=dict)
    provider_errors: list[dict[str, Any]] = field(default_factory=list)
    context_budgets: list[dict[str, Any]] = field(default_factory=list)
    provider_timing_totals_ms: dict[str, float] = field(default_factory=dict)

    def observe(self, call: dict[str, Any]) -> None:
        self.calls += 1
        model_turn = _int_or_none(call.get("model_turn"))
        if model_turn is not None:
            self.model_turns.add(model_turn)
        latency_s = _float_or_zero(call.get("latency_s"))
        self.latency_s_total += latency_s
        if call.get("success"):
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            self.failed_latency_s_total += latency_s
        self.retried_calls += int(bool(call.get("retry_scheduled")))
        self.timeout_calls += int(bool(call.get("timeout")))
        self.http_error_calls += int(call.get("http_status") is not None)
        self.retry_after_s_total += _float_or_zero(call.get("retry_after_s"))
        error_type = str(call.get("error_type") or "")
        context_window_error = error_type == "context_window_exceeded"
        self.context_window_error_calls += int(context_window_error)
        self._observe_context_budget(call)
        self._observe_provider_timing(call)
        finish_reason = call.get("finish_reason")
        if finish_reason:
            key = str(finish_reason)
            self.finish_reasons[key] = self.finish_reasons.get(key, 0) + 1
        if (call.get("error") or call.get("http_status")) and not context_window_error:
            self.provider_errors.append(_provider_error_summary(call))

    def _observe_context_budget(self, call: dict[str, Any]) -> None:
        budget = call.get("context_budget")
        if isinstance(budget, dict) and budget:
            self.context_budgets.append(budget)

    def _observe_provider_timing(self, call: dict[str, Any]) -> None:
        timing = call.get("provider_reported_timing")
        if not isinstance(timing, dict):
            return
        reported_components = 0
        for name in (
            "downstream_llm_api_ms",
            "proxy_pre_processing_ms",
            "proxy_post_processing_ms",
            "proxy_message_copy_ms",
        ):
            if timing.get(name) is None:
                continue
            self.provider_timing_totals_ms[name] = self.provider_timing_totals_ms.get(
                name, 0.0
            ) + _float_or_zero(timing.get(name))
            reported_components += 1
        self.provider_timing_call_count += int(bool(reported_components))

    def to_payload(self) -> dict[str, Any]:
        result = {
            "calls": self.calls,
            "model_turns": len(self.model_turns),
            "extra_model_turns_after_initial": max(0, len(self.model_turns) - 1),
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "retried_calls": self.retried_calls,
            "timeout_calls": self.timeout_calls,
            "http_error_calls": self.http_error_calls,
            "context_window_error_calls": self.context_window_error_calls,
            "endpoint_failed_calls": max(0, self.failed_calls - self.context_window_error_calls),
            "finish_reasons": self.finish_reasons,
            "retry_after_s_total": round(self.retry_after_s_total, 3),
            "latency_s_total": round(self.latency_s_total, 3),
            "failed_latency_s_total": round(self.failed_latency_s_total, 3),
            "provider_error_count": len(self.provider_errors),
            "provider_errors": self.provider_errors,
            **_context_budget_summary(self.context_budgets),
        }
        if self.provider_timing_call_count:
            result["provider_reported_timing"] = {
                "source": "response_headers",
                "reported_call_count": self.provider_timing_call_count,
                "components_ms_total": {
                    name: round(value, 3) for name, value in self.provider_timing_totals_ms.items()
                },
            }
        return result


def _provider_error_summary(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_turn": call.get("model_turn"),
        "attempt": call.get("attempt"),
        "latency_s": call.get("latency_s"),
        "error": call.get("error"),
        "error_type": call.get("error_type"),
        "http_status": call.get("http_status"),
        "retryable": call.get("retryable"),
        "retry_scheduled": call.get("retry_scheduled"),
        "retry_after_s": call.get("retry_after_s"),
        "response_body_sample": _truncate_text(str(call.get("response_body") or ""), 500),
    }


def _context_budget_summary(context_budgets: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "context_budget_adjusted_calls": sum(
            1 for budget in context_budgets if budget.get("adjusted")
        ),
        "context_window_tokens": next(
            (
                budget.get("context_window_tokens")
                for budget in context_budgets
                if budget.get("context_window_tokens") is not None
            ),
            None,
        ),
        "max_estimated_input_tokens": max(
            (_int_or_zero(budget.get("estimated_input_tokens")) for budget in context_budgets),
            default=0,
        ),
        "configured_max_output_tokens": max(
            (
                _int_or_zero(budget.get("configured_max_output_tokens"))
                for budget in context_budgets
            ),
            default=0,
        ),
        "minimum_effective_max_output_tokens": min(
            (
                _int_or_zero(budget.get("effective_max_output_tokens"))
                for budget in context_budgets
                if budget.get("effective_max_output_tokens") is not None
            ),
            default=0,
        ),
    }


def _structured_fact_lines(l0_bundle: dict[str, Any]) -> dict[int, list[str]]:
    bundle = l0_bundle.get("bundle") if isinstance(l0_bundle.get("bundle"), dict) else l0_bundle
    facts: dict[int, list[str]] = {}

    def add(line: Any, label: str) -> None:
        line_no = _int_or_none(line)
        if line_no is None:
            return
        labels = facts.setdefault(line_no, [])
        if label not in labels:
            labels.append(label)

    job_metadata = bundle.get("job_metadata") or {}
    if isinstance(job_metadata, dict):
        add(
            job_metadata.get("explicit_world_size_line"),
            "job_metadata.explicit_world_size",
        )

    progress_summary = bundle.get("run_progress_summary") or {}
    if isinstance(progress_summary, dict):
        for field in (
            "first_iteration_line",
            "last_iteration_line",
            "last_checkpoint_line",
            "checkpoint_load_line",
            "last_setup_line",
            "first_terminal_incident_line",
        ):
            add(progress_summary.get(field), f"run_progress_summary.{field}")

    progress = bundle.get("progress") or {}
    if isinstance(progress, dict):
        for group_name in (
            "progress_markers",
            "checkpoint_markers",
            "setup_markers",
        ):
            for marker in progress.get(group_name) or []:
                if not isinstance(marker, dict):
                    continue
                marker_type = marker.get("marker_type") or "unknown"
                add(marker.get("line"), f"progress.{group_name}.{marker_type}")

    for item in bundle.get("operation_artifact_comparisons") or []:
        if not isinstance(item, dict):
            continue
        operation = str(item.get("operation") or "unknown")
        for line in item.get("success_lines") or []:
            add(line, f"operation_artifact_comparisons.{operation}.success")
        add(
            item.get("current_start_line"),
            f"operation_artifact_comparisons.{operation}.current_start",
        )
        add(
            item.get("current_completion_line"),
            f"operation_artifact_comparisons.{operation}.current_completion",
        )
        add(
            item.get("failure_line"),
            f"operation_artifact_comparisons.{operation}.failure",
        )

    for item in bundle.get("distributed_failure_incidents") or []:
        if not isinstance(item, dict):
            continue
        incident_type = str(item.get("incident_type") or "unknown")
        add(
            item.get("primary_observed_line"),
            f"distributed_failure_incident.{incident_type}.primary",
        )

    return facts


@dataclass(frozen=True)
class ToolEfficiencyInput:
    """Normalized route payloads used by tool-efficiency scoring."""

    model_calls: tuple[Any, ...]
    tool_calls: tuple[Any, ...]
    timing: dict[str, Any]
    interaction_transcript: tuple[Any, ...]
    analysis: dict[str, Any]
    l0_bundle: dict[str, Any]

    @classmethod
    def from_payloads(
        cls,
        *,
        l1: dict[str, Any],
        timing: dict[str, Any],
        interaction_transcript: Any,
        analysis: dict[str, Any],
        l0_bundle: dict[str, Any] | None,
    ) -> "ToolEfficiencyInput":
        model_calls = l1.get("model_calls") or []
        tool_calls = l1.get("tool_calls") or []
        return cls(
            model_calls=tuple(model_calls) if isinstance(model_calls, list) else (),
            tool_calls=tuple(tool_calls) if isinstance(tool_calls, list) else (),
            timing=dict(timing) if isinstance(timing, dict) else {},
            interaction_transcript=(
                tuple(interaction_transcript) if isinstance(interaction_transcript, list) else ()
            ),
            analysis=dict(analysis) if isinstance(analysis, dict) else {},
            l0_bundle=dict(l0_bundle) if isinstance(l0_bundle, dict) else {},
        )


def tool_efficiency_summary(
    *,
    l1: dict[str, Any],
    timing: dict[str, Any],
    interaction_transcript: Any,
    analysis: dict[str, Any],
    l0_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = ToolEfficiencyInput.from_payloads(
        l1=l1,
        timing=timing,
        interaction_transcript=interaction_transcript,
        analysis=analysis,
        l0_bundle=l0_bundle,
    )
    return _tool_efficiency_from_input(context)


def _tool_efficiency_from_input(context: ToolEfficiencyInput) -> dict[str, Any]:
    model_calls = context.model_calls
    tool_calls = context.tool_calls
    timing = context.timing
    interaction_transcript = list(context.interaction_transcript)
    analysis = context.analysis

    prompted_lines = _prompt_excerpt_line_numbers(interaction_transcript)
    structured_fact_lines = _structured_fact_lines(context.l0_bundle)
    per_call, all_tool_result_lines = _classify_tool_calls(
        tool_calls,
        interaction_transcript=interaction_transcript,
        prompted_lines=prompted_lines,
        structured_fact_lines=structured_fact_lines,
    )

    distinct_result_lines = sorted(
        {
            line
            for item in per_call
            for line in item.get("result_line_numbers_sample", [])
            if isinstance(line, int)
        }
    )
    new_prompt_lines_total = sum(
        _int_or_zero(item.get("new_prompt_excerpt_line_count")) for item in per_call
    )
    already_prompt_lines_total = sum(
        _int_or_zero(item.get("already_in_prompt_excerpt_line_count")) for item in per_call
    )
    duplicate_prompt_context_calls = sum(
        1 for item in per_call if "duplicate_prompt_context" in item.get("flags", [])
    )
    no_new_prompt_line_calls = sum(
        1
        for item in per_call
        if item.get("result_lines") and _int_or_zero(item.get("new_prompt_excerpt_line_count")) == 0
    )
    total_latency_ms = sum(
        _float_or_zero(call.get("latency_ms")) for call in tool_calls if isinstance(call, dict)
    )
    model_turn_values = [
        _int_or_none(call.get("model_turn"))
        for call in [*model_calls, *tool_calls]
        if isinstance(call, dict)
    ]
    model_turn_values = [turn for turn in model_turn_values if turn is not None]
    distinct_model_turns = set(model_turn_values)
    tool_model_turns = {
        turn
        for call in tool_calls
        if isinstance(call, dict)
        for turn in [_int_or_none(call.get("model_turn"))]
        if turn is not None
    }
    tool_only_lines = all_tool_result_lines.difference(prompted_lines)
    primary = analysis.get("primary_failure") or {}
    primary_line = _int_or_none(primary.get("line")) if isinstance(primary, dict) else None
    final_evidence_lines = {
        line
        for item in analysis.get("evidence") or []
        if isinstance(item, dict)
        for line in [_int_or_none(item.get("line"))]
        if line is not None
    }
    tool_only_evidence_lines = sorted(final_evidence_lines.intersection(tool_only_lines))
    structured_fact_redundant_tool_only_lines = sorted(
        tool_only_lines.intersection(structured_fact_lines)
    )
    primary_from_tool_only = primary_line in tool_only_lines if primary_line is not None else False
    context_impact = _tool_only_context_impact(
        analysis=analysis,
        tool_only_evidence_lines=tool_only_evidence_lines,
        primary_from_tool_only=primary_from_tool_only,
        structured_fact_redundant_lines=structured_fact_redundant_tool_only_lines,
    )
    unused_tool_only_lines = sorted(
        tool_only_lines.difference(final_evidence_lines).difference(structured_fact_lines)
    )
    if not per_call:
        final_context_dependency = "no_tools"
    elif primary_from_tool_only or tool_only_evidence_lines:
        final_context_dependency = "final_evidence_depends_on_tool_only_lines"
    else:
        final_context_dependency = "no_final_citation_dependency_observed"

    return {
        "calls": len(per_call),
        "model_calls": len(model_calls),
        "model_turns": len(distinct_model_turns),
        "extra_model_turns_after_initial": max(0, len(distinct_model_turns) - 1),
        "tool_driven_model_turns": len(tool_model_turns),
        "max_model_turn": max(model_turn_values) if model_turn_values else None,
        "total_tool_latency_ms": round(total_latency_ms, 3),
        "total_tool_latency_s": round(total_latency_ms / 1000.0, 3),
        "total_result_lines": sum(_int_or_zero(item.get("result_lines")) for item in per_call),
        "total_result_chars": sum(_int_or_zero(item.get("result_chars")) for item in per_call),
        "distinct_result_line_count_sampled": len(distinct_result_lines),
        "new_prompt_excerpt_line_count": new_prompt_lines_total,
        "unique_new_prompt_excerpt_line_count": len(tool_only_lines),
        "new_line_existing_structured_fact_count": sum(
            _int_or_zero(item.get("new_line_existing_structured_fact_count")) for item in per_call
        ),
        "already_in_prompt_excerpt_line_count": already_prompt_lines_total,
        "duplicate_prompt_context_calls": duplicate_prompt_context_calls,
        "no_new_prompt_line_calls": no_new_prompt_line_calls,
        "error_calls": sum(1 for item in per_call if "error" in item.get("flags", [])),
        "truncated_calls": sum(1 for item in per_call if "truncated" in item.get("flags", [])),
        "tool_wall_clock_s": timing.get("l1_tool_wall_clock_s"),
        "model_call_wall_clock_s": timing.get("l1_model_call_wall_clock_s"),
        "optimization_note": _tool_optimization_note(
            calls=len(per_call),
            duplicate_prompt_context_calls=duplicate_prompt_context_calls,
            no_new_prompt_line_calls=no_new_prompt_line_calls,
        ),
        "final_context_dependency": final_context_dependency,
        "final_context_impact": context_impact["impact"],
        "final_primary_from_tool_only_context": primary_from_tool_only,
        "final_evidence_tool_only_lines": tool_only_evidence_lines,
        "decision_relevant_tool_only_lines": context_impact["decision_relevant_lines"],
        "structured_fact_redundant_tool_only_lines": (
            context_impact["structured_fact_redundant_lines"]
        ),
        "structured_fact_redundant_tool_only_line_labels": {
            str(line): structured_fact_lines.get(line, [])
            for line in context_impact["structured_fact_redundant_lines"]
        },
        "incidental_tool_only_lines": context_impact["incidental_lines"],
        "unused_tool_only_lines": unused_tool_only_lines,
        "attribution_or_policy_change_measurable": False,
        "attribution_or_policy_change_note": (
            "Tool-request turns do not contain a required provisional assessment; "
            "exact before/after attribution or policy change is therefore not observable."
        ),
        "per_call": per_call,
    }


def _classify_tool_calls(
    tool_calls: list[Any],
    *,
    interaction_transcript: list[Any],
    prompted_lines: set[int],
    structured_fact_lines: dict[int, list[str]],
) -> tuple[list[dict[str, Any]], set[int]]:
    result_events = _tool_result_events_by_id(interaction_transcript)
    all_result_lines: set[int] = set()
    classified: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        event = result_events.get(call.get("tool_call_id"), {})
        result = event.get("result") if isinstance(event, dict) else {}
        result_lines = sorted(set(_tool_result_line_numbers(result)))
        all_result_lines.update(result_lines)
        already_prompt_lines = sorted(line for line in result_lines if line in prompted_lines)
        new_prompt_lines = sorted(line for line in result_lines if line not in prompted_lines)
        repeated_structured_facts = sorted(
            line for line in new_prompt_lines if line in structured_fact_lines
        )
        line_count = _int_or_zero(call.get("result_lines"))
        char_count = _int_or_zero(call.get("result_chars"))
        if line_count == 0 and result_lines:
            line_count = len(result_lines)
        if char_count == 0 and result:
            char_count = len(json.dumps(result, sort_keys=True))
        flags = _tool_call_flags(
            call,
            result_lines=result_lines,
            new_prompt_lines=new_prompt_lines,
            repeated_structured_facts=repeated_structured_facts,
            line_count=line_count,
        )
        classified.append(
            {
                "model_turn": call.get("model_turn"),
                "name": call.get("name"),
                "args_summary": call.get("args_summary"),
                "latency_ms": call.get("latency_ms"),
                "result_lines": line_count,
                "result_chars": char_count,
                "total_matches": call.get("total_matches"),
                "new_prompt_excerpt_line_count": len(new_prompt_lines),
                "already_in_prompt_excerpt_line_count": len(already_prompt_lines),
                "result_line_numbers_sample": result_lines[:20],
                "new_prompt_excerpt_line_sample": new_prompt_lines[:20],
                "new_line_existing_structured_fact_count": len(repeated_structured_facts),
                "new_line_existing_structured_fact_sample": repeated_structured_facts[:20],
                "already_in_prompt_excerpt_line_sample": already_prompt_lines[:20],
                "flags": flags,
            }
        )
    return classified, all_result_lines


def _tool_call_flags(
    call: dict[str, Any],
    *,
    result_lines: list[int],
    new_prompt_lines: list[int],
    repeated_structured_facts: list[int],
    line_count: int,
) -> list[str]:
    flags = []
    if call.get("error"):
        flags.append("error")
    if call.get("truncated"):
        flags.append("truncated")
    if line_count == 0:
        flags.append("no_result_lines")
    if result_lines and not new_prompt_lines:
        flags.append("duplicate_prompt_context")
    if repeated_structured_facts:
        flags.append("existing_structured_fact")
    return flags


def _tool_only_context_impact(
    *,
    analysis: dict[str, Any],
    tool_only_evidence_lines: list[int],
    primary_from_tool_only: bool,
    structured_fact_redundant_lines: list[int] | None = None,
) -> dict[str, Any]:
    structured_fact_redundant = set(structured_fact_redundant_lines or [])
    if primary_from_tool_only and not set(tool_only_evidence_lines).issubset(
        structured_fact_redundant
    ):
        return {
            "impact": "decision_critical_primary",
            "decision_relevant_lines": sorted(
                set(tool_only_evidence_lines).difference(structured_fact_redundant)
            ),
            "structured_fact_redundant_lines": sorted(structured_fact_redundant),
            "incidental_lines": [],
        }
    if not tool_only_evidence_lines:
        return {
            "impact": "no_final_citation_dependency_observed",
            "decision_relevant_lines": [],
            "structured_fact_redundant_lines": [],
            "incidental_lines": [],
        }

    incidental_lines: set[int] = set()
    for item in [
        *(analysis.get("secondary_failures") or []),
        *(analysis.get("related_failures") or []),
    ]:
        if not isinstance(item, dict) or item.get("causal_role") not in {
            "cascade",
            "teardown",
        }:
            continue
        line = _int_or_none(item.get("line"))
        if line is not None:
            incidental_lines.add(line)

    for item in analysis.get("cascades") or []:
        if not isinstance(item, dict):
            continue
        first_line = _int_or_none(item.get("first_line"))
        last_line = _int_or_none(item.get("last_line"))
        if first_line is None or last_line is None:
            continue
        incidental_lines.update(
            line for line in tool_only_evidence_lines if first_line <= line <= last_line
        )

    for item in analysis.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        line = _int_or_none(item.get("line"))
        supports = str(item.get("supports") or "")
        if line is not None and re.search(
            r"\b(?:cascade|cleanup|downstream|scheduler cancellation|teardown)\b",
            supports,
            re.I,
        ):
            incidental_lines.add(line)

    cited = set(tool_only_evidence_lines).difference(structured_fact_redundant)
    incidental = sorted(cited.intersection(incidental_lines))
    decision_relevant = sorted(cited.difference(incidental_lines))
    if decision_relevant:
        impact = "decision_relevant_support"
    elif structured_fact_redundant:
        impact = "existing_structured_fact_redundancy"
    else:
        impact = "incidental_downstream_context"
    return {
        "impact": impact,
        "decision_relevant_lines": decision_relevant,
        "structured_fact_redundant_lines": sorted(structured_fact_redundant),
        "incidental_lines": incidental,
    }


def model_selection_signals(
    *,
    model_call_summary: dict[str, Any],
    tool_efficiency: dict[str, Any],
    semantic_safety: dict[str, Any],
    route_execution_status: str | None = None,
) -> dict[str, Any]:
    retried_calls = _int_or_zero(model_call_summary.get("retried_calls"))
    failed_calls = _int_or_zero(model_call_summary.get("failed_calls"))
    context_window_error_calls = _int_or_zero(model_call_summary.get("context_window_error_calls"))
    endpoint_failed_calls = (
        _int_or_zero(model_call_summary.get("endpoint_failed_calls"))
        if model_call_summary.get("endpoint_failed_calls") is not None
        else max(0, failed_calls - context_window_error_calls)
    )
    unnecessary_tool_calls = _int_or_zero(tool_efficiency.get("duplicate_prompt_context_calls"))
    low_yield_tool_calls = _int_or_zero(tool_efficiency.get("no_new_prompt_line_calls"))
    extra_model_turns = _int_or_zero(model_call_summary.get("extra_model_turns_after_initial"))

    endpoint_reliability = "ok"
    if route_execution_status == "deadline_exceeded":
        endpoint_reliability = "deadline_exceeded"
    elif endpoint_failed_calls or retried_calls:
        endpoint_reliability = "endpoint_issue"

    client_request_health = "context_budget_exceeded" if context_window_error_calls else "ok"

    context_efficiency = "good"
    if unnecessary_tool_calls:
        context_efficiency = "unnecessary_tool_context"
    elif tool_efficiency.get("final_context_impact") == ("existing_structured_fact_redundancy"):
        context_efficiency = "unnecessary_tool_context"
    elif low_yield_tool_calls:
        context_efficiency = "low_yield_tool_context"
    elif _int_or_zero(tool_efficiency.get("calls")):
        context_efficiency = "tool_added_context_or_non_line_context"

    semantic_safety_signal = semantic_safety.get("semantic_safety") or "not_reported"

    notes: list[str] = []
    if endpoint_reliability == "endpoint_issue":
        notes.append(
            "Provider failures, retries, or timeouts are endpoint/service reliability "
            "signals, not semantic model-quality failures."
        )
    elif endpoint_reliability == "deadline_exceeded":
        notes.append(
            "The route did not complete within the analysis deadline; zero recorded "
            "provider attempts does not imply endpoint success."
        )
    if context_window_error_calls:
        notes.append(
            "At least one request exceeded the declared model context window; this "
            "is a client/profile budget failure, not endpoint unreliability."
        )
    if unnecessary_tool_calls:
        notes.append(
            "At least one tool call returned only lines already present in the initial "
            "prompt excerpt; this is a context-efficiency/model-behavior signal."
        )
    elif tool_efficiency.get("final_context_impact") == ("existing_structured_fact_redundancy"):
        notes.append(
            "The final answer cited a tool-only raw line whose semantic fact was "
            "already present in the structured L0 bundle."
        )
    elif low_yield_tool_calls:
        notes.append(
            "At least one tool call returned no new prompt-excerpt lines; review whether "
            "the bundle or prompt made the needed context clear enough."
        )
    return {
        "context_efficiency": context_efficiency,
        "endpoint_reliability": endpoint_reliability,
        "client_request_health": client_request_health,
        "semantic_safety": semantic_safety_signal,
        "model_failure_domain": semantic_safety.get("model_failure_domain"),
        "model_failure_domain_status": semantic_safety.get("model_failure_domain_status"),
        "model_failure_domain_confidence": semantic_safety.get("model_failure_domain_confidence"),
        "model_retry_outlook_without_workload_change": semantic_safety.get(
            "model_retry_outlook_without_workload_change"
        ),
        "model_retry_outlook_status": semantic_safety.get("model_retry_outlook_status"),
        "model_retry_outlook_confidence": semantic_safety.get("model_retry_outlook_confidence"),
        "retry_policy_rule": semantic_safety.get("retry_policy_rule"),
        "retry_budget_exhausted": semantic_safety.get("retry_budget_exhausted"),
        "unnecessary_tool_calls": unnecessary_tool_calls,
        "low_yield_tool_calls": low_yield_tool_calls,
        "model_turns": model_call_summary.get("model_turns"),
        "extra_model_turns_after_initial": extra_model_turns,
        "failed_endpoint_attempts": endpoint_failed_calls,
        "failed_model_calls": failed_calls,
        "context_window_error_calls": context_window_error_calls,
        "retried_model_calls": retried_calls,
        "timeout_model_calls": _int_or_zero(model_call_summary.get("timeout_calls")),
        "provider_error_count": _int_or_zero(model_call_summary.get("provider_error_count")),
        "http_error_calls": _int_or_zero(model_call_summary.get("http_error_calls")),
        "failed_model_call_latency_s": model_call_summary.get("failed_latency_s_total"),
        "notes": notes,
    }


def l1_kpis(
    *,
    l1: dict[str, Any],
    l1_layer: dict[str, Any],
    timing: dict[str, Any],
    token_usage: dict[str, Any],
    token_limit: dict[str, Any],
    model_call_summary: dict[str, Any],
    tool_efficiency: dict[str, Any],
    model_selection_signals: dict[str, Any],
    route_execution_status: str | None = None,
) -> dict[str, Any]:
    errors = l1.get("errors") or []
    anomalies = l1.get("anomalies") or {}
    if not isinstance(errors, list):
        errors = []
    if not isinstance(anomalies, dict):
        anomalies = {}
    execution_status, execution_issues = l1_execution_status(
        l1_layer=l1_layer,
        model_call_summary=model_call_summary,
        route_execution_status=route_execution_status,
    )
    return {
        "response_parsed": l1.get("success") if l1.get("enabled") else None,
        "output_status": l1_layer.get("output_status"),
        "output_usable": l1_layer.get("output_usable"),
        "output_errors": l1_layer.get("output_errors") or [],
        "execution_status": execution_status,
        "execution_issues": execution_issues,
        "contract_repair_requested": bool(anomalies.get("contract_repair_requested")),
        "contract_repair_turns": (1 if anomalies.get("contract_repair_requested") else 0),
        "wall_clock_s": timing.get("l1_wall_clock_s"),
        "model_call_wall_clock_s": timing.get("l1_model_call_wall_clock_s"),
        "tool_wall_clock_s": timing.get("l1_tool_wall_clock_s"),
        "model_calls": model_call_summary.get("calls"),
        "model_turns": model_call_summary.get("model_turns"),
        "successful_model_calls": model_call_summary.get("successful_calls"),
        "failed_model_calls": model_call_summary.get("failed_calls"),
        "retried_model_calls": model_call_summary.get("retried_calls"),
        "timeout_model_calls": model_call_summary.get("timeout_calls"),
        "provider_error_count": model_call_summary.get("provider_error_count"),
        "provider_reported_timing": model_call_summary.get("provider_reported_timing"),
        "context_budget_adjusted_calls": model_call_summary.get("context_budget_adjusted_calls"),
        "context_window_tokens": model_call_summary.get("context_window_tokens"),
        "max_estimated_input_tokens": model_call_summary.get("max_estimated_input_tokens"),
        "configured_max_output_tokens": model_call_summary.get("configured_max_output_tokens"),
        "minimum_effective_max_output_tokens": model_call_summary.get(
            "minimum_effective_max_output_tokens"
        ),
        "finish_reasons": model_call_summary.get("finish_reasons") or {},
        "tool_calls": tool_efficiency.get("calls"),
        "tool_driven_model_turns": tool_efficiency.get("tool_driven_model_turns"),
        "extra_model_turns_after_initial": model_call_summary.get(
            "extra_model_turns_after_initial"
        ),
        "max_model_turn": tool_efficiency.get("max_model_turn"),
        "tool_result_lines": tool_efficiency.get("total_result_lines"),
        "tool_result_chars": tool_efficiency.get("total_result_chars"),
        "tool_calls_added_new_prompt_lines": tool_efficiency.get("new_prompt_excerpt_line_count"),
        "tool_calls_already_prompt_lines": tool_efficiency.get(
            "already_in_prompt_excerpt_line_count"
        ),
        "duplicate_tool_calls": tool_efficiency.get("duplicate_prompt_context_calls"),
        "no_new_prompt_line_tool_calls": tool_efficiency.get("no_new_prompt_line_calls"),
        "tool_error_calls": tool_efficiency.get("error_calls"),
        "tool_truncated_calls": tool_efficiency.get("truncated_calls"),
        "prompt_tokens": token_usage.get("prompt_tokens"),
        "completion_tokens": token_usage.get("completion_tokens"),
        "total_tokens": token_usage.get("total_tokens"),
        "token_limit_hit": token_limit.get("hit"),
        "context_efficiency": model_selection_signals.get("context_efficiency"),
        "endpoint_reliability": model_selection_signals.get("endpoint_reliability"),
        "client_request_health": model_selection_signals.get("client_request_health"),
        "failed_endpoint_attempts": model_selection_signals.get("failed_endpoint_attempts"),
        "context_window_error_calls": model_selection_signals.get("context_window_error_calls"),
        "http_error_calls": model_selection_signals.get("http_error_calls"),
        "errors_count": len(errors),
        "anomaly_keys": sorted(str(key) for key, value in anomalies.items() if value),
        "review_mode_note": (
            "L1 KPIs describe raw model, endpoint, token, and tool interaction health. "
            "They do not include L2 audit results or prove semantic correctness."
        ),
    }
