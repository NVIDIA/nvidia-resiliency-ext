# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Human-readable Markdown rendering for one-log review summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifact_io import write_text_atomic as _write_text_atomic
from .scoring import _int_or_zero


def write_review_markdown(path: Path, summary: dict[str, Any]) -> None:
    """Publish the human-readable review artifact for one route."""
    primary = summary["primary_failure"]
    retry_policy = summary.get("retry_policy") or {}
    timing = summary["timing"]
    token_usage = summary["token_usage"]
    leak_audit = summary["path_redaction_audit"]
    job_metadata = summary.get("job_metadata") or {}
    progress_summary = summary.get("run_progress_summary") or {}
    line_numbering = summary.get("line_numbering") or {}
    model_selection_signals = summary.get("model_selection_signals") or {}
    model_call_summary = summary.get("model_call_summary") or {}
    tool_efficiency = summary.get("tool_efficiency") or {}
    semantic_safety = summary.get("semantic_safety") or {}
    l0_bundle_kpis = summary.get("l0_kpis") or summary.get("l0_bundle_kpis") or {}
    l1_kpis = summary.get("l1_kpis") or {}
    l2_kpis = summary.get("l2_kpis") or {}
    l3_kpis = summary.get("l3_kpis") or {}
    l4_kpis = summary.get("l4_kpis") or {}
    recovery_assessment = summary.get("model_recovery_assessment") or {}
    root_cause = summary.get("root_cause_assessment") or {}
    tool_profile = summary.get("effective_tool_profile") or {}
    lines = [
        f"# Restart Agent Review: {summary['run_label']}",
        "",
        f"- target: `{summary['target']}`",
        f"- model: `{summary['model']}`",
        (
            "- tool_loop_profile: "
            f"id=`{tool_profile.get('profile_id')}`, "
            f"experimental=`{tool_profile.get('experimental')}`, "
            f"tools_enabled=`{tool_profile.get('tools_enabled')}`, "
            f"max_tool_rounds=`{tool_profile.get('max_tool_rounds')}`, "
            f"max_model_turns=`{tool_profile.get('max_model_turns')}`, "
            f"source=`{tool_profile.get('source')}`"
        ),
        f"- exit_code: `{summary['exit_code']}`",
        f"- l1_response_parsed: `{summary['l1_response_parsed']}`",
        f"- l2_audit_status: `{summary['l2_audit_status']}`",
        f"- decision: `{summary['decision']}`",
        f"- decision_basis: `{summary['decision_basis']}`",
        (
            "- retry_policy: "
            f"version=`{retry_policy.get('policy_version')}`, "
            f"rule=`{retry_policy.get('rule')}`, "
            f"allowed_retries=`{retry_policy.get('allowed_retries')}`, "
            f"matching_prior_failures=`{retry_policy.get('matching_prior_failures')}`, "
            f"exhausted=`{retry_policy.get('retry_budget_exhausted')}`"
        ),
        (
            "- primary: "
            f"`{primary.get('fine_class')}` line `{primary.get('line')}` "
            f"outcome `{primary.get('fault_outcome')}` "
            f"causal_role `{primary.get('causal_role')}`"
        ),
        (
            "- L1 recovery assessment: "
            f"root_cause_status=`{root_cause.get('status')}`, "
            f"domain=`{_claim_field(recovery_assessment, 'failure_domain', 'value')}`, "
            f"domain_status=`{_claim_field(recovery_assessment, 'failure_domain', 'status')}`, "
            f"domain_confidence=`{_claim_field(recovery_assessment, 'failure_domain', 'confidence')}`, "
            "retry_outlook_without_workload_change="
            f"`{_claim_field(recovery_assessment, 'retry_outlook_without_workload_change', 'value')}`, "
            f"retry_status=`{_claim_field(recovery_assessment, 'retry_outlook_without_workload_change', 'status')}`, "
            f"retry_confidence=`{_claim_field(recovery_assessment, 'retry_outlook_without_workload_change', 'confidence')}`"
        ),
        f"- model_calls: `{summary['model_calls']}`",
        f"- tool_calls: `{summary['tool_calls']}` names=`{summary['tool_names']}`",
        f"- total_wall_clock_s: `{timing.get('total_wall_clock_s')}`",
        f"- l1_wall_clock_s: `{timing.get('l1_wall_clock_s')}`",
        f"- l1_model_call_wall_clock_s: `{timing.get('l1_model_call_wall_clock_s')}`",
        f"- token_usage_total: `{token_usage.get('total_tokens')}`",
        f"- token_limit_hit: `{(summary.get('token_limit') or {}).get('hit')}`",
        f"- path_redaction_audit_passed: `{leak_audit.get('passed')}`",
        (
            "- job_metadata: "
            f"world_size=`{job_metadata.get('explicit_world_size')}`, "
            f"world_size_source=`{job_metadata.get('world_size_source')}`, "
            f"observed_rank_range=`{job_metadata.get('observed_rank_min')}.."
            f"{job_metadata.get('observed_rank_max')}`, "
            f"observed_rank_count=`{job_metadata.get('observed_rank_count')}`"
        ),
        (
            "- run_progress_summary: "
            f"successful_runtime_s=`{progress_summary.get('successful_runtime_seconds')}`, "
            f"first_iteration=`{progress_summary.get('first_iteration')}`, "
            f"last_iteration=`{progress_summary.get('last_iteration')}`, "
            f"last_checkpoint=`{progress_summary.get('last_checkpoint_iteration')}`, "
            f"iterations_since_checkpoint=`{progress_summary.get('iterations_since_checkpoint')}`, "
            f"checkpoint_load=`{progress_summary.get('checkpoint_load_iteration')}`, "
            f"observed_failure_iteration="
            f"`{progress_summary.get('latest_observed_failure_iteration')}`, "
            f"observed_failure_line="
            f"`{progress_summary.get('latest_observed_failure_iteration_line')}`, "
            f"observed_replay_distance="
            f"`{progress_summary.get('observed_iterations_after_checkpoint_load')}`, "
            f"progress_markers=`{progress_summary.get('progress_marker_count')}`, "
            f"checkpoint_markers=`{progress_summary.get('checkpoint_marker_count')}`, "
            f"progress_after_failure=`{progress_summary.get('progress_after_failure_episode')}`"
        ),
        (
            "- line_numbering: "
            f"scheme=`{line_numbering.get('scheme')}`, "
            f"splitlines=`{line_numbering.get('splitlines_count')}`, "
            f"LF=`{line_numbering.get('lf_count')}`, "
            f"CR=`{line_numbering.get('cr_count')}`, "
            f"CRLF=`{line_numbering.get('crlf_count')}`"
        ),
    ]
    lines.extend(_artifact_guide_markdown(summary))
    lines.extend(_l1_model_output_markdown(summary))
    lines.extend(_gold_score_markdown(summary.get("gold_score")))
    lines.extend(_failure_identity_markdown(primary.get("failure_identity")))
    lines.extend(_l0_bundle_kpis_markdown(l0_bundle_kpis))
    lines.extend(_l1_kpis_markdown(l1_kpis))
    lines.extend(_l2_kpis_markdown(l2_kpis))
    lines.extend(_l3_kpis_markdown(l3_kpis))
    lines.extend(_l4_kpis_markdown(l4_kpis))
    lines.extend(_model_selection_signals_markdown(model_selection_signals))
    lines.extend(_semantic_safety_markdown(semantic_safety))
    lines.extend(_model_call_reliability_markdown(model_call_summary))
    lines.extend(_tool_efficiency_markdown(tool_efficiency))
    if leak_audit.get("source_path_tokens_found"):
        lines.append(f"- path_redaction_tokens_found: `{leak_audit['source_path_tokens_found']}`")
    if leak_audit.get("source_content_overlap_tokens"):
        lines.append(
            "- path_redaction_source_content_overlap: "
            f"`{leak_audit['source_content_overlap_tokens']}`"
        )
    if summary["errors"]:
        lines.append(f"- errors: `{summary['errors']}`")
    lines.append("")
    _write_text_atomic(path, "\n".join(lines))


def _artifact_guide_markdown(summary: dict[str, Any]) -> list[str]:
    artifacts = summary.get("artifacts") or {}
    return [
        "",
        "## Artifact Guide",
        "",
        "- Review the model's complete parsed L1 answer in the next section.",
        f"- Final composed product result: `{artifacts.get('result_json')}`",
        f"- Deep pipeline trace, raw model response, and tool transcript: "
        f"`{artifacts.get('trace_json')}`",
        "- This document combines the L1 answer with stage KPIs and gold scoring; "
        "the final product result is not the raw model response.",
    ]


def _l1_model_output_markdown(summary: dict[str, Any]) -> list[str]:
    model_output = summary.get("l1_model_output")
    lines = [
        "",
        "## L1 Model Output",
        "",
        "This is the model's complete parsed `analyzer_trace.l1.parsed_evidence` "
        "object before L2 grounding, L3 history, or L4 policy.",
        "",
    ]
    if not isinstance(model_output, dict):
        status = "not run" if summary.get("l1_response_parsed") is None else "not available"
        lines.append(f"L1 parsed output: `{status}`.")
        return lines
    lines.extend(
        [
            "```json",
            json.dumps(model_output, indent=2, sort_keys=True, ensure_ascii=True),
            "```",
        ]
    )
    return lines


def _failure_identity_markdown(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    family = value.get("family") if isinstance(value.get("family"), dict) else {}
    concrete = value.get("concrete") if isinstance(value.get("concrete"), dict) else {}
    client_concrete = (
        value.get("client_concrete") if isinstance(value.get("client_concrete"), dict) else {}
    )
    return [
        "",
        "## Experimental Failure Identity",
        "",
        f"- policy_active: `{value.get('policy_active')}`",
        f"- family: `{family.get('label')}`",
        f"- family_fingerprint: `{family.get('fingerprint')}`",
        f"- family_complete: `{family.get('complete')}`",
        f"- concrete: `{concrete.get('label')}`",
        f"- concrete_fingerprint: `{concrete.get('fingerprint')}`",
        f"- concrete_complete: `{concrete.get('complete')}`",
        f"- client_concrete: `{client_concrete.get('label')}`",
        f"- client_concrete_fingerprint: `{client_concrete.get('fingerprint')}`",
        f"- client_concrete_complete: `{client_concrete.get('complete')}`",
    ]


def _gold_score_markdown(gold_score: Any) -> list[str]:
    if not isinstance(gold_score, dict):
        return []
    l1 = gold_score.get("l1") or {}
    l2 = gold_score.get("l2") or {}
    l4 = gold_score.get("l4") or {}
    fallback_l4 = gold_score.get("fallback_l4") or {}
    enriched_l4 = gold_score.get("enriched_l4") or {}
    path_comparison = gold_score.get("l4_path_comparison") or {}
    l0a = gold_score.get("l0a") or {}
    l0b = gold_score.get("l0b") or {}
    lines = [
        "",
        "## Gold Comparison",
        "",
    ]
    if l0a:
        lines.extend(
            [
                (
                    "- L0A quality: "
                    f"primary_coverage=`{l0a.get('primary_evidence_coverage')}`, "
                    f"selected_primary=`{l0a.get('selected_primary_accuracy')}`, "
                    f"root_fingerprint=`{l0a.get('root_fingerprint_accuracy')}`, "
                    f"progress_recall=`{l0a.get('progress_line_recall')}`, "
                    f"checkpoint_recall=`{l0a.get('checkpoint_line_recall')}`, "
                    f"overall=`{l0a.get('overall_pass')}`"
                ),
                (
                    "- L0A deterministic facts: "
                    f"setup_markers=`{l0a.get('setup_marker_types_correct')}`, "
                    f"count=`{l0a.get('setup_marker_count_correct')}`, "
                    f"coverage=`{l0a.get('coverage_checks')}`"
                ),
            ]
        )
    if l0b:
        lines.append(
            "- L0B quality: "
            f"required_line_recall=`{l0b.get('required_evidence_line_recall')}`, "
            f"primary_retained=`{l0b.get('primary_retained_from_l0a')}`, "
            f"reference_checks=`{l0b.get('required_reference_checks')}`, "
            f"projection_integrity=`{l0b.get('projection_integrity_ok')}`, "
            f"overall=`{l0b.get('overall_pass')}`"
        )
    lines.extend(
        [
            (
                "- L1 raw semantics: "
                f"root_cause=`{l1.get('root_cause_correct')}`, "
                f"recovery=`{l1.get('recovery_assessment_correct')}`, "
                f"cascades=`{l1.get('cascade_correct')}`, "
                f"overall=`{l1.get('overall_semantic_pass')}`"
            ),
            f"- L1 recovery fields: `{l1.get('recovery_field_results')}`",
            f"- L1 unsupported_claims: `{l1.get('unsupported_claims')}`",
            f"- L1 semantic_confidence: `{l1.get('model_recovery_confidence')}`",
            (
                "- L2 grounding, identity, and audit: "
                f"status=`{l2.get('audit_status')}`, "
                f"audit_correct=`{l2.get('audit_correct')}`, "
                f"history_identity=`{l2.get('history_identity_correct')}`, "
                f"anchor=`{l2.get('canonical_anchor_correct')}`, "
                f"operation=`{l2.get('operation_correct')}`, "
                f"mechanism=`{l2.get('mechanism_correct')}`, "
                f"reference_audit_effect=`{l2.get('reference_audit_effect')}`"
            ),
            (
                "- L4/final product: "
                f"root_cause=`{l4.get('root_cause_correct')}`, "
                f"rule=`{l4.get('retry_rule_correct')}`, "
                f"allowed_retries=`{l4.get('allowed_retries_correct')}`, "
                f"exhaustion=`{l4.get('retry_exhaustion_correct')}`, "
                f"action=`{l4.get('action_correct')}`, "
                f"cascades=`{l4.get('cascade_correct')}`, "
                f"overall=`{l4.get('overall_semantic_pass')}`"
            ),
            (
                "- fallback versus enriched action: "
                f"fallback=`{fallback_l4.get('action_correct')}`, "
                f"enriched=`{enriched_l4.get('action_correct')}`, "
                f"effect=`{path_comparison.get('action_effect')}`"
            ),
            (
                "- fallback versus enriched policy/action: "
                f"fallback=`{fallback_l4.get('policy_action_pass')}`, "
                f"enriched=`{enriched_l4.get('policy_action_pass')}`, "
                f"effect=`{path_comparison.get('policy_action_effect')}`"
            ),
            f"- calibration: `{gold_score.get('calibration_note')}`",
        ]
    )
    return lines


def _l1_kpis_markdown(l1_kpis: dict[str, Any]) -> list[str]:
    first_turn_usable = None
    response_delivered = (
        _int_or_zero(l1_kpis.get("successful_model_calls")) > 0
        if l1_kpis.get("successful_model_calls") is not None
        else bool(l1_kpis.get("response_parsed") or l1_kpis.get("output_usable"))
    )
    if response_delivered:
        first_turn_usable = bool(
            l1_kpis.get("output_usable")
            and _int_or_zero(l1_kpis.get("model_turns")) == 1
            and _int_or_zero(l1_kpis.get("tool_driven_model_turns")) == 0
            and _int_or_zero(l1_kpis.get("contract_repair_turns")) == 0
        )
    lines = [
        "",
        "## Model Route Qualification",
        "",
        "Semantic quality is reported in the gold section when available. Endpoint-only absence is not a semantic failure.",
        "",
        "### Route Outcome",
        "",
        (
            "- latency: "
            f"l1_wall_clock_s=`{l1_kpis.get('wall_clock_s')}`, "
            f"model_call_wall_clock_s=`{l1_kpis.get('model_call_wall_clock_s')}`, "
            f"tool_wall_clock_s=`{l1_kpis.get('tool_wall_clock_s')}`"
        ),
        (
            "- result: "
            f"execution_status=`{l1_kpis.get('execution_status')}`, "
            f"output_status=`{l1_kpis.get('output_status')}`, "
            f"usable=`{l1_kpis.get('output_usable')}`"
        ),
        "",
        "### Behavioral Efficiency",
        "",
        f"- first_turn_usable: `{first_turn_usable if first_turn_usable is not None else 'not_observed'}`",
        (
            "- interaction: "
            f"turns=`{l1_kpis.get('model_turns')}`, "
            f"tool_driven_turns=`{l1_kpis.get('tool_driven_model_turns')}`, "
            f"contract_repair_turns=`{l1_kpis.get('contract_repair_turns')}`, "
            f"all_extra_turns=`{l1_kpis.get('extra_model_turns_after_initial')}`"
        ),
        (
            "- tool_calls: "
            f"calls=`{l1_kpis.get('tool_calls')}`, "
            f"new_prompt_lines=`{l1_kpis.get('tool_calls_added_new_prompt_lines')}`, "
            f"duplicate_tool_calls=`{l1_kpis.get('duplicate_tool_calls')}`, "
            f"no_new_prompt_line_tool_calls=`{l1_kpis.get('no_new_prompt_line_tool_calls')}`, "
            f"tool_errors=`{l1_kpis.get('tool_error_calls')}`, "
            f"tool_truncated=`{l1_kpis.get('tool_truncated_calls')}`"
        ),
        (
            "- tokens: "
            f"prompt=`{l1_kpis.get('prompt_tokens')}`, "
            f"completion=`{l1_kpis.get('completion_tokens')}`, "
            f"total=`{l1_kpis.get('total_tokens')}`, "
            f"token_limit_hit=`{l1_kpis.get('token_limit_hit')}`"
        ),
        "",
        "### Endpoint Reliability",
        "",
        (
            "- provider_attempts: "
            f"calls=`{l1_kpis.get('model_calls')}`, "
            f"successful=`{l1_kpis.get('successful_model_calls')}`, "
            f"failed=`{l1_kpis.get('failed_model_calls')}`, "
            f"retried=`{l1_kpis.get('retried_model_calls')}`, "
            f"timeouts=`{l1_kpis.get('timeout_model_calls')}`, "
            f"provider_errors=`{l1_kpis.get('provider_error_count')}`, "
            f"status=`{l1_kpis.get('endpoint_reliability')}`"
        ),
        "",
        "### Contract And Context Diagnostics",
        "",
        (
            "- response: "
            f"execution_issues=`{l1_kpis.get('execution_issues')}`, "
            f"parsed=`{l1_kpis.get('response_parsed')}`, "
            f"contract_repair=`{l1_kpis.get('contract_repair_requested')}`, "
            f"context_efficiency=`{l1_kpis.get('context_efficiency')}`, "
            f"client_request_health=`{l1_kpis.get('client_request_health')}`"
        ),
        (
            "- errors: "
            f"errors_count=`{l1_kpis.get('errors_count')}`, "
            f"anomaly_keys=`{l1_kpis.get('anomaly_keys')}`, "
            f"finish_reasons=`{l1_kpis.get('finish_reasons')}`"
        ),
        f"- note: {l1_kpis.get('review_mode_note')}",
    ]
    return lines


def _l2_kpis_markdown(l2_kpis: dict[str, Any]) -> list[str]:
    return [
        "",
        "## L2 KPI Checklist",
        "",
        (
            "- functional grounding: "
            f"status=`{l2_kpis.get('grounding_status')}`, "
            f"method=`{l2_kpis.get('grounding_method')}`, "
            f"history_ready=`{l2_kpis.get('history_identity_ready')}`"
        ),
        (
            "- audit diagnostics: "
            f"status=`{l2_kpis.get('audit_status')}`, "
            f"ran=`{l2_kpis.get('audit_ran')}`, "
            f"primary=`{l2_kpis.get('primary_available')}`, "
            f"recovery_assessment=`{l2_kpis.get('recovery_assessment_available')}`, "
            f"related_failures=`{l2_kpis.get('related_failures_audited')}`"
        ),
        (
            "- grounding: "
            f"domain_support=`{l2_kpis.get('failure_domain_supporting_lines')}`, "
            f"retry_support=`{l2_kpis.get('retry_outlook_supporting_lines')}`, "
            f"unresolved=`{l2_kpis.get('unresolved_recovery_supporting_lines')}`, "
            f"citations_raw/rendered/abbrev/nearby/ungrounded="
            f"`{l2_kpis.get('exact_citation_count')}/"
            f"{l2_kpis.get('rendered_exact_citation_count')}/"
            f"{l2_kpis.get('abbreviated_exact_citation_count')}/"
            f"{l2_kpis.get('nearby_resolved_count')}/"
            f"{l2_kpis.get('ungrounded_citation_count')}`"
        ),
        (
            "- grounding outcomes: "
            f"adjustments=`{l2_kpis.get('grounding_adjustment_count')}`, "
            f"identity_anchor=`{l2_kpis.get('stable_identity_anchor_line')}`, "
            f"anchor_reason=`{l2_kpis.get('stable_identity_anchor_reason')}`"
        ),
        (
            "- root fingerprint: "
            f"owner=`{l2_kpis.get('root_fingerprint_owner')}`, "
            f"available=`{l2_kpis.get('root_fingerprint_available')}`, "
            f"history_ready=`{l2_kpis.get('history_identity_ready')}`, "
            f"source=`{l2_kpis.get('root_fingerprint_source')}`, "
            f"matches_l0=`{l2_kpis.get('matches_l0_root_fingerprint')}`, "
            f"id=`{l2_kpis.get('root_fingerprint')}`"
        ),
        (
            "- findings: "
            f"wall_clock_s=`{l2_kpis.get('wall_clock_s')}`, "
            f"count=`{l2_kpis.get('finding_count')}`, "
            f"material=`{l2_kpis.get('material_finding_count')}`, "
            f"severity_counts=`{l2_kpis.get('finding_severity_counts')}`"
        ),
    ]


def _l3_kpis_markdown(l3_kpis: dict[str, Any]) -> list[str]:
    return [
        "",
        "## L3 KPI Checklist",
        "",
        (
            "- current facts: "
            f"source=`{l3_kpis.get('current_failure_facts_source')}`, "
            f"history_identity_ready=`{l3_kpis.get('history_identity_ready')}`, "
            f"root=`{l3_kpis.get('current_root_fingerprint')}`"
        ),
        (
            "- history comparison: "
            f"available=`{l3_kpis.get('history_available')}`, "
            f"same_job=`{l3_kpis.get('same_job_attempts')}`, "
            f"same_root=`{l3_kpis.get('matching_root_attempts')}`, "
            f"advanced=`{l3_kpis.get('observed_advance_attempts')}`, "
            f"no_observed_advance=`{l3_kpis.get('no_observed_advance_attempts')}`, "
            f"unknown=`{l3_kpis.get('unknown_progress_attempts')}`"
        ),
        (
            "- stronger matches: "
            f"same_failure_position=`{l3_kpis.get('exact_failure_position_attempts')}`, "
            f"same_data_position=`{l3_kpis.get('same_data_position_attempts')}`, "
            f"same_artifact=`{l3_kpis.get('same_artifact_attempts')}`, "
            f"consecutive_no_advance="
            f"`{l3_kpis.get('consecutive_same_root_no_advance_attempts')}`"
        ),
        ("- timing: " f"wall_clock_s=`{l3_kpis.get('wall_clock_s')}`"),
    ]


def _l4_kpis_markdown(l4_kpis: dict[str, Any]) -> list[str]:
    return [
        "",
        "## L4 KPI Checklist",
        "",
        (
            "- retry policy: "
            f"decision=`{l4_kpis.get('decision')}`, "
            f"basis=`{l4_kpis.get('decision_basis')}`, "
            f"version=`{l4_kpis.get('policy_version')}`, "
            f"rule=`{l4_kpis.get('rule')}`, "
            f"matching/allowed="
            f"`{l4_kpis.get('matching_prior_failures')}/"
            f"{l4_kpis.get('allowed_retries')}`, "
            f"exhausted=`{l4_kpis.get('retry_budget_exhausted')}`"
        ),
        (
            "- semantic inputs: "
            f"domain=`{l4_kpis.get('failure_domain')}`, "
            f"domain_status=`{l4_kpis.get('failure_domain_status')}`, "
            f"retry_outlook=`{l4_kpis.get('retry_outlook_without_workload_change')}`, "
            f"retry_status=`{l4_kpis.get('retry_outlook_status')}`, "
            f"l2_policy_grounded=`"
            f"{l4_kpis.get('recovery_assessment_policy_grounded')}`, "
            f"immediate_stop_qualified=`{l4_kpis.get('current_evidence_qualified')}`"
        ),
        (
            "- output: "
            f"wall_clock_s=`{l4_kpis.get('wall_clock_s')}`, "
            f"evidence_source=`{l4_kpis.get('evidence_source')}`, "
            f"model_contribution=`{l4_kpis.get('model_contribution')}`, "
            f"quality=`{l4_kpis.get('result_quality')}`, "
            f"nvrx_use=`{l4_kpis.get('nvrx_use')}`"
        ),
        (
            "- downstream roles: "
            + (
                ", ".join(
                    f"{item.get('causal_role')}@{item.get('first_line')}"
                    + (
                        f"-{item.get('last_line')}"
                        if item.get("last_line") != item.get("first_line")
                        else ""
                    )
                    + f" x{item.get('count')}"
                    for item in l4_kpis.get("downstream_roles") or []
                )
                or "none"
            )
        ),
        (
            "- latency_scope: "
            f"mode=`{l4_kpis.get('latency_mode')}`, "
            f"terminal_total_s=`{l4_kpis.get('terminal_total_wall_clock_s')}`, "
            f"post_progressive_end_s=`{l4_kpis.get('post_progressive_end_wall_clock_s')}`, "
            f"decision_window_hit=`{l4_kpis.get('progressive_decision_window_hit')}`, "
            f"production_gate_measured=`{l4_kpis.get('production_gate_measured')}`"
        ),
    ]


def _l0_bundle_kpis_markdown(l0_bundle_kpis: dict[str, Any]) -> list[str]:
    selection = l0_bundle_kpis.get("selection_summary") or {}
    coverage = l0_bundle_kpis.get("evidence_coverage") or {}
    projection = l0_bundle_kpis.get("l0b_projection_metrics") or {}
    view_size = projection.get("view_size") or {}
    budget = projection.get("budget_utilization") or {}
    projected_selection = projection.get("selection_counts") or {}
    compaction = projection.get("compaction_counts") or {}
    integrity = projection.get("projection_integrity") or {}
    lines = [
        "",
        "## L0A Operations",
        "",
        (
            "- assembly: "
            f"l0_wall_clock_s=`{l0_bundle_kpis.get('l0_wall_clock_s')}`, "
            f"l0a_wall_clock_s=`{l0_bundle_kpis.get('l0a_wall_clock_s')}`, "
            f"line_count=`{l0_bundle_kpis.get('line_count')}`, "
            f"byte_size=`{l0_bundle_kpis.get('byte_size')}`"
        ),
        (
            "- shape: "
            f"context_windows=`{l0_bundle_kpis.get('context_window_count')}`, "
            f"candidate_anchors=`{l0_bundle_kpis.get('candidate_anchor_count')}`, "
            f"occurrence_groups=`{l0_bundle_kpis.get('occurrence_group_count')}`, "
            f"failure_episodes=`{l0_bundle_kpis.get('failure_episode_count')}`, "
            f"terminal_failure_episodes=`{l0_bundle_kpis.get('terminal_failure_episode_count')}`, "
            "distributed_failure_incidents="
            f"`{l0_bundle_kpis.get('distributed_failure_incident_count')}`"
        ),
        (
            "- selection/lossiness: "
            f"high_signal_lines=`{selection.get('high_signal_lines')}`, "
            f"sampled_candidate_lines=`{selection.get('sampled_candidate_lines')}`, "
            f"dropped_noise_lines=`{selection.get('dropped_noise_lines')}`, "
            f"caps_hit=`{selection.get('caps_hit')}`, "
            f"coverage=`{coverage}`"
        ),
        "",
        "## Decision Evidence",
        "",
        (
            "- selection: "
            "wall_clock_s="
            f"`{l0_bundle_kpis.get('decision_evidence_wall_clock_s')}`, "
            f"primary_line=`{l0_bundle_kpis.get('selected_primary_line')}`, "
            f"in_l0a=`{l0_bundle_kpis.get('selected_primary_in_bundle')}`, "
            f"referenced_windows=`{l0_bundle_kpis.get('selected_primary_context_window_ids')}`"
        ),
        (
            "- root fingerprint: "
            f"owner=`{l0_bundle_kpis.get('root_fingerprint_owner')}`, "
            f"available=`{l0_bundle_kpis.get('root_fingerprint_available')}`, "
            f"history_ready=`{l0_bundle_kpis.get('history_identity_ready')}`, "
            f"source=`{l0_bundle_kpis.get('root_fingerprint_source')}`, "
            f"id=`{l0_bundle_kpis.get('root_fingerprint')}`"
        ),
        (
            "- progress_context: "
            f"progress_after_fault_known=`{l0_bundle_kpis.get('progress_after_fault_known')}`, "
            f"progress_after_fault=`{l0_bundle_kpis.get('progress_after_fault')}`"
        ),
        (
            "- terminal_incident_timing: "
            f"line=`{l0_bundle_kpis.get('first_terminal_incident_line')}`, "
            f"timestamp=`{l0_bundle_kpis.get('first_terminal_incident_timestamp')}`, "
            "seconds_since_last_progress="
            f"`{l0_bundle_kpis.get('seconds_from_last_progress_to_terminal_incident')}`, "
            f"configured_timeout_s=`{l0_bundle_kpis.get('configured_terminal_timeout_seconds')}`, "
            f"detection_lag_s=`{l0_bundle_kpis.get('terminal_detection_lag_seconds')}`"
        ),
        (
            "- setup_progress: "
            f"count=`{l0_bundle_kpis.get('setup_marker_count')}`, "
            f"types=`{l0_bundle_kpis.get('setup_marker_types')}`, "
            f"lines=`{l0_bundle_kpis.get('setup_marker_lines')}`"
        ),
        "",
        "## L0B Operations",
        "",
        (
            "- projection: "
            f"schema=`{l0_bundle_kpis.get('l0b_schema_version')}`, "
            f"wall_clock_s=`{l0_bundle_kpis.get('l0b_wall_clock_s')}`, "
            f"characters=`{view_size.get('compact_json_characters')}`, "
            f"estimated_tokens=`{view_size.get('estimated_tokens')}`, "
            f"integrity=`{integrity.get('status')}`, "
            f"payload_hash=`{integrity.get('deterministic_payload_sha256')}`"
        ),
        (
            "- restart_environment_context: "
            f"`{l0_bundle_kpis.get('restart_environment_context')}`"
        ),
        f"- budget_utilization: `{budget}`",
        f"- selection_counts: `{projected_selection}`",
        f"- compaction_counts: `{compaction}`",
        "",
        "## L0B Model-Conditioned Diagnostics",
        "",
        (
            "- initial visibility: "
            f"selected_primary_in_view=`{l0_bundle_kpis.get('selected_primary_in_excerpt')}`, "
            "anchors_without_excerpt="
            f"`{l0_bundle_kpis.get('candidate_anchors_without_excerpt')}`"
        ),
        (
            "- downstream tool behavior: "
            f"tool_calls_needed=`{l0_bundle_kpis.get('tool_calls_needed')}`, "
            f"tool_calls_useful_proxy=`{l0_bundle_kpis.get('tool_calls_useful_proxy')}`, "
            f"new_prompt_lines=`{l0_bundle_kpis.get('tool_calls_added_new_prompt_lines')}`, "
            f"duplicate_tool_calls=`{l0_bundle_kpis.get('duplicate_tool_calls')}`, "
            f"no_new_prompt_line_tool_calls=`{l0_bundle_kpis.get('no_new_prompt_line_tool_calls')}`"
        ),
        (
            "- attention diagnostics: "
            f"top_anchor_progress_after_count=`{l0_bundle_kpis.get('top_anchor_progress_after_count')}`, "
            f"recovered_or_progressed_top_anchor_count=`{l0_bundle_kpis.get('recovered_or_progressed_top_anchor_count')}`"
        ),
        (
            "- attribution note: tool behavior is a cross-model/profile diagnostic; "
            "it is not by itself an L0B quality failure."
        ),
    ]
    return lines


def _model_selection_signals_markdown(signals: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## Model Selection Signals",
        "",
        (
            "- context_efficiency: "
            f"`{signals.get('context_efficiency')}` "
            f"(unnecessary_tool_calls=`{signals.get('unnecessary_tool_calls')}`, "
            f"low_yield_tool_calls=`{signals.get('low_yield_tool_calls')}`, "
            f"extra_model_turns_after_initial=`{signals.get('extra_model_turns_after_initial')}`)"
        ),
        (
            "- endpoint_reliability: "
            f"`{signals.get('endpoint_reliability')}` "
            f"(failed_endpoint_attempts=`{signals.get('failed_endpoint_attempts')}`, "
            f"retried_model_calls=`{signals.get('retried_model_calls')}`, "
            f"timeout_model_calls=`{signals.get('timeout_model_calls')}`, "
            f"http_error_calls=`{signals.get('http_error_calls')}`, "
            f"provider_error_count=`{signals.get('provider_error_count')}`, "
            f"failed_model_call_latency_s=`{signals.get('failed_model_call_latency_s')}`)"
        ),
        ("- semantic_safety: " f"`{signals.get('semantic_safety')}`"),
        (
            "- recovery_and_policy: "
            f"domain=`{signals.get('model_failure_domain')}`, "
            f"domain_status=`{signals.get('model_failure_domain_status')}`, "
            f"retry_outlook=`"
            f"{signals.get('model_retry_outlook_without_workload_change')}`, "
            f"retry_status=`{signals.get('model_retry_outlook_status')}`, "
            f"retry_rule=`{signals.get('retry_policy_rule')}`, "
            f"budget_exhausted=`{signals.get('retry_budget_exhausted')}`"
        ),
    ]
    notes = signals.get("notes") or []
    for note in notes:
        lines.append(f"- note: {note}")
    return lines


def _model_call_reliability_markdown(model_call_summary: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## Model-Call Reliability",
        "",
        (
            "- summary: "
            f"calls=`{model_call_summary.get('calls')}`, "
            f"turns=`{model_call_summary.get('model_turns')}`, "
            f"successful=`{model_call_summary.get('successful_calls')}`, "
            f"failed=`{model_call_summary.get('failed_calls')}`, "
            f"retried=`{model_call_summary.get('retried_calls')}`, "
            f"timeouts=`{model_call_summary.get('timeout_calls')}`"
        ),
        (
            "- latency: "
            f"model_call_latency_s_sum=`{model_call_summary.get('latency_s_total')}`, "
            f"failed_latency_s_sum=`{model_call_summary.get('failed_latency_s_total')}`, "
            f"retry_after_s_total=`{model_call_summary.get('retry_after_s_total')}`"
        ),
        f"- finish_reasons: `{model_call_summary.get('finish_reasons')}`",
        f"- provider_error_count: `{model_call_summary.get('provider_error_count')}`",
        f"- http_error_calls: `{model_call_summary.get('http_error_calls')}`",
    ]
    provider_errors = model_call_summary.get("provider_errors") or []
    provider_timing = model_call_summary.get("provider_reported_timing") or {}
    if provider_timing:
        components = provider_timing.get("components_ms_total") or {}
        lines.extend(
            [
                "",
                "### Provider-Reported L1 Timing",
                "",
                (
                    "- response-header totals: "
                    f"reported_calls=`{provider_timing.get('reported_call_count')}`, "
                    f"downstream_llm_api_ms=`{components.get('downstream_llm_api_ms')}`, "
                    f"proxy_pre_processing_ms=`{components.get('proxy_pre_processing_ms')}`, "
                    f"proxy_post_processing_ms=`{components.get('proxy_post_processing_ms')}`, "
                    f"proxy_message_copy_ms=`{components.get('proxy_message_copy_ms')}`"
                ),
                (
                    "- interpretation: downstream LLM API is the proxy's downstream-call "
                    "span; it is not model compute time and may include backend transport, "
                    "queueing, prefill, and decode."
                ),
            ]
        )
    if provider_errors:
        lines.extend(
            [
                "",
                "| turn | attempt | status | retryable | retry scheduled | latency s | error |",
                "|---:|---:|---:|---|---|---:|---|",
            ]
        )
        for error in provider_errors:
            error_text = _escape_md_table(str(error.get("error") or ""))
            lines.append(
                "| "
                f"{error.get('model_turn')} | "
                f"{error.get('attempt')} | "
                f"{error.get('http_status')} | "
                f"{error.get('retryable')} | "
                f"{error.get('retry_scheduled')} | "
                f"{error.get('latency_s')} | "
                f"{error_text} |"
            )
    return lines


def _semantic_safety_markdown(semantic_safety: dict[str, Any]) -> list[str]:
    normalizations = semantic_safety.get("normalizations") or []
    lines = [
        "",
        "## Semantic Safety",
        "",
        ("- summary: " f"semantic_safety=`{semantic_safety.get('semantic_safety')}`"),
        (
            "- recovery_and_policy: "
            f"domain=`{semantic_safety.get('model_failure_domain')}`, "
            f"domain_status=`{semantic_safety.get('model_failure_domain_status')}`, "
            f"retry_outlook=`"
            f"{semantic_safety.get('model_retry_outlook_without_workload_change')}`, "
            f"retry_status=`{semantic_safety.get('model_retry_outlook_status')}`, "
            f"retry_rule=`{semantic_safety.get('retry_policy_rule')}`, "
            f"budget_exhausted=`{semantic_safety.get('retry_budget_exhausted')}`"
        ),
    ]
    if normalizations:
        lines.extend(
            [
                "",
                "| field | from | to | reason | line |",
                "|---|---|---|---|---:|",
            ]
        )
        for item in normalizations:
            lines.append(
                "| "
                f"{item.get('field')} | "
                f"{_escape_md_table(str(item.get('from')))} | "
                f"{_escape_md_table(str(item.get('to')))} | "
                f"{item.get('reason')} | "
                f"{item.get('line')} |"
            )
    return lines


def _tool_efficiency_markdown(tool_efficiency: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## Tool-Use Efficiency",
        "",
        (
            "- summary: "
            f"calls=`{tool_efficiency.get('calls')}`, "
            f"tool_driven_turns=`{tool_efficiency.get('tool_driven_model_turns')}`, "
            f"all_model_turns=`{tool_efficiency.get('model_turns')}`, "
            f"all_extra_turns=`{tool_efficiency.get('extra_model_turns_after_initial')}`, "
            f"max_model_turn=`{tool_efficiency.get('max_model_turn')}`, "
            f"optimization_note=`{tool_efficiency.get('optimization_note')}`"
        ),
        (
            "- cost: "
            f"tool_wall_clock_s=`{tool_efficiency.get('tool_wall_clock_s')}`, "
            f"tool_latency_s_sum=`{tool_efficiency.get('total_tool_latency_s')}`, "
            f"model_call_wall_clock_s=`{tool_efficiency.get('model_call_wall_clock_s')}`"
        ),
        (
            "- context_yield: "
            f"result_lines=`{tool_efficiency.get('total_result_lines')}`, "
            f"result_chars=`{tool_efficiency.get('total_result_chars')}`, "
            f"new_prompt_excerpt_lines=`{tool_efficiency.get('new_prompt_excerpt_line_count')}`, "
            f"already_in_prompt_excerpt_lines=`{tool_efficiency.get('already_in_prompt_excerpt_line_count')}`"
        ),
        (
            "- flags: "
            f"duplicate_prompt_context_calls=`{tool_efficiency.get('duplicate_prompt_context_calls')}`, "
            f"no_new_prompt_line_calls=`{tool_efficiency.get('no_new_prompt_line_calls')}`, "
            f"error_calls=`{tool_efficiency.get('error_calls')}`, "
            f"truncated_calls=`{tool_efficiency.get('truncated_calls')}`"
        ),
        (
            "- final_answer_dependency: "
            f"status=`{tool_efficiency.get('final_context_dependency')}`, "
            f"impact=`{tool_efficiency.get('final_context_impact')}`, "
            f"primary_from_tool_only=`{tool_efficiency.get('final_primary_from_tool_only_context')}`, "
            f"decision_relevant_lines=`{tool_efficiency.get('decision_relevant_tool_only_lines')}`, "
            f"incidental_lines=`{tool_efficiency.get('incidental_tool_only_lines')}`"
        ),
        f"- attribution_policy_change: `{tool_efficiency.get('attribution_or_policy_change_note')}`",
    ]
    per_call = tool_efficiency.get("per_call") or []
    if per_call:
        lines.extend(
            [
                "",
                "| turn | tool | args | result lines | new prompt lines | already prompt lines | latency ms | flags |",
                "|---:|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for item in per_call:
            args_summary = _escape_md_table(str(item.get("args_summary") or ""))
            flags = _escape_md_table(",".join(item.get("flags") or []))
            lines.append(
                "| "
                f"{item.get('model_turn')} | "
                f"{item.get('name')} | "
                f"{args_summary} | "
                f"{item.get('result_lines')} | "
                f"{item.get('new_prompt_excerpt_line_count')} | "
                f"{item.get('already_in_prompt_excerpt_line_count')} | "
                f"{item.get('latency_ms')} | "
                f"{flags} |"
            )
    return lines


def _escape_md_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _claim_field(assessment: dict[str, Any], claim_name: str, field: str) -> Any:
    claim = assessment.get(claim_name) or {}
    return claim.get(field) if isinstance(claim, dict) else None
