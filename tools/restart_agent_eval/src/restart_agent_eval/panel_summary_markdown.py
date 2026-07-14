# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concise panel-summary Markdown rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .panel_format import _dict, _md, _primary_label, _short_identity, _yes_no
from .scoring import _int_or_zero


def _panel_markdown(panel: dict[str, Any]) -> str:
    """Render the compact human review; exhaustive stage detail lives separately."""

    rows = panel["rows"]
    model_rows = [row for row in rows if row.get("model")]
    config = _dict(panel.get("restart_agent_config"))
    manifest = _dict(panel.get("run_manifest"))
    repositories = _dict(manifest.get("repositories"))
    product = _dict(repositories.get("product"))
    harness = _dict(repositories.get("harness"))
    comparison_axes = _dict(panel.get("comparison_axes"))
    concerns = panel.get("concerns") or []
    gold_rows = [row for row in rows if row.get("gold_case_id")]
    lines = [
        "# Restart Agent Model Panel Summary",
        "",
        "## Run Identity",
        "",
        "| case | source SHA-256 | bytes | run | gold | routes |",
        "|---|---|---:|---|---|---:|",
        "| "
        f"{_md(panel.get('source_log_relative_path') or panel.get('source_log_name'))} | "
        f"{_md(_short_hash(panel.get('source_log_sha256')))} | "
        f"{_md(panel.get('source_log_byte_size'))} | "
        f"{_md(panel.get('run_id') or panel.get('run_dir_name'))} | "
        f"{_md('attached' if gold_rows else 'not reviewed')} | "
        f"{_md(panel.get('model_count'))} |",
        "",
        "| product commit | harness commit | config | config fingerprint |",
        "|---|---|---|---|",
        "| "
        f"{_md(_short_hash(panel.get('product_commit') or product.get('commit')))} | "
        f"{_md(_short_hash(panel.get('harness_commit') or harness.get('commit')))} | "
        f"{_md(config.get('config_id') or '<not recorded>')} | "
        f"{_md(_short_hash(config.get('config_fingerprint')))} |",
        "",
        f"Run directory: `{panel.get('run_dir')}`",
    ]
    _append_attention_required(lines, concerns)
    _append_gold_scorecard(lines, panel, gold_rows)
    _append_cross_route_outcome(lines, rows, comparison_axes)
    _append_primary_by_stage(lines, rows)
    _append_recovery_assessment(lines, model_rows, comparison_axes)
    _append_shared_deterministic_evidence(lines, panel)
    _append_history_and_policy(lines, rows)
    _append_conditional_diagnostics(lines, panel)
    _append_artifact_paths(lines, panel)
    lines.append("")
    return "\n".join(lines)


def _append_attention_required(lines: list[str], concerns: list[dict[str, Any]]) -> None:
    lines.extend(["", "## Attention Required", ""])
    if concerns:
        lines.extend(
            [
                "| severity | owner | target | concern | impact | evidence |",
                "|---|---|---|---|---|---|",
            ]
        )
        for concern in concerns:
            lines.append(
                "| "
                f"{_md(concern.get('severity'))} | "
                f"{_md(concern.get('owner'))} | "
                f"{_md(concern.get('target'))} | "
                f"{_md(concern.get('category'))} | "
                f"{_md(concern.get('impact'))} | "
                f"{_md(concern.get('summary'))} |"
            )
    else:
        lines.append("- none")


def _append_gold_scorecard(
    lines: list[str], panel: dict[str, Any], gold_rows: list[dict[str, Any]]
) -> None:
    if gold_rows:
        first = gold_rows[0]
        lines.extend(
            [
                "",
                "## Gold Scorecard",
                "",
                "Shared deterministic stages are scored once; model-dependent stages are shown per route.",
                "",
                "| case | L0A | L0B | primary coverage | selected primary | phase | checkpoint | post-fault progress | cascades |",
                "|---|---|---|---|---|---|---|---|---|",
                "| "
                f"{_md(first.get('gold_case_id'))} | "
                f"{_score(first.get('gold_l0a_overall'))} | "
                f"{_score(first.get('gold_l0b_overall'))} | "
                f"{_score(first.get('gold_l0a_primary_evidence_coverage'))} | "
                f"{_score(first.get('gold_l0a_selected_primary_accuracy'))} | "
                f"{_score(first.get('gold_l0a_primary_phase_correct'))} | "
                f"{_score(first.get('gold_l0a_checkpoint_load_iteration_correct'))} | "
                f"{_score(first.get('gold_l0a_progress_after_failure_correct'))} | "
                f"{_score(first.get('gold_l0a_cascade_line_recall'))} |",
                "",
                "| target | L1 RCA | L1 recovery | L1 related failures | L2 audit | L2 history identity | final cascades | L4 policy/action | unsupported claims |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for row in gold_rows:
            lines.append(
                "| "
                f"{_md(row.get('target'))} | "
                f"{_score(row.get('gold_l1_root_cause_correct'))} | "
                f"{_score(row.get('gold_l1_recovery_correct'))} | "
                f"{_score(row.get('gold_l1_related_failure_recall'))} | "
                f"{_score(row.get('gold_l2_audit_correct'))} | "
                f"{_score(row.get('gold_l2_history_identity_correct'))} | "
                f"{_score(row.get('gold_l4_cascade_correct'))} | "
                f"{_score(row.get('gold_l4_policy_action'))} | "
                f"{_md(','.join(row.get('gold_l1_unsupported_claims') or []))} |"
            )

        agreement = _dict(panel.get("l2_root_fingerprint_agreement"))
        lines.extend(
            [
                "",
                "L2 history identity stability: "
                f"`{_score(agreement.get('gold_stability_correct'))}` "
                f"(expected unique identities=`{_md(agreement.get('gold_expected_identity_count') or 'not_scored')}`, "
                f"observed=`{_md(agreement.get('unique_fingerprints'))}`).",
            ]
        )

        path_comparison = panel.get("decision_path_comparison") or {}
        fallback = path_comparison.get("shared_fallback") or {}
        lines.extend(
            [
                "",
                "### Fallback Versus L1-Enriched Policy",
                "",
                (
                    "The deterministic fallback is scored once. Each model route is "
                    "then scored against the same gold action and retry-policy expectations."
                ),
                "",
                "| fallback consistency | decision | rule | action correct | policy/action correct |",
                "|---|---|---|---|---|",
                "| "
                f"{_md(path_comparison.get('fallback_consistency'))} | "
                f"{_md(fallback.get('decision'))} | "
                f"{_md(fallback.get('retry_rule'))} | "
                f"{_score(fallback.get('gold_action_correct'))} | "
                f"{_score(fallback.get('gold_policy_action'))} |",
                "",
                "| target | enriched decision | enriched rule | action correct | policy/action correct | action effect | policy/action effect |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for route in path_comparison.get("model_routes") or []:
            lines.append(
                "| "
                f"{_md(route.get('target'))} | "
                f"{_md(route.get('enriched_decision'))} | "
                f"{_md(route.get('enriched_retry_rule'))} | "
                f"{_score(route.get('gold_enriched_action_correct'))} | "
                f"{_score(route.get('gold_enriched_policy_action'))} | "
                f"{_md(route.get('action_effect'))} | "
                f"{_md(route.get('policy_action_effect'))} |"
            )


def _append_cross_route_outcome(
    lines: list[str], rows: list[dict[str, Any]], comparison_axes: dict[str, Any]
) -> None:
    lines.extend(
        [
            "",
            "## Cross-Route Outcome",
            "",
            "| target | contribution | reason | L1 | semantic | endpoint | decision | rule | quality | NVRx use | route s |",
            "|---|---|---|---|---|---|---|---|---|---|---:|",
        ]
    )
    outcomes_by_target = {
        str(row.get("target")): row for row in comparison_axes.get("route_outcome") or []
    }
    for row in rows:
        outcome = outcomes_by_target.get(str(row.get("target")), {})
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(outcome.get('model_contribution'))} | "
            f"{_md(outcome.get('reason'))} | "
            f"{_md(row.get('l1_execution_status'))} | "
            f"{_md(outcome.get('semantic_quality'))} | "
            f"{_md(outcome.get('endpoint_reliability'))} | "
            f"{_md(row.get('decision'))} | "
            f"{_md(row.get('l4_retry_rule'))} | "
            f"{_md(row.get('l4_result_quality'))} | "
            f"{_md(outcome.get('nvrx_use'))} | "
            f"{_md(outcome.get('latency_s'))} |"
        )


def _append_primary_by_stage(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "",
            "## Semantic Comparison",
            "",
            "### Primary By Stage",
            "",
            "| target | L0 deterministic | L1 semantic | L2 grounded | L1/L0 relation | L2/L0 relation |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_primary_label(row, 'l0_deterministic')} | "
            f"{_primary_label(row, 'l1_semantic')} | "
            f"{_primary_label(row, 'l2_grounded')} | "
            f"{_md(row.get('l1_primary_relation_to_l0'))} | "
            f"{_md(row.get('l2_primary_relation_to_l0'))} |"
        )


def _append_recovery_assessment(
    lines: list[str], model_rows: list[dict[str, Any]], comparison_axes: dict[str, Any]
) -> None:
    if model_rows:
        lines.extend(
            [
                "",
                "### Recovery Assessment",
                "",
                "| target | RCA status | domain | domain status | domain confidence | retry outlook | retry status | retry confidence |",
                "|---|---|---|---|---:|---|---|---:|",
            ]
        )
        for row in model_rows:
            lines.append(
                "| "
                f"{_md(row.get('target'))} | "
                f"{_md(row.get('model_root_cause_status'))} | "
                f"{_md(row.get('model_failure_domain'))} | "
                f"{_md(row.get('model_failure_domain_status'))} | "
                f"{_md(row.get('model_failure_domain_confidence'))} | "
                f"{_md(row.get('model_retry_outlook_without_workload_change'))} | "
                f"{_md(row.get('model_retry_outlook_status'))} | "
                f"{_md(row.get('model_retry_outlook_confidence'))} |"
            )

        _append_model_operations(lines, model_rows, comparison_axes)


def _append_shared_deterministic_evidence(
    lines: list[str],
    panel: dict[str, Any],
) -> None:
    evidence = _dict(panel.get("shared_decision_evidence"))
    consistency = _dict(panel.get("decision_evidence_consistency"))
    primary = _dict(evidence.get("deterministic_primary_candidate"))
    progress = _dict(evidence.get("progress_checkpoint_state"))
    shape = _dict(panel.get("shared_l0_shape"))
    lines.extend(
        [
            "",
            "## Shared Deterministic Evidence",
            "",
            f"- Decision Evidence consistency: `{consistency.get('status')}` across "
            f"`{consistency.get('available_models')}/{consistency.get('total_models')}` routes.",
            "",
            "| primary | outcome | phase | causal role | fallback root |",
            "|---|---|---|---|---|",
            "| "
            f"{_md(primary.get('fine_class'))}@{_md(primary.get('line'))} | "
            f"{_md(primary.get('fault_outcome'))} | "
            f"{_md(primary.get('phase'))} | "
            f"{_md(primary.get('causal_role'))} | "
            f"{_md(_short_identity(primary.get('root_fingerprint')))} |",
            "",
            "| first/last iteration | last progress | checkpoint load | last checkpoint | progress after episode |",
            "|---|---:|---:|---:|---|",
            "| "
            f"{_md(progress.get('first_iteration'))}/{_md(progress.get('last_iteration'))} | "
            f"{_md(progress.get('last_progress_line'))} | "
            f"{_md(progress.get('checkpoint_load_iteration'))} | "
            f"{_md(progress.get('last_checkpoint_iteration'))} | "
            f"{_yes_no(progress.get('progress_after_failure_episode'))} |",
            "",
            "- L0A: "
            f"lines=`{shape.get('line_count')}`, windows=`{shape.get('context_window_count')}`, "
            f"anchors=`{shape.get('candidate_anchor_count')}`, "
            f"occurrence_groups=`{shape.get('occurrence_group_count')}`, "
            f"episodes=`{shape.get('failure_episode_count')}`, "
            f"incidents=`{shape.get('distributed_failure_incident_count')}`, "
            f"build=`{_l0_build_label(shape)}`.",
            "- L0B: "
            f"characters=`{shape.get('l0b_compact_json_characters')}`, "
            f"estimated_tokens=`{shape.get('l0b_estimated_evidence_tokens')}`, "
            f"model_lines=`{shape.get('l0b_model_facing_context_lines')}`, "
            f"truncated_windows=`{shape.get('l0b_truncated_context_windows')}`, "
            f"integrity=`{shape.get('l0b_projection_integrity_status')}`.",
        ]
    )


def _score(value: Any) -> str:
    return "not_scored" if value is None else _yes_no(value)


def _l0_build_label(shape: dict[str, Any]) -> str:
    if shape.get("replayed"):
        return "replayed"
    value = shape.get("l0a_wall_clock_s")
    return "not_available" if value is None else f"{value}s"


def _append_history_and_policy(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "",
            "## History And Policy",
            "",
            "| target | history root ready | matching root | advanced | no observed advance | rule | prior/allowed | exhausted | decision |",
            "|---|---|---:|---:|---:|---|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_yes_no(row.get('current_history_identity_ready'))} | "
            f"{_md(row.get('l3_matching_root_attempts'))} | "
            f"{_md(row.get('l3_observed_advance_attempts'))} | "
            f"{_md(row.get('l3_no_observed_advance_attempts'))} | "
            f"{_md(row.get('l4_retry_rule'))} | "
            f"{_md(row.get('l4_matching_prior_failures'))}/"
            f"{_md(row.get('l4_allowed_retries'))} | "
            f"{_yes_no(row.get('l4_retry_budget_exhausted'))} | "
            f"{_md(row.get('decision'))} |"
        )


def _append_conditional_diagnostics(lines: list[str], panel: dict[str, Any]) -> None:
    rows = panel["rows"]
    diagnostic_rows = [
        row
        for row in rows
        if row.get("l1_execution_status") not in {None, "ok", "not_run"}
        or _int_or_zero(row.get("l1_kpi_tool_calls"))
        or _int_or_zero(row.get("l2_material_finding_count"))
        or _int_or_zero(row.get("l1_kpi_context_budget_adjusted_calls"))
    ]
    root_agreement = _dict(panel.get("l2_root_fingerprint_agreement"))
    root_attention = (
        root_agreement.get("status") == "unstable"
        or root_agreement.get("disagreement_reason") == "missing_fingerprints"
    )
    if not diagnostic_rows and not root_attention:
        return
    lines.extend(
        [
            "",
            "## Conditional Diagnostics",
            "",
            "Only non-trivial interaction or validation signals are shown here. Full stage detail is in `panel_diagnostics.md`.",
        ]
    )
    if diagnostic_rows:
        lines.extend(
            [
                "",
                "| target | L1 status | tools | no-new | tool dependency | L2 grounding | material findings | budget adjustments |",
                "|---|---|---:|---:|---|---|---:|---:|",
            ]
        )
        for row in diagnostic_rows:
            lines.append(
                "| "
                f"{_md(row.get('target'))} | "
                f"{_md(row.get('l1_execution_status'))} | "
                f"{_md(row.get('l1_kpi_tool_calls'))} | "
                f"{_md(row.get('l1_kpi_no_new_prompt_line_tool_calls'))} | "
                f"{_md(row.get('tool_final_context_impact'))} | "
                f"{_md(row.get('l2_grounding_status'))} | "
                f"{_md(row.get('l2_material_finding_count'))} | "
                f"{_md(row.get('l1_kpi_context_budget_adjusted_calls'))} |"
            )
    lines.append(
        "- L2 root fingerprint agreement: "
        f"status=`{root_agreement.get('status')}`, "
        f"available=`{root_agreement.get('available_models')}/"
        f"{root_agreement.get('total_models')}`, "
        f"unique=`{root_agreement.get('unique_fingerprints')}`, "
        f"reason=`{root_agreement.get('disagreement_reason')}`."
    )
    if root_agreement.get("status") == "unstable":
        for row in rows:
            lines.append(
                f"  - `{row.get('target')}`: "
                f"`{_short_identity(row.get('l2_root_fingerprint'))}`"
            )


def _append_artifact_paths(lines: list[str], panel: dict[str, Any]) -> None:
    paths = _dict(panel.get("artifact_paths"))
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "| artifact | path |",
            "|---|---|",
        ]
    )
    for name, path in paths.items():
        lines.append(f"| {_md(name)} | `{Path(str(path)).name}` |")
    for row in panel["rows"]:
        artifacts = _dict(row.get("artifacts"))
        review_path = artifacts.get("review_md")
        if review_path:
            lines.append(f"| {_md(row.get('target'))} review | `{Path(str(review_path)).name}` |")


def _append_model_operations(
    lines: list[str],
    rows: list[dict[str, Any]],
    comparison_axes: dict[str, Any],
) -> None:
    lines.extend(
        [
            "",
            "## Operational Comparison",
            "",
            "Behavioral work and endpoint delivery remain separate dimensions.",
            "",
            "### Behavioral Efficiency",
            "",
            "| target | first-turn usable | turns | tool turns | tool calls | no-new | tokens | L1 s |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    efficiency_by_target = {
        str(row.get("target")): row for row in comparison_axes.get("behavioral_efficiency") or []
    }
    for row in rows:
        efficiency = efficiency_by_target.get(str(row.get("target")), {})
        first_turn = efficiency.get("first_turn_usable")
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_yes_no(first_turn) if first_turn is not None else 'not_observed'} | "
            f"{_md(efficiency.get('model_turns'))} | "
            f"{_md(efficiency.get('tool_driven_turns'))} | "
            f"{_md(efficiency.get('tool_calls'))} | "
            f"{_md(efficiency.get('no_new_context_tool_calls'))} | "
            f"{_md(efficiency.get('total_tokens'))} | "
            f"{_md(row.get('l1_kpi_wall_clock_s'))} |"
        )

    lines.extend(
        [
            "",
            "### Endpoint Reliability",
            "",
            "| target | status | attempts | failed | retries | timeouts | HTTP errors | provider errors |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    endpoint_by_target = {
        str(row.get("target")): row for row in comparison_axes.get("endpoint_reliability") or []
    }
    for row in rows:
        endpoint = endpoint_by_target.get(str(row.get("target")), {})
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(endpoint.get('status'))} | "
            f"{_md(endpoint.get('attempts'))} | "
            f"{_md(endpoint.get('failed_attempts'))} | "
            f"{_md(endpoint.get('retried_attempts'))} | "
            f"{_md(endpoint.get('timeouts'))} | "
            f"{_md(endpoint.get('http_errors'))} | "
            f"{_md(endpoint.get('provider_errors'))} |"
        )


def _short_hash(value: Any, *, length: int = 12) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text.startswith("sha256:"):
        return "sha256:" + text.split(":", 1)[1][:length]
    return text[:length]
