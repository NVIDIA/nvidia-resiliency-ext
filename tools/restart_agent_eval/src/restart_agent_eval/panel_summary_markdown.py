# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concise panel-summary Markdown rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .panel_format import (
    _affected_entity_label,
    _claim_confidence_label,
    _dict,
    _md,
    _primary_label,
    _short_identity,
    _yes_no,
)
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
    _append_failure_tracks_and_selection(lines, rows)
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
                "| observation selected | observation identity | primary absent |",
                "|---|---|---|",
                "| "
                f"{_score(first.get('gold_l0a_selected_observation_accuracy'))} | "
                f"{_score(first.get('gold_l0a_observation_fingerprint_accuracy'))} | "
                f"{_score(first.get('gold_l0a_primary_absence_correct'))} |",
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
        deterministic = path_comparison.get("shared_deterministic") or {}
        lines.extend(
            [
                "",
                "### Fallback Versus L1-Enriched Policy",
                "",
                (
                    "The deterministic recommendation is scored once. Each model route is "
                    "then scored against the same gold action and retry-policy expectations."
                ),
                "",
                "| deterministic consistency | decision | rule | action correct | policy/action correct |",
                "|---|---|---|---|---|",
                "| "
                f"{_md(path_comparison.get('deterministic_consistency'))} | "
                f"{_md(deterministic.get('decision'))} | "
                f"{_md(deterministic.get('retry_rule'))} | "
                f"{_score(deterministic.get('gold_action_correct'))} | "
                f"{_score(deterministic.get('gold_policy_action'))} |",
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
            f"{_policy_rule_cell(row)} | "
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
    lines.extend(
        [
            "",
            "### Selected Observation By Stage",
            "",
            "A selected observation preserves a visible terminal failure surface when no initiating primary is supportable. It is not a root cause.",
            "",
            "| target | L0 deterministic | L1 semantic | L2 grounded | L1/L0 relation | L2/L0 relation |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_observation_label(row, 'l0')} | "
            f"{_observation_label(row, 'l1')} | "
            f"{_observation_label(row, 'l2')} | "
            f"{_md(row.get('l1_observation_relation_to_l0'))} | "
            f"{_md(row.get('l2_observation_relation_to_l0'))} |"
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
                f"{_claim_confidence_label(row.get('model_failure_domain'), row.get('model_failure_domain_status'), row.get('model_failure_domain_confidence'))} | "
                f"{_md(row.get('model_retry_outlook_without_workload_change'))} | "
                f"{_md(row.get('model_retry_outlook_status'))} | "
                f"{_claim_confidence_label(row.get('model_retry_outlook_without_workload_change'), row.get('model_retry_outlook_status'), row.get('model_retry_outlook_confidence'))} |"
            )

        _append_model_operations(lines, model_rows, comparison_axes)


def _append_shared_deterministic_evidence(
    lines: list[str],
    panel: dict[str, Any],
) -> None:
    evidence = _dict(panel.get("shared_decision_evidence"))
    consistency = _dict(panel.get("decision_evidence_consistency"))
    primary = _dict(evidence.get("deterministic_primary_candidate"))
    observation = _dict(evidence.get("selected_observed_failure"))
    progress = _dict(evidence.get("progress_checkpoint_state"))
    shape = _dict(panel.get("shared_l0_shape"))
    l0_entity = next(
        (
            row.get("l0_affected_entity")
            for row in panel.get("rows") or []
            if row.get("l0_affected_entity")
        ),
        None,
    )
    lines.extend(
        [
            "",
            "## Shared Deterministic Evidence",
            "",
            f"- Decision Evidence consistency: `{consistency.get('status')}` across "
            f"`{consistency.get('available_models')}/{consistency.get('total_models')}` routes.",
            "",
            "| primary | outcome | phase | causal role | deterministic root | affected entity |",
            "|---|---|---|---|---|---|",
            "| "
            f"{_md(primary.get('failure_class'))}@{_md(primary.get('line'))} | "
            f"{_md(primary.get('fault_outcome'))} | "
            f"{_md(primary.get('phase'))} | "
            f"{_md(primary.get('causal_role'))} | "
            f"{_md(_short_identity(primary.get('root_fingerprint')))} | "
            f"{_md(_affected_entity_label(l0_entity))} |",
            "",
            "| selected observation | causal role | observation identity | policy role |",
            "|---|---|---|---|",
            "| "
            f"{_md(observation.get('failure_class'))}@{_md(observation.get('line'))} | "
            f"{_md(observation.get('causal_role'))} | "
            f"{_md(_short_identity(observation.get('observation_fingerprint')))} | "
            f"{_md('root-independent only' if observation else 'n/a')} |",
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
            f"narrative=`{shape.get('l0b_narrative_status')}`/"
            f"`{shape.get('l0b_narrative_event_count')}` events, "
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
            "## History",
            "",
            "| target | identity kind | identity ready | root no-advance | concrete no-advance | observation no-advance | job no-advance | job unknown | job advanced |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(row.get('current_identity_kind'))} | "
            f"{_yes_no(row.get('current_history_identity_ready'))} | "
            f"{_md(row.get('l3_consecutive_same_root_no_advance_attempts'))} | "
            f"{_md(row.get('l3_consecutive_same_root_and_entity_no_advance_attempts'))} | "
            f"{_md(row.get('l3_consecutive_same_observation_no_advance_attempts'))} | "
            f"{_md(row.get('l3_consecutive_same_job_no_advance_attempts'))} | "
            f"{_md(row.get('l3_consecutive_same_job_unknown_progress_attempts'))} | "
            f"{_yes_no(row.get('l3_job_progress_advanced'))} |"
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "| target | base -> effective | root ledger | selected ledger | job guards | exhausted by | decision |",
            "|---|---|---:|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_policy_rule_cell(row)} | "
            f"{_ledger_cell(row, 'l4_general_root')} | "
            f"{_selected_ledger_cell(row)} | "
            f"{_job_guard_cell(row)} | "
            f"{_md(','.join(row.get('l4_exhausted_by') or []) or '-')} | "
            f"{_md(row.get('decision'))} |"
        )


def _append_failure_tracks_and_selection(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "",
            "## Failure Tracks And L4 Selection",
            "",
            "Track state and like-kind history are shown independently; availability is not an accuracy score.",
            "",
            "| target | deterministic | primary | observation | deterministic history | primary history | observation history | L4 path | reason |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(row.get('deterministic_track_status'))} | "
            f"{_md(row.get('primary_track_status'))} | "
            f"{_md(row.get('observation_track_status'))} | "
            f"{_md(row.get('deterministic_history_status'))} | "
            f"{_md(row.get('primary_history_status'))} | "
            f"{_md(row.get('observation_history_status'))} | "
            f"{_md(row.get('l4_selected_path'))} | "
            f"{_md(row.get('l4_path_selection_reason'))} |"
        )


def _ledger_cell(row: dict[str, Any], prefix: str) -> str:
    if row.get(f"{prefix}_applicable") is False:
        return _md(row.get(f"{prefix}_inapplicable_reason") or "n/a")
    return (
        f"{_md(row.get(f'{prefix}_matching_prior_attempts'))}/"
        f"{_md(row.get(f'{prefix}_allowed_retries'))}"
    )


def _selected_ledger_cell(row: dict[str, Any]) -> str:
    if not row.get("l4_selected_policy_ledger_present"):
        return "-"
    return (
        f"{_md(row.get('l4_selected_policy_ledger_rule'))}:"
        f"{_md(row.get('l4_selected_policy_matching_prior_attempts'))}/"
        f"{_md(row.get('l4_selected_policy_allowed_retries'))}"
    )


def _policy_rule_cell(row: dict[str, Any]) -> str:
    base = _md(row.get("l4_base_rule"))
    effective = _md(row.get("l4_effective_rule"))
    return base if base == effective else f"{base} -> {effective}"


def _job_guard_cell(row: dict[str, Any]) -> str:
    no_progress = (
        f"no-progress "
        f"{_md(row.get('l4_job_no_progress_matching_prior_attempts'))}/"
        f"{_md(row.get('l4_job_no_progress_allowed_retries'))}"
    )
    unknown = (
        f"unknown "
        f"{_md(row.get('l4_job_unknown_matching_prior_attempts'))}/"
        f"{_md(row.get('l4_job_unknown_allowed_retries'))}"
    )
    return f"{no_progress}; {unknown}"


def _append_conditional_diagnostics(lines: list[str], panel: dict[str, Any]) -> None:
    rows = panel["rows"]
    diagnostic_rows = [
        row
        for row in rows
        if row.get("l1_execution_status") not in {None, "ok", "not_run"}
        or _int_or_zero(row.get("l1_kpi_tool_calls"))
        or _int_or_zero(row.get("l2_observational_finding_count"))
        or _int_or_zero(row.get("l1_kpi_context_budget_adjusted_calls"))
    ]
    root_agreement = _dict(panel.get("l2_root_fingerprint_agreement"))
    entity_agreement = _dict(panel.get("affected_entity_agreement"))
    root_attention = (
        root_agreement.get("status") == "unstable"
        or root_agreement.get("disagreement_reason") == "missing_fingerprints"
    )
    entity_attention = entity_agreement.get("status") == "unstable" or (
        _int_or_zero(entity_agreement.get("available_models")) > 0
        and entity_agreement.get("disagreement_reason") == "missing_fingerprints"
    )
    if not diagnostic_rows and not root_attention and not entity_attention:
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
                "| target | L1 status | tools | semantic new/no-new/unassessed | tool dependency | L2 grounding | observational findings | budget adjustments |",
                "|---|---|---:|---|---|---|---:|---:|",
            ]
        )
        for row in diagnostic_rows:
            lines.append(
                "| "
                f"{_md(row.get('target'))} | "
                f"{_md(row.get('l1_execution_status'))} | "
                f"{_md(row.get('l1_kpi_tool_calls'))} | "
                f"{_md(row.get('l1_kpi_tool_semantic_new_evidence_calls'))}/"
                f"{_md(row.get('l1_kpi_tool_semantic_no_new_evidence_calls'))}/"
                f"{_md(row.get('l1_kpi_tool_semantic_novelty_unassessed_calls'))} | "
                f"{_md(row.get('tool_final_context_impact'))} | "
                f"{_md(row.get('l2_grounding_status'))} | "
                f"{_md(row.get('l2_observational_finding_count'))} | "
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
    observation_agreement = _dict(panel.get("l2_observation_fingerprint_agreement"))
    if observation_agreement.get("available_models"):
        lines.append(
            "- L2 observation fingerprint agreement: "
            f"status=`{observation_agreement.get('status')}`, "
            f"available=`{observation_agreement.get('available_models')}/"
            f"{observation_agreement.get('total_models')}`, "
            f"unique=`{observation_agreement.get('unique_fingerprints')}`, "
            f"reason=`{observation_agreement.get('disagreement_reason')}`."
        )
    if root_agreement.get("status") == "unstable":
        for row in rows:
            lines.append(
                f"  - `{row.get('target')}`: "
                f"`{_short_identity(row.get('l2_root_fingerprint'))}`"
            )
    if entity_attention:
        lines.append(
            "- Affected-entity agreement: "
            f"status=`{entity_agreement.get('status')}`, "
            f"available=`{entity_agreement.get('available_models')}/"
            f"{entity_agreement.get('total_models')}`, "
            f"unique=`{entity_agreement.get('unique_fingerprints')}`, "
            f"reason=`{entity_agreement.get('disagreement_reason')}`."
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
            "| target | first-turn usable | turns | tool turns | tool calls | semantic new/no-new/unassessed | final impact | tokens | L1 s |",
            "|---|---|---:|---:|---:|---|---|---:|---:|",
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
            f"{_md(efficiency.get('semantic_new_evidence_calls'))}/"
            f"{_md(efficiency.get('semantic_no_new_evidence_calls'))}/"
            f"{_md(efficiency.get('semantic_novelty_unassessed_calls'))} | "
            f"{_md(efficiency.get('final_context_impact'))} | "
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


def _observation_label(row: dict[str, Any], stage: str) -> str:
    line = row.get(f"{stage}_selected_observation_line")
    failure_class = row.get(f"{stage}_selected_observation_class")
    if line is None and not failure_class:
        return "-"
    return f"{_md(failure_class)}@{_md(line)}"
