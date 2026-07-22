# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detailed panel diagnostics Markdown rendering."""

from __future__ import annotations

from typing import Any

from .panel_format import _dict, _md, _primary_label, _short_identity, _yes_no


def _panel_diagnostics_markdown(panel: dict[str, Any]) -> str:
    lines = _diagnostics_header(panel)
    _append_comparison_axes(panel, lines)
    _append_fallback_and_decision_evidence(panel, lines)
    _append_primary_identity_and_gold(panel, lines)
    _append_latency_and_cost(panel, lines)
    _append_shared_l0_and_execution_context(panel, lines)
    _append_l0_coverage_and_tool_profiles(panel, lines)
    _append_l1_call_diagnostics(panel, lines)
    _append_l2_diagnostics(panel, lines)
    _append_l1_semantic_assessment(panel, lines)
    _append_l3_history(panel, lines)
    _append_l4_policy(panel, lines)
    _append_concerns_and_notes(panel, lines)
    return "\n".join(lines)


def _diagnostics_header(panel: dict[str, Any]) -> list[str]:
    config = _dict(panel.get("restart_agent_config"))
    lines = [
        "# Restart Agent Model Panel Summary",
        "",
        f"- run_dir: `{panel['run_dir']}`",
        f"- run_dir_name: `{panel.get('run_dir_name')}`",
        f"- source_log_name: `{panel.get('source_log_name')}`",
        f"- model_count: `{panel['model_count']}`",
        f"- decision_counts: `{panel['decision_counts']}`",
        f"- primary_class_counts: `{panel['primary_class_counts']}`",
        f"- primary_line_counts: `{panel['primary_line_counts']}`",
        f"- config_id: `{config.get('config_id') or '<not recorded>'}`",
        f"- config_version: `{config.get('config_version') or '<not recorded>'}`",
        f"- config_fingerprint: `{config.get('config_fingerprint') or '<not recorded>'}`",
        "",
        "## Paths",
        "",
        "```text",
        f"source_log:  {panel.get('source_log_path') or '<unknown>'}",
        f"review_dir:  {panel.get('run_dir') or '<unknown>'}",
        f"product_repo: {panel.get('product_repo') or '<unknown>'}",
        f"shared_l0_bundle: {panel.get('shared_l0_bundle') or '<unknown>'}",
        f"gold_label: {panel.get('gold_label_path') or '<none>'}",
        "```",
        "",
        "## Route Outcome",
        "",
        "This is the combined operational view. It does not replace the three independent axes below.",
        "",
        "| target | contribution | L1 status | semantic | endpoint | decision | result quality | NVRx use | route s | basis |",
        "|---|---|---|---|---|---|---|---|---:|---|",
    ]
    return lines


def _append_comparison_axes(panel: dict[str, Any], lines: list[str]) -> None:
    comparison_axes = _dict(panel.get("comparison_axes"))
    for row in comparison_axes.get("route_outcome") or []:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(row.get('model_contribution'))} | "
            f"{_md(row.get('l1_execution_status'))} | "
            f"{_md(row.get('semantic_quality'))} | "
            f"{_md(row.get('endpoint_reliability'))} | "
            f"{_md(row.get('decision'))} | "
            f"{_md(row.get('result_quality'))} | "
            f"{_md(row.get('nvrx_use'))} | "
            f"{_md(row.get('latency_s'))} | "
            f"{_md(row.get('latency_basis'))} |"
        )

    lines.extend(
        [
            "",
            "## Semantic Quality",
            "",
            "Scored only when a model response was delivered and human-approved gold is available. Endpoint-only failures are `not_observed`, not semantic failures.",
            "",
            "| target | status | semantic primary | root cause | recovery assessment | related failures | final cascades | unsupported claims | semantic confidence |",
            "|---|---|---|---|---|---|---|---|---:|",
        ]
    )
    for row in comparison_axes.get("semantic_quality") or []:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(row.get('status'))} | "
            f"{_md(row.get('primary'))} | "
            f"{_yes_no(row.get('root_cause_correct'))} | "
            f"{_yes_no(row.get('recovery_correct'))} | "
            f"{_yes_no(row.get('related_failure_recall'))} | "
            f"{_yes_no(row.get('final_cascade_correct'))} | "
            f"{_md(','.join(row.get('unsupported_claims') or []))} | "
            f"{_md(row.get('confidence'))} |"
        )

    lines.extend(
        [
            "",
            "## Behavioral Efficiency",
            "",
            "Interaction work is reported independently from provider delivery failures. No synthetic efficiency score is applied.",
            "",
            "| target | first-turn usable | model turns | tool turns | repair turns | tool calls | duplicate | no-new | tokens |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_axes.get("behavioral_efficiency") or []:
        first_turn = row.get("first_turn_usable")
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_yes_no(first_turn) if first_turn is not None else 'not_observed'} | "
            f"{_md(row.get('model_turns'))} | "
            f"{_md(row.get('tool_driven_turns'))} | "
            f"{_md(row.get('contract_repair_turns'))} | "
            f"{_md(row.get('tool_calls'))} | "
            f"{_md(row.get('duplicate_tool_calls'))} | "
            f"{_md(row.get('no_new_context_tool_calls'))} | "
            f"{_md(row.get('total_tokens'))} |"
        )

    lines.extend(
        [
            "",
            "## Endpoint Reliability",
            "",
            "Provider delivery is measured across all attempts and is never counted as semantic correctness.",
            "",
            "| target | status | attempts | successful | failed | retries | timeouts | HTTP errors | provider errors |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_axes.get("endpoint_reliability") or []:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(row.get('status'))} | "
            f"{_md(row.get('attempts'))} | "
            f"{_md(row.get('successful_attempts'))} | "
            f"{_md(row.get('failed_attempts'))} | "
            f"{_md(row.get('retried_attempts'))} | "
            f"{_md(row.get('timeouts'))} | "
            f"{_md(row.get('http_errors'))} | "
            f"{_md(row.get('provider_errors'))} |"
        )

    provider_timing_rows = [
        row for row in panel["rows"] if row.get("l1_kpi_provider_reported_timing")
    ]
    if provider_timing_rows:
        lines.extend(
            [
                "",
                "## Provider-Reported L1 Timing",
                "",
                "These optional response-header spans refine L1 route timing. Downstream LLM API is the proxy's downstream-call span, not model compute time; it may include backend transport, queueing, prefill, and decode.",
                "",
                "| target | reported calls | client call s | downstream LLM API s | proxy pre ms | proxy post ms | message copy ms |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in provider_timing_rows:
            timing = _dict(row.get("l1_kpi_provider_reported_timing"))
            components = _dict(timing.get("components_ms_total"))
            downstream_ms = components.get("downstream_llm_api_ms")
            downstream_s = (
                round(float(downstream_ms) / 1000.0, 3) if downstream_ms is not None else None
            )
            lines.append(
                "| "
                f"{_md(row.get('target'))} | "
                f"{_md(timing.get('reported_call_count'))} | "
                f"{_md(row.get('l1_kpi_model_call_wall_clock_s'))} | "
                f"{_md(downstream_s)} | "
                f"{_md(components.get('proxy_pre_processing_ms'))} | "
                f"{_md(components.get('proxy_post_processing_ms'))} | "
                f"{_md(components.get('proxy_message_copy_ms'))} |"
            )


def _append_fallback_and_decision_evidence(panel: dict[str, Any], lines: list[str]) -> None:
    decision_evidence = _dict(panel.get("shared_decision_evidence"))
    decision_consistency = _dict(panel.get("decision_evidence_consistency"))
    lines.extend(
        [
            "",
            "## Deterministic Fallback Inputs",
            "",
            "This is the shared, model-independent L0 `DecisionEvidence` used by the "
            "fallback path when L1 is late or unavailable. It is shown once because L0 "
            "is built once per log and replayed unchanged to the model panel.",
            "",
            "- consistency: "
            f"status=`{decision_consistency.get('status')}`, "
            f"available_models=`{decision_consistency.get('available_models')}/"
            f"{decision_consistency.get('total_models')}`, "
            f"unique_payloads=`{decision_consistency.get('unique_payloads')}`",
        ]
    )
    if decision_evidence:
        fallback_primary = _dict(decision_evidence.get("deterministic_primary_candidate"))
        canonical_identity = _dict(decision_evidence.get("canonical_observed_identity"))
        progress = _dict(decision_evidence.get("progress_checkpoint_state"))
        references = _dict(decision_evidence.get("selected_evidence_references"))
        fallback_fingerprint = fallback_primary.get("root_fingerprint")
        canonical_fingerprint = canonical_identity.get("root_fingerprint")
        lines.extend(
            [
                f"- schema: `{decision_evidence.get('schema_version')}`",
                "",
                "### L0 Primary And State",
                "",
                "| primary | policy class | outcome | phase | causal role |",
                "|---|---|---|---|---|",
                "| "
                f"{_md(fallback_primary.get('fine_class'))}@"
                f"{_md(fallback_primary.get('line'))} | "
                f"{_md(fallback_primary.get('policy_class'))} | "
                f"{_md(fallback_primary.get('fault_outcome'))} | "
                f"{_md(fallback_primary.get('phase'))} | "
                f"{_md(fallback_primary.get('causal_role'))} |",
                "",
                "### L0 Fallback Root-Fingerprint KPI",
                "",
                "L0 owns the deterministic fallback fingerprint. L3 consumes this key "
                "when the model-enriched path is unavailable; L1 does not create it. "
                "Runtime availability is shown here. Accuracy and false merge/split "
                "rates require reviewed corpus labels.",
                "",
                "| owner | available | history ready | source | canonical anchor | anchor reason |",
                "|---|---|---|---|---:|---|",
                "| "
                "L0 | "
                f"{_yes_no(bool(fallback_fingerprint))} | "
                f"{_yes_no(bool(fallback_fingerprint))} | "
                f"{_md(fallback_primary.get('root_fingerprint_source'))} | "
                f"{_md(canonical_identity.get('identity_anchor_line'))} | "
                f"{_md(canonical_identity.get('identity_anchor_reason'))} |",
                "",
                "#### L0 Fallback Root Fingerprint",
                "",
                "```text",
                str(fallback_fingerprint or "<none>"),
                "```",
            ]
        )
        if canonical_fingerprint != fallback_fingerprint:
            lines.extend(
                [
                    "",
                    "#### Canonical L0 Root Fingerprint",
                    "",
                    "```text",
                    str(canonical_fingerprint or "<none>"),
                    "```",
                ]
            )
        lines.extend(
            [
                "",
                "### Fallback Progress And Checkpoint Facts",
                "",
                "| first iteration | last iteration | last progress line | checkpoint load | last completed checkpoint | progress after episode |",
                "|---:|---:|---:|---:|---:|---|",
                "| "
                f"{_md(progress.get('first_iteration'))} | "
                f"{_md(progress.get('last_iteration'))} | "
                f"{_md(progress.get('last_progress_line'))} | "
                f"{_md(progress.get('checkpoint_load_iteration'))} | "
                f"{_md(progress.get('last_checkpoint_iteration'))} | "
                f"{_yes_no(progress.get('progress_after_failure_episode'))} |",
                "",
                "- selected L0A references: "
                f"source_lines=`{references.get('source_lines') or []}`, "
                f"anchors=`{len(references.get('candidate_anchor_ids') or [])}`, "
                f"windows=`{len(references.get('context_window_ids') or [])}`, "
                f"episodes=`{len(references.get('failure_episode_ids') or [])}`, "
                f"incidents=`{len(references.get('distributed_incident_ids') or [])}`, "
                f"occurrence_groups=`{len(references.get('occurrence_group_ids') or [])}`",
            ]
        )
    else:
        lines.extend(["", "No shared L0 `DecisionEvidence` payload is available."])


def _append_primary_identity_and_gold(panel: dict[str, Any], lines: list[str]) -> None:
    _append_primary_selection(panel, lines)
    _append_root_fingerprints(panel, lines)
    _append_experimental_identity(panel, lines)
    _append_gold_results(panel, lines)


def _append_primary_selection(panel: dict[str, Any], lines: list[str]) -> None:
    lines.extend(
        [
            "",
            "## Primary Selection By Stage",
            "",
            "L0 is the deterministic candidate selected from current-log evidence. "
            "L1 is the model's raw semantic selection. L2 grounds its evidence and "
            "derives stable identity; separate audit findings do not replace L1 semantics.",
            "",
            "| target | L0 deterministic | L1 semantic | L2 grounded |",
            "|---|---|---|---|",
        ]
    )
    for row in panel["rows"]:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_primary_label(row, 'l0_deterministic')} | "
            f"{_primary_label(row, 'l1_semantic')} | "
            f"{_primary_label(row, 'l2_grounded')} |"
        )
    lines.extend(
        [
            "",
            "| target | L1 relative to L0 | L2 relative to L0 |",
            "|---|---|---|",
        ]
    )
    for row in panel["rows"]:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(row.get('l1_primary_relation_to_l0'))} | "
            f"{_md(row.get('l2_primary_relation_to_l0'))} |"
        )


def _append_root_fingerprints(panel: dict[str, Any], lines: list[str]) -> None:
    if any(row.get("l2_root_fingerprint") or row.get("l2_audit_ran") for row in panel["rows"]):
        root_agreement = _dict(panel.get("l2_root_fingerprint_agreement"))
        lines.extend(
            [
                "",
                "## L2 Enriched Root-Fingerprint KPI",
                "",
                "L2 owns the deterministic fingerprint for each usable model-enriched "
                "path. It grounds the L1-selected primary against observed evidence; "
                "L3 only compares the resulting key. `matches L0` is diagnostic because "
                "a different valid primary may legitimately produce a different key. "
                "Gold accuracy and false merge/split rates remain corpus KPIs.",
                "",
                "- agreement: "
                f"status=`{root_agreement.get('status')}`, "
                f"total_models=`{root_agreement.get('total_models')}`, "
                f"available_models=`{root_agreement.get('available_models')}`, "
                f"unique_fingerprints=`{root_agreement.get('unique_fingerprints')}`, "
                f"all_available_agree=`{root_agreement.get('all_available_agree')}`, "
                f"disagreement_reason=`{root_agreement.get('disagreement_reason')}`",
                "",
                "| target | available | history ready | matches L0 | identity anchor | source | id |",
                "|---|---|---|---|---:|---|---|",
            ]
        )
        for row in panel["rows"]:
            lines.append(
                "| "
                f"{_md(row.get('target'))} | "
                f"{_yes_no(row.get('l2_root_fingerprint_available'))} | "
                f"{_yes_no(row.get('l2_history_identity_ready'))} | "
                f"{_yes_no(row.get('l2_matches_l0_root_fingerprint'))} | "
                f"{_md(row.get('stable_identity_anchor_line'))} | "
                f"{_md(row.get('l2_root_fingerprint_source'))} | "
                f"{_md(_short_identity(row.get('l2_root_fingerprint')))} |"
            )
        history_roots = {
            str(row["l2_root_fingerprint"])
            for row in panel["rows"]
            if row.get("l2_root_fingerprint")
        }
        if len(history_roots) == 1:
            lines.extend(
                [
                    "",
                    "### Shared L2 Root Fingerprint",
                    "",
                    "```text",
                    next(iter(history_roots)),
                    "```",
                ]
            )
        elif history_roots:
            lines.extend(["", "### L2 Root Fingerprints", ""])
            for row in panel["rows"]:
                if row.get("l2_root_fingerprint"):
                    lines.append(
                        f"- {_md(row.get('target'))}: " f"`{row.get('l2_root_fingerprint')}`"
                    )


def _append_experimental_identity(panel: dict[str, Any], lines: list[str]) -> None:
    if any(row.get("family_fingerprint") for row in panel["rows"]):
        lines.extend(["", "## Experimental Failure Identity", ""])
        lines.append(
            "These identities are observational only and do not affect L3 history or L4 policy."
        )
        lines.extend(
            [
                "",
                "### Family Identity",
                "",
                "| target | operation | mechanism | exception | complete | id |",
                "|---|---|---|---|---|---|",
            ]
        )
        for row in panel["rows"]:
            lines.append(
                "| "
                f"{_md(row.get('target'))} | "
                f"{_md(row.get('family_operation'))} | "
                f"{_md(row.get('family_mechanism'))} | "
                f"{_md(row.get('family_exception_type'))} | "
                f"{_yes_no(row.get('family_complete'))} | "
                f"{_md(_short_identity(row.get('family_fingerprint')))} |"
            )

        lines.extend(
            [
                "",
                "### Concrete Identity",
                "",
                "| target | component | callsite | position | complete | id |",
                "|---|---|---|---|---|---|",
            ]
        )
        for row in panel["rows"]:
            lines.append(
                "| "
                f"{_md(row.get('target'))} | "
                f"{_md(row.get('concrete_component'))} | "
                f"{_md(row.get('concrete_callsite'))} | "
                f"{_md(row.get('concrete_failure_position'))} | "
                f"{_yes_no(row.get('concrete_complete'))} | "
                f"{_md(_short_identity(row.get('concrete_fingerprint')))} |"
            )
        artifact_rows = [row for row in panel["rows"] if row.get("concrete_artifact_path")]
        if artifact_rows:
            lines.extend(["", "### Grounded Artifact Paths", ""])
            for row in artifact_rows:
                lines.append(
                    f"- {_md(row.get('target'))}: " f"`{row.get('concrete_artifact_path')}`"
                )

        client_agreement = _dict(panel.get("client_concrete_agreement"))
        lines.extend(
            [
                "",
                "### Client Concrete Identity",
                "",
                "This additive fingerprint is built from the canonical terminal exception of the grounded primary's L0 failure episode and deterministic source-log context, with primary-line fallback when no episode exists.",
                "",
                "- agreement: "
                f"available_models=`{client_agreement.get('available_models')}`, "
                f"unique_fingerprints=`{client_agreement.get('unique_fingerprints')}`, "
                f"all_available_agree=`{client_agreement.get('all_available_agree')}`, "
                f"same_primary_consistent=`{client_agreement.get('same_primary_consistent')}`, "
                f"disagreement_reason=`{client_agreement.get('disagreement_reason')}`",
                "",
                "| target | exception | callsite | position | complete | id |",
                "|---|---|---|---|---|---|",
            ]
        )
        for row in panel["rows"]:
            lines.append(
                "| "
                f"{_md(row.get('target'))} | "
                f"{_md(row.get('client_concrete_exception_type'))} | "
                f"{_md(row.get('client_concrete_callsite'))} | "
                f"{_md(row.get('client_concrete_failure_position'))} | "
                f"{_yes_no(row.get('client_concrete_complete'))} | "
                f"{_md(_short_identity(row.get('client_concrete_fingerprint')))} |"
            )
        if any(
            row.get("client_concrete_phase")
            or row.get("client_concrete_checkpoint_iteration") is not None
            or row.get("client_concrete_operation_signature")
            for row in panel["rows"]
        ):
            lines.extend(
                [
                    "",
                    "### Client Observed Operation Context",
                    "",
                    "| target | phase | checkpoint iteration | operation signature |",
                    "|---|---|---:|---|",
                ]
            )
            for row in panel["rows"]:
                lines.append(
                    "| "
                    f"{_md(row.get('target'))} | "
                    f"{_md(row.get('client_concrete_phase'))} | "
                    f"{_md(row.get('client_concrete_checkpoint_iteration'))} | "
                    f"{_md(row.get('client_concrete_operation_signature'))} |"
                )
        client_labels = {
            str(row["client_concrete_label"])
            for row in panel["rows"]
            if row.get("client_concrete_label")
        }
        if len(client_labels) == 1:
            lines.extend(
                [
                    "",
                    "### Shared Client Concrete Label",
                    "",
                    "```text",
                    next(iter(client_labels)),
                    "```",
                ]
            )
        elif client_labels:
            lines.extend(["", "### Client Concrete Labels", ""])
            for row in panel["rows"]:
                if row.get("client_concrete_label"):
                    lines.append(
                        f"- {_md(row.get('target'))}: " f"`{row.get('client_concrete_label')}`"
                    )
        source_rows = [row for row in panel["rows"] if row.get("client_concrete_source_file")]
        if source_rows:
            lines.extend(["", "### Client Traceback Source Files", ""])
            for row in source_rows:
                lines.append(
                    f"- {_md(row.get('target'))}: " f"`{row.get('client_concrete_source_file')}`"
                )
        client_artifact_rows = [
            row for row in panel["rows"] if row.get("client_concrete_artifact_path")
        ]
        if client_artifact_rows:
            lines.extend(["", "### Client Artifact Paths", ""])
            for row in client_artifact_rows:
                lines.append(
                    f"- {_md(row.get('target'))}: " f"`{row.get('client_concrete_artifact_path')}`"
                )


def _append_gold_results(panel: dict[str, Any], lines: list[str]) -> None:
    if any(row.get("gold_case_id") for row in panel["rows"]):
        gold_l0a_row = next(
            (row for row in panel["rows"] if row.get("gold_l0a_overall") is not None),
            None,
        )
        if gold_l0a_row is not None:
            lines.extend(
                [
                    "",
                    "## Gold L0A Quality",
                    "",
                    f"- overall: `{gold_l0a_row.get('gold_l0a_overall')}`",
                    (
                        "- primary: "
                        "evidence_coverage="
                        f"`{gold_l0a_row.get('gold_l0a_primary_evidence_coverage')}`, "
                        "selected_accuracy="
                        f"`{gold_l0a_row.get('gold_l0a_selected_primary_accuracy')}`"
                    ),
                    (
                        "- fallback root fingerprint accuracy: "
                        f"`{gold_l0a_row.get('gold_l0a_root_fingerprint_accuracy')}`"
                    ),
                    (
                        "- markers: "
                        f"progress_recall=`{gold_l0a_row.get('gold_l0a_progress_line_recall')}`, "
                        "checkpoint_recall="
                        f"`{gold_l0a_row.get('gold_l0a_checkpoint_line_recall')}`"
                    ),
                    (
                        "- setup marker types: "
                        "required="
                        f"`{gold_l0a_row.get('gold_l0a_required_setup_marker_types')}`, "
                        "observed="
                        f"`{gold_l0a_row.get('gold_l0a_observed_setup_marker_types')}`"
                    ),
                    f"- coverage: `{gold_l0a_row.get('gold_l0a_coverage_checks')}`",
                ]
            )
        gold_l0b_row = next(
            (row for row in panel["rows"] if row.get("gold_l0b_overall") is not None),
            None,
        )
        if gold_l0b_row is not None:
            lines.extend(
                [
                    "",
                    "## Gold L0B Quality",
                    "",
                    f"- overall: `{gold_l0b_row.get('gold_l0b_overall')}`",
                    (
                        "- evidence: "
                        "required_line_recall="
                        f"`{gold_l0b_row.get('gold_l0b_required_evidence_line_recall')}`, "
                        "primary_retained="
                        f"`{gold_l0b_row.get('gold_l0b_primary_retained_from_l0a')}`"
                    ),
                    (
                        "- projection_integrity: "
                        f"`{gold_l0b_row.get('gold_l0b_projection_integrity_ok')}`"
                    ),
                ]
            )
        lines.extend(["", "## Gold L1 Raw Semantics", ""])
        lines.extend(
            [
                "| target | root cause | recovery | related failures | core semantics | unsupported | all L1 checks | confidence |",
                "|---|---|---|---|---|---|---|---:|",
            ]
        )
        for row in panel["rows"]:
            lines.append(
                "| "
                f"{_md(row.get('target'))} | "
                f"{_yes_no(row.get('gold_l1_root_cause_correct'))} | "
                f"{_yes_no(row.get('gold_l1_recovery_correct'))} | "
                f"{_yes_no(row.get('gold_l1_related_failure_recall'))} | "
                f"{_yes_no(row.get('gold_l1_core_semantic'))} | "
                f"{_md(','.join(row.get('gold_l1_unsupported_claims') or []))} | "
                f"{_yes_no(row.get('gold_l1_overall'))} | "
                f"{_md(row.get('gold_l1_confidence'))} |"
            )

        lines.extend(["", "## Gold L2 Identity And L4 Product", ""])
        lines.extend(
            [
                "| target | L2 audit | L2 history identity | reference-audit effect | L4 root | L4 policy | L4 action | L4 overall |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for row in panel["rows"]:
            lines.append(
                "| "
                f"{_md(row.get('target'))} | "
                f"{_yes_no(row.get('gold_l2_audit_correct'))} | "
                f"{_yes_no(row.get('gold_l2_history_identity_correct'))} | "
                f"{_md(row.get('gold_l2_reference_audit_effect'))} | "
                f"{_yes_no(row.get('gold_l4_root_cause_correct'))} | "
                f"{_yes_no(row.get('gold_l4_policy_correct'))} | "
                f"{_yes_no(row.get('gold_l4_action_correct'))} | "
                f"{_yes_no(row.get('gold_l4_overall'))} |"
            )


def _append_latency_and_cost(panel: dict[str, Any], lines: list[str]) -> None:
    lines.extend(["", "## Internal Stage Latency And Cost", ""])
    post_end_measured = any(
        row.get("post_progressive_end_wall_clock_s") is not None for row in panel["rows"]
    )
    latency_header = "| target | mode |"
    latency_separator = "|---|---|"
    if post_end_measured:
        latency_header += " post-end |"
        latency_separator += "---:|"
    latency_header += " L1 | L2 | L3 | L4 | tokens |"
    latency_separator += "---:|---:|---:|---:|---:|"
    lines.extend([latency_header, latency_separator])
    for row in panel["rows"]:
        latency_row = "| " f"{_md(row.get('target'))} | " f"{_md(row.get('latency_mode'))} | "
        if post_end_measured:
            latency_row += f"{_md(row.get('post_progressive_end_wall_clock_s'))} | "
        latency_row += (
            f"{_md(row.get('l1_kpi_wall_clock_s'))} | "
            f"{_md(row.get('l2_wall_clock_s'))} | "
            f"{_md(row.get('l3_wall_clock_s'))} | "
            f"{_md(row.get('l4_wall_clock_s'))} | "
            f"{_md(row.get('l1_kpi_total_tokens'))} |"
        )
        lines.append(latency_row)
    if post_end_measured:
        lines.append("\nPost-end is the production progressive decision-gate latency.")
    else:
        lines.append(
            "\nShared L0 build time is reported once below; L1 is the comparable "
            "per-model terminal interaction latency."
        )


def _append_shared_l0_and_execution_context(panel: dict[str, Any], lines: list[str]) -> None:
    _append_shared_l0(panel, lines)
    _append_execution_context(panel, lines)


def _append_shared_l0(panel: dict[str, Any], lines: list[str]) -> None:
    shared_l0 = _dict(panel.get("shared_l0_shape"))
    lines.extend(["", "## Shared L0A Operations", ""])
    lines.append(
        "- size: "
        f"log_lines=`{shared_l0.get('line_count')}`, "
        f"bytes=`{shared_l0.get('byte_size')}`"
    )
    lines.append(
        "- L0A assembly: "
        f"seconds=`{shared_l0.get('l0a_wall_clock_s')}`, "
        f"aggregate_l0_seconds=`{shared_l0.get('build_wall_clock_s')}` (once per panel)"
    )
    lines.append(
        "- shape: "
        f"windows=`{shared_l0.get('context_window_count')}`, "
        f"anchors=`{shared_l0.get('candidate_anchor_count')}`, "
        f"occurrence_groups=`{shared_l0.get('occurrence_group_count')}`, "
        f"episodes=`{shared_l0.get('failure_episode_count')}`, "
        f"distributed_incidents=`{shared_l0.get('distributed_failure_incident_count')}`, "
        "anchors_without_excerpt="
        f"`{shared_l0.get('candidate_anchors_without_excerpt')}`"
    )
    lines.append("- replay consistency: " f"`{shared_l0.get('consistent_across_models')}`")
    if shared_l0.get("path_access_fact_count"):
        lines.append(
            "- path access: "
            f"facts=`{shared_l0.get('path_access_fact_count')}`, "
            f"namespaces_by_role=`{shared_l0.get('path_namespaces_by_role')}`, "
            f"cross_namespace=`{shared_l0.get('cross_namespace_paths_observed')}`, "
            "failed_vs_configured_write_mismatch="
            f"`{shared_l0.get('failed_vs_configured_write_mismatch')}`, "
            f"ownership_verified=`{shared_l0.get('path_ownership_verified')}`"
        )
    lines.extend(["", "## Shared Decision Evidence", ""])
    lines.append("- selection: " f"seconds=`{shared_l0.get('decision_evidence_wall_clock_s')}`")
    lines.extend(["", "## Shared L0B Operations", ""])
    lines.append(
        "- projection: "
        f"schema=`{shared_l0.get('l0b_schema_version')}`, "
        f"seconds=`{shared_l0.get('l0b_wall_clock_s')}`, "
        f"characters=`{shared_l0.get('l0b_compact_json_characters')}`, "
        f"estimated_tokens=`{shared_l0.get('l0b_estimated_evidence_tokens')}`, "
        f"model_facing_lines=`{shared_l0.get('l0b_model_facing_context_lines')}`, "
        f"truncated_windows=`{shared_l0.get('l0b_truncated_context_windows')}`, "
        f"integrity=`{shared_l0.get('l0b_projection_integrity_status')}`"
    )
    lines.append(
        "- deterministic replay: "
        f"payload_hash=`{shared_l0.get('l0b_payload_hash')}`, "
        f"consistent_across_models=`{shared_l0.get('consistent_across_models')}`"
    )
    lines.append(f"- budget_utilization: `{shared_l0.get('l0b_budget_utilization')}`")
    lines.append(f"- selection_counts: `{shared_l0.get('l0b_selection_counts')}`")
    lines.append(f"- compaction_counts: `{shared_l0.get('l0b_compaction_counts')}`")


def _append_execution_context(panel: dict[str, Any], lines: list[str]) -> None:
    execution = _dict(panel.get("shared_l0_execution"))
    lines.extend(["", "## Current-Attempt Execution Context", ""])
    lines.append(
        "These are shared deterministic L0 facts. Later job progress does not "
        "by itself prove recovery of the same rank, path, network, or component."
    )
    lines.extend(
        [
            "",
            "### Observed Failure Position",
            "",
            "| checkpoint load | failure iteration | canonical incident line | latest observed copy | replay distance | phase |",
            "|---:|---:|---:|---:|---:|---|",
            (
                "| "
                f"{_md(execution.get('checkpoint_load_iteration'))} | "
                f"{_md(execution.get('latest_observed_failure_iteration'))} | "
                f"{_md(execution.get('first_terminal_incident_line'))} | "
                f"{_md(execution.get('latest_observed_failure_iteration_line'))} | "
                f"{_md(execution.get('observed_iterations_after_checkpoint_load'))} | "
                f"{_md(execution.get('observed_failure_phase'))} |"
            ),
            "",
            "### Completed Progress",
            "",
            "| runtime s | iteration first/last | delta | last checkpoint | replay distance | progress after terminal | later-progress observations/events |",
            "|---:|---:|---:|---:|---:|---|---:|",
            (
                "| "
                f"{_md(execution.get('successful_runtime_seconds'))} | "
                f"{_md(execution.get('first_iteration'))}/"
                f"{_md(execution.get('last_iteration'))} | "
                f"{_md(execution.get('iteration_delta'))} | "
                f"{_md(execution.get('last_checkpoint_iteration'))} | "
                f"{_md(execution.get('iterations_since_checkpoint'))} | "
                f"{_yes_no(execution.get('progress_after_failure_episode'))} | "
                f"{_md(execution.get('later_progress_after_fault_observation_count'))}/"
                f"{_md(execution.get('later_progress_after_fault_event_count'))} |"
            ),
        ]
    )
    if execution.get("first_terminal_incident_line") is not None:
        lines.append(
            "- terminal incident timing: "
            f"line=`{execution.get('first_terminal_incident_line')}`, "
            f"timestamp=`{execution.get('first_terminal_incident_timestamp')}`, "
            "seconds_since_last_progress="
            f"`{execution.get('seconds_from_last_progress_to_terminal_incident')}`, "
            f"configured_timeout_s=`{execution.get('configured_terminal_timeout_seconds')}`, "
            f"detection_lag_s=`{execution.get('terminal_detection_lag_seconds')}`"
        )
    operation_artifact_comparisons = execution.get("operation_artifact_comparisons") or []
    if operation_artifact_comparisons:
        lines.extend(["", "### Operation/Artifact Comparisons Within Current Log", ""])
        lines.extend(
            [
                "| operation | logical artifact | physical unit | comparison | successes | current outcome | failure line | observer ranks success/failure |",
                "|---|---|---|---|---:|---|---:|---|",
            ]
        )
        for item in operation_artifact_comparisons:
            lines.append(
                "| "
                f"{_md(item.get('operation'))} | "
                f"{_md(item.get('logical_artifact_id'))} | "
                f"{_md(item.get('physical_unit_id'))} | "
                f"{_md(item.get('comparison_level'))} | "
                f"{_md(item.get('success_count'))} | "
                f"{_md(item.get('current_outcome'))} | "
                f"{_md(item.get('failure_line'))} | "
                f"{_md(','.join(item.get('successful_observer_ranks') or []))}/"
                f"{_md(','.join(item.get('failed_observer_ranks') or []))} |"
            )
    distributed_incidents = execution.get("distributed_failure_incidents") or []
    if distributed_incidents:
        lines.extend(["", "### Distributed Failure Incidents", ""])
        for incident in distributed_incidents:
            lines.append(
                "- "
                f"id=`{incident.get('incident_id')}`, "
                f"kind=`{incident.get('incident_kind')}`, "
                f"type=`{incident.get('incident_type')}`, "
                f"primary_line=`{incident.get('primary_observed_line')}`, "
                f"events=`{incident.get('event_count')}`, "
                f"unique_operations=`{incident.get('unique_operation_count')}`, "
                f"observed_ranks=`{incident.get('observed_rank_count')}`, "
                f"root_cause_status=`{incident.get('root_cause_status')}`, "
                f"history_id=`{incident.get('history_fingerprint')}`"
            )
    later_progress_observations = execution.get("later_progress_after_fault_observations") or []
    if later_progress_observations:
        lines.extend(["", "### Later Progress After Fault-Like Events", ""])
        for observation in later_progress_observations:
            lines.append(
                "- "
                f"class=`{observation.get('fine_class')}`, "
                f"events=`{observation.get('event_count')}`, "
                f"event_lines=`{observation.get('sample_event_lines')}`, "
                "later_progress_lines="
                f"`{observation.get('sample_later_progress_lines')}`, "
                "component_recovery_proven="
                f"`{observation.get('component_recovery_proven')}`"
            )


def _append_l0_coverage_and_tool_profiles(panel: dict[str, Any], lines: list[str]) -> None:
    lines.extend(["", "## L0 Bundle Coverage", ""])
    lines.extend(
        [
            "| target | primary in bundle | primary in excerpt | setup markers | progress fact known | progress after fault | tool-added lines |",
            "|---|---|---|---|---|---|---:|",
        ]
    )
    for row in panel["rows"]:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(row.get('l0_selected_primary_in_bundle'))} | "
            f"{_md(row.get('l0_selected_primary_in_excerpt'))} | "
            f"{_md(','.join(row.get('l0_setup_marker_types') or []))} | "
            f"{_yes_no(row.get('l0_progress_after_fault_known'))} | "
            f"{_yes_no(row.get('l0_progress_after_fault'))} | "
            f"{_md(row.get('l0_tool_calls_added_new_prompt_lines'))} |"
        )

    if any(row.get("tool_profile_id") for row in panel["rows"]):
        lines.extend(
            [
                "",
                "## L1 Tool Profiles",
                "",
                "| target | profile | experimental | tools | max tool rounds | max model turns | source |",
                "|---|---|---|---|---:|---:|---|",
            ]
        )
        for row in panel["rows"]:
            lines.append(
                "| "
                f"{_md(row.get('target'))} | "
                f"{_md(row.get('tool_profile_id'))} | "
                f"{_yes_no(row.get('tool_profile_experimental'))} | "
                f"{_yes_no(row.get('tools_enabled'))} | "
                f"{_md(row.get('max_tool_rounds'))} | "
                f"{_md(row.get('max_model_turns'))} | "
                f"{_md(row.get('tool_profile_source'))} |"
            )


def _append_l1_call_diagnostics(panel: dict[str, Any], lines: list[str]) -> None:
    lines.extend(["", "## L1 Model Calls", ""])
    lines.extend(
        [
            "| target | response parsed | calls | turns | successful | failed | retried | timed out | HTTP errors | provider errors |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in panel["rows"]:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_yes_no(row.get('l1_kpi_response_parsed'))} | "
            f"{_md(row.get('l1_kpi_model_calls'))} | "
            f"{_md(row.get('l1_kpi_model_turns'))} | "
            f"{_md(row.get('l1_kpi_successful_model_calls'))} | "
            f"{_md(row.get('l1_kpi_failed_model_calls'))} | "
            f"{_md(row.get('l1_kpi_retried_model_calls'))} | "
            f"{_md(row.get('l1_kpi_timeout_model_calls'))} | "
            f"{_md(row.get('http_error_calls'))} | "
            f"{_md(row.get('l1_kpi_provider_error_count'))} |"
        )

    if any(row.get("l1_kpi_context_window_tokens") for row in panel["rows"]):
        lines.extend(
            [
                "",
                "### L1 Context Budget",
                "",
                "| target | context window | max estimated input | configured output | minimum effective output | adjusted calls |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in panel["rows"]:
            lines.append(
                "| "
                f"{_md(row.get('target'))} | "
                f"{_md(row.get('l1_kpi_context_window_tokens'))} | "
                f"{_md(row.get('l1_kpi_max_estimated_input_tokens'))} | "
                f"{_md(row.get('l1_kpi_configured_max_output_tokens'))} | "
                f"{_md(row.get('l1_kpi_minimum_effective_max_output_tokens'))} | "
                f"{_md(row.get('l1_kpi_context_budget_adjusted_calls'))} |"
            )

    lines.extend(["", "## L1 Tool Context", ""])
    lines.extend(
        [
            "| target | tool calls | tool-driven turns | new lines | duplicate calls | no-new calls | tool errors | truncated |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in panel["rows"]:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(row.get('l1_kpi_tool_calls'))} | "
            f"{_md(row.get('l1_kpi_tool_driven_model_turns'))} | "
            f"{_md(row.get('l1_kpi_tool_calls_added_new_prompt_lines'))} | "
            f"{_md(row.get('l1_kpi_duplicate_tool_calls'))} | "
            f"{_md(row.get('l1_kpi_no_new_prompt_line_tool_calls'))} | "
            f"{_md(row.get('l1_kpi_tool_error_calls'))} | "
            f"{_md(row.get('l1_kpi_tool_truncated_calls'))} |"
        )

    lines.extend(["", "### Final-Answer Tool Dependency", ""])
    lines.extend(
        [
            "| target | impact | primary tool-only | decision lines | structured repeats | incidental cited | unused new |",
            "|---|---|---|---|---|---|---:|",
        ]
    )
    for row in panel["rows"]:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(row.get('tool_final_context_impact') or row.get('tool_final_context_dependency'))} | "
            f"{_yes_no(row.get('tool_final_primary_from_tool_only_context'))} | "
            f"{_md(','.join(str(line) for line in row.get('tool_decision_relevant_tool_only_lines') or []))} | "
            f"{_md(','.join(str(line) for line in row.get('tool_structured_fact_redundant_lines') or []))} | "
            f"{_md(','.join(str(line) for line in row.get('tool_incidental_tool_only_lines') or []))} | "
            f"{_md(len(row.get('tool_unused_tool_only_lines') or []))} |"
        )

    lines.extend(["", "## L1 Runtime Health", ""])
    lines.extend(
        [
            "| target | execution | issues | output | usable | contract repair | client request | endpoint | token limit | L1 errors |",
            "|---|---|---|---|---|---|---|---|---|---:|",
        ]
    )
    for row in panel["rows"]:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(row.get('l1_execution_status'))} | "
            f"{_md(','.join(row.get('l1_execution_issues') or []))} | "
            f"{_md(row.get('l1_output_status'))} | "
            f"{_yes_no(row.get('l1_output_usable'))} | "
            f"{_yes_no(row.get('l1_contract_repair_requested'))} | "
            f"{_md(row.get('l1_kpi_client_request_health'))} | "
            f"{_md(row.get('l1_kpi_endpoint_reliability'))} | "
            f"{_yes_no(row.get('l1_kpi_token_limit_hit'))} | "
            f"{_md(row.get('l1_kpi_errors_count'))} |"
        )


def _append_l2_diagnostics(panel: dict[str, Any], lines: list[str]) -> None:
    lines.extend(["", "## L2 Grounding, Identity, And Audit", ""])
    lines.extend(
        [
            "| target | grounding | method | audit | ran | primary | recovery | related | domain/retry/unresolved support | citations raw/rendered/abbrev/nearby/U | findings all/material | severity |",
            "|---|---|---|---|---|---|---|---:|---|---|---|---|",
        ]
    )
    for row in panel["rows"]:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(row.get('l2_grounding_status'))} | "
            f"{_md(row.get('l2_grounding_method'))} | "
            f"{_md(row.get('l2_audit_status'))} | "
            f"{_yes_no(row.get('l2_audit_ran'))} | "
            f"{_yes_no(row.get('l2_primary_available'))} | "
            f"{_yes_no(row.get('l2_recovery_assessment_available'))} | "
            f"{_md(row.get('l2_related_failures_audited'))} | "
            f"{_md(len(row.get('l2_failure_domain_supporting_lines') or []))}/"
            f"{_md(len(row.get('l2_retry_outlook_supporting_lines') or []))}/"
            f"{_md(len(row.get('l2_unresolved_recovery_supporting_lines') or []))} | "
            f"{_md(row.get('l2_exact_citation_count'))}/"
            f"{_md(row.get('l2_rendered_exact_citation_count'))}/"
            f"{_md(row.get('l2_abbreviated_exact_citation_count'))}/"
            f"{_md(row.get('l2_nearby_resolved_count'))}/"
            f"{_md(row.get('l2_ungrounded_citation_count'))} | "
            f"{_md(row.get('l2_finding_count'))}/"
            f"{_md(row.get('l2_material_finding_count'))} | "
            f"{_md(row.get('l2_finding_severity_counts'))} |"
        )

    lines.extend(["", "## L2 Audit Outcomes", ""])
    lines.extend(
        [
            "| target | reference repairs | recovery observations | unresolved recovery lines | fingerprint source |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in panel["rows"]:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(row.get('l2_grounding_adjustment_count'))} | "
            f"{_md(row.get('l2_recovery_audit_observation_count'))} | "
            f"{_md(len(row.get('l2_unresolved_recovery_supporting_lines') or []))} | "
            f"{_md(row.get('l2_root_fingerprint_source'))} |"
        )


def _append_l1_semantic_assessment(panel: dict[str, Any], lines: list[str]) -> None:
    restart_context = panel.get("shared_restart_environment_context") or {}
    restart_context_consistency = panel.get("restart_environment_context_consistency") or {}
    lines.extend(["", "## L1 Restart Environment", ""])
    lines.append(
        "- shared context: "
        f"`{restart_context}`; consistency=`{restart_context_consistency.get('status')}`"
    )
    lines.append(
        "- interpretation: the workload remains unchanged, while normal restart may "
        "recreate process state and change hardware allocation or mutable service state."
    )

    lines.extend(["", "## L1 Cause Assessment", ""])
    lines.extend(
        [
            "| target | root status | domain | domain status | domain confidence |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in panel["rows"]:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(row.get('model_root_cause_status'))} | "
            f"{_md(row.get('model_failure_domain'))} | "
            f"{_md(row.get('model_failure_domain_status'))} | "
            f"{_md(row.get('model_failure_domain_confidence'))} |"
        )
    missing_evidence_rows = [
        row for row in panel["rows"] if row.get("model_root_cause_missing_evidence")
    ]
    if missing_evidence_rows:
        lines.extend(["", "### L1 Missing Evidence", ""])
        for row in missing_evidence_rows:
            lines.append(
                f"- {_md(row.get('target'))}: " f"`{row.get('model_root_cause_missing_evidence')}`"
            )

    lines.extend(["", "## L1 Recovery Assessment", ""])
    lines.extend(
        [
            "| target | retry outlook unchanged | retry status | retry confidence |",
            "|---|---|---|---:|",
        ]
    )
    for row in panel["rows"]:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(row.get('model_retry_outlook_without_workload_change'))} | "
            f"{_md(row.get('model_retry_outlook_status'))} | "
            f"{_md(row.get('model_retry_outlook_confidence'))} |"
        )


def _append_l3_history(panel: dict[str, Any], lines: list[str]) -> None:
    lines.extend(["", "## L3 History", ""])
    lines.extend(
        [
            "### Current-Failure Input",
            "",
            "| target | source | history identity ready | selected root |",
            "|---|---|---|---|",
        ]
    )
    for row in panel["rows"]:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(row.get('current_failure_facts_source'))} | "
            f"{_yes_no(row.get('current_history_identity_ready'))} | "
            f"{_md(_short_identity(row.get('current_root_fingerprint')))} |"
        )
    lines.extend(["", "### Prior-Attempt Comparison", ""])
    lines.extend(
        [
            "| target | available | same job/root | advanced | no observed advance | unknown | same failure/data position | streak |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in panel["rows"]:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_yes_no(row.get('l3_history_available'))} | "
            f"{_md(row.get('l3_same_job_attempts'))}/"
            f"{_md(row.get('l3_matching_root_attempts'))} | "
            f"{_md(row.get('l3_observed_advance_attempts'))} | "
            f"{_md(row.get('l3_no_observed_advance_attempts'))} | "
            f"{_md(row.get('l3_unknown_progress_attempts'))} | "
            f"{_md(row.get('l3_exact_failure_position_attempts'))}/"
            f"{_md(row.get('l3_same_data_position_attempts'))} | "
            f"{_md(row.get('l3_consecutive_same_root_no_advance_attempts'))} |"
        )


def _append_l4_policy(panel: dict[str, Any], lines: list[str]) -> None:
    lines.extend(["", "## L4 Policy Decision", ""])
    lines.extend(
        [
            "| target | decision | basis | retry rule | prior/allowed | exhausted | current evidence qualified | quality | NVRx use |",
            "|---|---|---|---|---:|---|---|---|---|",
        ]
    )
    for row in panel["rows"]:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(row.get('decision'))} | "
            f"{_md(row.get('decision_basis'))} | "
            f"{_md(row.get('l4_retry_rule'))} | "
            f"{_md(row.get('l4_matching_prior_failures'))}/"
            f"{_md(row.get('l4_allowed_retries'))} | "
            f"{_yes_no(row.get('l4_retry_budget_exhausted'))} | "
            f"{_yes_no(row.get('l4_current_evidence_qualified'))} | "
            f"{_md(row.get('l4_result_quality'))} | "
            f"{_md(row.get('l4_nvrx_use'))} |"
        )

    lines.extend(["", "### L4 Policy Inputs", ""])
    lines.extend(
        [
            "| target | domain | domain status | retry outlook unchanged | retry status | L2 policy grounded | observed advance |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in panel["rows"]:
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{_md(row.get('l4_failure_domain'))} | "
            f"{_md(row.get('l4_failure_domain_status'))} | "
            f"{_md(row.get('l4_retry_outlook_without_workload_change'))} | "
            f"{_md(row.get('l4_retry_outlook_status'))} | "
            f"{_yes_no(row.get('l4_recovery_assessment_policy_grounded'))} | "
            f"{_yes_no(row.get('l4_observed_advance'))} |"
        )

    lines.extend(["", "### Final Downstream Roles", ""])
    lines.extend(
        [
            "| target | cascades | teardown | downstream events |",
            "|---|---:|---:|---|",
        ]
    )
    for row in panel["rows"]:
        downstream = row.get("l4_downstream_roles") or []
        cascade_count = sum(
            int(item.get("count") or 0)
            for item in downstream
            if item.get("causal_role") == "cascade"
        )
        teardown_count = sum(
            int(item.get("count") or 0)
            for item in downstream
            if item.get("causal_role") == "teardown"
        )
        event_summary = ", ".join(
            f"{item.get('causal_role')}@{item.get('first_line')}"
            + (
                f"-{item.get('last_line')}"
                if item.get("last_line") != item.get("first_line")
                else ""
            )
            + f" x{item.get('count')}"
            for item in downstream
        )
        lines.append(
            "| "
            f"{_md(row.get('target'))} | "
            f"{cascade_count} | "
            f"{teardown_count} | "
            f"{_md(event_summary or 'none')} |"
        )


def _append_concerns_and_notes(panel: dict[str, Any], lines: list[str]) -> None:
    concerns = panel.get("concerns") or []
    lines.extend(["", "## Concerns", ""])
    if concerns:
        for concern in concerns:
            lines.append(
                f"- `{concern.get('target')}` `{concern.get('category')}`: "
                f"{concern.get('summary')}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Notes", ""])
    for note in panel.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
