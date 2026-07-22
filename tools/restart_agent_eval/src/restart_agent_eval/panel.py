#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize one-log restart-policy review artifacts across a model panel."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .artifact_io import read_json as _read_json
from .artifact_io import write_json as _write_json
from .panel_diagnostics_markdown import _panel_diagnostics_markdown
from .panel_summary_markdown import _panel_markdown
from .schemas import PANEL_SUMMARY_SCHEMA_VERSION, REVIEW_SUMMARY_SCHEMA_VERSION, require_schema

DEFAULT_TARGET_ORDER = MappingProxyType(
    {
        "deterministic": 0,
        "qwen235b": 10,
        "qwen397b": 15,
        "nemotron": 20,
        "gpt": 30,
        "claude": 40,
        "gemini": 50,
        "qwen": 90,
    }
)


@dataclass(frozen=True)
class PanelInput:
    """Loaded run metadata and normalized per-route summaries."""

    run_dir: Path
    index: dict[str, Any]
    run_manifest: dict[str, Any]
    summaries: tuple[dict[str, Any], ...]
    restart_agent_config: dict[str, Any]

    @classmethod
    def read(cls, run_dir: Path, summaries: list[dict[str, Any]]) -> "PanelInput":
        index = _read_json(run_dir / "review_index.json")
        index = index if isinstance(index, dict) else {}
        run_manifest = index.get("run_manifest")
        return cls(
            run_dir=run_dir,
            index=index,
            run_manifest=run_manifest if isinstance(run_manifest, dict) else {},
            summaries=tuple(summaries),
            restart_agent_config=_restart_agent_config(run_dir),
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = args.run_dir.expanduser()
    if not run_dir.is_dir():
        raise SystemExit(f"run directory does not exist: {run_dir}")

    summaries = _load_summaries(run_dir)
    if not summaries:
        raise SystemExit(f"no *.review.json files found in: {run_dir}")

    json_path, md_path = write_panel_summary(
        run_dir,
        summaries,
        json_path=args.json_out,
        md_path=args.md_out,
    )

    if not args.quiet:
        print(f"panel summary json: {json_path}")
        print(f"panel summary md:   {md_path}")
    return 0


def write_panel_summary(
    run_dir: Path,
    summaries: list[dict[str, Any]],
    *,
    json_path: Path | None = None,
    md_path: Path | None = None,
) -> tuple[Path, Path]:
    panel = _build_panel_summary(run_dir, summaries)
    json_path = json_path or run_dir / "panel_summary.json"
    md_path = md_path or run_dir / "panel_summary.md"
    diagnostics_path = run_dir / "panel_diagnostics.md"
    panel["artifact_paths"] = {
        "summary_markdown": str(md_path),
        "diagnostics_markdown": str(diagnostics_path),
        "summary_json": str(json_path),
        "run_manifest": str(run_dir / "run_manifest.json"),
        "review_index": str(run_dir / "review_index.json"),
    }
    live_dir = run_dir / "live"
    if live_dir.is_dir():
        panel["artifact_paths"].update(
            {
                "live_status": str(live_dir / "run_status.json"),
                "live_events": str(live_dir / "events.jsonl"),
            }
        )
    _write_json(json_path, panel)
    md_path.write_text(_panel_markdown(panel), encoding="utf-8")
    diagnostics_path.write_text(_panel_diagnostics_markdown(panel), encoding="utf-8")
    return json_path, md_path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read restart-agent one-log review artifacts from a run directory "
            "and emit a compact cross-model comparison."
        )
    )
    parser.add_argument("run_dir", type=Path, help="one-log run directory")
    parser.add_argument("--json-out", type=Path, help="output JSON path")
    parser.add_argument("--md-out", type=Path, help="output Markdown path")
    parser.add_argument("--quiet", action="store_true", help="do not print output paths")
    return parser.parse_args(argv)


def _load_summaries(run_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    index = _read_json(run_dir / "review_index.json")
    runs = index.get("runs") if isinstance(index, dict) else None
    if isinstance(runs, list):
        for item in runs:
            if not isinstance(item, dict):
                continue
            require_schema(item, REVIEW_SUMMARY_SCHEMA_VERSION, artifact="review summary")
            summaries.append(item)

    seen_labels = {
        str(summary.get("run_label") or summary.get("target") or "") for summary in summaries
    }
    for path in sorted(run_dir.glob("*.review.json")):
        summary = _read_json(path)
        if not isinstance(summary, dict):
            continue
        require_schema(summary, REVIEW_SUMMARY_SCHEMA_VERSION, artifact=str(path))
        label = str(summary.get("run_label") or summary.get("target") or path.name)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        summaries.append(summary)

    return sorted(summaries, key=_summary_sort_key)


def _summary_sort_key(summary: dict[str, Any]) -> tuple[int, str]:
    target = str(summary.get("target") or "")
    return (
        DEFAULT_TARGET_ORDER.get(target, 100),
        str(summary.get("run_label") or target),
    )


def _build_panel_summary(run_dir: Path, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return _build_panel_payload(PanelInput.read(run_dir, summaries))


def _build_panel_payload(context: PanelInput) -> dict[str, Any]:
    run_dir = context.run_dir
    index = context.index
    run_manifest = context.run_manifest
    summaries = list(context.summaries)
    source_log_path = index.get("source_log_path")
    product_repo = index.get("product_repo")
    shared_l0_bundle = index.get("shared_l0_bundle")
    gold_label_path = index.get("gold_label_path")
    restart_agent_config = context.restart_agent_config
    (
        shared_decision_evidence,
        decision_evidence_consistency,
        decision_evidence_by_target,
    ) = _shared_decision_evidence(summaries)
    rows = [_row(summary) for summary in summaries]
    restart_context = _restart_context_agreement(rows)
    comparison_axes = _comparison_axes(rows)
    decision_counts = Counter(str(row.get("decision")) for row in rows)
    primary_class_counts = Counter(str(row.get("primary_class")) for row in rows)
    primary_line_counts = Counter(str(row.get("primary_line")) for row in rows)
    model_rows = [row for row in rows if row.get("model")]
    client_agreement = _fingerprint_agreement(
        rows,
        fingerprint_field="client_concrete_fingerprint",
        primary_line_field="primary_line",
        require_multiple=False,
    )
    root_agreement = _fingerprint_agreement(
        model_rows,
        fingerprint_field="l2_root_fingerprint",
        primary_line_field="l2_grounded_primary_line",
        require_multiple=True,
    )
    expected_identity_counts = {
        int(row["gold_l2_expected_cross_route_identity_count"])
        for row in model_rows
        if row.get("gold_l2_expected_cross_route_identity_count") is not None
    }
    expected_identity_count = (
        next(iter(expected_identity_counts)) if len(expected_identity_counts) == 1 else None
    )
    root_agreement["gold_expected_identity_count"] = expected_identity_count
    root_agreement["gold_stability_correct"] = (
        root_agreement["unique_fingerprints"] == expected_identity_count
        and root_agreement["available_models"] == len(model_rows)
        if expected_identity_count is not None
        else None
    )

    concerns = _concerns(rows)
    if decision_evidence_consistency["status"] == "inconsistent":
        concerns.append(
            {
                "target": "panel",
                "category": "l0_decision_evidence_consistency",
                "summary": (
                    f"status={decision_evidence_consistency['status']} "
                    f"available_models={decision_evidence_consistency['available_models']}/"
                    f"{decision_evidence_consistency['total_models']} "
                    f"unique_payloads={decision_evidence_consistency['unique_payloads']}"
                ),
            }
        )
    if restart_context["status"] == "inconsistent":
        concerns.append(
            {
                "target": "panel",
                "category": "restart_environment_context_consistency",
                "summary": (
                    f"status={restart_context['status']} "
                    f"available_models={restart_context['available_models']}/{len(rows)} "
                    f"unique_payloads={restart_context['unique_payloads']}"
                ),
            }
        )
    if (
        root_agreement["status"] == "unstable"
        or root_agreement["disagreement_reason"] == "missing_fingerprints"
    ):
        concerns.append(
            {
                "target": "panel",
                "category": "l2_root_fingerprint_stability",
                "summary": (
                    f"status={root_agreement['status']} "
                    f"available_models={root_agreement['available_models']}/{len(model_rows)} "
                    f"unique_fingerprints={root_agreement['unique_fingerprints']} "
                    f"reason={root_agreement['disagreement_reason']}"
                ),
            }
        )
    if root_agreement.get("gold_stability_correct") is False:
        concerns.append(
            {
                "target": "panel",
                "category": "gold_l2_history_identity_stability",
                "summary": (
                    f"expected_unique_identities={expected_identity_count} "
                    f"observed_unique_identities={root_agreement['unique_fingerprints']}"
                ),
            }
        )
    has_gold = any(row.get("gold_case_id") for row in rows)
    shared_l0_shape = _shared_l0_shape(rows)
    shared_l0_execution = _shared_l0_execution(rows)
    decision_path_comparison = _decision_path_comparison(rows)
    source_content_overlap = sorted(
        {
            token
            for row in rows
            for token in row.get("redaction_source_content_overlap_tokens") or []
        }
    )
    concerns = [_enrich_concern(concern) for concern in concerns]
    return {
        "schema_version": PANEL_SUMMARY_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "run_dir_name": run_dir.name,
        "run_manifest": run_manifest,
        "run_id": run_manifest.get("run_id"),
        "source_log_path": source_log_path,
        "source_log_name": Path(str(source_log_path)).name if source_log_path else None,
        "source_log_relative_path": index.get("source_log_relative_path"),
        "source_log_sha256": index.get("source_log_sha256"),
        "source_log_byte_size": index.get("source_log_byte_size"),
        "product_repo": product_repo,
        "product_commit": index.get("product_commit"),
        "harness_repo": index.get("harness_repo"),
        "harness_commit": index.get("harness_commit"),
        "shared_l0_bundle": shared_l0_bundle,
        "shared_l0_shape": shared_l0_shape,
        "shared_l0_execution": shared_l0_execution,
        "shared_decision_evidence": shared_decision_evidence,
        "decision_evidence_consistency": decision_evidence_consistency,
        "decision_evidence_by_target": decision_evidence_by_target,
        "shared_restart_environment_context": restart_context["shared_context"],
        "restart_environment_context_consistency": {
            "status": restart_context["status"],
            "available_models": restart_context["available_models"],
            "total_models": len(rows),
            "unique_payloads": restart_context["unique_payloads"],
            "by_target": restart_context["by_target"],
        },
        "gold_label_path": gold_label_path,
        "restart_agent_config": restart_agent_config,
        "model_count": len(rows),
        "decision_counts": dict(sorted(decision_counts.items())),
        "primary_class_counts": dict(sorted(primary_class_counts.items())),
        "primary_line_counts": dict(sorted(primary_line_counts.items())),
        "rows": rows,
        "comparison_axes": comparison_axes,
        "decision_path_comparison": decision_path_comparison,
        "l2_root_fingerprint_agreement": root_agreement,
        "client_concrete_agreement": client_agreement,
        "concerns": concerns,
        "notes": [
            (
                "Semantic fields are scored against the attached human-approved gold label."
                if has_gold
                else "This is a comparison aid, not a ground-truth semantic score."
            ),
            (
                "Semantic quality, behavioral efficiency, and endpoint reliability are "
                "independent axes; route outcome reports their combined operational result."
            ),
            (
                "A model with endpoint issues may still be semantically useful; a model "
                "with clean endpoint behavior may still be wrong."
            ),
            (
                "Redaction candidates already present in source-log content are not path leaks: "
                f"{source_content_overlap}."
                if source_content_overlap
                else "Redaction checks found no source-content/path-token overlap."
            ),
        ],
    }


def _restart_context_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_target = {
        str(row.get("target")): row.get("restart_environment_context")
        for row in rows
        if row.get("restart_environment_context")
    }
    payloads = {
        json.dumps(context, sort_keys=True, separators=(",", ":")) for context in by_target.values()
    }
    if not by_target:
        status = "not_available"
    elif len(payloads) != 1:
        status = "inconsistent"
    elif len(by_target) != len(rows):
        status = "consistent_among_available"
    else:
        status = "consistent"
    return {
        "status": status,
        "available_models": len(by_target),
        "unique_payloads": len(payloads),
        "by_target": by_target,
        "shared_context": (
            next(iter(by_target.values()), {})
            if status in {"consistent", "consistent_among_available"}
            else {}
        ),
    }


def _fingerprint_agreement(
    rows: list[dict[str, Any]],
    *,
    fingerprint_field: str,
    primary_line_field: str,
    require_multiple: bool,
) -> dict[str, Any]:
    fingerprints = {str(row[fingerprint_field]) for row in rows if row.get(fingerprint_field)}
    by_primary: dict[str, set[str]] = {}
    for row in rows:
        line = row.get(primary_line_field)
        fingerprint = row.get(fingerprint_field)
        if line is not None and fingerprint:
            by_primary.setdefault(str(line), set()).add(str(fingerprint))
    same_primary_consistent = all(len(values) == 1 for values in by_primary.values())
    if len(fingerprints) <= 1:
        reason = "none"
    elif len(by_primary) > 1 and same_primary_consistent:
        reason = "primary_selection_disagreement"
    else:
        reason = "same_primary_identity_disagreement"
    available = sum(1 for row in rows if row.get(fingerprint_field))
    common = {
        "available_models": available,
        "unique_fingerprints": len(fingerprints),
        "all_available_agree": bool(fingerprints) and len(fingerprints) == 1,
        "same_primary_consistent": same_primary_consistent,
        "primary_line_groups": len(by_primary),
        "disagreement_reason": reason,
    }
    if not require_multiple:
        return common
    if len(fingerprints) > 1:
        status = "unstable"
    elif len(rows) < 2:
        status = "not_checkable"
        reason = "insufficient_models" if rows else "no_model_routes"
    elif available != len(rows):
        status = "not_checkable"
        reason = "missing_fingerprints"
    else:
        status = "stable"
    return {
        "status": status,
        "total_models": len(rows),
        **common,
        "disagreement_reason": reason,
    }


def _decision_path_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available_fallbacks = [
        {
            "decision": row.get("fallback_decision"),
            "retry_rule": row.get("fallback_retry_rule"),
            "gold_action_correct": row.get("gold_fallback_action_correct"),
            "gold_policy_action": row.get("gold_fallback_policy_action"),
        }
        for row in rows
        if row.get("fallback_available")
    ]
    serialized_fallbacks = {
        json.dumps(item, sort_keys=True, separators=(",", ":")) for item in available_fallbacks
    }
    if not available_fallbacks:
        fallback_status = "unavailable"
        shared_fallback: dict[str, Any] = {}
    elif len(serialized_fallbacks) == 1:
        fallback_status = "consistent"
        shared_fallback = available_fallbacks[0]
    else:
        fallback_status = "inconsistent"
        shared_fallback = {}

    model_routes = [
        {
            "target": row.get("target"),
            "enriched_available": row.get("enriched_available"),
            "enriched_decision": row.get("enriched_decision"),
            "enriched_retry_rule": row.get("enriched_retry_rule"),
            "gold_enriched_action_correct": row.get("gold_enriched_action_correct"),
            "gold_enriched_policy_action": row.get("gold_enriched_policy_action"),
            "action_effect": row.get("l1_action_effect"),
            "policy_action_effect": row.get("l1_policy_action_effect"),
        }
        for row in rows
        if row.get("model")
    ]
    action_effect_counts = Counter(
        str(route["action_effect"])
        for route in model_routes
        if route.get("action_effect") not in {None, "not_available", "unscored"}
    )
    policy_effect_counts = Counter(
        str(route["policy_action_effect"])
        for route in model_routes
        if route.get("policy_action_effect") not in {None, "not_available", "unscored"}
    )
    return {
        "fallback_consistency": fallback_status,
        "shared_fallback": shared_fallback,
        "model_routes": model_routes,
        "action_effect_counts": dict(sorted(action_effect_counts.items())),
        "policy_action_effect_counts": dict(sorted(policy_effect_counts.items())),
    }


def _enrich_concern(concern: dict[str, Any]) -> dict[str, Any]:
    category = str(concern.get("category") or "unknown")
    if category.startswith("l0_"):
        owner = "L0 bundle"
    elif category.startswith("l2_"):
        owner = "L2 audit/identity"
    elif category == "gold_final_cascade":
        owner = "analyzer output"
    elif category.startswith("l4_") or category.startswith("gold_l4_"):
        owner = "L4 policy"
    elif category in {"endpoint_reliability"}:
        owner = "endpoint"
    elif category in {"redaction"}:
        owner = "harness"
    else:
        owner = "model route"

    if category in {
        "redaction",
        "l0_decision_evidence_consistency",
        "gold_l4_action",
    }:
        severity = "high"
    elif category in {
        "endpoint_reliability",
        "l1_unavailable",
        "client_context_budget",
        "token_limit",
        "l0_bundle_gap",
        "l2_root_fingerprint_stability",
    } or category.startswith("gold_"):
        severity = "medium"
    else:
        severity = "advisory"

    impact = {
        "endpoint_reliability": "route availability or latency",
        "l1_unavailable": "model-enriched result unavailable",
        "l0_bundle_gap": "initial evidence may be incomplete",
        "l0_bundle_coverage": "initial evidence retention",
        "l2_root_fingerprint_stability": "history identity stability",
        "redaction": "source-path isolation",
        "tool_efficiency": "route latency and token cost",
    }.get(category, "review required")
    return {**concern, "severity": severity, "owner": owner, "impact": impact}


def _restart_agent_config(run_dir: Path) -> dict[str, Any] | None:
    batch_result = _read_json(run_dir / "restart_agent.result.json")
    if not isinstance(batch_result, dict):
        return None
    shared_analysis = batch_result.get("shared_analysis")
    if not isinstance(shared_analysis, dict):
        return None
    config = shared_analysis.get("restart_agent_config")
    return config if isinstance(config, dict) else None


def _comparison_axes(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    semantic_quality = []
    behavioral_efficiency = []
    endpoint_reliability = []
    route_outcome = []
    for row in rows:
        semantic_status = _semantic_quality_status(row)
        first_turn_usable = _first_turn_usable(row)
        fallback_only = not row.get("l1_output_usable")
        model_contribution = "fallback_only" if fallback_only else "model_enriched"
        route_latency_s = _coalesce(
            row.get("post_progressive_end_wall_clock_s"),
            row.get("l1_kpi_wall_clock_s"),
            row.get("l1_wall_clock_s"),
        )
        route_latency_basis = (
            "post_progressive_end"
            if row.get("post_progressive_end_wall_clock_s") is not None
            else "terminal_l1_interaction"
        )

        semantic_quality.append(
            {
                "target": row.get("target"),
                "status": semantic_status,
                "primary": _semantic_primary_label(row),
                "root_cause_correct": row.get("gold_l1_root_cause_correct"),
                "recovery_correct": row.get("gold_l1_recovery_correct"),
                "related_failure_recall": row.get("gold_l1_related_failure_recall"),
                "final_cascade_correct": row.get("gold_l4_cascade_correct"),
                "unsupported_claims": row.get("gold_l1_unsupported_claims") or [],
                "confidence": _coalesce(
                    row.get("gold_l1_confidence"),
                    row.get("model_recovery_confidence"),
                ),
            }
        )
        behavioral_efficiency.append(
            {
                "target": row.get("target"),
                "first_turn_usable": first_turn_usable,
                "model_turns": row.get("l1_kpi_model_turns"),
                "tool_driven_turns": row.get("l1_kpi_tool_driven_model_turns"),
                "contract_repair_turns": row.get("l1_contract_repair_turns"),
                "tool_calls": _coalesce(
                    row.get("l1_kpi_tool_calls"),
                    row.get("tool_calls"),
                ),
                "duplicate_tool_calls": row.get("l1_kpi_duplicate_tool_calls"),
                "no_new_context_tool_calls": row.get("l1_kpi_no_new_prompt_line_tool_calls"),
                "total_tokens": _coalesce(
                    row.get("l1_kpi_total_tokens"),
                    row.get("total_tokens"),
                ),
            }
        )
        endpoint_reliability.append(
            {
                "target": row.get("target"),
                "status": _coalesce(
                    row.get("l1_kpi_endpoint_reliability"),
                    row.get("endpoint_reliability"),
                ),
                "attempts": _coalesce(
                    row.get("l1_kpi_model_calls"),
                    row.get("model_calls"),
                ),
                "successful_attempts": row.get("l1_kpi_successful_model_calls"),
                "failed_attempts": _coalesce(
                    row.get("l1_kpi_failed_model_calls"),
                    row.get("failed_model_calls"),
                ),
                "retried_attempts": _coalesce(
                    row.get("l1_kpi_retried_model_calls"),
                    row.get("retried_model_calls"),
                ),
                "timeouts": _coalesce(
                    row.get("l1_kpi_timeout_model_calls"),
                    row.get("timeout_model_calls"),
                ),
                "http_errors": row.get("http_error_calls"),
                "provider_errors": _coalesce(
                    row.get("l1_kpi_provider_error_count"),
                    row.get("provider_error_count"),
                ),
            }
        )
        route_outcome.append(
            {
                "target": row.get("target"),
                "model_contribution": model_contribution,
                "l1_execution_status": row.get("l1_execution_status"),
                "semantic_quality": semantic_status,
                "endpoint_reliability": _coalesce(
                    row.get("l1_kpi_endpoint_reliability"),
                    row.get("endpoint_reliability"),
                ),
                "decision": row.get("decision"),
                "result_quality": row.get("l4_result_quality"),
                "nvrx_use": (
                    "eligible_fallback"
                    if fallback_only and row.get("l4_nvrx_use") == "eligible_degraded"
                    else row.get("l4_nvrx_use")
                ),
                "reason": "no_model_enrichment" if fallback_only else "model_enriched",
                "latency_s": route_latency_s,
                "latency_basis": route_latency_basis,
            }
        )
    return {
        "semantic_quality": semantic_quality,
        "behavioral_efficiency": behavioral_efficiency,
        "endpoint_reliability": endpoint_reliability,
        "route_outcome": route_outcome,
    }


def _semantic_quality_status(row: dict[str, Any]) -> str:
    if not row.get("gold_case_id"):
        return "unscored"
    if not _model_response_delivered(row):
        return "not_observed"
    core_semantic = row.get("gold_l1_core_semantic")
    if core_semantic is None:
        core_semantic = row.get("gold_l1_overall")
    return "pass" if core_semantic is True else "fail"


def _first_turn_usable(row: dict[str, Any]) -> bool | None:
    if not _model_response_delivered(row):
        return None
    return bool(
        row.get("l1_output_usable")
        and _int_or_zero(row.get("l1_kpi_model_turns")) == 1
        and _int_or_zero(row.get("l1_kpi_tool_driven_model_turns")) == 0
        and _int_or_zero(row.get("l1_contract_repair_turns")) == 0
    )


def _model_response_delivered(row: dict[str, Any]) -> bool:
    successful_calls = row.get("l1_kpi_successful_model_calls")
    if successful_calls is not None:
        return _int_or_zero(successful_calls) > 0
    return bool(row.get("l1_kpi_response_parsed") or row.get("l1_output_usable"))


def _coalesce(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _semantic_primary_label(row: dict[str, Any]) -> str:
    fine_class = row.get("l1_semantic_primary_class") or row.get("primary_class")
    line = row.get("l1_semantic_primary_line") or row.get("primary_line")
    if fine_class is None and line is None:
        return "not_available"
    return f"{fine_class}@{line}"


def _shared_decision_evidence(
    summaries: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    evidence_by_target: dict[str, dict[str, Any]] = {}
    serialized_payloads: set[str] = set()
    for summary in summaries:
        evidence = _dict(summary.get("decision_evidence"))
        if not evidence:
            continue
        target = str(summary.get("target") or summary.get("run_label") or "unknown")
        evidence_by_target[target] = evidence
        serialized_payloads.add(json.dumps(evidence, sort_keys=True, separators=(",", ":")))

    total_models = len(summaries)
    available_models = len(evidence_by_target)
    unique_payloads = len(serialized_payloads)
    if available_models == 0:
        status = "unavailable"
    elif total_models == 1:
        status = "not_checkable"
    elif unique_payloads > 1:
        status = "inconsistent"
    elif available_models != total_models:
        status = "consistent_among_available"
    else:
        status = "consistent"

    shared = next(iter(evidence_by_target.values()), {}) if unique_payloads == 1 else {}
    divergent = evidence_by_target if status == "inconsistent" else {}
    return (
        shared,
        {
            "status": status,
            "total_models": total_models,
            "available_models": available_models,
            "unique_payloads": unique_payloads,
            "shared_payload_available": bool(shared),
        },
        divergent,
    )


def _shared_l0_shape(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        "line_count": "l0_line_count",
        "byte_size": "l0_byte_size",
        "context_window_count": "l0_context_window_count",
        "candidate_anchor_count": "l0_candidate_anchor_count",
        "occurrence_group_count": "l0_occurrence_group_count",
        "failure_episode_count": "l0_failure_episode_count",
        "distributed_failure_incident_count": ("l0_distributed_failure_incident_count"),
        "candidate_anchors_without_excerpt": "l0_candidate_anchors_without_excerpt",
        "path_access_fact_count": "l0_path_access_fact_count",
        "path_namespaces_by_role": "l0_path_namespaces_by_role",
        "cross_namespace_paths_observed": "l0_cross_namespace_paths_observed",
        "failed_vs_configured_write_mismatch": ("l0_failed_vs_configured_write_mismatch"),
        "path_ownership_verified": "l0_path_ownership_verified",
    }
    populated = [row for row in rows if row.get("l0_line_count") is not None]
    if not populated:
        return {name: None for name in fields}
    first = populated[0]
    result = {name: first.get(row_field) for name, row_field in fields.items()}
    first_projection = _dict(first.get("l0b_projection_metrics"))
    first_view_size = _dict(first_projection.get("view_size"))
    first_compaction = _dict(first_projection.get("compaction_counts"))
    first_integrity = _dict(first_projection.get("projection_integrity"))
    result.update(
        {
            "l0b_schema_version": first.get("l0b_schema_version"),
            "l0b_compact_json_characters": first_view_size.get("compact_json_characters"),
            "l0b_estimated_evidence_tokens": first_view_size.get("estimated_tokens"),
            "l0b_model_facing_context_lines": first_compaction.get("model_facing_context_lines"),
            "l0b_truncated_context_windows": first_compaction.get("truncated_context_windows"),
            "l0b_budget_utilization": first_projection.get("budget_utilization") or {},
            "l0b_selection_counts": first_projection.get("selection_counts") or {},
            "l0b_compaction_counts": first_compaction,
            "l0b_projection_integrity_status": first_integrity.get("status"),
            "l0b_payload_hash": first_integrity.get("deterministic_payload_sha256"),
        }
    )
    build_times = [
        float(row["l0_wall_clock_s"]) for row in populated if row.get("l0_wall_clock_s") is not None
    ]
    result["build_wall_clock_s"] = max(build_times, default=None)
    result["replayed"] = all(bool(row.get("l0_reused")) for row in populated)
    l0a_times = [
        float(row["l0a_wall_clock_s"])
        for row in populated
        if row.get("l0a_wall_clock_s") is not None
    ]
    l0b_times = [
        float(row["l0b_wall_clock_s"])
        for row in populated
        if row.get("l0b_wall_clock_s") is not None
    ]
    decision_evidence_times = [
        float(row["decision_evidence_wall_clock_s"])
        for row in populated
        if row.get("decision_evidence_wall_clock_s") is not None
    ]
    result["l0a_wall_clock_s"] = max(l0a_times, default=None)
    result["decision_evidence_wall_clock_s"] = max(
        decision_evidence_times,
        default=None,
    )
    result["l0b_wall_clock_s"] = max(l0b_times, default=None)
    result["consistent_across_models"] = all(
        all(row.get(row_field) == result[name] for name, row_field in fields.items())
        and row.get("l0b_schema_version") == result["l0b_schema_version"]
        and _dict(row.get("l0b_projection_metrics")) == first_projection
        for row in populated[1:]
    )
    return result


def _shared_l0_execution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    fields = (
        "successful_runtime_seconds",
        "first_iteration",
        "last_iteration",
        "iteration_delta",
        "last_checkpoint_iteration",
        "iterations_since_checkpoint",
        "checkpoint_load_iteration",
        "latest_observed_failure_iteration",
        "latest_observed_failure_iteration_line",
        "observed_iterations_after_checkpoint_load",
        "observed_failure_phase",
        "progress_after_failure_episode",
        "first_terminal_incident_line",
        "first_terminal_incident_timestamp",
        "configured_terminal_timeout_seconds",
        "seconds_from_last_progress_to_terminal_incident",
        "terminal_detection_lag_seconds",
        "later_progress_after_fault_observation_count",
        "later_progress_after_fault_event_count",
    )
    first = rows[0]
    result = {field: first.get(f"l0_{field}") for field in fields}
    result["later_progress_after_fault_observations"] = (
        first.get("l0_later_progress_after_fault_observations") or []
    )
    result["distributed_failure_incidents"] = first.get("l0_distributed_failure_incidents") or []
    result["operation_artifact_comparisons"] = first.get("l0_operation_artifact_comparisons") or []
    result["consistent_across_models"] = all(
        all(row.get(f"l0_{field}") == result[field] for field in fields)
        and (row.get("l0_later_progress_after_fault_observations") or [])
        == result["later_progress_after_fault_observations"]
        and (row.get("l0_distributed_failure_incidents") or [])
        == result["distributed_failure_incidents"]
        and (row.get("l0_operation_artifact_comparisons") or [])
        == result["operation_artifact_comparisons"]
        for row in rows[1:]
    )
    return result


@dataclass(frozen=True)
class _PanelRowState:
    summary: dict[str, Any]
    is_model_route: bool
    primary: dict[str, Any]
    stage_selection: dict[str, Any]
    l0_deterministic_primary: dict[str, Any]
    l1_semantic_primary: dict[str, Any]
    l2_grounded_primary: dict[str, Any]
    current_failure_facts: dict[str, Any]
    failure_identity: dict[str, Any]
    family_identity: dict[str, Any]
    concrete_identity: dict[str, Any]
    client_concrete_identity: dict[str, Any]
    signals: dict[str, Any]
    timing: dict[str, Any]
    token_usage: dict[str, Any]
    token_limit: dict[str, Any]
    model_call_summary: dict[str, Any]
    tool_efficiency: dict[str, Any]
    semantic_safety: dict[str, Any]
    recovery_assessment: dict[str, Any]
    root_cause: dict[str, Any]
    retry_policy: dict[str, Any]
    l0_bundle_kpis: dict[str, Any]
    l1_kpis: dict[str, Any]
    l2_kpis: dict[str, Any]
    l3_kpis: dict[str, Any]
    l4_kpis: dict[str, Any]
    tool_profile: dict[str, Any]
    redaction: dict[str, Any]
    gold_score: dict[str, Any]
    gold_l0a: dict[str, Any]
    gold_l0b: dict[str, Any]
    gold_l1: dict[str, Any]
    gold_l2: dict[str, Any]
    gold_l4: dict[str, Any]
    gold_fallback_l4: dict[str, Any]
    gold_enriched_l4: dict[str, Any]
    gold_path_comparison: dict[str, Any]
    fallback_path: dict[str, Any]
    enriched_path: dict[str, Any]
    gold_l1_core_semantic: Any
    gold_l4_policy_action: Any


def _normalized_l1_kpis(
    *,
    summary: dict[str, Any],
    is_model_route: bool,
    timing: dict[str, Any],
    token_usage: dict[str, Any],
    token_limit: dict[str, Any],
    model_call_summary: dict[str, Any],
    tool_efficiency: dict[str, Any],
    semantic_safety: dict[str, Any],
    signals: dict[str, Any],
) -> dict[str, Any]:
    l1_kpis = _dict(summary.get("l1_kpis"))
    if not l1_kpis:
        l1_kpis = _fallback_l1_kpis(
            summary=summary,
            timing=timing,
            token_usage=token_usage,
            token_limit=token_limit,
            model_call_summary=model_call_summary,
            tool_efficiency=tool_efficiency,
            semantic_safety=semantic_safety,
            signals=signals,
        )
    l1_kpis.setdefault("tool_driven_model_turns", tool_efficiency.get("tool_driven_model_turns"))
    l1_kpis.setdefault(
        "contract_repair_turns",
        1 if l1_kpis.get("contract_repair_requested") else 0,
    )
    if not l1_kpis.get("execution_status"):
        issues = _l1_execution_issues(l1_kpis)
        l1_kpis["execution_status"] = (
            "degraded"
            if l1_kpis.get("output_usable") and issues
            else "ok" if l1_kpis.get("output_usable") else "failed"
        )
        l1_kpis["execution_issues"] = issues
    if not is_model_route:
        l1_kpis.update(
            execution_status="not_run",
            execution_issues=[],
            output_status="not_run",
            output_usable=False,
        )
    return l1_kpis


def _l1_execution_issues(l1_kpis: dict[str, Any]) -> list[str]:
    checks = (
        ("failed_model_calls", "model_call_failed"),
        ("retried_model_calls", "retry_used"),
        ("timeout_model_calls", "provider_timeout"),
        ("provider_error_count", "provider_error"),
    )
    return [issue for field, issue in checks if _int_or_zero(l1_kpis.get(field))]


def _gold_l1_core_semantic(gold_l1: dict[str, Any]) -> Any:
    result = gold_l1.get("core_semantic_pass")
    if result is not None or not gold_l1:
        return result
    checks = [gold_l1.get("root_cause_correct")]
    if gold_l1.get("recovery_assessment_correct") is not None:
        checks.append(gold_l1.get("recovery_assessment_correct"))
    return all(value is True for value in checks) and not bool(gold_l1.get("unsupported_claims"))


def _gold_l4_policy_action(gold_l4: dict[str, Any]) -> Any:
    result = gold_l4.get("policy_action_pass")
    if result is not None or not gold_l4:
        return result
    checks = [gold_l4.get("action_correct")]
    checks.extend(
        value
        for value in (
            gold_l4.get("retry_rule_correct"),
            gold_l4.get("allowed_retries_correct"),
            gold_l4.get("retry_exhaustion_correct"),
        )
        if value is not None
    )
    return all(value is True for value in checks)


def _prepare_panel_row_state(summary: dict[str, Any]) -> _PanelRowState:
    is_model_route = bool(summary.get("model"))
    primary = _dict(summary.get("primary_failure"))
    stage_selection = _dict(summary.get("primary_selection_by_stage"))
    l0_deterministic_primary = _dict(stage_selection.get("l0_deterministic"))
    l1_semantic_primary = _dict(stage_selection.get("l1_semantic"))
    l2_grounded_primary = _dict(stage_selection.get("l2_grounded"))
    current_failure_facts = _dict(summary.get("current_failure_facts"))
    failure_identity = _dict(primary.get("failure_identity"))
    family_identity = _dict(failure_identity.get("family"))
    concrete_identity = _dict(failure_identity.get("concrete"))
    client_concrete_identity = _dict(failure_identity.get("client_concrete"))
    signals = _dict(summary.get("model_selection_signals"))
    timing = _dict(summary.get("timing"))
    token_usage = _dict(summary.get("token_usage"))
    token_limit = _dict(summary.get("token_limit"))
    model_call_summary = _dict(summary.get("model_call_summary"))
    tool_efficiency = _dict(summary.get("tool_efficiency"))
    semantic_safety = _dict(summary.get("semantic_safety"))
    recovery_assessment = _dict(summary.get("model_recovery_assessment"))
    root_cause = _dict(summary.get("root_cause_assessment"))
    retry_policy = _dict(summary.get("retry_policy"))
    l0_bundle_kpis = _dict(summary.get("l0_kpis")) or _dict(summary.get("l0_bundle_kpis"))
    if not l0_bundle_kpis:
        l0_bundle_kpis = _fallback_l0_kpis(
            summary=summary,
            tool_efficiency=tool_efficiency,
        )
    l1_kpis = _normalized_l1_kpis(
        summary=summary,
        is_model_route=is_model_route,
        timing=timing,
        token_usage=token_usage,
        token_limit=token_limit,
        model_call_summary=model_call_summary,
        tool_efficiency=tool_efficiency,
        semantic_safety=semantic_safety,
        signals=signals,
    )
    l2_kpis = _dict(summary.get("l2_kpis"))
    l3_kpis = _dict(summary.get("l3_kpis"))
    l4_kpis = _dict(summary.get("l4_kpis"))
    tool_profile = _dict(summary.get("effective_tool_profile"))
    redaction = _dict(summary.get("path_redaction_audit"))
    gold_score = _dict(summary.get("gold_score"))
    gold_l0a = _dict(gold_score.get("l0a"))
    gold_l0b = _dict(gold_score.get("l0b"))
    gold_l1 = _dict(gold_score.get("l1"))
    gold_l2 = _dict(gold_score.get("l2"))
    gold_l4 = _dict(gold_score.get("l4"))
    gold_fallback_l4 = _dict(gold_score.get("fallback_l4"))
    gold_enriched_l4 = _dict(gold_score.get("enriched_l4"))
    gold_path_comparison = _dict(gold_score.get("l4_path_comparison"))
    decision_paths = _dict(summary.get("decision_paths"))
    fallback_path = _dict(decision_paths.get("deterministic_fallback"))
    enriched_path = _dict(decision_paths.get("l1_enriched"))
    if not is_model_route:
        gold_l1 = {}
    gold_l1_core_semantic = _gold_l1_core_semantic(gold_l1)
    gold_l4_policy_action = _gold_l4_policy_action(gold_l4)

    return _PanelRowState(
        summary=summary,
        is_model_route=is_model_route,
        primary=primary,
        stage_selection=stage_selection,
        l0_deterministic_primary=l0_deterministic_primary,
        l1_semantic_primary=l1_semantic_primary,
        l2_grounded_primary=l2_grounded_primary,
        current_failure_facts=current_failure_facts,
        failure_identity=failure_identity,
        family_identity=family_identity,
        concrete_identity=concrete_identity,
        client_concrete_identity=client_concrete_identity,
        signals=signals,
        timing=timing,
        token_usage=token_usage,
        token_limit=token_limit,
        model_call_summary=model_call_summary,
        tool_efficiency=tool_efficiency,
        semantic_safety=semantic_safety,
        recovery_assessment=recovery_assessment,
        root_cause=root_cause,
        retry_policy=retry_policy,
        l0_bundle_kpis=l0_bundle_kpis,
        l1_kpis=l1_kpis,
        l2_kpis=l2_kpis,
        l3_kpis=l3_kpis,
        l4_kpis=l4_kpis,
        tool_profile=tool_profile,
        redaction=redaction,
        gold_score=gold_score,
        gold_l0a=gold_l0a,
        gold_l0b=gold_l0b,
        gold_l1=gold_l1,
        gold_l2=gold_l2,
        gold_l4=gold_l4,
        gold_fallback_l4=gold_fallback_l4,
        gold_enriched_l4=gold_enriched_l4,
        gold_path_comparison=gold_path_comparison,
        fallback_path=fallback_path,
        enriched_path=enriched_path,
        gold_l1_core_semantic=gold_l1_core_semantic,
        gold_l4_policy_action=gold_l4_policy_action,
    )


def _row(summary: dict[str, Any]) -> dict[str, Any]:
    state = _prepare_panel_row_state(summary)
    return {
        **_panel_row_route_and_identity_fields(state),
        **_panel_row_gold_fields(state),
        **_panel_row_l0_fields(state),
        **_panel_row_l1_fields(state),
        **_panel_row_l2_l3_l4_fields(state),
        **_panel_row_operational_fields(state),
    }


def _panel_row_route_and_identity_fields(state: _PanelRowState) -> dict[str, Any]:
    return {
        'target': state.summary.get('target'),
        'model': state.summary.get('model'),
        'tool_profile_id': state.tool_profile.get('profile_id'),
        'tool_profile_experimental': state.tool_profile.get('experimental'),
        'tools_enabled': state.tool_profile.get('tools_enabled'),
        'max_tool_rounds': state.tool_profile.get('max_tool_rounds'),
        'max_model_turns': state.tool_profile.get('max_model_turns'),
        'tool_profile_source': state.tool_profile.get('source'),
        'decision': state.summary.get('decision'),
        'decision_basis': state.summary.get('decision_basis'),
        'fallback_available': state.fallback_path.get('available'),
        'fallback_decision': state.fallback_path.get('decision'),
        'fallback_retry_rule': state.fallback_path.get('retry_rule'),
        'fallback_ready_wall_clock_s': state.fallback_path.get('ready_wall_clock_s'),
        'enriched_available': state.enriched_path.get('available'),
        'enriched_decision': state.enriched_path.get('decision'),
        'enriched_retry_rule': state.enriched_path.get('retry_rule'),
        'enriched_ready_wall_clock_s': state.enriched_path.get('ready_wall_clock_s'),
        'model_recovery_confidence': {
            'failure_domain': _claim_field(
                state.recovery_assessment, 'failure_domain', 'confidence'
            ),
            'retry_outlook': _claim_field(
                state.recovery_assessment,
                'retry_outlook_without_workload_change',
                'confidence',
            ),
        },
        'primary_line': state.primary.get('line'),
        'primary_class': state.primary.get('fine_class'),
        'l0_deterministic_primary_line': state.l0_deterministic_primary.get('line'),
        'l0_deterministic_primary_class': state.l0_deterministic_primary.get('fine_class'),
        'l0_root_fingerprint_owner': state.l0_bundle_kpis.get('root_fingerprint_owner'),
        'l0_root_fingerprint': state.l0_bundle_kpis.get('root_fingerprint'),
        'l0_root_fingerprint_source': state.l0_bundle_kpis.get('root_fingerprint_source'),
        'l0_root_fingerprint_available': state.l0_bundle_kpis.get('root_fingerprint_available'),
        'l0_history_identity_ready': state.l0_bundle_kpis.get('history_identity_ready'),
        'l1_semantic_primary_line': state.l1_semantic_primary.get('line'),
        'l1_semantic_primary_class': state.l1_semantic_primary.get('fine_class'),
        'l2_grounded_primary_line': state.l2_grounded_primary.get('line'),
        'l2_grounded_primary_class': state.l2_grounded_primary.get('fine_class'),
        'l1_primary_relation_to_l0': state.stage_selection.get('l1_relation_to_l0'),
        'l2_primary_relation_to_l0': state.stage_selection.get('l2_relation_to_l0'),
        'policy_class': state.primary.get('policy_class'),
        'fault_outcome': state.primary.get('fault_outcome'),
        'causal_role': state.primary.get('causal_role'),
        'root_fingerprint': state.primary.get('root_fingerprint'),
        'root_fingerprint_source': state.primary.get('root_fingerprint_source'),
        'stable_identity_anchor_line': state.l2_kpis.get('stable_identity_anchor_line'),
        'stable_identity_anchor_reason': state.l2_kpis.get('stable_identity_anchor_reason'),
        'failure_identity_schema': state.failure_identity.get('schema_version'),
        'failure_identity_policy_active': state.failure_identity.get('policy_active'),
        'family_operation': state.family_identity.get('operation'),
        'family_mechanism': state.family_identity.get('mechanism'),
        'family_exception_type': state.family_identity.get('exception_type'),
        'family_label': state.family_identity.get('label'),
        'family_fingerprint': state.family_identity.get('fingerprint'),
        'family_complete': state.family_identity.get('complete'),
        'concrete_component': state.concrete_identity.get('component'),
        'concrete_callsite': state.concrete_identity.get('callsite'),
        'concrete_artifact_path': state.concrete_identity.get('artifact_path'),
        'concrete_failure_position': state.concrete_identity.get('failure_position'),
        'concrete_stack_path': state.concrete_identity.get('stack_path') or [],
        'concrete_label': state.concrete_identity.get('label'),
        'concrete_fingerprint': state.concrete_identity.get('fingerprint'),
        'concrete_complete': state.concrete_identity.get('complete'),
        'client_concrete_exception_type': state.client_concrete_identity.get('exception_type'),
        'client_concrete_message_signature': state.client_concrete_identity.get(
            'message_signature'
        ),
        'client_concrete_source_file': state.client_concrete_identity.get('source_file'),
        'client_concrete_callsite': state.client_concrete_identity.get('callsite'),
        'client_concrete_artifact_path': state.client_concrete_identity.get('artifact_path'),
        'client_concrete_failure_position': state.client_concrete_identity.get('failure_position'),
        'client_concrete_phase': state.client_concrete_identity.get('phase'),
        'client_concrete_checkpoint_iteration': state.client_concrete_identity.get(
            'checkpoint_iteration'
        ),
        'client_concrete_operation_signature': state.client_concrete_identity.get(
            'operation_signature'
        ),
        'client_concrete_stack_path': state.client_concrete_identity.get('stack_path') or [],
        'client_concrete_label': state.client_concrete_identity.get('label'),
        'client_concrete_fingerprint': state.client_concrete_identity.get('fingerprint'),
        'client_concrete_complete': state.client_concrete_identity.get('complete'),
    }


def _panel_row_gold_fields(state: _PanelRowState) -> dict[str, Any]:
    return {
        'gold_case_id': state.gold_score.get('case_id'),
        'gold_l0a_overall': state.gold_l0a.get('overall_pass'),
        'gold_l0a_primary_evidence_coverage': state.gold_l0a.get('primary_evidence_coverage'),
        'gold_l0a_selected_primary_accuracy': state.gold_l0a.get('selected_primary_accuracy'),
        'gold_l0a_root_fingerprint_accuracy': state.gold_l0a.get('root_fingerprint_accuracy'),
        'gold_l0a_progress_line_recall': state.gold_l0a.get('progress_line_recall'),
        'gold_l0a_checkpoint_line_recall': state.gold_l0a.get('checkpoint_line_recall'),
        'gold_l0a_primary_phase_correct': state.gold_l0a.get('primary_phase_correct'),
        'gold_l0a_checkpoint_load_iteration_correct': state.gold_l0a.get(
            'checkpoint_load_iteration_correct'
        ),
        'gold_l0a_progress_after_failure_correct': state.gold_l0a.get(
            'progress_after_failure_correct'
        ),
        'gold_l0a_cascade_line_recall': state.gold_l0a.get('cascade_line_recall'),
        'gold_l0a_required_setup_marker_types': state.gold_l0a.get('required_setup_marker_types')
        or [],
        'gold_l0a_observed_setup_marker_types': state.gold_l0a.get('observed_setup_marker_types')
        or [],
        'gold_l0a_coverage_checks': state.gold_l0a.get('coverage_checks') or {},
        'gold_l0b_overall': state.gold_l0b.get('overall_pass'),
        'gold_l0b_required_evidence_line_recall': state.gold_l0b.get(
            'required_evidence_line_recall'
        ),
        'gold_l0b_primary_retained_from_l0a': state.gold_l0b.get('primary_retained_from_l0a'),
        'gold_l0b_projection_integrity_ok': state.gold_l0b.get('projection_integrity_ok'),
        'gold_l1_root_cause_correct': state.gold_l1.get('root_cause_correct'),
        'gold_l1_recovery_correct': state.gold_l1.get('recovery_assessment_correct'),
        'gold_l1_operation_correct': state.gold_l1.get('root_cause_operation_correct'),
        'gold_l1_mechanism_contradiction': state.gold_l1.get('root_cause_mechanism_contradiction'),
        'gold_l1_recovery_fields': state.gold_l1.get('recovery_field_results') or {},
        'gold_l1_related_failure_recall': state.gold_l1.get(
            'related_failure_recall', state.gold_l1.get('cascade_correct')
        ),
        'gold_l1_cascade_correct': state.gold_l1.get('cascade_correct'),
        'gold_l1_unsupported_claims': state.gold_l1.get('unsupported_claims') or [],
        'gold_l1_confidence': state.gold_l1.get('model_recovery_confidence'),
        'gold_l1_core_semantic': state.gold_l1_core_semantic,
        'gold_l1_overall': state.gold_l1.get('overall_semantic_pass'),
        'gold_l2_audit_correct': state.gold_l2.get('audit_correct'),
        'gold_l2_history_identity_correct': state.gold_l2.get('history_identity_correct'),
        'gold_l2_canonical_anchor_correct': state.gold_l2.get('canonical_anchor_correct'),
        'gold_l2_operation_correct': state.gold_l2.get('operation_correct'),
        'gold_l2_mechanism_correct': state.gold_l2.get('mechanism_correct'),
        'gold_l2_expected_cross_route_identity_count': state.gold_l2.get(
            'expected_cross_route_identity_count'
        ),
        'gold_l2_reference_audit_effect': state.gold_l2.get('reference_audit_effect'),
        'gold_l4_root_cause_correct': state.gold_l4.get('root_cause_correct'),
        'gold_l4_retry_rule_correct': state.gold_l4.get('retry_rule_correct'),
        'gold_l4_allowed_retries_correct': state.gold_l4.get('allowed_retries_correct'),
        'gold_l4_exhaustion_correct': state.gold_l4.get('retry_exhaustion_correct'),
        'gold_l4_action_correct': state.gold_l4.get('action_correct'),
        'gold_l4_cascade_correct': state.gold_l4.get('cascade_correct'),
        'gold_l4_teardown_role_correct': state.gold_l4.get('teardown_role_correct'),
        'gold_l4_policy_action': state.gold_l4_policy_action,
        'gold_l4_overall': state.gold_l4.get('overall_semantic_pass'),
        'gold_fallback_action_correct': state.gold_fallback_l4.get('action_correct'),
        'gold_fallback_policy_action': state.gold_fallback_l4.get('policy_action_pass'),
        'gold_enriched_action_correct': state.gold_enriched_l4.get('action_correct'),
        'gold_enriched_policy_action': state.gold_enriched_l4.get('policy_action_pass'),
        'l1_action_effect': state.gold_path_comparison.get('action_effect'),
        'l1_policy_action_effect': state.gold_path_comparison.get('policy_action_effect'),
    }


def _panel_row_l0_fields(state: _PanelRowState) -> dict[str, Any]:
    return {
        'l0_wall_clock_s': state.l0_bundle_kpis.get('l0_wall_clock_s'),
        'l0_reused': state.l0_bundle_kpis.get('l0_reused'),
        'l0a_wall_clock_s': state.l0_bundle_kpis.get('l0a_wall_clock_s'),
        'decision_evidence_wall_clock_s': state.l0_bundle_kpis.get(
            'decision_evidence_wall_clock_s'
        ),
        'l0b_wall_clock_s': state.l0_bundle_kpis.get('l0b_wall_clock_s'),
        'l0b_schema_version': state.l0_bundle_kpis.get('l0b_schema_version'),
        'restart_environment_context': state.l0_bundle_kpis.get('restart_environment_context')
        or {},
        'l0b_projection_metrics': state.l0_bundle_kpis.get('l0b_projection_metrics') or {},
        'l0_line_count': state.l0_bundle_kpis.get('line_count'),
        'l0_byte_size': state.l0_bundle_kpis.get('byte_size'),
        'l0_context_window_count': state.l0_bundle_kpis.get('context_window_count'),
        'l0_candidate_anchor_count': state.l0_bundle_kpis.get('candidate_anchor_count'),
        'l0_occurrence_group_count': state.l0_bundle_kpis.get('occurrence_group_count'),
        'l0_failure_episode_count': state.l0_bundle_kpis.get('failure_episode_count'),
        'l0_distributed_failure_incident_count': state.l0_bundle_kpis.get(
            'distributed_failure_incident_count'
        ),
        'l0_distributed_failure_incidents': state.l0_bundle_kpis.get(
            'distributed_failure_incidents'
        )
        or state.summary.get('distributed_failure_incidents')
        or [],
        'l0_operation_artifact_comparisons': state.l0_bundle_kpis.get(
            'operation_artifact_comparisons'
        )
        or state.summary.get('operation_artifact_comparisons')
        or [],
        'l0_path_access_fact_count': state.l0_bundle_kpis.get('path_access_fact_count'),
        'l0_path_namespaces_by_role': state.l0_bundle_kpis.get('path_namespaces_by_role') or {},
        'l0_cross_namespace_paths_observed': state.l0_bundle_kpis.get(
            'cross_namespace_paths_observed'
        ),
        'l0_failed_vs_configured_write_mismatch': state.l0_bundle_kpis.get(
            'failed_vs_configured_write_mismatch'
        ),
        'l0_path_ownership_verified': state.l0_bundle_kpis.get('path_ownership_verified'),
        'l0_selected_primary_in_bundle': state.l0_bundle_kpis.get('selected_primary_in_bundle'),
        'l0_selected_primary_in_excerpt': state.l0_bundle_kpis.get('selected_primary_in_excerpt'),
        'l0_progress_after_fault_known': state.l0_bundle_kpis.get('progress_after_fault_known'),
        'l0_progress_after_fault': state.l0_bundle_kpis.get('progress_after_fault'),
        'l0_successful_runtime_seconds': state.l0_bundle_kpis.get('successful_runtime_seconds'),
        'l0_first_iteration': state.l0_bundle_kpis.get('first_iteration'),
        'l0_last_iteration': state.l0_bundle_kpis.get('last_iteration'),
        'l0_iteration_delta': state.l0_bundle_kpis.get('iteration_delta'),
        'l0_last_checkpoint_iteration': state.l0_bundle_kpis.get('last_checkpoint_iteration'),
        'l0_iterations_since_checkpoint': state.l0_bundle_kpis.get('iterations_since_checkpoint'),
        'l0_checkpoint_load_iteration': state.l0_bundle_kpis.get('checkpoint_load_iteration'),
        'l0_latest_observed_failure_iteration': state.l0_bundle_kpis.get(
            'latest_observed_failure_iteration'
        ),
        'l0_latest_observed_failure_iteration_line': state.l0_bundle_kpis.get(
            'latest_observed_failure_iteration_line'
        ),
        'l0_observed_iterations_after_checkpoint_load': state.l0_bundle_kpis.get(
            'observed_iterations_after_checkpoint_load'
        ),
        'l0_observed_failure_phase': state.l0_bundle_kpis.get('observed_failure_phase'),
        'l0_progress_after_failure_episode': state.l0_bundle_kpis.get('progress_after_fault'),
        'l0_first_terminal_incident_line': state.l0_bundle_kpis.get('first_terminal_incident_line'),
        'l0_first_terminal_incident_timestamp': state.l0_bundle_kpis.get(
            'first_terminal_incident_timestamp'
        ),
        'l0_configured_terminal_timeout_seconds': state.l0_bundle_kpis.get(
            'configured_terminal_timeout_seconds'
        ),
        'l0_seconds_from_last_progress_to_terminal_incident': state.l0_bundle_kpis.get(
            'seconds_from_last_progress_to_terminal_incident'
        ),
        'l0_terminal_detection_lag_seconds': state.l0_bundle_kpis.get(
            'terminal_detection_lag_seconds'
        ),
        'l0_later_progress_after_fault_observation_count': state.l0_bundle_kpis.get(
            'later_progress_after_fault_observation_count'
        ),
        'l0_later_progress_after_fault_event_count': state.l0_bundle_kpis.get(
            'later_progress_after_fault_event_count'
        ),
        'l0_later_progress_after_fault_observations': state.summary.get(
            'later_progress_after_fault_observations'
        )
        or [],
        'l0_setup_marker_count': state.l0_bundle_kpis.get('setup_marker_count'),
        'l0_setup_marker_types': state.l0_bundle_kpis.get('setup_marker_types') or [],
        'l0_setup_marker_lines': state.l0_bundle_kpis.get('setup_marker_lines') or [],
        'l0_tool_calls_useful_proxy': state.l0_bundle_kpis.get('tool_calls_useful_proxy'),
        'l0_tool_calls_added_new_prompt_lines': state.l0_bundle_kpis.get(
            'tool_calls_added_new_prompt_lines'
        ),
        'l0_candidate_anchors_without_excerpt': state.l0_bundle_kpis.get(
            'candidate_anchors_without_excerpt'
        ),
        'l0_recovered_or_progressed_top_anchor_count': state.l0_bundle_kpis.get(
            'recovered_or_progressed_top_anchor_count'
        ),
    }


def _panel_row_l1_fields(state: _PanelRowState) -> dict[str, Any]:
    return {
        'l1_response_parsed': state.summary.get('l1_response_parsed'),
        'l1_kpi_response_parsed': state.l1_kpis.get('response_parsed'),
        'l1_output_status': state.l1_kpis.get('output_status'),
        'l1_output_usable': state.l1_kpis.get('output_usable'),
        'l1_execution_status': state.l1_kpis.get('execution_status'),
        'l1_execution_issues': state.l1_kpis.get('execution_issues') or [],
        'l1_contract_repair_requested': state.l1_kpis.get('contract_repair_requested'),
        'l1_contract_repair_turns': state.l1_kpis.get('contract_repair_turns'),
        'l1_kpi_wall_clock_s': state.l1_kpis.get('wall_clock_s'),
        'l1_kpi_model_call_wall_clock_s': state.l1_kpis.get('model_call_wall_clock_s'),
        'l1_kpi_tool_wall_clock_s': state.l1_kpis.get('tool_wall_clock_s'),
        'l1_kpi_model_calls': state.l1_kpis.get('model_calls'),
        'l1_kpi_model_turns': state.l1_kpis.get('model_turns'),
        'l1_kpi_successful_model_calls': state.l1_kpis.get('successful_model_calls'),
        'l1_kpi_failed_model_calls': state.l1_kpis.get('failed_model_calls'),
        'l1_kpi_retried_model_calls': state.l1_kpis.get('retried_model_calls'),
        'l1_kpi_timeout_model_calls': state.l1_kpis.get('timeout_model_calls'),
        'l1_kpi_provider_error_count': state.l1_kpis.get('provider_error_count'),
        'l1_kpi_provider_reported_timing': state.l1_kpis.get('provider_reported_timing'),
        'l1_kpi_context_budget_adjusted_calls': state.l1_kpis.get('context_budget_adjusted_calls'),
        'l1_kpi_context_window_tokens': state.l1_kpis.get('context_window_tokens'),
        'l1_kpi_max_estimated_input_tokens': state.l1_kpis.get('max_estimated_input_tokens'),
        'l1_kpi_configured_max_output_tokens': state.l1_kpis.get('configured_max_output_tokens'),
        'l1_kpi_minimum_effective_max_output_tokens': state.l1_kpis.get(
            'minimum_effective_max_output_tokens'
        ),
        'l1_kpi_tool_calls': state.l1_kpis.get('tool_calls'),
        'l1_kpi_tool_driven_model_turns': state.l1_kpis.get('tool_driven_model_turns'),
        'l1_kpi_extra_model_turns_after_initial': state.l1_kpis.get(
            'extra_model_turns_after_initial'
        ),
        'l1_kpi_tool_calls_added_new_prompt_lines': state.l1_kpis.get(
            'tool_calls_added_new_prompt_lines'
        ),
        'l1_kpi_duplicate_tool_calls': state.l1_kpis.get('duplicate_tool_calls'),
        'l1_kpi_no_new_prompt_line_tool_calls': state.l1_kpis.get('no_new_prompt_line_tool_calls'),
        'l1_kpi_tool_error_calls': state.l1_kpis.get('tool_error_calls'),
        'l1_kpi_tool_truncated_calls': state.l1_kpis.get('tool_truncated_calls'),
        'l1_kpi_total_tokens': state.l1_kpis.get('total_tokens'),
        'l1_kpi_token_limit_hit': state.l1_kpis.get('token_limit_hit'),
        'l1_kpi_context_efficiency': state.l1_kpis.get('context_efficiency'),
        'l1_kpi_endpoint_reliability': state.l1_kpis.get('endpoint_reliability'),
        'l1_kpi_client_request_health': state.l1_kpis.get('client_request_health'),
        'l1_kpi_context_window_error_calls': state.l1_kpis.get('context_window_error_calls'),
        'l1_kpi_errors_count': state.l1_kpis.get('errors_count'),
    }


def _claim_field(assessment: Mapping[str, Any], claim_name: str, field: str) -> Any:
    claim = assessment.get(claim_name) or {}
    return claim.get(field) if isinstance(claim, Mapping) else None


def _panel_row_l2_l3_l4_fields(state: _PanelRowState) -> dict[str, Any]:
    return {
        'l2_audit_ran': state.summary.get('l2_audit_ran'),
        'l2_grounding_status': state.l2_kpis.get('grounding_status'),
        'l2_grounding_method': state.l2_kpis.get('grounding_method'),
        'l2_audit_status': state.l2_kpis.get('audit_status'),
        'l2_wall_clock_s': state.l2_kpis.get('wall_clock_s'),
        'l2_primary_available': state.l2_kpis.get('primary_available'),
        'l2_recovery_assessment_available': state.l2_kpis.get('recovery_assessment_available'),
        'l2_related_failures_audited': state.l2_kpis.get('related_failures_audited'),
        'l2_failure_domain_supporting_lines': state.l2_kpis.get('failure_domain_supporting_lines')
        or [],
        'l2_retry_outlook_supporting_lines': state.l2_kpis.get('retry_outlook_supporting_lines')
        or [],
        'l2_unresolved_recovery_supporting_lines': state.l2_kpis.get(
            'unresolved_recovery_supporting_lines'
        )
        or [],
        'l2_finding_count': state.l2_kpis.get('finding_count'),
        'l2_material_finding_count': state.l2_kpis.get('material_finding_count'),
        'l2_finding_severity_counts': state.l2_kpis.get('finding_severity_counts') or {},
        'l2_exact_citation_count': state.l2_kpis.get('exact_citation_count'),
        'l2_rendered_exact_citation_count': state.l2_kpis.get('rendered_exact_citation_count'),
        'l2_abbreviated_exact_citation_count': state.l2_kpis.get(
            'abbreviated_exact_citation_count', state.l2_kpis.get('abbreviated_exact_count', 0)
        ),
        'l2_nearby_resolved_count': state.l2_kpis.get('nearby_resolved_count'),
        'l2_ungrounded_citation_count': state.l2_kpis.get('ungrounded_citation_count'),
        'l2_grounding_adjustment_count': state.l2_kpis.get('grounding_adjustment_count'),
        'l2_recovery_audit_observation_count': state.l2_kpis.get(
            'recovery_audit_observation_count'
        ),
        'l2_recovery_field_audits': state.l2_kpis.get('recovery_field_audits') or [],
        'l2_operation_artifact_audit_observations': state.l2_kpis.get(
            'operation_artifact_audit_observations'
        )
        or [],
        'l2_root_fingerprint_owner': state.l2_kpis.get('root_fingerprint_owner'),
        'l2_root_fingerprint': state.l2_kpis.get('root_fingerprint'),
        'l2_root_fingerprint_source': state.l2_kpis.get('root_fingerprint_source'),
        'l2_root_fingerprint_available': state.l2_kpis.get('root_fingerprint_available'),
        'l2_history_identity_ready': state.l2_kpis.get('history_identity_ready'),
        'l2_matches_l0_root_fingerprint': state.l2_kpis.get('matches_l0_root_fingerprint'),
        'current_failure_facts_source': state.current_failure_facts.get('source'),
        'current_history_identity_ready': state.current_failure_facts.get('history_identity_ready'),
        'current_root_fingerprint': state.current_failure_facts.get('root_fingerprint'),
        'l3_wall_clock_s': state.l3_kpis.get('wall_clock_s'),
        'l3_history_available': state.l3_kpis.get('history_available'),
        'l3_same_job_attempts': state.l3_kpis.get('same_job_attempts'),
        'l3_matching_root_attempts': state.l3_kpis.get('matching_root_attempts'),
        'l3_observed_advance_attempts': state.l3_kpis.get('observed_advance_attempts'),
        'l3_no_observed_advance_attempts': state.l3_kpis.get('no_observed_advance_attempts'),
        'l3_unknown_progress_attempts': state.l3_kpis.get('unknown_progress_attempts'),
        'l3_exact_failure_position_attempts': state.l3_kpis.get('exact_failure_position_attempts'),
        'l3_same_data_position_attempts': state.l3_kpis.get('same_data_position_attempts'),
        'l3_same_artifact_attempts': state.l3_kpis.get('same_artifact_attempts'),
        'l3_consecutive_same_root_no_advance_attempts': state.l3_kpis.get(
            'consecutive_same_root_no_advance_attempts'
        ),
        'l3_advanced_beyond_all_comparable_attempts': state.l3_kpis.get(
            'advanced_beyond_all_comparable_attempts'
        ),
        'l4_wall_clock_s': state.l4_kpis.get('wall_clock_s'),
        'l4_result_quality': state.l4_kpis.get('result_quality'),
        'l4_nvrx_use': state.l4_kpis.get('nvrx_use'),
        'l4_policy_version': state.l4_kpis.get('policy_version'),
        'l4_retry_rule': state.l4_kpis.get('rule') or state.retry_policy.get('rule'),
        'l4_allowed_retries': state.l4_kpis.get('allowed_retries'),
        'l4_matching_prior_failures': state.l4_kpis.get('matching_prior_failures'),
        'l4_retry_budget_exhausted': state.l4_kpis.get('retry_budget_exhausted'),
        'l4_downstream_roles': state.l4_kpis.get('downstream_roles') or [],
        'l4_recovery_assessment_policy_grounded': state.l4_kpis.get(
            'recovery_assessment_policy_grounded'
        ),
        'l4_current_evidence_qualified': state.l4_kpis.get('current_evidence_qualified'),
        'l4_observed_advance': state.l4_kpis.get('observed_advance'),
        'l4_failure_domain': state.l4_kpis.get('failure_domain'),
        'l4_failure_domain_status': state.l4_kpis.get('failure_domain_status'),
        'l4_failure_domain_confidence': state.l4_kpis.get('failure_domain_confidence'),
        'l4_retry_outlook_without_workload_change': state.l4_kpis.get(
            'retry_outlook_without_workload_change'
        ),
        'l4_retry_outlook_status': state.l4_kpis.get('retry_outlook_status'),
        'l4_retry_outlook_confidence': state.l4_kpis.get('retry_outlook_confidence'),
        'latency_mode': state.l4_kpis.get('latency_mode'),
        'terminal_total_wall_clock_s': state.l4_kpis.get('terminal_total_wall_clock_s'),
        'post_progressive_end_wall_clock_s': state.l4_kpis.get('post_progressive_end_wall_clock_s'),
        'production_gate_measured': state.l4_kpis.get('production_gate_measured'),
    }


def _panel_row_operational_fields(state: _PanelRowState) -> dict[str, Any]:
    return {
        'l1_wall_clock_s': state.timing.get('l1_wall_clock_s'),
        'model_call_wall_clock_s': state.timing.get('l1_model_call_wall_clock_s'),
        'tool_wall_clock_s': state.timing.get('l1_tool_wall_clock_s'),
        'model_calls': state.summary.get('model_calls'),
        'tool_calls': state.summary.get('tool_calls'),
        'tool_names': state.summary.get('tool_names') or [],
        'context_efficiency': state.signals.get('context_efficiency'),
        'semantic_safety': state.signals.get('semantic_safety'),
        'endpoint_reliability': state.signals.get('endpoint_reliability'),
        'client_request_health': state.signals.get('client_request_health'),
        'failed_endpoint_attempts': state.signals.get('failed_endpoint_attempts'),
        'failed_model_calls': state.model_call_summary.get('failed_calls'),
        'retried_model_calls': state.model_call_summary.get('retried_calls'),
        'timeout_model_calls': state.model_call_summary.get('timeout_calls'),
        'http_error_calls': state.model_call_summary.get('http_error_calls'),
        'context_window_error_calls': state.model_call_summary.get('context_window_error_calls'),
        'provider_error_count': state.model_call_summary.get('provider_error_count'),
        'duplicate_prompt_context_calls': state.tool_efficiency.get(
            'duplicate_prompt_context_calls'
        ),
        'no_new_prompt_line_calls': state.tool_efficiency.get('no_new_prompt_line_calls'),
        'new_prompt_excerpt_line_count': state.tool_efficiency.get('new_prompt_excerpt_line_count'),
        'tool_final_context_dependency': state.tool_efficiency.get('final_context_dependency'),
        'tool_final_context_impact': state.tool_efficiency.get('final_context_impact'),
        'tool_final_primary_from_tool_only_context': state.tool_efficiency.get(
            'final_primary_from_tool_only_context'
        ),
        'tool_final_evidence_tool_only_lines': state.tool_efficiency.get(
            'final_evidence_tool_only_lines'
        )
        or [],
        'tool_decision_relevant_tool_only_lines': state.tool_efficiency.get(
            'decision_relevant_tool_only_lines'
        )
        or [],
        'tool_structured_fact_redundant_lines': state.tool_efficiency.get(
            'structured_fact_redundant_tool_only_lines'
        )
        or [],
        'tool_structured_fact_redundant_line_labels': state.tool_efficiency.get(
            'structured_fact_redundant_tool_only_line_labels'
        )
        or {},
        'tool_incidental_tool_only_lines': state.tool_efficiency.get('incidental_tool_only_lines')
        or [],
        'tool_unused_tool_only_lines': state.tool_efficiency.get('unused_tool_only_lines') or [],
        'tool_unique_new_prompt_line_count': state.tool_efficiency.get(
            'unique_new_prompt_excerpt_line_count'
        ),
        'total_tokens': state.token_usage.get('total_tokens'),
        'prompt_tokens': state.token_usage.get('prompt_tokens'),
        'completion_tokens': state.token_usage.get('completion_tokens'),
        'token_limit_hit': state.token_limit.get('hit'),
        'redaction_passed': state.redaction.get('passed'),
        'redaction_source_content_overlap_tokens': state.redaction.get(
            'source_content_overlap_tokens'
        )
        or [],
        'model_failure_domain': _claim_field(state.recovery_assessment, 'failure_domain', 'value'),
        'model_failure_domain_status': _claim_field(
            state.recovery_assessment, 'failure_domain', 'status'
        ),
        'model_failure_domain_confidence': _claim_field(
            state.recovery_assessment, 'failure_domain', 'confidence'
        ),
        'model_root_cause_status': state.root_cause.get('status'),
        'model_root_cause_missing_evidence': state.root_cause.get('missing_evidence') or [],
        'model_retry_outlook_without_workload_change': _claim_field(
            state.recovery_assessment,
            'retry_outlook_without_workload_change',
            'value',
        ),
        'model_retry_outlook_status': _claim_field(
            state.recovery_assessment,
            'retry_outlook_without_workload_change',
            'status',
        ),
        'model_retry_outlook_confidence': _claim_field(
            state.recovery_assessment,
            'retry_outlook_without_workload_change',
            'confidence',
        ),
        'l2_recovery_suggestion_applied': state.semantic_safety.get(
            'l2_recovery_suggestion_applied'
        ),
        'artifacts': state.summary.get('artifacts') or {},
    }


def _concerns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rules = (
        _gold_concerns,
        _endpoint_concerns,
        _l1_l2_concerns,
        _tool_and_semantic_concerns,
        _coverage_and_redaction_concerns,
    )
    return [concern for row in rows for rule in rules for concern in rule(row)]


def _concern(row: dict[str, Any], category: str, summary: str) -> dict[str, Any]:
    return {"target": row.get("target"), "category": category, "summary": summary}


def _gold_concerns(row: dict[str, Any]) -> list[dict[str, Any]]:
    if not row.get("gold_case_id"):
        return []
    checks = list(
        (
            ("gold_l1_root_cause_correct", "gold_l1_root_cause"),
            ("gold_l1_recovery_correct", "gold_l1_recovery"),
        )
        if row.get("model")
        else ()
    )
    checks.extend(
        (
            ("gold_l4_cascade_correct", "gold_final_cascade"),
            ("gold_l4_retry_rule_correct", "gold_l4_retry_rule"),
            ("gold_l4_allowed_retries_correct", "gold_l4_retry_budget"),
            ("gold_l4_exhaustion_correct", "gold_l4_exhaustion"),
            ("gold_l4_action_correct", "gold_l4_action"),
        )
    )
    concerns = [
        _concern(row, category, "result did not match the human-approved gold label")
        for field, category in checks
        if row.get(field) is False
    ]
    if row.get("model") and row.get("gold_l1_unsupported_claims"):
        concerns.append(
            _concern(
                row,
                "gold_l1_unsupported_claim",
                ", ".join(row.get("gold_l1_unsupported_claims") or []),
            )
        )
    return concerns


def _endpoint_concerns(row: dict[str, Any]) -> list[dict[str, Any]]:
    concerns = []
    if row.get("endpoint_reliability") != "ok":
        concerns.append(
            _concern(
                row,
                "endpoint_reliability",
                f"failed_attempts={row.get('failed_endpoint_attempts')} "
                f"retried={row.get('retried_model_calls')} "
                f"timeouts={row.get('timeout_model_calls')} "
                f"http_errors={row.get('http_error_calls')} "
                f"provider_errors={row.get('provider_error_count')}",
            )
        )
    if row.get("client_request_health") == "context_budget_exceeded":
        concerns.append(
            _concern(
                row,
                "client_context_budget",
                "request exceeded the declared model context window; "
                f"calls={row.get('context_window_error_calls')}",
            )
        )
    return concerns


def _l1_l2_concerns(row: dict[str, Any]) -> list[dict[str, Any]]:
    concerns = []
    if row.get("l1_response_parsed") is False:
        concerns.append(
            _concern(
                row,
                "l1_unavailable",
                "L1 did not produce a parseable model result; any final decision is "
                f"analyzer fallback via {row.get('decision_basis')}",
            )
        )
    if row.get("l1_contract_repair_requested"):
        concerns.append(
            _concern(row, "l1_contract_repair", "initial model response required contract repair")
        )
    if _int_or_zero(row.get("l2_material_finding_count")):
        concerns.append(
            _concern(
                row,
                "l2_audit",
                f"material findings={row.get('l2_material_finding_count')} "
                f"of {row.get('l2_finding_count')} total; "
                f"severity={row.get('l2_finding_severity_counts')}",
            )
        )
    return concerns


def _tool_and_semantic_concerns(row: dict[str, Any]) -> list[dict[str, Any]]:
    concerns = []
    if row.get("context_efficiency") in {
        "unnecessary_tool_context",
        "low_yield_tool_context",
    }:
        concerns.append(
            _concern(
                row,
                "tool_efficiency",
                f"context={row.get('context_efficiency')} tools={row.get('tool_calls')} "
                f"dup_tools={row.get('duplicate_prompt_context_calls')} "
                f"no_new={row.get('no_new_prompt_line_calls')}",
            )
        )
    if row.get("semantic_safety") == "stable_identity_adjusted":
        concerns.append(
            _concern(
                row,
                "semantic_safety",
                f"semantic={row.get('semantic_safety')} "
                f"domain={row.get('model_failure_domain')} "
                f"retry_outlook={row.get('model_retry_outlook_without_workload_change')} "
                f"retry_rule={row.get('l4_retry_rule')}",
            )
        )
    if row.get("token_limit_hit"):
        concerns.append(_concern(row, "token_limit", "token limit was hit"))
    return concerns


def _coverage_and_redaction_concerns(row: dict[str, Any]) -> list[dict[str, Any]]:
    concerns = []
    if row.get("redaction_passed") is False:
        concerns.append(
            _concern(row, "redaction", "source path token appeared in model transcript")
        )
    if row.get("l0_selected_primary_in_excerpt") == "no":
        concerns.append(
            _concern(
                row,
                "l0_bundle_coverage",
                "analyzer-selected primary was not present in the initial prompt excerpts",
            )
        )
    decision_relevant_lines = row.get("tool_decision_relevant_tool_only_lines") or []
    if decision_relevant_lines:
        concerns.append(
            _concern(
                row,
                "l0_bundle_gap",
                f"decision-relevant tool-only lines={decision_relevant_lines}; "
                "structured-fact repeats="
                f"{row.get('tool_structured_fact_redundant_lines') or []}; "
                "incidental cited lines="
                f"{row.get('tool_incidental_tool_only_lines') or []}; unused new lines="
                f"{len(row.get('tool_unused_tool_only_lines') or [])}",
            )
        )
    return concerns


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _fallback_l0_kpis(
    *,
    summary: dict[str, Any],
    tool_efficiency: dict[str, Any],
) -> dict[str, Any]:
    trace = _trace_from_summary(summary)
    l0_bundle = _dict(trace.get("l0_bundle"))
    if not l0_bundle:
        return {}
    analysis = _dict(trace.get("analysis_result"))
    if not analysis:
        analysis = {"primary_failure": summary.get("primary_failure")}
    analyzer_trace = _dict(trace.get("analyzer_trace"))
    timing = _dict(analyzer_trace.get("timing"))
    l0_model_view = _dict(analyzer_trace.get("l0_model_view"))
    l0b_projection = _dict(l0_model_view.get("projection_metrics"))
    progress = _dict(l0_bundle.get("run_progress_summary"))
    l0_primary = _dict(l0_bundle.get("deterministic_primary_candidate"))
    progress_facts = _dict(l0_bundle.get("progress"))
    setup_markers = _list_of_dicts(progress_facts.get("setup_markers"))
    primary = _dict(analysis.get("primary_failure"))
    primary_line = _int_or_none(primary.get("line"))
    in_excerpt = _line_in_context_windows(primary_line, l0_bundle)
    in_bundle = _line_in_l0_bundle(primary_line, l0_bundle, in_excerpt)
    candidate_anchors = _list_of_dicts(l0_bundle.get("candidate_anchors"))
    progress_after = progress.get("progress_after_failure_episode")
    later_progress_after_fault_observations = _list_of_dicts(
        l0_bundle.get("later_progress_after_fault_observations")
    )
    distributed_incidents = _distributed_incident_summaries(l0_bundle)
    operation_artifact_comparisons = _list_of_dicts(l0_bundle.get("operation_artifact_comparisons"))
    return {
        "l0_wall_clock_s": timing.get("l0_wall_clock_s"),
        "l0a_wall_clock_s": timing.get("l0a_wall_clock_s"),
        "decision_evidence_wall_clock_s": timing.get("decision_evidence_wall_clock_s"),
        "l0b_wall_clock_s": timing.get("l0b_wall_clock_s"),
        "l0b_schema_version": l0_model_view.get("schema_version"),
        "l0b_projection_metrics": l0b_projection,
        "line_count": l0_bundle.get("line_count"),
        "byte_size": l0_bundle.get("byte_size"),
        "context_window_count": len(_list_of_dicts(l0_bundle.get("context_windows"))),
        "candidate_anchor_count": len(candidate_anchors),
        "occurrence_group_count": len(_list_of_dicts(l0_bundle.get("occurrence_groups"))),
        "failure_episode_count": len(_list_of_dicts(l0_bundle.get("failure_episodes"))),
        "distributed_failure_incident_count": len(distributed_incidents),
        "distributed_failure_incidents": distributed_incidents,
        "operation_artifact_comparisons": operation_artifact_comparisons,
        "root_fingerprint_owner": "L0",
        "root_fingerprint": l0_primary.get("root_fingerprint"),
        "root_fingerprint_source": l0_primary.get("root_fingerprint_source"),
        "root_fingerprint_available": bool(l0_primary.get("root_fingerprint")),
        "history_identity_ready": bool(l0_primary.get("root_fingerprint")),
        "exact_physical_unit_comparison_count": sum(
            1
            for item in operation_artifact_comparisons
            if item.get("comparison_level") == "exact_physical_unit"
        ),
        "selected_primary_in_bundle": _yes_no_unknown(in_bundle),
        "selected_primary_in_excerpt": _yes_no_unknown(in_excerpt),
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
        "observed_failure_phase": l0_primary.get("phase"),
        "later_progress_after_fault_observation_count": len(
            later_progress_after_fault_observations
        ),
        "later_progress_after_fault_event_count": sum(
            _int_or_zero(item.get("event_count"))
            for item in later_progress_after_fault_observations
        ),
        "setup_marker_count": len(setup_markers),
        "setup_marker_types": [marker.get("marker_type") for marker in setup_markers],
        "setup_marker_lines": [marker.get("line") for marker in setup_markers],
        "tool_calls_added_new_prompt_lines": tool_efficiency.get("new_prompt_excerpt_line_count"),
        "candidate_anchors_without_excerpt": sum(
            1 for anchor in candidate_anchors if not anchor.get("context_window_ids")
        ),
    }


def _trace_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    artifacts = _dict(summary.get("artifacts"))
    raw_path = artifacts.get("trace_json")
    if raw_path:
        trace = _read_json(Path(str(raw_path)))
        if isinstance(trace, dict):
            return trace
    return {}


def _fallback_l1_kpis(
    *,
    summary: dict[str, Any],
    timing: dict[str, Any],
    token_usage: dict[str, Any],
    token_limit: dict[str, Any],
    model_call_summary: dict[str, Any],
    tool_efficiency: dict[str, Any],
    semantic_safety: dict[str, Any],
    signals: dict[str, Any],
) -> dict[str, Any]:
    failed = _int_or_zero(model_call_summary.get("failed_calls"))
    retried = _int_or_zero(model_call_summary.get("retried_calls"))
    timeouts = _int_or_zero(model_call_summary.get("timeout_calls"))
    provider_errors = _int_or_zero(model_call_summary.get("provider_error_count"))
    issues = []
    if failed:
        issues.append("model_call_failed")
    if retried:
        issues.append("retry_used")
    if timeouts:
        issues.append("provider_timeout")
    if provider_errors:
        issues.append("provider_error")
    parsed = summary.get("l1_response_parsed", summary.get("l1_success"))
    return {
        "response_parsed": parsed,
        "execution_status": ("degraded" if parsed and issues else "ok" if parsed else "failed"),
        "execution_issues": issues,
        "wall_clock_s": timing.get("l1_wall_clock_s"),
        "model_call_wall_clock_s": timing.get("l1_model_call_wall_clock_s"),
        "tool_wall_clock_s": timing.get("l1_tool_wall_clock_s"),
        "model_calls": model_call_summary.get("calls") or summary.get("model_calls"),
        "model_turns": model_call_summary.get("model_turns"),
        "successful_model_calls": model_call_summary.get("successful_calls"),
        "failed_model_calls": model_call_summary.get("failed_calls"),
        "retried_model_calls": model_call_summary.get("retried_calls"),
        "timeout_model_calls": model_call_summary.get("timeout_calls"),
        "provider_error_count": model_call_summary.get("provider_error_count"),
        "http_error_calls": model_call_summary.get("http_error_calls"),
        "tool_calls": tool_efficiency.get("calls") or summary.get("tool_calls"),
        "tool_driven_model_turns": tool_efficiency.get("tool_driven_model_turns"),
        "extra_model_turns_after_initial": model_call_summary.get(
            "extra_model_turns_after_initial"
        ),
        "contract_repair_requested": bool(summary.get("l1_contract_repair_requested")),
        "contract_repair_turns": (1 if summary.get("l1_contract_repair_requested") else 0),
        "tool_calls_added_new_prompt_lines": tool_efficiency.get("new_prompt_excerpt_line_count"),
        "duplicate_tool_calls": tool_efficiency.get("duplicate_prompt_context_calls"),
        "no_new_prompt_line_tool_calls": tool_efficiency.get("no_new_prompt_line_calls"),
        "tool_error_calls": tool_efficiency.get("error_calls"),
        "tool_truncated_calls": tool_efficiency.get("truncated_calls"),
        "total_tokens": token_usage.get("total_tokens"),
        "token_limit_hit": token_limit.get("hit"),
        "context_efficiency": signals.get("context_efficiency"),
        "endpoint_reliability": signals.get("endpoint_reliability"),
        "semantic_safety": signals.get("semantic_safety"),
        "normalization_count": semantic_safety.get("normalization_count"),
        "errors_count": len(summary.get("errors") or []),
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
    for window in _list_of_dicts(l0_bundle.get("context_windows")):
        start = _int_or_none(window.get("start_line"))
        end = _int_or_none(window.get("end_line"))
        if start is not None and end is not None and start <= line_no <= end:
            return True
    return False


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _distributed_incident_summaries(
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


def _yes_no_unknown(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


if __name__ == "__main__":
    raise SystemExit(main())
