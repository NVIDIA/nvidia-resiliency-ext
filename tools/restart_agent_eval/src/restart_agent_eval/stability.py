#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure repeated-run decision stability without invoking a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .runtime import SYSTEM_CLOCK, Clock
from .schemas import PANEL_SUMMARY_SCHEMA_VERSION, require_schema

SCHEMA_VERSION = "restart_agent_eval_stability.v1"
PANEL_FILENAME = "panel_summary.json"
INDEX_FILENAME = "review_index.json"
L0_FILENAME = "l0_bundle.json"
DEFAULT_MINIMUM_SAMPLES = 10

COMPARABILITY_FIELDS = (
    "source_sha256",
    "product_commit",
    "config_fingerprint",
    "request_sha256",
    "route_profile_sha256",
    "l0a_sha256",
    "l0b_sha256",
    "initial_request_sha256",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dirs = discover_run_dirs(args.run_dirs, args.runs_root)
    if args.latest is not None:
        run_dirs = run_dirs[-args.latest :]
    if not run_dirs:
        raise SystemExit("no completed restart-agent run directories found")

    summary = build_stability_summary(
        run_dirs,
        route_filters=args.route,
        minimum_samples=args.minimum_samples,
        clock=SYSTEM_CLOCK,
    )
    if not summary["accepted_sample_count"]:
        selected = ", ".join(args.route) if args.route else "model routes"
        raise SystemExit(f"no stability samples matched: {selected}")
    output_dir = _resolve_output_dir(
        args.output_dir,
        args.runs_root,
        run_dirs,
        clock=SYSTEM_CLOCK,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "stability_summary.json"
    md_path = output_dir / "stability_summary.md"
    _write_json(json_path, summary)
    md_path.write_text(render_stability_markdown(summary), encoding="utf-8")
    print(f"stability artifacts: {output_dir}")
    print(f"json: {json_path}")
    print(f"report: {md_path}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure decision stability over completed one-log result directories; "
            "this command never invokes a model"
        )
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        help="completed one-log run directories containing panel_summary.json",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        help="discover completed runs recursively below this per-log run root",
    )
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        help="target name or exact model name to include; may be repeated",
    )
    parser.add_argument(
        "--latest",
        type=_positive_int,
        help="use only the latest N discovered run directories",
    )
    parser.add_argument(
        "--minimum-samples",
        type=_positive_int,
        default=DEFAULT_MINIMUM_SAMPLES,
        help=f"minimum comparable samples; default {DEFAULT_MINIMUM_SAMPLES}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="artifact directory; defaults to <common-run-root>/stability/<timestamp>",
    )
    return parser.parse_args(argv)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def discover_run_dirs(explicit: Sequence[Path], runs_root: Path | None) -> list[Path]:
    discovered: set[Path] = set()
    for raw_path in explicit:
        path = raw_path.expanduser().resolve()
        if not (path / PANEL_FILENAME).is_file():
            raise SystemExit(f"run directory is missing {PANEL_FILENAME}: {path}")
        discovered.add(path)
    if runs_root is not None:
        root = runs_root.expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"--runs-root is not a directory: {root}")
        for panel_path in root.rglob(PANEL_FILENAME):
            discovered.add(panel_path.parent.resolve())
    return sorted(discovered, key=_run_sort_key)


def build_stability_summary(
    run_dirs: Sequence[Path],
    *,
    route_filters: Sequence[str] = (),
    minimum_samples: int = DEFAULT_MINIMUM_SAMPLES,
    clock: Clock = SYSTEM_CLOCK,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    rejected_runs: list[dict[str, str]] = []
    filters = frozenset(route_filters)

    for run_dir in run_dirs:
        try:
            loaded = _load_run_samples(run_dir, filters)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            rejected_runs.append({"run_dir": str(run_dir), "reason": str(exc)})
            continue
        samples.extend(loaded)

    cohorts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        cohorts[_cohort_id(sample)].append(sample)

    cohort_summaries = [
        _summarize_cohort(cohort_id, cohort_samples, minimum_samples)
        for cohort_id, cohort_samples in sorted(cohorts.items())
    ]
    source_values = sorted(
        {sample.get("source_sha256") for sample in samples if sample.get("source_sha256")}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": clock.now_utc().isoformat(),
        "minimum_samples": minimum_samples,
        "input_run_count": len(run_dirs),
        "accepted_sample_count": len(samples),
        "rejected_runs": rejected_runs,
        "route_filters": list(route_filters),
        "source_sha256_values": source_values,
        "cohort_count": len(cohort_summaries),
        "cohorts": cohort_summaries,
        "interpretation": {
            "stability_is_accuracy": False,
            "promotion_verdict_emitted": False,
            "notes": [
                "Stability is measured only within an exact comparability cohort.",
                "Gold accuracy and repeated-run stability are independent dimensions.",
                "Endpoint failures are retained separately from usable L1 semantic outputs.",
            ],
        },
    }


def _load_run_samples(run_dir: Path, route_filters: frozenset[str]) -> list[dict[str, Any]]:
    context = StabilityRunContext.read(run_dir)
    return [
        sample
        for row in context.rows
        if (sample := _sample_from_row(context, row, route_filters)) is not None
    ]


@dataclass(frozen=True)
class StabilityRunContext:
    """Run-level comparability metadata shared by every route sample."""

    run_dir: Path
    manifest: Mapping[str, Any]
    source_sha256: str | None
    product_commit: str | None
    product_dirty: bool | None
    config_fingerprint: str | None
    route_configs: Mapping[str, Mapping[str, Any]]
    l0a_sha256: str | None
    created_at: str
    rows: tuple[Mapping[str, Any], ...]

    @classmethod
    def read(cls, run_dir: Path) -> "StabilityRunContext":
        panel = _read_json(run_dir / PANEL_FILENAME)
        require_schema(panel, PANEL_SUMMARY_SCHEMA_VERSION, artifact=str(run_dir / PANEL_FILENAME))
        index_path = run_dir / INDEX_FILENAME
        index = _read_json(index_path) if index_path.is_file() else {}
        manifest = panel.get("run_manifest") or index.get("run_manifest") or {}
        product = _nested(manifest, "repositories", "product") or {}
        config = panel.get("restart_agent_config") or {}
        effective_config = config.get("effective_config") or {}
        routes = effective_config.get("model_routes") or []
        route_configs = {
            str(route.get("route_id")): route
            for route in routes
            if isinstance(route, Mapping) and route.get("route_id")
        }
        rows = panel.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"{run_dir}: panel rows are missing")
        l0_path = run_dir / L0_FILENAME
        return cls(
            run_dir=run_dir,
            manifest=manifest,
            source_sha256=(
                panel.get("source_log_sha256")
                or index.get("source_log_sha256")
                or _nested(manifest, "source", "sha256")
            ),
            product_commit=(
                panel.get("product_commit") or index.get("product_commit") or product.get("commit")
            ),
            product_dirty=product.get("dirty"),
            config_fingerprint=config.get("config_fingerprint"),
            route_configs=route_configs,
            l0a_sha256=_file_sha256(l0_path) if l0_path.is_file() else None,
            created_at=str(
                manifest.get("created_at_utc") or manifest.get("run_id") or run_dir.name
            ),
            rows=tuple(row for row in rows if isinstance(row, Mapping)),
        )


def _sample_from_row(
    context: StabilityRunContext,
    row: Mapping[str, Any],
    route_filters: frozenset[str],
) -> dict[str, Any] | None:
    if not row.get("model"):
        return None
    target = str(row.get("target") or "")
    model = str(row.get("model") or "")
    if route_filters and target not in route_filters and model not in route_filters:
        return None

    trace_path = _resolve_trace_path(context.run_dir, row)
    trace = _read_json(trace_path) if trace_path is not None else {}
    analysis_result = trace.get("analysis_result") or {}
    analyzer_trace = trace.get("analyzer_trace") or {}
    l1_trace = analyzer_trace.get("l1") or {}
    assessment = analysis_result.get("model_recovery_assessment") or {}
    if not isinstance(assessment, Mapping):
        assessment = {}
    policy_fields, policy_contract = _policy_fields(assessment)
    primary = analysis_result.get("primary_failure") or {}
    root_cause = analysis_result.get("root_cause_assessment") or {}
    route_config = context.route_configs.get(target) or _route_config_fallback(row)
    l0b_sha256 = _nested(
        row,
        "l0b_projection_metrics",
        "projection_integrity",
        "deterministic_payload_sha256",
    ) or _nested(
        analyzer_trace,
        "l0_model_view",
        "projection_metrics",
        "projection_integrity",
        "deterministic_payload_sha256",
    )
    l1_usable = row.get("l1_output_usable")
    if l1_usable is None:
        l1_usable = bool(l1_trace.get("success") and assessment)

    return {
        "run_dir": str(context.run_dir),
        "run_id": context.manifest.get("run_id") or context.run_dir.name,
        "created_at_utc": context.created_at,
        "target": target,
        "model": model,
        "source_sha256": context.source_sha256,
        "product_commit": context.product_commit,
        "product_dirty": context.product_dirty,
        "config_fingerprint": context.config_fingerprint,
        "request_sha256": _canonical_sha256(trace.get("request")),
        "route_profile_sha256": _canonical_sha256(route_config),
        "l0a_sha256": context.l0a_sha256,
        "l0b_sha256": l0b_sha256,
        "initial_request_sha256": _first_request_sha256(l1_trace),
        "l1_usable": bool(l1_usable),
        "l1_execution_status": row.get("l1_execution_status"),
        "decision": row.get("decision"),
        "decision_basis": row.get("decision_basis"),
        "policy_contract": policy_contract,
        "policy_fields": policy_fields,
        "policy_tuple_sha256": (_canonical_sha256(policy_fields) if policy_fields else None),
        "failure_domain_confidence": _claim_field(
            assessment,
            "failure_domain",
            "confidence",
        ),
        "retry_outlook_confidence": _claim_field(
            assessment,
            "retry_outlook_without_workload_change",
            "confidence",
        ),
        "root_cause_status": root_cause.get("status"),
        "primary_class": primary.get("fine_class") or row.get("l1_semantic_primary_class"),
        "primary_line": primary.get("line") or row.get("l1_semantic_primary_line"),
        "root_fingerprint": row.get("current_root_fingerprint") or row.get("l2_root_fingerprint"),
        "model_calls": row.get("model_calls"),
        "tool_calls": row.get("tool_calls"),
        "tool_sequence": _tool_sequence(l1_trace),
        "no_new_tool_calls": row.get("no_new_prompt_line_calls"),
        "l1_wall_clock_s": row.get("l1_wall_clock_s"),
        "total_tokens": row.get("total_tokens"),
        "endpoint_reliability": row.get("endpoint_reliability"),
        "failed_endpoint_attempts": row.get("failed_endpoint_attempts"),
        "retried_model_calls": row.get("retried_model_calls"),
        "timeout_model_calls": row.get("timeout_model_calls"),
        "gold_l1_core_semantic": row.get("gold_l1_core_semantic"),
        "gold_l4_action_correct": row.get("gold_l4_action_correct"),
        "gold_case_id": row.get("gold_case_id"),
    }


def _policy_fields(assessment: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    fields = {
        "failure_domain": _claim_field(assessment, "failure_domain", "value"),
        "failure_domain_status": _claim_field(assessment, "failure_domain", "status"),
        "retry_outlook_without_workload_change": _claim_field(
            assessment,
            "retry_outlook_without_workload_change",
            "value",
        ),
        "retry_outlook_status": _claim_field(
            assessment,
            "retry_outlook_without_workload_change",
            "status",
        ),
    }
    if all(value is not None for value in fields.values()):
        return fields, "current"
    return {}, "unavailable"


def _claim_field(assessment: Mapping[str, Any], claim_name: str, field: str) -> Any:
    claim = assessment.get(claim_name) or {}
    return claim.get(field) if isinstance(claim, Mapping) else None


def _summarize_cohort(
    cohort_id: str,
    samples: Sequence[dict[str, Any]],
    minimum_samples: int,
) -> dict[str, Any]:
    ordered = sorted(samples, key=lambda item: (str(item.get("created_at_utc")), item["run_id"]))
    usable = [
        sample for sample in ordered if sample.get("l1_usable") and sample.get("policy_fields")
    ]
    decisions = [sample.get("decision") for sample in ordered if sample.get("decision")]
    policy_tuples = [sample.get("policy_tuple_sha256") for sample in usable]
    missing_comparability = sorted(
        {field for sample in ordered for field in COMPARABILITY_FIELDS if not sample.get(field)}
    )
    product_dirty = any(sample.get("product_dirty") is True for sample in ordered)
    comparability_status = "verified"
    if missing_comparability:
        comparability_status = "incomplete"
    elif product_dirty:
        comparability_status = "provisional_dirty_product"

    decision_metrics = _categorical_metrics(decisions)
    decision_metrics["sequential_flips"] = _sequential_flips(decisions)
    decision_metrics["sequential_flip_rate"] = _ratio(
        decision_metrics["sequential_flips"], max(len(decisions) - 1, 0)
    )
    policy_metrics = _categorical_metrics(policy_tuples)
    policy_metrics["sequential_flips"] = _sequential_flips(policy_tuples)
    policy_metrics["sequential_flip_rate"] = _ratio(
        policy_metrics["sequential_flips"], max(len(policy_tuples) - 1, 0)
    )

    status = "insufficient_samples"
    if len(ordered) >= minimum_samples:
        if comparability_status == "incomplete":
            status = "comparability_incomplete"
        elif (
            len(usable) != len(ordered)
            or decision_metrics.get("modal_agreement") != 1.0
            or policy_metrics.get("modal_agreement") != 1.0
        ):
            status = "observed_unstable"
        else:
            status = "observed_stable"

    field_names = sorted({field for sample in usable for field in sample["policy_fields"]})
    field_stability = {
        field: _categorical_metrics([sample["policy_fields"].get(field) for sample in usable])
        for field in field_names
    }
    identity = {field: ordered[0].get(field) for field in COMPARABILITY_FIELDS}
    identity.update(
        {
            "target": ordered[0].get("target"),
            "model": ordered[0].get("model"),
            "product_dirty": product_dirty,
            "config_fingerprints": sorted(
                {
                    sample.get("config_fingerprint")
                    for sample in ordered
                    if sample.get("config_fingerprint")
                }
            ),
        }
    )
    return {
        "cohort_id": cohort_id,
        "target": ordered[0].get("target"),
        "model": ordered[0].get("model"),
        "sample_count": len(ordered),
        "minimum_samples": minimum_samples,
        "status": status,
        "comparability": {
            "status": comparability_status,
            "missing_fields": missing_comparability,
            "identity": identity,
        },
        "availability": {
            "usable_l1_count": len(usable),
            "usable_l1_rate": _ratio(len(usable), len(ordered)),
            "unusable_l1_count": len(ordered) - len(usable),
        },
        "decision_stability": decision_metrics,
        "semantic_stability": {
            "policy_contracts": _distribution([sample.get("policy_contract") for sample in usable]),
            "exact_policy_tuple": policy_metrics,
            "fields": field_stability,
            "root_cause_status": _categorical_metrics(
                [sample.get("root_cause_status") for sample in usable]
            ),
            "failure_domain_confidence": _numeric_metrics(
                [sample.get("failure_domain_confidence") for sample in usable]
            ),
            "retry_outlook_confidence": _numeric_metrics(
                [sample.get("retry_outlook_confidence") for sample in usable]
            ),
        },
        "primary_and_identity_stability": {
            "primary_class": _categorical_metrics(
                [sample.get("primary_class") for sample in usable]
            ),
            "primary_line": _categorical_metrics([sample.get("primary_line") for sample in usable]),
            "root_fingerprint": _categorical_metrics(
                [sample.get("root_fingerprint") for sample in usable]
            ),
        },
        "behavioral_variability": {
            "model_calls": _numeric_metrics([sample.get("model_calls") for sample in ordered]),
            "tool_calls": _numeric_metrics([sample.get("tool_calls") for sample in ordered]),
            "no_new_tool_calls": _numeric_metrics(
                [sample.get("no_new_tool_calls") for sample in ordered]
            ),
            "tool_sequences": _distribution(
                [sample.get("tool_sequence") or [] for sample in ordered]
            ),
            "l1_wall_clock_s": _numeric_metrics(
                [sample.get("l1_wall_clock_s") for sample in ordered]
            ),
            "total_tokens": _numeric_metrics([sample.get("total_tokens") for sample in ordered]),
        },
        "endpoint_reliability": {
            "status_distribution": _distribution(
                [sample.get("endpoint_reliability") for sample in ordered]
            ),
            "failed_endpoint_attempts": _numeric_metrics(
                [sample.get("failed_endpoint_attempts") for sample in ordered]
            ),
            "retried_model_calls": _numeric_metrics(
                [sample.get("retried_model_calls") for sample in ordered]
            ),
            "timeout_model_calls": _numeric_metrics(
                [sample.get("timeout_model_calls") for sample in ordered]
            ),
        },
        "gold_accuracy": {
            "case_ids": sorted(
                {sample.get("gold_case_id") for sample in ordered if sample.get("gold_case_id")}
            ),
            "l1_core_semantic": _boolean_score(
                [sample.get("gold_l1_core_semantic") for sample in ordered]
            ),
            "l4_action": _boolean_score(
                [sample.get("gold_l4_action_correct") for sample in ordered]
            ),
        },
        "samples": ordered,
    }


def render_stability_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Decision Stability",
        "",
        (
            f"Runs: `{summary.get('input_run_count')}` | Samples: "
            f"`{summary.get('accepted_sample_count')}` | Cohorts: "
            f"`{summary.get('cohort_count')}` | Minimum samples: "
            f"`{summary.get('minimum_samples')}`"
        ),
        "",
        "Stability and correctness are independent. This report never emits a promotion verdict.",
        "",
    ]
    rejected = summary.get("rejected_runs") or []
    if rejected:
        lines.extend(["## Rejected Runs", ""])
        for item in rejected:
            lines.append(f"- `{item.get('run_dir')}`: {item.get('reason')}")
        lines.append("")

    for cohort in summary.get("cohorts") or []:
        lines.extend(_render_cohort(cohort))
    return "\n".join(lines).rstrip() + "\n"


def _render_cohort(cohort: Mapping[str, Any]) -> list[str]:
    decision = cohort.get("decision_stability") or {}
    semantic = cohort.get("semantic_stability") or {}
    exact_policy = semantic.get("exact_policy_tuple") or {}
    availability = cohort.get("availability") or {}
    comparability = cohort.get("comparability") or {}
    identity = comparability.get("identity") or {}
    lines = [
        f"## {cohort.get('target')} - {cohort.get('model')}",
        "",
        (
            f"Status: `{cohort.get('status')}` | Samples: `{cohort.get('sample_count')}` | "
            f"Comparability: `{comparability.get('status')}` | Cohort: `{cohort.get('cohort_id')}`"
        ),
        "",
        "### Input Identity",
        "",
        "| identity | value |",
        "|---|---|",
        f"| source | `{_short(identity.get('source_sha256'))}` |",
        f"| product | `{_short(identity.get('product_commit'))}` |",
        f"| analyzer config | `{_short(identity.get('config_fingerprint'))}` |",
        f"| route profile | `{_short(identity.get('route_profile_sha256'))}` |",
        f"| L0A | `{_short(identity.get('l0a_sha256'))}` |",
        f"| L0B | `{_short(identity.get('l0b_sha256'))}` |",
        f"| initial request | `{_short(identity.get('initial_request_sha256'))}` |",
        "",
    ]
    if comparability.get("missing_fields"):
        lines.append(
            "Missing comparability fields: "
            + ", ".join(f"`{field}`" for field in comparability["missing_fields"])
        )
        lines.append("")
    if identity.get("product_dirty"):
        lines.extend(
            [
                "The product checkout was dirty; this cohort is provisional even when request hashes match.",
                "",
            ]
        )

    lines.extend(
        [
            "### Stability Headline",
            "",
            "| measure | result |",
            "|---|---|",
            (
                f"| usable L1 responses | {availability.get('usable_l1_count')}/"
                f"{cohort.get('sample_count')} ({_pct(availability.get('usable_l1_rate'))}) |"
            ),
            (
                f"| final action agreement | {_pct(decision.get('modal_agreement'))}; "
                f"{_format_distribution(decision.get('distribution'))} |"
            ),
            (
                f"| final action sequential flips | {decision.get('sequential_flips')}/"
                f"{max(int(decision.get('count') or 0) - 1, 0)} "
                f"({_pct(decision.get('sequential_flip_rate'))}) |"
            ),
            (f"| exact L1 policy-input agreement | {_pct(exact_policy.get('modal_agreement'))} |"),
            (
                f"| L1 policy-input sequential flips | {exact_policy.get('sequential_flips')}/"
                f"{max(int(exact_policy.get('count') or 0) - 1, 0)} "
                f"({_pct(exact_policy.get('sequential_flip_rate'))}) |"
            ),
            "",
            "### L1 Field Stability",
            "",
            "| field | modal agreement | values |",
            "|---|---:|---|",
        ]
    )
    fields = semantic.get("fields") or {}
    if fields:
        for field, metrics in fields.items():
            lines.append(
                f"| `{field}` | {_pct(metrics.get('modal_agreement'))} | "
                f"{_format_distribution(metrics.get('distribution'))} |"
            )
    else:
        lines.append("| _no usable policy fields_ | n/a | n/a |")

    primary = cohort.get("primary_and_identity_stability") or {}
    behavior = cohort.get("behavioral_variability") or {}
    endpoint = cohort.get("endpoint_reliability") or {}
    accuracy = cohort.get("gold_accuracy") or {}
    lines.extend(
        [
            "",
            "### Primary And Identity",
            "",
            "| measure | modal agreement | values |",
            "|---|---:|---|",
        ]
    )
    for label, key in (
        ("primary class", "primary_class"),
        ("primary line", "primary_line"),
        ("root fingerprint", "root_fingerprint"),
    ):
        metrics = primary.get(key) or {}
        lines.append(
            f"| {label} | {_pct(metrics.get('modal_agreement'))} | "
            f"{_format_distribution(metrics.get('distribution'), shorten=True)} |"
        )

    lines.extend(
        [
            "",
            "### Behavior And Endpoint",
            "",
            "| measure | result |",
            "|---|---|",
            f"| L1 latency | {_format_numeric(behavior.get('l1_wall_clock_s'))} |",
            f"| total tokens | {_format_numeric(behavior.get('total_tokens'))} |",
            f"| model calls | {_format_numeric(behavior.get('model_calls'))} |",
            f"| tool calls | {_format_numeric(behavior.get('tool_calls'))} |",
            (
                f"| endpoint status | "
                f"{_format_distribution(endpoint.get('status_distribution'))} |"
            ),
            "",
            "### Gold Accuracy",
            "",
            "| measure | scored | pass rate |",
            "|---|---:|---:|",
            _accuracy_row("L1 core semantics", accuracy.get("l1_core_semantic")),
            _accuracy_row("L4 action", accuracy.get("l4_action")),
            "",
            "### Samples",
            "",
            "| run | L1 usable | action | policy tuple | primary | tools | L1 sec | endpoint |",
            "|---|---:|---|---|---|---:|---:|---|",
        ]
    )
    for sample in cohort.get("samples") or []:
        primary_value = f"{sample.get('primary_class')}@{sample.get('primary_line')}"
        lines.append(
            f"| `{sample.get('run_id')}` | {sample.get('l1_usable')} | "
            f"{sample.get('decision')} | {_short(sample.get('policy_tuple_sha256'))} | "
            f"{primary_value} | {sample.get('tool_calls')} | "
            f"{sample.get('l1_wall_clock_s')} | {sample.get('endpoint_reliability')} |"
        )
    lines.append("")
    return lines


def _cohort_id(sample: Mapping[str, Any]) -> str:
    identity = {
        "target": sample.get("target"),
        "model": sample.get("model"),
        **{field: sample.get(field) for field in COMPARABILITY_FIELDS},
    }
    return _canonical_sha256(identity).split(":", 1)[1][:12]


def _route_config_fallback(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "model",
        "tools_enabled",
        "tool_profile_id",
        "max_tool_rounds",
        "max_model_turns",
    )
    return {key: row.get(key) for key in keys}


def _resolve_trace_path(run_dir: Path, row: Mapping[str, Any]) -> Path | None:
    raw_path = _nested(row, "artifacts", "trace_json")
    if raw_path:
        path = Path(str(raw_path))
        if path.is_file():
            return path
        local = run_dir / path.name
        if local.is_file():
            return local
    target = str(row.get("target") or "")
    candidates = sorted(run_dir.glob("*.trace.json"))
    for path in candidates:
        if target and target in path.name:
            return path
    return None


def _first_request_sha256(l1_trace: Mapping[str, Any]) -> str | None:
    transcript = l1_trace.get("interaction_transcript") or []
    for event in transcript:
        if isinstance(event, Mapping) and event.get("event_type") == "model_request":
            value = event.get("payload_sha256")
            return str(value) if value else None
    return None


def _tool_sequence(l1_trace: Mapping[str, Any]) -> list[str]:
    transcript = l1_trace.get("interaction_transcript") or []
    return [
        str(event.get("name"))
        for event in transcript
        if isinstance(event, Mapping)
        and event.get("event_type") == "tool_result"
        and event.get("name")
    ]


def _categorical_metrics(values: Iterable[Any]) -> dict[str, Any]:
    materialized = [value for value in values if value is not None]
    distribution = _distribution(materialized)
    modal_count = max(distribution.values(), default=0)
    return {
        "count": len(materialized),
        "unique_count": len(distribution),
        "distribution": distribution,
        "modal_agreement": _ratio(modal_count, len(materialized)),
    }


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    counter = Counter(_display_value(value) for value in values if value is not None)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _numeric_metrics(values: Iterable[Any]) -> dict[str, Any]:
    numbers = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not numbers:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None}
    ordered = sorted(numbers)
    return {
        "count": len(ordered),
        "min": _clean_number(ordered[0]),
        "p50": _clean_number(statistics.median(ordered)),
        "p90": _clean_number(_nearest_rank(ordered, 0.90)),
        "max": _clean_number(ordered[-1]),
    }


def _boolean_score(values: Iterable[Any]) -> dict[str, Any]:
    scored = [value for value in values if isinstance(value, bool)]
    passed = sum(1 for value in scored if value)
    return {
        "scored_count": len(scored),
        "pass_count": passed,
        "pass_rate": _ratio(passed, len(scored)),
    }


def _sequential_flips(values: Sequence[Any]) -> int:
    return sum(left != right for left, right in zip(values, values[1:]))


def _nearest_rank(ordered: Sequence[float], percentile: float) -> float:
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _clean_number(value: float) -> int | float:
    return int(value) if value.is_integer() else round(value, 3)


def _display_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _canonical_sha256(value: Any) -> str | None:
    if value is None:
        return None
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_sort_key(path: Path) -> tuple[str, str]:
    try:
        panel = _read_json(path / PANEL_FILENAME)
        manifest = panel.get("run_manifest") or {}
        created = str(manifest.get("created_at_utc") or manifest.get("run_id") or path.name)
    except (OSError, ValueError, json.JSONDecodeError):
        created = path.name
    return created, str(path)


def _resolve_output_dir(
    requested: Path | None,
    runs_root: Path | None,
    run_dirs: Sequence[Path],
    *,
    clock: Clock = SYSTEM_CLOCK,
) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    if runs_root is not None:
        base = runs_root.expanduser().resolve()
    else:
        common = Path(os.path.commonpath([str(path) for path in run_dirs]))
        base = common.parent if (common / PANEL_FILENAME).is_file() else common
    stamp = clock.now_utc().strftime("%Y%m%dT%H%M%S%fZ")
    return base / "stability" / stamp


def _short(value: Any, length: int = 12) -> str:
    if value is None:
        return "n/a"
    text = str(value)
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    return text if len(text) <= length else text[:length]


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.1f}%"


def _format_distribution(value: Any, *, shorten: bool = False) -> str:
    if not isinstance(value, Mapping) or not value:
        return "n/a"
    parts = []
    for key, count in value.items():
        label = _short(key, 24) if shorten else str(key)
        parts.append(f"`{label}`={count}")
    return ", ".join(parts)


def _format_numeric(value: Any) -> str:
    if not isinstance(value, Mapping) or not value.get("count"):
        return "n/a"
    return (
        f"p50={value.get('p50')}, p90={value.get('p90')}, "
        f"range={value.get('min')}..{value.get('max')}"
    )


def _accuracy_row(label: str, value: Any) -> str:
    value = value if isinstance(value, Mapping) else {}
    return f"| {label} | {value.get('scored_count', 0)} | {_pct(value.get('pass_rate'))} |"


if __name__ == "__main__":
    raise SystemExit(main())
