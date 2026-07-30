#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect the complete and model-facing L0 views in an analyzer trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

COLLECTION_FIELDS = (
    "occurrence_groups",
    "registry_matches",
    "candidate_anchors",
    "failure_episodes",
    "cascades",
    "context_windows",
)


class TraceInspectionError(ValueError):
    """Raised when a trace cannot provide the requested view."""


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        trace = _read_trace(args.trace)
        if args.view == "full-l0":
            output = _full_l0(trace)
        elif args.view == "decision-evidence":
            output = _decision_evidence(trace)
        else:
            snapshot, snapshot_count = _model_l0_snapshot(trace, args.snapshot)
            if args.view == "model-l0":
                output = snapshot
            else:
                output = _comparison(
                    trace_path=args.trace,
                    trace=trace,
                    model_l0=snapshot,
                    snapshot_index=args.snapshot,
                    snapshot_count=snapshot_count,
                )
    except (OSError, json.JSONDecodeError, TraceInspectionError) as exc:
        print(f"inspect_trace: {exc}", file=sys.stderr)
        return 2

    json.dump(output, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "View the exact model-facing L0 snapshot, the complete internal L0 "
            "bundle, shared Decision Evidence, or a comparison from a restart "
            "agent trace."
        )
    )
    parser.add_argument("trace", type=Path, help="Path to a model.*.trace.json file")
    parser.add_argument(
        "--view",
        choices=("model-l0", "full-l0", "decision-evidence", "comparison"),
        default="model-l0",
        help="View to print as JSON (default: model-l0)",
    )
    parser.add_argument(
        "--snapshot",
        type=int,
        default=1,
        help="One-based bundle_snapshot index when a trace has more than one (default: 1)",
    )
    args = parser.parse_args(argv)
    if args.snapshot < 1:
        parser.error("--snapshot must be at least 1")
    return args


def _read_trace(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        trace = json.load(stream)
    if not isinstance(trace, dict):
        raise TraceInspectionError("trace root must be a JSON object")
    return trace


def _full_l0(trace: dict[str, Any]) -> dict[str, Any]:
    bundle = trace.get("l0_bundle")
    if not isinstance(bundle, dict):
        raise TraceInspectionError("trace does not contain a top-level l0_bundle object")
    return bundle


def _decision_evidence(trace: dict[str, Any]) -> dict[str, Any]:
    analyzer_trace = trace.get("analyzer_trace")
    if not isinstance(analyzer_trace, dict):
        raise TraceInspectionError("trace does not contain analyzer_trace")
    decision_evidence = analyzer_trace.get("decision_evidence")
    if not isinstance(decision_evidence, dict):
        raise TraceInspectionError("trace does not contain analyzer_trace.decision_evidence")
    return decision_evidence


def _model_l0_snapshot(trace: dict[str, Any], snapshot_index: int) -> tuple[dict[str, Any], int]:
    analyzer_trace = trace.get("analyzer_trace")
    if not isinstance(analyzer_trace, dict):
        raise TraceInspectionError("trace does not contain analyzer_trace")

    model_view = analyzer_trace.get("l0_model_view")
    if not isinstance(model_view, dict):
        raise TraceInspectionError("trace does not contain analyzer_trace.l0_model_view")
    if snapshot_index != 1:
        raise TraceInspectionError("analyzer_trace.l0_model_view has exactly one snapshot")
    return model_view, 1


def _comparison(
    *,
    trace_path: Path,
    trace: dict[str, Any],
    model_l0: dict[str, Any],
    snapshot_index: int,
    snapshot_count: int,
) -> dict[str, Any]:
    full_l0 = _full_l0(trace)
    model_evidence = model_l0.get("evidence_bundle")
    if not isinstance(model_evidence, dict):
        raise TraceInspectionError(
            "model-facing L0 view does not contain an evidence_bundle object"
        )
    analyzer_trace = trace.get("analyzer_trace")
    l1 = analyzer_trace.get("l1") if isinstance(analyzer_trace, dict) else {}
    transcript = l1.get("interaction_transcript") if isinstance(l1, dict) else []
    event_types = [
        event.get("event_type")
        for event in transcript
        if isinstance(event, dict) and isinstance(event.get("event_type"), str)
    ]
    decision_evidence = (
        analyzer_trace.get("decision_evidence") if isinstance(analyzer_trace, dict) else None
    )
    model_decision_evidence = model_l0.get("decision_evidence")

    return {
        "trace_path": str(trace_path.resolve()),
        "schema_version": trace.get("schema_version"),
        "model": l1.get("model") if isinstance(l1, dict) else None,
        "bundle_snapshot": {
            "selected": snapshot_index,
            "count": snapshot_count,
        },
        "full_internal_l0": _bundle_stats(full_l0),
        "model_facing_l0": {
            **_bundle_stats(model_evidence),
            "schema_version": model_l0.get("schema_version"),
            "projection_metrics": model_l0.get("projection_metrics") or {},
        },
        "decision_evidence": {
            "schema_version": (
                decision_evidence.get("schema_version")
                if isinstance(decision_evidence, dict)
                else None
            ),
            "present_in_trace": isinstance(decision_evidence, dict),
            "present_in_model_view": isinstance(model_decision_evidence, dict),
            "exactly_shared": (
                isinstance(decision_evidence, dict) and decision_evidence == model_decision_evidence
            ),
        },
        "collection_counts": {
            field: {
                "full_internal": _collection_count(full_l0, field),
                "model_facing": _collection_count(model_evidence, field),
            }
            for field in COLLECTION_FIELDS
        },
        "excluded_top_level_fields": sorted(set(full_l0) - set(model_evidence)),
        "model_only_top_level_fields": sorted(set(model_evidence) - set(full_l0)),
        "full_model_request_recorded": "model_request" in event_types,
        "trace_note": (
            "l0_model_view is the exact typed L0B artifact supplied inside the "
            "initial model message. A complete request also contains the system "
            "prompt, task instructions, response schema, advertised tool schemas, "
            "and request options. Current traces do not expose those fields unless "
            "they include a model_request event."
        ),
    }


def _bundle_stats(bundle: dict[str, Any]) -> dict[str, Any]:
    windows = bundle.get("context_windows")
    if not isinstance(windows, list):
        windows = []
    line_numbers = {
        line.get("line")
        for window in windows
        if isinstance(window, dict)
        for line in _list_value(window.get("lines"))
        if isinstance(line, dict) and isinstance(line.get("line"), int)
    }
    return {
        "byte_size": bundle.get("byte_size"),
        "line_count": bundle.get("line_count"),
        "top_level_fields": sorted(bundle),
        "context_window_count": len(windows),
        "unique_excerpt_line_count": len(line_numbers),
    }


def _collection_count(bundle: dict[str, Any], field: str) -> int | None:
    value = bundle.get(field)
    return len(value) if isinstance(value, list) else None


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


if __name__ == "__main__":
    raise SystemExit(main())
