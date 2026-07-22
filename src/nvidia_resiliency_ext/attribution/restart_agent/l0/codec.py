# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned JSON persistence for deterministic L0 bundle replay."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from ..infrastructure.artifact_io import write_json_atomic
from ..models import (
    CandidateAnchor,
    CascadeEvidence,
    ContextWindow,
    DistributedFailureIncident,
    FailureEpisode,
    FailureEvidence,
    JobMetadata,
    L0Bundle,
    LaterProgressAfterFaultObservation,
    LogLine,
    NormalizedOccurrenceGroup,
    OperationArtifactComparisonEvidence,
    PostFaultSummary,
    ProgressFacts,
    ProgressMarker,
    RunProgressSummary,
)

BUNDLE_RECORD_SCHEMA_VERSION = "restart_agent_l0_bundle.v1"


def write_l0_bundle(path: str | Path, bundle: L0Bundle) -> None:
    source = Path(bundle.log_path)
    stat = source.stat()
    payload = {
        "schema_version": BUNDLE_RECORD_SCHEMA_VERSION,
        "source": {
            "log_path": bundle.log_path,
            "byte_size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "bundle": asdict(bundle),
    }
    write_json_atomic(path, payload)


def read_l0_bundle(path: str | Path, *, expected_log_path: str) -> L0Bundle:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("L0 bundle record must be an object")
    if payload.get("schema_version") != BUNDLE_RECORD_SCHEMA_VERSION:
        raise ValueError("L0 bundle record schema_version is invalid")
    source = _mapping(payload.get("source"), "source")
    if str(source.get("log_path")) != expected_log_path:
        raise ValueError("L0 bundle source log_path does not match runtime input")
    stat = Path(expected_log_path).stat()
    if int(source.get("byte_size", -1)) != stat.st_size:
        raise ValueError("source log byte size changed after L0 bundle construction")
    if int(source.get("mtime_ns", -1)) != stat.st_mtime_ns:
        raise ValueError("source log mtime changed after L0 bundle construction")
    return _bundle(_mapping(payload.get("bundle"), "bundle"))


def _bundle(value: Mapping[str, Any]) -> L0Bundle:
    return L0Bundle(
        log_path=str(value["log_path"]),
        byte_size=int(value["byte_size"]),
        line_count=int(value["line_count"]),
        path_hints=tuple(value.get("path_hints") or ()),
        path_access_facts=tuple(
            dict(_mapping(item, "path_access_fact"))
            for item in value.get("path_access_facts") or ()
        ),
        path_namespace_summary=dict(value.get("path_namespace_summary") or {}),
        occurrence_groups=tuple(
            _occurrence_group(item) for item in value.get("occurrence_groups") or ()
        ),
        context_windows=tuple(_window(item) for item in value.get("context_windows") or ()),
        candidate_anchors=tuple(_anchor(item) for item in value.get("candidate_anchors") or ()),
        registry_matches=tuple(_failure(item) for item in value.get("registry_matches") or ()),
        deterministic_primary_candidate=(
            _failure(value["deterministic_primary_candidate"])
            if value.get("deterministic_primary_candidate")
            else None
        ),
        cascades=tuple(_cascade(item) for item in value.get("cascades") or ()),
        cause_confirmations=tuple(
            _failure(item) for item in value.get("cause_confirmations") or ()
        ),
        failure_episodes=tuple(_episode(item) for item in value.get("failure_episodes") or ()),
        distributed_failure_incidents=tuple(
            _distributed_incident(item) for item in value.get("distributed_failure_incidents") or ()
        ),
        post_fault_summaries=tuple(
            _post_fault(item) for item in value.get("post_fault_summaries") or ()
        ),
        progress=_progress(_mapping(value.get("progress") or {}, "progress")),
        run_progress_summary=RunProgressSummary(
            **_mapping(value.get("run_progress_summary") or {}, "run_progress_summary")
        ),
        operation_artifact_comparisons=tuple(
            _operation_artifact_comparison(item)
            for item in value.get("operation_artifact_comparisons") or ()
        ),
        later_progress_after_fault_observations=tuple(
            _later_progress_after_fault_observation(item)
            for item in value.get("later_progress_after_fault_observations") or ()
        ),
        job_metadata=JobMetadata(**_mapping(value.get("job_metadata") or {}, "job_metadata")),
        evidence_coverage=dict(value.get("evidence_coverage") or {}),
        selection_summary=dict(value.get("selection_summary") or {}),
        anomalies=dict(value.get("anomalies") or {}),
    )


def _failure(value: Any) -> FailureEvidence:
    return FailureEvidence(**_mapping(value, "failure"))


def _cascade(value: Any) -> CascadeEvidence:
    payload = dict(_mapping(value, "cascade"))
    for field in ("sample_lines", "rank_spread", "node_spread", "gpu_spread"):
        payload[field] = tuple(payload.get(field) or ())
    return CascadeEvidence(**payload)


def _occurrence_group(value: Any) -> NormalizedOccurrenceGroup:
    payload = dict(_mapping(value, "normalized_occurrence_group"))
    for field in ("sample_lines", "rank_spread", "node_spread", "gpu_spread"):
        payload[field] = tuple(payload.get(field) or ())
    return NormalizedOccurrenceGroup(**payload)


def _window(value: Any) -> ContextWindow:
    payload = dict(_mapping(value, "context_window"))
    payload["seed_lines"] = tuple(payload.get("seed_lines") or ())
    payload["occurrence_group_ids"] = tuple(payload.get("occurrence_group_ids") or ())
    payload["lines"] = tuple(
        LogLine(**_mapping(item, "log_line")) for item in payload.get("lines") or ()
    )
    return ContextWindow(**payload)


def _anchor(value: Any) -> CandidateAnchor:
    payload = dict(_mapping(value, "candidate_anchor"))
    payload["sources"] = tuple(payload.get("sources") or ())
    payload["context_window_ids"] = tuple(payload.get("context_window_ids") or ())
    for field in (
        "taxonomy_match",
        "first_downstream_registry_match",
        "first_downstream_cascade",
    ):
        payload[field] = _failure(payload[field]) if payload.get(field) else None
    return CandidateAnchor(**payload)


def _marker(value: Any) -> ProgressMarker:
    payload = dict(_mapping(value, "progress_marker"))
    payload["secondary_value"] = dict(payload.get("secondary_value") or {})
    return ProgressMarker(**payload)


def _episode(value: Any) -> FailureEpisode:
    payload = dict(_mapping(value, "failure_episode"))
    payload["last_progress_before"] = (
        _marker(payload["last_progress_before"]) if payload.get("last_progress_before") else None
    )
    payload["first_progress_after"] = (
        _marker(payload["first_progress_after"]) if payload.get("first_progress_after") else None
    )
    payload["first_downstream_cascade"] = (
        _failure(payload["first_downstream_cascade"])
        if payload.get("first_downstream_cascade")
        else None
    )
    payload["cause_confirmations"] = tuple(
        _failure(item) for item in payload.get("cause_confirmations") or ()
    )
    payload["context_window_ids"] = tuple(payload.get("context_window_ids") or ())
    for field in (
        "precursor_lines",
        "exception_chain_lines",
        "duplicate_rendering_lines",
        "wrapper_exception_lines",
    ):
        payload[field] = tuple(payload.get(field) or ())
    return FailureEpisode(**payload)


def _distributed_incident(value: Any) -> DistributedFailureIncident:
    payload = dict(_mapping(value, "distributed_failure_incident"))
    for field in (
        "member_event_lines",
        "sample_lines",
        "operation_types",
        "operation_signatures",
        "rank_spread",
        "process_group_types",
    ):
        payload[field] = tuple(payload.get(field) or ())
    return DistributedFailureIncident(**payload)


def _later_progress_after_fault_observation(
    value: Any,
) -> LaterProgressAfterFaultObservation:
    payload = dict(_mapping(value, "later_progress_after_fault_observation"))
    payload["sample_event_lines"] = tuple(payload.get("sample_event_lines") or ())
    payload["sample_later_progress_lines"] = tuple(payload.get("sample_later_progress_lines") or ())
    return LaterProgressAfterFaultObservation(**payload)


def _operation_artifact_comparison(value: Any) -> OperationArtifactComparisonEvidence:
    payload = dict(_mapping(value, "operation_artifact_comparison"))
    for field in (
        "success_logical_artifact_ids",
        "success_physical_unit_ids",
        "success_data_regions",
        "success_integrity_markers",
        "success_lines",
        "successful_observer_ranks",
        "failed_observer_ranks",
    ):
        payload[field] = tuple(payload.get(field) or ())
    payload["comparison_counts"] = dict(payload.get("comparison_counts") or {})
    return OperationArtifactComparisonEvidence(**payload)


def _post_fault(value: Any) -> PostFaultSummary:
    payload = dict(_mapping(value, "post_fault_summary"))
    payload["later_matching_exception_lines"] = tuple(
        payload.get("later_matching_exception_lines") or ()
    )
    return PostFaultSummary(**payload)


def _progress(value: Mapping[str, Any]) -> ProgressFacts:
    payload = dict(value)
    for field in (
        "progress_lines",
        "checkpoint_lines",
        "setup_lines",
        "recovery_lines",
    ):
        payload[field] = tuple(payload.get(field) or ())
    payload["progress_markers"] = tuple(
        _marker(item) for item in payload.get("progress_markers") or ()
    )
    payload["checkpoint_markers"] = tuple(
        _marker(item) for item in payload.get("checkpoint_markers") or ()
    )
    payload["setup_markers"] = tuple(_marker(item) for item in payload.get("setup_markers") or ())
    return ProgressFacts(**payload)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value
