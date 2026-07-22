# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic L0B projection for the initial model-facing evidence view."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from ..identity import normalized_pattern
from ..models import (
    DEFAULT_RESTART_ENVIRONMENT_CONTEXT,
    CausalRole,
    DecisionEvidence,
    L0Bundle,
    L0ModelFacingView,
)
from .registry import diagnostic_context_kind, diagnostic_uncertainty_kind

PROMPT_PATTERN_MAX_CHARS = 240
PROMPT_QUOTE_MAX_CHARS = 500
PROMPT_CONTEXT_LINE_MAX_CHARS = 360
PROMPT_CONTEXT_PREVIEW_LINES = 8
PROMPT_CONTEXT_MAX_WINDOWS = 4
PROMPT_CONTEXT_EXCERPT_MAX_LINES = 240
PROMPT_CONTEXT_EXCERPT_MAX_CHARS = 50_000
PROMPT_CONTEXT_MERGE_GAP_LINES = 5
PROMPT_CONTEXT_HEAD_LINES = 3
PROMPT_CONTEXT_TAIL_LINES = 2
PROMPT_CONTEXT_HIGH_SIGNAL_LINES = 3
PROMPT_OCCURRENCE_GROUP_MAX = 30
PROMPT_REGISTRY_CANDIDATE_GROUP_MAX = 20
PROMPT_CANDIDATE_ANCHOR_MAX = 20
PROMPT_FAILURE_EPISODE_MAX = 10
PROMPT_DISTRIBUTED_INCIDENT_MAX = 10
PROMPT_CAUSE_CONFIRMATION_MAX = 10
PROMPT_LATER_PROGRESS_OBSERVATION_MAX = 10
PROMPT_POST_FAULT_SUMMARY_MAX = 10
PROMPT_CASCADE_MAX = 20
PROMPT_CONTEXT_HIGH_SIGNAL_RE = re.compile(
    r"\b(?:CRITICAL|FATAL|ERROR|Traceback|RuntimeError|AssertionError|AcceleratorError)\b"
    r"|(?:raising|raised)\s+.*\b(?:gpu|accelerator|device)\b.*\berror\b"
    r"|\b(?:assert(?:ion)? failed|out of bounds|bounds failure|timeout)\b",
    re.IGNORECASE,
)
ESTIMATED_CHARS_PER_EVIDENCE_TOKEN = 3


def build_l0_model_facing_view(
    bundle: L0Bundle,
    decision_evidence: DecisionEvidence,
    restart_environment_context: Mapping[str, bool] | None = None,
) -> L0ModelFacingView:
    """Build the deterministic, attention-efficient L0B projection once."""

    evidence_bundle = _model_evidence_for_projection(bundle, decision_evidence)
    attempt_execution_context = _attempt_execution_context(bundle)
    resolved_restart_environment_context = dict(DEFAULT_RESTART_ENVIRONMENT_CONTEXT)
    if restart_environment_context:
        resolved_restart_environment_context.update(restart_environment_context)
    serialized_evidence = json.dumps(
        {
            "decision_evidence": decision_evidence.to_payload(),
            "attempt_execution_context": attempt_execution_context,
            "restart_environment_context": resolved_restart_environment_context,
            "evidence_bundle": evidence_bundle,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    projection_metrics = _projection_metrics(
        bundle,
        decision_evidence,
        evidence_bundle,
        serialized_evidence,
    )
    return L0ModelFacingView(
        decision_evidence=decision_evidence,
        evidence_bundle=evidence_bundle,
        attempt_execution_context=attempt_execution_context,
        restart_environment_context=resolved_restart_environment_context,
        projection_metrics=projection_metrics,
    )


def _projection_metrics(
    bundle: L0Bundle,
    decision_evidence: DecisionEvidence,
    evidence_bundle: dict[str, Any],
    serialized_evidence: str,
) -> dict[str, Any]:
    context_windows = list(evidence_bundle.get("context_windows") or ())
    context_stats = _context_projection_stats(bundle, context_windows)
    selection_counts = _projection_selection_counts(
        bundle,
        evidence_bundle,
        projected_context_window_count=len(context_windows),
        selected_source_window_count=context_stats["selected_source_window_count"],
    )
    budget_utilization = _projection_budget_utilization(
        evidence_bundle,
        max_window_lines=context_stats["max_window_lines"],
        max_window_characters=context_stats["max_window_characters"],
        selected_source_window_count=context_stats["selected_source_window_count"],
    )
    compaction_counts = _projection_compaction_counts(
        bundle,
        context_windows,
        context_stats,
    )
    integrity_checks = _projection_integrity_checks(
        bundle,
        decision_evidence,
        context_windows,
        selection_counts,
        budget_utilization,
        unresolved_projected_window_ids=context_stats["unresolved_projected_window_ids"],
    )
    return {
        "view_size": {
            "compact_json_characters": len(serialized_evidence),
            "estimated_tokens": (len(serialized_evidence) + ESTIMATED_CHARS_PER_EVIDENCE_TOKEN - 1)
            // ESTIMATED_CHARS_PER_EVIDENCE_TOKEN,
            "estimation_characters_per_token": ESTIMATED_CHARS_PER_EVIDENCE_TOKEN,
        },
        "budget_utilization": budget_utilization,
        "selection_counts": selection_counts,
        "compaction_counts": compaction_counts,
        "projection_integrity": {
            "status": "ok" if all(integrity_checks.values()) else "failed",
            "checks": integrity_checks,
            "unresolved_projected_window_ids": context_stats["unresolved_projected_window_ids"],
            "deterministic_payload_sha256": (
                "sha256:" + hashlib.sha256(serialized_evidence.encode("utf-8")).hexdigest()
            ),
        },
    }


def _context_projection_stats(
    bundle: L0Bundle,
    context_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    projected_source_window_ids = {
        source_id
        for window in context_windows
        for source_id in str(window.get("window_id") or "").split("+")
        if source_id
    }
    known_window_ids = {window.window_id for window in bundle.context_windows}
    return {
        "selected_source_window_count": len(
            projected_source_window_ids.intersection(known_window_ids)
        ),
        "unresolved_projected_window_ids": sorted(projected_source_window_ids - known_window_ids),
        "source_context_lines": sum(
            int(window.get("source_line_count") or 0) for window in context_windows
        ),
        "model_facing_context_lines": sum(
            int(window.get("lines_in_prompt") or 0) for window in context_windows
        ),
        "max_window_lines": max(
            (int(window.get("lines_in_prompt") or 0) for window in context_windows),
            default=0,
        ),
        "max_window_characters": max(
            (
                sum(len(str(line.get("text") or "")) for line in window.get("lines") or ())
                for window in context_windows
            ),
            default=0,
        ),
    }


def _projection_selection_counts(
    bundle: L0Bundle,
    evidence_bundle: dict[str, Any],
    *,
    projected_context_window_count: int,
    selected_source_window_count: int,
) -> dict[str, dict[str, Any]]:
    return {
        "context_windows": _selection_count(
            len(bundle.context_windows),
            selected_source_window_count,
            PROMPT_CONTEXT_MAX_WINDOWS,
            projected=projected_context_window_count,
        ),
        "occurrence_groups": _selection_count(
            len(bundle.occurrence_groups),
            len(evidence_bundle.get("occurrence_groups") or ()),
            PROMPT_OCCURRENCE_GROUP_MAX,
        ),
        "registry_candidate_groups": _selection_count(
            _eligible_registry_candidate_group_count(bundle),
            len(evidence_bundle.get("registry_candidate_groups") or ()),
            PROMPT_REGISTRY_CANDIDATE_GROUP_MAX,
        ),
        "candidate_anchors": _selection_count(
            len(bundle.candidate_anchors),
            len(evidence_bundle.get("candidate_anchors") or ()),
            PROMPT_CANDIDATE_ANCHOR_MAX,
        ),
        "failure_episodes": _selection_count(
            len(bundle.failure_episodes),
            len(evidence_bundle.get("failure_episodes") or ()),
            PROMPT_FAILURE_EPISODE_MAX,
        ),
        "distributed_incidents": _selection_count(
            len(bundle.distributed_failure_incidents),
            len(evidence_bundle.get("distributed_failure_incidents") or ()),
            PROMPT_DISTRIBUTED_INCIDENT_MAX,
        ),
        "cause_confirmations": _selection_count(
            len(bundle.cause_confirmations),
            len(evidence_bundle.get("cause_confirmations") or ()),
            PROMPT_CAUSE_CONFIRMATION_MAX,
        ),
        "later_progress_observations": _selection_count(
            len(bundle.later_progress_after_fault_observations),
            len(evidence_bundle.get("later_progress_after_fault_observations") or ()),
            PROMPT_LATER_PROGRESS_OBSERVATION_MAX,
        ),
        "post_fault_summaries": _selection_count(
            len(bundle.post_fault_summaries),
            len(evidence_bundle.get("post_fault_summaries") or ()),
            PROMPT_POST_FAULT_SUMMARY_MAX,
        ),
        "cascades": _selection_count(
            len(bundle.cascades),
            len(evidence_bundle.get("cascades") or ()),
            PROMPT_CASCADE_MAX,
        ),
    }


def _projection_budget_utilization(
    evidence_bundle: dict[str, Any],
    *,
    max_window_lines: int,
    max_window_characters: int,
    selected_source_window_count: int,
) -> dict[str, Any]:
    return {
        "context_window_slots": _budget_usage(
            selected_source_window_count,
            PROMPT_CONTEXT_MAX_WINDOWS,
        ),
        "occurrence_group_slots": _budget_usage(
            len(evidence_bundle.get("occurrence_groups") or ()),
            PROMPT_OCCURRENCE_GROUP_MAX,
        ),
        "candidate_anchor_slots": _budget_usage(
            len(evidence_bundle.get("candidate_anchors") or ()),
            PROMPT_CANDIDATE_ANCHOR_MAX,
        ),
        "failure_episode_slots": _budget_usage(
            len(evidence_bundle.get("failure_episodes") or ()),
            PROMPT_FAILURE_EPISODE_MAX,
        ),
        "max_window_lines": _budget_usage(
            max_window_lines,
            PROMPT_CONTEXT_EXCERPT_MAX_LINES,
        ),
        "max_window_characters": _budget_usage(
            max_window_characters,
            PROMPT_CONTEXT_EXCERPT_MAX_CHARS,
        ),
        "overall_view_limit_configured": False,
        "overall_view_utilization_pct": None,
    }


def _projection_compaction_counts(
    bundle: L0Bundle,
    context_windows: list[dict[str, Any]],
    context_stats: dict[str, Any],
) -> dict[str, int]:
    selected_count = int(context_stats["selected_source_window_count"])
    source_line_count = int(context_stats["source_context_lines"])
    model_line_count = int(context_stats["model_facing_context_lines"])
    return {
        "source_context_windows_selected": selected_count,
        "projected_context_windows_after_merge": len(context_windows),
        "context_windows_merged": max(0, selected_count - len(context_windows)),
        "context_windows_omitted": max(
            0,
            len(bundle.context_windows) - selected_count,
        ),
        "source_context_lines": source_line_count,
        "model_facing_context_lines": model_line_count,
        "context_lines_omitted": max(0, source_line_count - model_line_count),
        "truncated_context_windows": sum(
            1 for window in context_windows if window.get("truncated")
        ),
        "truncated_model_facing_lines": sum(
            1
            for window in context_windows
            for line in window.get("lines") or ()
            if line.get("line_truncated")
        ),
    }


def _projection_integrity_checks(
    bundle: L0Bundle,
    decision_evidence: DecisionEvidence,
    context_windows: list[dict[str, Any]],
    selection_counts: dict[str, dict[str, Any]],
    budget_utilization: dict[str, Any],
    *,
    unresolved_projected_window_ids: list[str],
) -> dict[str, bool]:
    return {
        "payload_serializable": True,
        "decision_evidence_references_resolve": _decision_evidence_references_resolve(
            bundle,
            decision_evidence,
        ),
        "projected_context_references_resolve": not unresolved_projected_window_ids,
        "projected_line_numbers_valid": _projected_line_numbers_valid(
            context_windows,
            bundle.line_count,
        ),
        "selection_accounting_consistent": all(
            int(counts["available"]) == int(counts["selected"]) + int(counts["omitted"])
            for counts in selection_counts.values()
        ),
        "within_declared_section_limits": all(
            value.get("utilization_pct") is None or float(value["utilization_pct"]) <= 100.0
            for value in budget_utilization.values()
            if isinstance(value, dict)
        ),
        "lossiness_accounted": all(
            "omitted" in counts and "limit" in counts for counts in selection_counts.values()
        ),
    }


def _selection_count(
    available: int,
    selected: int,
    limit: int,
    *,
    projected: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": available,
        "selected": selected,
        "omitted": max(0, available - selected),
        "limit": limit,
    }
    if projected is not None:
        result["projected_after_merge"] = projected
    return result


def _budget_usage(used: int, limit: int) -> dict[str, Any]:
    return {
        "used": used,
        "limit": limit,
        "utilization_pct": round((used / limit) * 100, 1) if limit else None,
    }


def _eligible_registry_candidate_group_count(bundle: L0Bundle) -> int:
    registry_shapes = {
        (match.registry_id, normalized_pattern(match.quote or match.signature))
        for match in bundle.registry_matches
    }
    return sum(
        1
        for group in bundle.occurrence_groups
        if group.registry_id is not None
        and (group.registry_id, group.normalized_shape) in registry_shapes
    )


def _decision_evidence_references_resolve(
    bundle: L0Bundle,
    decision_evidence: DecisionEvidence,
) -> bool:
    references = decision_evidence.selected_evidence_references
    known = {
        "candidate_anchor_ids": {item.anchor_id for item in bundle.candidate_anchors},
        "context_window_ids": {item.window_id for item in bundle.context_windows},
        "failure_episode_ids": {item.episode_id for item in bundle.failure_episodes},
        "distributed_incident_ids": {
            item.incident_id for item in bundle.distributed_failure_incidents
        },
        "occurrence_group_ids": {item.occurrence_group_id for item in bundle.occurrence_groups},
    }
    for name, known_ids in known.items():
        if not set(references.get(name) or ()).issubset(known_ids):
            return False
    return all(
        isinstance(line, int) and 1 <= line <= bundle.line_count
        for line in references.get("source_lines") or ()
    )


def _projected_line_numbers_valid(
    context_windows: list[dict[str, Any]],
    line_count: int,
) -> bool:
    for window in context_windows:
        start = int(window.get("start_line") or 0)
        end = int(window.get("end_line") or 0)
        lines = [int(line.get("line") or 0) for line in window.get("lines") or ()]
        if start < 1 or end < start or end > line_count:
            return False
        if lines != sorted(lines):
            return False
        if any(line < start or line > end for line in lines):
            return False
    return True


def _model_evidence_for_projection(
    bundle: L0Bundle,
    decision_evidence: DecisionEvidence,
) -> dict[str, Any]:
    primary = decision_evidence.deterministic_primary_candidate
    context_windows = _context_windows_for_prompt(
        bundle,
        primary_line=primary.line if primary is not None else None,
    )
    return {
        "byte_size": bundle.byte_size,
        "line_count": bundle.line_count,
        "path_access_facts": [dict(item) for item in bundle.path_access_facts],
        "path_namespace_summary": dict(bundle.path_namespace_summary),
        "occurrence_groups": _occurrence_groups_for_prompt(bundle),
        "registry_candidate_groups": _registry_candidate_groups_for_prompt(bundle),
        "registry_candidate": _registry_hint_for_prompt(
            primary,
            causal_role_hint=_causal_role_hint_for_line(
                bundle,
                primary.line if primary is not None else None,
            ),
        ),
        "candidate_anchors": _candidate_anchors_for_prompt(bundle, context_windows),
        "failure_episodes": [
            _failure_episode_for_prompt(episode)
            for episode in bundle.failure_episodes[:PROMPT_FAILURE_EPISODE_MAX]
        ],
        "distributed_failure_incidents": _distributed_incidents_for_prompt(bundle),
        "cause_confirmations": [
            _cause_confirmation_for_prompt(confirmation)
            for confirmation in bundle.cause_confirmations[:PROMPT_CAUSE_CONFIRMATION_MAX]
        ],
        "later_progress_after_fault_observations": (
            _later_progress_after_fault_observations_for_prompt(bundle)
        ),
        "post_fault_summaries": _post_fault_summaries_for_prompt(bundle),
        "cascades": _cascades_for_prompt(bundle),
        "context_windows": context_windows,
        "progress": _progress_for_prompt(bundle),
        "run_progress_summary": _run_progress_summary_for_prompt(bundle.run_progress_summary),
        "job_metadata": _job_metadata_for_prompt(bundle.job_metadata),
        "evidence_coverage": dict(bundle.evidence_coverage),
        "selection_summary": dict(bundle.selection_summary),
    }


def _occurrence_groups_for_prompt(bundle: L0Bundle) -> list[dict[str, Any]]:
    return [
        {
            "occurrence_group_id": group.occurrence_group_id,
            "normalized_shape": _truncate_text(group.normalized_shape, PROMPT_PATTERN_MAX_CHARS),
            "first_line": group.first_line,
            "count": group.count,
            "sample_lines": list(group.sample_lines),
            "registry_id": group.registry_id,
            "classification": group.classification,
        }
        for group in bundle.occurrence_groups[:PROMPT_OCCURRENCE_GROUP_MAX]
    ]


def _distributed_incidents_for_prompt(bundle: L0Bundle) -> list[dict[str, Any]]:
    return [
        {
            "incident_id": incident.incident_id,
            "incident_kind": incident.incident_kind,
            "incident_type": incident.incident_type,
            "status": incident.status,
            "primary_observed_line": incident.primary_observed_line,
            "primary_observed_quote": _truncate_text(
                incident.primary_observed_quote,
                PROMPT_QUOTE_MAX_CHARS,
            ),
            "sample_lines": list(incident.sample_lines),
            "event_count": incident.event_count,
            "unique_operation_count": incident.unique_operation_count,
            "operation_types": list(incident.operation_types),
            "operation_signatures": list(incident.operation_signatures),
            "observed_rank_count": incident.observed_rank_count,
            "rank_spread_sample": list(incident.rank_spread),
            "process_group_types": list(incident.process_group_types),
            "phase": incident.phase,
            "configured_timeout_seconds": incident.configured_timeout_seconds,
            "last_progress_line": incident.last_progress_line,
            "last_progress_timestamp": incident.last_progress_timestamp,
            "first_detection_timestamp": incident.first_detection_timestamp,
            "seconds_since_last_progress": incident.seconds_since_last_progress,
            "detection_lag_seconds": incident.detection_lag_seconds,
            "root_cause_status": incident.root_cause_status,
            "interpretation": incident.interpretation,
        }
        for incident in bundle.distributed_failure_incidents[:PROMPT_DISTRIBUTED_INCIDENT_MAX]
    ]


def _post_fault_summaries_for_prompt(bundle: L0Bundle) -> list[dict[str, Any]]:
    return [
        {
            "episode_id": summary.episode_id,
            "anchor_line": summary.anchor_line,
            "lines_after_anchor": summary.lines_after_anchor,
            "progress_after_observed": summary.progress_after_observed,
            "first_progress_after_line": summary.first_progress_after_line,
            "later_matching_exception_count": summary.later_matching_exception_count,
            "later_matching_exception_lines": list(summary.later_matching_exception_lines),
            "later_high_signal_count": summary.later_high_signal_count,
            "last_high_signal_line": summary.last_high_signal_line,
            "last_high_signal_quote": _truncate_text(
                summary.last_high_signal_quote,
                PROMPT_QUOTE_MAX_CHARS,
            ),
            "first_teardown_line": summary.first_teardown_line,
            "first_process_termination_line": summary.first_process_termination_line,
            "first_scheduler_cancel_line": summary.first_scheduler_cancel_line,
            "first_cascade_line": summary.first_cascade_line,
        }
        for summary in bundle.post_fault_summaries[:PROMPT_POST_FAULT_SUMMARY_MAX]
    ]


def _cascades_for_prompt(bundle: L0Bundle) -> list[dict[str, Any]]:
    return [
        {
            "fine_class": cascade.fine_class,
            "policy_class": cascade.policy_class,
            "cascade_fingerprint": cascade.cascade_fingerprint,
            "first_line": cascade.first_line,
            "last_line": cascade.last_line,
            "count": cascade.count,
            "sample_lines": list(cascade.sample_lines),
            "rank_spread": list(cascade.rank_spread),
            "node_spread": list(cascade.node_spread),
            "gpu_spread": list(cascade.gpu_spread),
            "reason": cascade.reason,
        }
        for cascade in bundle.cascades[:PROMPT_CASCADE_MAX]
    ]


def _progress_for_prompt(bundle: L0Bundle) -> dict[str, Any]:
    progress = bundle.progress
    return {
        "highest_completed_step": progress.highest_completed_step,
        "last_progress_line": progress.last_progress_line,
        "last_checkpoint_step": progress.last_checkpoint_step,
        "last_checkpoint_line": progress.last_checkpoint_line,
        "latest_observed_failure_iteration": progress.latest_observed_failure_iteration,
        "latest_observed_failure_iteration_line": progress.latest_observed_failure_iteration_line,
        "progress_lines": list(progress.progress_lines[-50:]),
        "checkpoint_lines": list(progress.checkpoint_lines[-50:]),
        "setup_lines": list(progress.setup_lines[-50:]),
        "recovery_lines": list(progress.recovery_lines[:50]),
        "recent_progress_markers": [
            _progress_marker_for_prompt(marker) for marker in progress.progress_markers[-20:]
        ],
        "recent_checkpoint_markers": [
            _progress_marker_for_prompt(marker) for marker in progress.checkpoint_markers[-10:]
        ],
        "recent_setup_markers": [
            _progress_marker_for_prompt(marker) for marker in progress.setup_markers[-10:]
        ],
    }


def _run_progress_summary_for_prompt(summary: Any) -> dict[str, Any]:
    return {
        "first_iteration": summary.first_iteration,
        "first_iteration_line": summary.first_iteration_line,
        "first_iteration_timestamp": summary.first_iteration_timestamp,
        "last_iteration": summary.last_iteration,
        "last_iteration_line": summary.last_iteration_line,
        "last_iteration_timestamp": summary.last_iteration_timestamp,
        "iteration_delta": summary.iteration_delta,
        "total_iterations": summary.total_iterations,
        "first_consumed_samples": summary.first_consumed_samples,
        "last_consumed_samples": summary.last_consumed_samples,
        "consumed_samples_delta": summary.consumed_samples_delta,
        "progress_marker_count": summary.progress_marker_count,
        "checkpoint_marker_count": summary.checkpoint_marker_count,
        "setup_marker_count": summary.setup_marker_count,
        "last_checkpoint_iteration": summary.last_checkpoint_iteration,
        "last_checkpoint_line": summary.last_checkpoint_line,
        "checkpoint_load_iteration": summary.checkpoint_load_iteration,
        "checkpoint_load_line": summary.checkpoint_load_line,
        "latest_observed_failure_iteration": summary.latest_observed_failure_iteration,
        "latest_observed_failure_iteration_line": (summary.latest_observed_failure_iteration_line),
        "observed_iterations_after_checkpoint_load": (
            summary.observed_iterations_after_checkpoint_load
        ),
        "last_setup_marker_type": summary.last_setup_marker_type,
        "last_setup_line": summary.last_setup_line,
        "successful_runtime_seconds": summary.successful_runtime_seconds,
        "iterations_since_checkpoint": summary.iterations_since_checkpoint,
        "progress_after_failure_episode": summary.progress_after_failure_episode,
        "first_terminal_incident_line": summary.first_terminal_incident_line,
        "first_terminal_incident_timestamp": summary.first_terminal_incident_timestamp,
        "configured_terminal_timeout_seconds": summary.configured_terminal_timeout_seconds,
        "seconds_from_last_progress_to_terminal_incident": (
            summary.seconds_from_last_progress_to_terminal_incident
        ),
        "terminal_detection_lag_seconds": summary.terminal_detection_lag_seconds,
    }


def _attempt_execution_context(bundle: L0Bundle) -> dict[str, Any]:
    summary = bundle.run_progress_summary
    return {
        "scope": "current_log_only",
        "terminal_timing": {
            "configured_terminal_timeout_seconds": summary.configured_terminal_timeout_seconds,
            "seconds_from_last_progress_to_terminal_incident": (
                summary.seconds_from_last_progress_to_terminal_incident
            ),
            "terminal_detection_lag_seconds": summary.terminal_detection_lag_seconds,
        },
    }


def _later_progress_after_fault_observations_for_prompt(
    bundle: L0Bundle,
) -> list[dict[str, Any]]:
    return [
        {
            "fine_class": item.fine_class,
            "root_fingerprint": item.root_fingerprint,
            "event_count": item.event_count,
            "sample_event_lines": list(item.sample_event_lines),
            "sample_later_progress_lines": list(item.sample_later_progress_lines),
            "matches_terminal_fingerprint": item.matches_terminal_fingerprint,
            "ordering_basis": item.ordering_basis,
            "interpretation": item.interpretation,
            "component_recovery_proven": item.component_recovery_proven,
        }
        for item in bundle.later_progress_after_fault_observations[
            :PROMPT_LATER_PROGRESS_OBSERVATION_MAX
        ]
    ]


def _job_metadata_for_prompt(metadata: Any) -> dict[str, Any]:
    return {
        "explicit_world_size": metadata.explicit_world_size,
        "explicit_world_size_line": metadata.explicit_world_size_line,
        "observed_rank_min": metadata.observed_rank_min,
        "observed_rank_max": metadata.observed_rank_max,
        "observed_rank_count": metadata.observed_rank_count,
        "inferred_world_size_lower_bound": metadata.inferred_world_size_lower_bound,
        "world_size_source": metadata.world_size_source,
        "world_size_confidence": metadata.world_size_confidence,
        "observed_node_count": metadata.observed_node_count,
        "rank_to_gpu_mapping_available": metadata.rank_to_gpu_mapping_available,
    }


def _candidate_anchors_for_prompt(
    bundle: L0Bundle,
    context_windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lines_in_prompt = {
        line["line"]
        for window in context_windows
        for line in window.get("lines", [])
        if isinstance(line, dict) and "line" in line
    }
    return [
        {
            "anchor_id": anchor.anchor_id,
            "line": anchor.line,
            "quote": _truncate_text(anchor.quote, PROMPT_QUOTE_MAX_CHARS),
            "sources": list(anchor.sources),
            "high_signal": anchor.high_signal,
            "causal_role_hint": anchor.causal_role_hint,
            "anchor_rank": anchor.anchor_rank,
            "taxonomy_hint": _registry_hint_for_prompt(
                anchor.taxonomy_match,
                causal_role_hint=anchor.causal_role_hint,
            ),
            "nearby_progress_observations": {
                "prior_observed_progress_line": anchor.prior_observed_progress_line,
                "later_observed_progress_line": anchor.later_observed_progress_line,
                "prior_progress_rank": anchor.prior_progress_rank,
                "later_progress_rank": anchor.later_progress_rank,
                "later_progress_rank_relation": anchor.later_progress_rank_relation,
                "later_observation_proves_recovery": (anchor.later_observation_proves_recovery),
            },
            "first_downstream_registry_hint": _registry_hint_for_prompt(
                anchor.first_downstream_registry_match,
                causal_role_hint=_causal_role_hint_for_line(
                    bundle,
                    (
                        anchor.first_downstream_registry_match.line
                        if anchor.first_downstream_registry_match
                        else None
                    ),
                ),
            ),
            "first_downstream_cascade": _failure_for_prompt(anchor.first_downstream_cascade),
            "context_window_ids": list(anchor.context_window_ids),
            "covered_by_excerpt": anchor.line in lines_in_prompt,
        }
        for anchor in bundle.candidate_anchors[:PROMPT_CANDIDATE_ANCHOR_MAX]
    ]


def _failure_episode_for_prompt(episode: Any) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "status": episode.status,
        "start_line": episode.start_line,
        "end_line": episode.end_line,
        "first_exception_line": episode.first_exception_line,
        "terminal_exception_line": episode.terminal_exception_line,
        "terminal_exception_quote": _truncate_text(
            episode.terminal_exception_quote,
            PROMPT_QUOTE_MAX_CHARS,
        ),
        "terminal_exception_iteration": episode.terminal_exception_iteration,
        "terminal_exception_causal_role_hint": (episode.terminal_exception_causal_role_hint),
        "precursor_lines": list(episode.precursor_lines),
        "identity_anchor_line": episode.identity_anchor_line,
        "identity_anchor_reason": episode.identity_anchor_reason,
        "exception_chain_lines": list(episode.exception_chain_lines),
        "duplicate_rendering_lines": list(episode.duplicate_rendering_lines),
        "wrapper_exception_lines": list(episode.wrapper_exception_lines),
        "exception_rank": episode.exception_rank,
        "exception_node": episode.exception_node,
        "exception_gpu": episode.exception_gpu,
        "last_progress_before": _progress_marker_for_prompt(episode.last_progress_before),
        "first_progress_after": _progress_marker_for_prompt(episode.first_progress_after),
        "first_teardown_line": episode.first_teardown_line,
        "first_process_termination_line": episode.first_process_termination_line,
        "first_scheduler_cancel_line": episode.first_scheduler_cancel_line,
        "first_downstream_cascade": _failure_for_prompt(episode.first_downstream_cascade),
        "cause_confirmations": [
            _cause_confirmation_for_prompt(confirmation)
            for confirmation in episode.cause_confirmations
        ],
        "context_window_ids": list(episode.context_window_ids),
        "reason": episode.reason,
    }


def _progress_marker_for_prompt(marker: Any | None) -> dict[str, Any] | None:
    if marker is None:
        return None
    return {
        "marker_id": marker.marker_id,
        "marker_type": marker.marker_type,
        "value": marker.value,
        "state": marker.state,
        "line": marker.line,
        "quote": _truncate_text(marker.quote, PROMPT_QUOTE_MAX_CHARS),
        "timestamp": marker.timestamp,
        "rank": marker.rank,
        "node": marker.node,
        "gpu": marker.gpu,
        "pattern_id": marker.pattern_id,
        "secondary_value": dict(marker.secondary_value),
    }


def _failure_for_prompt(match: Any | None) -> dict[str, Any] | None:
    if match is None:
        return None
    return {
        **match.to_failure_payload(),
        "quote": _truncate_text(match.quote, PROMPT_QUOTE_MAX_CHARS),
        "registry_id": match.registry_id,
        "role": match.role,
    }


def _cause_confirmation_for_prompt(match: Any) -> dict[str, Any]:
    return {
        "confirmation_kind": match.fine_class,
        "signature": match.signature,
        "line": match.line,
        "quote": _truncate_text(match.quote, PROMPT_QUOTE_MAX_CHARS),
        "rank": match.rank,
        "node": match.node,
        "phase": match.phase,
        "registry_id": match.registry_id,
    }


def _registry_candidate_groups_for_prompt(bundle: L0Bundle) -> list[dict[str, Any]]:
    matches_by_pattern: dict[tuple[str | None, str], Any] = {}
    for match in bundle.registry_matches:
        key = (match.registry_id, normalized_pattern(match.quote or match.signature))
        matches_by_pattern.setdefault(key, match)

    result: list[dict[str, Any]] = []
    for group in bundle.occurrence_groups:
        if group.registry_id is None:
            continue
        match = matches_by_pattern.get((group.registry_id, group.normalized_shape))
        if match is None:
            continue
        result.append(
            {
                "registry_id": match.registry_id,
                "fine_class_hint": match.fine_class,
                "signature_hint": match.signature,
                "first_line": group.first_line,
                "representative_quote": _truncate_text(
                    match.quote,
                    PROMPT_QUOTE_MAX_CHARS,
                ),
                "count": group.count,
                "sample_lines": list(group.sample_lines),
                "rank_spread": list(group.rank_spread[:8]),
                "causal_role_hint": _causal_role_hint_for_line(bundle, match.line),
                "provisional": True,
            }
        )
    return result[:PROMPT_REGISTRY_CANDIDATE_GROUP_MAX]


def _registry_hint_for_prompt(
    match: Any | None,
    *,
    causal_role_hint: str = CausalRole.UNKNOWN.value,
) -> dict[str, Any] | None:
    if match is None:
        return None
    return {
        "fine_class_hint": match.fine_class,
        "signature_hint": match.signature,
        "line": match.line,
        "quote": _truncate_text(match.quote, PROMPT_QUOTE_MAX_CHARS),
        "rank": match.rank,
        "phase": match.phase,
        "failure_iteration": match.failure_iteration,
        "registry_id": match.registry_id,
        "causal_role_hint": causal_role_hint,
        "provisional": True,
    }


def _causal_role_hint_for_line(bundle: L0Bundle, line: int | None) -> str:
    if line is None:
        return CausalRole.UNKNOWN.value
    for episode in bundle.failure_episodes:
        if episode.terminal_exception_line == line:
            return episode.terminal_exception_causal_role_hint
        if line in episode.wrapper_exception_lines:
            return CausalRole.TEARDOWN.value
    return CausalRole.UNKNOWN.value


def _context_windows_for_prompt(
    bundle: L0Bundle,
    *,
    primary_line: int | None,
) -> list[dict[str, Any]]:
    selected_windows = _seed_context_windows(bundle.context_windows, primary_line=primary_line)
    for window in sorted(
        bundle.context_windows,
        key=lambda window: (
            -_context_window_score(window, primary_line=primary_line),
            window.start_line,
            window.end_line,
        ),
    ):
        if len(selected_windows) >= PROMPT_CONTEXT_MAX_WINDOWS:
            break
        if window not in selected_windows:
            selected_windows.append(window)

    merged_windows = _merge_context_windows(selected_windows)
    return [_context_window_payload(window) for window in merged_windows]


def _seed_context_windows(windows: Any, *, primary_line: int | None) -> list[Any]:
    selected: list[Any] = []
    earliest_high_signal = _earliest_high_signal_window(windows)
    if earliest_high_signal is not None:
        selected.append(earliest_high_signal)
    if primary_line is not None:
        primary_windows = [
            window for window in windows if window.start_line <= primary_line <= window.end_line
        ]
        if primary_windows:
            primary_window = sorted(primary_windows, key=lambda window: window.start_line)[0]
            if primary_window not in selected:
                selected.append(primary_window)
    return selected


def _earliest_high_signal_window(windows: Any) -> Any | None:
    best: tuple[int, Any] | None = None
    for window in windows:
        signal_lines = [item.line for item in window.lines if _is_high_signal_line(item)]
        if not signal_lines:
            continue
        candidate = (min(signal_lines), window)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best[1] if best else None


def _context_window_score(window: Any, *, primary_line: int | None) -> int:
    score = 0
    if primary_line is not None and window.start_line <= primary_line <= window.end_line:
        score += 1000
    score += len(window.seed_lines) * 10
    score += min(sum(1 for item in window.lines if _is_high_signal_line(item)), 10) * 20
    return score


def _merge_context_windows(windows: list[Any]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for window in sorted(windows, key=lambda item: (item.start_line, item.end_line)):
        current = {
            "window_id": window.window_id,
            "selected_by": window.selected_by,
            "start_line": window.start_line,
            "end_line": window.end_line,
            "seed_lines": set(window.seed_lines),
            "occurrence_group_ids": set(window.occurrence_group_ids),
            "truncated": window.truncated,
            "lines_by_number": {item.line: item for item in window.lines},
        }
        if (
            merged
            and current["start_line"] <= merged[-1]["end_line"] + PROMPT_CONTEXT_MERGE_GAP_LINES
        ):
            previous = merged[-1]
            previous["window_id"] = f"{previous['window_id']}+{current['window_id']}"
            previous["selected_by"] = _merge_selected_by(
                str(previous["selected_by"]), str(current["selected_by"])
            )
            previous["end_line"] = max(int(previous["end_line"]), int(current["end_line"]))
            previous["seed_lines"].update(current["seed_lines"])
            previous["occurrence_group_ids"].update(current["occurrence_group_ids"])
            previous["truncated"] = bool(previous["truncated"] or current["truncated"])
            previous["lines_by_number"].update(current["lines_by_number"])
            continue
        merged.append(current)
    return merged[:PROMPT_CONTEXT_MAX_WINDOWS]


def _merge_selected_by(first: str, second: str) -> str:
    if first == second:
        return first
    parts = []
    for value in (first, second):
        for part in value.split("+"):
            if part not in parts:
                parts.append(part)
    return "+".join(parts)


def _context_window_payload(window: dict[str, Any]) -> dict[str, Any]:
    lines = [window["lines_by_number"][line] for line in sorted(window["lines_by_number"])]
    excerpt, excerpt_truncated = _window_excerpt_lines(lines)
    return {
        "window_id": window["window_id"],
        "selected_by": window["selected_by"],
        "start_line": window["start_line"],
        "end_line": window["end_line"],
        "seed_lines": sorted(window["seed_lines"]),
        "occurrence_group_ids": sorted(window["occurrence_group_ids"]),
        "prompt_view": "bounded_excerpt",
        "source_line_count": len(lines),
        "lines_in_prompt": len(excerpt),
        "truncated": bool(window["truncated"] or excerpt_truncated),
        "lines": excerpt,
    }


def _window_excerpt_lines(lines: Any) -> tuple[list[dict[str, Any]], bool]:
    result: list[dict[str, Any]] = []
    used_chars = 0
    truncated = False
    for item in list(lines):
        if len(result) >= PROMPT_CONTEXT_EXCERPT_MAX_LINES:
            truncated = True
            break
        text = _truncate_text(item.text, PROMPT_CONTEXT_LINE_MAX_CHARS)
        next_used = used_chars + len(text or "")
        if next_used > PROMPT_CONTEXT_EXCERPT_MAX_CHARS:
            truncated = True
            break
        result.append(
            {
                "line": item.line,
                "text": text,
                "line_truncated": len(item.text) > PROMPT_CONTEXT_LINE_MAX_CHARS,
                "line_role": _prompt_line_role(item.text),
                "diagnostic_kind": diagnostic_context_kind(item.text),
                "diagnostic_uncertainty_kind": diagnostic_uncertainty_kind(item.text),
            }
        )
        used_chars = next_used
    return result, truncated


def _prompt_line_role(text: str) -> str:
    if diagnostic_context_kind(text) is not None:
        return "diagnostic_context"
    if diagnostic_uncertainty_kind(text) is not None:
        return "observed_log_with_causal_hypothesis"
    return "observed_log"


def _window_preview_lines(
    lines: Any,
    seed_lines: Any,
) -> list[dict[str, Any]]:
    line_list = list(lines)
    seed_set = set(seed_lines)
    selected_lines: set[int] = set()

    def add_items(items: Any) -> None:
        for item in items:
            if len(selected_lines) >= PROMPT_CONTEXT_PREVIEW_LINES:
                return
            selected_lines.add(item.line)

    add_items(item for item in line_list if item.line in seed_set)
    add_items(_high_signal_lines(line_list, limit=PROMPT_CONTEXT_HIGH_SIGNAL_LINES))
    add_items(line_list[:PROMPT_CONTEXT_HEAD_LINES])
    add_items(line_list[-PROMPT_CONTEXT_TAIL_LINES:])

    result: list[dict[str, Any]] = []
    for item in line_list:
        if item.line not in selected_lines:
            continue
        result.append(
            {
                "line": item.line,
                "text": _truncate_text(item.text, PROMPT_CONTEXT_LINE_MAX_CHARS),
                "line_truncated": len(item.text) > PROMPT_CONTEXT_LINE_MAX_CHARS,
            }
        )
    return result


def _high_signal_lines(lines: list[Any], *, limit: int) -> list[Any]:
    result: list[Any] = []
    for item in lines:
        if _is_high_signal_line(item):
            result.append(item)
            if len(result) >= limit:
                break
    return result


def _is_high_signal_line(item: Any) -> bool:
    if diagnostic_context_kind(item.text) is not None:
        return False
    return bool(PROMPT_CONTEXT_HIGH_SIGNAL_RE.search(item.text))


def _truncate_text(value: str | None, max_chars: int) -> str | None:
    if value is None:
        return None
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "...[truncated]"
