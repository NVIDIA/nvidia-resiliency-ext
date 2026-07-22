# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only log tools over an L0 evidence bundle."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Callable, Iterable

from ..infrastructure.log_source import LogSnapshot
from ..models import CausalRole, L0Bundle, L0ModelFacingView, LogLine
from .contracts import EvidenceTools, L1EvidenceContext

OVERVIEW_HEAD_LINES = 40
OVERVIEW_TAIL_LINES = 80
OVERVIEW_MAX_CHARS = 12_000
GREP_MAX_MATCHES = 50
GREP_MAX_MATCHES_HARD_LIMIT = 200
READ_WINDOW_MAX_LINES = 240
READ_WINDOW_MAX_CHARS = 50_000
TOOL_LINE_MAX_CHARS = 2_000
EVIDENCE_OBJECTS_SCHEMA_VERSION = "restart_agent_evidence_objects.v1"
EVIDENCE_OBJECTS_MAX_REFS = 8
EVIDENCE_OBJECTS_MAX_CHARS = 50_000
EVIDENCE_OBJECT_REF_MAX_CHARS = 128
EVIDENCE_OBJECTS_METADATA_RESERVE_CHARS = 2_048


EvidenceToolsFactory = Callable[[L0Bundle, LogSnapshot], EvidenceTools]


def build_l1_evidence_context(
    bundle: L0Bundle,
    model_view: L0ModelFacingView,
    source_log: LogSnapshot,
    tools_factory: EvidenceToolsFactory | None = None,
) -> L1EvidenceContext:
    """Assemble the provider-neutral L1 view and controlled expansion tools."""

    factory = tools_factory or LogTools
    return L1EvidenceContext(model_view=model_view, tools=factory(bundle, source_log))


class LogTools:
    def __init__(self, bundle: L0Bundle, source_log: LogSnapshot):
        self._bundle = bundle
        self._source_log = source_log

    def overview(self) -> dict[str, Any]:
        return _build_overview(self._bundle, self._source_log)

    def grep_log(
        self,
        pattern: str,
        *,
        ignore_case: bool = True,
        max_matches: int = GREP_MAX_MATCHES,
    ) -> dict[str, Any]:
        max_matches = min(max(max_matches, 0), GREP_MAX_MATCHES_HARD_LIMIT)
        flags = re.I if ignore_case else 0
        regex = re.compile(pattern, flags)
        matches: list[dict[str, Any]] = []
        total = 0
        for item in self._source_log.log_lines():
            if not regex.search(item.text):
                continue
            total += 1
            if len(matches) < max_matches:
                matches.append(_line_payload(item))
        return {
            "pattern": pattern,
            "matches": matches,
            "total_matches": total,
            "truncated": total > len(matches),
        }

    def read_window(
        self,
        center_line: int,
        *,
        before: int = 20,
        after: int = 80,
    ) -> dict[str, Any]:
        before = min(max(before, 0), 120)
        after = min(max(after, 0), 120)
        start = max(1, center_line - before)
        end = min(self._bundle.line_count, center_line + after)
        if end - start + 1 > READ_WINDOW_MAX_LINES:
            end = start + READ_WINDOW_MAX_LINES - 1

        selected, chars_truncated = _line_payload_with_char_cap(
            self._source_log.log_lines(start_line=start, end_line=end),
            READ_WINDOW_MAX_CHARS,
        )
        return {
            "start_line": start,
            "end_line": end,
            "lines": selected,
            "truncated": end < min(self._bundle.line_count, center_line + after) or chars_truncated,
        }

    def get_evidence_objects(self, refs: list[str]) -> dict[str, Any]:
        """Resolve attempt-scoped L0A references without rereading the log."""

        requested_refs: list[str] = []
        invalid_refs: list[str] = []
        request_truncated = len(refs) > EVIDENCE_OBJECTS_MAX_REFS
        for ref in refs[:EVIDENCE_OBJECTS_MAX_REFS]:
            if len(ref) > EVIDENCE_OBJECT_REF_MAX_CHARS:
                invalid_refs.append(ref[:EVIDENCE_OBJECT_REF_MAX_CHARS])
                request_truncated = True
                continue
            if ref not in requested_refs:
                requested_refs.append(ref)

        index = _evidence_object_index(self._bundle)
        objects: list[dict[str, Any]] = []
        missing_refs: list[str] = []
        omitted_refs: list[str] = []
        object_truncated = False
        for ref in requested_refs:
            indexed = index.get(ref)
            if indexed is None:
                missing_refs.append(ref)
                continue
            object_type, value = indexed
            response_without_object = {
                "schema_version": EVIDENCE_OBJECTS_SCHEMA_VERSION,
                "requested_refs": requested_refs,
                "objects": objects,
                "missing_refs": missing_refs,
                "invalid_refs": invalid_refs,
                "omitted_refs": omitted_refs,
                "truncated": True,
            }
            remaining = (
                EVIDENCE_OBJECTS_MAX_CHARS
                - _json_chars(response_without_object)
                - EVIDENCE_OBJECTS_METADATA_RESERVE_CHARS
            )
            if remaining <= 256:
                omitted_refs.append(ref)
                object_truncated = True
                continue
            payload, payload_truncated = _bounded_json_payload(
                _json_compatible(asdict(value)),
                remaining,
            )
            candidate = {
                "ref": ref,
                "object_type": object_type,
                "payload": payload,
                "truncated": payload_truncated,
            }
            if _json_chars({**response_without_object, "objects": [*objects, candidate]}) > (
                EVIDENCE_OBJECTS_MAX_CHARS
            ):
                omitted_refs.append(ref)
                object_truncated = True
                continue
            objects.append(candidate)
            object_truncated = object_truncated or payload_truncated

        return {
            "schema_version": EVIDENCE_OBJECTS_SCHEMA_VERSION,
            "requested_refs": requested_refs,
            "objects": objects,
            "missing_refs": missing_refs,
            "invalid_refs": invalid_refs,
            "omitted_refs": omitted_refs,
            "limits": {
                "max_refs": EVIDENCE_OBJECTS_MAX_REFS,
                "max_chars": EVIDENCE_OBJECTS_MAX_CHARS,
            },
            "truncated": request_truncated or object_truncated or bool(omitted_refs),
        }


def _build_overview(bundle: L0Bundle, source_log: LogSnapshot) -> dict[str, Any]:
    head = list(source_log.log_lines(end_line=OVERVIEW_HEAD_LINES))
    tail_start = max(1, len(source_log.lines) - OVERVIEW_TAIL_LINES + 1)
    tail = list(source_log.log_lines(start_line=tail_start))
    head_payload, head_truncated = _line_payload_with_char_cap(head, OVERVIEW_MAX_CHARS // 2)
    tail_payload, tail_truncated = _line_payload_with_char_cap(tail, OVERVIEW_MAX_CHARS // 2)
    return {
        "line_count": bundle.line_count,
        "byte_size": bundle.byte_size,
        "head": head_payload,
        "tail": tail_payload,
        "deterministic_summary": {
            "path_access_facts": [dict(item) for item in bundle.path_access_facts],
            "path_namespace_summary": dict(bundle.path_namespace_summary),
            "run_progress_summary": _run_progress_summary_payload(bundle.run_progress_summary),
            "job_metadata": _job_metadata_payload(bundle.job_metadata),
            "later_progress_after_fault_observations": [
                {
                    "fine_class": item.fine_class,
                    "root_fingerprint": item.root_fingerprint,
                    "event_count": item.event_count,
                    "sample_event_lines": list(item.sample_event_lines),
                    "sample_later_progress_lines": list(item.sample_later_progress_lines),
                    "matches_terminal_fingerprint": (item.matches_terminal_fingerprint),
                    "ordering_basis": item.ordering_basis,
                    "interpretation": item.interpretation,
                    "component_recovery_proven": item.component_recovery_proven,
                }
                for item in bundle.later_progress_after_fault_observations[:10]
            ],
            "progress_lines": list(bundle.progress.progress_lines[-20:]),
            "recent_progress_markers": [
                _progress_marker_payload(marker)
                for marker in bundle.progress.progress_markers[-20:]
            ],
            "recent_checkpoint_markers": [
                _progress_marker_payload(marker)
                for marker in bundle.progress.checkpoint_markers[-10:]
            ],
            "recent_setup_markers": [
                _progress_marker_payload(marker) for marker in bundle.progress.setup_markers[-10:]
            ],
            "failure_episodes": [
                {
                    "episode_id": episode.episode_id,
                    "status": episode.status,
                    "start_line": episode.start_line,
                    "end_line": episode.end_line,
                    "first_exception_line": episode.first_exception_line,
                    "terminal_exception_line": episode.terminal_exception_line,
                    "terminal_exception_iteration": episode.terminal_exception_iteration,
                    "terminal_exception_causal_role_hint": (
                        episode.terminal_exception_causal_role_hint
                    ),
                    "exception_chain_lines": list(episode.exception_chain_lines),
                    "duplicate_rendering_lines": list(episode.duplicate_rendering_lines),
                    "wrapper_exception_lines": list(episode.wrapper_exception_lines),
                    "exception_rank": episode.exception_rank,
                    "exception_node": episode.exception_node,
                    "exception_gpu": episode.exception_gpu,
                    "last_progress_before": _progress_marker_payload(episode.last_progress_before),
                    "first_progress_after": _progress_marker_payload(episode.first_progress_after),
                    "first_teardown_line": episode.first_teardown_line,
                    "first_process_termination_line": (episode.first_process_termination_line),
                    "first_scheduler_cancel_line": episode.first_scheduler_cancel_line,
                    "cause_confirmations": [
                        _registry_hint(bundle, confirmation)
                        for confirmation in episode.cause_confirmations
                    ],
                    "context_window_ids": list(episode.context_window_ids),
                    "reason": episode.reason,
                }
                for episode in bundle.failure_episodes[:10]
            ],
            "distributed_failure_incidents": [
                {
                    "incident_id": incident.incident_id,
                    "incident_kind": incident.incident_kind,
                    "incident_type": incident.incident_type,
                    "status": incident.status,
                    "first_observed_line": incident.first_observed_line,
                    "last_observed_line": incident.last_observed_line,
                    "primary_observed_line": incident.primary_observed_line,
                    "sample_lines": list(incident.sample_lines),
                    "event_count": incident.event_count,
                    "unique_operation_count": incident.unique_operation_count,
                    "operation_types": list(incident.operation_types),
                    "operation_signatures": list(incident.operation_signatures),
                    "observed_rank_count": incident.observed_rank_count,
                    "rank_spread_sample": list(incident.rank_spread),
                    "process_group_types": list(incident.process_group_types),
                    "phase": incident.phase,
                    "configured_timeout_seconds": (incident.configured_timeout_seconds),
                    "last_progress_line": incident.last_progress_line,
                    "last_progress_timestamp": incident.last_progress_timestamp,
                    "first_detection_timestamp": incident.first_detection_timestamp,
                    "seconds_since_last_progress": incident.seconds_since_last_progress,
                    "detection_lag_seconds": incident.detection_lag_seconds,
                    "current_attempt_incident_fingerprint": incident.history_fingerprint,
                    "root_cause_status": incident.root_cause_status,
                    "interpretation": incident.interpretation,
                }
                for incident in bundle.distributed_failure_incidents[:10]
            ],
            "cause_confirmations": [
                _registry_hint(bundle, confirmation)
                for confirmation in bundle.cause_confirmations[:10]
            ],
            "registry_candidate_groups": _registry_candidate_groups(bundle),
            "registry_candidate": _registry_hint(
                bundle,
                bundle.deterministic_primary_candidate,
            ),
            "candidate_anchors": [
                {
                    "anchor_id": anchor.anchor_id,
                    "line": anchor.line,
                    "sources": list(anchor.sources),
                    "high_signal": anchor.high_signal,
                    "causal_role_hint": anchor.causal_role_hint,
                    "anchor_rank": anchor.anchor_rank,
                    "taxonomy_hint": _registry_hint(
                        bundle,
                        anchor.taxonomy_match,
                    ),
                    "nearby_progress_observations": {
                        "prior_observed_progress_line": (anchor.prior_observed_progress_line),
                        "later_observed_progress_line": (anchor.later_observed_progress_line),
                        "prior_progress_rank": anchor.prior_progress_rank,
                        "later_progress_rank": anchor.later_progress_rank,
                        "later_progress_rank_relation": (anchor.later_progress_rank_relation),
                        "later_observation_proves_recovery": (
                            anchor.later_observation_proves_recovery
                        ),
                    },
                    "first_downstream_registry_hint": _registry_hint(
                        bundle,
                        anchor.first_downstream_registry_match,
                    ),
                    "first_downstream_cascade": (
                        anchor.first_downstream_cascade.to_failure_payload()
                        if anchor.first_downstream_cascade
                        else None
                    ),
                    "context_window_ids": list(anchor.context_window_ids),
                }
                for anchor in bundle.candidate_anchors[:20]
            ],
            "cascade_groups": [
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
                for cascade in bundle.cascades[:20]
            ],
            "termination_candidates": [],
        },
        "truncated": (len(bundle.registry_matches) > 20 or head_truncated or tail_truncated),
    }


def _evidence_object_index(bundle: L0Bundle) -> dict[str, tuple[str, Any]]:
    index: dict[str, tuple[str, Any]] = {}
    collections = (
        ("occurrence_group", bundle.occurrence_groups, "occurrence_group_id"),
        ("context_window", bundle.context_windows, "window_id"),
        ("candidate_anchor", bundle.candidate_anchors, "anchor_id"),
        ("failure_episode", bundle.failure_episodes, "episode_id"),
        ("distributed_incident", bundle.distributed_failure_incidents, "incident_id"),
        ("progress_marker", bundle.progress.progress_markers, "marker_id"),
        ("checkpoint_marker", bundle.progress.checkpoint_markers, "marker_id"),
        ("setup_marker", bundle.progress.setup_markers, "marker_id"),
    )
    for object_type, values, id_field in collections:
        for value in values:
            ref = str(getattr(value, id_field))
            index.setdefault(ref, (object_type, value))
    return index


def _bounded_json_payload(value: Any, max_chars: int) -> tuple[Any, bool]:
    if _json_chars(value) <= max_chars:
        return value, False
    if isinstance(value, str):
        keep = max(0, max_chars - 32)
        return value[:keep], True
    if isinstance(value, list):
        result: list[Any] = []
        truncated = False
        for item in value:
            remaining = max_chars - _json_chars(result) - 2
            if remaining <= 0:
                truncated = True
                break
            bounded_item, item_truncated = _bounded_json_payload(item, remaining)
            candidate = [*result, bounded_item]
            if _json_chars(candidate) > max_chars:
                truncated = True
                break
            result.append(bounded_item)
            truncated = truncated or item_truncated
        return result, truncated or len(result) < len(value)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        truncated = False
        for key, item in value.items():
            remaining = max_chars - _json_chars(result) - len(str(key)) - 8
            if remaining <= 0:
                truncated = True
                break
            bounded_item, item_truncated = _bounded_json_payload(item, remaining)
            candidate = {**result, key: bounded_item}
            if _json_chars(candidate) > max_chars:
                truncated = True
                break
            result[key] = bounded_item
            truncated = truncated or item_truncated
        return result, truncated or len(result) < len(value)
    return value, True


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _json_chars(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True))


def _line_payload_with_char_cap(
    lines: Iterable[LogLine],
    max_chars: int,
) -> tuple[list[dict[str, Any]], bool]:
    payload: list[dict[str, Any]] = []
    used = 0
    truncated = False
    for item in lines:
        text = item.text
        next_used = used + len(text)
        if next_used > max_chars:
            remaining = max(0, max_chars - used)
            if remaining > 0:
                payload.append({"line": item.line, "text": text[:remaining]})
            truncated = True
            break
        payload.append(_line_payload(item))
        used = next_used
    return payload, truncated


def _line_payload(item: LogLine) -> dict[str, Any]:
    text = item.text
    truncated = len(text) > TOOL_LINE_MAX_CHARS
    payload = {"line": item.line, "text": text[:TOOL_LINE_MAX_CHARS]}
    if truncated:
        payload["line_truncated"] = True
        payload["original_chars"] = len(text)
    return payload


def _progress_marker_payload(marker: Any | None) -> dict[str, Any] | None:
    if marker is None:
        return None
    return {
        "marker_id": marker.marker_id,
        "marker_type": marker.marker_type,
        "value": marker.value,
        "state": marker.state,
        "line": marker.line,
        "timestamp": marker.timestamp,
        "rank": marker.rank,
        "node": marker.node,
        "gpu": marker.gpu,
        "pattern_id": marker.pattern_id,
        "secondary_value": dict(marker.secondary_value),
    }


def _run_progress_summary_payload(summary: Any) -> dict[str, Any]:
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


def _job_metadata_payload(metadata: Any) -> dict[str, Any]:
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


def _registry_candidate_groups(bundle: L0Bundle) -> list[dict[str, Any]]:
    groups: dict[tuple[str | None, str], dict[str, Any]] = {}
    for match in bundle.registry_matches:
        key = (match.registry_id, match.signature)
        group = groups.get(key)
        if group is None:
            group = {
                "registry_id": match.registry_id,
                "fine_class_hint": match.fine_class,
                "signature_hint": match.signature,
                "first_line": match.line,
                "count": 0,
                "sample_lines": [],
                "causal_role_hint": _causal_role_hint(bundle, match.line),
                "provisional": True,
            }
            groups[key] = group
        group["count"] += 1
        if match.line is not None and len(group["sample_lines"]) < 5:
            group["sample_lines"].append(match.line)
        if _causal_role_hint(bundle, match.line) == CausalRole.TEARDOWN.value:
            group["causal_role_hint"] = CausalRole.TEARDOWN.value
    return list(groups.values())[:20]


def _registry_hint(bundle: L0Bundle, match: Any | None) -> dict[str, Any] | None:
    if match is None:
        return None
    return {
        "fine_class_hint": match.fine_class,
        "signature_hint": match.signature,
        "line": match.line,
        "rank": match.rank,
        "phase": match.phase,
        "failure_iteration": match.failure_iteration,
        "registry_id": match.registry_id,
        "causal_role_hint": _causal_role_hint(bundle, match.line),
        "provisional": True,
    }


def _causal_role_hint(bundle: L0Bundle, line: int | None) -> str:
    if line is None:
        return CausalRole.UNKNOWN.value
    for episode in bundle.failure_episodes:
        if episode.terminal_exception_line == line:
            return episode.terminal_exception_causal_role_hint
    return CausalRole.UNKNOWN.value
