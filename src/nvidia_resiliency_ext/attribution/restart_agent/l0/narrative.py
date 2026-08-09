# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic narrative and compact Decision Evidence projections for L0B."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..models import DecisionEvidence, FailureEvidence, L0Bundle

MODEL_VIEW_SAMPLE_LIMIT = 8


def build_failure_narrative(
    bundle: L0Bundle,
    decision_evidence: DecisionEvidence,
) -> dict[str, Any]:
    """Render typed L0 facts in a compact, deterministic causal reading order."""

    primary = decision_evidence.deterministic_primary_candidate
    observation = decision_evidence.selected_observed_failure
    selected = primary or observation
    events: list[dict[str, Any]] = []

    operation = _relevant_operation_fact(decision_evidence.operation_artifact_facts, selected)
    if operation is not None:
        success_lines = _integer_sequence(operation.get("success_lines"))
        success_count = _non_negative_int(operation.get("success_count")) or len(success_lines)
        if success_count:
            latest_success = max(success_lines, default=None)
            operation_name = str(operation.get("operation") or "operation")
            events.append(
                _event(
                    "prior_operation_success",
                    latest_success,
                    (
                        f"{operation_name} completed {success_count} prior "
                        f"time(s) in this log"
                        + (
                            f"; latest success was at line {latest_success}"
                            if latest_success is not None
                            else ""
                        )
                    ),
                    line_references=success_lines,
                    occurrence_count=success_count,
                    line_samples=success_lines,
                )
            )

    progress = decision_evidence.progress_checkpoint_state
    checkpoint_line = _positive_int(progress.get("last_checkpoint_line"))
    checkpoint_step = progress.get("last_checkpoint_iteration")
    if checkpoint_line is not None:
        events.append(
            _event(
                "last_checkpoint",
                checkpoint_line,
                _marker_summary("checkpoint", checkpoint_step, checkpoint_line),
            )
        )

    progress_line = _positive_int(progress.get("last_progress_line"))
    progress_step = progress.get("highest_completed_step")
    if progress_line is not None and progress_line != checkpoint_line:
        events.append(
            _event(
                "last_progress",
                progress_line,
                _marker_summary("progress", progress_step, progress_line),
            )
        )

    if operation is not None:
        start_line = _positive_int(operation.get("current_start_line"))
        if start_line is not None:
            events.append(
                _event(
                    "current_operation_start",
                    start_line,
                    f"{str(operation.get('operation') or 'operation')} started at line {start_line}",
                )
            )

    episode = _selected_episode(bundle, decision_evidence)
    if episode is not None and (selected is None or episode.first_exception_line != selected.line):
        events.append(
            _event(
                "fault_observation",
                episode.first_exception_line,
                f"failure episode began at line {episode.first_exception_line}",
                object_references=(episode.episode_id,),
            )
        )
    if episode is not None and episode.first_progress_after is not None:
        recovery_line = episode.first_progress_after.line
        events.append(
            _event(
                "same_attempt_recovery",
                recovery_line,
                f"application progress resumed at line {recovery_line}",
                object_references=(episode.episode_id,),
            )
        )

    if selected is not None:
        kind = "primary_failure" if primary is not None else "selected_observation"
        events.append(
            _event(
                kind,
                selected.line,
                _failure_summary(selected),
                object_references=_selected_object_references(decision_evidence),
            )
        )

    incident = _selected_incident(bundle, decision_evidence, selected)
    if incident is not None:
        events.append(
            _event(
                "distributed_fanout",
                incident.primary_observed_line,
                (
                    f"{incident.incident_type} was observed {incident.event_count} time(s) "
                    f"across {incident.observed_rank_count} rank(s)"
                ),
                object_references=(incident.incident_id,),
                occurrence_count=incident.event_count,
                rank_count=incident.observed_rank_count,
                unattributed_occurrence_count=_non_negative_int(
                    decision_evidence.locality.get("unattributed_root_occurrence_count")
                ),
                rank_samples=incident.rank_spread,
                line_samples=incident.sample_lines,
            )
        )

    cascades = [
        cascade
        for cascade in bundle.cascades
        if selected is None or selected.line is None or cascade.first_line >= selected.line
    ]
    if cascades:
        occurrence_count = sum(cascade.count for cascade in cascades)
        sample_lines = [line for cascade in cascades for line in cascade.sample_lines]
        events.append(
            _event(
                "cascade_summary",
                min(cascade.first_line for cascade in cascades),
                (
                    f"{occurrence_count} downstream cascade occurrence(s) were grouped "
                    f"into {len(cascades)} cascade class(es)"
                ),
                occurrence_count=occurrence_count,
                line_samples=sample_lines,
            )
        )

    teardown_lines = _teardown_lines(episode)
    if teardown_lines:
        events.append(
            _event(
                "teardown_summary",
                min(teardown_lines),
                f"teardown or process termination was observed at {len(teardown_lines)} line(s)",
                object_references=(episode.episode_id,) if episode is not None else (),
                occurrence_count=len(teardown_lines),
                line_samples=teardown_lines,
            )
        )

    progress_after = progress.get("progress_after_failure_episode")
    if isinstance(progress_after, bool):
        later_line = (
            episode.first_progress_after.line
            if progress_after and episode is not None and episode.first_progress_after is not None
            else None
        )
        events.append(
            _event(
                "later_progress_outcome",
                later_line,
                (
                    f"later application progress was observed at line {later_line}"
                    if later_line is not None
                    else "no later application progress was observed in the inspected log"
                ),
                object_references=(episode.episode_id,) if episode is not None else (),
            )
        )

    events = [{**event, "sequence": index} for index, event in enumerate(events, start=1)]
    known_unknowns: list[dict[str, Any]] = []
    if selected is not None and not bundle.cause_confirmations:
        known_unknowns.append(
            {
                "id": "no_typed_cause_confirmation_selected",
                "summary": "no typed cause-confirmation match was selected in the current log",
                "coverage_references": ["collection:cause_confirmations"],
            }
        )
    if selected is None and events:
        known_unknowns.append(
            {
                "id": "failure_identity_not_selected",
                "summary": "typed current-log facts did not select a failure identity",
                "coverage_references": ["decision_evidence:canonical_observed_identity"],
            }
        )

    if selected is not None:
        status = "available"
        identity_kind = "primary" if primary is not None else "observation_only"
    elif events:
        status = "partial"
        identity_kind = "none"
    else:
        status = "not_available"
        identity_kind = "none"
    return {
        "status": status,
        "identity_kind": identity_kind,
        "events": events,
        "known_unknowns": known_unknowns,
    }


def build_decision_evidence_view(decision_evidence: DecisionEvidence) -> dict[str, Any]:
    """Project exact internal Decision Evidence into a bounded model-facing view."""

    locality = dict(decision_evidence.locality)
    root_ranks = locality.pop("root_observer_ranks", None)
    rank_spread = locality.pop("rank_spread", None)
    root_rank_values = _string_sequence(root_ranks) if root_ranks is not None else None
    rank_spread_values = _string_sequence(rank_spread) if rank_spread is not None else None
    compact_locality = {
        **locality,
        "root_observer_count": len(root_rank_values) if root_rank_values is not None else None,
        "root_observer_rank_samples": _rank_samples(root_rank_values or ()),
        "rank_spread_count": len(rank_spread_values) if rank_spread_values is not None else None,
        "rank_spread_samples": _rank_samples(rank_spread_values or ()),
    }
    return {
        "source_schema_version": decision_evidence.schema_version,
        "deterministic_primary_candidate": _failure_view(
            decision_evidence.deterministic_primary_candidate
        ),
        "selected_observed_failure": _failure_view(decision_evidence.selected_observed_failure),
        "canonical_observed_identity": dict(decision_evidence.canonical_observed_identity),
        "selected_evidence_references": dict(decision_evidence.selected_evidence_references),
        "failure_position": dict(decision_evidence.failure_position),
        "progress_checkpoint_state": dict(decision_evidence.progress_checkpoint_state),
        "operation_artifact_facts": [
            _compact_mapping(dict(fact)) for fact in decision_evidence.operation_artifact_facts
        ],
        "later_progress_recovery": _compact_mapping(
            dict(decision_evidence.later_progress_recovery)
        ),
        "locality": compact_locality,
        "coverage_lossiness": dict(decision_evidence.coverage_lossiness),
        "provenance": dict(decision_evidence.provenance),
    }


def decision_evidence_view_is_consistent(
    decision_evidence: DecisionEvidence,
    view: Mapping[str, Any],
) -> bool:
    """Check that compaction preserved canonical scalar and count semantics."""

    if view.get("source_schema_version") != decision_evidence.schema_version:
        return False
    for name in (
        "canonical_observed_identity",
        "selected_evidence_references",
        "failure_position",
        "progress_checkpoint_state",
        "coverage_lossiness",
        "provenance",
    ):
        if dict(view.get(name) or {}) != dict(getattr(decision_evidence, name)):
            return False
    locality = view.get("locality")
    if not isinstance(locality, Mapping):
        return False
    exact_ranks = decision_evidence.locality.get("root_observer_ranks")
    expected_count = len(exact_ranks) if exact_ranks is not None else None
    return locality.get("root_observer_count") == expected_count


def narrative_references_resolve(bundle: L0Bundle, narrative: Mapping[str, Any]) -> bool:
    """Validate references emitted by deterministic narrative templates."""

    known_objects = {
        *(episode.episode_id for episode in bundle.failure_episodes),
        *(incident.incident_id for incident in bundle.distributed_failure_incidents),
        *(f"cascade-{index + 1}" for index in range(len(bundle.cascades))),
    }
    for event in narrative.get("events") or ():
        if not isinstance(event, Mapping):
            return False
        for reference in event.get("evidence_references") or ():
            if not isinstance(reference, str):
                return False
            if reference.startswith("line-"):
                try:
                    line = int(reference.removeprefix("line-"))
                except ValueError:
                    return False
                if line < 1 or line > bundle.line_count:
                    return False
            elif reference not in known_objects:
                return False
    return True


def _event(
    kind: str,
    line: int | None,
    summary: str,
    *,
    line_references: Sequence[int] = (),
    object_references: Sequence[str] = (),
    occurrence_count: int | None = None,
    rank_count: int | None = None,
    unattributed_occurrence_count: int | None = None,
    rank_samples: Sequence[str] = (),
    line_samples: Sequence[int] = (),
) -> dict[str, Any]:
    lines = [*line_references]
    if line is not None and line not in lines:
        lines.append(line)
    return {
        "sequence": 0,
        "kind": kind,
        "line": line,
        "summary": summary,
        "evidence_references": [
            *(f"line-{item}" for item in _bounded(sorted(set(lines)))),
            *_bounded(object_references),
        ],
        "occurrence_count": occurrence_count,
        "rank_count": rank_count,
        "unattributed_occurrence_count": unattributed_occurrence_count,
        "rank_samples": _rank_samples(_string_sequence(rank_samples)),
        "line_samples": _bounded(sorted(set(_integer_sequence(line_samples)))),
    }


def _failure_view(failure: FailureEvidence | None) -> dict[str, Any] | None:
    if failure is None:
        return None
    return {
        **failure.to_failure_payload(),
        "signature": failure.signature,
        "quote": failure.quote,
        "registry_id": failure.registry_id,
    }


def _failure_summary(failure: FailureEvidence) -> str:
    signature = failure.signature.strip().rstrip(":")
    label = signature if signature else failure.failure_class or "failure"
    if failure.line is not None:
        return f"{label} was selected at line {failure.line}"
    return f"{label} was selected"


def _marker_summary(kind: str, value: Any, line: int) -> str:
    if value is None:
        return f"last {kind} marker was observed at line {line}"
    return f"{kind} {value} completed at line {line}"


def _relevant_operation_fact(
    facts: Sequence[Mapping[str, Any]],
    selected: FailureEvidence | None,
) -> Mapping[str, Any] | None:
    if not facts:
        return None
    selected_line = selected.line if selected is not None else None

    def score(fact: Mapping[str, Any]) -> tuple[int, int]:
        failure_line = _positive_int(fact.get("failure_line"))
        start_line = _positive_int(fact.get("current_start_line"))
        exact_failure = int(selected_line is not None and failure_line == selected_line)
        active_before = int(
            selected_line is not None
            and start_line is not None
            and start_line <= selected_line
            and str(fact.get("current_outcome") or "unknown") != "completed"
        )
        return exact_failure * 2 + active_before, start_line or 0

    return max(facts, key=score)


def _selected_episode(bundle: L0Bundle, evidence: DecisionEvidence) -> Any | None:
    ids = list(evidence.selected_evidence_references.get("failure_episode_ids") or ())
    by_id = {episode.episode_id: episode for episode in bundle.failure_episodes}
    for episode_id in ids:
        if episode_id in by_id:
            return by_id[episode_id]
    selected = evidence.deterministic_primary_candidate or evidence.selected_observed_failure
    if selected is not None and selected.line is not None:
        return next(
            (
                episode
                for episode in bundle.failure_episodes
                if episode.start_line <= selected.line <= episode.end_line
            ),
            None,
        )
    return None


def _selected_incident(
    bundle: L0Bundle,
    evidence: DecisionEvidence,
    selected: FailureEvidence | None,
) -> Any | None:
    ids = list(evidence.selected_evidence_references.get("distributed_incident_ids") or ())
    by_id = {incident.incident_id: incident for incident in bundle.distributed_failure_incidents}
    for incident_id in ids:
        if incident_id in by_id:
            return by_id[incident_id]
    if selected is not None and selected.line is not None:
        return next(
            (
                incident
                for incident in bundle.distributed_failure_incidents
                if selected.line in incident.member_event_lines
                or selected.line == incident.primary_observed_line
            ),
            None,
        )
    return None


def _selected_object_references(evidence: DecisionEvidence) -> list[str]:
    references = evidence.selected_evidence_references
    return [
        *list(references.get("failure_episode_ids") or ()),
        *list(references.get("distributed_incident_ids") or ()),
    ]


def _teardown_lines(episode: Any | None) -> list[int]:
    if episode is None:
        return []
    return sorted(
        {
            line
            for line in (
                episode.first_teardown_line,
                episode.first_process_termination_line,
                episode.first_scheduler_cancel_line,
            )
            if line is not None
        }
    )


def _compact_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            compact[key] = _compact_mapping(item)
        elif isinstance(item, (list, tuple)):
            compact[f"{key}_count"] = len(item)
            compact[key] = _bounded(item)
        else:
            compact[key] = item
    return compact


def _bounded(value: Sequence[Any]) -> list[Any]:
    return list(value[:MODEL_VIEW_SAMPLE_LIMIT])


def _rank_samples(value: Sequence[str]) -> list[str]:
    def key(rank: str) -> tuple[int, int | str]:
        return (0, int(rank)) if rank.isdigit() else (1, rank)

    return sorted(set(value), key=key)[:MODEL_VIEW_SAMPLE_LIMIT]


def _integer_sequence(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, int) and not isinstance(item, bool)]


def _string_sequence(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
