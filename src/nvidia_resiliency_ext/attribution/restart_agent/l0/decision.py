# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Decision Evidence selection from the complete L0A bundle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from ..models import DecisionEvidence, FailureEpisode, L0Bundle


@dataclass(frozen=True)
class _DecisionSelection:
    primary: Any | None
    primary_line: int | None
    identity_line: int | None
    identity_reason: str
    episode: FailureEpisode | None
    incident: Any | None
    root_fingerprint: str | None
    root_fingerprint_source: str | None
    references: Mapping[str, Any]


def build_decision_evidence(bundle: L0Bundle) -> DecisionEvidence:
    """Select canonical policy-relevant facts and L0A references once."""

    selected = _select_decision_context(bundle)
    primary = selected.primary
    primary_line = selected.primary_line
    identity_line = selected.identity_line
    episode = selected.episode
    incident = selected.incident
    root_fingerprint = selected.root_fingerprint

    run = bundle.run_progress_summary
    progress = bundle.progress
    terminal_observations = [
        asdict(observation)
        for observation in bundle.later_progress_after_fault_observations
        if primary is not None
        and (
            observation.matches_terminal_fingerprint
            or observation.root_fingerprint == root_fingerprint
            or observation.fine_class == primary.fine_class
        )
    ]

    return DecisionEvidence(
        deterministic_primary_candidate=primary,
        canonical_observed_identity={
            "available": primary is not None,
            "fine_class": primary.fine_class if primary is not None else None,
            "signature": primary.signature if primary is not None else None,
            "identity_anchor_line": identity_line,
            "identity_anchor_reason": selected.identity_reason,
            "root_fingerprint": root_fingerprint,
            "root_fingerprint_source": selected.root_fingerprint_source,
            "registry_id": primary.registry_id if primary is not None else None,
        },
        selected_evidence_references=selected.references,
        failure_position={
            "primary_line": primary_line,
            "identity_anchor_line": identity_line,
            "failure_iteration": primary.failure_iteration if primary is not None else None,
            "phase": primary.phase if primary is not None else None,
            "fault_outcome": primary.fault_outcome if primary is not None else None,
            "causal_role": primary.causal_role if primary is not None else None,
            "data_position_fingerprint": (
                primary.data_position_fingerprint if primary is not None else None
            ),
            "first_terminal_incident_line": run.first_terminal_incident_line,
            "latest_observed_failure_iteration": run.latest_observed_failure_iteration,
            "latest_observed_failure_iteration_line": (run.latest_observed_failure_iteration_line),
        },
        progress_checkpoint_state={
            "first_iteration": run.first_iteration,
            "last_iteration": run.last_iteration,
            "iteration_delta": run.iteration_delta,
            "successful_runtime_seconds": run.successful_runtime_seconds,
            "highest_completed_step": progress.highest_completed_step,
            "last_progress_line": progress.last_progress_line,
            "latest_observed_failure_iteration": (progress.latest_observed_failure_iteration),
            "latest_observed_failure_iteration_line": (
                progress.latest_observed_failure_iteration_line
            ),
            "last_checkpoint_iteration": run.last_checkpoint_iteration,
            "last_checkpoint_line": run.last_checkpoint_line,
            "checkpoint_load_iteration": run.checkpoint_load_iteration,
            "checkpoint_load_line": run.checkpoint_load_line,
            "iterations_since_checkpoint": run.iterations_since_checkpoint,
            "observed_iterations_after_checkpoint_load": (
                run.observed_iterations_after_checkpoint_load
            ),
            "progress_after_failure_episode": run.progress_after_failure_episode,
            "progress_marker_count": run.progress_marker_count,
            "checkpoint_marker_count": run.checkpoint_marker_count,
            "setup_marker_count": run.setup_marker_count,
        },
        operation_artifact_facts=tuple(
            asdict(item) for item in bundle.operation_artifact_comparisons
        ),
        later_progress_recovery={
            "matching_observations": terminal_observations,
            "recovery_lines": list(progress.recovery_lines),
            "progress_after_failure_episode": run.progress_after_failure_episode,
        },
        locality={
            "rank": primary.rank if primary is not None else None,
            "node": primary.node if primary is not None else None,
            "gpu": primary.gpu if primary is not None else None,
            "distributed_incident_kind": (incident.incident_kind if incident is not None else None),
            "distributed_incident_type": (incident.incident_type if incident is not None else None),
            "observed_rank_count": (incident.observed_rank_count if incident is not None else 0),
            "rank_spread": list(incident.rank_spread) if incident is not None else [],
            "job_metadata": asdict(bundle.job_metadata),
        },
        coverage_lossiness={
            "evidence_coverage": dict(bundle.evidence_coverage),
            "selection_summary": dict(bundle.selection_summary),
        },
        provenance={
            "source": "l0a_deterministic_selection",
            "log_line_count": bundle.line_count,
            "log_byte_size": bundle.byte_size,
            "log_rescanned": False,
            "model_used": False,
        },
    )


def _select_decision_context(bundle: L0Bundle) -> _DecisionSelection:
    primary = bundle.deterministic_primary_candidate
    primary_line = primary.line if primary is not None else None
    identity_line = primary_line
    identity_reason = "no_deterministic_primary"
    if primary_line is not None:
        identity_line, identity_reason = canonical_identity_anchor_line(
            bundle,
            primary_line,
            selection_label="deterministic_primary",
        )

    episode = _failure_episode_for_lines(bundle, primary_line, identity_line)
    incident = (
        distributed_incident_for_line(bundle, identity_line) if identity_line is not None else None
    )
    identity_match = _registry_match_for_line(bundle, identity_line)
    root_fingerprint = None
    root_fingerprint_source = None
    if incident is not None and incident.history_fingerprint:
        root_fingerprint = incident.history_fingerprint
        root_fingerprint_source = incident.history_fingerprint_source
    elif identity_match is not None and identity_match.root_fingerprint:
        root_fingerprint = identity_match.root_fingerprint
        root_fingerprint_source = identity_match.root_fingerprint_source
    elif primary is not None:
        root_fingerprint = primary.root_fingerprint
        root_fingerprint_source = primary.root_fingerprint_source

    referenced_lines = {line for line in (primary_line, identity_line) if line is not None}
    if episode is not None:
        referenced_lines.update(episode.precursor_lines)
        referenced_lines.update(episode.exception_chain_lines)
        if episode.terminal_exception_line is not None:
            referenced_lines.add(episode.terminal_exception_line)
    anchors = tuple(
        anchor for anchor in bundle.candidate_anchors if anchor.line in referenced_lines
    )
    window_ids = {window_id for anchor in anchors for window_id in anchor.context_window_ids}
    if episode is not None:
        window_ids.update(episode.context_window_ids)
    window_ids.update(
        window.window_id
        for window in bundle.context_windows
        if any(window.start_line <= line <= window.end_line for line in referenced_lines)
    )
    occurrence_group_ids = {
        group.occurrence_group_id
        for group in bundle.occurrence_groups
        if bool(referenced_lines.intersection({group.first_line, *group.sample_lines}))
        or (
            primary is not None
            and primary.registry_id is not None
            and group.registry_id == primary.registry_id
        )
    }
    references = {
        "semantics": "provenance_only",
        "resolution": "get_evidence_objects_when_advertised",
        "source_lines": sorted(referenced_lines),
        "candidate_anchor_ids": [anchor.anchor_id for anchor in anchors],
        "context_window_ids": sorted(window_ids),
        "failure_episode_ids": [episode.episode_id] if episode is not None else [],
        "distributed_incident_ids": [incident.incident_id] if incident is not None else [],
        "occurrence_group_ids": sorted(occurrence_group_ids),
    }
    return _DecisionSelection(
        primary=primary,
        primary_line=primary_line,
        identity_line=identity_line,
        identity_reason=identity_reason,
        episode=episode,
        incident=incident,
        root_fingerprint=root_fingerprint,
        root_fingerprint_source=root_fingerprint_source,
        references=references,
    )


def canonical_identity_anchor_line(
    bundle: L0Bundle,
    line: int,
    *,
    selection_label: str,
) -> tuple[int, str]:
    """Return the stable identity anchor within the matching failure episode."""

    candidates: list[tuple[int, int, int, str]] = []
    for episode in bundle.failure_episodes:
        terminal_line = episode.terminal_exception_line
        if terminal_line is None:
            continue
        chain_end = max((*episode.exception_chain_lines, terminal_line))
        observed_episode_lines = {
            *episode.precursor_lines,
            *episode.exception_chain_lines,
            *(confirmation.line for confirmation in episode.cause_confirmations),
            terminal_line,
        }
        if line in observed_episode_lines or episode.start_line - 1 <= line <= chain_end:
            anchor_line = episode.identity_anchor_line or terminal_line
            candidates.append(
                (
                    abs(anchor_line - line),
                    anchor_line - min((*episode.precursor_lines, episode.start_line)),
                    anchor_line,
                    episode.identity_anchor_reason or "terminal_exception",
                )
            )
    if not candidates:
        return line, f"{selection_label}_line"

    _, _, anchor_line, anchor_reason = min(candidates)
    if anchor_line == line:
        return line, f"{selection_label}_is_episode_identity_anchor:{anchor_reason}"
    return anchor_line, f"failure_episode_identity_anchor:{anchor_reason}"


def distributed_incident_for_line(bundle: L0Bundle, line: int) -> Any | None:
    for incident in bundle.distributed_failure_incidents:
        if line == incident.primary_observed_line or line in incident.member_event_lines:
            return incident
    return None


def _failure_episode_for_lines(
    bundle: L0Bundle,
    *lines: int | None,
) -> FailureEpisode | None:
    observed_lines = {line for line in lines if line is not None}
    for episode in bundle.failure_episodes:
        episode_lines = {
            episode.first_exception_line,
            episode.terminal_exception_line,
            episode.identity_anchor_line,
            *episode.precursor_lines,
            *episode.exception_chain_lines,
        }
        if observed_lines.intersection(line for line in episode_lines if line is not None):
            return episode
        if any(episode.start_line <= line <= episode.end_line for line in observed_lines):
            return episode
    return None


def _registry_match_for_line(bundle: L0Bundle, line: int | None) -> Any | None:
    if line is None:
        return None
    return next((match for match in bundle.registry_matches if match.line == line), None)
