# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic evidence assembly for one interleaved training log."""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..identity import (
    canonical_observed_fingerprint,
    extract_data_position_fingerprint,
    extract_failure_iteration,
    extract_gpu,
    extract_node,
    extract_rank,
    fingerprint_for,
    normalized_pattern,
    path_hints,
)
from ..infrastructure.log_source import LogSnapshot, read_log_lines
from ..models import (
    ArtifactComparisonLevel,
    ArtifactObservationKind,
    AssessmentStatus,
    CandidateAnchor,
    CascadeEvidence,
    CausalRole,
    ContextWindow,
    CoverageStatus,
    DistributedFailureIncident,
    DistributedIncidentKind,
    FailureEpisode,
    FailureEvidence,
    FaultOutcome,
    JobMetadata,
    L0Bundle,
    LaterProgressAfterFaultObservation,
    LogLine,
    NormalizedOccurrenceGroup,
    OperationArtifactComparisonEvidence,
    PolicyClass,
    PostFaultSummary,
    ProgressFacts,
    ProgressMarker,
    RegistryRole,
    RunProgressSummary,
)
from .registry import (
    diagnostic_context_kind,
    diagnostic_uncertainty_kind,
    match_registry,
    root_fingerprint,
    signature_for,
)

_MEGATRON_ITERATION_RE = re.compile(
    r"^\s*(?:(?P<rank_prefix>\d+):\s*)?"
    r"\[(?P<timestamp>[^\]]+)\]\s+iteration\s+"
    r"(?P<iteration>\d+)\s*/\s*(?P<total_iterations>\d+)\s*\|\s*"
    r"consumed samples:\s*(?P<consumed_samples>\d+)\s*\|",
    re.I,
)
_CHECKPOINT_STEP_RE = re.compile(r"\bcheckpoint\b.*?\b(?:step|iter(?:ation)?)[=:\s#/]+(\d+)", re.I)
_CHECKPOINT_RE = re.compile(r"\bcheckpoint\b.*\b(?:saved|complete|completed|success|wrote)\b", re.I)
_CHECKPOINT_SAVED_ITERATION_RE = re.compile(
    r"\bsuccessfully saved checkpoint from iteration\s+(?P<iteration>\d+)\b"
    r"|\bsaved checkpoint from iteration\s+(?P<iteration_alt>\d+)\b",
    re.I,
)
_CHECKPOINT_SAVE_START_RE = re.compile(
    r"\bsaving checkpoint at iteration\s+(?P<iteration>\d+)\s+to\s+" r"(?P<artifact_path>\S+)",
    re.I,
)
_CHECKPOINT_LOADED_RE = re.compile(r"\bsuccessfully loaded checkpoint\b", re.I)
_CHECKPOINT_LOAD_COMPLETE_RE = re.compile(
    r"\bsuccessfully loaded checkpoint(?:\s+from\s+(?P<artifact_path>\S+))?.*?"
    r"\bat iteration\s+(?P<iteration>\d+)\b",
    re.I,
)
_CHECKPOINT_LOAD_START_RE = re.compile(
    r"\bloading\s+(?:a\s+)?(?:distributed\s+)?checkpoint\b"
    r"(?:\s+from\s+(?P<artifact_path>\S+))?.*?"
    r"\bat iteration\s+(?P<iteration>\d+)\b",
    re.I,
)
_DATALOADER_READ_EVENT_RE = re.compile(
    r"\b(?:dataloader|data loader|dataset reader)\b.*?"
    r"(?P<outcome>successfully\s+(?:read|loaded|opened)|finished\s+reading|"
    r"read\s+complete|reading|loading|opening|failed\s+(?:to\s+)?(?:read|load|open)|"
    r"error\s+(?:while\s+)?(?:reading|loading|opening))\b.*?"
    r"\b(?:file|shard|object|path)\b\s*(?:=|:|from)?\s*['\"]?"
    r"(?P<physical_unit>/[^'\"\s,\]\)]+)",
    re.I,
)
_CHECKPOINT_METADATA_LOADED_RE = re.compile(
    r"\b(?:checkpoint|sharded_state_dict)\s+metadata\s+loaded\b",
    re.I,
)
_CHECKPOINT_RESHARD_RE = re.compile(r"\bjob sharding has changed\b", re.I)
_OPTIMIZER_SETUP_RE = re.compile(r"\bsetting up optimizer\b", re.I)
_CUDA_GRAPH_BUILT_RE = re.compile(r"\bbuilt cuda graph(?:\(s\))?\b", re.I)
_RECOVERY_RE = re.compile(
    r"\b(?:retry succeeded|recovered|continuing|skipping|quarantine|resumed|successfully retried)\b",
    re.I,
)
_TERMINAL_RE = re.compile(
    r"\b(?:fatal|failed|aborted|terminated|exiting|uncaught|raise|runtimeerror|traceback)\b",
    re.I,
)
_PHASE_CHECKPOINT_RE = re.compile(r"\bcheckpoint\b", re.I)
_TRAINING_ITERATIONS_TOTAL_RE = re.compile(r"\bsetting training iterations to (?P<total>\d+)\b")
_TRAINING_START_RE = re.compile(r"\[before the start of training step\] datetime:", re.I)
_RERUN_ITERATION_RESET_RE = re.compile(
    r"\bOverwriting rerun_state_machine\.current_iteration from -?\d+ to -?\d+",
    re.I,
)
_RANK_GPU_MAPPING_WARNING_RE = re.compile(
    r"Guessing device ID based on global rank\. This can cause a hang if rank to GPU "
    r"mapping is heterogeneous",
    re.I,
)
_NCCL_VERSION_RE = re.compile(r"\bNCCL version (?P<version>\S+)")
_WORLD_SIZE_RE = re.compile(r"\bworld_size\b\s*(?:[.=:\-\s]+)\s*(?P<world_size>\d+)\b", re.I)
_HIGH_SIGNAL_RE = re.compile(
    r"\b(?:CRITICAL|FATAL|ERROR|Traceback|RuntimeError|AssertionError|AcceleratorError)\b"
    r"|(?:raising|raised)\s+.*\b(?:gpu|accelerator|device)\b.*\berror\b"
    r"|\b(?:assert(?:ion)? failed|out of bounds|bounds failure|timeout)\b",
    re.I,
)
_STRUCTURED_LOG_PREFIX_RE = re.compile(
    r"^\s*(?:\d+:\s*)?(?:\[[^\]]+\]\s*)?"
    r"(?P<severity>DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL):"
    r"(?P<logger>[A-Za-z_][A-Za-z0-9_.-]*):(?P<message>.*)$",
    re.I,
)
_FAILURE_ANNOUNCEMENT_RE = re.compile(
    r"\b(?:raising|raised|injecting|injected|simulating|simulated)\b.*"
    r"\b(?:error|failure|fault)\b",
    re.I,
)
_EXCEPTION_SUMMARY_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):",
)
_WATCHDOG_EXCEPTION_RE = re.compile(
    r"\bwatchdog thread terminated with exception:\s*"
    r"(?P<message>.*\b(?:CUDA|NCCL|c10)\s+error:.*)$",
    re.I,
)
_CUDA_RUNTIME_STATUS_RE = re.compile(r"\bcudaError[A-Za-z0-9_]*:", re.I)
_TRACEBACK_RE = re.compile(r"\bTraceback \(most recent call last\):", re.I)
_TERMINAL_OPERATION_TIMEOUT_RE = re.compile(
    r"\b(?:watchdog\s+)?(?:caught\s+)?(?:collective\s+)?operation\s+timeout\b"
    r"|\boperation\b.*\btimed out\b",
    re.I,
)
_TIMED_OPERATION_SIGNATURE_RE = re.compile(
    r"\b(?:watchdog\s+caught\s+)?collective\s+operation\s+timeout:?\s*"
    r"(?P<operation>Work[A-Za-z0-9_]+\([^)]*\))",
    re.I,
)
_TIMED_OPERATION_FIELD_RE = re.compile(
    r"\b(?P<name>SeqNum|OpType)=(?P<value>[A-Za-z0-9_.-]+)\b",
    re.I,
)
_TIME_OF_DAY_RE = re.compile(
    r"(?<!\d)(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})" r"(?:\.(?P<fraction>\d+))?",
)
_TIMEOUT_MS_RE = re.compile(r"\bTimeout\(ms\)=(?P<milliseconds>\d+)\b", re.I)
_PROCESS_GROUP_TYPE_RE = re.compile(
    r"\bPG\s+GUID\s+\d+\((?P<group_type>[^)]+)\)",
    re.I,
)
_EXCEPTION_CHAIN_RE = re.compile(
    r"The above exception was the direct cause of the following exception"
    r"|During handling of the above exception, another exception occurred",
    re.I,
)
_TEARDOWN_RE = re.compile(
    r"\b(?:destroy_process_group|program exit|before program exit|exiting)\b",
    re.I,
)
_BARE_PROCESS_KILLED_RE = re.compile(r"^\s*(?:\d+:\s*)?Killed\s*$", re.I)
_PROCESS_TERMINATION_RE = re.compile(
    r"\b(?:Producer process has been terminated|process has been terminated|terminated)\b"
    r"|Fatal Python error:\s*(?:Aborted|Segmentation fault)\b",
    re.I,
)
_SCHEDULER_CANCEL_RE = re.compile(
    r"\b(?:CANCELLED AT|STEP\b.*\bCANCELLED|slurmstepd\b.*\bCANCELLED)\b",
    re.I,
)
_CLEANUP_FRAME_RE = re.compile(
    r"\b(?:_run_finalizers|finalizer|_cleanup|cleanup|shutdown|atexit|"
    r"destroy_process_group|sem_unlink)\b",
    re.I,
)
_ITERATION_VALUE_RE = re.compile(r"\biteration\s+(?P<iteration>\d+)\b", re.I)
_CONFIG_PATH_RE = re.compile(
    r"^\s*(?:\d+:\s*)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s+\.{2,}\s+" r"(?P<path>/\S+)"
)
_INLINE_PATH_RE = re.compile(
    r"\b(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<quote>['\"]?)" r"(?P<path>/[^\s'\"(),]+)(?P=quote)"
)
_PERMISSION_DENIED_PATH_RE = re.compile(
    r"\b(?:PermissionError\b.*\bpermission denied|permission denied)\b.*?"
    r"(?P<quote>['\"])(?P<path>/[^'\"]+)(?P=quote)",
    re.I,
)
_USER_NAMESPACE_RE = re.compile(r"/users/(?P<namespace>[^/]+)/", re.I)
_READ_PATH_KEYS = {
    "data_path",
    "load",
    "per_split_data_args_path",
    "pretrained_checkpoint",
    "tokenizer_model",
}
_WRITE_PATH_KEYS = {
    "data_cache_path",
    "log_dir",
    "output_dir",
    "save",
    "tensorboard_dir",
}
_CACHE_PATH_KEYS = {"cache_dir", "hf_home", "path_to_cache"}
CONTEXT_WINDOW_BEFORE_LINES = 40
CONTEXT_WINDOW_AFTER_LINES = 140
MAX_CONTEXT_WINDOW_SEEDS = 8
MAX_CONTEXT_HIGH_SIGNAL_SEEDS = 3
MAX_CANDIDATE_ANCHORS = 16
MAX_HIGH_SIGNAL_ANCHORS = 8
MAX_FAILURE_EPISODES = 3
HIGH_SIGNAL_CLUSTER_GAP_LINES = 25
NEARBY_EPISODE_PRECURSOR_LINES = 8
REGISTRY_MATCH_HEAD_PER_PATTERN = 5
REGISTRY_MATCH_TAIL_PER_PATTERN = 2
DISTRIBUTED_TIMEOUT_WAVE_SECONDS = 5.0
MAX_DISTRIBUTED_TIMEOUT_WAVE_SECONDS = 60.0
MAX_DISTRIBUTED_INCIDENT_SAMPLE_LINES = 12
MAX_DISTRIBUTED_INCIDENT_RANK_SAMPLES = 16
MAX_PATH_ACCESS_FACTS = 40


@dataclass(frozen=True)
class _OccurrenceSeed:
    detector_id: str
    normalized_shape: str
    line: int
    classification: str
    registry_id: str | None = None
    rank: str | None = None
    node: str | None = None
    gpu: str | None = None


@dataclass(frozen=True)
class _TimedOperationEvent:
    line: int
    text: str
    timestamp_seconds: float | None
    timestamp_text: str | None
    configured_timeout_seconds: float | None
    sequence_number: str
    operation_type: str
    rank: str | None


@dataclass(frozen=True)
class _DetectedEvidence:
    path_access_facts: tuple[Mapping[str, object], ...]
    path_namespace_summary: Mapping[str, object]
    progress: ProgressFacts
    all_registry_matches: tuple[FailureEvidence, ...]
    all_cause_confirmations: tuple[FailureEvidence, ...]
    cause_confirmations: tuple[FailureEvidence, ...]
    high_signal_lines: tuple[int, ...]
    occurrence_groups: tuple[NormalizedOccurrenceGroup, ...]
    distributed_failure_incidents: tuple[DistributedFailureIncident, ...]
    registry_matches: tuple[FailureEvidence, ...]
    dropped_registry_matches: int
    primary: FailureEvidence | None
    preliminary_failure_episodes: tuple[FailureEpisode, ...]
    prompt_high_signal_lines: tuple[int, ...]


@dataclass(frozen=True)
class _ContextualEvidence:
    primary: FailureEvidence | None
    context_windows: tuple[ContextWindow, ...]
    failure_episodes: tuple[FailureEpisode, ...]
    cascades: tuple[CascadeEvidence, ...]
    post_fault_summaries: tuple[PostFaultSummary, ...]
    candidate_anchors: tuple[CandidateAnchor, ...]
    later_progress_after_fault_observations: tuple[LaterProgressAfterFaultObservation, ...]
    job_metadata: JobMetadata
    run_progress_summary: RunProgressSummary
    operation_artifact_comparisons: tuple[OperationArtifactComparisonEvidence, ...]


def build_l0_bundle(
    log_path: str | Path,
    *,
    source_log: LogSnapshot | None = None,
) -> L0Bundle:
    """Build current-log evidence without job history or policy context."""

    log_path = str(log_path)
    path = Path(log_path)
    if source_log is not None and source_log.path != log_path:
        raise ValueError("source log snapshot path does not match log_path")
    byte_size = source_log.byte_size if source_log is not None else path.stat().st_size
    lines = list(source_log.log_lines()) if source_log is not None else _read_lines(path)
    detected = _detect_evidence(lines)
    contextual = _contextualize_evidence(lines, detected)
    return _assemble_bundle(log_path, byte_size, lines, detected, contextual)


def _detect_evidence(lines: Sequence[LogLine]) -> _DetectedEvidence:
    path_access_facts, path_namespace_summary = _collect_path_access_facts(lines)
    progress, deterministic_occurrence_seeds = _collect_progress(lines)
    terminal_lines = tuple(item.line for item in lines if _TERMINAL_RE.search(item.text))
    episode_event_lines = _episode_event_lines(lines)
    all_registry_matches = tuple(_collect_registry_matches(lines, progress, terminal_lines))
    all_cause_confirmations = tuple(
        match
        for match in all_registry_matches
        if match.role == RegistryRole.CAUSE_CONFIRMATION.value
    )
    cause_confirmations = tuple(_sample_cause_confirmations(all_cause_confirmations))
    diagnostic_occurrence_seeds = _collect_diagnostic_occurrence_seeds(lines)
    high_signal_lines = _collect_high_signal_lines(lines)
    occurrence_groups = tuple(
        _build_occurrence_groups(
            all_registry_matches,
            (*deterministic_occurrence_seeds, *diagnostic_occurrence_seeds),
        )
    )
    distributed_failure_incidents = tuple(
        _build_distributed_failure_incidents(lines, progress, all_registry_matches)
    )
    registry_matches, dropped_registry_matches = _compact_registry_matches(all_registry_matches)
    primary = _select_primary_candidate(registry_matches)
    preliminary_failure_episodes = tuple(
        _build_failure_episodes(
            lines,
            high_signal_lines,
            registry_matches,
            progress,
            context_windows=(),
            event_lines=episode_event_lines,
            cause_confirmations=cause_confirmations,
            distributed_failure_incidents=distributed_failure_incidents,
        )
    )
    secondary_episode_lines = _secondary_episode_high_signal_lines(
        lines,
        preliminary_failure_episodes,
    )
    prompt_high_signal_lines = tuple(
        line_no for line_no in high_signal_lines if line_no not in secondary_episode_lines
    )
    return _DetectedEvidence(
        path_access_facts=path_access_facts,
        path_namespace_summary=path_namespace_summary,
        progress=progress,
        all_registry_matches=all_registry_matches,
        all_cause_confirmations=all_cause_confirmations,
        cause_confirmations=cause_confirmations,
        high_signal_lines=high_signal_lines,
        occurrence_groups=occurrence_groups,
        distributed_failure_incidents=distributed_failure_incidents,
        registry_matches=tuple(registry_matches),
        dropped_registry_matches=dropped_registry_matches,
        primary=primary,
        preliminary_failure_episodes=preliminary_failure_episodes,
        prompt_high_signal_lines=prompt_high_signal_lines,
    )


def _contextualize_evidence(
    lines: Sequence[LogLine],
    detected: _DetectedEvidence,
) -> _ContextualEvidence:
    episode_event_lines = _episode_event_lines(lines)
    context_windows = tuple(
        _build_context_windows(
            lines,
            detected.occurrence_groups,
            detected.registry_matches,
            detected.prompt_high_signal_lines,
            failure_episode_lines=_failure_episode_seed_lines(
                detected.preliminary_failure_episodes
            ),
            cause_confirmation_lines=tuple(
                match.line for match in detected.cause_confirmations if match.line is not None
            ),
        )
    )
    failure_episodes = tuple(
        _build_failure_episodes(
            lines,
            detected.high_signal_lines,
            detected.registry_matches,
            detected.progress,
            context_windows=context_windows,
            event_lines=episode_event_lines,
            cause_confirmations=detected.cause_confirmations,
            distributed_failure_incidents=detected.distributed_failure_incidents,
        )
    )
    primary = _canonicalize_episode_primary(
        detected.primary,
        detected.registry_matches,
        failure_episodes,
        lines,
        detected.progress,
    )
    primary = _apply_distributed_incident_identity(
        primary,
        detected.distributed_failure_incidents,
    )
    cascades = tuple(
        build_cascades_for_primary(
            detected.all_registry_matches,
            primary,
            detected.distributed_failure_incidents,
        )
    )
    post_fault_summaries = tuple(
        _build_post_fault_summaries(
            lines,
            detected.prompt_high_signal_lines,
            failure_episodes,
        )
    )
    candidate_anchors = tuple(
        _build_candidate_anchors(
            lines,
            detected.prompt_high_signal_lines,
            detected.registry_matches,
            primary,
            detected.progress,
            context_windows,
            failure_episodes,
            detected.cause_confirmations,
        )
    )
    later_progress = tuple(
        _build_later_progress_after_fault_observations(
            detected.all_registry_matches,
            detected.progress,
            failure_episodes,
        )
    )
    job_metadata = _collect_job_metadata(lines)
    run_progress_summary = _build_run_progress_summary(
        detected.progress,
        failure_episodes,
        detected.distributed_failure_incidents,
    )
    operation_artifact_comparisons = tuple(
        _build_operation_artifact_comparisons(
            lines,
            detected.progress,
            primary,
            detected.distributed_failure_incidents,
        )
    )
    return _ContextualEvidence(
        primary=primary,
        context_windows=context_windows,
        failure_episodes=failure_episodes,
        cascades=cascades,
        post_fault_summaries=post_fault_summaries,
        candidate_anchors=candidate_anchors,
        later_progress_after_fault_observations=later_progress,
        job_metadata=job_metadata,
        run_progress_summary=run_progress_summary,
        operation_artifact_comparisons=operation_artifact_comparisons,
    )


def _assemble_bundle(
    log_path: str,
    byte_size: int,
    lines: Sequence[LogLine],
    detected: _DetectedEvidence,
    contextual: _ContextualEvidence,
) -> L0Bundle:
    coverage = dict(
        _coverage(
            path_hint_count=len(path_hints(log_path)),
            path_access_fact_count=len(detected.path_access_facts),
            occurrence_group_count=len(detected.occurrence_groups),
            context_count=len(contextual.context_windows),
            candidate_anchor_count=len(contextual.candidate_anchors),
            failure_episode_count=len(contextual.failure_episodes),
            distributed_incident_count=len(detected.distributed_failure_incidents),
            progress=detected.progress,
            job_metadata=contextual.job_metadata,
            primary=contextual.primary,
            cascade_count=len(contextual.cascades),
        )
    )
    coverage["operation_artifact_comparisons"] = (
        "found" if contextual.operation_artifact_comparisons else "not_found"
    )
    return L0Bundle(
        log_path=log_path,
        byte_size=byte_size,
        line_count=len(lines),
        path_hints=tuple(path_hints(log_path)),
        path_access_facts=detected.path_access_facts,
        path_namespace_summary=detected.path_namespace_summary,
        occurrence_groups=detected.occurrence_groups,
        context_windows=contextual.context_windows,
        candidate_anchors=contextual.candidate_anchors,
        registry_matches=detected.registry_matches,
        deterministic_primary_candidate=contextual.primary,
        cascades=contextual.cascades,
        cause_confirmations=detected.cause_confirmations,
        failure_episodes=contextual.failure_episodes,
        distributed_failure_incidents=detected.distributed_failure_incidents,
        post_fault_summaries=contextual.post_fault_summaries,
        progress=detected.progress,
        run_progress_summary=contextual.run_progress_summary,
        operation_artifact_comparisons=contextual.operation_artifact_comparisons,
        later_progress_after_fault_observations=(
            contextual.later_progress_after_fault_observations
        ),
        job_metadata=contextual.job_metadata,
        evidence_coverage=coverage,
        selection_summary=_selection_summary(lines, detected, contextual),
        anomalies={"line_numbering": _line_numbering_anomaly()},
    )


def _selection_summary(
    lines: Sequence[LogLine],
    detected: _DetectedEvidence,
    contextual: _ContextualEvidence,
) -> Mapping[str, object]:
    run = contextual.run_progress_summary
    operations = contextual.operation_artifact_comparisons
    later_progress = contextual.later_progress_after_fault_observations
    return {
        "raw_lines": len(lines),
        "candidate_lines_after_filters": len(detected.all_registry_matches),
        "retained_registry_matches": len(detected.registry_matches),
        "dropped_duplicate_registry_matches": detected.dropped_registry_matches,
        "candidate_anchors": len(contextual.candidate_anchors),
        "failure_episodes": len(contextual.failure_episodes),
        "distributed_failure_incidents": len(detected.distributed_failure_incidents),
        "high_signal_lines": len(detected.high_signal_lines),
        "occurrence_groups": len(detected.occurrence_groups),
        "world_size_source": contextual.job_metadata.world_size_source,
        "observed_rank_count": contextual.job_metadata.observed_rank_count,
        "progress_marker_count": run.progress_marker_count,
        "checkpoint_marker_count": run.checkpoint_marker_count,
        "setup_marker_count": run.setup_marker_count,
        "latest_observed_failure_iteration": run.latest_observed_failure_iteration,
        "observed_iterations_after_checkpoint_load": run.observed_iterations_after_checkpoint_load,
        "operation_artifact_comparison_count": len(operations),
        "exact_physical_unit_comparison_count": sum(
            1
            for item in operations
            if item.comparison_level == ArtifactComparisonLevel.EXACT_PHYSICAL_UNIT.value
        ),
        "later_progress_after_fault_observation_count": len(later_progress),
        "later_progress_after_fault_event_count": sum(item.event_count for item in later_progress),
        "cause_confirmation_count": len(detected.all_cause_confirmations),
        "path_access_fact_count": len(detected.path_access_facts),
        "path_namespace_mismatch_observed": bool(
            detected.path_namespace_summary.get("failed_vs_configured_write_mismatch")
        ),
        "dropped_noise_lines": 0,
        "sampled_candidate_lines": sum(
            max(0, group.count - len(group.sample_lines)) for group in detected.occurrence_groups
        ),
        "caps_hit": (["registry_matches_per_pattern"] if detected.dropped_registry_matches else []),
        "primary_after_context_available": _has_after_context(contextual.primary, lines),
        "cited_error_only_evidence": False,
    }


def _line_numbering_anomaly() -> Mapping[str, str]:
    return {
        "scheme": "python_text_universal_newlines",
        "line_field": (
            "1-based logical line after Python text-mode universal-newline "
            "splitting on LF, CR, or CRLF"
        ),
        "shell_tool_note": (
            "Rank prefixes also look like '<rank>:'. Tools that count only "
            "LF-delimited records, or render embedded CR-delimited records, "
            "may show different line numbers for the same text."
        ),
    }


def _read_lines(path: Path) -> list[LogLine]:
    return read_log_lines(path)


def _collect_path_access_facts(
    lines: Sequence[LogLine],
) -> tuple[tuple[Mapping[str, object], ...], Mapping[str, object]]:
    facts: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    def add_fact(
        item: LogLine,
        *,
        path: str,
        role: str,
        access_intent: str,
        source: str,
    ) -> None:
        normalized_path = path.rstrip(":;].")
        key = (role, normalized_path)
        if key in seen or len(facts) >= MAX_PATH_ACCESS_FACTS:
            return
        seen.add(key)
        namespace_match = _USER_NAMESPACE_RE.search(normalized_path)
        facts.append(
            {
                "line": item.line,
                "path": normalized_path,
                "role": role,
                "access_intent": access_intent,
                "source": source,
                "user_namespace": (namespace_match.group("namespace") if namespace_match else None),
            }
        )

    for item in lines:
        permission_match = _PERMISSION_DENIED_PATH_RE.search(item.text)
        if permission_match is not None:
            failed_path = permission_match.group("path")
            add_fact(
                item,
                path=failed_path,
                role="failed_access",
                access_intent=(
                    "write_or_create" if failed_path.lower().endswith(".lock") else "unknown"
                ),
                source="permission_denied_exception",
            )

        config_match = _CONFIG_PATH_RE.search(item.text)
        if config_match is not None:
            role, access_intent = _configured_path_role(config_match.group("key"))
            if role is not None:
                add_fact(
                    item,
                    path=config_match.group("path"),
                    role=role,
                    access_intent=access_intent,
                    source=f"config:{config_match.group('key').lower()}",
                )

        for inline_match in _INLINE_PATH_RE.finditer(item.text):
            role, access_intent = _configured_path_role(inline_match.group("key"))
            if role is None:
                continue
            add_fact(
                item,
                path=inline_match.group("path"),
                role=role,
                access_intent=access_intent,
                source=f"inline_config:{inline_match.group('key').lower()}",
            )

    namespaces_by_role: dict[str, list[str]] = {}
    for fact in facts:
        namespace = fact.get("user_namespace")
        if not namespace:
            continue
        role = str(fact["role"])
        namespaces_by_role.setdefault(role, [])
        if namespace not in namespaces_by_role[role]:
            namespaces_by_role[role].append(str(namespace))
    for namespaces in namespaces_by_role.values():
        namespaces.sort()

    failed_namespaces = set(namespaces_by_role.get("failed_access", []))
    configured_write_namespaces = set(namespaces_by_role.get("configured_write", []))
    configured_write_namespaces.update(namespaces_by_role.get("configured_cache_write", []))
    all_namespaces = sorted(
        {namespace for namespaces in namespaces_by_role.values() for namespace in namespaces}
    )
    summary: dict[str, object] = {
        "namespaces_by_role": namespaces_by_role,
        "all_user_namespaces": all_namespaces,
        "cross_namespace_paths_observed": len(all_namespaces) > 1,
        "failed_vs_configured_write_mismatch": bool(
            failed_namespaces
            and configured_write_namespaces
            and failed_namespaces.isdisjoint(configured_write_namespaces)
        ),
        "effective_user": None,
        "ownership_verified": False,
        "interpretation": (
            "Path namespaces are deterministic string evidence only; they do not prove "
            "the effective process user, file owner, mode, or ACL."
        ),
    }
    return tuple(facts), summary


def _configured_path_role(key: str) -> tuple[str | None, str]:
    normalized = key.lower()
    if normalized in _READ_PATH_KEYS:
        return "configured_read", "read"
    if normalized in _WRITE_PATH_KEYS:
        return "configured_write", "write"
    if normalized in _CACHE_PATH_KEYS:
        return "configured_cache_write", "write_or_create"
    return None, "unknown"


def _collect_progress(
    lines: Iterable[LogLine],
) -> tuple[ProgressFacts, tuple[_OccurrenceSeed, ...]]:
    progress_lines: list[int] = []
    checkpoint_lines: list[int] = []
    setup_lines: list[int] = []
    recovery_lines: list[int] = []
    progress_markers: list[ProgressMarker] = []
    checkpoint_markers: list[ProgressMarker] = []
    setup_markers: list[ProgressMarker] = []
    occurrence_seeds: list[_OccurrenceSeed] = []
    seen_setup_markers: set[tuple[str, int | str | None]] = set()
    highest_completed_step: int | None = None
    last_progress_line: int | None = None
    last_checkpoint_step: int | None = None
    last_checkpoint_line: int | None = None
    latest_observed_failure_iteration: int | None = None
    latest_observed_failure_iteration_line: int | None = None

    for item in lines:
        failure_iteration = _observed_failure_iteration(item.text)
        if failure_iteration is not None:
            latest_observed_failure_iteration = failure_iteration
            latest_observed_failure_iteration_line = item.line

        iteration_marker = _iteration_progress_marker(item, len(progress_markers) + 1)
        if iteration_marker is not None:
            progress_lines.append(item.line)
            progress_markers.append(iteration_marker)
            highest_completed_step = max(highest_completed_step or 0, int(iteration_marker.value))
            last_progress_line = item.line
            occurrence_seeds.append(
                _progress_occurrence_seed(
                    item,
                    detector_id="megatron_iteration_summary.v1",
                    normalized_shape="megatron iteration summary",
                    classification="progress",
                )
            )

        checkpoint_marker = _checkpoint_progress_marker(item, len(checkpoint_markers) + 1)
        if checkpoint_marker is not None:
            checkpoint_lines.append(item.line)
            checkpoint_markers.append(checkpoint_marker)
            checkpoint_step = int(checkpoint_marker.value)
            last_checkpoint_line = item.line
            last_checkpoint_step = max(last_checkpoint_step or 0, checkpoint_step)
            occurrence_seeds.append(
                _progress_occurrence_seed(
                    item,
                    detector_id="checkpoint_complete.v1",
                    normalized_shape="checkpoint completion",
                    classification="checkpoint",
                )
            )

        setup_detection = _setup_progress_marker(item, len(setup_markers) + 1)
        if setup_detection is not None:
            setup_marker, marker_key, pattern_id, normalized = setup_detection
            if marker_key not in seen_setup_markers:
                seen_setup_markers.add(marker_key)
                setup_lines.append(item.line)
                setup_markers.append(setup_marker)
            occurrence_seeds.append(
                _progress_occurrence_seed(
                    item,
                    detector_id=pattern_id,
                    normalized_shape=normalized,
                    classification="setup_progress",
                )
            )

        if _RECOVERY_RE.search(item.text):
            recovery_lines.append(item.line)
            occurrence_seeds.append(
                _progress_occurrence_seed(
                    item,
                    detector_id="recovery_marker.v1",
                    normalized_shape="recovery marker",
                    classification="recovery",
                )
            )

    return (
        ProgressFacts(
            highest_completed_step=highest_completed_step,
            last_progress_line=last_progress_line,
            last_checkpoint_step=last_checkpoint_step,
            last_checkpoint_line=last_checkpoint_line,
            latest_observed_failure_iteration=latest_observed_failure_iteration,
            latest_observed_failure_iteration_line=latest_observed_failure_iteration_line,
            progress_lines=tuple(progress_lines),
            checkpoint_lines=tuple(checkpoint_lines),
            setup_lines=tuple(setup_lines),
            recovery_lines=tuple(recovery_lines),
            progress_markers=tuple(progress_markers),
            checkpoint_markers=tuple(checkpoint_markers),
            setup_markers=tuple(setup_markers),
        ),
        tuple(occurrence_seeds),
    )


def _iteration_progress_marker(item: LogLine, marker_number: int) -> ProgressMarker | None:
    match = _MEGATRON_ITERATION_RE.search(item.text)
    if match is None:
        return None
    return ProgressMarker(
        marker_id=f"pm-{marker_number}",
        marker_type="iteration",
        value=int(match.group("iteration")),
        state="completed",
        line=item.line,
        quote=item.text,
        timestamp=match.group("timestamp"),
        rank=extract_rank(item.text),
        node=extract_node(item.text),
        gpu=extract_gpu(item.text),
        pattern_id="megatron_iteration_summary.v1",
        secondary_value={
            "total_iterations": int(match.group("total_iterations")),
            "consumed_samples": int(match.group("consumed_samples")),
        },
    )


def _checkpoint_progress_marker(item: LogLine, marker_number: int) -> ProgressMarker | None:
    step = _checkpoint_step(item.text)
    if step is None:
        return None
    return ProgressMarker(
        marker_id=f"ckpt-{marker_number}",
        marker_type="checkpoint",
        value=step,
        state="completed",
        line=item.line,
        quote=item.text,
        rank=extract_rank(item.text),
        node=extract_node(item.text),
        gpu=extract_gpu(item.text),
        pattern_id="checkpoint_complete.v1",
    )


def _setup_progress_marker(
    item: LogLine,
    marker_number: int,
) -> tuple[ProgressMarker, tuple[str, int | str | None], str, str] | None:
    detected = _setup_marker(item.text)
    if detected is None:
        return None
    marker_type, value, state, pattern_id, normalized = detected
    marker = ProgressMarker(
        marker_id=f"setup-{marker_number}",
        marker_type=marker_type,
        value=value,
        state=state,
        line=item.line,
        quote=item.text,
        rank=extract_rank(item.text),
        node=extract_node(item.text),
        gpu=extract_gpu(item.text),
        pattern_id=pattern_id,
    )
    return marker, (marker_type, value), pattern_id, normalized


def _progress_occurrence_seed(
    item: LogLine,
    *,
    detector_id: str,
    normalized_shape: str,
    classification: str,
) -> _OccurrenceSeed:
    return _OccurrenceSeed(
        detector_id=detector_id,
        normalized_shape=normalized_shape,
        line=item.line,
        classification=classification,
        rank=extract_rank(item.text),
        node=extract_node(item.text),
        gpu=extract_gpu(item.text),
    )


def _checkpoint_step(text: str) -> int | None:
    saved_iteration = _CHECKPOINT_SAVED_ITERATION_RE.search(text)
    if saved_iteration:
        value = saved_iteration.group("iteration") or saved_iteration.group("iteration_alt")
        return int(value)
    if not _CHECKPOINT_RE.search(text):
        return None
    explicit_step = _CHECKPOINT_STEP_RE.search(text)
    if explicit_step:
        return int(explicit_step.group(1))
    return None


def _setup_marker(
    text: str,
) -> tuple[str, int | str | None, str, str, str] | None:
    if _CHECKPOINT_LOADED_RE.search(text):
        iteration = _ITERATION_VALUE_RE.search(text)
        return (
            "checkpoint_load",
            int(iteration.group("iteration")) if iteration else None,
            "completed",
            "checkpoint_load_complete.v1",
            "checkpoint load completed",
        )
    checkpoint_start = _CHECKPOINT_LOAD_START_RE.search(text)
    if checkpoint_start:
        return (
            "checkpoint_load_start",
            int(checkpoint_start.group("iteration")),
            "started",
            "checkpoint_load_start.v1",
            "checkpoint load started",
        )
    if _CHECKPOINT_RESHARD_RE.search(text):
        return (
            "checkpoint_reshard",
            None,
            "observed",
            "checkpoint_reshard.v1",
            "checkpoint sharding change observed",
        )
    if _CHECKPOINT_METADATA_LOADED_RE.search(text):
        return (
            "checkpoint_metadata_load",
            None,
            "completed",
            "checkpoint_metadata_load_complete.v1",
            "checkpoint metadata load completed",
        )
    if _OPTIMIZER_SETUP_RE.search(text):
        return (
            "optimizer_setup",
            None,
            "started",
            "optimizer_setup_start.v1",
            "optimizer setup started",
        )
    if _CUDA_GRAPH_BUILT_RE.search(text):
        return (
            "cuda_graph_build",
            None,
            "completed",
            "cuda_graph_build_complete.v1",
            "cuda graph build completed",
        )
    return None


def _collect_registry_matches(
    lines: list[LogLine],
    progress: ProgressFacts,
    terminal_lines: Sequence[int],
) -> list[FailureEvidence]:
    all_line_text = {item.line: item.text for item in lines}
    result: list[FailureEvidence] = []
    for item in lines:
        if diagnostic_context_kind(item.text) is not None:
            continue
        teardown_exception = _exception_is_teardown(lines, item.line)
        rows = match_registry(item.text, diagnostic_checked=True)
        if teardown_exception:
            rows = [row for row in rows if row.registry_id == "observed_exception"]
        for row in rows:
            rank = extract_rank(item.text)
            node = extract_node(item.text)
            gpu = extract_gpu(item.text)
            outcome = _candidate_outcome(item.line, item.text, progress, terminal_lines)
            observed_fingerprint = root_fingerprint(row, item.text)
            fingerprint_source = "l0_registry"
            if row.registry_id == "observed_exception":
                context_start = max(0, item.line - 16)
                observed_fingerprint = canonical_observed_fingerprint(
                    item.text,
                    tuple(line.text for line in lines[context_start : item.line - 1]),
                )
                fingerprint_source = "observed_exception"
            if teardown_exception:
                observed_fingerprint = fingerprint_for(
                    "teardown_cleanup",
                    [signature_for(row, item.text)],
                )
                fingerprint_source = "l0_teardown_structure"
            result.append(
                FailureEvidence(
                    fine_class=row.registry_id,
                    policy_class=(
                        PolicyClass.CASCADE.value if teardown_exception else row.policy_class
                    ),
                    signature=signature_for(row, item.text),
                    root_fingerprint=observed_fingerprint,
                    fault_outcome=outcome,
                    causal_role=(
                        CausalRole.TEARDOWN.value
                        if teardown_exception
                        else (
                            CausalRole.CASCADE.value
                            if row.policy_class == PolicyClass.CASCADE.value
                            else CausalRole.UNKNOWN.value
                        )
                    ),
                    line=item.line,
                    quote=all_line_text.get(item.line),
                    rank=rank,
                    phase=(
                        "teardown"
                        if teardown_exception
                        else _phase_for_line(item.line, item.text, progress)
                    ),
                    node=node,
                    gpu=gpu,
                    failure_iteration=extract_failure_iteration(item.text),
                    data_position_fingerprint=extract_data_position_fingerprint(item.text),
                    registry_id=row.registry_id,
                    role=(RegistryRole.CASCADE_CANDIDATE.value if teardown_exception else row.role),
                    recovery_behavior=row.recovery_behavior,
                    root_fingerprint_source=fingerprint_source,
                )
            )
    return result


def _compact_registry_matches(
    matches: Sequence[FailureEvidence],
) -> tuple[list[FailureEvidence], int]:
    groups: OrderedDict[tuple[str | None, str], list[FailureEvidence]] = OrderedDict()
    for match in matches:
        key = (match.registry_id, normalized_pattern(match.quote or match.signature))
        groups.setdefault(key, []).append(match)

    retained: list[FailureEvidence] = []
    for group in groups.values():
        if len(group) <= REGISTRY_MATCH_HEAD_PER_PATTERN + REGISTRY_MATCH_TAIL_PER_PATTERN:
            retained.extend(group)
            continue
        retained.extend(group[:REGISTRY_MATCH_HEAD_PER_PATTERN])
        retained.extend(group[-REGISTRY_MATCH_TAIL_PER_PATTERN:])
    retained.sort(key=lambda match: match.line or 0)
    return retained, len(matches) - len(retained)


def _collect_diagnostic_occurrence_seeds(
    lines: Iterable[LogLine],
) -> tuple[_OccurrenceSeed, ...]:
    seeds: list[_OccurrenceSeed] = []
    for item in lines:
        text = item.text
        diagnostic_kind = diagnostic_context_kind(text)
        if diagnostic_kind is not None:
            seeds.append(
                _diagnostic_seed(
                    item,
                    detector_id=f"{diagnostic_kind}.v1",
                    normalized_shape=diagnostic_kind,
                    classification="diagnostic_context",
                )
            )
        uncertainty_kind = diagnostic_uncertainty_kind(text)
        if uncertainty_kind is not None:
            seeds.append(
                _diagnostic_seed(
                    item,
                    detector_id=f"{uncertainty_kind}.v1",
                    normalized_shape=uncertainty_kind,
                    classification="diagnostic_hypothesis",
                )
            )
        if _TRAINING_ITERATIONS_TOTAL_RE.search(text):
            seeds.append(
                _diagnostic_seed(
                    item,
                    detector_id="megatron_training_iterations_total.v1",
                    normalized_shape="setting training iterations",
                    classification="lifecycle",
                )
            )
        if _TRAINING_START_RE.search(text):
            seeds.append(
                _diagnostic_seed(
                    item,
                    detector_id="megatron_training_start_datetime.v1",
                    normalized_shape="before start of training step datetime",
                    classification="lifecycle",
                )
            )
        if _RERUN_ITERATION_RESET_RE.search(text):
            seeds.append(
                _diagnostic_seed(
                    item,
                    detector_id="megatron_rerun_iteration_reset.v1",
                    normalized_shape="rerun current iteration reset",
                    classification="lifecycle",
                )
            )
        if _RANK_GPU_MAPPING_WARNING_RE.search(text):
            seeds.append(
                _diagnostic_seed(
                    item,
                    detector_id="rank_gpu_mapping_warning.v1",
                    normalized_shape="guessing device id based on global rank",
                    classification="diagnostic",
                )
            )
        if _NCCL_VERSION_RE.search(text):
            seeds.append(
                _diagnostic_seed(
                    item,
                    detector_id="nccl_version.v1",
                    normalized_shape="nccl version",
                    classification="diagnostic",
                )
            )
        if _WORLD_SIZE_RE.search(text):
            seeds.append(
                _diagnostic_seed(
                    item,
                    detector_id="world_size_config.v1",
                    normalized_shape="world size config",
                    classification="job_metadata",
                )
            )
    return tuple(seeds)


def _collect_high_signal_lines(lines: Iterable[LogLine]) -> tuple[int, ...]:
    result: list[int] = []
    for item in lines:
        if diagnostic_context_kind(item.text) is not None:
            continue
        signal_text = _structured_log_signal_text(item.text)
        if _HIGH_SIGNAL_RE.search(signal_text) or _BARE_PROCESS_KILLED_RE.search(item.text):
            result.append(item.line)
    return tuple(result)


def _structured_log_signal_text(text: str) -> str:
    """Exclude logger names from severity and message signal matching."""

    match = _STRUCTURED_LOG_PREFIX_RE.match(text)
    if match is None:
        return text
    severity = match.group("severity").upper()
    message = match.group("message")
    if severity in {"ERROR", "CRITICAL", "FATAL"}:
        return f"{severity}: {message}"
    return message


def _select_high_signal_lines(
    lines: Sequence[LogLine],
    high_signal_lines: Sequence[int],
    *,
    limit: int,
) -> tuple[int, ...]:
    if not high_signal_lines or limit <= 0:
        return ()

    text_by_line = {item.line: item.text for item in lines}
    selected: list[int] = []

    def add(line_no: int | None) -> None:
        if line_no is None or line_no not in text_by_line:
            return
        if line_no in selected:
            return
        if any(abs(line_no - existing) <= HIGH_SIGNAL_CLUSTER_GAP_LINES for existing in selected):
            return
        selected.append(line_no)

    add(high_signal_lines[0])

    terminal_ranked = sorted(
        high_signal_lines,
        key=lambda line_no: (
            -_high_signal_priority(text_by_line.get(line_no, "")),
            -line_no,
        ),
    )
    for line_no in terminal_ranked:
        if len(selected) >= limit - 1:
            break
        add(line_no)

    for line_no in reversed(high_signal_lines):
        if len(selected) >= limit:
            break
        add(line_no)

    if len(selected) < limit:
        for line_no in high_signal_lines:
            if len(selected) >= limit:
                break
            if line_no not in selected:
                selected.append(line_no)

    return tuple(sorted(selected[:limit]))


def _high_signal_priority(text: str) -> int:
    lowered = text.lower()
    if "traceback" in lowered:
        return 100
    if _BARE_PROCESS_KILLED_RE.search(text):
        return 98
    if "runtimeerror" in lowered or "exception" in lowered:
        return 95
    if "fatal" in lowered or "critical" in lowered:
        return 90
    if "assert" in lowered or "out of bounds" in lowered or "bounds failure" in lowered:
        return 85
    if "error" in lowered:
        return 70
    if "timeout" in lowered:
        return 60
    return 10


def _diagnostic_seed(
    item: LogLine,
    *,
    detector_id: str,
    normalized_shape: str,
    classification: str,
) -> _OccurrenceSeed:
    return _OccurrenceSeed(
        detector_id=detector_id,
        normalized_shape=normalized_shape,
        line=item.line,
        classification=classification,
        rank=extract_rank(item.text),
        node=extract_node(item.text),
        gpu=extract_gpu(item.text),
    )


def _candidate_outcome(
    line_no: int,
    text: str,
    progress: ProgressFacts,
    terminal_lines: Sequence[int],
) -> str:
    if _has_later(progress.progress_lines, line_no) or _has_later(
        progress.checkpoint_lines, line_no
    ):
        return FaultOutcome.PROGRESSED_AFTER.value
    if _has_later(progress.recovery_lines, line_no):
        return FaultOutcome.RECOVERED.value
    if _TERMINAL_RE.search(text) or _has_later(terminal_lines, line_no):
        return FaultOutcome.TERMINAL.value
    if terminal_lines and line_no >= max(1, terminal_lines[-1] - 80):
        return FaultOutcome.TERMINAL.value
    return FaultOutcome.UNRESOLVED.value


def _has_later(line_numbers: Sequence[int], line_no: int) -> bool:
    return bisect_right(line_numbers, line_no) < len(line_numbers)


def _observed_failure_iteration(text: str) -> int | None:
    if not (
        _EXCEPTION_SUMMARY_RE.search(text)
        or _WATCHDOG_EXCEPTION_RE.search(text)
        or _CUDA_RUNTIME_STATUS_RE.search(text)
        or _TERMINAL_OPERATION_TIMEOUT_RE.search(text)
    ):
        return None
    return extract_failure_iteration(text)


def _exception_is_teardown(lines: Sequence[LogLine], line_no: int) -> bool:
    if line_no < 1 or line_no > len(lines):
        return False
    terminal = lines[line_no - 1]
    if not _EXCEPTION_SUMMARY_RE.search(terminal.text):
        return False

    preferred_rank = extract_rank(terminal.text)
    context: list[LogLine] = []
    traceback_found = False
    for item in reversed(lines[max(0, line_no - 50) : line_no - 1]):
        item_rank = extract_rank(item.text)
        if preferred_rank is not None and item_rank not in {None, preferred_rank}:
            continue
        context.append(item)
        if _TRACEBACK_RE.search(item.text):
            traceback_found = True
            break
    return traceback_found and any(_CLEANUP_FRAME_RE.search(item.text) for item in context)


def _phase_for_line(line_no: int, text: str, progress: ProgressFacts) -> str | None:
    if _PHASE_CHECKPOINT_RE.search(text):
        return "checkpoint"
    failure_iteration = _observed_failure_iteration(text)
    if failure_iteration is not None:
        checkpoint_load = next(
            (
                marker
                for marker in reversed(progress.setup_markers)
                if marker.line <= line_no
                and marker.marker_type in {"checkpoint_load", "checkpoint_load_start"}
                and isinstance(marker.value, int)
            ),
            None,
        )
        if checkpoint_load is not None and failure_iteration >= int(checkpoint_load.value):
            return (
                "first_iter"
                if failure_iteration - int(checkpoint_load.value) <= 1
                else "steady_mid"
            )
        return "first_iter" if failure_iteration <= 1 else "steady_mid"
    if progress.last_progress_line is None or line_no < progress.progress_lines[0]:
        return "setup"
    if line_no <= progress.progress_lines[0]:
        return "first_iter"
    return "steady_mid"


def _build_occurrence_groups(
    matches: Sequence[FailureEvidence],
    occurrence_seeds: Sequence[_OccurrenceSeed],
) -> list[NormalizedOccurrenceGroup]:
    groups: OrderedDict[tuple[str | None, str], list[FailureEvidence | _OccurrenceSeed]] = (
        OrderedDict()
    )
    for seed in occurrence_seeds:
        groups.setdefault((seed.registry_id, seed.normalized_shape), []).append(seed)
    for match in matches:
        quote = match.quote or match.signature
        pattern_identity = normalized_pattern(quote)
        if (
            match.policy_class == PolicyClass.CASCADE.value
            and match.root_fingerprint
            and match.root_fingerprint_source.startswith("l0_")
        ):
            pattern_identity = f"stable:{match.root_fingerprint}"
        key = (match.registry_id, pattern_identity)
        groups.setdefault(key, []).append(match)

    result: list[NormalizedOccurrenceGroup] = []
    for index, ((registry_id, _), group_matches) in enumerate(groups.items(), start=1):
        first = group_matches[0]
        first_line = first.line or 0
        pattern = (
            first.normalized_shape
            if isinstance(first, _OccurrenceSeed)
            else normalized_pattern(first.quote or first.signature)
        )
        result.append(
            NormalizedOccurrenceGroup(
                occurrence_group_id=f"og-{index}",
                normalized_shape=pattern,
                first_line=first_line,
                count=len(group_matches),
                sample_lines=tuple(
                    match.line for match in group_matches[:5] if match.line is not None
                ),
                rank_spread=tuple(sorted({match.rank for match in group_matches if match.rank})),
                node_spread=tuple(sorted({match.node for match in group_matches if match.node})),
                gpu_spread=tuple(sorted({match.gpu for match in group_matches if match.gpu})),
                registry_id=registry_id,
                classification=_classification_for(first),
                classification_source="deterministic",
            )
        )
    return result


def _classification_for(match: FailureEvidence | _OccurrenceSeed) -> str:
    if isinstance(match, _OccurrenceSeed):
        return match.classification
    if match.role == RegistryRole.CAUSE_CONFIRMATION.value:
        return "cause_confirmation"
    if match.policy_class == PolicyClass.CASCADE.value:
        return "cascade"
    return "error"


def _select_primary_candidate(
    matches: Sequence[FailureEvidence],
) -> FailureEvidence | None:
    root_matches = [
        match
        for match in matches
        if match.role in {RegistryRole.ROOT_CANDIDATE.value, RegistryRole.EITHER.value}
        and match.policy_class != PolicyClass.CASCADE.value
        and match.fault_outcome
        not in {FaultOutcome.RECOVERED.value, FaultOutcome.PROGRESSED_AFTER.value}
    ]
    if root_matches:
        return sorted(root_matches, key=lambda match: match.line or 0)[0]

    progressed_roots = [
        match
        for match in matches
        if match.role in {RegistryRole.ROOT_CANDIDATE.value, RegistryRole.EITHER.value}
        and match.policy_class != PolicyClass.CASCADE.value
    ]
    if progressed_roots:
        return sorted(progressed_roots, key=lambda match: match.line or 0)[0]
    return None


def _canonicalize_episode_primary(
    primary: FailureEvidence | None,
    matches: Sequence[FailureEvidence],
    episodes: Sequence[FailureEpisode],
    lines: Sequence[LogLine],
    progress: ProgressFacts,
) -> FailureEvidence | None:
    if primary is None:
        return None
    for episode in episodes:
        terminal_line = episode.terminal_exception_line
        identity_line = episode.identity_anchor_line or terminal_line
        observed_lines = {
            *episode.precursor_lines,
            *episode.exception_chain_lines,
            terminal_line,
            identity_line,
        }
        if terminal_line is None or primary.line not in observed_lines:
            continue
        assert identity_line is not None
        identity_match = next(
            (match for match in matches if match.line == identity_line),
            None,
        )
        if identity_match is not None:
            return replace(
                identity_match,
                failure_iteration=(
                    identity_match.failure_iteration or episode.terminal_exception_iteration
                ),
            )
        if not 1 <= identity_line <= len(lines):
            continue
        identity_text = lines[identity_line - 1].text
        terminal_match = next(
            (match for match in matches if match.line == terminal_line),
            None,
        )
        context_start = max(0, identity_line - 17)
        return FailureEvidence(
            fine_class="observed_failure",
            policy_class=PolicyClass.AMBIGUOUS.value,
            signature=identity_text.strip(),
            root_fingerprint=canonical_observed_fingerprint(
                identity_text,
                tuple(item.text for item in lines[context_start : identity_line - 1]),
            ),
            fault_outcome=episode.status,
            causal_role=CausalRole.INITIATING.value,
            line=identity_line,
            quote=identity_text,
            rank=extract_rank(identity_text),
            phase=(
                terminal_match.phase
                if terminal_match is not None and terminal_match.phase
                else _phase_for_line(identity_line, identity_text, progress)
            ),
            node=extract_node(identity_text),
            gpu=extract_gpu(identity_text),
            failure_iteration=(
                extract_failure_iteration(identity_text) or episode.terminal_exception_iteration
            ),
            data_position_fingerprint=extract_data_position_fingerprint(identity_text),
            role=RegistryRole.ROOT_CANDIDATE.value,
            recovery_behavior=primary.recovery_behavior,
            root_fingerprint_source="observed_exception",
        )
    return primary


def _build_failure_episodes(
    lines: Sequence[LogLine],
    high_signal_lines: Sequence[int],
    matches: Sequence[FailureEvidence],
    progress: ProgressFacts,
    *,
    context_windows: Sequence[ContextWindow],
    event_lines: Mapping[str, Sequence[int]],
    cause_confirmations: Sequence[FailureEvidence],
    distributed_failure_incidents: Sequence[DistributedFailureIncident],
) -> list[FailureEpisode]:
    if not high_signal_lines:
        return []

    text_by_line = {item.line: item.text for item in lines}
    latest_progress = _last_progress_marker(progress)
    min_line = (latest_progress.line if latest_progress else 0) + 1
    terminal_lines = [
        line_no
        for line_no in high_signal_lines
        if line_no >= min_line and _terminal_episode_priority(text_by_line.get(line_no, "")) > 0
    ]
    if not terminal_lines and latest_progress is not None:
        terminal_lines = [
            line_no
            for line_no in high_signal_lines
            if _terminal_episode_priority(text_by_line.get(line_no, "")) > 0
        ]
    if not terminal_lines:
        return []

    starts: list[int] = []
    seen_terminal_keys: set[str] = set()
    for line_no in terminal_lines:
        start_line = _traceback_start_for_line(lines, line_no, lower_bound=min_line)
        terminal_key = (
            f"traceback:{start_line}"
            if _TRACEBACK_RE.search(text_by_line.get(start_line, ""))
            else _terminal_episode_key(text_by_line.get(line_no, ""))
        )
        if terminal_key in seen_terminal_keys:
            continue
        seen_terminal_keys.add(terminal_key)
        if start_line in starts:
            continue
        starts.append(start_line)
        if len(starts) >= MAX_FAILURE_EPISODES:
            break

    episodes: list[FailureEpisode] = []
    for start_line in starts:
        last_progress = _nearest_prior_progress_marker(progress, start_line)
        terminal_line = _terminal_exception_line(lines, start_line)
        terminal_text = text_by_line.get(terminal_line or start_line)
        progress_after = _first_progress_after(progress, last_progress, start_line)
        first_teardown_line = _first_later_line(event_lines["teardown"], start_line)
        first_process_termination_line = _first_at_or_after_line(
            event_lines["process_termination"], start_line
        )
        first_scheduler_cancel_line = _first_later_line(event_lines["scheduler_cancel"], start_line)
        downstream_cascade = _first_downstream_cascade(matches, start_line)
        status = _failure_episode_status(
            terminal_line=terminal_line,
            progress_after=progress_after,
            first_teardown_line=first_teardown_line,
            first_process_termination_line=first_process_termination_line,
            first_scheduler_cancel_line=first_scheduler_cancel_line,
            downstream_cascade=downstream_cascade,
        )
        end_line = max(
            candidate
            for candidate in (
                start_line,
                terminal_line,
                first_teardown_line,
                first_process_termination_line,
                first_scheduler_cancel_line,
                downstream_cascade.line if downstream_cascade else None,
            )
            if candidate is not None
        )
        episodes.append(
            FailureEpisode(
                episode_id=f"fe-{len(episodes) + 1}",
                status=status,
                start_line=start_line,
                end_line=end_line,
                first_exception_line=start_line,
                terminal_exception_line=terminal_line,
                terminal_exception_quote=terminal_text,
                terminal_exception_iteration=_iteration_value(terminal_text),
                terminal_exception_causal_role_hint=_traceback_causal_role_hint(
                    lines,
                    start_line,
                    terminal_line,
                ),
                exception_rank=extract_rank(terminal_text or text_by_line.get(start_line, "")),
                exception_node=extract_node(terminal_text or text_by_line.get(start_line, "")),
                exception_gpu=extract_gpu(terminal_text or text_by_line.get(start_line, "")),
                last_progress_before=last_progress,
                first_progress_after=progress_after,
                first_teardown_line=first_teardown_line,
                first_process_termination_line=first_process_termination_line,
                first_scheduler_cancel_line=first_scheduler_cancel_line,
                first_downstream_cascade=downstream_cascade,
                context_window_ids=_context_window_ids_for_episode(
                    context_windows,
                    start_line,
                    terminal_line,
                ),
                reason=_failure_episode_reason(
                    status,
                    prior_progress_observed=last_progress is not None,
                ),
            )
        )
    consolidated = _consolidate_failure_episodes(
        lines,
        episodes,
        progress,
        distributed_failure_incidents,
    )
    with_precursors = _attach_timeout_aligned_precursors(
        lines,
        consolidated,
        high_signal_lines,
        progress,
    )
    with_precursors = _attach_nearby_high_signal_precursors(
        lines,
        with_precursors,
        high_signal_lines,
        progress,
    )
    return _attach_cause_confirmations(
        with_precursors,
        cause_confirmations,
        context_windows,
    )


def _consolidate_failure_episodes(
    lines: Sequence[LogLine],
    episodes: Sequence[FailureEpisode],
    progress: ProgressFacts,
    distributed_failure_incidents: Sequence[DistributedFailureIncident],
) -> list[FailureEpisode]:
    if len(episodes) < 2:
        return list(episodes)

    groups: list[list[tuple[FailureEpisode, str | None]]] = []
    current: list[tuple[FailureEpisode, str | None]] = [(episodes[0], None)]
    for episode in episodes[1:]:
        previous = current[-1][0]
        relation = _episode_relation(
            lines,
            previous,
            episode,
            progress,
            distributed_failure_incidents,
        )
        if relation is None:
            groups.append(current)
            current = [(episode, None)]
        else:
            current.append((episode, relation))
    groups.append(current)

    consolidated: list[FailureEpisode] = []
    for group in groups:
        members = [item[0] for item in group]
        if len(members) == 1:
            consolidated.append(members[0])
            continue

        duplicate_lines = {
            previous.terminal_exception_line
            for previous, current_item in zip(members, group[1:])
            if current_item[1] == "serialized_duplicate"
            and previous.terminal_exception_line is not None
        }
        duplicate_lines.update(
            member.terminal_exception_line
            for member, relation in group
            if relation in {"distributed_fanout", "duplicate_rendering"}
            and member.terminal_exception_line is not None
        )
        wrapper_lines = {
            member.terminal_exception_line
            for member, relation in group
            if relation in {"outer_wrapper", "teardown_follow_on"}
            and member.terminal_exception_line is not None
        }
        causal = next(
            (
                member
                for member in members
                if member.terminal_exception_line not in duplicate_lines
                and member.terminal_exception_line not in wrapper_lines
            ),
            members[0],
        )
        downstream = min(
            (
                member.first_downstream_cascade
                for member in members
                if member.first_downstream_cascade is not None
                and member.first_downstream_cascade.line is not None
            ),
            key=lambda item: item.line or 0,
            default=None,
        )
        context_ids = tuple(
            dict.fromkeys(
                window_id for member in members for window_id in member.context_window_ids
            )
        )
        relations = {relation for _, relation in group if relation is not None}
        if relations.issubset({"distributed_fanout", "distributed_timeout_wave"}):
            reason = (
                "consolidated distributed timeout wave; earliest observed terminal "
                f"timeout at line {causal.terminal_exception_line}"
            )
        else:
            reason = (
                "consolidated exception chain; canonical causal exception at "
                f"line {causal.terminal_exception_line}"
            )
        consolidated.append(
            FailureEpisode(
                episode_id=f"fe-{len(consolidated) + 1}",
                status=(
                    FaultOutcome.PROGRESSED_AFTER.value
                    if any(member.first_progress_after is not None for member in members)
                    else causal.status
                ),
                start_line=min(member.start_line for member in members),
                end_line=max(member.end_line for member in members),
                first_exception_line=min(member.first_exception_line for member in members),
                terminal_exception_line=causal.terminal_exception_line,
                terminal_exception_quote=causal.terminal_exception_quote,
                terminal_exception_iteration=causal.terminal_exception_iteration,
                terminal_exception_causal_role_hint=(causal.terminal_exception_causal_role_hint),
                precursor_lines=tuple(
                    dict.fromkeys(
                        line_no for member in members for line_no in member.precursor_lines
                    )
                ),
                identity_anchor_line=causal.identity_anchor_line,
                identity_anchor_reason=causal.identity_anchor_reason,
                exception_chain_lines=tuple(
                    member.terminal_exception_line
                    for member in members
                    if member.terminal_exception_line is not None
                ),
                duplicate_rendering_lines=tuple(sorted(duplicate_lines)),
                wrapper_exception_lines=tuple(sorted(wrapper_lines)),
                exception_rank=causal.exception_rank,
                exception_node=causal.exception_node,
                exception_gpu=causal.exception_gpu,
                last_progress_before=causal.last_progress_before,
                first_progress_after=min(
                    (
                        member.first_progress_after
                        for member in members
                        if member.first_progress_after is not None
                    ),
                    key=lambda marker: marker.line,
                    default=None,
                ),
                first_teardown_line=_minimum_optional(
                    member.first_teardown_line for member in members
                ),
                first_process_termination_line=_minimum_optional(
                    member.first_process_termination_line for member in members
                ),
                first_scheduler_cancel_line=_minimum_optional(
                    member.first_scheduler_cancel_line for member in members
                ),
                first_downstream_cascade=downstream,
                context_window_ids=context_ids,
                reason=reason,
            )
        )
    return consolidated


def _episode_relation(
    lines: Sequence[LogLine],
    previous: FailureEpisode,
    current: FailureEpisode,
    progress: ProgressFacts,
    distributed_failure_incidents: Sequence[DistributedFailureIncident],
) -> str | None:
    previous_terminal = previous.terminal_exception_line or previous.start_line
    current_start = current.start_line
    if current_start <= previous_terminal:
        return None
    if any(previous_terminal < line_no < current_start for line_no in progress.progress_lines):
        return None
    current_terminal = current.terminal_exception_line or current.start_line
    if any(
        previous_terminal in incident.member_event_lines
        and current_terminal in incident.member_event_lines
        for incident in distributed_failure_incidents
    ):
        return "distributed_fanout"
    previous_watchdog = _WATCHDOG_EXCEPTION_RE.search(previous.terminal_exception_quote or "")
    current_watchdog = _WATCHDOG_EXCEPTION_RE.search(current.terminal_exception_quote or "")
    if (
        previous_watchdog is not None
        and current_watchdog is not None
        and normalized_pattern(previous_watchdog.group("message"))
        == normalized_pattern(current_watchdog.group("message"))
        and previous.exception_rank == current.exception_rank
    ):
        return "duplicate_rendering"
    previous_operation = _timed_operation_identity(previous.terminal_exception_quote)
    current_operation = _timed_operation_identity(current.terminal_exception_quote)
    if previous_operation is not None and previous_operation == current_operation:
        return "distributed_fanout"
    if (
        previous_operation is not None
        and current_operation is not None
        and _same_timeout_detection_wave(
            previous.terminal_exception_quote,
            current.terminal_exception_quote,
        )
    ):
        return "distributed_timeout_wave"
    prior_terminal_markers = (
        previous.first_teardown_line,
        previous.first_process_termination_line,
        previous.first_scheduler_cancel_line,
    )
    if current.terminal_exception_causal_role_hint == CausalRole.TEARDOWN.value:
        if (
            previous.terminal_exception_causal_role_hint == CausalRole.TEARDOWN.value
            or current_start <= previous.end_line
            or any(
                marker is not None and marker < current_start for marker in prior_terminal_markers
            )
        ):
            return "teardown_follow_on"
    between = lines[previous_terminal:current_start]
    if any(_EXCEPTION_CHAIN_RE.search(item.text) for item in between):
        return "outer_wrapper"
    if (
        current_start - previous.start_line <= 10
        and _serialized_traceback(previous.terminal_exception_quote)
        and previous.exception_rank == current.exception_rank
    ):
        return "serialized_duplicate"
    return None


def _same_timeout_detection_wave(first: str | None, second: str | None) -> bool:
    if not first or not second:
        return False
    first_time = _time_of_day_seconds(first)
    second_time = _time_of_day_seconds(second)
    if first_time is None or second_time is None:
        return False
    first_timeout = _configured_timeout_seconds(first)
    second_timeout = _configured_timeout_seconds(second)
    if (
        first_timeout is not None
        and second_timeout is not None
        and abs(first_timeout - second_timeout) > 1.0
    ):
        return False
    tolerance = _distributed_timeout_wave_tolerance(first_timeout, second_timeout)
    return _time_of_day_distance(first_time, second_time) <= tolerance


def _timed_operation_identity(text: str | None) -> tuple[tuple[str, str], ...] | None:
    fields = _timed_operation_fields(text)
    if fields is None:
        return None
    return tuple((name, fields[name]) for name in ("seqnum", "optype"))


def _timed_operation_fields(text: str | None) -> dict[str, str] | None:
    if not text or not _TERMINAL_OPERATION_TIMEOUT_RE.search(text):
        return None
    fields = {
        match.group("name").lower(): match.group("value").lower()
        for match in _TIMED_OPERATION_FIELD_RE.finditer(text)
    }
    if not {"seqnum", "optype"}.issubset(fields):
        return None
    return fields


def _build_distributed_failure_incidents(
    lines: Sequence[LogLine],
    progress: ProgressFacts,
    matches: Sequence[FailureEvidence],
) -> list[DistributedFailureIncident]:
    incidents = _build_collective_timeout_incidents(lines, progress)
    incidents.extend(
        _build_distributed_exception_fanout_incidents(
            lines,
            progress,
            matches,
            first_incident_number=len(incidents) + 1,
            excluded_lines={
                line_no for incident in incidents for line_no in incident.member_event_lines
            },
        )
    )
    return incidents


def _build_collective_timeout_incidents(
    lines: Sequence[LogLine],
    progress: ProgressFacts,
) -> list[DistributedFailureIncident]:
    events: list[_TimedOperationEvent] = []
    for item in lines:
        fields = _timed_operation_fields(item.text)
        if fields is None:
            continue
        events.append(
            _TimedOperationEvent(
                line=item.line,
                text=item.text,
                timestamp_seconds=_time_of_day_seconds(item.text),
                timestamp_text=_time_of_day_text(item.text),
                configured_timeout_seconds=_configured_timeout_seconds(item.text),
                sequence_number=fields["seqnum"],
                operation_type=fields["optype"].lstrip("_"),
                rank=extract_rank(item.text),
            )
        )
    if not events:
        return []

    groups: list[list[_TimedOperationEvent]] = []
    current = [events[0]]
    for event in events[1:]:
        if _same_distributed_timeout_wave(current, event, progress):
            current.append(event)
        else:
            groups.append(current)
            current = [event]
    groups.append(current)

    incidents: list[DistributedFailureIncident] = []
    for group in groups:
        primary = min(group, key=lambda event: event.line)
        last_progress = _nearest_prior_progress_marker(progress, primary.line)
        first_detection = min(
            (event for event in group if event.timestamp_seconds is not None),
            key=lambda event: event.timestamp_seconds or 0.0,
            default=primary,
        )
        configured_timeout = _common_configured_timeout(group)
        seconds_since_progress = _elapsed_time_of_day(
            last_progress.timestamp if last_progress is not None else None,
            first_detection.text,
        )
        detection_lag = (
            max(0.0, seconds_since_progress - configured_timeout)
            if seconds_since_progress is not None and configured_timeout is not None
            else None
        )
        phase = _incident_phase_for_line(primary.line, primary.text, progress)
        operation_signatures = tuple(
            sorted(
                {f"optype={event.operation_type},seqnum={event.sequence_number}" for event in group}
            )
        )
        operation_types = tuple(sorted({event.operation_type for event in group}))
        ranks = tuple(
            sorted(
                {event.rank for event in group if event.rank is not None},
                key=_rank_sort_key,
            )
        )
        later_progress = _first_progress_after(
            progress, last_progress, max(event.line for event in group)
        )
        status = (
            FaultOutcome.PROGRESSED_AFTER.value
            if later_progress is not None
            else FaultOutcome.TERMINAL.value
        )
        history_fingerprint = fingerprint_for(
            "distributed_incident",
            ["collective_operation_timeout", phase or "unknown_phase"],
        )
        incidents.append(
            DistributedFailureIncident(
                incident_id=f"di-{len(incidents) + 1}",
                incident_kind=DistributedIncidentKind.DISTRIBUTED_MECHANISM.value,
                incident_type="distributed_collective_timeout_wave",
                status=status,
                first_observed_line=min(event.line for event in group),
                last_observed_line=max(event.line for event in group),
                primary_observed_line=primary.line,
                primary_observed_quote=primary.text,
                member_event_lines=tuple(event.line for event in group),
                sample_lines=_distributed_incident_sample_lines(group),
                event_count=len(group),
                unique_operation_count=len(operation_signatures),
                operation_types=operation_types,
                operation_signatures=operation_signatures,
                observed_rank_count=len(ranks),
                rank_spread=ranks[:MAX_DISTRIBUTED_INCIDENT_RANK_SAMPLES],
                process_group_types=_process_group_types_for_wave(lines, group),
                phase=phase,
                configured_timeout_seconds=configured_timeout,
                last_progress_line=(last_progress.line if last_progress is not None else None),
                last_progress_timestamp=(
                    last_progress.timestamp if last_progress is not None else None
                ),
                first_detection_timestamp=first_detection.timestamp_text,
                seconds_since_last_progress=_rounded_seconds(seconds_since_progress),
                detection_lag_seconds=_rounded_seconds(detection_lag),
                history_fingerprint=history_fingerprint,
            )
        )
    return incidents


def _build_distributed_exception_fanout_incidents(
    lines: Sequence[LogLine],
    progress: ProgressFacts,
    matches: Sequence[FailureEvidence],
    *,
    first_incident_number: int,
    excluded_lines: set[int],
) -> list[DistributedFailureIncident]:
    text_by_line = {item.line: item.text for item in lines}
    grouped: OrderedDict[tuple[str, str], dict[int, FailureEvidence]] = OrderedDict()
    for match in matches:
        if (
            match.registry_id != "observed_exception"
            or match.line is None
            or match.line in excluded_lines
            or match.rank is None
        ):
            continue
        text = match.quote or text_by_line.get(match.line, "")
        if not _EXCEPTION_SUMMARY_RE.search(text):
            continue
        prior_progress = _nearest_prior_progress_marker(progress, match.line)
        progress_segment = str(prior_progress.line if prior_progress is not None else 0)
        grouped.setdefault(
            (progress_segment, normalized_pattern(text)),
            {},
        )[match.line] = match

    incidents: list[DistributedFailureIncident] = []
    for (_, pattern), by_line in grouped.items():
        members = [by_line[line_no] for line_no in sorted(by_line)]
        ranks = tuple(
            sorted(
                {member.rank for member in members if member.rank is not None},
                key=_rank_sort_key,
            )
        )
        if len(members) < 2 or len(ranks) < 2:
            continue
        primary = members[0]
        primary_line = primary.line or 0
        last_line = members[-1].line or primary_line
        last_progress = _nearest_prior_progress_marker(progress, primary_line)
        later_progress = _first_progress_after(progress, last_progress, last_line)
        primary_text = text_by_line.get(primary_line, primary.quote or primary.signature)
        exception_match = _EXCEPTION_SUMMARY_RE.search(primary_text)
        exception_type = (
            exception_match.group(0).rstrip(":") if exception_match is not None else "exception"
        )
        history_fingerprint = primary.root_fingerprint or canonical_observed_fingerprint(
            primary_text,
            (),
        )
        incidents.append(
            DistributedFailureIncident(
                incident_id=f"di-{first_incident_number + len(incidents)}",
                incident_kind=DistributedIncidentKind.DISTRIBUTED_FANOUT.value,
                incident_type="distributed_exception_fanout",
                status=(
                    FaultOutcome.PROGRESSED_AFTER.value
                    if later_progress is not None
                    else FaultOutcome.TERMINAL.value
                ),
                first_observed_line=primary_line,
                last_observed_line=last_line,
                primary_observed_line=primary_line,
                primary_observed_quote=primary_text,
                member_event_lines=tuple(
                    member.line for member in members if member.line is not None
                ),
                sample_lines=_sample_line_numbers(
                    tuple(member.line for member in members if member.line is not None),
                    MAX_DISTRIBUTED_INCIDENT_SAMPLE_LINES,
                ),
                event_count=len(members),
                unique_operation_count=1,
                operation_types=(exception_type,),
                operation_signatures=(pattern,),
                observed_rank_count=len(ranks),
                rank_spread=ranks[:MAX_DISTRIBUTED_INCIDENT_RANK_SAMPLES],
                phase=primary.phase
                or _incident_phase_for_line(
                    primary_line,
                    primary_text,
                    progress,
                ),
                last_progress_line=(last_progress.line if last_progress is not None else None),
                last_progress_timestamp=(
                    last_progress.timestamp if last_progress is not None else None
                ),
                history_fingerprint=history_fingerprint,
                history_fingerprint_source="l0_distributed_exception_fanout",
                root_cause_status=AssessmentStatus.UNKNOWN.value,
                interpretation="same_attempt_rank_fanout_not_cross_cycle_recurrence",
            )
        )
    return incidents


def _same_distributed_timeout_wave(
    current_wave: Sequence[_TimedOperationEvent],
    current: _TimedOperationEvent,
    progress: ProgressFacts,
) -> bool:
    first = current_wave[0]
    previous = current_wave[-1]
    progress_lines = (*progress.progress_lines, *progress.checkpoint_lines)
    if any(previous.line < line_no < current.line for line_no in progress_lines):
        return False
    if (
        first.configured_timeout_seconds is not None
        and current.configured_timeout_seconds is not None
        and abs(first.configured_timeout_seconds - current.configured_timeout_seconds) > 1.0
    ):
        return False
    reference_times = [
        event.timestamp_seconds for event in current_wave if event.timestamp_seconds is not None
    ]
    if reference_times and current.timestamp_seconds is not None:
        tolerance = _distributed_timeout_wave_tolerance(
            *(event.configured_timeout_seconds for event in current_wave),
            current.configured_timeout_seconds,
        )
        return (
            min(
                _time_of_day_distance(reference, current.timestamp_seconds)
                for reference in reference_times
            )
            <= tolerance
        )
    return current.line - previous.line <= 1000


def _common_configured_timeout(events: Sequence[_TimedOperationEvent]) -> float | None:
    counts: dict[float, int] = defaultdict(int)
    for event in events:
        if event.configured_timeout_seconds is not None:
            counts[event.configured_timeout_seconds] += 1
    if not counts:
        return None
    return max(counts, key=lambda timeout: (counts[timeout], -timeout))


def _distributed_incident_sample_lines(
    events: Sequence[_TimedOperationEvent],
) -> tuple[int, ...]:
    selected: list[int] = []
    seen_signatures: set[tuple[str, str]] = set()
    for event in events:
        signature = (event.operation_type, event.sequence_number)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        selected.append(event.line)
    for line_no in (events[0].line, events[-1].line):
        if line_no not in selected:
            selected.append(line_no)
    return tuple(sorted(selected)[:MAX_DISTRIBUTED_INCIDENT_SAMPLE_LINES])


def _sample_line_numbers(line_numbers: Sequence[int], limit: int) -> tuple[int, ...]:
    ordered = tuple(sorted(dict.fromkeys(line_numbers)))
    if len(ordered) <= limit:
        return ordered
    head_count = max(1, limit - 1)
    return (*ordered[:head_count], ordered[-1])


def _process_group_types_for_wave(
    lines: Sequence[LogLine],
    events: Sequence[_TimedOperationEvent],
) -> tuple[str, ...]:
    event_times = [
        event.timestamp_seconds for event in events if event.timestamp_seconds is not None
    ]
    if not event_times:
        return ()
    tolerance = _distributed_timeout_wave_tolerance(
        *(event.configured_timeout_seconds for event in events)
    )
    lower_line = max(1, min(event.line for event in events) - 100)
    upper_line = min(len(lines), max(event.line for event in events) + 500)
    group_types: set[str] = set()
    for item in lines[lower_line - 1 : upper_line]:
        match = _PROCESS_GROUP_TYPE_RE.search(item.text)
        timestamp = _time_of_day_seconds(item.text)
        if match is None or timestamp is None:
            continue
        if (
            min(_time_of_day_distance(timestamp, event_time) for event_time in event_times)
            > tolerance
        ):
            continue
        group_types.add(match.group("group_type").lower())
    return tuple(sorted(group_types))


def _apply_distributed_incident_identity(
    primary: FailureEvidence | None,
    incidents: Sequence[DistributedFailureIncident],
) -> FailureEvidence | None:
    if primary is None or primary.line is None:
        return primary
    for incident in incidents:
        if primary.line not in incident.member_event_lines:
            continue
        return replace(
            primary,
            root_fingerprint=incident.history_fingerprint,
            root_fingerprint_source=incident.history_fingerprint_source,
        )
    return primary


def _incident_phase_for_line(
    line_no: int,
    text: str,
    progress: ProgressFacts,
) -> str | None:
    prior_setup = [marker for marker in progress.setup_markers if marker.line <= line_no]
    prior_progress = [marker for marker in progress.progress_markers if marker.line <= line_no]
    if prior_setup:
        latest_setup = max(prior_setup, key=lambda marker: marker.line)
        latest_progress_line = max(
            (marker.line for marker in prior_progress),
            default=0,
        )
        if latest_setup.line > latest_progress_line:
            return latest_setup.marker_type
    return _phase_for_line(line_no, text, progress)


def _time_of_day_text(text: str) -> str | None:
    match = _TIME_OF_DAY_RE.search(text)
    return match.group(0) if match is not None else None


def _elapsed_time_of_day(start: str | None, end_text: str | None) -> float | None:
    if not start or not end_text:
        return None
    start_seconds = _time_of_day_seconds(start)
    end_seconds = _time_of_day_seconds(end_text)
    if start_seconds is None or end_seconds is None:
        return None
    elapsed = end_seconds - start_seconds
    if elapsed < 0:
        elapsed += 24 * 60 * 60
    return elapsed


def _time_of_day_distance(first: float, second: float) -> float:
    distance = abs(first - second)
    return min(distance, 24 * 60 * 60 - distance)


def _distributed_timeout_wave_tolerance(*timeouts: float | None) -> float:
    configured = [timeout for timeout in timeouts if timeout is not None]
    if not configured:
        return DISTRIBUTED_TIMEOUT_WAVE_SECONDS
    return max(
        DISTRIBUTED_TIMEOUT_WAVE_SECONDS,
        min(MAX_DISTRIBUTED_TIMEOUT_WAVE_SECONDS, min(configured) * 0.1),
    )


def _rounded_seconds(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def _rank_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _attach_timeout_aligned_precursors(
    lines: Sequence[LogLine],
    episodes: Sequence[FailureEpisode],
    high_signal_lines: Sequence[int],
    progress: ProgressFacts,
) -> list[FailureEpisode]:
    text_by_line = {item.line: item.text for item in lines}
    result: list[FailureEpisode] = []
    for episode in episodes:
        terminal_line = episode.terminal_exception_line
        terminal_text = episode.terminal_exception_quote or ""
        timeout_seconds = _configured_timeout_seconds(terminal_text)
        terminal_time = _time_of_day_seconds(terminal_text)
        if terminal_line is None or timeout_seconds is None or terminal_time is None:
            result.append(
                replace(
                    episode,
                    identity_anchor_line=episode.identity_anchor_line or terminal_line,
                    identity_anchor_reason=(episode.identity_anchor_reason or "terminal_exception"),
                )
            )
            continue

        last_progress_line = (
            episode.last_progress_before.line if episode.last_progress_before is not None else 0
        )
        precursor_lines: list[int] = []
        for line_no in high_signal_lines:
            if not last_progress_line < line_no < terminal_line:
                continue
            candidate_text = text_by_line.get(line_no, "")
            if (
                _TERMINAL_OPERATION_TIMEOUT_RE.search(candidate_text)
                or _TRACEBACK_RE.search(candidate_text)
                or _CLEANUP_FRAME_RE.search(candidate_text)
            ):
                continue
            candidate_time = _time_of_day_seconds(candidate_text)
            if candidate_time is None:
                continue
            elapsed = terminal_time - candidate_time
            if elapsed < 0:
                elapsed += 24 * 60 * 60
            tolerance = max(5.0, min(60.0, timeout_seconds * 0.1))
            if abs(elapsed - timeout_seconds) <= tolerance:
                precursor_lines.append(line_no)

        if precursor_lines:
            identity_anchor_line = min(precursor_lines)
            identity_anchor_reason = "observed_precursor_aligned_with_terminal_timeout"
        else:
            identity_anchor_line = episode.identity_anchor_line or terminal_line
            identity_anchor_reason = episode.identity_anchor_reason or "terminal_exception"
        result.append(
            replace(
                episode,
                precursor_lines=tuple(dict.fromkeys((*episode.precursor_lines, *precursor_lines))),
                identity_anchor_line=identity_anchor_line,
                identity_anchor_reason=identity_anchor_reason,
            )
        )
    return result


def _attach_nearby_high_signal_precursors(
    lines: Sequence[LogLine],
    episodes: Sequence[FailureEpisode],
    high_signal_lines: Sequence[int],
    progress: ProgressFacts,
) -> list[FailureEpisode]:
    """Attach a nearby observed error that explains a following traceback or wrapper."""

    text_by_line = {item.line: item.text for item in lines}
    result: list[FailureEpisode] = []
    for episode in episodes:
        if episode.precursor_lines:
            result.append(episode)
            continue
        lower_bound = max(
            episode.start_line - NEARBY_EPISODE_PRECURSOR_LINES,
            (episode.last_progress_before.line + 1) if episode.last_progress_before else 1,
        )
        candidates = [
            line_no
            for line_no in high_signal_lines
            if lower_bound <= line_no < episode.start_line
            and _terminal_episode_priority(text_by_line.get(line_no, "")) == 0
            and _high_signal_priority(text_by_line.get(line_no, "")) >= 70
            and not _FAILURE_ANNOUNCEMENT_RE.search(text_by_line.get(line_no, ""))
            and not _CLEANUP_FRAME_RE.search(text_by_line.get(line_no, ""))
            and not any(
                line_no < progress_line < episode.start_line
                for progress_line in (*progress.progress_lines, *progress.checkpoint_lines)
            )
        ]
        if not candidates:
            result.append(episode)
            continue
        identity_anchor_line = min(candidates)
        result.append(
            replace(
                episode,
                precursor_lines=tuple(dict.fromkeys((*episode.precursor_lines, *candidates))),
                identity_anchor_line=identity_anchor_line,
                identity_anchor_reason="nearby_high_signal_error_precedes_failure_episode",
            )
        )
    return result


def _attach_cause_confirmations(
    episodes: Sequence[FailureEpisode],
    confirmations: Sequence[FailureEvidence],
    context_windows: Sequence[ContextWindow],
) -> list[FailureEpisode]:
    if not episodes or not confirmations:
        return list(episodes)

    assigned: dict[int, list[FailureEvidence]] = defaultdict(list)
    for confirmation in sorted(confirmations, key=lambda item: item.line or 0):
        if confirmation.line is None:
            continue
        eligible = [
            (index, episode)
            for index, episode in enumerate(episodes)
            if episode.start_line <= confirmation.line
            and (
                episode.first_progress_after is None
                or episode.first_progress_after.line > confirmation.line
            )
        ]
        if not eligible:
            continue
        termination_episodes = [
            item
            for item in eligible
            if _BARE_PROCESS_KILLED_RE.search(item[1].terminal_exception_quote or "")
            or (
                item[1].first_process_termination_line is not None
                and item[1].first_process_termination_line <= confirmation.line
            )
        ]
        selected_index, _ = max(
            termination_episodes or eligible,
            key=lambda item: item[1].start_line,
        )
        assigned[selected_index].append(confirmation)

    result: list[FailureEpisode] = []
    for index, episode in enumerate(episodes):
        sampled = _sample_cause_confirmations(assigned.get(index, ()))
        if not sampled:
            result.append(episode)
            continue
        confirmation_lines = tuple(item.line for item in sampled if item.line is not None)
        context_ids = list(episode.context_window_ids)
        for line_no in confirmation_lines:
            for window_id in _context_window_ids_for_line(context_windows, line_no):
                if window_id not in context_ids:
                    context_ids.append(window_id)
        result.append(
            replace(
                episode,
                end_line=max(episode.end_line, *confirmation_lines),
                identity_anchor_line=confirmation_lines[0],
                identity_anchor_reason="explicit_cause_confirmation",
                cause_confirmations=sampled,
                context_window_ids=tuple(context_ids),
            )
        )
    return result


def _sample_cause_confirmations(
    confirmations: Sequence[FailureEvidence],
) -> tuple[FailureEvidence, ...]:
    if len(confirmations) <= 3:
        return tuple(confirmations)
    return tuple((*confirmations[:2], confirmations[-1]))


def _configured_timeout_seconds(text: str) -> float | None:
    match = _TIMEOUT_MS_RE.search(text)
    if match is None:
        return None
    return int(match.group("milliseconds")) / 1000.0


def _time_of_day_seconds(text: str) -> float | None:
    match = _TIME_OF_DAY_RE.search(text)
    if match is None:
        return None
    fraction = match.group("fraction") or ""
    fractional_seconds = float(f"0.{fraction}") if fraction else 0.0
    return (
        int(match.group("hour")) * 3600
        + int(match.group("minute")) * 60
        + int(match.group("second"))
        + fractional_seconds
    )


def _serialized_traceback(text: str | None) -> bool:
    if not text:
        return False
    return bool(_TRACEBACK_RE.search(text) and ("\\n" in text or text.lstrip().startswith("[")))


def _minimum_optional(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _last_progress_marker(progress: ProgressFacts) -> ProgressMarker | None:
    markers = [*progress.progress_markers, *progress.checkpoint_markers]
    if not markers:
        return None
    return max(markers, key=lambda marker: marker.line)


def _nearest_prior_progress_marker(
    progress: ProgressFacts,
    line_no: int,
) -> ProgressMarker | None:
    markers = [
        marker
        for marker in [*progress.progress_markers, *progress.checkpoint_markers]
        if marker.line < line_no
    ]
    if not markers:
        return None
    return max(markers, key=lambda marker: marker.line)


def _terminal_episode_priority(text: str) -> int:
    if _TRACEBACK_RE.search(text):
        return 100
    if _EXCEPTION_SUMMARY_RE.search(text):
        return 95
    if _WATCHDOG_EXCEPTION_RE.search(text):
        return 94
    if _CUDA_RUNTIME_STATUS_RE.search(text):
        return 94
    if _BARE_PROCESS_KILLED_RE.search(text):
        return 90
    if _TERMINAL_OPERATION_TIMEOUT_RE.search(text):
        return 80
    return 0


def _traceback_start_for_line(
    lines: Sequence[LogLine],
    line_no: int,
    *,
    lower_bound: int,
) -> int:
    if 1 <= line_no <= len(lines) and _TRACEBACK_RE.search(lines[line_no - 1].text):
        return line_no
    start_index = max(0, line_no - 41)
    end_index = max(0, line_no - 1)
    for item in reversed(lines[start_index:end_index]):
        if item.line < lower_bound:
            break
        if _TRACEBACK_RE.search(item.text):
            return item.line
    return line_no


def _terminal_exception_line(lines: Sequence[LogLine], start_line: int) -> int | None:
    for item in lines[start_line - 1 : min(len(lines), start_line + 120)]:
        if _EXCEPTION_SUMMARY_RE.search(item.text):
            return item.line
        if _WATCHDOG_EXCEPTION_RE.search(item.text):
            return item.line
        if _CUDA_RUNTIME_STATUS_RE.search(item.text):
            return item.line
        if item.line == start_line and _TERMINAL_OPERATION_TIMEOUT_RE.search(item.text):
            return item.line
    return start_line


def _traceback_causal_role_hint(
    lines: Sequence[LogLine],
    start_line: int,
    terminal_line: int | None,
) -> str:
    if terminal_line is None:
        return CausalRole.UNKNOWN.value
    if terminal_line >= start_line and _exception_is_teardown(lines, terminal_line):
        return CausalRole.TEARDOWN.value
    return CausalRole.UNKNOWN.value


def _first_progress_after(
    progress: ProgressFacts,
    last_progress: ProgressMarker | None,
    line_no: int,
) -> ProgressMarker | None:
    markers = sorted(
        [*progress.progress_markers, *progress.checkpoint_markers],
        key=lambda marker: marker.line,
    )
    for marker in markers:
        if marker.line <= line_no:
            continue
        if _compatible_progress_advance(last_progress, marker):
            return marker
    return None


def _compatible_progress_advance(
    previous: ProgressMarker | None,
    candidate: ProgressMarker,
) -> bool:
    if candidate.value is None:
        return False
    if previous is None:
        return True
    if candidate.marker_type != previous.marker_type:
        return True
    if not isinstance(candidate.value, int) or not isinstance(previous.value, int):
        return candidate.value != previous.value
    return candidate.value > previous.value


def _episode_event_lines(lines: Sequence[LogLine]) -> dict[str, tuple[int, ...]]:
    result: dict[str, list[int]] = {
        "teardown": [],
        "process_termination": [],
        "scheduler_cancel": [],
    }
    for item in lines:
        if _TEARDOWN_RE.search(item.text):
            result["teardown"].append(item.line)
        if _PROCESS_TERMINATION_RE.search(item.text) or _BARE_PROCESS_KILLED_RE.search(item.text):
            result["process_termination"].append(item.line)
        if _SCHEDULER_CANCEL_RE.search(item.text):
            result["scheduler_cancel"].append(item.line)
    return {name: tuple(line_numbers) for name, line_numbers in result.items()}


def _first_later_line(line_numbers: Sequence[int], line_no: int) -> int | None:
    index = bisect_right(line_numbers, line_no)
    return line_numbers[index] if index < len(line_numbers) else None


def _first_at_or_after_line(line_numbers: Sequence[int], line_no: int) -> int | None:
    for candidate in line_numbers:
        if candidate >= line_no:
            return candidate
    return None


def _terminal_episode_key(text: str) -> str:
    operation = _TIMED_OPERATION_SIGNATURE_RE.search(text)
    if operation:
        return "timed_operation:" + normalized_pattern(operation.group("operation"))
    return normalized_pattern(text)


def _failure_episode_status(
    *,
    terminal_line: int | None,
    progress_after: ProgressMarker | None,
    first_teardown_line: int | None,
    first_process_termination_line: int | None,
    first_scheduler_cancel_line: int | None,
    downstream_cascade: FailureEvidence | None,
) -> str:
    if progress_after is not None:
        return FaultOutcome.PROGRESSED_AFTER.value
    if terminal_line is not None and (
        first_teardown_line is not None
        or first_process_termination_line is not None
        or first_scheduler_cancel_line is not None
        or downstream_cascade is not None
    ):
        return FaultOutcome.TERMINAL.value
    if terminal_line is not None:
        return FaultOutcome.TERMINAL.value
    return FaultOutcome.UNRESOLVED.value


def _failure_episode_reason(status: str, *, prior_progress_observed: bool) -> str:
    if status == FaultOutcome.PROGRESSED_AFTER.value:
        return "compatible application or checkpoint progress appears after the episode"
    if status == FaultOutcome.TERMINAL.value:
        prefix = (
            "terminal-looking exception after observed prior progress"
            if prior_progress_observed
            else "terminal-looking exception; no prior progress marker was observed"
        )
        return f"{prefix}, with no later compatible progress"
    return "terminal-looking exception without enough later context"


def _build_post_fault_summaries(
    lines: Sequence[LogLine],
    high_signal_lines: Sequence[int],
    episodes: Sequence[FailureEpisode],
) -> list[PostFaultSummary]:
    text_by_line = {item.line: item.text for item in lines}
    episode_patterns = {
        episode.episode_id: normalized_pattern(text_by_line.get(anchor_line, ""))
        for episode in episodes
        for anchor_line in [episode.terminal_exception_line]
        if anchor_line is not None
    }
    target_patterns = {pattern for pattern in episode_patterns.values() if pattern}
    matching_by_pattern: dict[str, list[int]] = defaultdict(list)
    if target_patterns:
        for item in lines:
            if not _EXCEPTION_SUMMARY_RE.search(item.text):
                continue
            pattern = normalized_pattern(item.text)
            if pattern in target_patterns:
                matching_by_pattern[pattern].append(item.line)
    summaries: list[PostFaultSummary] = []
    for episode in episodes:
        anchor_line = episode.terminal_exception_line
        if anchor_line is None:
            continue
        anchor_pattern = episode_patterns.get(episode.episode_id, "")
        pattern_lines = matching_by_pattern.get(anchor_pattern, [])
        matching_lines = pattern_lines[bisect_right(pattern_lines, anchor_line) :]
        later_high_signal = high_signal_lines[bisect_right(high_signal_lines, anchor_line) :]
        last_high_signal_line = later_high_signal[-1] if later_high_signal else None
        summaries.append(
            PostFaultSummary(
                episode_id=episode.episode_id,
                anchor_line=anchor_line,
                lines_after_anchor=max(0, len(lines) - anchor_line),
                progress_after_observed=episode.first_progress_after is not None,
                first_progress_after_line=(
                    episode.first_progress_after.line
                    if episode.first_progress_after is not None
                    else None
                ),
                later_matching_exception_count=len(matching_lines),
                later_matching_exception_lines=tuple(matching_lines[:20]),
                later_high_signal_count=len(later_high_signal),
                last_high_signal_line=last_high_signal_line,
                last_high_signal_quote=(
                    text_by_line.get(last_high_signal_line)
                    if last_high_signal_line is not None
                    else None
                ),
                first_teardown_line=episode.first_teardown_line,
                first_process_termination_line=episode.first_process_termination_line,
                first_scheduler_cancel_line=episode.first_scheduler_cancel_line,
                first_cascade_line=(
                    episode.first_downstream_cascade.line
                    if episode.first_downstream_cascade is not None
                    else None
                ),
            )
        )
    return summaries


def _iteration_value(text: str | None) -> int | None:
    if text is None:
        return None
    match = _ITERATION_VALUE_RE.search(text)
    return int(match.group("iteration")) if match else None


def _failure_episode_seed_lines(episodes: Sequence[FailureEpisode]) -> tuple[int, ...]:
    result: list[int] = []
    for episode in episodes:
        for line_no in (episode.start_line, episode.terminal_exception_line):
            if line_no is not None and line_no not in result:
                result.append(line_no)
    return tuple(result)


def _secondary_episode_high_signal_lines(
    lines: Sequence[LogLine],
    episodes: Sequence[FailureEpisode],
) -> set[int]:
    result: set[int] = set()
    for episode in episodes:
        for line_no in (
            *episode.duplicate_rendering_lines,
            *episode.wrapper_exception_lines,
        ):
            result.add(line_no)
            result.add(
                _traceback_start_for_line(
                    lines,
                    line_no,
                    lower_bound=max(1, line_no - 120),
                )
            )
    return result


def _context_window_ids_for_episode(
    context_windows: Sequence[ContextWindow],
    start_line: int,
    terminal_line: int | None,
) -> tuple[str, ...]:
    lines = [start_line]
    if terminal_line is not None:
        lines.append(terminal_line)
    result: list[str] = []
    for line_no in lines:
        for window_id in _context_window_ids_for_line(context_windows, line_no):
            if window_id not in result:
                result.append(window_id)
    return tuple(result)


def _build_context_windows(
    lines: list[LogLine],
    occurrence_groups: Sequence[NormalizedOccurrenceGroup],
    matches: Sequence[FailureEvidence],
    high_signal_lines: Sequence[int] = (),
    failure_episode_lines: Sequence[int] = (),
    cause_confirmation_lines: Sequence[int] = (),
) -> list[ContextWindow]:
    occurrence_group_by_line: dict[int, str] = {}
    for group in occurrence_groups:
        for line_no in group.sample_lines:
            occurrence_group_by_line[line_no] = group.occurrence_group_id

    seed_lines: list[tuple[int, str]] = []
    seen_seed_lines: set[int] = set()

    def add_seed(line_no: int | None, source: str) -> None:
        if line_no is None or line_no in seen_seed_lines:
            return
        seed_lines.append((line_no, source))
        seen_seed_lines.add(line_no)

    for line_no in failure_episode_lines:
        add_seed(line_no, "failure_episode")

    for line_no in cause_confirmation_lines:
        add_seed(line_no, "cause_confirmation")

    for line_no in _select_high_signal_lines(
        lines,
        high_signal_lines,
        limit=MAX_CONTEXT_HIGH_SIGNAL_SEEDS,
    ):
        add_seed(line_no, "high_signal")

    seen_registry_groups: set[tuple[str | None, str]] = set()
    for match in matches:
        group_key = (
            match.registry_id,
            normalized_pattern(match.quote or match.signature),
        )
        if group_key in seen_registry_groups:
            continue
        seen_registry_groups.add(group_key)
        add_seed(match.line, "registry_candidate")
        if len(seed_lines) >= MAX_CONTEXT_WINDOW_SEEDS:
            break

    windows: list[ContextWindow] = []
    for index, (seed, selected_by) in enumerate(seed_lines, start=1):
        start = max(1, seed - CONTEXT_WINDOW_BEFORE_LINES)
        end = min(len(lines), seed + CONTEXT_WINDOW_AFTER_LINES)
        window_lines = tuple(lines[start - 1 : end])
        occurrence_group_id = occurrence_group_by_line.get(seed)
        windows.append(
            ContextWindow(
                window_id=f"w-{index}",
                selected_by=selected_by,
                start_line=start,
                end_line=end,
                seed_lines=(seed,),
                occurrence_group_ids=(occurrence_group_id,) if occurrence_group_id else (),
                lines=window_lines,
                truncated=False,
            )
        )
    return windows


def _build_candidate_anchors(
    lines: Sequence[LogLine],
    high_signal_lines: Sequence[int],
    matches: Sequence[FailureEvidence],
    primary: FailureEvidence | None,
    progress: ProgressFacts,
    context_windows: Sequence[ContextWindow],
    failure_episodes: Sequence[FailureEpisode],
    cause_confirmations: Sequence[FailureEvidence],
) -> list[CandidateAnchor]:
    text_by_line = {item.line: item.text for item in lines}
    anchors: OrderedDict[int, list[str]] = OrderedDict()

    def add_anchor(line_no: int | None, source: str) -> None:
        if line_no is None or line_no not in text_by_line:
            return
        sources = anchors.setdefault(line_no, [])
        if source not in sources:
            sources.append(source)

    for line_no in _select_high_signal_lines(
        lines,
        high_signal_lines,
        limit=MAX_HIGH_SIGNAL_ANCHORS,
    ):
        add_anchor(line_no, "high_signal")
    for episode in failure_episodes:
        add_anchor(episode.start_line, "failure_episode")
        add_anchor(episode.terminal_exception_line, "terminal_exception")
    for confirmation in cause_confirmations:
        add_anchor(confirmation.line, "cause_confirmation")
    if primary is not None:
        add_anchor(primary.line, "registry_selection")
    seen_registry_groups: set[tuple[str | None, str]] = set()
    for match in matches:
        group_key = (
            match.registry_id,
            normalized_pattern(match.quote or match.signature),
        )
        if group_key in seen_registry_groups:
            continue
        seen_registry_groups.add(group_key)
        add_anchor(match.line, "registry_candidate")
        if len(anchors) >= MAX_CANDIDATE_ANCHORS:
            break

    match_by_line: dict[int, FailureEvidence] = {}
    for match in matches:
        if match.line is not None and match.line not in match_by_line:
            match_by_line[match.line] = match

    result: list[CandidateAnchor] = []
    episode_role_by_line: dict[int, str] = {}
    for episode in failure_episodes:
        if episode.terminal_exception_line is not None:
            episode_role_by_line[episode.terminal_exception_line] = (
                episode.terminal_exception_causal_role_hint
            )
        for wrapper_line in episode.wrapper_exception_lines:
            episode_role_by_line[wrapper_line] = CausalRole.TEARDOWN.value
    for index, (line_no, sources) in enumerate(
        sorted(anchors.items(), key=lambda item: item[0])[:MAX_CANDIDATE_ANCHORS],
        start=1,
    ):
        prior_progress_line = _nearest_prior_line(progress.progress_lines, line_no)
        later_progress_line = _nearest_next_line(progress.progress_lines, line_no)
        anchor_rank = extract_rank(text_by_line[line_no])
        later_progress_rank = _rank_for_line(text_by_line, later_progress_line)
        result.append(
            CandidateAnchor(
                anchor_id=f"ca-{index}",
                line=line_no,
                quote=text_by_line[line_no],
                sources=tuple(sources),
                high_signal="high_signal" in sources,
                causal_role_hint=episode_role_by_line.get(
                    line_no,
                    CausalRole.UNKNOWN.value,
                ),
                anchor_rank=anchor_rank,
                taxonomy_match=match_by_line.get(line_no),
                prior_observed_progress_line=prior_progress_line,
                later_observed_progress_line=later_progress_line,
                prior_progress_rank=_rank_for_line(text_by_line, prior_progress_line),
                later_progress_rank=later_progress_rank,
                later_progress_rank_relation=_rank_relation(anchor_rank, later_progress_rank),
                later_observation_proves_recovery=False,
                first_downstream_registry_match=_first_downstream_match(matches, line_no),
                first_downstream_cascade=_first_downstream_cascade(matches, line_no),
                context_window_ids=_context_window_ids_for_line(context_windows, line_no),
            )
        )
    return result


def _nearest_prior_line(line_numbers: Sequence[int], line_no: int) -> int | None:
    prior = [candidate for candidate in line_numbers if candidate < line_no]
    return max(prior) if prior else None


def _nearest_next_line(line_numbers: Sequence[int], line_no: int) -> int | None:
    after = [candidate for candidate in line_numbers if candidate > line_no]
    return min(after) if after else None


def _rank_for_line(text_by_line: Mapping[int, str], line_no: int | None) -> str | None:
    if line_no is None:
        return None
    text = text_by_line.get(line_no)
    if text is None:
        return None
    return extract_rank(text)


def _rank_relation(anchor_rank: str | None, progress_rank: str | None) -> str | None:
    if progress_rank is None:
        return None
    if anchor_rank is None:
        return "anchor_rank_unknown"
    if anchor_rank == progress_rank:
        return "same_rank"
    return "different_rank"


def _first_downstream_match(
    matches: Sequence[FailureEvidence],
    line_no: int,
) -> FailureEvidence | None:
    downstream = [match for match in matches if match.line is not None and match.line > line_no]
    return min(downstream, key=lambda match: match.line or 0) if downstream else None


def _first_downstream_cascade(
    matches: Sequence[FailureEvidence],
    line_no: int,
) -> FailureEvidence | None:
    downstream = [
        match
        for match in matches
        if match.line is not None
        and match.line > line_no
        and (
            match.policy_class == PolicyClass.CASCADE.value
            or match.role == RegistryRole.CASCADE_CANDIDATE.value
            or match.registry_id == "observed_distributed_operation_timeout"
        )
    ]
    return min(downstream, key=lambda match: match.line or 0) if downstream else None


def _context_window_ids_for_line(
    context_windows: Sequence[ContextWindow],
    line_no: int,
) -> tuple[str, ...]:
    return tuple(
        window.window_id
        for window in context_windows
        if window.start_line <= line_no <= window.end_line
    )


def build_cascades_for_primary(
    matches: Sequence[FailureEvidence],
    primary: FailureEvidence | None,
    distributed_incidents: Sequence[DistributedFailureIncident] = (),
) -> list[CascadeEvidence]:
    if primary is None or primary.line is None:
        return []

    primary_incident_member_lines: set[int] = set()
    downstream_incident_member_lines: set[int] = set()
    cascades: list[CascadeEvidence] = []
    for incident in distributed_incidents:
        if primary.line in incident.member_event_lines:
            primary_incident_member_lines.update(incident.member_event_lines)
            continue
        if incident.first_observed_line <= primary.line:
            continue
        downstream_incident_member_lines.update(incident.member_event_lines)
        first_match = next(
            (match for match in matches if match.line == incident.primary_observed_line),
            None,
        )
        cascades.append(
            CascadeEvidence(
                fine_class=(
                    first_match.fine_class if first_match is not None else incident.incident_type
                ),
                policy_class=PolicyClass.CASCADE.value,
                cascade_fingerprint=incident.history_fingerprint,
                causal_role=CausalRole.CASCADE.value,
                first_line=incident.first_observed_line,
                last_line=incident.last_observed_line,
                count=incident.event_count,
                sample_lines=incident.sample_lines,
                rank_spread=incident.rank_spread,
                reason=(
                    "distributed fanout appears after primary candidate at line " f"{primary.line}"
                ),
            )
        )

    groups: dict[tuple[str, str | None, str], list[FailureEvidence]] = defaultdict(list)
    for match in matches:
        if match.line is None or match.line <= primary.line:
            continue
        if match.line in primary_incident_member_lines:
            continue
        if match.line in downstream_incident_member_lines:
            continue
        if (
            match.policy_class == PolicyClass.CASCADE.value
            or match.role == RegistryRole.CASCADE_CANDIDATE.value
            or match.registry_id == "observed_distributed_operation_timeout"
        ):
            causal_role = (
                match.causal_role
                if match.causal_role in {CausalRole.CASCADE.value, CausalRole.TEARDOWN.value}
                else CausalRole.CASCADE.value
            )
            groups[(match.fine_class, match.root_fingerprint, causal_role)].append(match)

    for (fine_class, fingerprint, causal_role), group in sorted(
        groups.items(), key=lambda item: item[1][0].line or 0
    ):
        lines = [match.line for match in group if match.line is not None]
        cascades.append(
            CascadeEvidence(
                fine_class=fine_class,
                policy_class=PolicyClass.CASCADE.value,
                cascade_fingerprint=fingerprint,
                causal_role=causal_role,
                first_line=min(lines),
                last_line=max(lines),
                count=len(group),
                sample_lines=tuple(lines[:5]),
                rank_spread=tuple(sorted({match.rank for match in group if match.rank})),
                node_spread=tuple(sorted({match.node for match in group if match.node})),
                gpu_spread=tuple(sorted({match.gpu for match in group if match.gpu})),
                reason=f"appears after primary candidate at line {primary.line}",
            )
        )
    return cascades


def _collect_job_metadata(lines: Sequence[LogLine]) -> JobMetadata:
    explicit_world_size: int | None = None
    explicit_world_size_line: int | None = None
    ranks: set[int] = set()
    nodes: set[str] = set()

    for item in lines:
        text = item.text
        if explicit_world_size is None:
            world_size_match = _WORLD_SIZE_RE.search(text)
            if world_size_match:
                explicit_world_size = int(world_size_match.group("world_size"))
                explicit_world_size_line = item.line

        rank = _numeric_field(extract_rank(text))
        if rank is not None:
            ranks.add(rank)

        node = extract_node(text)
        if node:
            nodes.add(node)

    observed_rank_min = min(ranks) if ranks else None
    observed_rank_max = max(ranks) if ranks else None
    lower_bound = observed_rank_max + 1 if observed_rank_max is not None else None
    if explicit_world_size is not None:
        world_size_source = "explicit"
        world_size_confidence = "explicit"
    elif lower_bound is not None:
        world_size_source = "observed_rank_lower_bound"
        world_size_confidence = "observed_lower_bound"
    else:
        world_size_source = "not_found"
        world_size_confidence = "not_found"

    return JobMetadata(
        explicit_world_size=explicit_world_size,
        explicit_world_size_line=explicit_world_size_line,
        observed_rank_min=observed_rank_min,
        observed_rank_max=observed_rank_max,
        observed_rank_count=len(ranks),
        inferred_world_size_lower_bound=lower_bound,
        world_size_source=world_size_source,
        world_size_confidence=world_size_confidence,
        observed_node_count=len(nodes),
        rank_to_gpu_mapping_available=False,
    )


def _build_run_progress_summary(
    progress: ProgressFacts,
    failure_episodes: Sequence[FailureEpisode],
    distributed_incidents: Sequence[DistributedFailureIncident],
) -> RunProgressSummary:
    iteration_markers = [
        marker
        for marker in progress.progress_markers
        if marker.marker_type == "iteration" and isinstance(marker.value, int)
    ]
    checkpoints = [
        marker for marker in progress.checkpoint_markers if isinstance(marker.value, int)
    ]
    checkpoint_loads = [
        marker
        for marker in progress.setup_markers
        if marker.marker_type == "checkpoint_load"
        and marker.state == "completed"
        and isinstance(marker.value, int)
    ]
    checkpoint_load_iteration = checkpoint_loads[-1].value if checkpoint_loads else None
    checkpoint_load_line = checkpoint_loads[-1].line if checkpoint_loads else None
    observed_failure_iteration = progress.latest_observed_failure_iteration
    observed_iterations_after_checkpoint_load = (
        observed_failure_iteration - int(checkpoint_load_iteration)
        if observed_failure_iteration is not None
        and checkpoint_load_iteration is not None
        and observed_failure_iteration >= int(checkpoint_load_iteration)
        else None
    )
    last_setup = progress.setup_markers[-1] if progress.setup_markers else None
    first_terminal_incident = next(
        (
            incident
            for incident in distributed_incidents
            if incident.status == FaultOutcome.TERMINAL.value
        ),
        None,
    )
    terminal_fields = {
        "first_terminal_incident_line": (
            first_terminal_incident.primary_observed_line if first_terminal_incident else None
        ),
        "first_terminal_incident_timestamp": (
            first_terminal_incident.first_detection_timestamp if first_terminal_incident else None
        ),
        "configured_terminal_timeout_seconds": (
            first_terminal_incident.configured_timeout_seconds if first_terminal_incident else None
        ),
        "seconds_from_last_progress_to_terminal_incident": (
            first_terminal_incident.seconds_since_last_progress if first_terminal_incident else None
        ),
        "terminal_detection_lag_seconds": (
            first_terminal_incident.detection_lag_seconds if first_terminal_incident else None
        ),
    }
    if not iteration_markers:
        return RunProgressSummary(
            progress_marker_count=len(progress.progress_markers),
            checkpoint_marker_count=len(progress.checkpoint_markers),
            setup_marker_count=len(progress.setup_markers),
            last_checkpoint_iteration=checkpoints[-1].value if checkpoints else None,
            last_checkpoint_line=checkpoints[-1].line if checkpoints else None,
            checkpoint_load_iteration=checkpoint_load_iteration,
            checkpoint_load_line=checkpoint_load_line,
            latest_observed_failure_iteration=observed_failure_iteration,
            latest_observed_failure_iteration_line=(
                progress.latest_observed_failure_iteration_line
            ),
            observed_iterations_after_checkpoint_load=(observed_iterations_after_checkpoint_load),
            last_setup_marker_type=last_setup.marker_type if last_setup else None,
            last_setup_line=last_setup.line if last_setup else None,
            progress_after_failure_episode=_progress_after_failure_episode(failure_episodes),
            **terminal_fields,
        )

    first = iteration_markers[0]
    last = iteration_markers[-1]
    first_iteration = int(first.value)
    last_iteration = int(last.value)
    first_consumed_samples = _secondary_int(first, "consumed_samples")
    last_consumed_samples = _secondary_int(last, "consumed_samples")
    return RunProgressSummary(
        first_iteration=first_iteration,
        first_iteration_line=first.line,
        first_iteration_timestamp=first.timestamp,
        last_iteration=last_iteration,
        last_iteration_line=last.line,
        last_iteration_timestamp=last.timestamp,
        iteration_delta=last_iteration - first_iteration,
        total_iterations=_secondary_int(last, "total_iterations"),
        first_consumed_samples=first_consumed_samples,
        last_consumed_samples=last_consumed_samples,
        consumed_samples_delta=(
            last_consumed_samples - first_consumed_samples
            if first_consumed_samples is not None and last_consumed_samples is not None
            else None
        ),
        progress_marker_count=len(progress.progress_markers),
        checkpoint_marker_count=len(progress.checkpoint_markers),
        setup_marker_count=len(progress.setup_markers),
        last_checkpoint_iteration=checkpoints[-1].value if checkpoints else None,
        last_checkpoint_line=checkpoints[-1].line if checkpoints else None,
        checkpoint_load_iteration=checkpoint_load_iteration,
        checkpoint_load_line=checkpoint_load_line,
        latest_observed_failure_iteration=observed_failure_iteration,
        latest_observed_failure_iteration_line=progress.latest_observed_failure_iteration_line,
        observed_iterations_after_checkpoint_load=observed_iterations_after_checkpoint_load,
        last_setup_marker_type=last_setup.marker_type if last_setup else None,
        last_setup_line=last_setup.line if last_setup else None,
        successful_runtime_seconds=_elapsed_seconds(first.timestamp, last.timestamp),
        iterations_since_checkpoint=(
            last_iteration - int(checkpoints[-1].value)
            if checkpoints and int(checkpoints[-1].value) <= last_iteration
            else None
        ),
        progress_after_failure_episode=_progress_after_failure_episode(failure_episodes),
        **terminal_fields,
    )


def _build_operation_artifact_comparisons(
    lines: Sequence[LogLine],
    progress: ProgressFacts,
    primary: FailureEvidence | None,
    distributed_incidents: Sequence[DistributedFailureIncident],
) -> tuple[OperationArtifactComparisonEvidence, ...]:
    return (
        *_build_checkpoint_save_comparisons(
            lines,
            progress,
            primary,
            distributed_incidents,
        ),
        *_build_checkpoint_load_comparisons(lines, primary),
        *_build_dataloader_read_comparisons(lines),
    )


def _build_checkpoint_save_comparisons(
    lines: Sequence[LogLine],
    progress: ProgressFacts,
    primary: FailureEvidence | None,
    distributed_incidents: Sequence[DistributedFailureIncident],
) -> tuple[OperationArtifactComparisonEvidence, ...]:
    starts_by_artifact: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for item in lines:
        match = _CHECKPOINT_SAVE_START_RE.search(item.text)
        if match is None:
            continue
        artifact_path = _normalized_artifact_path(match.group("artifact_path"))
        starts_by_artifact[artifact_path].append((item.line, int(match.group("iteration"))))
    if not starts_by_artifact:
        return ()

    completion_markers = tuple(
        marker for marker in progress.checkpoint_markers if isinstance(marker.value, int)
    )
    terminal_lines = sorted(
        {
            incident.primary_observed_line
            for incident in distributed_incidents
            if incident.status == FaultOutcome.TERMINAL.value
        }
        | ({primary.line} if primary is not None and primary.line is not None else set())
    )
    comparisons: list[OperationArtifactComparisonEvidence] = []
    for artifact_path, starts in sorted(starts_by_artifact.items()):
        starts.sort()
        completions: list[tuple[int, int]] = []
        for index, (start_line, value) in enumerate(starts):
            next_start_line = starts[index + 1][0] if index + 1 < len(starts) else None
            completion = next(
                (
                    marker
                    for marker in completion_markers
                    if marker.value == value
                    and marker.line > start_line
                    and (next_start_line is None or marker.line < next_start_line)
                ),
                None,
            )
            if completion is not None:
                completions.append((completion.line, value))

        current_start_line, current_value = starts[-1]
        terminal_line = next(
            (line for line in terminal_lines if line > current_start_line),
            None,
        )
        current_completion_line = next(
            (
                line
                for line, value in completions
                if value == current_value and line > current_start_line
            ),
            None,
        )
        if current_completion_line is not None:
            current_outcome = "completed"
            failure_line = None
        elif terminal_line is not None and terminal_line > current_start_line:
            current_outcome = "started_not_completed"
            failure_line = terminal_line
        else:
            current_outcome = "started_unresolved"
            failure_line = None

        successes = [(line, value) for line, value in completions if line < current_start_line]
        current_logical_id = _checkpoint_artifact_id(artifact_path, current_value)
        success_pairs = tuple(
            (_checkpoint_artifact_id(artifact_path, value), None) for _, value in successes
        )
        comparison_level, comparison_counts = _artifact_comparison_summary(
            current_logical_artifact_id=current_logical_id,
            current_physical_unit_id=None,
            success_identities=success_pairs,
        )
        line_by_number = {item.line: item.text for item in lines}
        comparisons.append(
            OperationArtifactComparisonEvidence(
                operation="checkpoint_save",
                artifact_path=artifact_path,
                logical_artifact_id=current_logical_id,
                observation_kind=ArtifactObservationKind.CURRENT_LOG_COMPARISON.value,
                comparison_level=comparison_level,
                comparison_counts=comparison_counts,
                success_count=len(successes),
                success_logical_artifact_ids=tuple(value for value, _ in success_pairs),
                success_lines=tuple(line for line, _ in successes),
                successful_observer_ranks=_observer_ranks(
                    line_by_number,
                    (line for line, _ in successes),
                ),
                failed_observer_ranks=_observer_ranks(
                    line_by_number,
                    (failure_line,) if failure_line is not None else (),
                ),
                current_start_line=current_start_line,
                current_completion_line=current_completion_line,
                current_outcome=current_outcome,
                failure_line=failure_line,
                interpretation=_comparison_interpretation(comparison_level),
            )
        )
    return tuple(comparisons)


def _build_checkpoint_load_comparisons(
    lines: Sequence[LogLine],
    primary: FailureEvidence | None,
) -> tuple[OperationArtifactComparisonEvidence, ...]:
    starts: list[tuple[int, str | None, int]] = []
    completions: list[tuple[int, str | None, int, str | None]] = []
    line_by_number = {item.line: item.text for item in lines}
    for item in lines:
        start = _CHECKPOINT_LOAD_START_RE.search(item.text)
        if start is not None:
            starts.append(
                (
                    item.line,
                    _optional_artifact_path(start.group("artifact_path")),
                    int(start.group("iteration")),
                )
            )
        completion = _CHECKPOINT_LOAD_COMPLETE_RE.search(item.text)
        if completion is not None:
            completions.append(
                (
                    item.line,
                    _optional_artifact_path(completion.group("artifact_path")),
                    int(completion.group("iteration")),
                    extract_rank(item.text),
                )
            )
    if not starts:
        return ()

    known_paths_by_iteration: dict[int, set[str]] = defaultdict(set)
    for _, path, iteration in starts:
        if path is not None:
            known_paths_by_iteration[iteration].add(path)
    resolved_completions: list[tuple[int, str | None, int, str | None]] = []
    for line, path, iteration, rank in completions:
        if path is None and len(known_paths_by_iteration[iteration]) == 1:
            path = next(iter(known_paths_by_iteration[iteration]))
        resolved_completions.append((line, path, iteration, rank))

    grouped_starts: dict[str | None, list[tuple[int, int]]] = defaultdict(list)
    for line, path, iteration in starts:
        grouped_starts[path].append((line, iteration))

    primary_text = " ".join(
        value
        for value in (
            primary.fine_class if primary is not None else None,
            primary.signature if primary is not None else None,
            primary.quote if primary is not None else None,
            primary.phase if primary is not None else None,
        )
        if value
    )
    primary_is_checkpoint_load = bool(
        primary is not None
        and primary.line is not None
        and re.search(
            r"\b(?:checkpoint|deserial|decode|state[_ ]dict|tensor_to_object)\b",
            primary_text,
            re.I,
        )
    )
    comparisons: list[OperationArtifactComparisonEvidence] = []
    for artifact_path, artifact_starts in sorted(
        grouped_starts.items(),
        key=lambda item: (item[0] or "", item[1][-1][0]),
    ):
        artifact_starts.sort()
        current_start_line, current_iteration = artifact_starts[-1]
        failure_line = (
            primary.line
            if primary_is_checkpoint_load
            and primary is not None
            and primary.line is not None
            and primary.line > current_start_line
            else None
        )
        current_logical_id = _checkpoint_artifact_id(artifact_path, current_iteration)
        success_rows: list[tuple[int, str, str | None]] = []
        current_completion_line = None
        for line, path, iteration, rank in resolved_completions:
            if artifact_path is not None and path != artifact_path:
                continue
            logical_id = _checkpoint_artifact_id(path, iteration)
            if line > current_start_line and iteration == current_iteration:
                current_completion_line = current_completion_line or line
                if failure_line is not None and line < failure_line:
                    success_rows.append((line, logical_id, rank))
            elif line < current_start_line:
                success_rows.append((line, logical_id, rank))

        if failure_line is not None and current_completion_line is not None:
            current_outcome = "mixed_success_and_failure"
        elif failure_line is not None:
            current_outcome = "started_not_completed"
        elif current_completion_line is not None:
            current_outcome = "completed"
        else:
            current_outcome = "started_unresolved"
        success_pairs = tuple((logical_id, None) for _, logical_id, _ in success_rows)
        comparison_level, comparison_counts = _artifact_comparison_summary(
            current_logical_artifact_id=current_logical_id,
            current_physical_unit_id=None,
            success_identities=success_pairs,
        )
        successful_ranks = tuple(
            sorted(
                {rank for _, _, rank in success_rows if rank is not None},
                key=_rank_sort_key,
            )
        )
        failed_ranks = _observer_ranks(
            line_by_number,
            (failure_line,) if failure_line is not None else (),
        )
        observation_kind = (
            ArtifactObservationKind.DISTRIBUTED_FANOUT.value
            if successful_ranks and failed_ranks and set(successful_ranks) != set(failed_ranks)
            else ArtifactObservationKind.CURRENT_LOG_COMPARISON.value
        )
        comparisons.append(
            OperationArtifactComparisonEvidence(
                operation="checkpoint_load",
                artifact_path=artifact_path,
                logical_artifact_id=current_logical_id,
                observation_kind=observation_kind,
                comparison_level=comparison_level,
                comparison_counts=comparison_counts,
                success_count=len(success_rows),
                success_logical_artifact_ids=tuple(logical_id for _, logical_id, _ in success_rows),
                success_lines=tuple(line for line, _, _ in success_rows),
                successful_observer_ranks=successful_ranks,
                failed_observer_ranks=failed_ranks,
                current_start_line=current_start_line,
                current_completion_line=current_completion_line,
                current_outcome=current_outcome,
                failure_line=failure_line,
                interpretation=_comparison_interpretation(comparison_level),
            )
        )
    return tuple(comparisons)


def _build_dataloader_read_comparisons(
    lines: Sequence[LogLine],
) -> tuple[OperationArtifactComparisonEvidence, ...]:
    events_by_unit: dict[str, list[tuple[int, str, str | None, str | None, str | None]]] = (
        defaultdict(list)
    )
    for item in lines:
        match = _DATALOADER_READ_EVENT_RE.search(item.text)
        if match is None:
            continue
        outcome_text = match.group("outcome").lower()
        if "success" in outcome_text or "finished" in outcome_text or "complete" in outcome_text:
            outcome = "success"
        elif "failed" in outcome_text or "error" in outcome_text:
            outcome = "failure"
        else:
            outcome = "start"
        physical_unit = _normalized_artifact_path(match.group("physical_unit"))
        events_by_unit[physical_unit].append(
            (
                item.line,
                outcome,
                extract_rank(item.text),
                _data_region_identity(item.text),
                _integrity_marker(item.text),
            )
        )

    comparisons: list[OperationArtifactComparisonEvidence] = []
    all_successes = [
        (unit, event)
        for unit, unit_events in events_by_unit.items()
        for event in unit_events
        if event[1] == "success"
    ]
    for physical_unit, events in sorted(events_by_unit.items()):
        events.sort()
        failures = [event for event in events if event[1] == "failure"]
        if not failures:
            continue
        failure_line, _, failure_rank, data_region, integrity_marker = failures[-1]
        starts = [event for event in events if event[1] == "start" and event[0] < failure_line]
        current_start_line = starts[-1][0] if starts else None
        successes = [(unit, event) for unit, event in all_successes if event[0] < failure_line]
        same_unit_successes = [item for item in successes if item[0] == physical_unit]
        success_pairs = tuple((None, unit) for unit, _ in successes)
        comparison_level, comparison_counts = _artifact_comparison_summary(
            current_logical_artifact_id=None,
            current_physical_unit_id=physical_unit,
            success_identities=success_pairs,
        )
        successful_ranks = tuple(
            sorted(
                {rank for _, (_, _, rank, _, _) in successes if rank is not None},
                key=_rank_sort_key,
            )
        )
        same_unit_successful_ranks = {
            rank for _, (_, _, rank, _, _) in same_unit_successes if rank is not None
        }
        failed_ranks = (failure_rank,) if failure_rank is not None else ()
        observation_kind = (
            ArtifactObservationKind.DISTRIBUTED_FANOUT.value
            if same_unit_successes
            and same_unit_successful_ranks
            and failed_ranks
            and same_unit_successful_ranks != set(failed_ranks)
            else ArtifactObservationKind.CURRENT_LOG_COMPARISON.value
        )
        comparisons.append(
            OperationArtifactComparisonEvidence(
                operation="dataloader_read",
                artifact_path=physical_unit,
                physical_unit_id=physical_unit,
                data_region=data_region,
                integrity_marker=integrity_marker,
                observation_kind=observation_kind,
                comparison_level=comparison_level,
                comparison_counts=comparison_counts,
                success_count=len(successes),
                success_physical_unit_ids=tuple(unit for unit, _ in successes),
                success_data_regions=tuple(
                    region for _, (_, _, _, region, _) in successes if region is not None
                ),
                success_integrity_markers=tuple(
                    marker for _, (_, _, _, _, marker) in successes if marker is not None
                ),
                success_lines=tuple(line for _, (line, _, _, _, _) in successes),
                successful_observer_ranks=successful_ranks,
                failed_observer_ranks=failed_ranks,
                current_start_line=current_start_line,
                current_outcome=(
                    "mixed_success_and_failure" if same_unit_successes else "started_not_completed"
                ),
                failure_line=failure_line,
                interpretation=_comparison_interpretation(comparison_level),
            )
        )
    return tuple(comparisons)


def _checkpoint_artifact_id(artifact_path: str | None, iteration: int) -> str:
    path = artifact_path or "unknown_path"
    return f"{path}#checkpoint_iteration={iteration}"


def _optional_artifact_path(value: str | None) -> str | None:
    return _normalized_artifact_path(value) if value else None


def _normalized_artifact_path(value: str) -> str:
    stripped = value.strip().rstrip(",;:)]}")
    if stripped != "/":
        stripped = stripped.rstrip("/")
    return stripped


def _artifact_comparison_summary(
    *,
    current_logical_artifact_id: str | None,
    current_physical_unit_id: str | None,
    success_identities: Sequence[tuple[str | None, str | None]],
) -> tuple[str, Mapping[str, int]]:
    counts = {level.value: 0 for level in ArtifactComparisonLevel}
    for logical_artifact_id, physical_unit_id in success_identities:
        if current_physical_unit_id and physical_unit_id == current_physical_unit_id:
            level = ArtifactComparisonLevel.EXACT_PHYSICAL_UNIT.value
        elif current_logical_artifact_id and logical_artifact_id == current_logical_artifact_id:
            level = ArtifactComparisonLevel.SAME_LOGICAL_ARTIFACT_OTHER_OR_UNKNOWN_UNIT.value
        elif logical_artifact_id or physical_unit_id:
            level = ArtifactComparisonLevel.SAME_OPERATION_DIFFERENT_ARTIFACT.value
        else:
            level = ArtifactComparisonLevel.UNKNOWN_COMPARABILITY.value
        counts[level] += 1
    for level in (
        ArtifactComparisonLevel.EXACT_PHYSICAL_UNIT.value,
        ArtifactComparisonLevel.SAME_LOGICAL_ARTIFACT_OTHER_OR_UNKNOWN_UNIT.value,
        ArtifactComparisonLevel.SAME_OPERATION_DIFFERENT_ARTIFACT.value,
        ArtifactComparisonLevel.UNKNOWN_COMPARABILITY.value,
    ):
        if counts[level]:
            return level, counts
    return ArtifactComparisonLevel.UNKNOWN_COMPARABILITY.value, counts


def _comparison_interpretation(level: str) -> str:
    return {
        ArtifactComparisonLevel.EXACT_PHYSICAL_UNIT.value: (
            "same_physical_unit_success_observed; data_region_and_observer_locality_remain_explicit"
        ),
        ArtifactComparisonLevel.SAME_LOGICAL_ARTIFACT_OTHER_OR_UNKNOWN_UNIT.value: (
            "same_logical_artifact_success_observed_on_another_or_unknown_shard"
        ),
        ArtifactComparisonLevel.SAME_OPERATION_DIFFERENT_ARTIFACT.value: (
            "operation_pipeline_was_runnable; current_artifact_health_not_established"
        ),
        ArtifactComparisonLevel.UNKNOWN_COMPARABILITY.value: (
            "identity_is_insufficient_for_artifact_health_inference"
        ),
    }[level]


def _observer_ranks(
    line_by_number: Mapping[int, str],
    line_numbers: Iterable[int],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                rank
                for line in line_numbers
                if (rank := extract_rank(line_by_number.get(line, ""))) is not None
            },
            key=_rank_sort_key,
        )
    )


def _rank_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _data_region_identity(text: str) -> str | None:
    match = re.search(
        r"\b(?P<kind>offset|byte(?:\s+position)?|sample|record|index|token)\b"
        r"\s*(?:=|:)\s*(?P<value>\d+)\b",
        text,
        re.I,
    )
    if match is None:
        return None
    kind = re.sub(r"\s+", "_", match.group("kind").lower())
    return f"{kind}={match.group('value')}"


def _integrity_marker(text: str) -> str | None:
    match = re.search(r"\b(?:checksum|crc|hash)\b[^,;\n]{0,120}", text, re.I)
    return match.group(0).strip() if match is not None else None


def _elapsed_seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        elapsed = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except ValueError:
        return None
    return max(0.0, elapsed)


def _build_later_progress_after_fault_observations(
    matches: Sequence[FailureEvidence],
    progress: ProgressFacts,
    failure_episodes: Sequence[FailureEpisode],
) -> list[LaterProgressAfterFaultObservation]:
    progress_lines = sorted({*progress.progress_lines, *progress.checkpoint_lines})
    terminal_fingerprints = {
        match.root_fingerprint
        for match in matches
        if match.root_fingerprint
        and any(
            episode.start_line <= (match.line or 0) <= episode.end_line
            for episode in failure_episodes
        )
    }
    groups: OrderedDict[tuple[str, str | None], list[tuple[int, int]]] = OrderedDict()
    for match in matches:
        if (
            match.line is None
            or match.policy_class == PolicyClass.CASCADE.value
            or match.fault_outcome
            not in {FaultOutcome.PROGRESSED_AFTER.value, FaultOutcome.RECOVERED.value}
        ):
            continue
        later_progress = _nearest_next_line(progress_lines, match.line)
        if later_progress is None:
            continue
        groups.setdefault((match.fine_class, match.root_fingerprint), []).append(
            (match.line, later_progress)
        )

    result: list[LaterProgressAfterFaultObservation] = []
    for (fine_class, fingerprint), observations in groups.items():
        result.append(
            LaterProgressAfterFaultObservation(
                fine_class=fine_class,
                root_fingerprint=fingerprint,
                event_count=len(observations),
                sample_event_lines=tuple(line for line, _ in observations[:5]),
                sample_later_progress_lines=tuple(line for _, line in observations[:5]),
                matches_terminal_fingerprint=bool(
                    fingerprint and fingerprint in terminal_fingerprints
                ),
            )
        )
    return result


def _numeric_field(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value) if value.isdigit() else None


def _secondary_int(marker: ProgressMarker, name: str) -> int | None:
    value = marker.secondary_value.get(name)
    return value if isinstance(value, int) else None


def _progress_after_failure_episode(
    failure_episodes: Sequence[FailureEpisode],
) -> bool | None:
    if not failure_episodes:
        return None
    return any(
        episode.status == FaultOutcome.PROGRESSED_AFTER.value
        or episode.first_progress_after is not None
        for episode in failure_episodes
    )


def _coverage(
    *,
    path_hint_count: int,
    path_access_fact_count: int,
    occurrence_group_count: int,
    context_count: int,
    candidate_anchor_count: int,
    failure_episode_count: int,
    distributed_incident_count: int,
    progress: ProgressFacts,
    job_metadata: JobMetadata,
    primary: FailureEvidence | None,
    cascade_count: int,
) -> dict[str, str]:
    return {
        "path_hints": (
            CoverageStatus.CHECKED.value if path_hint_count else CoverageStatus.NOT_FOUND.value
        ),
        "path_access_facts": (
            CoverageStatus.FOUND.value if path_access_fact_count else CoverageStatus.NOT_FOUND.value
        ),
        "occurrence_groups": (
            CoverageStatus.FOUND.value if occurrence_group_count else CoverageStatus.NOT_FOUND.value
        ),
        "context_windows": (
            CoverageStatus.FOUND.value if context_count else CoverageStatus.NOT_FOUND.value
        ),
        "candidate_anchors": (
            CoverageStatus.FOUND.value if candidate_anchor_count else CoverageStatus.NOT_FOUND.value
        ),
        "application_progress": (
            CoverageStatus.FOUND.value
            if progress.progress_lines
            else CoverageStatus.NOT_FOUND.value
        ),
        "checkpoint_progress": (
            CoverageStatus.FOUND.value
            if progress.checkpoint_lines
            else CoverageStatus.NOT_FOUND.value
        ),
        "setup_progress": (
            CoverageStatus.FOUND.value if progress.setup_lines else CoverageStatus.NOT_FOUND.value
        ),
        "observed_failure_iteration": (
            CoverageStatus.FOUND.value
            if progress.latest_observed_failure_iteration is not None
            else CoverageStatus.NOT_FOUND.value
        ),
        "progress_segments": (
            CoverageStatus.FOUND.value if failure_episode_count else CoverageStatus.NOT_FOUND.value
        ),
        "distributed_failure_incidents": (
            CoverageStatus.FOUND.value
            if distributed_incident_count
            else CoverageStatus.NOT_FOUND.value
        ),
        "job_metadata": (
            CoverageStatus.FOUND.value
            if (
                job_metadata.explicit_world_size is not None
                or job_metadata.inferred_world_size_lower_bound is not None
            )
            else CoverageStatus.NOT_FOUND.value
        ),
        "first_failure_candidate": (
            CoverageStatus.FOUND.value if candidate_anchor_count else CoverageStatus.NOT_FOUND.value
        ),
        "deterministic_taxonomy_primary": (
            CoverageStatus.FOUND.value if primary else CoverageStatus.NOT_FOUND.value
        ),
        "cascade": CoverageStatus.FOUND.value if cascade_count else CoverageStatus.NOT_FOUND.value,
    }


def _has_after_context(primary: FailureEvidence | None, lines: Sequence[LogLine]) -> bool:
    return bool(primary and primary.line and primary.line < len(lines))
