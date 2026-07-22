# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared dataclasses for the restart agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .immutable import freeze_json_value

RESTART_AGENT_REQUEST_SCHEMA_VERSION = "restart_agent_request.v1"
RESTART_AGENT_RESPONSE_SCHEMA_VERSION = "restart_agent_response.v1"
L1_EVIDENCE_SCHEMA_VERSION = "restart_agent_evidence.v1"
DECISION_EVIDENCE_SCHEMA_VERSION = "restart_agent_decision_evidence.v1"
L0_MODEL_VIEW_SCHEMA_VERSION = "restart_agent_l0_model_view.v1"
COLLECT_ALL_SCHEMA_VERSION = "restart_agent_collect_all.v1"
DEFAULT_RESTART_ENVIRONMENT_CONTEXT: Mapping[str, bool] = MappingProxyType(
    {
        "workload_unchanged": True,
        "process_state_recreated": True,
        "normal_restart_delay_applies": True,
        "hardware_allocation_may_change": True,
        "external_service_state_may_change": True,
    }
)
RETRY_POLICY_VERSION = "retry_budget.v1"
DEFAULT_RETRY_POLICY: Mapping[str, Any] = MappingProxyType(
    {
        "bounded_retry_allowed_retries": 1,
        "general_retry_allowed_retries": 3,
    }
)


class Decision(str, Enum):
    STOP = "STOP"
    RESTART = "RESTART"


class DecisionBasis(str, Enum):
    LOG_UNAVAILABLE = "log_unavailable"
    WORKLOAD_UNRECOVERABLE = "workload_unrecoverable"
    RETRY_BUDGET_EXHAUSTED = "retry_budget_exhausted"
    RETRY_RECOVERY_AVAILABLE = "retry_recovery_available"
    GENERAL_RETRY_AVAILABLE = "general_retry_available"
    OBSERVED_ADVANCE = "observed_advance"
    NO_PRIMARY_FAILURE = "no_primary_failure"
    TIME_LIMIT = "time_limit"
    MALFORMED_MODEL_OUTPUT = "malformed_model_output"


class PolicyClass(str, Enum):
    USER_FAILURE = "user_failure"
    NOT_USER_FAILURE = "not_user_failure"
    AMBIGUOUS = "ambiguous"
    CASCADE = "cascade"


class FailureDomain(str, Enum):
    WORKLOAD = "workload"
    INFRASTRUCTURE = "infrastructure"
    UNKNOWN = "unknown"


class RetryOutlookWithoutWorkloadChange(str, Enum):
    CANNOT_RECOVER = "cannot_recover"
    MAY_RECOVER = "may_recover"
    UNKNOWN = "unknown"


class AssessmentStatus(str, Enum):
    ESTABLISHED_BY_CURRENT_LOG = "established_by_current_log"
    SUPPORTED_BUT_UNCONFIRMED = "supported_but_unconfirmed"
    HYPOTHESIS_ONLY = "hypothesis_only"
    UNKNOWN = "unknown"


class L1AnalysisStatus(str, Enum):
    PRIMARY_IDENTIFIED = "primary_identified"
    NO_FAILURE_OBSERVED = "no_failure_observed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class FaultOutcome(str, Enum):
    TERMINAL = "terminal"
    RECOVERED = "recovered"
    PROGRESSED_AFTER = "progressed_after"
    UNRESOLVED = "unresolved"


class CausalRole(str, Enum):
    INITIATING = "initiating"
    CASCADE = "cascade"
    TEARDOWN = "teardown"
    UNKNOWN = "unknown"


class DistributedIncidentKind(str, Enum):
    DISTRIBUTED_MECHANISM = "distributed_mechanism"
    DISTRIBUTED_FANOUT = "distributed_fanout"


class AnalysisMode(str, Enum):
    TERMINAL = "terminal"
    PROGRESSIVE_START = "progressive_start"
    PROGRESSIVE_END = "progressive_end"


class DecisionCandidateKind(str, Enum):
    DETERMINISTIC_FALLBACK = "deterministic_fallback"
    L1_ENRICHED = "l1_enriched"


class ArtifactComparisonLevel(str, Enum):
    EXACT_PHYSICAL_UNIT = "exact_physical_unit"
    SAME_LOGICAL_ARTIFACT_OTHER_OR_UNKNOWN_UNIT = "same_logical_artifact_other_or_unknown_unit"
    SAME_OPERATION_DIFFERENT_ARTIFACT = "same_operation_different_artifact"
    UNKNOWN_COMPARABILITY = "unknown_comparability"


class ArtifactObservationKind(str, Enum):
    CURRENT_LOG_COMPARISON = "current_log_comparison"
    DISTRIBUTED_FANOUT = "distributed_fanout"


class CoverageStatus(str, Enum):
    CHECKED = "checked"
    FOUND = "found"
    NOT_FOUND = "not_found"
    NOT_AVAILABLE = "not_available"
    NOT_CHECKED = "not_checked"


class RegistryRole(str, Enum):
    ROOT_CANDIDATE = "root_candidate"
    CASCADE_CANDIDATE = "cascade_candidate"
    CAUSE_CONFIRMATION = "cause_confirmation"
    EITHER = "either"


class RecoveryBehavior(str, Enum):
    NONE = "none"
    RETRY_THEN_SKIP = "retry_then_skip"


class HistoryProgressRelation(str, Enum):
    ADVANCED = "advanced"
    SAME = "same"
    REGRESSED = "regressed"
    UNKNOWN = "unknown"


class RetryPolicyRule(str, Enum):
    NO_PRIMARY = "no_primary"
    TIME_LIMIT = "time_limit"
    WORKLOAD_UNRECOVERABLE = "workload_unrecoverable"
    BOUNDED_RETRY = "bounded_retry"
    GENERAL_RETRY = "general_retry"


class AttemptFailureFactsSource(str, Enum):
    L0_DETERMINISTIC = "l0_deterministic"
    L2_GROUNDED = "l2_grounded"


@dataclass(frozen=True)
class AttemptProgressSummary:
    """Route-independent progress facts derived once from the current log."""

    training_progress: str = "unknown"
    first_completed_step: int | None = None
    last_completed_step: int | None = None
    completed_step_delta: int | None = None
    progress_marker_count: int = 0
    checkpoint_progress: str = "unknown"
    checkpoint_load_step: int | None = None
    first_checkpoint_step: int | None = None
    last_checkpoint_step: int | None = None
    checkpoint_step_delta: int | None = None
    checkpoint_marker_count: int = 0
    failure_position: str = "unknown"
    progress_after_failure: str = "unknown"

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class AttemptFailureFacts:
    """Compact branch-specific failure observations used by L3."""

    source: AttemptFailureFactsSource
    fine_class: str | None
    root_fingerprint: str | None
    root_fingerprint_source: str | None
    fault_outcome: str | None
    primary_line: int | None = None
    identity_anchor_line: int | None = None
    identity_anchor_reason: str | None = None
    failure_iteration: int | None = None
    data_position_fingerprint: str | None = None
    artifact_path: str | None = None
    faulting_rank: str | None = None
    faulting_node: str | None = None
    faulting_gpu: str | None = None
    rank_to_gpu_map: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rank_to_gpu_map", freeze_json_value(self.rank_to_gpu_map))

    @property
    def history_identity_ready(self) -> bool:
        return bool(self.root_fingerprint)

    def to_payload(self) -> dict[str, Any]:
        return {
            **_to_payload(self),
            "history_identity_ready": self.history_identity_ready,
        }


@dataclass(frozen=True)
class EnrichedAttemptFacts:
    """One route-keyed L2-grounded fact block."""

    route_id: str
    facts: AttemptFailureFacts

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class AttemptRecord:
    """Neutral immutable record for a current or prior workload attempt."""

    job_id: str
    cycle_id: int
    progress: AttemptProgressSummary
    deterministic: AttemptFailureFacts
    enriched: Sequence[EnrichedAttemptFacts] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        entries = tuple(self.enriched)
        route_ids = [entry.route_id for entry in entries]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("AttemptRecord enriched route_id values must be unique")
        object.__setattr__(self, "enriched", entries)

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class PriorAttemptView:
    """Immutable runtime-selected earlier records for one invocation."""

    records: Sequence[AttemptRecord] = field(default_factory=tuple)
    available: bool = False
    availability_reason: str = "history_disabled"

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class RestartAgentRequest:
    """Validated caller-owned input to one restart-agent invocation."""

    log_path: str
    job_id: str | None = None
    cycle_id: int | None = None
    analysis_mode: str = AnalysisMode.TERMINAL.value
    schema_version: str = RESTART_AGENT_REQUEST_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class AnalysisExecutionContext:
    """Internal context assembled from a request, history, and product config."""

    request: RestartAgentRequest
    prior_attempts: PriorAttemptView = field(default_factory=PriorAttemptView)
    restart_environment_context: Mapping[str, bool] = field(
        default_factory=lambda: dict(DEFAULT_RESTART_ENVIRONMENT_CONTEXT)
    )
    retry_policy: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_RETRY_POLICY))

    @property
    def log_path(self) -> str:
        return self.request.log_path

    @property
    def job_id(self) -> str | None:
        return self.request.job_id

    @property
    def cycle_id(self) -> int | None:
        return self.request.cycle_id

    @property
    def analysis_mode(self) -> str:
        return self.request.analysis_mode


@dataclass(frozen=True)
class FailureDomainAssessment:
    """Typed L1 claim about which domain owns the observed failure."""

    value: FailureDomain
    status: AssessmentStatus
    confidence: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FailureDomainAssessment":
        return cls(
            value=FailureDomain(value.get("value")),
            status=AssessmentStatus(value.get("status")),
            confidence=_assessment_confidence(value, "failure domain"),
        )

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class RetryOutlookAssessment:
    """Typed L1 claim about recovery after the declared restart transition."""

    value: RetryOutlookWithoutWorkloadChange
    status: AssessmentStatus
    confidence: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RetryOutlookAssessment":
        return cls(
            value=RetryOutlookWithoutWorkloadChange(value.get("value")),
            status=AssessmentStatus(value.get("status")),
            confidence=_assessment_confidence(value, "retry outlook"),
        )

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


def _assessment_confidence(value: Mapping[str, Any], label: str) -> int:
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, int):
        raise TypeError(f"model {label} confidence must be an integer")
    if not 1 <= confidence <= 99:
        raise ValueError(f"model {label} confidence must be from 1 to 99")
    return confidence


@dataclass(frozen=True)
class ModelRecoveryAssessment:
    """Typed L1 recovery semantics grounded by L2 and consumed by L4."""

    failure_domain: FailureDomainAssessment
    retry_outlook_without_workload_change: RetryOutlookAssessment
    rationale: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelRecoveryAssessment":
        rationale = value.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("model recovery rationale must not be empty")
        failure_domain = value.get("failure_domain")
        retry_outlook = value.get("retry_outlook_without_workload_change")
        if not isinstance(failure_domain, Mapping):
            raise TypeError("model failure_domain must be an object")
        if not isinstance(retry_outlook, Mapping):
            raise TypeError("model retry_outlook_without_workload_change must be an object")
        return cls(
            failure_domain=FailureDomainAssessment.from_mapping(failure_domain),
            retry_outlook_without_workload_change=RetryOutlookAssessment.from_mapping(
                retry_outlook
            ),
            rationale=rationale,
        )

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class RetryPolicyConfig:
    """Validated L4 retry-budget configuration."""

    policy_version: str = RETRY_POLICY_VERSION
    bounded_retry_allowed_retries: int = 1
    general_retry_allowed_retries: int = 3

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RetryPolicyConfig":
        configured = normalize_retry_policy(value or {})
        return cls(
            bounded_retry_allowed_retries=int(configured["bounded_retry_allowed_retries"]),
            general_retry_allowed_retries=int(configured["general_retry_allowed_retries"]),
        )


@dataclass(frozen=True)
class LogLine:
    line: int
    text: str


@dataclass(frozen=True)
class FailureEvidence:
    fine_class: str
    policy_class: str
    signature: str
    root_fingerprint: str | None
    fault_outcome: str | None
    causal_role: str = CausalRole.UNKNOWN.value
    line: int | None = None
    quote: str | None = None
    rank: str | None = None
    phase: str | None = None
    node: str | None = None
    gpu: str | None = None
    failure_iteration: int | None = None
    data_position_fingerprint: str | None = None
    registry_id: str | None = None
    role: str | None = None
    recovery_behavior: str = RecoveryBehavior.NONE.value
    root_fingerprint_source: str = "l0_registry"
    failure_identity: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.failure_identity is not None:
            object.__setattr__(
                self,
                "failure_identity",
                freeze_json_value(self.failure_identity),
            )

    def to_failure_payload(self) -> dict[str, Any]:
        return {
            "fine_class": self.fine_class,
            "policy_class": self.policy_class,
            "signature": self.signature,
            "root_fingerprint": self.root_fingerprint,
            "root_fingerprint_source": self.root_fingerprint_source,
            "fault_outcome": self.fault_outcome,
            "causal_role": self.causal_role,
            "failure_iteration": self.failure_iteration,
            "data_position_fingerprint": self.data_position_fingerprint,
            "line": self.line,
            "rank": self.rank,
            "phase": self.phase,
            "node": self.node,
            "gpu": self.gpu,
            "failure_identity": dict(self.failure_identity) if self.failure_identity else None,
        }


@dataclass(frozen=True)
class CascadeEvidence:
    fine_class: str
    policy_class: str
    cascade_fingerprint: str | None
    causal_role: str
    first_line: int
    last_line: int
    count: int
    sample_lines: Sequence[int] = field(default_factory=tuple)
    rank_spread: Sequence[str] = field(default_factory=tuple)
    node_spread: Sequence[str] = field(default_factory=tuple)
    gpu_spread: Sequence[str] = field(default_factory=tuple)
    reason: str = ""
    relationship_rationales: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class NormalizedOccurrenceGroup:
    occurrence_group_id: str
    normalized_shape: str
    first_line: int
    count: int
    sample_lines: Sequence[int] = field(default_factory=tuple)
    rank_spread: Sequence[str] = field(default_factory=tuple)
    node_spread: Sequence[str] = field(default_factory=tuple)
    gpu_spread: Sequence[str] = field(default_factory=tuple)
    registry_id: str | None = None
    classification: str = "unknown"
    classification_source: str = "deterministic"


@dataclass(frozen=True)
class ContextWindow:
    window_id: str
    selected_by: str
    start_line: int
    end_line: int
    seed_lines: Sequence[int] = field(default_factory=tuple)
    occurrence_group_ids: Sequence[str] = field(default_factory=tuple)
    lines: Sequence[LogLine] = field(default_factory=tuple)
    truncated: bool = False


@dataclass(frozen=True)
class CandidateAnchor:
    anchor_id: str
    line: int
    quote: str
    sources: Sequence[str] = field(default_factory=tuple)
    high_signal: bool = False
    causal_role_hint: str = CausalRole.UNKNOWN.value
    anchor_rank: str | None = None
    taxonomy_match: FailureEvidence | None = None
    prior_observed_progress_line: int | None = None
    later_observed_progress_line: int | None = None
    prior_progress_rank: str | None = None
    later_progress_rank: str | None = None
    later_progress_rank_relation: str | None = None
    later_observation_proves_recovery: bool = False
    first_downstream_registry_match: FailureEvidence | None = None
    first_downstream_cascade: FailureEvidence | None = None
    context_window_ids: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProgressMarker:
    marker_id: str
    marker_type: str
    value: int | str | None
    state: str
    line: int
    quote: str | None = None
    timestamp: str | None = None
    rank: str | None = None
    node: str | None = None
    gpu: str | None = None
    pattern_id: str | None = None
    secondary_value: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "secondary_value", freeze_json_value(self.secondary_value))


@dataclass(frozen=True)
class FailureEpisode:
    episode_id: str
    status: str
    start_line: int
    end_line: int
    first_exception_line: int
    terminal_exception_line: int | None = None
    terminal_exception_quote: str | None = None
    terminal_exception_iteration: int | None = None
    terminal_exception_causal_role_hint: str = CausalRole.UNKNOWN.value
    precursor_lines: Sequence[int] = field(default_factory=tuple)
    identity_anchor_line: int | None = None
    identity_anchor_reason: str | None = None
    exception_chain_lines: Sequence[int] = field(default_factory=tuple)
    duplicate_rendering_lines: Sequence[int] = field(default_factory=tuple)
    wrapper_exception_lines: Sequence[int] = field(default_factory=tuple)
    exception_rank: str | None = None
    exception_node: str | None = None
    exception_gpu: str | None = None
    last_progress_before: ProgressMarker | None = None
    first_progress_after: ProgressMarker | None = None
    first_teardown_line: int | None = None
    first_process_termination_line: int | None = None
    first_scheduler_cancel_line: int | None = None
    first_downstream_cascade: FailureEvidence | None = None
    cause_confirmations: Sequence[FailureEvidence] = field(default_factory=tuple)
    context_window_ids: Sequence[str] = field(default_factory=tuple)
    reason: str = ""


@dataclass(frozen=True)
class DistributedFailureIncident:
    incident_id: str
    incident_kind: str
    incident_type: str
    status: str
    first_observed_line: int
    last_observed_line: int
    primary_observed_line: int
    primary_observed_quote: str
    member_event_lines: Sequence[int] = field(default_factory=tuple)
    sample_lines: Sequence[int] = field(default_factory=tuple)
    event_count: int = 0
    unique_operation_count: int = 0
    operation_types: Sequence[str] = field(default_factory=tuple)
    operation_signatures: Sequence[str] = field(default_factory=tuple)
    observed_rank_count: int = 0
    rank_spread: Sequence[str] = field(default_factory=tuple)
    process_group_types: Sequence[str] = field(default_factory=tuple)
    phase: str | None = None
    configured_timeout_seconds: float | None = None
    last_progress_line: int | None = None
    last_progress_timestamp: str | None = None
    first_detection_timestamp: str | None = None
    seconds_since_last_progress: float | None = None
    detection_lag_seconds: float | None = None
    history_fingerprint: str | None = None
    history_fingerprint_source: str = "l0_distributed_incident"
    root_cause_status: str = "unknown"
    interpretation: str = "observed_terminal_mechanism_not_root_cause"

    def __post_init__(self) -> None:
        valid_kinds = {item.value for item in DistributedIncidentKind}
        if self.incident_kind not in valid_kinds:
            raise ValueError(f"invalid distributed incident kind: {self.incident_kind}")
        if self.event_count < 1:
            raise ValueError("distributed incident must have at least one observed event")
        if (
            self.incident_kind == DistributedIncidentKind.DISTRIBUTED_FANOUT.value
            and self.observed_rank_count < 2
        ):
            raise ValueError("distributed fanout incident requires at least two distinct ranks")


@dataclass(frozen=True)
class PostFaultSummary:
    episode_id: str
    anchor_line: int
    lines_after_anchor: int
    progress_after_observed: bool
    first_progress_after_line: int | None = None
    later_matching_exception_count: int = 0
    later_matching_exception_lines: Sequence[int] = field(default_factory=tuple)
    later_high_signal_count: int = 0
    last_high_signal_line: int | None = None
    last_high_signal_quote: str | None = None
    first_teardown_line: int | None = None
    first_process_termination_line: int | None = None
    first_scheduler_cancel_line: int | None = None
    first_cascade_line: int | None = None


@dataclass(frozen=True)
class ProgressFacts:
    highest_completed_step: int | None = None
    last_progress_line: int | None = None
    last_checkpoint_step: int | None = None
    last_checkpoint_line: int | None = None
    latest_observed_failure_iteration: int | None = None
    latest_observed_failure_iteration_line: int | None = None
    progress_lines: Sequence[int] = field(default_factory=tuple)
    checkpoint_lines: Sequence[int] = field(default_factory=tuple)
    setup_lines: Sequence[int] = field(default_factory=tuple)
    recovery_lines: Sequence[int] = field(default_factory=tuple)
    progress_markers: Sequence[ProgressMarker] = field(default_factory=tuple)
    checkpoint_markers: Sequence[ProgressMarker] = field(default_factory=tuple)
    setup_markers: Sequence[ProgressMarker] = field(default_factory=tuple)


@dataclass(frozen=True)
class RunProgressSummary:
    first_iteration: int | None = None
    first_iteration_line: int | None = None
    first_iteration_timestamp: str | None = None
    last_iteration: int | None = None
    last_iteration_line: int | None = None
    last_iteration_timestamp: str | None = None
    iteration_delta: int | None = None
    total_iterations: int | None = None
    first_consumed_samples: int | None = None
    last_consumed_samples: int | None = None
    consumed_samples_delta: int | None = None
    progress_marker_count: int = 0
    checkpoint_marker_count: int = 0
    setup_marker_count: int = 0
    last_checkpoint_iteration: int | None = None
    last_checkpoint_line: int | None = None
    checkpoint_load_iteration: int | None = None
    checkpoint_load_line: int | None = None
    latest_observed_failure_iteration: int | None = None
    latest_observed_failure_iteration_line: int | None = None
    observed_iterations_after_checkpoint_load: int | None = None
    last_setup_marker_type: str | None = None
    last_setup_line: int | None = None
    successful_runtime_seconds: float | None = None
    iterations_since_checkpoint: int | None = None
    progress_after_failure_episode: bool | None = None
    first_terminal_incident_line: int | None = None
    first_terminal_incident_timestamp: str | None = None
    configured_terminal_timeout_seconds: float | None = None
    seconds_from_last_progress_to_terminal_incident: float | None = None
    terminal_detection_lag_seconds: float | None = None


@dataclass(frozen=True)
class OperationArtifactComparisonEvidence:
    operation: str
    artifact_path: str | None = None
    logical_artifact_id: str | None = None
    physical_unit_id: str | None = None
    data_region: str | None = None
    integrity_marker: str | None = None
    observation_kind: str = ArtifactObservationKind.CURRENT_LOG_COMPARISON.value
    comparison_level: str = ArtifactComparisonLevel.UNKNOWN_COMPARABILITY.value
    comparison_counts: Mapping[str, int] = field(default_factory=dict)
    success_count: int = 0
    success_logical_artifact_ids: Sequence[str] = field(default_factory=tuple)
    success_physical_unit_ids: Sequence[str] = field(default_factory=tuple)
    success_data_regions: Sequence[str] = field(default_factory=tuple)
    success_integrity_markers: Sequence[str] = field(default_factory=tuple)
    success_lines: Sequence[int] = field(default_factory=tuple)
    successful_observer_ranks: Sequence[str] = field(default_factory=tuple)
    failed_observer_ranks: Sequence[str] = field(default_factory=tuple)
    current_start_line: int | None = None
    current_completion_line: int | None = None
    current_outcome: str = "unknown"
    failure_line: int | None = None
    evidence_scope: str = "current_log"
    interpretation: str = "comparison_strength_is_identity_scoped"


@dataclass(frozen=True)
class LaterProgressAfterFaultObservation:
    fine_class: str
    root_fingerprint: str | None
    event_count: int
    sample_event_lines: Sequence[int] = field(default_factory=tuple)
    sample_later_progress_lines: Sequence[int] = field(default_factory=tuple)
    matches_terminal_fingerprint: bool = False
    ordering_basis: str = "log_order"
    interpretation: str = "job_progress_observed_after_event"
    component_recovery_proven: bool = False


@dataclass(frozen=True)
class JobMetadata:
    explicit_world_size: int | None = None
    explicit_world_size_line: int | None = None
    observed_rank_min: int | None = None
    observed_rank_max: int | None = None
    observed_rank_count: int = 0
    inferred_world_size_lower_bound: int | None = None
    world_size_source: str = "not_found"
    world_size_confidence: str = "not_found"
    observed_node_count: int = 0
    rank_to_gpu_mapping_available: bool = False


@dataclass(frozen=True)
class L0Bundle:
    log_path: str
    byte_size: int
    line_count: int
    path_hints: Sequence[str] = field(default_factory=tuple)
    path_access_facts: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    path_namespace_summary: Mapping[str, Any] = field(default_factory=dict)
    occurrence_groups: Sequence[NormalizedOccurrenceGroup] = field(default_factory=tuple)
    context_windows: Sequence[ContextWindow] = field(default_factory=tuple)
    candidate_anchors: Sequence[CandidateAnchor] = field(default_factory=tuple)
    registry_matches: Sequence[FailureEvidence] = field(default_factory=tuple)
    deterministic_primary_candidate: FailureEvidence | None = None
    cascades: Sequence[CascadeEvidence] = field(default_factory=tuple)
    cause_confirmations: Sequence[FailureEvidence] = field(default_factory=tuple)
    failure_episodes: Sequence[FailureEpisode] = field(default_factory=tuple)
    distributed_failure_incidents: Sequence[DistributedFailureIncident] = field(
        default_factory=tuple
    )
    post_fault_summaries: Sequence[PostFaultSummary] = field(default_factory=tuple)
    progress: ProgressFacts = field(default_factory=ProgressFacts)
    run_progress_summary: RunProgressSummary = field(default_factory=RunProgressSummary)
    operation_artifact_comparisons: Sequence[OperationArtifactComparisonEvidence] = field(
        default_factory=tuple
    )
    later_progress_after_fault_observations: Sequence[LaterProgressAfterFaultObservation] = field(
        default_factory=tuple
    )
    job_metadata: JobMetadata = field(default_factory=JobMetadata)
    evidence_coverage: Mapping[str, str] = field(default_factory=dict)
    selection_summary: Mapping[str, Any] = field(default_factory=dict)
    anomalies: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionEvidence:
    """Canonical deterministic decision facts selected from L0A."""

    deterministic_primary_candidate: FailureEvidence | None
    canonical_observed_identity: Mapping[str, Any]
    selected_evidence_references: Mapping[str, Any]
    failure_position: Mapping[str, Any]
    progress_checkpoint_state: Mapping[str, Any]
    operation_artifact_facts: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    later_progress_recovery: Mapping[str, Any] = field(default_factory=dict)
    locality: Mapping[str, Any] = field(default_factory=dict)
    coverage_lossiness: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = DECISION_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "canonical_observed_identity",
            "selected_evidence_references",
            "failure_position",
            "progress_checkpoint_state",
            "operation_artifact_facts",
            "later_progress_recovery",
            "locality",
            "coverage_lossiness",
            "provenance",
        ):
            object.__setattr__(self, name, freeze_json_value(getattr(self, name)))

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class L0ModelFacingView:
    """Deterministic L0B projection consumed by L1."""

    decision_evidence: DecisionEvidence
    evidence_bundle: Mapping[str, Any]
    attempt_execution_context: Mapping[str, Any]
    restart_environment_context: Mapping[str, bool] = field(
        default_factory=lambda: dict(DEFAULT_RESTART_ENVIRONMENT_CONTEXT)
    )
    projection_metrics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = L0_MODEL_VIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "evidence_bundle",
            "attempt_execution_context",
            "restart_environment_context",
            "projection_metrics",
        ):
            object.__setattr__(self, name, freeze_json_value(getattr(self, name)))

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "decision_evidence": self.decision_evidence.to_payload(),
            "attempt_execution_context": _to_payload(self.attempt_execution_context),
            "restart_environment_context": _to_payload(self.restart_environment_context),
            "evidence_bundle": _to_payload(self.evidence_bundle),
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            **self.prompt_payload(),
            "projection_metrics": _to_payload(self.projection_metrics),
        }


@dataclass(frozen=True)
class HistoryDimensionComparison:
    dimension: str
    prior_observation_status: str
    current_observation_status: str
    prior_value: int | None = None
    current_value: int | None = None
    delta: int | None = None
    relation: str = HistoryProgressRelation.UNKNOWN.value


@dataclass(frozen=True)
class HistoryProgressComparison:
    prior_cycle_id: int
    selected_basis: str = "none"
    dimension_comparisons: Sequence[HistoryDimensionComparison] = field(default_factory=tuple)
    positive_progress_conflict: bool = False
    relation: str = HistoryProgressRelation.UNKNOWN.value
    prior_attempt_progress: Mapping[str, Any] = field(default_factory=dict)
    prior_fault_outcome: str | None = None
    same_failure_iteration: bool = False
    same_rank: bool = False
    same_data_position: bool = False
    same_artifact: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension_comparisons", tuple(self.dimension_comparisons))
        object.__setattr__(
            self,
            "prior_attempt_progress",
            freeze_json_value(self.prior_attempt_progress),
        )


@dataclass(frozen=True)
class HistorySummary:
    available: bool = False
    availability_reason: str = "history_disabled"
    same_job_attempts: int = 0
    matching_root_attempts: int = 0
    comparisons: Sequence[HistoryProgressComparison] = field(default_factory=tuple)
    observed_advance_attempts: int = 0
    same_progress_attempts: int = 0
    regressed_progress_attempts: int = 0
    unknown_progress_attempts: int = 0
    no_observed_advance_attempts: int = 0
    matching_root_attempts_with_observed_training_progress: int = 0
    matching_root_attempts_before_observed_training_progress: int = 0
    matching_root_attempts_with_unknown_training_progress: int = 0
    exact_failure_position_attempts: int = 0
    same_rank_iteration_attempts: int = 0
    same_data_position_attempts: int = 0
    same_artifact_attempts: int = 0
    consecutive_same_root_no_advance_attempts: int = 0
    advanced_beyond_all_comparable_attempts: bool = False
    cross_node_recurrence: bool = False
    same_node_recurrence: bool = False
    same_gpu_recurrence: bool = False
    same_rank_only_recurrence: bool = False
    rank_to_gpu_mapping_available: bool = False


@dataclass(frozen=True)
class AnalysisResult:
    decision: str
    decision_basis: str
    retry_policy: Mapping[str, Any] = field(default_factory=dict)
    failure_domain: str | None = None
    result_provenance: Mapping[str, Any] = field(default_factory=dict)
    primary_failure: Mapping[str, Any] | None = None
    root_cause_assessment: Mapping[str, Any] | None = None
    model_recovery_assessment: Mapping[str, Any] | None = None
    secondary_failures: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    cascades: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    evidence_coverage: Mapping[str, str] = field(default_factory=dict)
    evidence: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    justification: str = ""
    schema_version: str = RESTART_AGENT_RESPONSE_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class DecisionCandidate:
    """A deadline-usable decision candidate produced during analysis."""

    candidate_kind: str
    result: AnalysisResult
    ready_wall_clock_s: float
    l1_execution_status: str
    history_summary: Mapping[str, Any] = field(default_factory=dict)
    stage_timings: Mapping[str, float] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class ModelAnalysisResult:
    """One independently completed model route in collect-all mode."""

    route_id: str
    model: str | None
    endpoint: str | None
    credential_ref: str | None
    execution_status: str
    l1_usable: bool
    analysis_result: AnalysisResult
    error: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class CollectAllAnalysisResult:
    """Shared-L0 result containing every model route without arbitration."""

    deterministic_result: AnalysisResult
    model_results: Sequence[ModelAnalysisResult]
    shared_analysis: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = COLLECT_ALL_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


def normalize_restart_agent_request(
    value: RestartAgentRequest | Mapping[str, Any],
) -> RestartAgentRequest:
    if isinstance(value, RestartAgentRequest):
        log_path = value.log_path
        job_id = value.job_id
        cycle_id = value.cycle_id
        analysis_mode = value.analysis_mode
        schema_version = value.schema_version
    elif isinstance(value, Mapping):
        allowed_fields = {
            "schema_version",
            "log_path",
            "job_id",
            "cycle_id",
            "analysis_mode",
        }
        unknown = sorted(set(value).difference(allowed_fields))
        if unknown:
            raise ValueError("unsupported restart-agent request fields: " + ", ".join(unknown))
        log_path = value.get("log_path")
        job_id = _optional_request_str(value.get("job_id"), "job_id")
        cycle_id = value.get("cycle_id")
        analysis_mode = str(value.get("analysis_mode") or AnalysisMode.TERMINAL.value)
        schema_version = value.get("schema_version")
    else:
        raise TypeError("restart-agent request must be RestartAgentRequest or mapping")

    if schema_version != RESTART_AGENT_REQUEST_SCHEMA_VERSION:
        raise ValueError(
            "restart-agent request schema_version must be "
            f"{RESTART_AGENT_REQUEST_SCHEMA_VERSION!r}"
        )

    if not log_path:
        raise TypeError("log_path is required")
    normalized_log_path = str(log_path)
    if not Path(normalized_log_path).is_absolute():
        raise ValueError("log_path must be absolute")

    try:
        normalized_analysis_mode = AnalysisMode(str(analysis_mode)).value
    except ValueError as exc:
        raise ValueError(f"unsupported analysis_mode: {analysis_mode!r}") from exc

    return RestartAgentRequest(
        log_path=normalized_log_path,
        job_id=_optional_request_str(job_id, "job_id"),
        cycle_id=_cycle_id(cycle_id),
        analysis_mode=normalized_analysis_mode,
        schema_version=RESTART_AGENT_REQUEST_SCHEMA_VERSION,
    )


def build_analysis_execution_context(
    request: RestartAgentRequest,
    *,
    prior_attempts: PriorAttemptView | None = None,
    restart_environment_context: Mapping[str, bool] | None = None,
    retry_policy: Mapping[str, Any] | None = None,
) -> AnalysisExecutionContext:
    """Assemble validated agent-owned state around a public request."""

    return AnalysisExecutionContext(
        request=request,
        prior_attempts=prior_attempts or PriorAttemptView(),
        restart_environment_context=normalize_restart_environment_context(
            restart_environment_context or {}
        ),
        retry_policy=normalize_retry_policy(retry_policy or {}),
    )


def log_unavailable_result(reason: str) -> AnalysisResult:
    coverage = {
        "path_hints": CoverageStatus.NOT_AVAILABLE.value,
        "occurrence_groups": CoverageStatus.NOT_AVAILABLE.value,
        "context_windows": CoverageStatus.NOT_AVAILABLE.value,
        "candidate_anchors": CoverageStatus.NOT_AVAILABLE.value,
        "application_progress": CoverageStatus.NOT_AVAILABLE.value,
        "checkpoint_progress": CoverageStatus.NOT_AVAILABLE.value,
        "setup_progress": CoverageStatus.NOT_AVAILABLE.value,
        "progress_segments": CoverageStatus.NOT_AVAILABLE.value,
        "job_metadata": CoverageStatus.NOT_AVAILABLE.value,
        "first_failure_candidate": CoverageStatus.NOT_AVAILABLE.value,
        "deterministic_taxonomy_primary": CoverageStatus.NOT_AVAILABLE.value,
        "cascade": CoverageStatus.NOT_AVAILABLE.value,
        "history": CoverageStatus.NOT_AVAILABLE.value,
    }
    return AnalysisResult(
        decision=Decision.RESTART.value,
        decision_basis=DecisionBasis.LOG_UNAVAILABLE.value,
        retry_policy={
            "policy_version": RETRY_POLICY_VERSION,
            "rule": RetryPolicyRule.NO_PRIMARY.value,
            "allowed_retries": DEFAULT_RETRY_POLICY["general_retry_allowed_retries"],
            "matching_prior_failures": 0,
            "retry_budget_exhausted": False,
        },
        result_provenance={
            "candidate_kind": DecisionCandidateKind.DETERMINISTIC_FALLBACK.value,
            "evidence_source": "fallback_log_unavailable",
            "model_contribution": "not_enabled",
            "history_contribution": "not_available",
            "result_quality": "fallback_only",
            "nvrx_use": "fallback_to_nvrx_default",
            "l1_execution_status": "not_run",
            "l1_execution_issues": [],
            "notes": ["log_unavailable"],
        },
        primary_failure=None,
        secondary_failures=(),
        cascades=(),
        evidence_coverage=coverage,
        evidence=(),
        justification=reason,
    )


def normalize_restart_environment_context(value: Any) -> Mapping[str, bool]:
    if not isinstance(value, Mapping):
        raise TypeError("restart_environment_context must be a mapping")
    unknown = sorted(set(value).difference(DEFAULT_RESTART_ENVIRONMENT_CONTEXT))
    if unknown:
        raise ValueError("unknown restart_environment_context fields: " + ", ".join(unknown))
    result = dict(DEFAULT_RESTART_ENVIRONMENT_CONTEXT)
    for key, configured in value.items():
        if not isinstance(configured, bool):
            raise TypeError(f"restart_environment_context.{key} must be a boolean")
        result[str(key)] = configured
    return result


def normalize_retry_policy(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("retry_policy must be a mapping")
    unknown = sorted(set(value).difference(DEFAULT_RETRY_POLICY))
    if unknown:
        raise ValueError("unknown retry_policy fields: " + ", ".join(unknown))
    result = dict(DEFAULT_RETRY_POLICY)
    result.update(value)
    for key in ("bounded_retry_allowed_retries", "general_retry_allowed_retries"):
        configured = result[key]
        if isinstance(configured, bool) or not isinstance(configured, int):
            raise TypeError(f"retry_policy.{key} must be an integer")
        if configured < 0:
            raise ValueError(f"retry_policy.{key} must not be negative")
    return result


def normalize_attempt_records(value: Sequence[Any]) -> tuple[AttemptRecord, ...]:
    """Validate a manual attempt-record fixture into deterministic key order."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("attempt records must be a sequence")
    records: list[AttemptRecord] = []
    seen_cycles: set[tuple[str, int]] = set()
    for index, item in enumerate(value):
        record = _attempt_record(
            item.to_payload() if isinstance(item, AttemptRecord) else item, index
        )
        identity = (record.job_id, record.cycle_id)
        if identity in seen_cycles:
            raise ValueError(
                "attempt records contain duplicate job_id/cycle_id record: "
                f"{record.job_id}/{record.cycle_id}"
            )
        seen_cycles.add(identity)
        records.append(record)
    return tuple(sorted(records, key=lambda record: (record.job_id, record.cycle_id)))


def _attempt_record(value: Any, index: int) -> AttemptRecord:
    if not isinstance(value, Mapping):
        raise TypeError(f"attempt records[{index}] must be a mapping")
    allowed = {"job_id", "cycle_id", "progress", "deterministic", "enriched"}
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"attempt records[{index}] has unsupported fields: {', '.join(unknown)}")
    job_id = _required_record_string(value.get("job_id"), index, "job_id")
    cycle_id = _required_record_int(value.get("cycle_id"), index, "cycle_id")
    progress = _attempt_progress_summary(value.get("progress"), index)
    deterministic = _attempt_failure_facts(
        value.get("deterministic"),
        index,
        "deterministic",
        required_identity=True,
    )
    enriched_value = value.get("enriched") or []
    if not isinstance(enriched_value, Sequence) or isinstance(enriched_value, (str, bytes)):
        raise TypeError(f"attempt records[{index}].enriched must be an array")
    enriched: list[EnrichedAttemptFacts] = []
    for entry_index, entry in enumerate(enriched_value):
        if not isinstance(entry, Mapping):
            raise TypeError(f"attempt records[{index}].enriched[{entry_index}] must be an object")
        route_id = _required_record_string(
            entry.get("route_id"), index, f"enriched[{entry_index}].route_id"
        )
        enriched.append(
            EnrichedAttemptFacts(
                route_id=route_id,
                facts=_attempt_failure_facts(
                    entry.get("facts"),
                    index,
                    f"enriched[{entry_index}].facts",
                    required_identity=False,
                ),
            )
        )
    return AttemptRecord(
        job_id=job_id,
        cycle_id=cycle_id,
        progress=progress,
        deterministic=deterministic,
        enriched=tuple(enriched),
    )


def _attempt_progress_summary(value: Any, index: int) -> AttemptProgressSummary:
    if not isinstance(value, Mapping):
        raise TypeError(f"attempt records[{index}].progress must be an object")
    allowed = {item.name for item in fields(AttemptProgressSummary)}
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(
            f"attempt records[{index}].progress has unsupported fields: {', '.join(unknown)}"
        )
    statuses = {"observed", "not_observed", "unknown"}
    for field_name in ("training_progress", "checkpoint_progress", "progress_after_failure"):
        if value.get(field_name, "unknown") not in statuses:
            raise ValueError(f"attempt records[{index}].progress.{field_name} is invalid")
    failure_position = value.get("failure_position", "unknown")
    if failure_position not in {
        "before_observed_training_progress",
        "after_observed_training_progress",
        "unknown",
    }:
        raise ValueError(f"attempt records[{index}].progress.failure_position is invalid")
    kwargs: dict[str, Any] = {
        "training_progress": value.get("training_progress", "unknown"),
        "checkpoint_progress": value.get("checkpoint_progress", "unknown"),
        "failure_position": failure_position,
        "progress_after_failure": value.get("progress_after_failure", "unknown"),
    }
    for field_name in allowed.difference(kwargs):
        number = value.get(field_name, 0 if field_name.endswith("_count") else None)
        if number is not None and (isinstance(number, bool) or not isinstance(number, int)):
            raise TypeError(f"attempt records[{index}].progress.{field_name} must be an integer")
        if field_name.endswith("_count") and number is not None and number < 0:
            raise ValueError(f"attempt records[{index}].progress.{field_name} must be non-negative")
        kwargs[field_name] = number
    return AttemptProgressSummary(**kwargs)


def _attempt_failure_facts(
    value: Any,
    index: int,
    field_prefix: str,
    *,
    required_identity: bool,
) -> AttemptFailureFacts:
    if not isinstance(value, Mapping):
        raise TypeError(f"attempt records[{index}].{field_prefix} must be an object")
    allowed = {item.name for item in fields(AttemptFailureFacts)}
    unknown = sorted(set(value).difference(allowed).difference({"history_identity_ready"}))
    if unknown:
        raise ValueError(
            f"attempt records[{index}].{field_prefix} has unsupported fields: " + ", ".join(unknown)
        )
    try:
        source = AttemptFailureFactsSource(value.get("source"))
    except ValueError as exc:
        raise ValueError(f"attempt records[{index}].{field_prefix}.source is invalid") from exc
    root_fingerprint = _optional_str(value.get("root_fingerprint"))
    root_source = _optional_str(value.get("root_fingerprint_source"))
    if required_identity and (not root_fingerprint or not root_source):
        raise ValueError(
            f"attempt records[{index}].{field_prefix} requires root fingerprint and source"
        )
    fault_outcome = _optional_str(value.get("fault_outcome"))
    if fault_outcome is not None and fault_outcome not in {item.value for item in FaultOutcome}:
        raise ValueError(f"attempt records[{index}].{field_prefix}.fault_outcome is invalid")
    rank_to_gpu_map = value.get("rank_to_gpu_map") or {}
    if not isinstance(rank_to_gpu_map, Mapping):
        raise TypeError(
            f"attempt records[{index}].{field_prefix}.rank_to_gpu_map must be an object"
        )
    int_fields = ("primary_line", "identity_anchor_line", "failure_iteration")
    numbers = {
        field_name: _optional_record_int(
            value.get(field_name), index, f"{field_prefix}.{field_name}"
        )
        for field_name in int_fields
    }
    return AttemptFailureFacts(
        source=source,
        fine_class=_optional_str(value.get("fine_class")),
        root_fingerprint=root_fingerprint,
        root_fingerprint_source=root_source,
        fault_outcome=fault_outcome,
        primary_line=numbers["primary_line"],
        identity_anchor_line=numbers["identity_anchor_line"],
        identity_anchor_reason=_optional_str(value.get("identity_anchor_reason")),
        failure_iteration=numbers["failure_iteration"],
        data_position_fingerprint=_optional_str(value.get("data_position_fingerprint")),
        artifact_path=_optional_str(value.get("artifact_path")),
        faulting_rank=_optional_str(value.get("faulting_rank")),
        faulting_node=_optional_str(value.get("faulting_node")),
        faulting_gpu=_optional_str(value.get("faulting_gpu")),
        rank_to_gpu_map={str(key): str(item) for key, item in rank_to_gpu_map.items()},
    )


def _required_record_string(value: Any, index: int, field_name: str) -> str:
    result = _optional_str(value)
    if result is None:
        raise ValueError(f"attempt records[{index}].{field_name} is required")
    return result


def _required_record_int(value: Any, index: int, field_name: str) -> int:
    result = _optional_record_int(value, index, field_name)
    if result is None:
        raise ValueError(f"attempt records[{index}].{field_name} is required")
    return result


def _optional_record_int(value: Any, index: int, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"attempt records[{index}].{field_name} must be an integer")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_request_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value or None


def _cycle_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("cycle_id must be an integer")
    return value


def _to_payload(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        result: dict[str, Any] = {}
        for name in value.__dataclass_fields__:
            result[name] = _to_payload(getattr(value, name))
        return result
    if isinstance(value, Mapping):
        return {str(k): _to_payload(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_payload(item) for item in value]
    if isinstance(value, list):
        return [_to_payload(item) for item in value]
    return value
