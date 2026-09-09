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
RETRY_POLICY_VERSION = "retry_budget.v1"
DEFAULT_RETRY_POLICY: Mapping[str, Any] = MappingProxyType(
    {
        "concrete_confirmation_retry_allowed_retries": 1,
        "workload_confirmation_retry_allowed_retries": 1,
        "general_retry_allowed_retries": 2,
        "job_no_progress_allowed_retries": 3,
        "job_unknown_progress_allowed_retries": 3,
    }
)
REJECTED_ITERATION_RETRY_THEN_SKIP_CONTEXT_ID = "rejected_iteration_retry_then_skip"
CUDA_OOM_NO_RETRY_CONTEXT_ID = "cuda_oom_no_retry"
PORT_BIND_CONFIRMATION_RETRY_CONTEXT_ID = "port_bind_confirmation_retry"
DEFAULT_POLICY_CONTEXTS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        CUDA_OOM_NO_RETRY_CONTEXT_ID: MappingProxyType(
            {
                "enabled": True,
            }
        ),
        PORT_BIND_CONFIRMATION_RETRY_CONTEXT_ID: MappingProxyType(
            {
                "enabled": True,
                "allowed_retries": 1,
            }
        ),
        REJECTED_ITERATION_RETRY_THEN_SKIP_CONTEXT_ID: MappingProxyType(
            {
                "enabled": True,
                "allowed_retries": 2,
            }
        ),
    }
)


class Decision(str, Enum):
    STOP = "STOP"
    RESTART = "RESTART"


class DecisionBasis(str, Enum):
    LOG_UNAVAILABLE = "log_unavailable"
    WORKLOAD_UNRECOVERABLE = "workload_unrecoverable"
    POLICY_CONTEXT_NO_RETRY = "policy_context_no_retry"
    RETRY_BUDGET_EXHAUSTED = "retry_budget_exhausted"
    JOB_NO_PROGRESS_BUDGET_EXHAUSTED = "job_no_progress_budget_exhausted"
    PROGRESS_UNVERIFIABLE_BUDGET_EXHAUSTED = "progress_unverifiable_budget_exhausted"
    CONCRETE_CONFIRMATION_RETRY_AVAILABLE = "concrete_confirmation_retry_available"
    WORKLOAD_CONFIRMATION_RETRY_AVAILABLE = "workload_confirmation_retry_available"
    GENERAL_RETRY_AVAILABLE = "general_retry_available"
    POLICY_CONTEXT_RETRY_AVAILABLE = "policy_context_retry_available"
    OBSERVED_ADVANCE = "observed_advance"
    NO_PRIMARY_FAILURE = "no_primary_failure"
    MALFORMED_MODEL_OUTPUT = "malformed_model_output"


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
    RETRY_PENDING = "retry_pending"
    UNRESOLVED = "unresolved"


class RetryLifecycleState(str, Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    EXHAUSTED = "exhausted"


class CausalRole(str, Enum):
    INITIATING = "initiating"
    CASCADE = "cascade"
    TEARDOWN = "teardown"
    UNKNOWN = "unknown"


class DistributedIncidentKind(str, Enum):
    DISTRIBUTED_MECHANISM = "distributed_mechanism"
    DISTRIBUTED_FANOUT = "distributed_fanout"


class DecisionCandidateKind(str, Enum):
    DETERMINISTIC = "deterministic"
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


class HistoryProgressRelation(str, Enum):
    ADVANCED = "advanced"
    SAME = "same"
    REGRESSED = "regressed"
    UNKNOWN = "unknown"


class AffectedEntityKind(str, Enum):
    ARTIFACT = "artifact"


class AffectedEntityRelation(str, Enum):
    SAME = "same"
    DIFFERENT = "different"
    UNKNOWN = "unknown"


class HistoryMatchScope(str, Enum):
    ROOT_ONLY = "root_only"
    ROOT_AND_ENTITY = "root_and_entity"
    SAME_JOB_NO_PROGRESS = "same_job_no_progress"
    SAME_JOB_UNKNOWN_PROGRESS = "same_job_unknown_progress"
    REJECTED_ITERATION_SIGNATURE = "rejected_iteration_signature"


class RetryPolicyRule(str, Enum):
    WORKLOAD_UNRECOVERABLE = "workload_unrecoverable"
    CONCRETE_CONFIRMATION_RETRY = "concrete_confirmation_retry"
    WORKLOAD_CONFIRMATION_RETRY = "workload_confirmation_retry"
    GENERAL_RETRY = "general_retry"
    CUDA_OOM_NO_RETRY = "cuda_oom_no_retry"
    PORT_BIND_CONFIRMATION_RETRY = "port_bind_confirmation_retry"
    REJECTED_ITERATION_RETRY_THEN_SKIP = "rejected_iteration_retry_then_skip"


class FailureClassifier(str, Enum):
    """Typed, policy-neutral classifiers derived from observed failure text."""

    CUDA_OOM = "cuda_oom"
    NAN_OR_INF = "nan_or_inf"
    PORT_BIND_CONFLICT = "port_bind_conflict"
    REJECTED_NONFINITE_ITERATION = "rejected_nonfinite_iteration"


class AttemptFailureFactsSource(str, Enum):
    L0_DETERMINISTIC = "l0_deterministic"
    L2_GROUNDED = "l2_grounded"


class HistoryIdentityKind(str, Enum):
    ROOT = "root"
    OBSERVATION_ONLY = "observation_only"
    NONE = "none"


@dataclass(frozen=True)
class AffectedEntity:
    """Exact object involved in a failure, independent of failure mechanism."""

    kind: AffectedEntityKind
    identity: str
    fingerprint: str
    evidence_line: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AffectedEntityKind):
            raise TypeError("affected entity kind must be typed")
        if not isinstance(self.identity, str) or not self.identity:
            raise TypeError("affected entity identity must be a non-empty string")
        if not isinstance(self.fingerprint, str) or not self.fingerprint:
            raise TypeError("affected entity fingerprint must be a non-empty string")
        if self.evidence_line is not None and (
            isinstance(self.evidence_line, bool)
            or not isinstance(self.evidence_line, int)
            or self.evidence_line < 1
        ):
            raise TypeError("affected entity evidence_line must be a positive integer")

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


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
    root_fingerprint: str | None
    root_fingerprint_source: str | None
    fault_outcome: str | None
    identity_kind: str = HistoryIdentityKind.ROOT.value
    observation_fingerprint: str | None = None
    observation_fingerprint_source: str | None = None
    primary_line: int | None = None
    selected_observation_line: int | None = None
    selected_observation_causal_role: str | None = None
    identity_anchor_line: int | None = None
    identity_anchor_reason: str | None = None
    failure_iteration: int | None = None
    classifiers: Sequence[str] = field(default_factory=tuple)
    affected_entity: AffectedEntity | None = None
    faulting_rank: str | None = None
    faulting_node: str | None = None
    faulting_gpu: str | None = None
    root_observer_ranks: Sequence[str] | None = None
    unattributed_root_occurrence_count: int | None = None
    rank_to_gpu_map: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_identity_kinds = {item.value for item in HistoryIdentityKind}
        if self.identity_kind not in valid_identity_kinds:
            raise ValueError(f"invalid history identity kind: {self.identity_kind}")
        if self.identity_kind == HistoryIdentityKind.ROOT.value:
            if not self.root_fingerprint or self.primary_line is None:
                raise ValueError("root identity requires root_fingerprint and primary_line")
            if (
                self.observation_fingerprint is not None
                or self.selected_observation_line is not None
            ):
                raise ValueError("root identity forbids observation-only fields")
        elif self.identity_kind == HistoryIdentityKind.OBSERVATION_ONLY.value:
            if not self.observation_fingerprint or self.selected_observation_line is None:
                raise ValueError("observation-only identity requires fingerprint and selected line")
            if self.root_fingerprint is not None or self.primary_line is not None:
                raise ValueError("observation-only identity forbids root fields")
            if self.affected_entity is not None or self.root_observer_ranks is not None:
                raise ValueError("observation-only identity forbids root-scoped facts")
        elif any(
            value is not None
            for value in (
                self.root_fingerprint,
                self.observation_fingerprint,
                self.primary_line,
                self.selected_observation_line,
            )
        ):
            raise ValueError("identity kind none forbids root and observation identities")
        if any(not isinstance(item, str) or not item for item in self.classifiers):
            raise TypeError("classifiers items must be non-empty strings")
        object.__setattr__(self, "classifiers", tuple(sorted(set(self.classifiers))))
        if self.root_observer_ranks is not None:
            object.__setattr__(
                self,
                "root_observer_ranks",
                tuple(sorted({str(rank) for rank in self.root_observer_ranks})),
            )
        if self.unattributed_root_occurrence_count is not None and (
            isinstance(self.unattributed_root_occurrence_count, bool)
            or not isinstance(self.unattributed_root_occurrence_count, int)
            or self.unattributed_root_occurrence_count < 0
        ):
            raise TypeError("unattributed_root_occurrence_count must be a non-negative integer")
        if (self.root_observer_ranks is None) != (self.unattributed_root_occurrence_count is None):
            raise ValueError(
                "root_observer_ranks and unattributed_root_occurrence_count "
                "must be available or unavailable together"
            )
        object.__setattr__(self, "rank_to_gpu_map", freeze_json_value(self.rank_to_gpu_map))

    @property
    def history_identity_ready(self) -> bool:
        return bool(self.root_fingerprint or self.observation_fingerprint)

    def to_payload(self) -> dict[str, Any]:
        return {
            **_to_payload(self),
            "history_identity_ready": self.history_identity_ready,
        }


@dataclass(frozen=True)
class EnrichedAttemptFacts:
    """Independent L2-grounded primary and observation tracks for one route."""

    route_id: str
    primary: AttemptFailureFacts | None = None
    observation: AttemptFailureFacts | None = None

    def __post_init__(self) -> None:
        if not self.route_id:
            raise ValueError("route_id is required for enriched attempt facts")
        if self.primary is None and self.observation is None:
            raise ValueError("enriched attempt facts require a primary or observation track")
        if (
            self.primary is not None
            and self.primary.identity_kind != HistoryIdentityKind.ROOT.value
        ):
            raise ValueError("enriched primary track requires a root identity")
        if (
            self.observation is not None
            and self.observation.identity_kind != HistoryIdentityKind.OBSERVATION_ONLY.value
        ):
            raise ValueError("enriched observation track requires an observation-only identity")

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
    schema_version: str = RESTART_AGENT_REQUEST_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class AnalysisExecutionContext:
    """Internal context assembled from a request, history, and product config."""

    request: RestartAgentRequest
    prior_attempts: PriorAttemptView = field(default_factory=PriorAttemptView)
    retry_policy: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_RETRY_POLICY))
    policy_contexts: PolicyContextConfig = field(default_factory=lambda: PolicyContextConfig())

    @property
    def log_path(self) -> str:
        return self.request.log_path

    @property
    def job_id(self) -> str | None:
        return self.request.job_id

    @property
    def cycle_id(self) -> int | None:
        return self.request.cycle_id


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
    """Exact typed L1 recovery semantics consumed by L4 after primary grounding."""

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
    concrete_confirmation_retry_allowed_retries: int = 1
    workload_confirmation_retry_allowed_retries: int = 1
    general_retry_allowed_retries: int = 2
    job_no_progress_allowed_retries: int = 3
    job_unknown_progress_allowed_retries: int = 3

    def __post_init__(self) -> None:
        for field_name in (
            "concrete_confirmation_retry_allowed_retries",
            "workload_confirmation_retry_allowed_retries",
            "general_retry_allowed_retries",
            "job_no_progress_allowed_retries",
            "job_unknown_progress_allowed_retries",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must not be negative")
        for field_name in (
            "concrete_confirmation_retry_allowed_retries",
            "workload_confirmation_retry_allowed_retries",
        ):
            if getattr(self, field_name) > self.general_retry_allowed_retries:
                raise ValueError(f"{field_name} must not exceed general_retry_allowed_retries")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RetryPolicyConfig":
        configured = normalize_retry_policy(value or {})
        return cls(
            concrete_confirmation_retry_allowed_retries=int(
                configured["concrete_confirmation_retry_allowed_retries"]
            ),
            workload_confirmation_retry_allowed_retries=int(
                configured["workload_confirmation_retry_allowed_retries"]
            ),
            general_retry_allowed_retries=int(configured["general_retry_allowed_retries"]),
            job_no_progress_allowed_retries=int(configured["job_no_progress_allowed_retries"]),
            job_unknown_progress_allowed_retries=int(
                configured["job_unknown_progress_allowed_retries"]
            ),
        )


@dataclass(frozen=True)
class RejectedIterationRetryThenSkipConfig:
    """External workload recovery context for rejected nonfinite iterations."""

    enabled: bool = True
    allowed_retries: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("rejected_iteration_retry_then_skip.enabled must be boolean")
        if isinstance(self.allowed_retries, bool) or not isinstance(self.allowed_retries, int):
            raise TypeError("rejected_iteration_retry_then_skip.allowed_retries must be an integer")
        if self.allowed_retries < 0:
            raise ValueError(
                "rejected_iteration_retry_then_skip.allowed_retries must not be negative"
            )

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class CudaOomNoRetryConfig:
    """External product policy for a selected terminal CUDA OOM."""

    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("cuda_oom_no_retry.enabled must be boolean")

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class PortBindConfirmationRetryConfig:
    """External confirmation policy for an address-in-use bind failure."""

    enabled: bool = True
    allowed_retries: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("port_bind_confirmation_retry.enabled must be boolean")
        if isinstance(self.allowed_retries, bool) or not isinstance(self.allowed_retries, int):
            raise TypeError("port_bind_confirmation_retry.allowed_retries must be an integer")
        if self.allowed_retries < 0:
            raise ValueError("port_bind_confirmation_retry.allowed_retries must not be negative")

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class PolicyContextConfig:
    """Validated external policy contexts available to L4."""

    cuda_oom_no_retry: CudaOomNoRetryConfig = field(default_factory=CudaOomNoRetryConfig)
    port_bind_confirmation_retry: PortBindConfirmationRetryConfig = field(
        default_factory=PortBindConfirmationRetryConfig
    )
    rejected_iteration_retry_then_skip: RejectedIterationRetryThenSkipConfig = field(
        default_factory=RejectedIterationRetryThenSkipConfig
    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "PolicyContextConfig":
        configured = normalize_policy_contexts(value or {})
        cuda_oom = configured[CUDA_OOM_NO_RETRY_CONTEXT_ID]
        port_bind = configured[PORT_BIND_CONFIRMATION_RETRY_CONTEXT_ID]
        context = configured[REJECTED_ITERATION_RETRY_THEN_SKIP_CONTEXT_ID]
        return cls(
            cuda_oom_no_retry=CudaOomNoRetryConfig(
                enabled=bool(cuda_oom["enabled"]),
            ),
            port_bind_confirmation_retry=PortBindConfirmationRetryConfig(
                enabled=bool(port_bind["enabled"]),
                allowed_retries=int(port_bind["allowed_retries"]),
            ),
            rejected_iteration_retry_then_skip=RejectedIterationRetryThenSkipConfig(
                enabled=bool(context["enabled"]),
                allowed_retries=int(context["allowed_retries"]),
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            CUDA_OOM_NO_RETRY_CONTEXT_ID: self.cuda_oom_no_retry.to_payload(),
            PORT_BIND_CONFIRMATION_RETRY_CONTEXT_ID: (
                self.port_bind_confirmation_retry.to_payload()
            ),
            REJECTED_ITERATION_RETRY_THEN_SKIP_CONTEXT_ID: (
                self.rejected_iteration_retry_then_skip.to_payload()
            ),
        }


@dataclass(frozen=True)
class LogLine:
    line: int
    text: str


@dataclass(frozen=True)
class RetryLifecycle:
    state: RetryLifecycleState
    attempt: int | None = None
    max_attempts: int | None = None

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class FailureEvidence:
    failure_class: str
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
    registry_id: str | None = None
    role: str | None = None
    root_fingerprint_source: str | None = "l0_registry"
    affected_entity: AffectedEntity | None = None
    retry_lifecycle: RetryLifecycle | None = None
    observation_fingerprint: str | None = None
    observation_fingerprint_source: str | None = None

    def to_failure_payload(self) -> dict[str, Any]:
        return {
            "failure_class": self.failure_class,
            "signature": self.signature,
            "root_fingerprint": self.root_fingerprint,
            "root_fingerprint_source": self.root_fingerprint_source,
            "observation_fingerprint": self.observation_fingerprint,
            "observation_fingerprint_source": self.observation_fingerprint_source,
            "fault_outcome": self.fault_outcome,
            "retry_lifecycle": (
                self.retry_lifecycle.to_payload() if self.retry_lifecycle is not None else None
            ),
            "causal_role": self.causal_role,
            "failure_iteration": self.failure_iteration,
            "line": self.line,
            "rank": self.rank,
            "phase": self.phase,
            "node": self.node,
            "gpu": self.gpu,
            "affected_entity": (
                self.affected_entity.to_payload() if self.affected_entity is not None else None
            ),
        }


@dataclass(frozen=True)
class CascadeEvidence:
    failure_class: str
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
    unattributed_occurrence_count: int = 0
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
    lifecycle_family: str | None = None
    lifecycle_source_dialects: Sequence[str] = field(default_factory=tuple)
    lifecycle_entities: Sequence[str] = field(default_factory=tuple)
    lifecycle_fault_lines: Sequence[int] = field(default_factory=tuple)
    recovery_attempt_lines: Sequence[int] = field(default_factory=tuple)
    recovery_confirmation_lines: Sequence[int] = field(default_factory=tuple)
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
    training_progress_dialect_recognized: bool = False
    checkpoint_progress_dialect_recognized: bool = False


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
    incident_configured_timeout_seconds: float | None = None
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
    failure_class: str
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
    selected_observed_failure: FailureEvidence | None = None
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
    selected_observed_failure: FailureEvidence | None
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
    failure_narrative: Mapping[str, Any]
    decision_evidence_view: Mapping[str, Any]
    evidence_bundle: Mapping[str, Any]
    attempt_execution_context: Mapping[str, Any]
    projection_metrics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = L0_MODEL_VIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "failure_narrative",
            "decision_evidence_view",
            "evidence_bundle",
            "attempt_execution_context",
            "projection_metrics",
        ):
            object.__setattr__(self, name, freeze_json_value(getattr(self, name)))

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "failure_narrative": _to_payload(self.failure_narrative),
            "decision_evidence_view": _to_payload(self.decision_evidence_view),
            "attempt_execution_context": _to_payload(self.attempt_execution_context),
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
    affected_entity_relation: str = AffectedEntityRelation.UNKNOWN.value
    same_root_observer_count: bool = False
    same_unattributed_root_occurrence_count: bool = False

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
    same_entity_attempts: int = 0
    different_entity_attempts: int = 0
    unknown_entity_attempts: int = 0
    consecutive_same_root_no_advance_attempts: int = 0
    consecutive_same_root_and_entity_no_advance_attempts: int = 0
    advanced_beyond_all_comparable_attempts: bool = False
    advanced_beyond_all_same_entity_comparable_attempts: bool = False
    cross_node_recurrence: bool = False
    same_node_recurrence: bool = False
    same_gpu_recurrence: bool = False
    same_rank_only_recurrence: bool = False
    rank_to_gpu_mapping_available: bool = False
    job_history_available: bool = False
    job_history_availability_reason: str = "history_disabled"
    job_comparisons: Sequence[HistoryProgressComparison] = field(default_factory=tuple)
    consecutive_same_job_no_advance_attempts: int = 0
    consecutive_same_job_unknown_progress_attempts: int = 0
    job_progress_advanced: bool = False
    identity_kind: str = HistoryIdentityKind.NONE.value
    observation_history_available: bool = False
    observation_history_availability_reason: str = "history_disabled"
    matching_observation_attempts: int = 0
    observation_comparisons: Sequence[HistoryProgressComparison] = field(default_factory=tuple)
    consecutive_same_observation_no_advance_attempts: int = 0


@dataclass(frozen=True)
class JobProgressHistory:
    """Route-independent progress comparison computed once for the current cycle."""

    available: bool = False
    availability_reason: str = "history_disabled"
    same_job_attempts: int = 0
    comparisons: Sequence[HistoryProgressComparison] = field(default_factory=tuple)
    consecutive_no_advance_attempts: int = 0
    consecutive_unknown_progress_attempts: int = 0
    progress_advanced: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "comparisons", tuple(self.comparisons))

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class RouteHistorySummary:
    """Like-kind L3 comparisons for one L1 route."""

    route_id: str
    primary: HistorySummary | None = None
    observation: HistorySummary | None = None

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class CycleHistoryComparison:
    """Shared job progress plus independent deterministic and route histories."""

    job_progress: JobProgressHistory
    deterministic: HistorySummary
    routes: Sequence[RouteHistorySummary] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        routes = tuple(self.routes)
        route_ids = [item.route_id for item in routes]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("cycle history route_id values must be unique")
        object.__setattr__(self, "routes", routes)

    def route(self, route_id: str) -> RouteHistorySummary | None:
        return next((item for item in self.routes if item.route_id == route_id), None)

    def to_payload(self) -> dict[str, Any]:
        return _to_payload(self)


@dataclass(frozen=True)
class AnalysisResult:
    decision: str
    decision_basis: str
    retry_policy: Mapping[str, Any] = field(default_factory=dict)
    failure_domain: str | None = None
    result_provenance: Mapping[str, Any] = field(default_factory=dict)
    l1_assessment: Mapping[str, Any] | None = None
    l2_grounding: Mapping[str, Any] = field(default_factory=dict)
    primary_failure: Mapping[str, Any] | None = None
    observed_failures: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    selected_observed_failure: Mapping[str, Any] | None = None
    secondary_failures: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    cascades: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    evidence_coverage: Mapping[str, str] = field(default_factory=dict)
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
    l1_execution_assessment: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def selected_candidate_kind(self) -> str:
        """Return the policy candidate represented by ``analysis_result``."""

        value = self.analysis_result.result_provenance.get("candidate_kind")
        if value in (
            DecisionCandidateKind.DETERMINISTIC.value,
            DecisionCandidateKind.L1_ENRICHED.value,
        ):
            return str(value)
        return DecisionCandidateKind.DETERMINISTIC.value

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
        schema_version = value.schema_version
    elif isinstance(value, Mapping):
        allowed_fields = {
            "schema_version",
            "log_path",
            "job_id",
            "cycle_id",
        }
        unknown = sorted(set(value).difference(allowed_fields))
        if unknown:
            raise ValueError("unsupported restart-agent request fields: " + ", ".join(unknown))
        log_path = value.get("log_path")
        job_id = _optional_request_str(value.get("job_id"), "job_id")
        cycle_id = value.get("cycle_id")
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

    return RestartAgentRequest(
        log_path=normalized_log_path,
        job_id=_optional_request_str(job_id, "job_id"),
        cycle_id=_cycle_id(cycle_id),
        schema_version=RESTART_AGENT_REQUEST_SCHEMA_VERSION,
    )


def build_analysis_execution_context(
    request: RestartAgentRequest,
    *,
    prior_attempts: PriorAttemptView | None = None,
    retry_policy: Mapping[str, Any] | None = None,
    policy_contexts: Mapping[str, Any] | PolicyContextConfig | None = None,
) -> AnalysisExecutionContext:
    """Assemble validated agent-owned state around a public request."""

    normalized_retry_policy = normalize_retry_policy(retry_policy or {})
    normalized_policy_contexts = (
        policy_contexts
        if isinstance(policy_contexts, PolicyContextConfig)
        else PolicyContextConfig.from_mapping(policy_contexts)
    )
    return AnalysisExecutionContext(
        request=request,
        prior_attempts=prior_attempts or PriorAttemptView(),
        retry_policy=normalized_retry_policy,
        policy_contexts=normalized_policy_contexts,
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
            "base_rule": None,
            "effective_policy": None,
            "applied_policy_context": None,
            "decision": Decision.RESTART.value,
            "decision_basis": DecisionBasis.LOG_UNAVAILABLE.value,
            "retry_budget_exhausted": False,
            "exhausted_by": [],
            "general_root_ceiling": {
                "ledger_id": "general_root_ceiling",
                "applicable": False,
                "rule": RetryPolicyRule.GENERAL_RETRY.value,
                "history_match_scope": HistoryMatchScope.ROOT_ONLY.value,
                "allowed_retries": None,
                "matching_prior_attempts": 0,
                "observed_advance": False,
                "exhausted": False,
                "inapplicable_reason": "missing_primary",
            },
            "selected_policy_ledger": None,
            "job_no_progress_guard": {
                "ledger_id": "job_no_progress_guard",
                "applicable": False,
                "rule": "job_no_progress_guard",
                "history_match_scope": HistoryMatchScope.SAME_JOB_NO_PROGRESS.value,
                "allowed_retries": None,
                "matching_prior_attempts": 0,
                "observed_advance": False,
                "exhausted": False,
                "inapplicable_reason": "log_unavailable",
            },
            "job_unknown_progress_guard": {
                "ledger_id": "job_unknown_progress_guard",
                "applicable": False,
                "rule": "job_unknown_progress_guard",
                "history_match_scope": HistoryMatchScope.SAME_JOB_UNKNOWN_PROGRESS.value,
                "allowed_retries": None,
                "matching_prior_attempts": 0,
                "observed_advance": False,
                "exhausted": False,
                "inapplicable_reason": "log_unavailable",
            },
        },
        result_provenance={
            "candidate_kind": DecisionCandidateKind.DETERMINISTIC.value,
            "evidence_source": "log_unavailable",
            "model_contribution": "not_enabled",
            "history_contribution": "not_available",
            "result_quality": "unusable",
            "nvrx_use": "fallback_to_nvrx_default",
            "l1_execution_status": "not_run",
            "l1_execution_issues": [],
            "notes": ["log_unavailable"],
        },
        l1_assessment=None,
        l2_grounding={
            "used": False,
            "grounding_status": "not_run",
            "audit_status": "not_run",
            "not_run_reason": "log_unavailable",
            "grounded_primary_failure": None,
            "grounded_related_failures": [],
            "grounded_evidence": [],
            "audit_influence": "observational_only",
            "grounded_failure_identities": {"primary": None, "observation": None},
            "history_identity": {
                "ready": False,
                "anchor_line": None,
                "anchor_reason": None,
                "root_fingerprint": None,
                "root_fingerprint_source": "unavailable",
            },
            "grounding_adjustments": [],
            "findings": [],
        },
        primary_failure=None,
        secondary_failures=(),
        cascades=(),
        evidence_coverage=coverage,
        justification=reason,
    )


def normalize_retry_policy(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("retry_policy must be a mapping")
    unknown = sorted(set(value).difference(DEFAULT_RETRY_POLICY))
    if unknown:
        raise ValueError("unknown retry_policy fields: " + ", ".join(unknown))
    result = dict(DEFAULT_RETRY_POLICY)
    result.update(value)
    for key in (
        "concrete_confirmation_retry_allowed_retries",
        "workload_confirmation_retry_allowed_retries",
        "general_retry_allowed_retries",
        "job_no_progress_allowed_retries",
        "job_unknown_progress_allowed_retries",
    ):
        configured = result[key]
        if isinstance(configured, bool) or not isinstance(configured, int):
            raise TypeError(f"retry_policy.{key} must be an integer")
        if configured < 0:
            raise ValueError(f"retry_policy.{key} must not be negative")
    for key in (
        "concrete_confirmation_retry_allowed_retries",
        "workload_confirmation_retry_allowed_retries",
    ):
        if result[key] > result["general_retry_allowed_retries"]:
            raise ValueError(
                f"retry_policy.{key} must not exceed " "retry_policy.general_retry_allowed_retries"
            )
    return result


def normalize_policy_contexts(value: Any) -> Mapping[str, Mapping[str, Any]]:
    """Validate externally declared L4 policy contexts."""

    if not isinstance(value, Mapping):
        raise TypeError("policy_contexts must be a mapping")
    unknown_contexts = sorted(set(value).difference(DEFAULT_POLICY_CONTEXTS))
    if unknown_contexts:
        raise ValueError("unknown policy_contexts: " + ", ".join(unknown_contexts))

    result = {context_id: dict(config) for context_id, config in DEFAULT_POLICY_CONTEXTS.items()}
    for context_id, raw_config in value.items():
        if not isinstance(raw_config, Mapping):
            raise TypeError(f"policy_contexts.{context_id} must be a mapping")
        unknown_fields = sorted(set(raw_config).difference(result[context_id]))
        if unknown_fields:
            raise ValueError(
                f"policy_contexts.{context_id} has unsupported fields: " + ", ".join(unknown_fields)
            )
        result[context_id].update(raw_config)

    cuda_oom = result[CUDA_OOM_NO_RETRY_CONTEXT_ID]
    if not isinstance(cuda_oom["enabled"], bool):
        raise TypeError("policy_contexts.cuda_oom_no_retry.enabled must be boolean")

    port_bind = result[PORT_BIND_CONFIRMATION_RETRY_CONTEXT_ID]
    if not isinstance(port_bind["enabled"], bool):
        raise TypeError("policy_contexts.port_bind_confirmation_retry.enabled must be boolean")
    port_bind_allowed_retries = port_bind["allowed_retries"]
    if isinstance(port_bind_allowed_retries, bool) or not isinstance(
        port_bind_allowed_retries, int
    ):
        raise TypeError(
            "policy_contexts.port_bind_confirmation_retry.allowed_retries must be an integer"
        )
    if port_bind_allowed_retries < 0:
        raise ValueError(
            "policy_contexts.port_bind_confirmation_retry.allowed_retries must not be negative"
        )

    rejected_iteration = result[REJECTED_ITERATION_RETRY_THEN_SKIP_CONTEXT_ID]
    enabled = rejected_iteration["enabled"]
    if not isinstance(enabled, bool):
        raise TypeError(
            "policy_contexts.rejected_iteration_retry_then_skip.enabled must be boolean"
        )
    allowed_retries = rejected_iteration["allowed_retries"]
    if isinstance(allowed_retries, bool) or not isinstance(allowed_retries, int):
        raise TypeError(
            "policy_contexts.rejected_iteration_retry_then_skip.allowed_retries "
            "must be an integer"
        )
    if allowed_retries < 0:
        raise ValueError(
            "policy_contexts.rejected_iteration_retry_then_skip.allowed_retries "
            "must not be negative"
        )
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
    )
    enriched_value = value.get("enriched") or []
    if not isinstance(enriched_value, Sequence) or isinstance(enriched_value, (str, bytes)):
        raise TypeError(f"attempt records[{index}].enriched must be an array")
    enriched: list[EnrichedAttemptFacts] = []
    for entry_index, entry in enumerate(enriched_value):
        if not isinstance(entry, Mapping):
            raise TypeError(f"attempt records[{index}].enriched[{entry_index}] must be an object")
        allowed_entry_fields = {"route_id", "primary", "observation"}
        unknown_entry_fields = sorted(set(entry).difference(allowed_entry_fields))
        if unknown_entry_fields:
            raise ValueError(
                f"attempt records[{index}].enriched[{entry_index}] has unsupported fields: "
                + ", ".join(unknown_entry_fields)
            )
        route_id = _required_record_string(
            entry.get("route_id"), index, f"enriched[{entry_index}].route_id"
        )
        primary_value = entry.get("primary")
        observation_value = entry.get("observation")
        enriched.append(
            EnrichedAttemptFacts(
                route_id=route_id,
                primary=(
                    _attempt_failure_facts(
                        primary_value,
                        index,
                        f"enriched[{entry_index}].primary",
                    )
                    if primary_value is not None
                    else None
                ),
                observation=(
                    _attempt_failure_facts(
                        observation_value,
                        index,
                        f"enriched[{entry_index}].observation",
                    )
                    if observation_value is not None
                    else None
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
    if bool(root_fingerprint) != bool(root_source):
        raise ValueError(
            f"attempt records[{index}].{field_prefix} root fingerprint and source "
            "must be supplied together"
        )
    fault_outcome = _optional_str(value.get("fault_outcome"))
    if fault_outcome is not None and fault_outcome not in {item.value for item in FaultOutcome}:
        raise ValueError(f"attempt records[{index}].{field_prefix}.fault_outcome is invalid")
    rank_to_gpu_map = value.get("rank_to_gpu_map") or {}
    if not isinstance(rank_to_gpu_map, Mapping):
        raise TypeError(
            f"attempt records[{index}].{field_prefix}.rank_to_gpu_map must be an object"
        )
    observation_fingerprint = _optional_str(value.get("observation_fingerprint"))
    observation_source = _optional_str(value.get("observation_fingerprint_source"))
    if bool(observation_fingerprint) != bool(observation_source):
        raise ValueError(
            f"attempt records[{index}].{field_prefix} observation fingerprint and source "
            "must be supplied together"
        )
    identity_kind = _optional_str(value.get("identity_kind")) or (
        HistoryIdentityKind.ROOT.value
        if root_fingerprint
        else (
            HistoryIdentityKind.OBSERVATION_ONLY.value
            if observation_fingerprint
            else HistoryIdentityKind.NONE.value
        )
    )
    int_fields = (
        "primary_line",
        "selected_observation_line",
        "identity_anchor_line",
        "failure_iteration",
    )
    numbers = {
        field_name: _optional_record_int(
            value.get(field_name), index, f"{field_prefix}.{field_name}"
        )
        for field_name in int_fields
    }
    affected_entity = _affected_entity(
        value.get("affected_entity"),
        index,
        field_prefix,
    )
    classifiers_value = value.get("classifiers", ())
    if not isinstance(classifiers_value, Sequence) or isinstance(classifiers_value, (str, bytes)):
        raise TypeError(f"attempt records[{index}].{field_prefix}.classifiers must be an array")
    if any(not isinstance(item, str) or not item for item in classifiers_value):
        raise TypeError(
            f"attempt records[{index}].{field_prefix}.classifiers items "
            "must be non-empty strings"
        )
    root_observer_ranks_value = value.get("root_observer_ranks")
    root_observer_ranks: tuple[str, ...] | None
    if root_observer_ranks_value is None:
        root_observer_ranks = None
    else:
        if not isinstance(root_observer_ranks_value, Sequence) or isinstance(
            root_observer_ranks_value, (str, bytes)
        ):
            raise TypeError(
                f"attempt records[{index}].{field_prefix}.root_observer_ranks "
                "must be an array or null"
            )
        if any(not isinstance(rank, str) or not rank for rank in root_observer_ranks_value):
            raise TypeError(
                f"attempt records[{index}].{field_prefix}.root_observer_ranks "
                "items must be non-empty strings"
            )
        root_observer_ranks = tuple(root_observer_ranks_value)
    unattributed_count = _optional_record_int(
        value.get("unattributed_root_occurrence_count"),
        index,
        f"{field_prefix}.unattributed_root_occurrence_count",
    )
    if unattributed_count is not None and unattributed_count < 0:
        raise ValueError(
            f"attempt records[{index}].{field_prefix}.unattributed_root_occurrence_count "
            "must be non-negative"
        )
    return AttemptFailureFacts(
        source=source,
        identity_kind=identity_kind,
        root_fingerprint=root_fingerprint,
        root_fingerprint_source=root_source,
        observation_fingerprint=observation_fingerprint,
        observation_fingerprint_source=observation_source,
        fault_outcome=fault_outcome,
        primary_line=numbers["primary_line"],
        selected_observation_line=numbers["selected_observation_line"],
        selected_observation_causal_role=_optional_str(
            value.get("selected_observation_causal_role")
        ),
        identity_anchor_line=numbers["identity_anchor_line"],
        identity_anchor_reason=_optional_str(value.get("identity_anchor_reason")),
        failure_iteration=numbers["failure_iteration"],
        classifiers=tuple(classifiers_value),
        affected_entity=affected_entity,
        faulting_rank=_optional_str(value.get("faulting_rank")),
        faulting_node=_optional_str(value.get("faulting_node")),
        faulting_gpu=_optional_str(value.get("faulting_gpu")),
        root_observer_ranks=root_observer_ranks,
        unattributed_root_occurrence_count=unattributed_count,
        rank_to_gpu_map={str(key): str(item) for key, item in rank_to_gpu_map.items()},
    )


def _affected_entity(
    value: Any,
    index: int,
    field_prefix: str,
) -> AffectedEntity | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(
            f"attempt records[{index}].{field_prefix}.affected_entity must be an object"
        )
    expected = {"kind", "identity", "fingerprint", "evidence_line"}
    unknown = sorted(set(value).difference(expected))
    missing = sorted({"kind", "identity", "fingerprint"}.difference(value))
    if unknown:
        raise ValueError(
            f"attempt records[{index}].{field_prefix}.affected_entity "
            "has unsupported fields: " + ", ".join(unknown)
        )
    if missing:
        raise ValueError(
            f"attempt records[{index}].{field_prefix}.affected_entity is missing fields: "
            + ", ".join(missing)
        )
    try:
        kind = AffectedEntityKind(str(value["kind"]))
    except ValueError as exc:
        raise ValueError(
            f"attempt records[{index}].{field_prefix}.affected_entity.kind is invalid"
        ) from exc
    return AffectedEntity(
        kind=kind,
        identity=_required_record_string(
            value["identity"],
            index,
            f"{field_prefix}.affected_entity.identity",
        ),
        fingerprint=_required_record_string(
            value["fingerprint"],
            index,
            f"{field_prefix}.affected_entity.fingerprint",
        ),
        evidence_line=_optional_record_int(
            value.get("evidence_line"),
            index,
            f"{field_prefix}.affected_entity.evidence_line",
        ),
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
