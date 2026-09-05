# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral contracts between restart-agent pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from ..immutable import freeze_json_value
from ..models import L0ModelFacingView

DEFAULT_ANALYSIS_TIMEOUT_SECONDS = 240.0


class L1ExecutionStatus(str, Enum):
    """Closed lifecycle state for one L1 route invocation."""

    NOT_RUN = "not_run"
    IN_FLIGHT = "in_flight"
    COMPLETED = "completed"
    FAILED = "failed"


class L1ResultQuality(str, Enum):
    """Whether L1 produced semantics that L2 may consume."""

    NOT_APPLICABLE = "not_applicable"
    USABLE = "usable"
    DEGRADED = "degraded"
    UNUSABLE = "unusable"


class L1ParseStatus(str, Enum):
    """Closed parsing and response-contract state for model output."""

    NOT_RUN = "not_run"
    NOT_AVAILABLE = "not_available"
    VALID = "valid"
    MALFORMED = "malformed"
    CONTRACT_INVALID = "contract_invalid"


class L1FinalEvidenceReason(str, Enum):
    """Closed reasons for the one optional tools-disabled final turn."""

    CONTRACT_REPAIR = "contract_repair"
    FORCED_FINAL_AFTER_TOOL_EXHAUSTION = "forced_final_after_tool_exhaustion"
    FORCED_FINAL_AFTER_OUTPUT_LIMIT = "forced_final_after_output_limit"


class L1ExecutionReason(str, Enum):
    """Closed diagnostic reasons for degraded or unusable L1 execution."""

    ANALYSIS_DEADLINE_EXCEEDED = "analysis_deadline_exceeded"
    CONTEXT_BUDGET_EXCEEDED = "context_budget_exceeded"
    CONTEXT_WINDOW_EXCEEDED = "context_window_exceeded"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_ERROR = "provider_error"
    PROVIDER_HTTP_ERROR = "provider_http_error"
    OUTPUT_TRUNCATED = "output_truncated"
    MALFORMED_OUTPUT = "malformed_output"
    CONTRACT_INVALID = "contract_invalid"
    NO_VALID_EVIDENCE = "no_valid_evidence"
    MODEL_CALL_FAILED = "model_call_failed"
    RETRY_USED = "retry_used"
    UNSUPPORTED_TOOL_REQUEST = "unsupported_tool_request"
    TOOL_ROUND_EXHAUSTED = "tool_round_exhausted"
    CONTRACT_REPAIR = "contract_repair"
    ORCHESTRATION_ERROR = "orchestration_error"


@dataclass(frozen=True)
class L1ExecutionAssessment:
    """Provider-neutral execution envelope consumed by downstream stages."""

    execution_status: L1ExecutionStatus
    result_quality: L1ResultQuality
    parse_status: L1ParseStatus
    evidence_present: bool
    final_evidence_reason: L1FinalEvidenceReason | None = None
    reason_codes: tuple[L1ExecutionReason, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return self.result_quality in (L1ResultQuality.USABLE, L1ResultQuality.DEGRADED)

    @property
    def degraded(self) -> bool:
        return self.result_quality is L1ResultQuality.DEGRADED

    @property
    def unusable_reason(self) -> L1ExecutionReason | None:
        if self.result_quality is not L1ResultQuality.UNUSABLE or not self.reason_codes:
            return None
        return self.reason_codes[0]

    def to_payload(self) -> dict[str, Any]:
        return {
            "execution_status": self.execution_status.value,
            "result_quality": self.result_quality.value,
            "parse_status": self.parse_status.value,
            "usable": self.usable,
            "degraded": self.degraded,
            "evidence_present": self.evidence_present,
            "final_evidence_reason": (
                self.final_evidence_reason.value if self.final_evidence_reason is not None else None
            ),
            "reason_codes": [reason.value for reason in self.reason_codes],
            "unusable_reason": (
                self.unusable_reason.value if self.unusable_reason is not None else None
            ),
            "errors": list(self.errors),
        }


class EvidenceTools(Protocol):
    """Read-only evidence expansion capabilities available to L1 adapters."""

    @property
    def line_count(self) -> int: ...

    def overview(self) -> dict[str, Any]: ...

    def grep_log(
        self,
        pattern: str,
        *,
        ignore_case: bool = True,
        max_matches: int = 50,
        result_mode: str = "compact",
    ) -> dict[str, Any]: ...

    def read_window(
        self,
        center_line: int,
        *,
        before: int = 20,
        after: int = 80,
    ) -> dict[str, Any]: ...

    def get_evidence_objects(self, refs: Sequence[str]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class L1EvidenceResult:
    """Raw, provider-neutral result returned by one L1 evidence extractor."""

    semantic_payload: Mapping[str, Any] | None
    model: str
    raw_model_output: str | None = None
    success: bool = False
    malformed: bool = False
    errors: tuple[str, ...] = ()
    model_calls: tuple[Mapping[str, Any], ...] = ()
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    unsupported_tool_requests: tuple[Mapping[str, Any], ...] = ()
    transcript_events: tuple[Mapping[str, Any], ...] = ()
    anomalies: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.semantic_payload is not None:
            object.__setattr__(
                self,
                "semantic_payload",
                freeze_json_value(self.semantic_payload),
            )
        for name in (
            "model_calls",
            "tool_calls",
            "unsupported_tool_requests",
            "transcript_events",
        ):
            object.__setattr__(
                self,
                name,
                tuple(freeze_json_value(item) for item in getattr(self, name)),
            )
        object.__setattr__(self, "anomalies", freeze_json_value(self.anomalies))

    @classmethod
    def disabled(cls) -> "L1EvidenceResult":
        return cls(
            semantic_payload=None,
            model="",
            success=False,
            anomalies={"l1_enabled": False},
        )

    def category_selection(self) -> Mapping[str, Any] | None:
        """Return the optional L1 category_selection block if the model emitted it.

        The block is None when: L1 didn't run, the response was malformed, or
        the model chose not to include the optional field. Callers must treat
        None as "no signal available" and MUST NOT synthesize a default.
        """

        if not isinstance(self.semantic_payload, Mapping):
            return None
        selection = self.semantic_payload.get("category_selection")
        if not isinstance(selection, Mapping):
            return None
        return selection

    def to_trace(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.model),
            "model": self.model or None,
            "success": self.success,
            "malformed": self.malformed,
            "errors": list(self.errors),
            "raw_model_output": self.raw_model_output,
            "semantic_payload": (
                dict(self.semantic_payload) if self.semantic_payload is not None else None
            ),
            "model_calls": [dict(item) for item in self.model_calls],
            "tool_calls": [dict(item) for item in self.tool_calls],
            "unsupported_tool_requests": [dict(item) for item in self.unsupported_tool_requests],
            "interaction_transcript": [dict(item) for item in self.transcript_events],
            "anomalies": dict(self.anomalies),
        }


@dataclass(frozen=True)
class L1EvidenceContext:
    """The bounded L0B model view and controlled read-only expansion tools."""

    model_view: L0ModelFacingView
    tools: EvidenceTools


class EvidenceExtractor(Protocol):
    """Infrastructure adapter that converts L0B evidence into an L1 result."""

    def extract_evidence(
        self,
        context: L1EvidenceContext,
        *,
        deadline_monotonic: float | None = None,
    ) -> L1EvidenceResult:
        """Interpret the model view and optionally use controlled tools."""


@dataclass(frozen=True)
class ModelRoute:
    """One independently configured L1 route for parallel analysis."""

    route_id: str
    evidence_extractor: EvidenceExtractor
    model: str | None = None
    endpoint: str | None = None
    credential_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.route_id.strip():
            raise ValueError("model route_id must not be empty")
