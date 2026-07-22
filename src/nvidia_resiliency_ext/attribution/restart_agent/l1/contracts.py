# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral contracts between restart-agent pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from ..immutable import freeze_json_value
from ..models import L0ModelFacingView

DEFAULT_ANALYSIS_TIMEOUT_SECONDS = 600.0


class EvidenceTools(Protocol):
    """Read-only evidence expansion capabilities available to L1 adapters."""

    def overview(self) -> dict[str, Any]: ...

    def grep_log(
        self,
        pattern: str,
        *,
        ignore_case: bool = True,
        max_matches: int = 50,
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

    evidence: Mapping[str, Any] | None
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
        if self.evidence is not None:
            object.__setattr__(self, "evidence", freeze_json_value(self.evidence))
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
            evidence=None,
            model="",
            success=False,
            anomalies={"l1_enabled": False},
        )

    def to_trace(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.model),
            "model": self.model or None,
            "success": self.success,
            "malformed": self.malformed,
            "errors": list(self.errors),
            "raw_model_output": self.raw_model_output,
            "parsed_evidence": dict(self.evidence) if self.evidence is not None else None,
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
