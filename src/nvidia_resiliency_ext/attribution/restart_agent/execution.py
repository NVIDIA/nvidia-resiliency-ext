# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Return-owned execution artifacts for restart-agent callers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .immutable import freeze_json_value
from .models import (
    AnalysisResult,
    AttemptRecord,
    CollectAllAnalysisResult,
    DecisionCandidate,
    DecisionEvidence,
    L0Bundle,
    L0ModelFacingView,
)


@dataclass(frozen=True)
class L0Artifacts:
    """Complete shared L0 products available before model-route fanout."""

    bundle: L0Bundle
    decision_evidence: DecisionEvidence
    model_view: L0ModelFacingView | None
    l0a_wall_clock_s: float
    decision_evidence_wall_clock_s: float
    l0b_wall_clock_s: float
    l0_wall_clock_s: float
    l0_reused: bool


@dataclass(frozen=True)
class AnalysisRun:
    """One analysis result together with the exact artifacts that produced it."""

    result: AnalysisResult
    trace: Mapping[str, Any]
    bundle: L0Bundle | None = None
    decision_evidence: DecisionEvidence | None = None
    model_view: L0ModelFacingView | None = None
    fallback_candidate: DecisionCandidate | None = None
    attempt_record: AttemptRecord | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace", freeze_json_value(self.trace))


@dataclass(frozen=True)
class CollectAllAnalysisRun:
    """One parallel route run and its shared deterministic artifacts."""

    result: CollectAllAnalysisResult
    trace: Mapping[str, Any]
    bundle: L0Bundle | None = None
    decision_evidence: DecisionEvidence | None = None
    model_view: L0ModelFacingView | None = None
    fallback_candidate: DecisionCandidate | None = None
    deterministic_attempt_record: AttemptRecord | None = None
    attempt_records_by_route: Mapping[str, AttemptRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace", freeze_json_value(self.trace))
        object.__setattr__(
            self,
            "attempt_records_by_route",
            freeze_json_value(self.attempt_records_by_route or {}),
        )
