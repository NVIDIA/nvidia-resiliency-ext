# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Library entrypoint for restart-policy decisions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Callable, TypeVar

from .execution import L0Artifacts
from .infrastructure.log_source import LocalLogSource, LogSnapshot, LogSource
from .l0 import build_l0_model_facing_view
from .l0.streaming import FinalizedL0A, ProgressiveL0Accumulator, finalize_log_snapshot
from .models import (
    AnalysisExecutionContext,
    AnalysisMode,
    DecisionEvidence,
    L0Bundle,
    L0ModelFacingView,
)
from .runtime import SYSTEM_CLOCK, Clock

_T = TypeVar("_T")
LogSourceFactory = Callable[[str], LogSource]


@dataclass(frozen=True)
class PreparedAnalysis:
    clock: Clock
    analysis_started: float
    execution_context: AnalysisExecutionContext
    bundle: L0Bundle | None
    decision_evidence: DecisionEvidence | None
    model_view: L0ModelFacingView | None
    source_log: LogSnapshot | None
    unavailable_reason: str | None
    l0a_wall_clock_s: float = 0.0
    decision_evidence_wall_clock_s: float = 0.0
    l0b_wall_clock_s: float = 0.0
    l0_wall_clock_s: float = 0.0
    l0_reused: bool = False
    l0_canonical_hash: str | None = None
    progressive_metrics: dict[str, object] | None = None


def prepare_analysis(
    execution_context: AnalysisExecutionContext,
    *,
    l0_bundle: L0Bundle | None,
    include_model_view: bool,
    log_source_factory: LogSourceFactory = LocalLogSource,
    clock: Clock = SYSTEM_CLOCK,
) -> PreparedAnalysis:
    analysis_started = clock.monotonic()
    execution_context = _snapshot_execution_context(execution_context)
    if execution_context.analysis_mode != AnalysisMode.TERMINAL.value:
        raise NotImplementedError(
            f"analysis_mode={execution_context.analysis_mode!r} is not implemented"
        )

    source = log_source_factory(execution_context.log_path)
    if source.path != execution_context.log_path:
        raise ValueError("log source path does not match runtime input")
    unavailable = source.unavailable_reason()
    if unavailable is not None:
        return PreparedAnalysis(
            clock=clock,
            analysis_started=analysis_started,
            execution_context=execution_context,
            bundle=None,
            decision_evidence=None,
            model_view=None,
            source_log=None,
            unavailable_reason=unavailable,
            l0_reused=l0_bundle is not None,
        )

    if l0_bundle is None and isinstance(source, LocalLogSource):
        finalized = ProgressiveL0Accumulator(
            source.path,
            reader=source.chunk_reader,
            clock=clock,
        ).finalize()
    else:
        finalized = finalize_log_snapshot(
            source.snapshot(),
            l0_bundle=l0_bundle,
            clock=clock,
        )
    if finalized.bundle.log_path != execution_context.log_path:
        raise ValueError("replayed L0 bundle log_path does not match runtime input")
    return _prepare_finalized_analysis(
        execution_context,
        finalized,
        include_model_view=include_model_view,
        clock=clock,
        analysis_started=analysis_started,
        l0_reused=l0_bundle is not None,
    )


def prepare_finalized_analysis(
    execution_context: AnalysisExecutionContext,
    finalized: FinalizedL0A,
    *,
    include_model_view: bool,
    clock: Clock = SYSTEM_CLOCK,
) -> PreparedAnalysis:
    """Build route-ready input from a caller-owned terminal L0A boundary."""

    return _prepare_finalized_analysis(
        _snapshot_execution_context(execution_context),
        finalized,
        include_model_view=include_model_view,
        clock=clock,
        analysis_started=clock.monotonic(),
        l0_reused=True,
    )


def _prepare_finalized_analysis(
    execution_context: AnalysisExecutionContext,
    finalized: FinalizedL0A,
    *,
    include_model_view: bool,
    clock: Clock,
    analysis_started: float,
    l0_reused: bool,
) -> PreparedAnalysis:
    if finalized.bundle.log_path != execution_context.log_path:
        raise ValueError("finalized L0A log_path does not match runtime input")
    model_view = None
    l0b_wall_clock_s = 0.0
    if include_model_view:
        l0b_started = clock.monotonic()
        model_view = build_l0_model_facing_view(
            finalized.bundle,
            finalized.decision_evidence,
        )
        l0b_wall_clock_s = round(clock.monotonic() - l0b_started, 3)

    return PreparedAnalysis(
        clock=clock,
        analysis_started=analysis_started,
        execution_context=execution_context,
        bundle=finalized.bundle,
        decision_evidence=finalized.decision_evidence,
        model_view=model_view,
        source_log=finalized.source_log,
        unavailable_reason=None,
        l0a_wall_clock_s=finalized.l0a_wall_clock_s,
        decision_evidence_wall_clock_s=finalized.decision_evidence_wall_clock_s,
        l0b_wall_clock_s=l0b_wall_clock_s,
        l0_wall_clock_s=round(
            finalized.l0a_wall_clock_s
            + finalized.decision_evidence_wall_clock_s
            + l0b_wall_clock_s,
            3,
        ),
        l0_reused=l0_reused,
        l0_canonical_hash=finalized.canonical_hash,
        progressive_metrics=dict(finalized.progressive_metrics),
    )


def _snapshot_execution_context(
    execution_context: AnalysisExecutionContext,
) -> AnalysisExecutionContext:
    return replace(
        execution_context,
        request=replace(execution_context.request),
        prior_attempts=deepcopy(execution_context.prior_attempts),
        retry_policy=deepcopy(dict(execution_context.retry_policy)),
        declared_recovery_capabilities=tuple(execution_context.declared_recovery_capabilities),
    )


def validate_timeout_seconds(timeout_seconds: float) -> float:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise TypeError("timeout_seconds must be a number")
    value = float(timeout_seconds)
    if value <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    return value


def remaining_deadline_seconds(
    deadline_monotonic: float,
    *,
    clock: Clock = SYSTEM_CLOCK,
) -> float:
    return max(0.0, deadline_monotonic - clock.monotonic())


def l0_artifacts(prepared: PreparedAnalysis) -> L0Artifacts:
    if prepared.bundle is None or prepared.decision_evidence is None:
        raise AssertionError("prepared analysis is missing L0 evidence")
    return L0Artifacts(
        bundle=prepared.bundle,
        decision_evidence=prepared.decision_evidence,
        model_view=prepared.model_view,
        l0a_wall_clock_s=prepared.l0a_wall_clock_s,
        decision_evidence_wall_clock_s=prepared.decision_evidence_wall_clock_s,
        l0b_wall_clock_s=prepared.l0b_wall_clock_s,
        l0_wall_clock_s=prepared.l0_wall_clock_s,
        l0_reused=prepared.l0_reused,
    )


def notify_l0_ready(
    callback: Callable[[L0Artifacts], None] | None,
    artifacts: L0Artifacts,
    *,
    clock: Clock = SYSTEM_CLOCK,
) -> tuple[str | None, float]:
    return notify_ready(callback, artifacts, clock=clock)


def notify_ready(
    callback: Callable[[_T], None] | None,
    value: _T,
    *,
    clock: Clock = SYSTEM_CLOCK,
) -> tuple[str | None, float]:
    """Invoke a non-critical publication callback and measure its cost."""

    if callback is None:
        return None, 0.0
    callback_started = clock.monotonic()
    error = None
    try:
        callback(value)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return error, round(clock.monotonic() - callback_started, 3)
