# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-run restart-agent coordination."""

from __future__ import annotations

from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .decision_pipeline import build_decision_outcome
from .execution import AnalysisRun, L0Artifacts
from .infrastructure.log_source import LocalLogSource, LogSnapshot, LogSource
from .l1 import (
    DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
    EvidenceExtractor,
    EvidenceToolsFactory,
    L1EvidenceResult,
    deadline_exceeded_result,
    extract_timed,
    output_health,
)
from .l2 import L2GroundingInput, L2Result, ground_and_audit_model_evidence
from .models import (
    AnalysisExecutionContext,
    DecisionCandidate,
    DecisionCandidateKind,
    DecisionEvidence,
    L0Bundle,
    L0ModelFacingView,
    log_unavailable_result,
)
from .observability import (
    DecisionTraceInputs,
    build_decision_trace,
    build_log_unavailable_trace,
    history_trace,
)
from .preparation import (
    PreparedAnalysis,
    l0_artifacts,
    notify_l0_ready,
    notify_ready,
    prepare_analysis,
    remaining_deadline_seconds,
    validate_timeout_seconds,
)
from .runtime import SYSTEM_CLOCK, THREAD_EXECUTOR_FACTORY, Clock, ExecutorFactory


class SingleRunCoordinator:
    """Coordinate one deterministic and optionally model-enriched analysis run.

    L0A always builds the complete deterministic evidence bundle first. When an
    evidence extractor is supplied, L0B projects that bundle into the bounded
    model-facing view consumed by L1. L0 matches are candidate evidence; they do
    not produce semantic STOP decisions by themselves.
    """

    def __init__(
        self,
        evidence_extractor: EvidenceExtractor | None = None,
        log_source_factory: Callable[[str], LogSource] = LocalLogSource,
        evidence_tools_factory: EvidenceToolsFactory | None = None,
        clock: Clock = SYSTEM_CLOCK,
        executor_factory: ExecutorFactory = THREAD_EXECUTOR_FACTORY,
    ):
        self._evidence_extractor = evidence_extractor
        self._log_source_factory = log_source_factory
        self._evidence_tools_factory = evidence_tools_factory
        self._clock = clock
        self._executor_factory = executor_factory

    def run(
        self,
        execution_context: AnalysisExecutionContext,
        *,
        l0_bundle: L0Bundle | None = None,
        on_l0_ready: Callable[[L0Artifacts], None] | None = None,
        on_fallback_ready: Callable[[DecisionCandidate], None] | None = None,
        timeout_seconds: float = DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
        route_id: str = "single",
    ) -> AnalysisRun:
        """Return the result and all artifacts owned by this invocation."""

        timeout_seconds = validate_timeout_seconds(timeout_seconds)
        prepared = prepare_analysis(
            execution_context,
            l0_bundle=l0_bundle,
            include_model_view=self._evidence_extractor is not None,
            log_source_factory=self._log_source_factory,
            clock=self._clock,
        )
        return self.run_prepared(
            prepared,
            deadline_monotonic=prepared.analysis_started + timeout_seconds,
            on_l0_ready=on_l0_ready,
            on_fallback_ready=on_fallback_ready,
            route_id=route_id,
        )

    def run_prepared(
        self,
        prepared: PreparedAnalysis,
        *,
        deadline_monotonic: float,
        on_l0_ready: Callable[[L0Artifacts], None] | None = None,
        on_fallback_ready: Callable[[DecisionCandidate], None] | None = None,
        route_id: str = "single",
    ) -> AnalysisRun:
        """Execute one route over immutable evidence prepared by the caller."""

        analysis_started = prepared.analysis_started
        execution_context = prepared.execution_context

        if prepared.unavailable_reason is not None:
            result = log_unavailable_result(prepared.unavailable_reason)
            trace = build_log_unavailable_trace(
                result,
                execution_context,
                prepared.unavailable_reason,
                total_wall_clock_s=round(prepared.clock.monotonic() - analysis_started, 3),
            )
            return AnalysisRun(result=result, trace=trace)

        bundle = prepared.bundle
        decision_evidence = prepared.decision_evidence
        if bundle is None or decision_evidence is None:
            raise AssertionError("prepared analysis is missing L0 evidence")
        l0_callback_error, l0_callback_wall_clock_s = notify_l0_ready(
            on_l0_ready,
            l0_artifacts(prepared),
            clock=prepared.clock,
        )
        should_run_l1 = self._evidence_extractor is not None
        model_view = prepared.model_view if should_run_l1 else None

        if should_run_l1:
            extractor = self._evidence_extractor
            if extractor is None:
                raise AssertionError("L1 was selected without an evidence extractor")
            if model_view is None:
                raise AssertionError("L1 was selected without an L0B model-facing view")
            enrichment = _run_enrichment(
                extractor=extractor,
                prepared=prepared,
                bundle=bundle,
                decision_evidence=decision_evidence,
                model_view=model_view,
                deadline_monotonic=deadline_monotonic,
                on_fallback_ready=on_fallback_ready,
                evidence_tools_factory=self._evidence_tools_factory,
                executor_factory=self._executor_factory,
                route_id=route_id,
            )
        else:
            enrichment = _EnrichmentRun.disabled()

        selected_candidate_kind = (
            DecisionCandidateKind.L1_ENRICHED.value
            if enrichment.l2_result.used
            else DecisionCandidateKind.DETERMINISTIC_FALLBACK.value
        )
        outcome = build_decision_outcome(
            bundle=bundle,
            decision_evidence=decision_evidence,
            execution_context=execution_context,
            l1_configured=self._evidence_extractor is not None,
            l1_result=enrichment.l1_result,
            l1_output_health=enrichment.l1_output_health,
            l2_result=enrichment.l2_result,
            candidate_kind=selected_candidate_kind,
            l1_pending=False,
            route_id=route_id,
            clock=prepared.clock,
        )
        trace = _build_single_run_trace(
            prepared=prepared,
            enrichment=enrichment,
            outcome=outcome,
            selected_candidate_kind=selected_candidate_kind,
            deadline_monotonic=deadline_monotonic,
        )
        trace["l0_ready_callback"] = {
            "error": l0_callback_error,
            "wall_clock_s": l0_callback_wall_clock_s,
        }
        return AnalysisRun(
            result=outcome.result,
            trace=trace,
            bundle=bundle,
            decision_evidence=decision_evidence,
            model_view=model_view,
            fallback_candidate=enrichment.fallback_candidate,
            attempt_record=outcome.attempt_record,
        )


def _build_single_run_trace(
    *,
    prepared: PreparedAnalysis,
    enrichment: "_EnrichmentRun",
    outcome: Any,
    selected_candidate_kind: str,
    deadline_monotonic: float,
) -> dict[str, Any]:
    bundle = prepared.bundle
    decision_evidence = prepared.decision_evidence
    if bundle is None or decision_evidence is None:
        raise AssertionError("prepared analysis is missing L0 evidence")
    return build_decision_trace(
        DecisionTraceInputs(
            execution_context=prepared.execution_context,
            result=outcome.result,
            bundle=bundle,
            decision_evidence=decision_evidence,
            primary=outcome.primary,
            l2_primary=outcome.l2_primary,
            total_wall_clock_s=round(
                prepared.clock.monotonic() - prepared.analysis_started,
                3,
            ),
            l0_wall_clock_s=prepared.l0_wall_clock_s,
            l0a_wall_clock_s=prepared.l0a_wall_clock_s,
            decision_evidence_wall_clock_s=prepared.decision_evidence_wall_clock_s,
            l0b_wall_clock_s=prepared.l0b_wall_clock_s,
            model_view=prepared.model_view if enrichment.l1_result.model else None,
            l1_result=enrichment.l1_result,
            l1_output_health=enrichment.l1_output_health,
            l1_wall_clock_s=enrichment.l1_wall_clock_s,
            l2_audit=outcome.l2_audit,
            selected_failure_facts=outcome.selected_failure_facts,
            history=outcome.history,
            retry_policy=outcome.retry_policy,
            l0_reused=prepared.l0_reused,
            l2_wall_clock_s=enrichment.l2_wall_clock_s,
            l3_wall_clock_s=outcome.l3_wall_clock_s,
            l4_wall_clock_s=outcome.l4_wall_clock_s,
            fallback_candidate=enrichment.fallback_candidate,
            selected_candidate_kind=selected_candidate_kind,
            fallback_callback_error=enrichment.fallback_callback_error,
            fallback_callback_wall_clock_s=enrichment.fallback_callback_wall_clock_s,
            analysis_timeout_seconds=round(
                deadline_monotonic - prepared.analysis_started,
                3,
            ),
        )
    )


@dataclass(frozen=True)
class _EnrichmentRun:
    l1_result: L1EvidenceResult
    l1_output_health: Mapping[str, Any]
    l2_result: L2Result
    l1_wall_clock_s: float
    l2_wall_clock_s: float
    fallback_candidate: DecisionCandidate | None
    fallback_callback_error: str | None
    fallback_callback_wall_clock_s: float

    @classmethod
    def disabled(cls) -> "_EnrichmentRun":
        l1_result = L1EvidenceResult.disabled()
        return cls(
            l1_result=l1_result,
            l1_output_health=output_health(l1_result),
            l2_result=L2Result.not_run("l1_not_run"),
            l1_wall_clock_s=0.0,
            l2_wall_clock_s=0.0,
            fallback_candidate=None,
            fallback_callback_error=None,
            fallback_callback_wall_clock_s=0.0,
        )


def _run_enrichment(
    *,
    extractor: EvidenceExtractor,
    prepared: PreparedAnalysis,
    bundle: L0Bundle,
    decision_evidence: DecisionEvidence,
    model_view: L0ModelFacingView,
    deadline_monotonic: float,
    on_fallback_ready: Callable[[DecisionCandidate], None] | None,
    evidence_tools_factory: EvidenceToolsFactory | None,
    executor_factory: ExecutorFactory,
    route_id: str,
) -> _EnrichmentRun:
    pending_l1 = L1EvidenceResult.disabled()
    fallback_outcome = build_decision_outcome(
        bundle=bundle,
        decision_evidence=decision_evidence,
        execution_context=prepared.execution_context,
        l1_configured=True,
        l1_result=pending_l1,
        l1_output_health={"status": "pending", "usable": False, "errors": []},
        l2_result=L2Result.not_run("l1_pending"),
        candidate_kind=DecisionCandidateKind.DETERMINISTIC_FALLBACK.value,
        l1_pending=True,
        route_id=route_id,
        clock=prepared.clock,
    )
    fallback_candidate = DecisionCandidate(
        candidate_kind=DecisionCandidateKind.DETERMINISTIC_FALLBACK.value,
        result=fallback_outcome.result,
        ready_wall_clock_s=round(
            prepared.clock.monotonic() - prepared.analysis_started,
            3,
        ),
        l1_execution_status="in_flight",
        history_summary=history_trace(fallback_outcome.history),
        stage_timings={
            "l0_wall_clock_s": prepared.l0_wall_clock_s,
            "l0a_wall_clock_s": prepared.l0a_wall_clock_s,
            "decision_evidence_wall_clock_s": prepared.decision_evidence_wall_clock_s,
            "l0b_wall_clock_s": prepared.l0b_wall_clock_s,
            "l3_wall_clock_s": fallback_outcome.l3_wall_clock_s,
            "l4_wall_clock_s": fallback_outcome.l4_wall_clock_s,
        },
    )
    callback_error, callback_wall_clock_s = notify_ready(
        on_fallback_ready,
        fallback_candidate,
        clock=prepared.clock,
    )
    l1_result, l1_wall_clock_s = _run_l1_until_deadline(
        extractor,
        bundle,
        model_view,
        _prepared_source_log(prepared),
        deadline_monotonic,
        prepared.analysis_started,
        evidence_tools_factory,
        prepared.clock,
        executor_factory,
    )
    l1_output_health = output_health(l1_result)
    if l1_output_health["usable"]:
        l2_started = prepared.clock.monotonic()
        l2_result = ground_and_audit_model_evidence(
            L2GroundingInput(
                bundle=bundle,
                model_view=model_view,
                l1_result=l1_result,
                source_log=_prepared_source_log(prepared),
            )
        )
        l2_wall_clock_s = round(prepared.clock.monotonic() - l2_started, 3)
    else:
        l2_result = L2Result.not_run("l1_output_unusable")
        l2_wall_clock_s = 0.0
    return _EnrichmentRun(
        l1_result=l1_result,
        l1_output_health=l1_output_health,
        l2_result=l2_result,
        l1_wall_clock_s=l1_wall_clock_s,
        l2_wall_clock_s=l2_wall_clock_s,
        fallback_candidate=fallback_candidate,
        fallback_callback_error=callback_error,
        fallback_callback_wall_clock_s=callback_wall_clock_s,
    )


def _run_l1_until_deadline(
    extractor: EvidenceExtractor,
    bundle: L0Bundle,
    model_view: L0ModelFacingView,
    source_log: LogSnapshot,
    deadline_monotonic: float,
    analysis_started: float,
    evidence_tools_factory: EvidenceToolsFactory | None,
    clock: Clock,
    executor_factory: ExecutorFactory,
) -> tuple[L1EvidenceResult, float]:
    if remaining_deadline_seconds(deadline_monotonic, clock=clock) <= 0:
        return (
            deadline_exceeded_result(model=type(extractor).__name__),
            round(clock.monotonic() - analysis_started, 3),
        )
    pool = executor_factory(max_workers=1, thread_name_prefix="restart-agent-l1")
    future = pool.submit(
        extract_timed,
        extractor,
        bundle,
        model_view,
        source_log,
        deadline_monotonic,
        evidence_tools_factory,
        clock,
    )
    try:
        try:
            l1_result, wall_clock_s, completed_monotonic = future.result(
                timeout=remaining_deadline_seconds(deadline_monotonic, clock=clock)
            )
            if completed_monotonic > deadline_monotonic:
                l1_result = deadline_exceeded_result(model=type(extractor).__name__)
            return l1_result, wall_clock_s
        except FutureTimeoutError:
            future.cancel()
            return (
                deadline_exceeded_result(model=type(extractor).__name__),
                round(clock.monotonic() - analysis_started, 3),
            )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _prepared_source_log(prepared: PreparedAnalysis) -> LogSnapshot:
    if prepared.source_log is None:
        raise AssertionError("prepared analysis is missing its source-log snapshot")
    return prepared.source_log
