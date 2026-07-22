# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel model-route coordination over one prepared analysis."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import FIRST_COMPLETED, Future, wait
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from .decision_pipeline import DecisionOutcome, build_decision_outcome
from .execution import CollectAllAnalysisRun, L0Artifacts
from .infrastructure.log_source import LocalLogSource, LogSource
from .l1 import (
    DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
    EvidenceExtractor,
    EvidenceToolsFactory,
    L1EvidenceResult,
    ModelRoute,
)
from .l2 import L2Result
from .models import (
    AnalysisExecutionContext,
    AnalysisResult,
    AttemptRecord,
    CollectAllAnalysisResult,
    DecisionCandidate,
    DecisionCandidateKind,
    DecisionEvidence,
    L0Bundle,
    ModelAnalysisResult,
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
from .single_run import SingleRunCoordinator


class PreparedRouteRunner(Protocol):
    def run_prepared(
        self,
        prepared: PreparedAnalysis,
        *,
        deadline_monotonic: float,
        route_id: str = "single",
    ) -> Any: ...


RouteRunnerFactory = Callable[[EvidenceExtractor], PreparedRouteRunner]
LogSourceFactory = Callable[[str], LogSource]
FutureWaiter = Callable[..., Any]


class MultiRouteCoordinator:
    """Coordinate independent model routes over one prepared L0 analysis."""

    def __init__(
        self,
        *,
        route_runner_factory: RouteRunnerFactory | None = None,
        log_source_factory: LogSourceFactory = LocalLogSource,
        evidence_tools_factory: EvidenceToolsFactory | None = None,
        clock: Clock = SYSTEM_CLOCK,
        executor_factory: ExecutorFactory = THREAD_EXECUTOR_FACTORY,
        future_waiter: FutureWaiter | None = None,
    ) -> None:
        self._route_runner_factory = route_runner_factory or (
            lambda extractor: SingleRunCoordinator(
                evidence_extractor=extractor,
                log_source_factory=log_source_factory,
                evidence_tools_factory=evidence_tools_factory,
                clock=clock,
                executor_factory=executor_factory,
            )
        )
        self._log_source_factory = log_source_factory
        self._clock = clock
        self._executor_factory = executor_factory
        self._future_waiter = future_waiter or wait

    def run_many(
        self,
        execution_context: AnalysisExecutionContext,
        model_routes: Sequence[ModelRoute],
        *,
        l0_bundle: L0Bundle | None = None,
        max_parallel_models: int | None = None,
        config_metadata: Mapping[str, Any] | None = None,
        on_l0_ready: Callable[[L0Artifacts], None] | None = None,
        on_fallback_ready: Callable[[DecisionCandidate], None] | None = None,
        on_route_complete: Callable[[ModelAnalysisResult, Mapping[str, Any]], None] | None = None,
        on_attempt_record_ready: Callable[[str, AttemptRecord], None] | None = None,
        timeout_seconds: float = DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
    ) -> CollectAllAnalysisRun:
        """Run independent model routes concurrently over one prepared L0 view."""

        timeout_seconds = validate_timeout_seconds(timeout_seconds)
        routes = tuple(model_routes)
        if not routes:
            raise ValueError("collect-all analysis requires at least one model route")
        route_ids = [route.route_id for route in routes]
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("collect-all model route_id values must be unique")
        worker_count = max_parallel_models or len(routes)
        if worker_count < 1:
            raise ValueError("max_parallel_models must be at least 1")
        worker_count = min(worker_count, len(routes))

        prepared = prepare_analysis(
            execution_context,
            l0_bundle=l0_bundle,
            include_model_view=True,
            log_source_factory=self._log_source_factory,
            clock=self._clock,
        )
        deadline_monotonic = prepared.analysis_started + timeout_seconds
        if prepared.unavailable_reason is not None:
            return _log_unavailable_collect_all_run(
                prepared=prepared,
                routes=routes,
                worker_count=worker_count,
                config_metadata=config_metadata,
                timeout_seconds=timeout_seconds,
                on_route_complete=on_route_complete,
            )

        bundle, decision_evidence = _prepared_evidence(prepared)
        l0_callback_error, l0_callback_wall_clock_s = notify_l0_ready(
            on_l0_ready,
            l0_artifacts(prepared),
            clock=prepared.clock,
        )
        fallback = _publish_collect_all_fallback(
            prepared=prepared,
            timeout_seconds=timeout_seconds,
            on_fallback_ready=on_fallback_ready,
        )

        completed_runs, route_callback_errors = _execute_model_routes(
            routes=routes,
            prepared=prepared,
            deterministic_result=fallback.outcome.result,
            deadline_monotonic=deadline_monotonic,
            worker_count=worker_count,
            on_route_complete=on_route_complete,
            on_attempt_record_ready=on_attempt_record_ready,
            route_runner_factory=self._route_runner_factory,
            executor_factory=self._executor_factory,
            future_waiter=self._future_waiter,
        )
        route_runs = tuple(completed_runs[route.route_id] for route in routes)

        model_results = tuple(run[0] for run in route_runs)
        deadline_exceeded_route_count = sum(
            result.execution_status == "deadline_exceeded" for result in model_results
        )
        shared_analysis = _collect_all_shared_analysis(
            prepared,
            route_count=len(routes),
            max_parallel_models=worker_count,
            config_metadata=config_metadata,
            timeout_seconds=timeout_seconds,
            deadline_exceeded_route_count=deadline_exceeded_route_count,
            fallback_candidate=fallback.candidate,
            fallback_callback_error=fallback.callback_error,
            fallback_callback_wall_clock_s=fallback.callback_wall_clock_s,
            route_completion_callback_errors=route_callback_errors,
            l0_ready_callback_error=l0_callback_error,
            l0_ready_callback_wall_clock_s=l0_callback_wall_clock_s,
        )
        batch_result = CollectAllAnalysisResult(
            deterministic_result=fallback.outcome.result,
            model_results=model_results,
            shared_analysis=shared_analysis,
        )
        trace = {
            "routing_mode": "collect_all",
            "shared_analysis": shared_analysis,
            "deterministic": {
                "analysis_result": fallback.outcome.result.to_payload(),
                "analyzer_trace": fallback.trace,
            },
            "model_routes": {
                route.route_id: {
                    "route": _model_route_metadata(route),
                    "execution_status": route_result.execution_status,
                    "l1_usable": route_result.l1_usable,
                    "error": route_result.error,
                    "analysis_result": route_result.analysis_result.to_payload(),
                    "analyzer_trace": route_trace,
                }
                for route, (route_result, route_trace, _attempt_record) in zip(routes, route_runs)
            },
        }
        return CollectAllAnalysisRun(
            result=batch_result,
            trace=trace,
            bundle=bundle,
            decision_evidence=decision_evidence,
            model_view=prepared.model_view,
            fallback_candidate=fallback.candidate,
            deterministic_attempt_record=fallback.outcome.attempt_record,
            attempt_records_by_route={
                route.route_id: attempt_record
                for route, (_result, _trace, attempt_record) in zip(routes, route_runs)
                if attempt_record is not None
            },
        )


@dataclass(frozen=True)
class _FallbackPublication:
    outcome: DecisionOutcome
    candidate: DecisionCandidate
    trace: dict[str, Any]
    callback_error: str | None
    callback_wall_clock_s: float


def _prepared_evidence(prepared: PreparedAnalysis) -> tuple[L0Bundle, DecisionEvidence]:
    if prepared.bundle is None or prepared.decision_evidence is None:
        raise AssertionError("prepared analysis is missing L0 evidence")
    return prepared.bundle, prepared.decision_evidence


def _publish_collect_all_fallback(
    *,
    prepared: PreparedAnalysis,
    timeout_seconds: float,
    on_fallback_ready: Callable[[DecisionCandidate], None] | None,
) -> _FallbackPublication:
    bundle, decision_evidence = _prepared_evidence(prepared)
    pending_l1_result = L1EvidenceResult.disabled()
    pending_l1_health = {"status": "pending", "usable": False, "errors": []}
    fallback_outcome = build_decision_outcome(
        bundle=bundle,
        decision_evidence=decision_evidence,
        execution_context=prepared.execution_context,
        l1_configured=True,
        l1_result=pending_l1_result,
        l1_output_health=pending_l1_health,
        l2_result=L2Result.not_run("l1_pending"),
        candidate_kind=DecisionCandidateKind.DETERMINISTIC_FALLBACK.value,
        l1_pending=True,
        clock=prepared.clock,
    )
    candidate = DecisionCandidate(
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
        candidate,
        clock=prepared.clock,
    )
    trace = build_decision_trace(
        DecisionTraceInputs(
            execution_context=prepared.execution_context,
            result=fallback_outcome.result,
            bundle=bundle,
            decision_evidence=decision_evidence,
            primary=fallback_outcome.primary,
            l2_primary=None,
            total_wall_clock_s=round(
                prepared.clock.monotonic() - prepared.analysis_started,
                3,
            ),
            l0_wall_clock_s=prepared.l0_wall_clock_s,
            l0a_wall_clock_s=prepared.l0a_wall_clock_s,
            decision_evidence_wall_clock_s=prepared.decision_evidence_wall_clock_s,
            l0b_wall_clock_s=prepared.l0b_wall_clock_s,
            model_view=prepared.model_view,
            l1_result=pending_l1_result,
            l1_output_health=pending_l1_health,
            l1_wall_clock_s=0.0,
            l2_audit=fallback_outcome.l2_audit,
            selected_failure_facts=fallback_outcome.selected_failure_facts,
            history=fallback_outcome.history,
            retry_policy=fallback_outcome.retry_policy,
            l0_reused=prepared.l0_reused,
            l2_wall_clock_s=0.0,
            l3_wall_clock_s=fallback_outcome.l3_wall_clock_s,
            l4_wall_clock_s=fallback_outcome.l4_wall_clock_s,
            fallback_candidate=candidate,
            selected_candidate_kind=DecisionCandidateKind.DETERMINISTIC_FALLBACK.value,
            fallback_callback_error=callback_error,
            fallback_callback_wall_clock_s=callback_wall_clock_s,
            analysis_timeout_seconds=timeout_seconds,
        )
    )
    return _FallbackPublication(
        outcome=fallback_outcome,
        candidate=candidate,
        trace=trace,
        callback_error=callback_error,
        callback_wall_clock_s=callback_wall_clock_s,
    )


def _log_unavailable_collect_all_run(
    *,
    prepared: PreparedAnalysis,
    routes: tuple[ModelRoute, ...],
    worker_count: int,
    config_metadata: Mapping[str, Any] | None,
    timeout_seconds: float,
    on_route_complete: Callable[[ModelAnalysisResult, Mapping[str, Any]], None] | None,
) -> CollectAllAnalysisRun:
    reason = prepared.unavailable_reason
    if reason is None:
        raise AssertionError("log-unavailable result requires an unavailable reason")
    deterministic_result = log_unavailable_result(reason)
    deterministic_trace = build_log_unavailable_trace(
        deterministic_result,
        prepared.execution_context,
        reason,
        total_wall_clock_s=round(
            prepared.clock.monotonic() - prepared.analysis_started,
            3,
        ),
    )
    model_results = tuple(
        ModelAnalysisResult(
            route_id=route.route_id,
            model=route.model,
            endpoint=route.endpoint,
            credential_ref=route.credential_ref,
            execution_status="not_run_log_unavailable",
            l1_usable=False,
            analysis_result=deterministic_result,
        )
        for route in routes
    )
    callback_errors: dict[str, str] = {}
    for route_result in model_results:
        _notify_route_complete(
            on_route_complete,
            route_result,
            deterministic_trace,
            callback_errors,
        )
    shared_analysis = _collect_all_shared_analysis(
        prepared,
        route_count=len(routes),
        max_parallel_models=worker_count,
        config_metadata=config_metadata,
        timeout_seconds=timeout_seconds,
        route_completion_callback_errors=callback_errors,
    )
    batch_result = CollectAllAnalysisResult(
        deterministic_result=deterministic_result,
        model_results=model_results,
        shared_analysis=shared_analysis,
    )
    trace = {
        "routing_mode": "collect_all",
        "shared_analysis": shared_analysis,
        "deterministic": {
            "analysis_result": deterministic_result.to_payload(),
            "analyzer_trace": deterministic_trace,
        },
        "model_routes": {
            route.route_id: {
                "route": _model_route_metadata(route),
                "execution_status": "not_run_log_unavailable",
                "l1_usable": False,
                "analysis_result": deterministic_result.to_payload(),
                "analyzer_trace": deterministic_trace,
            }
            for route in routes
        },
    }
    return CollectAllAnalysisRun(result=batch_result, trace=trace)


def _execute_model_routes(
    *,
    routes: tuple[ModelRoute, ...],
    prepared: PreparedAnalysis,
    deterministic_result: AnalysisResult,
    deadline_monotonic: float,
    worker_count: int,
    on_route_complete: Callable[[ModelAnalysisResult, Mapping[str, Any]], None] | None,
    on_attempt_record_ready: Callable[[str, AttemptRecord], None] | None,
    route_runner_factory: RouteRunnerFactory,
    executor_factory: ExecutorFactory,
    future_waiter: FutureWaiter,
) -> tuple[
    dict[str, tuple[ModelAnalysisResult, dict[str, Any], AttemptRecord | None]],
    dict[str, str],
]:
    completed: dict[str, tuple[ModelAnalysisResult, dict[str, Any], AttemptRecord | None]] = {}
    callback_errors: dict[str, str] = {}
    if remaining_deadline_seconds(deadline_monotonic, clock=prepared.clock) <= 0:
        for route in routes:
            run = _deadline_exceeded_route_result(route, deterministic_result)
            completed[route.route_id] = run
            _notify_route_complete(on_route_complete, run[0], run[1], callback_errors)
            _notify_attempt_record_ready(
                on_attempt_record_ready, route.route_id, run[2], callback_errors
            )
        return completed, callback_errors

    pool = executor_factory(
        max_workers=worker_count,
        thread_name_prefix="restart-agent-model-route",
    )
    future_routes: dict[
        Future[tuple[ModelAnalysisResult, dict[str, Any], AttemptRecord | None]], ModelRoute
    ] = {
        pool.submit(
            _run_model_route,
            route,
            prepared,
            deterministic_result,
            deadline_monotonic,
            route_runner_factory,
        ): route
        for route in routes
    }
    pending: set[Future[tuple[ModelAnalysisResult, dict[str, Any], AttemptRecord | None]]] = set(
        future_routes
    )
    route_order = {route.route_id: index for index, route in enumerate(routes)}
    while (
        pending
        and remaining_deadline_seconds(
            deadline_monotonic,
            clock=prepared.clock,
        )
        > 0
    ):
        newly_done, pending = future_waiter(
            pending,
            timeout=remaining_deadline_seconds(
                deadline_monotonic,
                clock=prepared.clock,
            ),
            return_when=FIRST_COMPLETED,
        )
        if not newly_done:
            break
        for future in sorted(
            newly_done,
            key=lambda item: route_order[future_routes[item].route_id],
        ):
            route = future_routes[future]
            try:
                run = future.result()
            except Exception as exc:  # pragma: no cover - defensive future boundary
                run = _orchestration_error_route_result(
                    route,
                    deterministic_result,
                    exc,
                )
            completed[route.route_id] = run
            _notify_route_complete(on_route_complete, run[0], run[1], callback_errors)
            _notify_attempt_record_ready(
                on_attempt_record_ready, route.route_id, run[2], callback_errors
            )
    for future in pending:
        route = future_routes[future]
        future.cancel()
        run = _deadline_exceeded_route_result(route, deterministic_result)
        completed[route.route_id] = run
        _notify_route_complete(on_route_complete, run[0], run[1], callback_errors)
        _notify_attempt_record_ready(
            on_attempt_record_ready, route.route_id, run[2], callback_errors
        )
    pool.shutdown(wait=False, cancel_futures=True)
    return completed, callback_errors


def _run_model_route(
    route: ModelRoute,
    prepared: PreparedAnalysis,
    deterministic_result: AnalysisResult,
    deadline_monotonic: float,
    route_runner_factory: RouteRunnerFactory,
) -> tuple[ModelAnalysisResult, dict[str, Any], AttemptRecord | None]:
    analyzer = route_runner_factory(route.evidence_extractor)
    try:
        run = analyzer.run_prepared(
            prepared,
            deadline_monotonic=deadline_monotonic,
            route_id=route.route_id,
        )
    except Exception as exc:  # pragma: no cover - defensive route isolation
        error = f"{type(exc).__name__}: {exc}"
        return (
            ModelAnalysisResult(
                route_id=route.route_id,
                model=route.model,
                endpoint=route.endpoint,
                credential_ref=route.credential_ref,
                execution_status="orchestration_error",
                l1_usable=False,
                analysis_result=deterministic_result,
                error=error,
            ),
            {
                "routing_error": error,
                "route": _model_route_metadata(route),
            },
            None,
        )

    result = run.result
    trace = run.trace
    l1_layer = (trace.get("layers") or {}).get("L1") or {}
    l1_usable = bool(l1_layer.get("output_usable"))
    l1_trace = trace.get("l1") or {}
    errors = [str(error) for error in l1_trace.get("errors") or ()]
    anomalies = l1_trace.get("anomalies") or {}
    if l1_usable:
        execution_status = "completed"
    elif anomalies.get("deadline_exceeded"):
        execution_status = "deadline_exceeded"
    elif anomalies.get("provider_error"):
        execution_status = "provider_error"
    elif l1_trace.get("malformed"):
        execution_status = "malformed"
    else:
        execution_status = "degraded"
    route_analysis_result = (
        deterministic_result if execution_status == "deadline_exceeded" else result
    )
    return (
        ModelAnalysisResult(
            route_id=route.route_id,
            model=route.model or l1_trace.get("model"),
            endpoint=route.endpoint,
            credential_ref=route.credential_ref,
            execution_status=execution_status,
            l1_usable=l1_usable,
            analysis_result=route_analysis_result,
            error="; ".join(errors) if errors else None,
        ),
        dict(trace),
        run.attempt_record,
    )


def _collect_all_shared_analysis(
    prepared: PreparedAnalysis,
    *,
    route_count: int,
    max_parallel_models: int,
    config_metadata: Mapping[str, Any] | None = None,
    timeout_seconds: float | None = None,
    deadline_exceeded_route_count: int = 0,
    fallback_candidate: DecisionCandidate | None = None,
    fallback_callback_error: str | None = None,
    fallback_callback_wall_clock_s: float = 0.0,
    route_completion_callback_errors: Mapping[str, str] | None = None,
    l0_ready_callback_error: str | None = None,
    l0_ready_callback_wall_clock_s: float = 0.0,
) -> dict[str, Any]:
    shared_analysis = {
        "routing_mode": "collect_all",
        "route_count": route_count,
        "max_parallel_models": max_parallel_models,
        "batch_wall_clock_s": round(
            prepared.clock.monotonic() - prepared.analysis_started,
            3,
        ),
        "l0_wall_clock_s": prepared.l0_wall_clock_s,
        "l0a_wall_clock_s": prepared.l0a_wall_clock_s,
        "decision_evidence_wall_clock_s": prepared.decision_evidence_wall_clock_s,
        "l0b_wall_clock_s": prepared.l0b_wall_clock_s,
        "l0_reused": prepared.l0_reused,
        "l0_bundle_hash": _stable_payload_hash(prepared.bundle),
        "l0_model_view_hash": _stable_payload_hash(prepared.model_view),
        "analysis_timeout_seconds": timeout_seconds,
        "deadline_exceeded_route_count": deadline_exceeded_route_count,
        "fallback_ready_wall_clock_s": (
            fallback_candidate.ready_wall_clock_s if fallback_candidate is not None else None
        ),
        "fallback_callback_error": fallback_callback_error,
        "fallback_callback_wall_clock_s": fallback_callback_wall_clock_s,
        "route_completion_callback_errors": dict(route_completion_callback_errors or {}),
        "l0_ready_callback_error": l0_ready_callback_error,
        "l0_ready_callback_wall_clock_s": l0_ready_callback_wall_clock_s,
    }
    if config_metadata is not None:
        shared_analysis["restart_agent_config"] = deepcopy(dict(config_metadata))
    return shared_analysis


def _notify_route_complete(
    callback: Callable[[ModelAnalysisResult, Mapping[str, Any]], None] | None,
    result: ModelAnalysisResult,
    trace: Mapping[str, Any],
    errors: dict[str, str],
) -> None:
    if callback is None:
        return
    try:
        callback(result, trace)
    except Exception as exc:
        errors[result.route_id] = f"{type(exc).__name__}: {exc}"


def _notify_attempt_record_ready(
    callback: Callable[[str, AttemptRecord], None] | None,
    route_id: str,
    record: AttemptRecord | None,
    errors: dict[str, str],
) -> None:
    if callback is None or record is None:
        return
    try:
        callback(route_id, record)
    except Exception as exc:
        errors[f"attempt_record:{route_id}"] = f"{type(exc).__name__}: {exc}"


def _orchestration_error_route_result(
    route: ModelRoute,
    deterministic_result: AnalysisResult,
    exc: BaseException,
) -> tuple[ModelAnalysisResult, dict[str, Any], AttemptRecord | None]:
    error = f"{type(exc).__name__}: {exc}"
    return (
        ModelAnalysisResult(
            route_id=route.route_id,
            model=route.model,
            endpoint=route.endpoint,
            credential_ref=route.credential_ref,
            execution_status="orchestration_error",
            l1_usable=False,
            analysis_result=deterministic_result,
            error=error,
        ),
        {"routing_error": error, "route": _model_route_metadata(route)},
        None,
    )


def _deadline_exceeded_route_result(
    route: ModelRoute,
    deterministic_result: AnalysisResult,
) -> tuple[ModelAnalysisResult, dict[str, Any], AttemptRecord | None]:
    error = "analysis deadline exceeded before the model route completed"
    return (
        ModelAnalysisResult(
            route_id=route.route_id,
            model=route.model,
            endpoint=route.endpoint,
            credential_ref=route.credential_ref,
            execution_status="deadline_exceeded",
            l1_usable=False,
            analysis_result=deterministic_result,
            error=error,
        ),
        {
            "route": _model_route_metadata(route),
            "execution_status": "deadline_exceeded",
            "deadline_exceeded": True,
            "routing_error": error,
        },
        None,
    )


def _model_route_metadata(route: ModelRoute) -> dict[str, Any]:
    return {
        "route_id": route.route_id,
        "model": route.model,
        "endpoint": route.endpoint,
        "credential_ref": route.credential_ref,
    }


def _stable_payload_hash(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "to_payload"):
        payload = value.to_payload()
    elif is_dataclass(value):
        payload = asdict(value)
    else:
        payload = value
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
