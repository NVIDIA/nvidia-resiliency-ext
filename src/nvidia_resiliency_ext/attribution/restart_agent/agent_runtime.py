# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateful restart-agent orchestration around the stateless L0-L4 core."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Any, Callable, Iterator, Mapping, Sequence

from .attempt_records import (
    AttemptRecordAssembler,
    AttemptRecordControl,
    AttemptRecordStore,
    InMemoryAttemptRecordStore,
    NullAttemptRecordStore,
)
from .config import RestartAgentConfig, build_model_routes
from .execution import AnalysisRun, CollectAllAnalysisRun, L0Artifacts
from .l1 import DEFAULT_ANALYSIS_TIMEOUT_SECONDS, LlmEvidenceExtractor, ModelRoute
from .models import (
    AttemptRecord,
    DecisionCandidate,
    ModelAnalysisResult,
    PriorAttemptView,
    RestartAgentRequest,
    normalize_restart_agent_request,
)
from .pipeline import RestartAgent
from .runtime import SYSTEM_CLOCK, Clock


@dataclass
class _InvocationRecordState:
    key: tuple[str, int] | None
    generation: int | None
    prior_attempts: PriorAttemptView
    record: AttemptRecord | None = None
    open: bool = True
    initial_upserted: bool = False
    enriched_updates: int = 0
    ignored_updates: int = 0
    upsert_reason: str = "l0_not_ready"
    excluded_current_or_future_records: int = 0
    store_metrics_before: Mapping[str, int | bool] | None = None
    enriched_route_updates: list[str] = field(default_factory=list)
    record_operation_wall_clock_s: float = 0.0


@dataclass
class _InvocationGate:
    lock: RLock = field(default_factory=RLock)
    users: int = 0


class RestartAgentRuntime:
    """Own current-process history, invocation generations, and core execution."""

    def __init__(
        self,
        analyzer: RestartAgent,
        *,
        attempt_record_store: AttemptRecordStore | None = None,
        attempt_record_assembler: AttemptRecordAssembler | None = None,
        model_routes: Sequence[ModelRoute] = (),
        max_parallel_models: int | None = None,
        timeout_seconds: float = DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
        config_metadata: Mapping[str, Any] | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._analyzer = analyzer
        self._store = attempt_record_store or InMemoryAttemptRecordStore()
        self._assembler = attempt_record_assembler or AttemptRecordAssembler()
        self._model_routes = tuple(model_routes)
        self._max_parallel_models = max_parallel_models
        self._timeout_seconds = timeout_seconds
        self._config_metadata = dict(config_metadata or {})
        self._clock = clock
        self._state_lock = RLock()
        self._key_gates: dict[tuple[str, int], _InvocationGate] = {}
        self._generations: dict[tuple[str, int], int] = {}
        self.attempt_record_control = AttemptRecordControl(self._store)

    @property
    def model_routes(self) -> tuple[ModelRoute, ...]:
        return self._model_routes

    def analyze(
        self,
        request: RestartAgentRequest | Mapping[str, Any],
        **kwargs: Any,
    ) -> AnalysisRun | CollectAllAnalysisRun:
        """Use configured routes when present, otherwise run the single core path."""

        if self._model_routes:
            return self.analyze_many(request, self._model_routes, **kwargs)
        return self.analyze_one(request, **kwargs)

    def analyze_one(
        self,
        request: RestartAgentRequest | Mapping[str, Any],
        *,
        l0_bundle: Any = None,
        on_l0_ready: Callable[[L0Artifacts], None] | None = None,
        on_fallback_ready: Callable[[DecisionCandidate], None] | None = None,
        timeout_seconds: float | None = None,
    ) -> AnalysisRun:
        normalized = normalize_restart_agent_request(request)
        with self._key_guard(normalized):
            state = self._begin(normalized)

            def l0_callback(artifacts: L0Artifacts) -> None:
                self._publish_initial(state, normalized, artifacts)
                if on_l0_ready is not None:
                    on_l0_ready(artifacts)

            try:
                run = self._analyzer.run(
                    normalized,
                    l0_bundle=l0_bundle,
                    prior_attempts=state.prior_attempts,
                    on_l0_ready=l0_callback,
                    on_fallback_ready=on_fallback_ready,
                    timeout_seconds=(
                        self._timeout_seconds if timeout_seconds is None else timeout_seconds
                    ),
                )
                self._publish_single_enriched(state, run.attempt_record)
            finally:
                self._close(state)
            return replace(run, trace=self._runtime_trace(run.trace, state))

    def analyze_many(
        self,
        request: RestartAgentRequest | Mapping[str, Any],
        model_routes: Sequence[ModelRoute] | None = None,
        *,
        l0_bundle: Any = None,
        max_parallel_models: int | None = None,
        config_metadata: Mapping[str, Any] | None = None,
        on_l0_ready: Callable[[L0Artifacts], None] | None = None,
        on_fallback_ready: Callable[[DecisionCandidate], None] | None = None,
        on_route_complete: Callable[[ModelAnalysisResult, Mapping[str, Any]], None] | None = None,
        timeout_seconds: float | None = None,
    ) -> CollectAllAnalysisRun:
        normalized = normalize_restart_agent_request(request)
        routes = tuple(model_routes or self._model_routes)
        if not routes:
            raise ValueError("analyze_many requires at least one model route")
        with self._key_guard(normalized):
            state = self._begin(normalized)

            def l0_callback(artifacts: L0Artifacts) -> None:
                self._publish_initial(state, normalized, artifacts)
                if on_l0_ready is not None:
                    on_l0_ready(artifacts)

            def record_callback(route_id: str, route_record: AttemptRecord) -> None:
                self._publish_enriched(state, route_id, route_record)

            try:
                run = self._analyzer.run_many(
                    normalized,
                    routes,
                    l0_bundle=l0_bundle,
                    prior_attempts=state.prior_attempts,
                    max_parallel_models=(
                        self._max_parallel_models
                        if max_parallel_models is None
                        else max_parallel_models
                    ),
                    config_metadata=(
                        self._config_metadata if config_metadata is None else config_metadata
                    ),
                    on_l0_ready=l0_callback,
                    on_fallback_ready=on_fallback_ready,
                    on_route_complete=on_route_complete,
                    on_attempt_record_ready=record_callback,
                    timeout_seconds=(
                        self._timeout_seconds if timeout_seconds is None else timeout_seconds
                    ),
                )
            finally:
                self._close(state)
            return replace(run, trace=self._runtime_trace(run.trace, state))

    def _begin(self, request: RestartAgentRequest) -> _InvocationRecordState:
        prior_attempts = self._prior_attempts(request)
        key = _request_key(request)
        generation = None
        if key is not None:
            with self._state_lock:
                generation = self._generations.get(key, 0) + 1
                self._generations[key] = generation
        return _InvocationRecordState(
            key=key,
            generation=generation,
            prior_attempts=prior_attempts,
            upsert_reason=(
                prior_attempts.availability_reason
                if key is None or not self._store.enabled
                else "l0_not_ready"
            ),
            excluded_current_or_future_records=(
                sum(record.cycle_id >= key[1] for record in self._store.records(key[0]))
                if key is not None and self._store.enabled
                else 0
            ),
            store_metrics_before=self._store.metrics(),
        )

    def _prior_attempts(self, request: RestartAgentRequest) -> PriorAttemptView:
        if not self._store.enabled:
            return PriorAttemptView(availability_reason="history_disabled")
        if request.job_id is None:
            return PriorAttemptView(availability_reason="missing_job_id")
        if request.cycle_id is None:
            return PriorAttemptView(availability_reason="missing_cycle_id")
        return self._store.get_prior_attempts(request.job_id, request.cycle_id)

    def _publish_initial(
        self,
        state: _InvocationRecordState,
        request: RestartAgentRequest,
        artifacts: L0Artifacts,
    ) -> None:
        started = self._clock.monotonic()
        try:
            if state.key is None or request.job_id is None or request.cycle_id is None:
                return
            record = self._assembler.initial_record(
                job_id=request.job_id,
                cycle_id=request.cycle_id,
                bundle=artifacts.bundle,
                decision_evidence=artifacts.decision_evidence,
            )
            state.record = record
            if not self._store.enabled:
                state.upsert_reason = "history_disabled"
                return
            if not record.deterministic.root_fingerprint:
                state.upsert_reason = "missing_root_fingerprint"
                return
            if not self._accepts_update(state):
                state.ignored_updates += 1
                return
            self._store.upsert_attempt(record)
            state.record = record
            state.initial_upserted = True
            state.upsert_reason = "stored"
        finally:
            state.record_operation_wall_clock_s += self._clock.monotonic() - started

    def _publish_single_enriched(
        self,
        state: _InvocationRecordState,
        route_record: AttemptRecord | None,
    ) -> None:
        if route_record is None:
            return
        for entry in route_record.enriched:
            self._publish_enriched(state, entry.route_id, route_record)

    def _publish_enriched(
        self,
        state: _InvocationRecordState,
        route_id: str,
        route_record: AttemptRecord,
    ) -> None:
        started = self._clock.monotonic()
        try:
            entry = next(
                (item for item in route_record.enriched if item.route_id == route_id),
                None,
            )
            if entry is None or state.record is None or not state.initial_upserted:
                return
            if not self._accepts_update(state):
                state.ignored_updates += 1
                return
            replacement = self._assembler.with_enriched(
                state.record,
                route_id=route_id,
                facts=entry.facts,
            )
            self._store.upsert_attempt(replacement)
            state.record = replacement
            state.enriched_updates += 1
            state.enriched_route_updates.append(route_id)
        finally:
            state.record_operation_wall_clock_s += self._clock.monotonic() - started

    def _accepts_update(self, state: _InvocationRecordState) -> bool:
        if not state.open or state.key is None or state.generation is None:
            return False
        with self._state_lock:
            return self._generations.get(state.key) == state.generation

    def _close(self, state: _InvocationRecordState) -> None:
        state.open = False

    @contextmanager
    def _key_guard(self, request: RestartAgentRequest) -> Iterator[None]:
        key = _request_key(request)
        if key is None:
            yield
            return
        with self._state_lock:
            gate = self._key_gates.setdefault(key, _InvocationGate())
            gate.users += 1
        try:
            with gate.lock:
                yield
        finally:
            with self._state_lock:
                gate.users -= 1
                if gate.users == 0 and self._key_gates.get(key) is gate:
                    self._key_gates.pop(key, None)
                    self._generations.pop(key, None)

    def _runtime_trace(
        self,
        trace: Mapping[str, Any],
        state: _InvocationRecordState,
    ) -> dict[str, Any]:
        payload = dict(trace)
        payload["runtime_history"] = {
            "enabled": self._store.enabled,
            "availability_reason": state.prior_attempts.availability_reason,
            "prior_attempt_count": len(state.prior_attempts.records),
            "excluded_current_or_future_record_count": (state.excluded_current_or_future_records),
            "record_key": (
                {"job_id": state.key[0], "cycle_id": state.key[1]}
                if state.key is not None
                else None
            ),
            "generation": state.generation,
            "initial_upserted": state.initial_upserted,
            "upsert_reason": state.upsert_reason,
            "enriched_updates": state.enriched_updates,
            "enriched_route_updates": tuple(state.enriched_route_updates),
            "ignored_closed_or_stale_updates": state.ignored_updates,
            "closed": not state.open,
            "record_operation_wall_clock_s": round(
                state.record_operation_wall_clock_s,
                6,
            ),
            "current_attempt_record": (
                state.record.to_payload() if state.record is not None else None
            ),
            "store_before": dict(state.store_metrics_before or {}),
            "store_after": self._store.metrics(),
        }
        return payload


def build_restart_agent_runtime(
    config: RestartAgentConfig,
    *,
    attempt_record_store: AttemptRecordStore | None = None,
) -> RestartAgentRuntime:
    """Composition root for validated config, provider routes, and runtime state."""

    store = attempt_record_store
    if store is None:
        store = (
            InMemoryAttemptRecordStore(
                max_attempts_per_job=config.history.max_attempts_per_job,
                max_total_records=config.history.max_total_records,
            )
            if config.history.enabled
            else NullAttemptRecordStore()
        )
    analyzer = RestartAgent(
        restart_environment_context=config.restart_environment_context,
        retry_policy=config.retry_policy,
    )
    return RestartAgentRuntime(
        analyzer,
        attempt_record_store=store,
        model_routes=build_model_routes(config, LlmEvidenceExtractor),
        max_parallel_models=config.max_parallel_models,
        timeout_seconds=config.timeout_seconds,
        config_metadata=config.metadata(),
    )


def _request_key(request: RestartAgentRequest) -> tuple[str, int] | None:
    if request.job_id is None or request.cycle_id is None:
        return None
    return request.job_id, request.cycle_id
