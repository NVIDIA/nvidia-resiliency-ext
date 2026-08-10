# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process attrsvc backend for the Restart Agent runtime."""

from __future__ import annotations

import asyncio
import os
import stat
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from queue import SimpleQueue
from threading import Event, RLock, Thread
from typing import Any, Callable, Hashable, Mapping

from nvidia_resiliency_ext.attribution.coalescing import (
    CacheResult,
    InflightResult,
    SubmittedResult,
)
from nvidia_resiliency_ext.attribution.orchestration.config import ErrorCode
from nvidia_resiliency_ext.attribution.orchestration.log_path_metadata import (
    CYCLE_NUM_PATTERN,
    extract_job_metadata,
)
from nvidia_resiliency_ext.attribution.orchestration.progressive import (
    ANALYSIS_INTENT_PROGRESSIVE,
    ANALYSIS_INTENT_TERMINAL,
    normalize_analysis_intent,
)
from nvidia_resiliency_ext.attribution.orchestration.types import (
    RECOMMENDATION_RESTART,
    RECOMMENDATION_STOP,
    RECOMMENDATION_UNKNOWN,
    LogAnalysisCycleResult,
    LogAnalyzerError,
    LogAnalyzerFilePreview,
    LogAnalyzerSubmitResult,
)
from nvidia_resiliency_ext.attribution.orchestration.utils import validate_log_path
from nvidia_resiliency_ext.attribution.path_utils import path_is_under_allowed_root
from nvidia_resiliency_ext.attribution.restart_agent import (
    AnalysisResult,
    DecisionCandidate,
    ModelAnalysisResult,
    ProgressiveL0Accumulator,
    ProgressiveSourceUnavailable,
    RestartAgentConfig,
    RestartAgentRequest,
    RestartAgentRuntime,
)

_STATUS_REGISTERED = "registered"
_STATUS_ANALYZING = "analyzing"
_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"
_PUBLIC_PENDING = "pending"
_PUBLIC_IN_FLIGHT = "in_flight"
_ELIGIBLE_NVRX_USES = frozenset({"eligible", "eligible_degraded"})


@dataclass(frozen=True)
class LogConvergencePolicy:
    """Bounded wait for log-funnel writes visible after terminal notification."""

    minimum_wait_seconds: float = 10.0
    quiet_seconds: float = 5.0
    max_wait_seconds: float = 40.0
    poll_seconds: float = 0.25


@dataclass(frozen=True)
class _DrainOutcome:
    converged: bool
    max_wait_expired: bool
    wall_clock_s: float = 0.0
    poll_count: int = 0
    growth_count: int = 0
    completion_reason: str = "unknown"
    minimum_wait_seconds: float = 0.0
    quiet_seconds: float = 0.0
    max_wait_seconds: float = 0.0

    @property
    def include_incomplete_tail(self) -> bool:
        return self.converged or not self.max_wait_expired

    def to_payload(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "max_wait_expired": self.max_wait_expired,
            "wall_clock_s": round(self.wall_clock_s, 6),
            "poll_count": self.poll_count,
            "growth_count": self.growth_count,
            "completion_reason": self.completion_reason,
            "minimum_wait_seconds": self.minimum_wait_seconds,
            "quiet_seconds": self.quiet_seconds,
            "max_wait_seconds": self.max_wait_seconds,
        }


@dataclass(frozen=True)
class ProgressiveAnalysisPolicy:
    """Service-owned scheduling and bounded-state policy."""

    enabled: bool = False
    pre_end_poll_seconds: float = 180.0
    active_idle_seconds: float = 900.0
    max_active_states: int = 64
    max_completed_results: int = 3000

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be boolean")
        for name in ("pre_end_poll_seconds", "active_idle_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        for name in ("max_active_states", "max_completed_results"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be greater than zero")
        if self.active_idle_seconds < self.pre_end_poll_seconds:
            raise ValueError("active_idle_seconds must be at least pre_end_poll_seconds")


@dataclass
class _AttemptExecution:
    key: Hashable
    log_path: str
    user: str
    job_id: str | None
    cycle_id: int | None
    status: str = _STATUS_REGISTERED
    registered_at: float = field(default_factory=time.monotonic)
    terminal_at: float | None = None
    completed_at: float | None = None
    future: Future[None] | None = None
    best_result: AnalysisResult | None = None
    best_source: str = ""
    candidate_result: AnalysisResult | None = None
    candidate_source: str = ""
    final_result: AnalysisResult | None = None
    error: str | None = None
    deterministic_ready_count: int = 0
    route_complete_count: int = 0
    terminal_drain_wall_clock_s: float | None = None
    terminal_l0a_ready_wall_clock_s: float | None = None
    deterministic_ready_wall_clock_s: float | None = None
    first_route_ready_wall_clock_s: float | None = None
    analysis_completed_wall_clock_s: float | None = None
    accumulator: ProgressiveL0Accumulator | None = None
    precompute_future: Future[bool] | None = None
    next_poll_at: float | None = None
    progressive_phase: str = "not_started"
    progressive_error: str | None = None
    progressive_evicted: bool = False


class RestartAgentServiceBackend:
    """Adapt attrsvc's progressive HTTP lifecycle to one RestartAgentRuntime."""

    DEFAULT_MAX_CONCURRENT_ATTEMPTS = 4

    def __init__(
        self,
        *,
        allowed_root: str,
        runtime: RestartAgentRuntime,
        config: RestartAgentConfig,
        convergence: LogConvergencePolicy = LogConvergencePolicy(),
        progressive: ProgressiveAnalysisPolicy = ProgressiveAnalysisPolicy(),
        executor: ThreadPoolExecutor | None = None,
        precompute_executor: ThreadPoolExecutor | None = None,
        accumulator_factory: Callable[[str], ProgressiveL0Accumulator] = (ProgressiveL0Accumulator),
    ) -> None:
        self._allowed_root = os.path.realpath(allowed_root)
        self._runtime = runtime
        self._config = config
        self._convergence = convergence
        self._progressive = progressive
        self._accumulator_factory = accumulator_factory
        self._max_completed_results = progressive.max_completed_results
        self._lock = RLock()
        self._entries: dict[Hashable, _AttemptExecution] = {}
        self._path_index: dict[str, Hashable] = {}
        self._executor = executor or ThreadPoolExecutor(
            max_workers=self.DEFAULT_MAX_CONCURRENT_ATTEMPTS,
            thread_name_prefix="nvrx-restart-agent",
        )
        self._owns_executor = executor is None
        self._precompute_executor = precompute_executor or ThreadPoolExecutor(
            max_workers=min(2, progressive.max_active_states),
            thread_name_prefix="nvrx-restart-agent-l0",
        )
        self._owns_precompute_executor = precompute_executor is None
        self._scheduler_wakeup = Event()
        self._scheduler = Thread(
            target=self._schedule_progressive,
            name="nvrx-restart-agent-progressive",
            daemon=True,
        )
        self._shutdown = False
        self._eviction_count = 0
        self._execution_errors = 0
        self._progressive_eviction_count = 0
        self._progressive_precompute_errors = 0
        self._scheduler.start()

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
        self._scheduler_wakeup.set()
        self._scheduler.join(timeout=1.0)
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
        if self._owns_precompute_executor:
            self._precompute_executor.shutdown(wait=False, cancel_futures=True)
        close_runtime = getattr(self._runtime, "close", None)
        if close_runtime is not None:
            close_runtime()

    async def shutdown_async(self) -> None:
        self.shutdown()

    async def start(self, loop: asyncio.AbstractEventLoop | None = None) -> dict[str, Any]:
        del loop
        return {"cache_entries_loaded": 0, "backend": "lib"}

    async def save_cache(self, cache_file: str | None = None) -> bool:
        del cache_file
        return False

    async def load_cache(self, cache_file: str | None = None) -> int:
        del cache_file
        return 0

    async def check_mcp_health(self, timeout_seconds: float = 5.0) -> tuple[str, str]:
        del timeout_seconds
        return "unused", "Restart Agent runs directly in attrsvc"

    def validate_path(
        self,
        user_path: str,
        *,
        require_regular_file: bool = True,
        reject_empty: bool = False,
    ) -> str | LogAnalyzerError:
        if os.path.exists(user_path):
            return validate_log_path(
                user_path,
                self._allowed_root,
                require_regular_file=require_regular_file,
                reject_empty=reject_empty,
            )
        return self._validate_expected_path(user_path)

    async def submit_log(
        self,
        log_path: str,
        user: str = "unknown",
        job_id: str | None = None,
        cycle_id: int | None = None,
        analysis_intent: str | None = None,
    ) -> LogAnalyzerSubmitResult | LogAnalyzerError:
        try:
            intent = normalize_analysis_intent(analysis_intent)
        except ValueError as exc:
            return LogAnalyzerError(ErrorCode.INVALID_PARAMETER, str(exc))

        normalized = self.validate_path(log_path, require_regular_file=False, reject_empty=False)
        if isinstance(normalized, LogAnalyzerError):
            return normalized
        if cycle_id is not None and (isinstance(cycle_id, bool) or not isinstance(cycle_id, int)):
            return LogAnalyzerError(ErrorCode.INVALID_PARAMETER, "cycle_id must be an integer")
        identity = self._identity(normalized, job_id, cycle_id)
        with self._lock:
            if self._shutdown:
                return LogAnalyzerError(
                    ErrorCode.INTERNAL_ERROR, "Restart Agent backend is shut down"
                )
            entry_or_error = self._register(
                normalized,
                user,
                *identity,
                explicit_job_id=job_id,
                explicit_cycle_id=cycle_id,
            )
            if isinstance(entry_or_error, LogAnalyzerError):
                return entry_or_error
            entry = entry_or_error
            if intent == ANALYSIS_INTENT_TERMINAL:
                self._start_terminal(entry)
            elif intent == ANALYSIS_INTENT_PROGRESSIVE:
                self._start_progressive(entry)
            else:
                # track_only has the same registration semantics in the direct backend.
                pass

        return LogAnalyzerSubmitResult(submitted=True, normalized_path=normalized)

    async def analyze_log(
        self,
        log_path: str,
        file: str | None = None,
        wl_restart: int | None = None,
        wait: bool = True,
    ) -> LogAnalysisCycleResult | LogAnalyzerError:
        if file is not None or wl_restart is not None:
            return LogAnalyzerError(
                ErrorCode.INVALID_PARAMETER,
                "file and wl_restart are not supported by the Restart Agent backend",
            )
        normalized = self._normalize_lookup_path(log_path)
        if isinstance(normalized, LogAnalyzerError):
            return normalized
        with self._lock:
            key = self._path_index.get(normalized)
            entry = self._entries.get(key) if key is not None else None
            future = entry.future if entry is not None else None
        if entry is None:
            return LogAnalyzerError(ErrorCode.NOT_FOUND, "attempt has not been submitted")
        if wait and future is not None and not future.done():
            try:
                await asyncio.shield(asyncio.wrap_future(future))
            except asyncio.CancelledError:
                raise
            except Exception:
                # _execute_terminal records the service-visible failure state.
                pass
        return self._public_result(entry)

    def read_file_preview(
        self, log_path: str, max_bytes: int = 4096
    ) -> LogAnalyzerFilePreview | LogAnalyzerError:
        validated = validate_log_path(
            log_path,
            self._allowed_root,
            require_regular_file=True,
            reject_empty=False,
        )
        if isinstance(validated, LogAnalyzerError):
            return validated
        try:
            with open(validated, "r", encoding="utf-8", errors="ignore") as handle:
                return LogAnalyzerFilePreview(content=handle.read(max_bytes), path=validated)
        except OSError as exc:
            return LogAnalyzerError(ErrorCode.INTERNAL_ERROR, f"file read error: {exc}")

    async def get_stats(self) -> dict[str, Any]:
        with self._lock:
            counts = self._status_counts()
            deterministic_count = sum(
                entry.deterministic_ready_count for entry in self._entries.values()
            )
            route_count = sum(entry.route_complete_count for entry in self._entries.values())
            execution_errors = self._execution_errors
            eviction_count = self._eviction_count
            progressive_evictions = self._progressive_eviction_count
            progressive_errors = self._progressive_precompute_errors
            progressive_states = self._progressive_state_counts()
            terminal_entries = tuple(
                entry for entry in self._entries.values() if entry.terminal_at is not None
            )
            accumulators = tuple(
                entry.accumulator
                for entry in self._entries.values()
                if entry.accumulator is not None
            )
        accumulator_states = [accumulator.state() for accumulator in accumulators]
        history_records = self._runtime.attempt_record_control.records()
        return {
            "backend": "lib",
            "restart_agent": {
                "config": self._config.metadata(),
                "attempts": counts,
                "deterministic_ready": deterministic_count,
                "route_complete": route_count,
                "execution_errors": execution_errors,
                "registry_evictions": eviction_count,
                "history_record_count": len(history_records),
                "terminal_timing": self._latest_terminal_timing(terminal_entries),
                "progressive": {
                    "policy": {
                        "pre_end_poll_seconds": self._progressive.pre_end_poll_seconds,
                        "enabled": self._progressive.enabled,
                        "active_idle_seconds": self._progressive.active_idle_seconds,
                        "max_active_states": self._progressive.max_active_states,
                        "max_completed_results": self._progressive.max_completed_results,
                    },
                    "states": progressive_states,
                    "active_state_count": len(accumulator_states),
                    "state_evictions": progressive_evictions,
                    "precompute_errors": progressive_errors,
                    "poll_count": sum(state.poll_count for state in accumulator_states),
                    "growth_count": sum(state.growth_count for state in accumulator_states),
                    "l0a_build_count": sum(state.l0a_build_count for state in accumulator_states),
                    "bytes_ingested": sum(state.bytes_ingested for state in accumulator_states),
                    "bytes_reread": sum(state.bytes_reread for state in accumulator_states),
                },
            },
        }

    async def get_cache(self) -> CacheResult:
        return {"count": 0, "entries": []}

    async def get_inflight(self) -> InflightResult:
        with self._lock:
            paths = [
                entry.log_path
                for entry in self._entries.values()
                if entry.status == _STATUS_ANALYZING
            ]
        return {"count": len(paths), "paths": sorted(paths)}

    async def get_submitted(self) -> SubmittedResult:
        now = time.monotonic()
        with self._lock:
            entries = [
                {
                    "path": entry.log_path,
                    "age_seconds": max(0.0, now - entry.registered_at),
                    "status": entry.status,
                }
                for entry in self._entries.values()
            ]
        return {"count": len(entries), "entries": sorted(entries, key=lambda item: item["path"])}

    def get_all_jobs(self) -> dict[str, Any]:
        jobs: dict[str, list[dict[str, Any]]] = {}
        with self._lock:
            for entry in self._entries.values():
                jobs.setdefault(entry.job_id or "unknown", []).append(
                    {
                        "cycle_id": entry.cycle_id,
                        "log_path": entry.log_path,
                        "status": entry.status,
                        "progressive_phase": entry.progressive_phase,
                    }
                )
        return {
            job_id: sorted(
                records, key=lambda item: (-1 if item["cycle_id"] is None else item["cycle_id"])
            )
            for job_id, records in sorted(jobs.items())
        }

    async def status(self) -> dict[str, Any]:
        with self._lock:
            counts = self._status_counts()
            shutdown = self._shutdown
        return {
            "status": "fail" if shutdown else "ok",
            "backend": "lib",
            "issues": (["Restart Agent backend is shut down"] if shutdown else []),
            "restart_agent": {
                "config_id": self._config.config_id,
                "config_version": self._config.config_version,
                "config_fingerprint": self._config.config_fingerprint,
                "attempts": counts,
            },
        }

    def _register(
        self,
        normalized_path: str,
        user: str,
        job_id: str | None,
        cycle_id: int | None,
        *,
        explicit_job_id: str | None,
        explicit_cycle_id: int | None,
    ) -> _AttemptExecution | LogAnalyzerError:
        path_key = self._path_index.get(normalized_path)
        if path_key is not None:
            existing = self._entries.get(path_key)
            if existing is None:
                self._path_index.pop(normalized_path, None)
            else:
                if (explicit_job_id or "").strip() and existing.job_id != job_id:
                    return LogAnalyzerError(
                        ErrorCode.INVALID_PARAMETER,
                        "log path is already registered with different job_id",
                    )
                if explicit_cycle_id is not None and existing.cycle_id != cycle_id:
                    return LogAnalyzerError(
                        ErrorCode.INVALID_PARAMETER,
                        "log path is already registered with different cycle_id",
                    )
                if user and existing.user == "unknown":
                    existing.user = user
                return existing

        key: Hashable = (
            (job_id, cycle_id) if job_id is not None and cycle_id is not None else normalized_path
        )
        existing = self._entries.get(key)
        if existing is not None:
            if existing.log_path != normalized_path:
                return LogAnalyzerError(
                    ErrorCode.INVALID_PARAMETER,
                    "job_id and cycle_id already identify a different log path",
                )
            if user and existing.user == "unknown":
                existing.user = user
            return existing
        if not self._make_room():
            return LogAnalyzerError(
                ErrorCode.JOB_LIMIT_REACHED,
                "Restart Agent attempt registry has no evictable entry",
            )
        entry = _AttemptExecution(
            key=key,
            log_path=normalized_path,
            user=user or "unknown",
            job_id=job_id,
            cycle_id=cycle_id,
        )
        self._entries[key] = entry
        self._path_index[normalized_path] = key
        return entry

    def _start_progressive(self, entry: _AttemptExecution) -> None:
        if entry.status in {_STATUS_ANALYZING, _STATUS_COMPLETED}:
            return
        if not self._progressive.enabled:
            entry.progressive_phase = "disabled"
            return
        self._ensure_accumulator(entry)
        if entry.accumulator is not None:
            entry.progressive_phase = "scheduled"
            entry.next_poll_at = time.monotonic()
            self._scheduler_wakeup.set()

    def _start_terminal(self, entry: _AttemptExecution) -> None:
        if entry.status in {_STATUS_ANALYZING, _STATUS_COMPLETED}:
            return
        self._ensure_accumulator(entry, allow_eviction=False)
        entry.status = _STATUS_ANALYZING
        entry.progressive_phase = "finalizing"
        entry.terminal_at = time.monotonic()
        entry.error = None
        entry.next_poll_at = None
        self._scheduler_wakeup.set()
        entry.future = self._executor.submit(self._execute_terminal, entry.key)

    def _execute_terminal(self, key: Hashable) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            request = RestartAgentRequest(
                log_path=entry.log_path,
                job_id=entry.job_id,
                cycle_id=entry.cycle_id,
            )
            accumulator = entry.accumulator
        try:
            drain = self._wait_for_log_convergence(request.log_path, accumulator)

            def deterministic_ready(candidate: DecisionCandidate) -> None:
                with self._lock:
                    current = self._entries.get(key)
                    if current is None:
                        return
                    current.best_result = candidate.result
                    current.best_source = candidate.candidate_kind
                    current.candidate_result = candidate.result
                    current.candidate_source = candidate.candidate_kind
                    current.deterministic_ready_count += 1
                    if current.terminal_at is not None:
                        current.deterministic_ready_wall_clock_s = (
                            time.monotonic() - current.terminal_at
                        )

            def route_complete(result: ModelAnalysisResult, trace: Mapping[str, Any]) -> None:
                del trace
                with self._lock:
                    current = self._entries.get(key)
                    if current is None:
                        return
                    current.best_result = result.analysis_result
                    current.best_source = (
                        f"l1_enriched:{result.route_id}" if result.l1_usable else "deterministic"
                    )
                    current.candidate_result = result.analysis_result
                    current.candidate_source = current.best_source
                    current.route_complete_count += 1
                    if (
                        current.first_route_ready_wall_clock_s is None
                        and current.terminal_at is not None
                    ):
                        current.first_route_ready_wall_clock_s = (
                            time.monotonic() - current.terminal_at
                        )

            finalized_l0a = None
            if accumulator is not None:
                try:
                    finalized_l0a = accumulator.finalize(
                        include_incomplete_tail=drain.include_incomplete_tail,
                    )
                    terminal_to_l0a_ready_s = (
                        time.monotonic() - entry.terminal_at
                        if entry.terminal_at is not None
                        else None
                    )
                    finalized_l0a = replace(
                        finalized_l0a,
                        progressive_metrics={
                            **dict(finalized_l0a.progressive_metrics),
                            "terminal_drain": drain.to_payload(),
                            "terminal_to_l0a_ready_wall_clock_s": (
                                round(terminal_to_l0a_ready_s, 6)
                                if terminal_to_l0a_ready_s is not None
                                else None
                            ),
                        },
                    )
                    with self._lock:
                        current = self._entries.get(key)
                        if current is not None:
                            current.terminal_drain_wall_clock_s = drain.wall_clock_s
                            current.terminal_l0a_ready_wall_clock_s = terminal_to_l0a_ready_s
                except ProgressiveSourceUnavailable:
                    finalized_l0a = None
                finally:
                    with self._lock:
                        current = self._entries.get(key)
                        if current is not None and current.accumulator is accumulator:
                            current.accumulator = None
                            current.precompute_future = None
                            current.next_poll_at = None
                    accumulator = None
            if finalized_l0a is not None and hasattr(self._runtime, "analyze_prepared"):
                run = self._runtime.analyze_prepared(
                    request,
                    finalized_l0a,
                    on_deterministic_ready=deterministic_ready,
                    on_route_complete=route_complete,
                    retain_detailed_artifacts=False,
                )
            else:
                run = self._runtime.analyze(
                    request,
                    on_deterministic_ready=deterministic_ready,
                    on_route_complete=route_complete,
                    retain_detailed_artifacts=False,
                )
            model_results = tuple(getattr(run.result, "model_results", ()))
            final = model_results[0].analysis_result if model_results else run.result
            if not isinstance(final, AnalysisResult):
                raise TypeError("Restart Agent service run did not produce AnalysisResult")
            with self._lock:
                current = self._entries.get(key)
                if current is not None:
                    current.final_result = final
                    current.best_result = final
                    if not current.best_source:
                        current.best_source = "deterministic"
                    current.status = _STATUS_COMPLETED
                    current.progressive_phase = "completed"
                    current.completed_at = time.monotonic()
                    if current.terminal_at is not None:
                        current.analysis_completed_wall_clock_s = (
                            current.completed_at - current.terminal_at
                        )
                    current.future = None
                    self._trim_completed_results()
        except Exception as exc:
            with self._lock:
                self._execution_errors += 1
                current = self._entries.get(key)
                if current is not None:
                    current.error = f"{type(exc).__name__}: {exc}"
                    current.completed_at = time.monotonic()
                    current.status = (
                        _STATUS_COMPLETED if current.best_result is not None else _STATUS_FAILED
                    )
                    current.progressive_phase = current.status
                    current.accumulator = None
                    current.precompute_future = None
                    current.next_poll_at = None
                    current.future = None
                    self._trim_completed_results()

    def _public_result(self, entry: _AttemptExecution) -> LogAnalysisCycleResult:
        with self._lock:
            result = entry.final_result or entry.best_result
            candidate_result = entry.candidate_result
            status = entry.status
            source = entry.best_source
            candidate_source = entry.candidate_source
            error = entry.error
            cycle_id = entry.cycle_id
        public_status = {
            _STATUS_REGISTERED: _PUBLIC_PENDING,
            _STATUS_ANALYZING: _PUBLIC_IN_FLIGHT,
            _STATUS_COMPLETED: _STATUS_COMPLETED,
            _STATUS_FAILED: _STATUS_FAILED,
        }[status]
        payload = result.to_payload() if result is not None else ({"error": error} if error else {})
        return LogAnalysisCycleResult(
            result=payload,
            status=public_status,
            wl_restart=cycle_id or 0,
            recommendation=self._recommendation(result, source),
            candidate_recommendation=self._recommendation(candidate_result, candidate_source),
        )

    @staticmethod
    def _recommendation(result: AnalysisResult | None, source: str) -> dict[str, str]:
        if result is None:
            return {"action": RECOMMENDATION_UNKNOWN, "reason": "", "source": source}
        nvrx_use = str(result.result_provenance.get("nvrx_use") or "")
        action = result.decision if nvrx_use in _ELIGIBLE_NVRX_USES else RECOMMENDATION_UNKNOWN
        if action not in {RECOMMENDATION_STOP, RECOMMENDATION_RESTART}:
            action = RECOMMENDATION_UNKNOWN
        return {
            "action": action,
            "reason": result.justification,
            "source": source or "restart_agent",
        }

    def _identity(
        self,
        path: str,
        explicit_job_id: str | None,
        explicit_cycle_id: int | None,
    ) -> tuple[str | None, int | None]:
        metadata = extract_job_metadata(path, warn_on_missing_job_id=False)
        job_id = (explicit_job_id or "").strip() or metadata.job_id or None
        if explicit_cycle_id is not None:
            cycle_id = explicit_cycle_id
        else:
            match = CYCLE_NUM_PATTERN.search(path)
            cycle_id = int(match.group(1)) if match else None
        return job_id, cycle_id

    def _validate_expected_path(self, user_path: str) -> str | LogAnalyzerError:
        if not os.path.isabs(user_path):
            return LogAnalyzerError(ErrorCode.INVALID_PATH, "path must be absolute")
        real = os.path.realpath(user_path)
        if not path_is_under_allowed_root(real, self._allowed_root):
            return LogAnalyzerError(
                ErrorCode.OUTSIDE_ROOT,
                "access outside allowed root is not permitted",
            )
        parent = os.path.dirname(real)
        try:
            parent_stat = os.stat(parent)
        except FileNotFoundError:
            return LogAnalyzerError(ErrorCode.NOT_FOUND, "path parent not found")
        except PermissionError:
            return LogAnalyzerError(ErrorCode.NOT_READABLE, "path parent is not readable")
        if not stat.S_ISDIR(parent_stat.st_mode):
            return LogAnalyzerError(ErrorCode.INVALID_PATH, "path parent is not a directory")
        if not os.access(parent, os.R_OK | os.X_OK):
            return LogAnalyzerError(ErrorCode.NOT_READABLE, "path parent is not accessible")
        return real

    def _normalize_lookup_path(self, path: str) -> str | LogAnalyzerError:
        if not os.path.isabs(path):
            return LogAnalyzerError(ErrorCode.INVALID_PATH, "path must be absolute")
        normalized = os.path.realpath(path)
        if not path_is_under_allowed_root(normalized, self._allowed_root):
            return LogAnalyzerError(
                ErrorCode.OUTSIDE_ROOT,
                "access outside allowed root is not permitted",
            )
        return normalized

    def _wait_for_log_convergence(
        self,
        path: str,
        accumulator: ProgressiveL0Accumulator | None,
    ) -> _DrainOutcome:
        policy = self._convergence
        if policy.max_wait_seconds <= 0:
            return _DrainOutcome(
                converged=True,
                max_wait_expired=False,
                completion_reason="disabled",
                minimum_wait_seconds=policy.minimum_wait_seconds,
                quiet_seconds=policy.quiet_seconds,
                max_wait_seconds=policy.max_wait_seconds,
            )
        reader_ready = Event()
        stop_observer = Event()
        notifications: SimpleQueue[str] = SimpleQueue()
        outcomes: list[_DrainOutcome] = []

        def observe() -> None:
            try:
                outcome = self._observe_log_convergence(
                    path,
                    reader_ready=reader_ready,
                    notifications=notifications,
                    stop_observer=stop_observer,
                )
                if outcome is not None:
                    outcomes.append(outcome)
            finally:
                notifications.put("done")

        observer = Thread(
            target=observe,
            name="nvrx-restart-agent-log-drain",
            daemon=True,
        )
        observer.start()
        try:
            if accumulator is not None:
                accumulator.refresh(precompute=False)
        except Exception:
            stop_observer.set()
            reader_ready.set()
            observer.join()
            raise
        reader_ready.set()
        try:
            while True:
                notification = notifications.get()
                if notification == "growth":
                    if accumulator is not None:
                        accumulator.refresh(precompute=False)
                    continue
                if notification == "quiet_candidate":
                    if accumulator is not None:
                        accumulator.refresh(precompute=True)
                    continue
                if notification == "done":
                    break
            observer.join()
            if accumulator is not None:
                accumulator.refresh(precompute=False)
        except Exception:
            stop_observer.set()
            reader_ready.set()
            observer.join()
            raise
        if not outcomes:
            raise RuntimeError("terminal log-drain observer stopped without an outcome")
        return outcomes[0]

    def _observe_log_convergence(
        self,
        path: str,
        *,
        reader_ready: Event,
        notifications: SimpleQueue[str],
        stop_observer: Event,
    ) -> _DrainOutcome | None:
        policy = self._convergence
        started = time.monotonic()
        quiet_eligible_at = started + policy.minimum_wait_seconds
        unchanged_since: float | None = None
        previous: tuple[int, int] | None = None
        precompute_notified_for: tuple[int, int] | None = None
        poll_count = 0
        growth_count = 0
        while True:
            now = time.monotonic()
            poll_count += 1
            try:
                current_stat = os.stat(path)
                current = (current_stat.st_size, current_stat.st_mtime_ns)
            except OSError:
                current = None
            if current is not None and current == previous:
                if unchanged_since is None:
                    unchanged_since = now
                if reader_ready.is_set() and precompute_notified_for != current:
                    notifications.put("quiet_candidate")
                    precompute_notified_for = current
            else:
                if previous is not None and current is not None:
                    growth_count += 1
                    notifications.put("growth")
                previous = current
                precompute_notified_for = None
                unchanged_since = now if current is not None else None
            if (
                current is not None
                and unchanged_since is not None
                and now >= quiet_eligible_at
                and now - max(unchanged_since, quiet_eligible_at) >= policy.quiet_seconds
                and reader_ready.is_set()
            ):
                return _DrainOutcome(
                    converged=True,
                    max_wait_expired=False,
                    wall_clock_s=now - started,
                    poll_count=poll_count,
                    growth_count=growth_count,
                    completion_reason="quiet_after_minimum_wait",
                    minimum_wait_seconds=policy.minimum_wait_seconds,
                    quiet_seconds=policy.quiet_seconds,
                    max_wait_seconds=policy.max_wait_seconds,
                )
            if now - started >= policy.max_wait_seconds:
                return _DrainOutcome(
                    converged=False,
                    max_wait_expired=True,
                    wall_clock_s=now - started,
                    poll_count=poll_count,
                    growth_count=growth_count,
                    completion_reason="max_wait_expired",
                    minimum_wait_seconds=policy.minimum_wait_seconds,
                    quiet_seconds=policy.quiet_seconds,
                    max_wait_seconds=policy.max_wait_seconds,
                )
            remaining = policy.max_wait_seconds - (time.monotonic() - started)
            if remaining <= 0:
                return _DrainOutcome(
                    converged=False,
                    max_wait_expired=True,
                    wall_clock_s=time.monotonic() - started,
                    poll_count=poll_count,
                    growth_count=growth_count,
                    completion_reason="max_wait_expired",
                    minimum_wait_seconds=policy.minimum_wait_seconds,
                    quiet_seconds=policy.quiet_seconds,
                    max_wait_seconds=policy.max_wait_seconds,
                )
            if stop_observer.wait(min(max(policy.poll_seconds, 0.01), remaining)):
                return None

    @staticmethod
    def _latest_terminal_timing(entries: tuple[_AttemptExecution, ...]) -> dict[str, Any]:
        if not entries:
            return {}
        latest = max(entries, key=lambda entry: entry.terminal_at or 0.0)
        return {
            "job_id": latest.job_id,
            "cycle_id": latest.cycle_id,
            "drain_wall_clock_s": latest.terminal_drain_wall_clock_s,
            "l0a_ready_wall_clock_s": latest.terminal_l0a_ready_wall_clock_s,
            "deterministic_ready_wall_clock_s": latest.deterministic_ready_wall_clock_s,
            "first_route_ready_wall_clock_s": latest.first_route_ready_wall_clock_s,
            "analysis_completed_wall_clock_s": latest.analysis_completed_wall_clock_s,
        }

    def _ensure_accumulator(
        self,
        entry: _AttemptExecution,
        *,
        allow_eviction: bool = True,
    ) -> None:
        if entry.accumulator is not None:
            return
        active = [
            candidate
            for candidate in self._entries.values()
            if candidate.accumulator is not None and candidate.status == _STATUS_REGISTERED
        ]
        if allow_eviction and len(active) >= self._progressive.max_active_states:
            evictable = [
                candidate
                for candidate in active
                if candidate.precompute_future is None or candidate.precompute_future.done()
            ]
            if not evictable:
                entry.progressive_phase = "precompute_skipped_capacity"
                entry.progressive_evicted = True
                return
            victim = min(
                evictable,
                key=lambda candidate: (
                    (
                        candidate.accumulator.state().last_growth_monotonic
                        if candidate.accumulator is not None
                        else None
                    )
                    or candidate.registered_at
                ),
            )
            victim.accumulator = None
            victim.precompute_future = None
            victim.next_poll_at = None
            victim.progressive_phase = "evicted"
            victim.progressive_evicted = True
            self._progressive_eviction_count += 1
        entry.accumulator = self._accumulator_factory(entry.log_path)
        entry.progressive_evicted = False

    def _schedule_progressive(self) -> None:
        """Schedule due pre-end reads from one service coordinator thread."""

        while True:
            self._scheduler_wakeup.clear()
            now = time.monotonic()
            next_due: float | None = None
            with self._lock:
                if self._shutdown:
                    return
                for entry in self._entries.values():
                    if (
                        entry.status != _STATUS_REGISTERED
                        or entry.accumulator is None
                        or entry.next_poll_at is None
                    ):
                        continue
                    if entry.precompute_future is not None and not entry.precompute_future.done():
                        continue
                    if entry.next_poll_at <= now:
                        entry.progressive_phase = "precomputing"
                        entry.precompute_future = self._precompute_executor.submit(
                            self._execute_precompute,
                            entry.key,
                            entry.accumulator,
                        )
                        continue
                    next_due = (
                        entry.next_poll_at
                        if next_due is None
                        else min(next_due, entry.next_poll_at)
                    )
            timeout = (
                self._progressive.pre_end_poll_seconds
                if next_due is None
                else max(0.01, next_due - time.monotonic())
            )
            self._scheduler_wakeup.wait(timeout=timeout)

    def _execute_precompute(
        self,
        key: Hashable,
        accumulator: ProgressiveL0Accumulator,
    ) -> bool:
        changed = False
        error: str | None = None
        try:
            changed = accumulator.refresh()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        now = time.monotonic()
        with self._lock:
            current = self._entries.get(key)
            if current is None or current.accumulator is not accumulator:
                return changed
            current.progressive_error = error
            if current.status != _STATUS_REGISTERED:
                return changed
            current.next_poll_at = now + self._progressive.pre_end_poll_seconds
            state = accumulator.state()
            last_activity = state.last_growth_monotonic or current.registered_at
            current.progressive_phase = (
                "idle"
                if now - last_activity >= self._progressive.active_idle_seconds
                else "scheduled"
            )
            if error is not None:
                self._progressive_precompute_errors += 1
                current.progressive_phase = "precompute_error"
        self._scheduler_wakeup.set()
        return changed

    def _make_room(self) -> bool:
        self._trim_completed_results(reserve=1)
        return True

    def _trim_completed_results(self, *, reserve: int = 0) -> None:
        limit = max(0, self._max_completed_results - reserve)
        completed = [
            entry
            for entry in self._entries.values()
            if entry.status in {_STATUS_COMPLETED, _STATUS_FAILED}
        ]
        while len(completed) > limit:
            victim = min(
                completed,
                key=lambda item: item.completed_at or item.registered_at,
            )
            completed.remove(victim)
            self._entries.pop(victim.key, None)
            self._path_index.pop(victim.log_path, None)
            self._eviction_count += 1

    def _status_counts(self) -> dict[str, int]:
        counts = {
            _STATUS_REGISTERED: 0,
            _STATUS_ANALYZING: 0,
            _STATUS_COMPLETED: 0,
            _STATUS_FAILED: 0,
        }
        for entry in self._entries.values():
            counts[entry.status] += 1
        return counts

    def _progressive_state_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self._entries.values():
            counts[entry.progressive_phase] = counts.get(entry.progressive_phase, 0) + 1
        return counts
