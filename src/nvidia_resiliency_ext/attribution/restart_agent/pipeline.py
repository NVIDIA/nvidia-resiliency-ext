# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small public facade over restart-agent run coordinators."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence

from .execution import AnalysisRun, CollectAllAnalysisRun, L0Artifacts
from .infrastructure.log_source import LocalLogSource
from .l0.streaming import FinalizedL0A
from .l1 import (
    DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
    EvidenceExtractor,
    EvidenceToolsFactory,
    ModelRoute,
)
from .models import (
    AnalysisExecutionContext,
    AttemptRecord,
    DecisionCandidate,
    L0Bundle,
    ModelAnalysisResult,
    PolicyContextConfig,
    PriorAttemptView,
    RestartAgentRequest,
    build_analysis_execution_context,
    normalize_restart_agent_request,
)
from .multi_route import FutureWaiter, LogSourceFactory, MultiRouteCoordinator, RouteRunnerFactory
from .preparation import prepare_finalized_analysis, validate_timeout_seconds
from .runtime import SYSTEM_CLOCK, THREAD_EXECUTOR_FACTORY, Clock, ExecutorFactory
from .single_run import SingleRunCoordinator


class RestartAgent:
    """Run terminal analysis without retaining invocation-owned artifacts."""

    def __init__(
        self,
        evidence_extractor: EvidenceExtractor | None = None,
        *,
        log_source_factory: LogSourceFactory = LocalLogSource,
        route_runner_factory: RouteRunnerFactory | None = None,
        evidence_tools_factory: EvidenceToolsFactory | None = None,
        clock: Clock = SYSTEM_CLOCK,
        executor_factory: ExecutorFactory = THREAD_EXECUTOR_FACTORY,
        future_waiter: FutureWaiter | None = None,
        retry_policy: Mapping[str, Any] | None = None,
        policy_contexts: Mapping[str, Any] | PolicyContextConfig | None = None,
    ) -> None:
        self._retry_policy = retry_policy
        self._policy_contexts = policy_contexts
        self._clock = clock
        self._single_run = SingleRunCoordinator(
            evidence_extractor=evidence_extractor,
            log_source_factory=log_source_factory,
            evidence_tools_factory=evidence_tools_factory,
            clock=clock,
            executor_factory=executor_factory,
        )
        self._multi_route = MultiRouteCoordinator(
            route_runner_factory=route_runner_factory,
            log_source_factory=log_source_factory,
            evidence_tools_factory=evidence_tools_factory,
            clock=clock,
            executor_factory=executor_factory,
            future_waiter=future_waiter,
        )

    def run(
        self,
        request: RestartAgentRequest | Mapping[str, Any],
        *,
        l0_bundle: L0Bundle | None = None,
        prior_attempts: PriorAttemptView | None = None,
        on_l0_ready: Callable[[L0Artifacts], None] | None = None,
        on_deterministic_ready: Callable[[DecisionCandidate], None] | None = None,
        timeout_seconds: float = DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
        retain_detailed_artifacts: bool = True,
    ) -> AnalysisRun:
        run = self._single_run.run(
            self._execution_context(request, prior_attempts=prior_attempts),
            l0_bundle=l0_bundle,
            on_l0_ready=on_l0_ready,
            on_deterministic_ready=on_deterministic_ready,
            timeout_seconds=timeout_seconds,
        )
        return _retained_single_run(run, retain_detailed_artifacts)

    def run_many(
        self,
        request: RestartAgentRequest | Mapping[str, Any],
        model_routes: Sequence[ModelRoute],
        *,
        l0_bundle: L0Bundle | None = None,
        prior_attempts: PriorAttemptView | None = None,
        max_parallel_models: int | None = None,
        config_metadata: Mapping[str, Any] | None = None,
        on_l0_ready: Callable[[L0Artifacts], None] | None = None,
        on_deterministic_ready: Callable[[DecisionCandidate], None] | None = None,
        on_route_complete: Callable[[ModelAnalysisResult, Mapping[str, Any]], None] | None = None,
        on_attempt_record_ready: Callable[[str, AttemptRecord], None] | None = None,
        timeout_seconds: float = DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
        retain_detailed_artifacts: bool = True,
    ) -> CollectAllAnalysisRun:
        return self._multi_route.run_many(
            self._execution_context(request, prior_attempts=prior_attempts),
            model_routes,
            l0_bundle=l0_bundle,
            max_parallel_models=max_parallel_models,
            config_metadata=config_metadata,
            on_l0_ready=on_l0_ready,
            on_deterministic_ready=on_deterministic_ready,
            on_route_complete=on_route_complete,
            on_attempt_record_ready=on_attempt_record_ready,
            timeout_seconds=timeout_seconds,
            retain_detailed_artifacts=retain_detailed_artifacts,
        )

    def run_prepared(
        self,
        request: RestartAgentRequest | Mapping[str, Any],
        finalized_l0a: FinalizedL0A,
        *,
        prior_attempts: PriorAttemptView | None = None,
        on_l0_ready: Callable[[L0Artifacts], None] | None = None,
        on_deterministic_ready: Callable[[DecisionCandidate], None] | None = None,
        timeout_seconds: float = DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
        retain_detailed_artifacts: bool = True,
    ) -> AnalysisRun:
        """Run one route without rereading or rebuilding caller-finalized L0A."""

        timeout_seconds = validate_timeout_seconds(timeout_seconds)
        prepared = prepare_finalized_analysis(
            self._execution_context(request, prior_attempts=prior_attempts),
            finalized_l0a,
            include_model_view=self._single_run.has_evidence_extractor,
            clock=self._clock,
        )
        run = self._single_run.run_prepared(
            prepared,
            deadline_monotonic=prepared.analysis_started + timeout_seconds,
            on_l0_ready=on_l0_ready,
            on_deterministic_ready=on_deterministic_ready,
        )
        return _retained_single_run(run, retain_detailed_artifacts)

    def run_many_prepared(
        self,
        request: RestartAgentRequest | Mapping[str, Any],
        finalized_l0a: FinalizedL0A,
        model_routes: Sequence[ModelRoute],
        *,
        prior_attempts: PriorAttemptView | None = None,
        max_parallel_models: int | None = None,
        config_metadata: Mapping[str, Any] | None = None,
        on_l0_ready: Callable[[L0Artifacts], None] | None = None,
        on_deterministic_ready: Callable[[DecisionCandidate], None] | None = None,
        on_route_complete: Callable[[ModelAnalysisResult, Mapping[str, Any]], None] | None = None,
        on_attempt_record_ready: Callable[[str, AttemptRecord], None] | None = None,
        timeout_seconds: float = DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
        retain_detailed_artifacts: bool = True,
    ) -> CollectAllAnalysisRun:
        """Run model routes without rereading caller-finalized L0A."""

        prepared = prepare_finalized_analysis(
            self._execution_context(request, prior_attempts=prior_attempts),
            finalized_l0a,
            include_model_view=True,
            clock=self._clock,
        )
        return self._multi_route.run_many_prepared(
            prepared,
            model_routes,
            max_parallel_models=max_parallel_models,
            config_metadata=config_metadata,
            on_l0_ready=on_l0_ready,
            on_deterministic_ready=on_deterministic_ready,
            on_route_complete=on_route_complete,
            on_attempt_record_ready=on_attempt_record_ready,
            timeout_seconds=timeout_seconds,
            retain_detailed_artifacts=retain_detailed_artifacts,
        )

    def _execution_context(
        self,
        request: RestartAgentRequest | Mapping[str, Any],
        *,
        prior_attempts: PriorAttemptView | None = None,
    ) -> AnalysisExecutionContext:
        normalized_request = normalize_restart_agent_request(request)
        return build_analysis_execution_context(
            normalized_request,
            prior_attempts=prior_attempts,
            retry_policy=self._retry_policy,
            policy_contexts=self._policy_contexts,
        )


def _retained_single_run(run: AnalysisRun, retain_detailed_artifacts: bool) -> AnalysisRun:
    if retain_detailed_artifacts:
        return run
    return replace(
        run,
        trace={"retention_mode": "compact_service"},
        bundle=None,
        decision_evidence=None,
        model_view=None,
        deterministic_candidate=None,
    )
