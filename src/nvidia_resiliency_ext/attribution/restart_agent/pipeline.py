# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small public facade over restart-agent run coordinators."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .execution import AnalysisRun, CollectAllAnalysisRun, L0Artifacts
from .infrastructure.log_source import LocalLogSource
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
    PriorAttemptView,
    RestartAgentRequest,
    build_analysis_execution_context,
    normalize_restart_agent_request,
)
from .multi_route import FutureWaiter, LogSourceFactory, MultiRouteCoordinator, RouteRunnerFactory
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
        restart_environment_context: Mapping[str, bool] | None = None,
        retry_policy: Mapping[str, Any] | None = None,
    ) -> None:
        self._restart_environment_context = restart_environment_context
        self._retry_policy = retry_policy
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
        on_fallback_ready: Callable[[DecisionCandidate], None] | None = None,
        timeout_seconds: float = DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
    ) -> AnalysisRun:
        return self._single_run.run(
            self._execution_context(request, prior_attempts=prior_attempts),
            l0_bundle=l0_bundle,
            on_l0_ready=on_l0_ready,
            on_fallback_ready=on_fallback_ready,
            timeout_seconds=timeout_seconds,
        )

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
        on_fallback_ready: Callable[[DecisionCandidate], None] | None = None,
        on_route_complete: Callable[[ModelAnalysisResult, Mapping[str, Any]], None] | None = None,
        on_attempt_record_ready: Callable[[str, AttemptRecord], None] | None = None,
        timeout_seconds: float = DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
    ) -> CollectAllAnalysisRun:
        return self._multi_route.run_many(
            self._execution_context(request, prior_attempts=prior_attempts),
            model_routes,
            l0_bundle=l0_bundle,
            max_parallel_models=max_parallel_models,
            config_metadata=config_metadata,
            on_l0_ready=on_l0_ready,
            on_fallback_ready=on_fallback_ready,
            on_route_complete=on_route_complete,
            on_attempt_record_ready=on_attempt_record_ready,
            timeout_seconds=timeout_seconds,
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
            restart_environment_context=self._restart_environment_context,
            retry_policy=self._retry_policy,
        )
