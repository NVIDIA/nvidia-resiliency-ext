# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restart agent for terminal distributed-training log decisions."""

from __future__ import annotations

from .agent_runtime import RestartAgentRuntime, build_restart_agent_runtime
from .attempt_records import (
    AttemptRecordAssembler,
    AttemptRecordControl,
    AttemptRecordStore,
    InMemoryAttemptRecordStore,
    NullAttemptRecordStore,
)
from .config import (
    HistoryConfig,
    ModelRouteSpec,
    RestartAgentConfig,
    build_model_routes,
    load_restart_agent_config,
    parse_restart_agent_config,
)
from .execution import AnalysisRun, CollectAllAnalysisRun, L0Artifacts
from .l1 import ModelRoute
from .models import (
    AnalysisResult,
    AttemptFailureFacts,
    AttemptFailureFactsSource,
    AttemptProgressSummary,
    AttemptRecord,
    CausalRole,
    CollectAllAnalysisResult,
    Decision,
    DecisionBasis,
    DecisionCandidate,
    DecisionCandidateKind,
    DecisionEvidence,
    DistributedIncidentKind,
    L0Bundle,
    L0ModelFacingView,
    ModelAnalysisResult,
    PriorAttemptView,
    RestartAgentRequest,
)
from .pipeline import RestartAgent

__all__ = [
    "AnalysisResult",
    "AnalysisRun",
    "AttemptFailureFacts",
    "AttemptFailureFactsSource",
    "AttemptProgressSummary",
    "AttemptRecord",
    "AttemptRecordAssembler",
    "AttemptRecordControl",
    "AttemptRecordStore",
    "CollectAllAnalysisResult",
    "CollectAllAnalysisRun",
    "CausalRole",
    "Decision",
    "DecisionBasis",
    "DecisionCandidate",
    "DecisionCandidateKind",
    "DecisionEvidence",
    "DistributedIncidentKind",
    "HistoryConfig",
    "InMemoryAttemptRecordStore",
    "L0Bundle",
    "L0Artifacts",
    "L0ModelFacingView",
    "ModelAnalysisResult",
    "ModelRoute",
    "ModelRouteSpec",
    "NullAttemptRecordStore",
    "PriorAttemptView",
    "RestartAgentConfig",
    "RestartAgent",
    "RestartAgentRuntime",
    "RestartAgentRequest",
    "build_model_routes",
    "build_restart_agent_runtime",
    "load_restart_agent_config",
    "parse_restart_agent_config",
]
