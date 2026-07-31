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
    L0SourceConfig,
    ModelRouteSpec,
    RestartAgentConfig,
    build_log_source_factory,
    build_model_routes,
    load_restart_agent_config,
    parse_restart_agent_config,
)
from .execution import AnalysisRun, CollectAllAnalysisRun, L0Artifacts
from .l0 import (
    FinalizedL0A,
    ProgressiveL0Accumulator,
    ProgressiveL0State,
    ProgressiveSourceUnavailable,
)
from .l1 import ModelRoute
from .models import (
    AffectedEntity,
    AffectedEntityKind,
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
    DeclaredRecoveryCapability,
    DistributedIncidentKind,
    HistoryMatchScope,
    L0Bundle,
    L0ModelFacingView,
    ModelAnalysisResult,
    PriorAttemptView,
    RecoveryBehavior,
    RecoveryCapabilityId,
    RestartAgentRequest,
)
from .pipeline import RestartAgent

__all__ = [
    "AffectedEntity",
    "AffectedEntityKind",
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
    "DeclaredRecoveryCapability",
    "DistributedIncidentKind",
    "HistoryConfig",
    "HistoryMatchScope",
    "InMemoryAttemptRecordStore",
    "L0SourceConfig",
    "L0Bundle",
    "L0Artifacts",
    "L0ModelFacingView",
    "FinalizedL0A",
    "ModelAnalysisResult",
    "ModelRoute",
    "ModelRouteSpec",
    "NullAttemptRecordStore",
    "PriorAttemptView",
    "ProgressiveL0Accumulator",
    "ProgressiveL0State",
    "ProgressiveSourceUnavailable",
    "RecoveryBehavior",
    "RecoveryCapabilityId",
    "RestartAgentConfig",
    "RestartAgent",
    "RestartAgentRuntime",
    "RestartAgentRequest",
    "build_model_routes",
    "build_log_source_factory",
    "build_restart_agent_runtime",
    "load_restart_agent_config",
    "parse_restart_agent_config",
]
