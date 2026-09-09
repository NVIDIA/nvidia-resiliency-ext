# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L1 semantic recovery assessment and provider adapters."""

from .advisories import model_evidence_contract_advisories
from .cluster_context import (
    DEFAULT_CLUSTER_EXECUTION_CONTEXT,
    ClusterExecutionContext,
    render_cluster_execution_context,
)
from .contracts import (
    DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
    EvidenceExtractor,
    EvidenceTools,
    L1EvidenceContext,
    L1EvidenceResult,
    L1ExecutionAssessment,
    L1ExecutionReason,
    L1ExecutionStatus,
    L1FinalEvidenceReason,
    L1ParseStatus,
    L1ResultQuality,
    ModelRoute,
)
from .execution import (
    assess_execution,
    deadline_exceeded_result,
    extract_timed,
    l1_contract_advisories,
    output_health,
    pending_output_health,
)
from .openai_compatible import (
    DEFAULT_ADVERTISED_TOOLS,
    THINKING_MODES,
    ChatTransport,
    ConfigCredentialProvider,
    CredentialProvider,
    HttpClient,
    LlmConfig,
    LlmEvidenceExtractor,
    OpenAICompatibleTransport,
)
from .provider_profiles import NVIDIA_INFERENCE_HUB, ProviderProfile
from .response_contract import L1_RESPONSE_CONTRACT, model_response_schema
from .tools import EvidenceToolsFactory, LogTools, build_l1_evidence_context
from .validation import model_evidence_contract_errors

__all__ = [
    "DEFAULT_ADVERTISED_TOOLS",
    "DEFAULT_ANALYSIS_TIMEOUT_SECONDS",
    "DEFAULT_CLUSTER_EXECUTION_CONTEXT",
    "ChatTransport",
    "ClusterExecutionContext",
    "ConfigCredentialProvider",
    "CredentialProvider",
    "EvidenceExtractor",
    "EvidenceTools",
    "EvidenceToolsFactory",
    "L1EvidenceContext",
    "L1EvidenceResult",
    "L1ExecutionAssessment",
    "L1FinalEvidenceReason",
    "L1ExecutionReason",
    "L1ExecutionStatus",
    "L1ParseStatus",
    "L1_RESPONSE_CONTRACT",
    "L1ResultQuality",
    "LlmConfig",
    "LlmEvidenceExtractor",
    "HttpClient",
    "LogTools",
    "ModelRoute",
    "NVIDIA_INFERENCE_HUB",
    "OpenAICompatibleTransport",
    "ProviderProfile",
    "THINKING_MODES",
    "assess_execution",
    "build_l1_evidence_context",
    "deadline_exceeded_result",
    "extract_timed",
    "l1_contract_advisories",
    "model_evidence_contract_advisories",
    "model_evidence_contract_errors",
    "model_response_schema",
    "output_health",
    "pending_output_health",
    "render_cluster_execution_context",
]
