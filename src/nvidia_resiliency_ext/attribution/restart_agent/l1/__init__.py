# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L1 semantic recovery assessment and provider adapters."""

from .contracts import (
    DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
    EvidenceExtractor,
    EvidenceTools,
    L1EvidenceContext,
    L1EvidenceResult,
    ModelRoute,
)
from .execution import deadline_exceeded_result, execution_status, extract_timed, output_health
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
    "ChatTransport",
    "ConfigCredentialProvider",
    "CredentialProvider",
    "EvidenceExtractor",
    "EvidenceTools",
    "EvidenceToolsFactory",
    "L1EvidenceContext",
    "L1EvidenceResult",
    "L1_RESPONSE_CONTRACT",
    "LlmConfig",
    "LlmEvidenceExtractor",
    "HttpClient",
    "LogTools",
    "ModelRoute",
    "NVIDIA_INFERENCE_HUB",
    "OpenAICompatibleTransport",
    "ProviderProfile",
    "THINKING_MODES",
    "build_l1_evidence_context",
    "deadline_exceeded_result",
    "execution_status",
    "extract_timed",
    "model_evidence_contract_errors",
    "model_response_schema",
    "output_health",
]
