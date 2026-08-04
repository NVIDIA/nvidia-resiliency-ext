# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve attrsvc settings into the canonical Restart Agent configuration."""

from __future__ import annotations

import os
from typing import Mapping

from nvidia_resiliency_ext.attribution.restart_agent import (
    RestartAgentConfig,
    load_restart_agent_config,
    parse_restart_agent_config,
)
from nvidia_resiliency_ext.attribution.restart_agent.config import (
    RESTART_AGENT_CONFIG_SCHEMA_VERSION,
)
from nvidia_resiliency_ext.attribution.restart_agent.l1 import LlmConfig

from .config import Settings


def restart_agent_config_from_settings(
    settings: Settings,
    *,
    environ: Mapping[str, str] | None = None,
) -> RestartAgentConfig:
    """Load an authoritative file or create one attrsvc-managed model route."""

    environment = os.environ if environ is None else environ
    if settings.ANALYSIS_BACKEND != "lib":
        if settings.RESTART_AGENT_CONFIG:
            raise ValueError("RESTART_AGENT_CONFIG requires ANALYSIS_BACKEND=lib")
        raise ValueError("Restart Agent configuration requires ANALYSIS_BACKEND=lib")

    if settings.RESTART_AGENT_CONFIG:
        config = load_restart_agent_config(settings.RESTART_AGENT_CONFIG, environ=environment)
    elif settings.RESTART_AGENT_ENRICHMENT_ENABLED:
        defaults = LlmConfig.from_env(environ=environment)
        request: dict[str, object] = {}
        if settings.LLM_TEMPERATURE is not None:
            request["temperature"] = settings.LLM_TEMPERATURE
        if settings.LLM_TOP_P is not None:
            request["top_p"] = settings.LLM_TOP_P
        if settings.LLM_MAX_TOKENS is not None:
            request["max_output_tokens"] = settings.LLM_MAX_TOKENS
        route: dict[str, object] = {
            "route_id": "nvrx-default",
            "model": settings.LLM_MODEL or defaults.model,
            "base_url": settings.LLM_BASE_URL or defaults.base_url,
            "credential_ref": "LLM_API_KEY_FILE",
        }
        if request:
            route["request"] = request
        config = parse_restart_agent_config(
            {
                "schema_version": RESTART_AGENT_CONFIG_SCHEMA_VERSION,
                "config_id": "nvrx-attrsvc-environment",
                "config_version": 1,
                "enrichment": {"enabled": True},
                "routing": {"mode": "collect_all", "max_parallel_models": 1},
                "model_routes": [route],
            },
            environ=environment,
        )
    else:
        config = parse_restart_agent_config(
            {
                "schema_version": RESTART_AGENT_CONFIG_SCHEMA_VERSION,
                "config_id": "nvrx-attrsvc-deterministic",
                "config_version": 1,
                "enrichment": {"enabled": False},
                "routing": {"mode": "collect_all", "max_parallel_models": 0},
                "model_routes": [],
            },
            environ=environment,
        )

    if config.enrichment_enabled and len(config.model_route_specs) != 1:
        raise ValueError("attrsvc Restart Agent integration requires exactly one model route")
    return config
