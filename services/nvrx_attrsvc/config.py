# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-layer configuration for nvrx_attrsvc HTTP service.

Core constants and ErrorCode are in the library layer:
    from nvidia_resiliency_ext.attribution import ErrorCode, MIN_FILE_SIZE_KB, ...

This module contains HTTP/service-specific settings only.
"""

import logging
import os
import re

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Re-export ErrorCode from library layer so service consumers can use:
#   from nvrx_attrsvc.config import ErrorCode
from nvidia_resiliency_ext.attribution import ErrorCode as ErrorCode

from nvidia_resiliency_ext.attribution.log_analyzer.config import (
    DEFAULT_LLM_BASE_URL as DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL as DEFAULT_LLM_MODEL,
)

logger = logging.getLogger(__name__)

# Service-specific constants
DEFAULT_HOST = "0.0.0.0"

DEFAULT_PORT = 8000
PRINT_PREVIEW_MAX_BYTES = 4096  # Max bytes to return for /print endpoint


class Settings(BaseSettings):
    """Typed configuration loaded from environment/.env (pydantic-settings v2).

    LLM fields (``NVRX_ATTRSVC_LLM_*``) are passed into
    :class:`~nvidia_resiliency_ext.attribution.log_analyzer.config.LogSageExecutionConfig` when set,
    then into the library :class:`~nvidia_resiliency_ext.attribution.analyzer.engine.Analyzer` via
    :class:`~nvrx_attrsvc.service.AttributionService`; unset fields keep library defaults.

    ``LOG_LEVEL`` sets the root log level, FastAPI ``debug`` (when ``LOG_LEVEL`` is ``DEBUG``), MCP
    subprocess ``--log-level``, and verbosity for in-process MCP client loggers. Allowed values:
    ``DEBUG``, ``INFO``, ``WARNING`` (default ``INFO``). Legacy env: ``NVRX_ATTRSVC_LOG_LEVEL_NAME``.
    """

    FAST_API_ROOT_PATH: str = Field(default="", description="FastAPI root path")
    ALLOWED_ROOT: str = Field(
        ..., description="Absolute base directory allowed for input paths (required)"
    )
    HOST: str = Field(default=DEFAULT_HOST)
    PORT: int = Field(default=DEFAULT_PORT)
    LOG_LEVEL: str = Field(
        default="INFO",
        description=(
            "Service log level: DEBUG, INFO, or WARNING. Drives logging.basicConfig, MCP "
            "``nvrx-mcp-analysis --log-level``, and FastAPI debug when set to DEBUG."
        ),
        validation_alias=AliasChoices("log_level", "log_level_name"),
    )
    COMPUTE_TIMEOUT: float | None = Field(
        default=None, description="Timeout for compute_fn in seconds (None = library default)"
    )

    # LLM settings → LogSageExecutionConfig when set (see AttributionService)
    LLM_MODEL: str | None = Field(default=DEFAULT_LLM_MODEL, description="LLM model identifier")
    LLM_BASE_URL: str | None = Field(default=DEFAULT_LLM_BASE_URL, description="LLM base url")
    LLM_TEMPERATURE: float | None = Field(
        default=None, description="LLM temperature (0.0 = deterministic)"
    )
    LLM_TOP_P: float | None = Field(default=None, description="LLM top-p for nucleus sampling")
    LLM_MAX_TOKENS: int | None = Field(default=None, description="Max tokens for LLM response")

    # Log + FR analysis backend: "lib" = in-process, "mcp" = subprocess MCP (same stdio client)
    ANALYSIS_BACKEND: str = Field(
        default="mcp",
        description=(
            "How to run LogSage and flight-recorder analysis: "
            "'mcp' (subprocess MCP, default) or 'lib' (in-process)."
        ),
        validation_alias=AliasChoices("analysis_backend", "log_analysis_backend"),
    )

    CLUSTER_NAME: str = Field(default="", description="Cluster name for dataflow")
    DATAFLOW_INDEX: str = Field(
        default="", description="Dataflow/elasticsearch index for posting results"
    )

    # Slack integration (optional; env vars have no NVRX_ATTRSVC_ prefix)
    SLACK_BOT_TOKEN: str = Field(
        default="",
        description="Slack bot token; if empty, setup() falls back to load_slack_bot_token() "
        "(SLACK_BOT_TOKEN_FILE, ~/.slack_bot_token, ~/.slack_token, ~/.config/nvrx/slack_bot_token)",
        validation_alias="SLACK_BOT_TOKEN",
    )
    SLACK_CHANNEL: str = Field(
        default="",
        description="Slack channel for alerts",
        validation_alias="SLACK_CHANNEL",
    )

    # Cache persistence (optional - empty = no persistence)
    CACHE_FILE: str = Field(
        default="",
        description="Path to cache file for persistence across restarts (empty = disabled)",
    )
    CACHE_GRACE_PERIOD_SECONDS: float = Field(
        default=600.0,
        description="Grace period before validating file mtime/size on cache hit (default 10 min)",
    )

    # Rate limiting (slowapi format: "N/period" e.g. "60/minute", "100/hour")
    RATE_LIMIT_SUBMIT: str = Field(
        default="1200/minute", description="Rate limit for POST /logs (submit endpoint)"
    )
    RATE_LIMIT_ANALYZE: str = Field(
        default="60/minute", description="Rate limit for GET /logs (analyze endpoint)"
    )
    RATE_LIMIT_PREVIEW: str = Field(
        default="120/minute", description="Rate limit for GET /print (file preview endpoint)"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_prefix="NVRX_ATTRSVC_",
        populate_by_name=True,
    )

    @field_validator("ANALYSIS_BACKEND")
    @classmethod
    def validate_analysis_backend(cls, v: str) -> str:
        allowed = ("lib", "mcp")
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(f"ANALYSIS_BACKEND must be one of {allowed}, got '{v}'")
        return v_lower

    @property
    def DEBUG(self) -> bool:
        """True when ``LOG_LEVEL`` is DEBUG (FastAPI debug mode)."""
        return self.LOG_LEVEL == "DEBUG"

    @field_validator("PORT")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"PORT must be between 1 and 65535, got {v}")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = ("DEBUG", "INFO", "WARNING")
        v_upper = v.strip().upper()
        if v_upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}, got '{v}'")
        return v_upper

    @field_validator("COMPUTE_TIMEOUT")
    @classmethod
    def validate_compute_timeout(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"COMPUTE_TIMEOUT must be positive, got {v}")
        return v

    @field_validator("LLM_TEMPERATURE")
    @classmethod
    def validate_llm_temperature(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 2.0):
            raise ValueError(f"LLM_TEMPERATURE must be between 0.0 and 2.0, got {v}")
        return v

    @field_validator("LLM_TOP_P")
    @classmethod
    def validate_llm_top_p(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"LLM_TOP_P must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("LLM_MAX_TOKENS")
    @classmethod
    def validate_llm_max_tokens(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError(f"LLM_MAX_TOKENS must be positive, got {v}")
        return v

    @field_validator("ALLOWED_ROOT")
    @classmethod
    def validate_allowed_root(cls, v: str) -> str:
        if not v:
            raise ValueError("ALLOWED_ROOT is required")
        real_path = os.path.realpath(v)
        if not os.path.isabs(real_path):
            raise ValueError(f"ALLOWED_ROOT must be an absolute path, got '{v}'")
        if not os.path.isdir(real_path):
            raise ValueError(f"ALLOWED_ROOT is not a directory: {real_path}")
        if not os.access(real_path, os.X_OK | os.R_OK):
            raise ValueError(f"ALLOWED_ROOT is not accessible: {real_path}")
        return v

    @field_validator("RATE_LIMIT_SUBMIT", "RATE_LIMIT_ANALYZE", "RATE_LIMIT_PREVIEW")
    @classmethod
    def validate_rate_limit(cls, v: str) -> str:
        # slowapi format: "N/period" e.g., "60/minute", "100/hour"
        pattern = r"^\d+/(second|minute|hour|day)$"
        if not re.match(pattern, v):
            raise ValueError(
                f"Rate limit must be in format 'N/period' (e.g., '60/minute'), got '{v}'"
            )
        return v

    @field_validator("CACHE_FILE")
    @classmethod
    def validate_cache_file(cls, v: str) -> str:
        """When set, require parent directory to exist and be writable (fail fast at startup)."""
        if not (v or "").strip():
            return ""
        path = os.path.abspath(os.path.expanduser(v.strip()))
        parent = os.path.dirname(path)
        if not parent:
            parent = os.getcwd()
        if not os.path.isdir(parent):
            raise ValueError(
                f"CACHE_FILE parent directory does not exist or is not a directory: {parent}"
            )
        if not os.access(parent, os.W_OK):
            raise ValueError(f"CACHE_FILE directory is not writable: {parent}")
        return path


def setup() -> Settings:
    """
    Group environment configuration and logging setup for nvrx_attrsvc.
    Returns a configured Settings instance.
    Also wires postprocessing config (poster, dataflow, Slack) from cfg.

    Field validators handle validation of PORT, LOG_LEVEL, COMPUTE_TIMEOUT,
    ALLOWED_ROOT, CACHE_FILE (when set), and rate limits. See Settings class for details.
    """
    try:
        cfg = Settings()  # type: ignore[call-arg]
    except Exception as e:
        # Fail fast if required settings are missing or invalid
        raise SystemExit(f"nvrx_attrsvc configuration error: {e}") from e

    _root_lvl = getattr(logging, cfg.LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=_root_lvl,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Suppress verbose logs from dependencies
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(_root_lvl)
    logging.getLogger("nvidia_resiliency_ext.attribution.mcp_integration").setLevel(_root_lvl)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    from nvidia_resiliency_ext.attribution.api_keys import load_nvidia_api_key, load_slack_bot_token

    nvidia_key = load_nvidia_api_key()
    if not nvidia_key:
        logger.error(
            "NVIDIA API key not found or empty. Attribution requires a key. Set NVIDIA_API_KEY "
            "or NVIDIA_API_KEY_FILE, or place a key in ~/.nvidia_api_key or "
            "~/.config/nvrx/nvidia_api_key. Slack notifications remain optional (SLACK_BOT_TOKEN)."
        )
        raise SystemExit(1)

    # Wire postprocessing config (lib singleton)
    from nvidia_resiliency_ext.attribution.postprocessing import ResultPoster, configure
    from nvidia_resiliency_ext.attribution.postprocessing.post_backend import post

    slack_token = (cfg.SLACK_BOT_TOKEN or "").strip() or load_slack_bot_token()

    configure(
        default_poster=ResultPoster(post_fn=post),
        cluster_name=cfg.CLUSTER_NAME or "",
        dataflow_index=cfg.DATAFLOW_INDEX or "",
        slack_bot_token=slack_token,
        slack_channel=cfg.SLACK_CHANNEL or "",
    )
    if slack_token:
        logger.info(f"Slack notifications enabled for channel: {cfg.SLACK_CHANNEL}")

    return cfg
