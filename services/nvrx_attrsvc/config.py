#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Service-layer configuration for nvrx_attrsvc HTTP service.

Core constants and ErrorCode are in the library layer:
    from nvidia_resiliency_ext.attribution import ErrorCode, MIN_FILE_SIZE_KB, ...

This module contains HTTP/service-specific settings only.
"""

import logging
import os
import re

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Re-export ErrorCode from library layer so service consumers can use:
#   from nvrx_attrsvc.config import ErrorCode
from nvidia_resiliency_ext.attribution import ErrorCode as ErrorCode

logger = logging.getLogger(__name__)

# Service-specific constants
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
PRINT_PREVIEW_MAX_BYTES = 4096  # Max bytes to return for /print endpoint


class Settings(BaseSettings):
    """Typed configuration loaded from environment/.env (pydantic-settings v2).

    LLM settings (LLM_MODEL, LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_TOKENS) default to None,
    meaning the library defaults in AnalyzerConfig are used. Set via environment to override.
    """

    FAST_API_ROOT_PATH: str = Field(default="", description="FastAPI root path")
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    ALLOWED_ROOT: str = Field(
        ..., description="Absolute base directory allowed for input paths (required)"
    )
    HOST: str = Field(default=DEFAULT_HOST)
    PORT: int = Field(default=DEFAULT_PORT)
    LOG_LEVEL_NAME: str = Field(default="INFO")
    COMPUTE_TIMEOUT: float | None = Field(
        default=None, description="Timeout for compute_fn in seconds (None = library default)"
    )

    # LLM settings - None means use library defaults from AnalyzerConfig
    LLM_MODEL: str | None = Field(default=None, description="LLM model identifier")
    LLM_TEMPERATURE: float | None = Field(
        default=None, description="LLM temperature (0.0 = deterministic)"
    )
    LLM_TOP_P: float | None = Field(default=None, description="LLM top-p for nucleus sampling")
    LLM_MAX_TOKENS: int | None = Field(default=None, description="Max tokens for LLM response")

    CLUSTER_NAME: str = Field(default="", description="Cluster name for dataflow")
    DATAFLOW_INDEX: str = Field(
        default="", description="Dataflow/elasticsearch index for posting results"
    )

    # Slack integration (optional - set SLACK_BOT_TOKEN to enable; env vars have no NVRX_ATTRSVC_ prefix)
    SLACK_BOT_TOKEN: str = Field(
        default="",
        description="Slack bot token (empty = disabled)",
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
    )

    @field_validator("PORT")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"PORT must be between 1 and 65535, got {v}")
        return v

    @field_validator("LOG_LEVEL_NAME")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"LOG_LEVEL_NAME must be one of {valid_levels}, got '{v}'")
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


def setup() -> Settings:
    """
    Group environment configuration and logging setup for nvrx_attrsvc.
    Returns a configured Settings instance.
    Also wires postprocessing config (poster, dataflow, Slack) from cfg.

    Field validators handle validation of PORT, LOG_LEVEL_NAME, COMPUTE_TIMEOUT,
    ALLOWED_ROOT, and rate limits. See Settings class for details.
    """
    try:
        cfg = Settings()  # type: ignore[call-arg]
    except Exception as e:
        # Fail fast if required settings are missing or invalid
        raise SystemExit(f"nvrx_attrsvc configuration error: {e}") from e

    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL_NAME, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Suppress verbose logs from dependencies
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.WARNING)
    logging.getLogger("nvidia_resiliency_ext.attribution.mcp_integration").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # Wire postprocessing config (lib singleton)
    from nvidia_resiliency_ext.attribution.postprocessing import ResultPoster, configure

    from . import dataflow

    configure(
        default_poster=ResultPoster(post_fn=dataflow.post),
        cluster_name=cfg.CLUSTER_NAME or "",
        dataflow_index=cfg.DATAFLOW_INDEX or "",
        slack_bot_token=cfg.SLACK_BOT_TOKEN or "",
        slack_channel=cfg.SLACK_CHANNEL or "",
    )
    if cfg.SLACK_BOT_TOKEN:
        logger.info(f"Slack notifications enabled for channel: {cfg.SLACK_CHANNEL}")

    return cfg
