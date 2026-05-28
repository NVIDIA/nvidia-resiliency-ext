# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core configuration and constants for log analysis.

This module contains library-level constants and error codes used by
the log analyzer components. Service-specific configuration (HTTP settings,
pydantic Settings class) should remain in the service layer.

Constants overview:
- TTL_* : Time-to-live values for job cleanup
- POLL_INTERVAL_SECONDS: How often splitlog tracker polls for changes
- MAX_JOBS: Maximum number of tracked jobs
- MIN_FILE_SIZE_KB: Minimum file size for analysis

Compute timeout defaults live on :class:`~nvidia_resiliency_ext.attribution.coalescing.RequestCoalescer`
(see ``DEFAULT_COMPUTE_TIMEOUT_SECONDS`` in ``attribution.coalescing``); :class:`~nvidia_resiliency_ext.attribution.analyzer.engine.Analyzer` accepts ``compute_timeout`` / ``grace_period_seconds`` for the coalescer.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

# LLM defaults — override with NVRX_LLM_MODEL / NVRX_LLM_BASE_URL env vars.
# Default endpoint is build.nvidia.com (publicly accessible).
# Internal NVIDIA users can override to inference.nvidia.com via NVRX_LLM_BASE_URL.
DEFAULT_LLM_MODEL = os.environ.get("NVRX_LLM_MODEL", "nvidia/nvidia/nemotron-3-super-v3")
DEFAULT_LLM_BASE_URL = os.environ.get("NVRX_LLM_BASE_URL", "https://inference-api.nvidia.com/v1")
DEFAULT_LLM_TEMPERATURE = 0.2
DEFAULT_LLM_TOP_P = 0.7
DEFAULT_LLM_MAX_TOKENS = 8192

# TTL constants (see spec Section 3.2)
TTL_PENDING_SECONDS = 7 * 24 * 60 * 60  # 1 week - pending job expiry
TTL_TERMINATED_SECONDS = 60 * 60  # 1 hour - terminated job expiry (after GET)
TTL_MAX_JOB_AGE_SECONDS = 6 * 30 * 24 * 60 * 60  # 6 months - non-terminated safety net

# Poll/tracking constants
POLL_INTERVAL_SECONDS = 5 * 60  # 5 minutes - background poll interval

# Limits (see spec Section 3.2)
MAX_JOBS = 100_000  # Maximum tracked jobs
MIN_FILE_SIZE_KB = 4  # Minimum file size (KB) for classification


@dataclass
class LogSageExecutionConfig:
    """Lib/MCP runtime override knobs for :class:`~nvidia_resiliency_ext.attribution.orchestration.log_analyzer.LogAnalyzer`.

    ``use_lib_log_analysis`` selects **both** LogSage and flight-recorder analysis: in-process vs the
    same MCP subprocess used for ``log_analyzer`` / ``fr_analyzer`` tools.
    LLM fields are optional overrides; ``None`` means orchestration omits the key so the lower
    LogSage/MCP/merge layer applies its own default.

    Subset of orchestration :class:`~nvidia_resiliency_ext.attribution.orchestration.types.LogAnalyzerConfig`
    (no ``allowed_root`` — path policy stays in the attribution :class:`~nvidia_resiliency_ext.attribution.analyzer.engine.Analyzer`).
    """

    use_lib_log_analysis: bool = False
    #: Subprocess MCP server (:func:`~nvidia_resiliency_ext.attribution.mcp_integration.mcp_client.get_server_command`).
    mcp_server_log_level: str = "INFO"
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_temperature: float | None = None
    llm_top_p: float | None = None
    llm_max_tokens: int | None = None

    def llm_runtime_overrides(self) -> dict[str, Any]:
        """LLM kwargs for LogSage/MCP/runtime calls, omitting unset overrides."""
        return llm_runtime_overrides(
            model=self.llm_model,
            base_url=self.llm_base_url,
            temperature=self.llm_temperature,
            top_p=self.llm_top_p,
            max_tokens=self.llm_max_tokens,
        )

    def llm_endpoint_overrides(self) -> dict[str, Any]:
        """Endpoint-only LLM kwargs for callers that do not consume sampling settings."""
        return drop_none_values(
            {
                "model": self.llm_model,
                "base_url": self.llm_base_url,
            }
        )

    def pipeline_llm_overrides(self) -> dict[str, Any]:
        """LLM override kwargs for :func:`run_attribution_pipeline`."""
        return drop_none_values(
            {
                "llm_model": self.llm_model,
                "llm_base_url": self.llm_base_url,
                "llm_temperature": self.llm_temperature,
                "llm_top_p": self.llm_top_p,
                "llm_max_tokens": self.llm_max_tokens,
            }
        )


def drop_none_values(values: dict[str, Any]) -> dict[str, Any]:
    """Return a copy without ``None`` values, preserving falsy overrides like ``0``."""
    return {key: value for key, value in values.items() if value is not None}


def llm_runtime_overrides(
    *,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Runtime LLM kwargs with only explicitly supplied overrides."""
    return drop_none_values(
        {
            "model": model,
            "base_url": base_url,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
    )


def _value_or_default(values: Mapping[str, Any], key: str, default: Any) -> Any:
    value = values.get(key)
    return default if value is None else value


def resolved_llm_runtime_kwargs(values: Mapping[str, Any]) -> dict[str, Any]:
    """Runtime LLM kwargs with lower-layer defaults applied."""
    return {
        "model": _value_or_default(values, "model", DEFAULT_LLM_MODEL),
        "base_url": _value_or_default(values, "base_url", DEFAULT_LLM_BASE_URL),
        "temperature": float(_value_or_default(values, "temperature", DEFAULT_LLM_TEMPERATURE)),
        "top_p": float(_value_or_default(values, "top_p", DEFAULT_LLM_TOP_P)),
        "max_tokens": int(_value_or_default(values, "max_tokens", DEFAULT_LLM_MAX_TOKENS)),
    }


# Result/response keys (serialized shape of orchestration results; see svc.types)
# Used by library and HTTP layer for consistent parsing. Job mode values are JobMode enum.
RESP_MODE = "mode"
RESP_RESULT = "result"
RESP_STATUS = "status"
RESP_LOG_FILE = "log_file"
RESP_WL_RESTART = "wl_restart"
RESP_WL_RESTART_COUNT = "wl_restart_count"
RESP_SCHED_RESTARTS = "sched_restarts"
RESP_LOGS_DIR = "logs_dir"
RESP_FILES_ANALYZED = "files_analyzed"
# Inner result dict (RESP_RESULT value from analysis pipeline)
RESP_MODULE = "module"
RESP_STATE = "state"
RESP_ERROR = "error"
RESP_RESULT_ID = "result_id"
# Inner result RESP_MODULE values
MODULE_LOG_ANALYZER = "log_analyzer"
MODULE_LOG_FR_ANALYZER = "log_fr_analyzer"
MODULE_FR_ONLY = "fr_only"
# Inner result RESP_STATE values
STATE_NO_LOG = "no_log"
STATE_TIMEOUT = "timeout"

# Stats / job detail keys (get_stats, get_jobs_detail, get_all_jobs response shape)
STATS_JOBS = "jobs"
STATS_JOBS_TERMINATED = "jobs_terminated"
STATS_JOBS_CLEANED = "jobs_cleaned"
STATS_FILES = "files"
STATS_FILES_TRACKED = "tracked"
STATS_FILES_ANALYZED = "analyzed"  # key under STATS_FILES (count of analyzed files)
STATS_SCHED_RESTARTS = "sched_restarts"
STATS_JOB_ID = "job_id"
STATS_LOG_PATH = "log_path"
STATS_USER = "user"
STATS_TERMINATED = "terminated"
STATS_LOG_FILES = "log_files"


class ErrorCode(str, Enum):
    """Error codes for log analysis operations.

    See spec Section 7 for HTTP status mapping when used in HTTP context.
    """

    # Path validation errors (400 in HTTP)
    INVALID_PATH = "invalid_path"  # Path not absolute, null bytes, etc.
    INVALID_PARAMETER = "invalid_parameter"  # Request field failed validation
    NOT_REGULAR = "not_regular"  # Not a regular file (directory, device, etc.)
    EMPTY_FILE = "empty_file"  # File is empty (GET only)

    # Permission errors (403 in HTTP)
    OUTSIDE_ROOT = "outside_root"  # Path (or symlink target) outside allowed root
    NOT_READABLE = "not_readable"  # File permission denied
    LOGS_DIR_NOT_READABLE = "logs_dir_not_readable"  # LOGS_DIR permission denied

    # Not found (404 in HTTP)
    NOT_FOUND = "not_found"  # File doesn't exist
    FR_DUMP_NOT_FOUND = "fr_dump_not_found"  # FR dump path not discoverable for trace-only analysis

    # Server errors (5xx in HTTP)
    JOB_LIMIT_REACHED = "job_limit_reached"  # MAX_JOBS exceeded (503)
    INTERNAL_ERROR = "internal_error"  # Unexpected server error (500)
