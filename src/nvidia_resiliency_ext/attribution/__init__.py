# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Current packaged attribution APIs and shared service wire types.

The built wheel ships restart-agent, FR analysis, request coalescing, and
shared attrsvc response/config types. Legacy LogSage, SPLITLOG orchestration,
and LogSage-backed MCP tools live under ``legacy_logsage`` in source checkouts
and are intentionally excluded from built wheels.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .coalescing import (
        DEFAULT_COMPUTE_TIMEOUT_SECONDS,
        CacheResult,
        CoalescerStats,
        InflightResult,
        LogAnalysisCoalesced,
        RequestCoalescer,
        StatsResult,
        SubmittedResult,
        coalesced_from_cache,
    )
    from .orchestration.analysis_pipeline import (
        AnalysisPipelineMode,
        CombinedAnalysisResult,
        FrDumpPathNotFoundError,
        run_attribution_pipeline,
    )
    from .orchestration.client_response import (
        AttrSvcResult,
        parse_attrsvc_response,
        recommendation_should_stop,
    )
    from .orchestration.config import (
        MAX_JOBS,
        MIN_FILE_SIZE_KB,
        POLL_INTERVAL_SECONDS,
        RESP_ERROR,
        RESP_FILES_ANALYZED,
        RESP_LOG_FILE,
        RESP_LOGS_DIR,
        RESP_MODE,
        RESP_MODULE,
        RESP_RESULT,
        RESP_RESULT_ID,
        RESP_SCHED_RESTARTS,
        RESP_STATE,
        RESP_STATUS,
        RESP_WL_RESTART,
        RESP_WL_RESTART_COUNT,
        STATE_TIMEOUT,
        TTL_MAX_JOB_AGE_SECONDS,
        TTL_PENDING_SECONDS,
        TTL_TERMINATED_SECONDS,
        ErrorCode,
    )
    from .orchestration.job import FileInfo, Job, JobMode
    from .orchestration.types import (
        AttributionRecommendation,
        LogAnalysisCycleResult,
        LogAnalysisSplitlogResult,
        LogAnalyzerConfig,
        LogAnalyzerError,
        LogAnalyzerFilePreview,
        LogAnalyzerOutcome,
        LogAnalyzerSubmitResult,
        RawAnalysisResultItem,
    )
    from .trace_analyzer.trace_analyzer import TraceAnalyzer

_EXPORTS = {
    "AnalysisPipelineMode": ".orchestration.analysis_pipeline",
    "CombinedAnalysisResult": ".orchestration.analysis_pipeline",
    "FrDumpPathNotFoundError": ".orchestration.analysis_pipeline",
    "TraceAnalyzer": ".trace_analyzer.trace_analyzer",
    "run_attribution_pipeline": ".orchestration.analysis_pipeline",
    "DEFAULT_COMPUTE_TIMEOUT_SECONDS": ".coalescing",
    "CacheResult": ".coalescing",
    "CoalescerStats": ".coalescing",
    "InflightResult": ".coalescing",
    "LogAnalysisCoalesced": ".coalescing",
    "RequestCoalescer": ".coalescing",
    "StatsResult": ".coalescing",
    "SubmittedResult": ".coalescing",
    "coalesced_from_cache": ".coalescing",
    "MAX_JOBS": ".orchestration.config",
    "MIN_FILE_SIZE_KB": ".orchestration.config",
    "POLL_INTERVAL_SECONDS": ".orchestration.config",
    "RESP_ERROR": ".orchestration.config",
    "RESP_FILES_ANALYZED": ".orchestration.config",
    "RESP_LOG_FILE": ".orchestration.config",
    "RESP_LOGS_DIR": ".orchestration.config",
    "RESP_MODE": ".orchestration.config",
    "RESP_MODULE": ".orchestration.config",
    "RESP_RESULT": ".orchestration.config",
    "RESP_RESULT_ID": ".orchestration.config",
    "RESP_SCHED_RESTARTS": ".orchestration.config",
    "RESP_STATE": ".orchestration.config",
    "RESP_STATUS": ".orchestration.config",
    "RESP_WL_RESTART": ".orchestration.config",
    "RESP_WL_RESTART_COUNT": ".orchestration.config",
    "STATE_TIMEOUT": ".orchestration.config",
    "TTL_MAX_JOB_AGE_SECONDS": ".orchestration.config",
    "TTL_PENDING_SECONDS": ".orchestration.config",
    "TTL_TERMINATED_SECONDS": ".orchestration.config",
    "ErrorCode": ".orchestration.config",
    "AttrSvcResult": ".orchestration.client_response",
    "parse_attrsvc_response": ".orchestration.client_response",
    "recommendation_should_stop": ".orchestration.client_response",
    "FileInfo": ".orchestration.job",
    "Job": ".orchestration.job",
    "JobMode": ".orchestration.job",
    "AttributionRecommendation": ".orchestration.types",
    "LogAnalysisCycleResult": ".orchestration.types",
    "LogAnalysisSplitlogResult": ".orchestration.types",
    "LogAnalyzerConfig": ".orchestration.types",
    "LogAnalyzerError": ".orchestration.types",
    "LogAnalyzerFilePreview": ".orchestration.types",
    "LogAnalyzerOutcome": ".orchestration.types",
    "LogAnalyzerSubmitResult": ".orchestration.types",
    "RawAnalysisResultItem": ".orchestration.types",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, __name__)

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))


__all__ = [
    # Log + FR orchestration helpers (LogSage runner is source-only legacy)
    "AnalysisPipelineMode",
    "CombinedAnalysisResult",
    "FrDumpPathNotFoundError",
    "run_attribution_pipeline",
    # Coalescer cache payload (LogSage + optional FR)
    "LogAnalysisCoalesced",
    "coalesced_from_cache",
    # Main API
    "TraceAnalyzer",
    "LogAnalyzerConfig",
    "AttributionRecommendation",
    "AttrSvcResult",
    "parse_attrsvc_response",
    "recommendation_should_stop",
    "LogAnalyzerError",
    "LogAnalyzerOutcome",
    "LogAnalysisCycleResult",
    "LogAnalyzerSubmitResult",
    "LogAnalysisSplitlogResult",
    "LogAnalyzerFilePreview",
    "RawAnalysisResultItem",
    # Configuration and error codes
    "ErrorCode",
    "TTL_PENDING_SECONDS",
    "TTL_TERMINATED_SECONDS",
    "TTL_MAX_JOB_AGE_SECONDS",
    "POLL_INTERVAL_SECONDS",
    "DEFAULT_COMPUTE_TIMEOUT_SECONDS",
    "MAX_JOBS",
    "MIN_FILE_SIZE_KB",
    "RESP_MODE",
    "RESP_RESULT",
    "RESP_STATUS",
    "RESP_LOG_FILE",
    "RESP_WL_RESTART",
    "RESP_WL_RESTART_COUNT",
    "RESP_SCHED_RESTARTS",
    "RESP_LOGS_DIR",
    "RESP_FILES_ANALYZED",
    "RESP_MODULE",
    "RESP_STATE",
    "RESP_ERROR",
    "RESP_RESULT_ID",
    "STATE_TIMEOUT",
    # Request coalescing
    "RequestCoalescer",
    "CoalescerStats",
    "StatsResult",
    "CacheResult",
    "InflightResult",
    "SubmittedResult",
    # Job data model
    "Job",
    "FileInfo",
    "JobMode",
]
