# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attribution module for failure attribution and log analysis.

This module provides:
- Analyzer: main API for analyzing logs (usable without HTTP)
- Core configuration and error codes
- Request coalescing for deduplication and caching
- Job/FileInfo data model for tracking analysis state
- SLURM parser for extracting metadata from scheduler output
- SplitlogTracker for multi-file job tracking

Example usage (no HTTP required):
    from nvidia_resiliency_ext.attribution import Analyzer

    analyzer = Analyzer(allowed_root="/logs", use_lib_log_analysis=False)
    
    # Submit and analyze
    await analyzer.submit("/logs/slurm-12345.out", user="alice")
    result = await analyzer.analyze("/logs/slurm-12345.out")
    
    analyzer.shutdown()

See ``README.md`` and ``ARCHITECTURE.md`` in this package for how the library is organized.

Log-analysis features use optional dependencies. Install them with
``pip install nvidia-resiliency-ext[attribution]`` when you need ``Analyzer``, LogSage,
MCP integration, or related attribution tooling.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from ._optional import reraise_if_missing_attribution_dependency

if TYPE_CHECKING:
    from .analyzer import (
        AnalysisPipelineMode,
        Analyzer,
        CombinedAnalysisResult,
        FrDumpPathNotFoundError,
        TraceAnalyzer,
        run_attribution_pipeline,
    )
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
    from .controller import (
        AttributionAnalysisConfig,
        AttributionCacheConfig,
        AttributionController,
        AttributionControllerConfig,
        AttributionCredentialsConfig,
        AttributionPostprocessingConfig,
    )
    from .svc.config import (
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
    from .svc.job import FileInfo, Job, JobMode
    from .svc.splitlog import SplitlogTracker
    from .svc.types import (
        LogAnalysisCycleResult,
        LogAnalysisSplitlogResult,
        LogAnalyzerConfig,
        LogAnalyzerError,
        LogAnalyzerFilePreview,
        LogAnalyzerOutcome,
        LogAnalyzerSubmitResult,
    )

_EXPORTS = {
    "AnalysisPipelineMode": ".analyzer",
    "Analyzer": ".analyzer",
    "CombinedAnalysisResult": ".analyzer",
    "FrDumpPathNotFoundError": ".analyzer",
    "TraceAnalyzer": ".analyzer",
    "run_attribution_pipeline": ".analyzer",
    "DEFAULT_COMPUTE_TIMEOUT_SECONDS": ".coalescing",
    "CacheResult": ".coalescing",
    "CoalescerStats": ".coalescing",
    "InflightResult": ".coalescing",
    "LogAnalysisCoalesced": ".coalescing",
    "RequestCoalescer": ".coalescing",
    "StatsResult": ".coalescing",
    "SubmittedResult": ".coalescing",
    "coalesced_from_cache": ".coalescing",
    "AttributionController": ".controller",
    "AttributionControllerConfig": ".controller",
    "AttributionAnalysisConfig": ".controller",
    "AttributionCacheConfig": ".controller",
    "AttributionCredentialsConfig": ".controller",
    "AttributionPostprocessingConfig": ".controller",
    "MAX_JOBS": ".svc.config",
    "MIN_FILE_SIZE_KB": ".svc.config",
    "POLL_INTERVAL_SECONDS": ".svc.config",
    "RESP_ERROR": ".svc.config",
    "RESP_FILES_ANALYZED": ".svc.config",
    "RESP_LOG_FILE": ".svc.config",
    "RESP_LOGS_DIR": ".svc.config",
    "RESP_MODE": ".svc.config",
    "RESP_MODULE": ".svc.config",
    "RESP_RESULT": ".svc.config",
    "RESP_RESULT_ID": ".svc.config",
    "RESP_SCHED_RESTARTS": ".svc.config",
    "RESP_STATE": ".svc.config",
    "RESP_STATUS": ".svc.config",
    "RESP_WL_RESTART": ".svc.config",
    "RESP_WL_RESTART_COUNT": ".svc.config",
    "STATE_TIMEOUT": ".svc.config",
    "TTL_MAX_JOB_AGE_SECONDS": ".svc.config",
    "TTL_PENDING_SECONDS": ".svc.config",
    "TTL_TERMINATED_SECONDS": ".svc.config",
    "ErrorCode": ".svc.config",
    "FileInfo": ".svc.job",
    "Job": ".svc.job",
    "JobMode": ".svc.job",
    "SplitlogTracker": ".svc.splitlog",
    "LogAnalysisCycleResult": ".svc.types",
    "LogAnalysisSplitlogResult": ".svc.types",
    "LogAnalyzerConfig": ".svc.types",
    "LogAnalyzerError": ".svc.types",
    "LogAnalyzerFilePreview": ".svc.types",
    "LogAnalyzerOutcome": ".svc.types",
    "LogAnalyzerSubmitResult": ".svc.types",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        module = import_module(module_name, __name__)
    except ModuleNotFoundError as exc:
        reraise_if_missing_attribution_dependency(
            exc,
            feature=f"{__name__}.{name}",
        )
        raise

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))


__all__ = [
    # Log + FR orchestration (no LogSage import here)
    "AnalysisPipelineMode",
    "CombinedAnalysisResult",
    "FrDumpPathNotFoundError",
    "run_attribution_pipeline",
    # Coalescer cache payload (LogSage + optional FR)
    "LogAnalysisCoalesced",
    "coalesced_from_cache",
    # Main API
    "Analyzer",
    "TraceAnalyzer",
    "LogAnalyzerConfig",
    "LogAnalyzerError",
    "LogAnalyzerOutcome",
    "LogAnalysisCycleResult",
    "LogAnalyzerSubmitResult",
    "LogAnalysisSplitlogResult",
    "LogAnalyzerFilePreview",
    "AttributionController",
    "AttributionControllerConfig",
    "AttributionAnalysisConfig",
    "AttributionCacheConfig",
    "AttributionCredentialsConfig",
    "AttributionPostprocessingConfig",
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
    # Splitlog tracking
    "SplitlogTracker",
]
