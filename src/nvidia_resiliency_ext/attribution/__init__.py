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
"""

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

# Re-export from svc submodule (jobs, splitlog, wire keys, config — not orchestration types)
from .svc.config import (  # Infrastructure
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
