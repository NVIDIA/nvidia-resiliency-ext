#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Attribution module for failure attribution and log analysis.

This module provides:
- LogAnalyzer: Main API for analyzing logs (usable without HTTP)
- Core configuration and error codes
- Request coalescing for deduplication and caching
- Job/FileInfo data model for tracking analysis state
- SLURM parser for extracting metadata from scheduler output
- SplitlogTracker for multi-file job tracking

Example usage (no HTTP required):
    from nvidia_resiliency_ext.attribution import LogAnalyzer, AnalyzerConfig
    
    config = AnalyzerConfig(allowed_root="/logs")
    analyzer = LogAnalyzer(config)
    
    # Submit and analyze
    await analyzer.submit("/logs/slurm-12345.out", user="alice")
    result = await analyzer.analyze("/logs/slurm-12345.out")
    
    analyzer.shutdown()

See nvrx_attrsvc_spec.md for architecture overview.
"""

# Re-export from log_analyzer submodule
from .log_analyzer import (  # Main API; Infrastructure
    DEFAULT_COMPUTE_TIMEOUT_SECONDS,
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
    AnalysisResult,
    AnalyzerConfig,
    AnalyzerError,
    AnalyzerResult,
    CacheResult,
    CoalescerStats,
    ErrorCode,
    FileInfo,
    FilePreviewResult,
    InflightResult,
    Job,
    JobMode,
    LogAnalyzer,
    RequestCoalescer,
    SplitlogAnalysisResult,
    SplitlogTracker,
    StatsResult,
    SubmitResult,
    SubmittedResult,
)

__all__ = [
    # Main API
    "LogAnalyzer",
    "AnalyzerConfig",
    "AnalyzerError",
    "AnalyzerResult",
    "AnalysisResult",
    "SubmitResult",
    "SplitlogAnalysisResult",
    "FilePreviewResult",
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
