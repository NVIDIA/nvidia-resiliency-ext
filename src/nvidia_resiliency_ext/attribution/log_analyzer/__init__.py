# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Log-side infrastructure: LogSage, SLURM parsing, splitlog, job model, wire keys.

Orchestration API types (:class:`~nvidia_resiliency_ext.attribution.log_analyzer.types.LogAnalyzerConfig`,
result dataclasses) live in :mod:`nvidia_resiliency_ext.attribution.log_analyzer.types`;
:class:`~nvidia_resiliency_ext.attribution.analyzer.engine.Analyzer` composes this package with request coalescing.

Also re-exports coalescing types from ``attribution.coalescing`` and postprocessing symbols.

Example:
    from nvidia_resiliency_ext.attribution.analyzer import Analyzer

    analyzer = Analyzer(allowed_root="/logs", use_lib_log_analysis=False)
    result = await analyzer.analyze("/logs/slurm-12345.out")
"""

from nvidia_resiliency_ext.attribution.coalescing import (
    DEFAULT_COMPUTE_TIMEOUT_SECONDS,
    CacheResult,
    CoalescerStats,
    ComputeStats,
    InflightResult,
    RequestCoalescer,
    StatsResult,
    SubmittedResult,
)
from nvidia_resiliency_ext.attribution.postprocessing import (
    PostFunction,
    PostingStats,
    ResultPoster,
    build_dataflow_record,
    config,
    get_default_poster,
    get_posting_stats,
    post_results,
)

from ..coalescing import LogAnalysisCoalesced
from .analysis_pipeline import (
    AnalysisPipelineMode,
    CombinedAnalysisResult,
    FrDumpPathNotFoundError,
    run_attribution_pipeline,
)
from .config import (
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
    LogSageExecutionConfig,
)
from .job import FileInfo, Job, JobMode
from .llm_output import (
    ParsedLLMResponse,
    attribution_no_restart,
    log_fields_for_dataflow_record,
    parse_llm_response,
)
from .log_analyzer import LogAnalyzer, LogSageRunner
from .log_path_metadata import (
    CYCLE_LOG_PATTERN,
    CYCLE_NUM_PATTERN,
    DATE_TIME_PATTERN,
    JOB_ID_PATTERNS,
    JobMetadata,
    extract_job_metadata,
)
from .parser_base import BaseParser, ParseResult
from .runner import ensure_analyzer_ready, notify_log_path_sync, run_log_analysis_sync
from .slurm_parser import (
    SlurmOutputInfo,
    SlurmParser,
    parse_slurm_output,
    read_and_parse_slurm_output,
)
from .splitlog import (
    DEFAULT_MAX_JOB_AGE_SECONDS,
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_TERMINATED_JOB_TTL_SECONDS,
    SplitlogTracker,
)
from .tracked_jobs import TrackedJobs
from .types import (
    LogAnalysisCycleResult,
    LogAnalysisSplitlogResult,
    LogAnalyzerConfig,
    LogAnalyzerError,
    LogAnalyzerFilePreview,
    LogAnalyzerOutcome,
    LogAnalyzerSubmitResult,
)

__all__ = [
    "LogAnalyzer",
    "LogSageRunner",
    "LogAnalyzerConfig",
    "LogAnalyzerError",
    "LogAnalyzerOutcome",
    "LogAnalysisCycleResult",
    "LogAnalysisSplitlogResult",
    "LogAnalyzerSubmitResult",
    "LogAnalyzerFilePreview",
    "LogSageExecutionConfig",
    "LogAnalysisCoalesced",
    "AnalysisPipelineMode",
    "CombinedAnalysisResult",
    "FrDumpPathNotFoundError",
    "run_attribution_pipeline",
    # Configuration and error codes
    "ErrorCode",
    "TTL_PENDING_SECONDS",
    "TTL_TERMINATED_SECONDS",
    "TTL_MAX_JOB_AGE_SECONDS",
    "POLL_INTERVAL_SECONDS",
    "DEFAULT_COMPUTE_TIMEOUT_SECONDS",
    "MAX_JOBS",
    "MIN_FILE_SIZE_KB",
    # Result/response keys (serialized result shape)
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
    "ComputeStats",
    "StatsResult",
    "CacheResult",
    "InflightResult",
    "SubmittedResult",
    # Job data model
    "Job",
    "FileInfo",
    "JobMode",
    # Scheduler parsers
    "BaseParser",
    "ParseResult",
    "SlurmParser",
    "SlurmOutputInfo",
    "parse_slurm_output",
    "read_and_parse_slurm_output",
    # Splitlog tracking
    "SplitlogTracker",
    "TrackedJobs",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "DEFAULT_TERMINATED_JOB_TTL_SECONDS",
    "DEFAULT_MAX_JOB_AGE_SECONDS",
    # Log path regexes + JobMetadata (log_path_metadata)
    "CYCLE_LOG_PATTERN",
    "CYCLE_NUM_PATTERN",
    "DATE_TIME_PATTERN",
    "JOB_ID_PATTERNS",
    "JobMetadata",
    "extract_job_metadata",
    # LLM response parsing (llm_output)
    "ParsedLLMResponse",
    "parse_llm_response",
    # Dataflow: LogSage keys (full record via postprocessing.pipeline.build_dataflow_record)
    "log_fields_for_dataflow_record",
    "build_dataflow_record",
    # Sync lib/MCP runner (e.g. FT path)
    "ensure_analyzer_ready",
    "notify_log_path_sync",
    "run_log_analysis_sync",
    # Attribution decision helper
    "attribution_no_restart",
    # Postprocessing
    "ResultPoster",
    "PostingStats",
    "PostFunction",
    "get_posting_stats",
    "get_default_poster",
    "config",
    "post_results",
]
