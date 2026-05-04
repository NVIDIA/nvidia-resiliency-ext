# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and result types for log-side analysis and the unified attribution API.

Compute/cache timing (:class:`~nvidia_resiliency_ext.attribution.coalescing.RequestCoalescer`)
is configured on :class:`~nvidia_resiliency_ext.attribution.analyzer.engine.Analyzer`, not
``LogAnalyzerConfig``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from .analysis_pipeline import AnalysisPipelineMode
from .config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, ErrorCode

if TYPE_CHECKING:
    from .config import LogSageExecutionConfig

from nvidia_resiliency_ext.attribution.trace_analyzer import FRAnalysisResult

from .job import JobMode

RECOMMENDATION_STOP = "STOP"
RECOMMENDATION_RESTART = "RESTART"
RECOMMENDATION_CONTINUE = "CONTINUE"
RECOMMENDATION_UNKNOWN = "UNKNOWN"
RECOMMENDATION_TIMEOUT = "TIMEOUT"


@dataclass
class LogAnalyzerConfig:
    """Bundled settings shape (e.g. docs, tests, or HTTP settings aggregation).

    :class:`~nvidia_resiliency_ext.attribution.analyzer.engine.Analyzer` takes ``allowed_root``,
    optional ``log_sage`` (:class:`~nvidia_resiliency_ext.attribution.orchestration.config.LogSageExecutionConfig`),
    coalescer timing, etc. — not this dataclass as a single ``config=`` argument. Use
    :meth:`log_sage_execution` to build ``log_sage`` from the LLM / lib-vs-MCP fields here; use a
    custom :class:`~nvidia_resiliency_ext.attribution.orchestration.log_analyzer.LogAnalyzer` if you
    need ``analysis_pipeline_mode`` from this bundle.
    """

    allowed_root: str
    llm_model: str = DEFAULT_LLM_MODEL
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_max_tokens: int = 8192
    use_lib_log_analysis: bool = False
    #: How LogSage and NCCL flight-recorder analysis are combined; see
    #: :class:`~nvidia_resiliency_ext.attribution.orchestration.analysis_pipeline.AnalysisPipelineMode`.
    analysis_pipeline_mode: AnalysisPipelineMode = AnalysisPipelineMode.LOG_AND_TRACE

    def __post_init__(self) -> None:
        if not self.allowed_root:
            raise ValueError("allowed_root is required")
        if not os.path.isabs(self.allowed_root):
            raise ValueError("allowed_root must be an absolute path")

    def log_sage_execution(self) -> LogSageExecutionConfig:
        """Settings for :class:`~nvidia_resiliency_ext.attribution.orchestration.log_analyzer.LogAnalyzer` (lib/MCP)."""
        from .config import LogSageExecutionConfig

        return LogSageExecutionConfig(
            use_lib_log_analysis=self.use_lib_log_analysis,
            llm_model=self.llm_model,
            llm_base_url=self.llm_base_url,
            llm_temperature=self.llm_temperature,
            llm_top_p=self.llm_top_p,
            llm_max_tokens=self.llm_max_tokens,
        )


@dataclass
class LogAnalyzerError:
    """Error result from :class:`~nvidia_resiliency_ext.attribution.analyzer.engine.Analyzer` operations."""

    error_code: ErrorCode
    message: str


@dataclass
class AttributionRecommendation:
    """Normalized client-facing restart/stop recommendation.

    ``UNKNOWN`` means attribution completed without a usable action, for example
    missing results, FR-only ``no_log`` output, invalid recommendation payloads,
    or an unrecognized backend result shape.
    """

    action: str = RECOMMENDATION_UNKNOWN
    reason: str = ""
    source: str = ""


@dataclass
class LogAnalysisCycleResult:
    """Single-file mode: one row per workload cycle (``wl_restart``) after orchestrated analysis."""

    result: Dict[str, Any]
    status: str = "completed"
    wl_restart: int = 0
    wl_restart_count: Optional[int] = None
    fr_dump_path: Optional[str] = None
    fr_analysis: Optional[FRAnalysisResult] = None
    llm_merged_summary: Optional[str] = None
    recommendation: AttributionRecommendation = field(default_factory=AttributionRecommendation)


@dataclass
class LogAnalyzerSubmitResult:
    """Outcome of submitting a log path for job tracking."""

    submitted: bool
    normalized_path: str
    mode: str = JobMode.SINGLE.value
    logs_dir: Optional[str] = None
    sched_restarts: int = 0
    files_analyzed: int = 0


@dataclass
class LogAnalysisSplitlogResult:
    """Split-log mode: aggregated result for a tracked job with ``LOGS_DIR``."""

    result: Dict[str, Any]
    status: str = "completed"
    mode: str = JobMode.SPLITLOG.value
    sched_restarts: int = 0
    log_file: str = ""
    wl_restart: int = 0
    fr_dump_path: Optional[str] = None
    fr_analysis: Optional[FRAnalysisResult] = None
    llm_merged_summary: Optional[str] = None
    recommendation: AttributionRecommendation = field(default_factory=AttributionRecommendation)


@dataclass
class LogAnalyzerFilePreview:
    """First-chunk file preview (e.g. HTTP ``GET`` print-style APIs)."""

    content: str
    path: str


LogAnalyzerOutcome = Union[LogAnalysisCycleResult, LogAnalysisSplitlogResult, LogAnalyzerError]
