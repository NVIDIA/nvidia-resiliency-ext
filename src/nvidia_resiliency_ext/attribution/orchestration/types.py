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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .analysis_pipeline import AnalysisPipelineMode
from .config import ErrorCode

if TYPE_CHECKING:
    from .config import LogSageExecutionConfig

from nvidia_resiliency_ext.attribution.trace_analyzer import FRAnalysisResult

from .job import JobMode

RECOMMENDATION_STOP = "STOP"
RECOMMENDATION_RESTART = "RESTART"
RECOMMENDATION_CONTINUE = "CONTINUE"
RECOMMENDATION_UNKNOWN = "UNKNOWN"
RECOMMENDATION_TIMEOUT = "TIMEOUT"

RECOMMENDATION_ACTIONS = (
    RECOMMENDATION_STOP,
    RECOMMENDATION_RESTART,
    RECOMMENDATION_CONTINUE,
    RECOMMENDATION_UNKNOWN,
    RECOMMENDATION_TIMEOUT,
)
RECOMMENDATION_PAYLOAD_FIELDS = ("action", "source")
RAW_ANALYSIS_RESULT_ITEM_PAYLOAD_FIELDS = (
    "raw_text",
    "auto_resume",
    "auto_resume_explanation",
    "attribution_text",
    "checkpoint_saved_flag",
    "action",
    "primary_issues",
    "secondary_issues",
)

_VALID_RECOMMENDATION_ACTIONS = set(RECOMMENDATION_ACTIONS)


def _normalized_state_name(value: Any) -> str:
    if value is None:
        return ""
    state = getattr(value, "name", None)
    if state is None:
        state = str(value)
    if "." in state:
        state = state.rsplit(".", 1)[-1]
    return state.strip()


def normalize_recommendation_action(value: Any) -> str:
    """Normalize a client-facing recommendation action."""
    action = _normalized_state_name(value).upper().replace(" ", "_")
    return action if action in _VALID_RECOMMENDATION_ACTIONS else RECOMMENDATION_UNKNOWN


@dataclass(frozen=True)
class RawAnalysisResultItem:
    """Single attribution result item with raw text and parsed LogSage fields."""

    raw_text: str
    auto_resume: str
    auto_resume_explanation: str
    attribution_text: str
    checkpoint_saved_flag: int
    action: str
    primary_issues: List[str] = field(default_factory=list)
    secondary_issues: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_text", str(self.raw_text))
        object.__setattr__(self, "auto_resume", str(self.auto_resume))
        object.__setattr__(self, "auto_resume_explanation", str(self.auto_resume_explanation))
        object.__setattr__(self, "attribution_text", str(self.attribution_text))
        checkpoint_saved_flag = 1 if self.checkpoint_saved_flag else 0
        object.__setattr__(self, "checkpoint_saved_flag", checkpoint_saved_flag)
        object.__setattr__(self, "action", normalize_recommendation_action(self.action))
        object.__setattr__(self, "primary_issues", _string_list(self.primary_issues))
        object.__setattr__(self, "secondary_issues", _string_list(self.secondary_issues))

    @classmethod
    def from_payload(cls, value: Any) -> "RawAnalysisResultItem":
        """Build from the canonical in-process or JSON payload shape."""
        if isinstance(value, cls):
            return value
        if isinstance(value, dict) and "raw_text" in value:
            missing = sorted(set(RAW_ANALYSIS_RESULT_ITEM_PAYLOAD_FIELDS) - value.keys())
            if missing:
                raise TypeError(
                    "raw analysis result item payload missing required fields: "
                    + ", ".join(missing)
                )
            action = normalize_recommendation_action(value.get("action"))
            return cls(
                raw_text=str(value.get("raw_text")),
                auto_resume=str(value.get("auto_resume")),
                auto_resume_explanation=str(value.get("auto_resume_explanation")),
                attribution_text=str(value.get("attribution_text")),
                checkpoint_saved_flag=1 if value.get("checkpoint_saved_flag") else 0,
                action=action,
                primary_issues=_string_list(value.get("primary_issues")),
                secondary_issues=_string_list(value.get("secondary_issues")),
            )
        raise TypeError(
            "raw analysis result item must be RawAnalysisResultItem or "
            f"dict with structured LogSage fields, not {type(value).__name__}"
        )

    def to_payload(self) -> Dict[str, Any]:
        """Return the canonical JSON payload shape."""
        return {
            field_name: getattr(self, field_name)
            for field_name in RAW_ANALYSIS_RESULT_ITEM_PAYLOAD_FIELDS
        }


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _string_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


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
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_temperature: Optional[float] = None
    llm_top_p: Optional[float] = None
    llm_max_tokens: Optional[int] = None
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
    """Normalized restart/stop recommendation.

    ``action`` is the client-facing policy decision. LogSage-specific parsing
    details stay at the source; downstream code should not re-derive decisions
    from raw text.

    ``UNKNOWN`` means attribution completed without a usable action, for example
    missing results, FR-only ``no_log`` output, invalid recommendation payloads,
    or an unrecognized backend result shape.
    """

    action: str = RECOMMENDATION_UNKNOWN
    reason: str = ""
    source: str = ""

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        source: str = "",
    ) -> Optional["AttributionRecommendation"]:
        """Build from the standard serialized recommendation payload."""
        if not isinstance(payload, dict):
            return None
        return cls(
            action=normalize_recommendation_action(payload.get("action")),
            reason=_string_value(payload.get("reason")),
            source=_string_value(payload.get("source")) or source,
        )


@dataclass
class LogSageAnalysisResult:
    """Structured LogSage handoff: parsed cycle items plus the derived action."""

    items: List[RawAnalysisResultItem] = field(default_factory=list)
    recommendation: AttributionRecommendation = field(default_factory=AttributionRecommendation)

    def __post_init__(self) -> None:
        self.items = [RawAnalysisResultItem.from_payload(item) for item in self.items]


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
    recommendation: Dict[str, str] = field(
        default_factory=lambda: {"action": RECOMMENDATION_UNKNOWN, "source": ""}
    )


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
    recommendation: Dict[str, str] = field(
        default_factory=lambda: {"action": RECOMMENDATION_UNKNOWN, "source": ""}
    )


@dataclass
class LogAnalyzerFilePreview:
    """First-chunk file preview (e.g. HTTP ``GET`` print-style APIs)."""

    content: str
    path: str


LogAnalyzerOutcome = Union[LogAnalysisCycleResult, LogAnalysisSplitlogResult, LogAnalyzerError]
