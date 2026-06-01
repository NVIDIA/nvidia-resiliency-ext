# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attribution result pipeline: build record, log, post to configured backend, optional Slack.

:func:`build_dataflow_record` composes the posting dict: service fields (cluster, user) plus
:func:`~nvidia_resiliency_ext.attribution.orchestration.llm_output.log_fields_for_dataflow_record` and
:func:`~nvidia_resiliency_ext.attribution.trace_analyzer.fr_support.fr_fields_for_dataflow_record`.

:func:`post_analysis_items` posts each cycle item (LogSage + optional FR) via :func:`post_results`.

``post_results`` is the entry point after analysis. It uses the shared :data:`~.config.config`
singleton (poster, cluster, Slack). The actual direct dataflow HTTP send is injected via
:class:`ResultPoster` — use :mod:`nvidia_resiliency_ext.attribution.postprocessing.post_backend`
for the shared retrying implementation.

Example:
    from nvidia_resiliency_ext.attribution.postprocessing import config, ResultPoster, post_results
    from nvidia_resiliency_ext.attribution.postprocessing import post_backend

    config.default_poster = ResultPoster(post_fn=post_backend.post)
    post_results(item, metadata, log_path, ...)
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from nvidia_resiliency_ext.attribution.logging_utils import bounded_log_value
from nvidia_resiliency_ext.attribution.orchestration.llm_output import (
    log_fields_for_dataflow_record,
)
from nvidia_resiliency_ext.attribution.orchestration.log_path_metadata import (
    JobMetadata,
    extract_job_metadata,
)
from nvidia_resiliency_ext.attribution.orchestration.posting_markdown import (
    format_posting_markdown_body,
)
from nvidia_resiliency_ext.attribution.orchestration.types import (
    RECOMMENDATION_UNKNOWN,
    AttributionRecommendation,
    RawAnalysisResultItem,
)
from nvidia_resiliency_ext.attribution.trace_analyzer import FRAnalysisResult
from nvidia_resiliency_ext.attribution.trace_analyzer.fr_support import (
    fr_fields_for_dataflow_record,
)

from .config import config, dataflow_posting_enabled
from .slack import maybe_send_slack_notification

logger = logging.getLogger(__name__)


@dataclass
class PostingStats:
    """Counters for posts attempted through the default :class:`ResultPoster`.

    A call to :meth:`ResultPoster.send` always increments ``total_posts``.
    ``successful_posts`` / ``failed_posts`` reflect whether data was actually delivered
    (missing ``post_fn`` counts as failed).
    """

    total_posts: int = 0
    successful_posts: int = 0
    failed_posts: int = 0


PostFunction = Callable[[Dict[str, Any], str], bool]


class ResultPoster:
    """Thin wrapper around a single ``post_fn(data, label) -> bool``.

    The second argument is retained for custom posters; the built-in direct
    dataflow HTTP path ignores it because the destination is the configured URL.
    """

    def __init__(self, post_fn: Optional[PostFunction] = None):
        self._post_fn = post_fn
        self._stats = PostingStats()
        self._missing_post_fn_warned = False

    @property
    def stats(self) -> PostingStats:
        return self._stats

    def send(self, data: Dict[str, Any], index: str) -> bool:
        """Invoke ``post_fn`` once; update :attr:`stats`.

        If ``post_fn`` is ``None``, nothing is sent: returns ``False`` and increments
        ``failed_posts``.
        """
        self._stats.total_posts += 1
        if self._post_fn is None:
            self._stats.failed_posts += 1
            if not self._missing_post_fn_warned:
                logger.warning(
                    "post_fn is not configured; attribution posts will fail until "
                    "a poster is wired (e.g. configure(default_poster=ResultPoster(post_fn=...))). "
                    "Further failures are logged at DEBUG; see posting stats."
                )
                self._missing_post_fn_warned = True
            else:
                logger.debug("post_fn not configured; skipping post")
            return False
        success = self._post_fn(data, index)
        if success:
            self._stats.successful_posts += 1
        else:
            self._stats.failed_posts += 1
        return success


def get_default_poster() -> ResultPoster:
    """Return ``config.default_poster``, creating a no-op poster if unset."""
    if config.default_poster is None:
        config.default_poster = ResultPoster()
    return config.default_poster


def get_posting_stats() -> PostingStats:
    """Posting counters from the default poster."""
    return get_default_poster().stats


def build_dataflow_record(
    item: RawAnalysisResultItem,
    metadata: JobMetadata,
    log_path: str,
    attribution_analysis_duration_seconds: float,
    cluster_name: str,
    user: str = "unknown",
    fr_dump_path: Optional[str] = None,
    fr_analysis: Optional[FRAnalysisResult] = None,
    attribution_analysis_completed_ms: Optional[int] = None,
    recommendation: Any = None,
) -> Dict[str, Any]:
    """
    Build a dataflow record from a structured LogSage result item plus recommendation envelope.

    This creates the data dictionary suitable for posting to dataflow.
    The actual posting is left to the caller.

    Args:
        item: Structured LogSage result item
        metadata: Job metadata from path
        log_path: Path to the log file
        attribution_analysis_duration_seconds: Time taken for analysis in seconds
        cluster_name: Cluster name for dataflow
        user: User identifier (default: ``"unknown"``)
        fr_dump_path: NCCL flight recorder dump path if configured
        fr_analysis: CollectiveAnalyzer result if dump was analyzed
        attribution_analysis_completed_ms: Epoch milliseconds when analysis completed
        recommendation: Normalized ``{"action": ..., "source": ...}`` envelope from the
            analyzer result. Posting and Slack use this instead of re-deriving a
            decision from LogSage item fields.

    Returns:
        Dictionary with dataflow record fields
    """
    normalized_recommendation = _recommendation_from_payload(recommendation)
    record: Dict[str, Any] = {
        "s_cluster": cluster_name,
        "s_user": user,
        "s_recommendation_action": normalized_recommendation.action,
        "s_recommendation_source": normalized_recommendation.source,
    }
    record.update(
        log_fields_for_dataflow_record(
            item,
            metadata,
            log_path,
            attribution_analysis_duration_seconds,
            attribution_analysis_completed_ms,
        )
    )
    record.update(fr_fields_for_dataflow_record(fr_dump_path, fr_analysis))
    return record


def post_results(
    item: RawAnalysisResultItem,
    metadata: JobMetadata,
    log_path: str,
    attribution_analysis_duration_seconds: float,
    user: str = "unknown",
    fr_dump_path: Optional[str] = None,
    fr_analysis: Optional[FRAnalysisResult] = None,
    attribution_analysis_completed_ms: Optional[int] = None,
    recommendation: Any = None,
) -> bool:
    """Build one attribution record; log; send when dataflow posting is configured; maybe Slack."""
    data = build_dataflow_record(
        item=item,
        metadata=metadata,
        log_path=log_path,
        attribution_analysis_duration_seconds=attribution_analysis_duration_seconds,
        cluster_name=config.cluster_name,
        user=user,
        fr_dump_path=fr_dump_path,
        fr_analysis=fr_analysis,
        attribution_analysis_completed_ms=attribution_analysis_completed_ms,
        recommendation=recommendation,
    )

    logger.info("jobid: %s", metadata.job_id)
    logger.info("log_path: %s", log_path)
    logger.info("auto_resume: %s", data["s_auto_resume"])
    logger.info(
        "attribution timing: duration_seconds=%.2f completed_at_ms=%s",
        data["d_attribution_analysis_duration_seconds"],
        data["ts_attribution_analysis_completed_ms"],
    )
    logger.info(
        "analysis summary:\n%s",
        bounded_log_value(format_posting_markdown_body(data)),
    )

    poster = get_default_poster()
    success = True
    if dataflow_posting_enabled():
        success = poster.send(data, "")
    maybe_send_slack_notification(data)
    return success


def post_analysis_items(
    result_items: List[Any],
    attribution_analysis_duration_seconds: float,
    path: str,
    user: str,
    job_id: Optional[str],
    fr_dump_path: Optional[str] = None,
    fr_analysis: Optional[FRAnalysisResult] = None,
    attribution_analysis_completed_ms: Optional[int] = None,
    recommendation: Any = None,
) -> None:
    """Call :func:`post_results` once per cycle item (LogSage + optional FR context)."""
    if not result_items:
        if fr_dump_path is None and fr_analysis is None:
            return
        if job_id:
            path_metadata = extract_job_metadata(path, warn_on_missing_job_id=False)
            metadata = JobMetadata(job_id=job_id, cycle_id=path_metadata.cycle_id)
        else:
            metadata = extract_job_metadata(path)
        # Trace-only / FR-only: no LogSage cycle rows; synthesize the canonical item
        # while FR columns come from fr_* args.
        result_item = RawAnalysisResultItem(
            raw_text="(trace-only analysis; no LogSage cycle items)",
            auto_resume="unknown",
            auto_resume_explanation="",
            attribution_text="(trace-only analysis; no LogSage cycle items)",
            checkpoint_saved_flag=0,
            action=RECOMMENDATION_UNKNOWN,
        )
        post_results(
            result_item,
            metadata,
            path,
            attribution_analysis_duration_seconds,
            user,
            fr_dump_path=fr_dump_path,
            fr_analysis=fr_analysis,
            attribution_analysis_completed_ms=attribution_analysis_completed_ms,
            recommendation=recommendation,
        )
        return
    for item in result_items:
        result_item = RawAnalysisResultItem.from_payload(item)
        if job_id:
            path_metadata = extract_job_metadata(path, warn_on_missing_job_id=False)
            metadata = JobMetadata(job_id=job_id, cycle_id=path_metadata.cycle_id)
        else:
            metadata = extract_job_metadata(path)
        post_results(
            result_item,
            metadata,
            path,
            attribution_analysis_duration_seconds,
            user,
            fr_dump_path=fr_dump_path,
            fr_analysis=fr_analysis,
            attribution_analysis_completed_ms=attribution_analysis_completed_ms,
            recommendation=recommendation,
        )


def _recommendation_from_payload(value: Any) -> AttributionRecommendation:
    if isinstance(value, AttributionRecommendation):
        return value
    return AttributionRecommendation.from_payload(value) or AttributionRecommendation(
        action=RECOMMENDATION_UNKNOWN
    )
