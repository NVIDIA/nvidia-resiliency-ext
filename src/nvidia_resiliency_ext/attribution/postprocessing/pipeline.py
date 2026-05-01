# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attribution result pipeline: build record, log, post to configured backend, optional Slack.

:func:`build_dataflow_record` composes the posting dict: service fields (cluster, user) plus
:func:`~nvidia_resiliency_ext.attribution.svc.llm_output.log_fields_for_dataflow_record` and
:func:`~nvidia_resiliency_ext.attribution.trace_analyzer.fr_support.fr_fields_for_dataflow_record`.

:func:`post_analysis_items` posts each cycle item (LogSage + optional FR) via :func:`post_results`.

``post_results`` is the entry point after analysis. It uses the shared :data:`~.config.config`
singleton (poster, cluster, index, Slack). The actual HTTP/ES send is injected via
:class:`ResultPoster` — use :mod:`nvidia_resiliency_ext.attribution.postprocessing.post_backend`
for the shared retrying implementation (custom override or nvdataflow).

Example:
    from nvidia_resiliency_ext.attribution.postprocessing import config, ResultPoster, post_results
    from nvidia_resiliency_ext.attribution.postprocessing import post_backend

    config.default_poster = ResultPoster(post_fn=post_backend.post)
    post_results(parsed, metadata, log_path, ...)
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from nvidia_resiliency_ext.attribution.svc.llm_output import (
    ParsedLLMResponse,
    log_fields_for_dataflow_record,
    parse_llm_response,
)
from nvidia_resiliency_ext.attribution.svc.log_path_metadata import (
    JobMetadata,
    extract_job_metadata,
)
from nvidia_resiliency_ext.attribution.svc.posting_markdown import format_posting_markdown_body
from nvidia_resiliency_ext.attribution.trace_analyzer import FRAnalysisResult
from nvidia_resiliency_ext.attribution.trace_analyzer.fr_support import (
    fr_fields_for_dataflow_record,
)

from .config import config
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
    """Thin wrapper around a single ``post_fn(data, index) -> bool``."""

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
        ``failed_posts`` (caller expected a post, e.g. ``dataflow_index`` is set but poster
        was not wired).
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
                logger.debug(
                    "post_fn not configured; skipping post (dataflow index=%r)",
                    index,
                )
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
    parsed: ParsedLLMResponse,
    metadata: JobMetadata,
    log_path: str,
    processing_time: float,
    cluster_name: str,
    user: str = "unknown",
    fr_dump_path: Optional[str] = None,
    fr_analysis: Optional[FRAnalysisResult] = None,
) -> Dict[str, Any]:
    """
    Build a dataflow record from parsed LLM results.

    This creates the data dictionary suitable for posting to dataflow/elasticsearch.
    The actual posting is left to the caller (to avoid nvdataflow dependency in library).

    Args:
        parsed: Parsed LLM response
        metadata: Job metadata from path
        log_path: Path to the log file
        processing_time: Time taken for analysis in seconds
        cluster_name: Cluster name for dataflow
        user: User identifier (default: ``"unknown"``)
        fr_dump_path: NCCL flight recorder dump path if configured
        fr_analysis: CollectiveAnalyzer result if dump was analyzed

    Returns:
        Dictionary with dataflow record fields
    """
    record: Dict[str, Any] = {
        "s_cluster": cluster_name,
        "s_user": user,
    }
    record.update(log_fields_for_dataflow_record(parsed, metadata, log_path, processing_time))
    record.update(fr_fields_for_dataflow_record(fr_dump_path, fr_analysis))
    return record


def post_results(
    parsed: ParsedLLMResponse,
    metadata: JobMetadata,
    log_path: str,
    processing_time: float,
    user: str = "unknown",
    fr_dump_path: Optional[str] = None,
    fr_analysis: Optional[FRAnalysisResult] = None,
) -> bool:
    """Build one attribution record; log; send via poster when ``config.dataflow_index`` is set; maybe Slack."""
    data = build_dataflow_record(
        parsed=parsed,
        metadata=metadata,
        log_path=log_path,
        processing_time=processing_time,
        cluster_name=config.cluster_name,
        user=user,
        fr_dump_path=fr_dump_path,
        fr_analysis=fr_analysis,
    )

    logger.info("jobid: %s", metadata.job_id)
    logger.info("log_path: %s", log_path)
    logger.info("auto_resume: %s", parsed.auto_resume)
    logger.info(
        "analysis summary:\n%s",
        format_posting_markdown_body(data),
    )

    poster = get_default_poster()
    success = True
    if config.dataflow_index:
        success = poster.send(data, config.dataflow_index)
    maybe_send_slack_notification(data)
    return success


def post_analysis_items(
    result_items: List[Any],
    processing_time: float,
    path: str,
    user: str,
    job_id: Optional[str],
    fr_dump_path: Optional[str] = None,
    fr_analysis: Optional[FRAnalysisResult] = None,
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
        # Trace-only / FR-only: no LogSage cycle rows to parse; use a minimal ParsedLLMResponse so
        # build_dataflow_record still gets structured fields while FR columns come from fr_* args.
        parsed = ParsedLLMResponse(
            auto_resume="unknown",
            auto_resume_explanation="",
            attribution_text="(trace-only analysis; no LogSage cycle items)",
            checkpoint_saved_flag=0,
        )
        post_results(
            parsed,
            metadata,
            path,
            processing_time,
            user,
            fr_dump_path=fr_dump_path,
            fr_analysis=fr_analysis,
        )
        return
    for item in result_items:
        # LogSage ``llm_analyze`` rows are 5-tuples (auto_resume, explanation, attribution line, …,
        # checkpoint). :func:`parse_llm_response` needs the same newline-joined text as
        # ``NVRxLogAnalyzer.print_output``, not ``item[0]`` alone.
        if isinstance(item, (list, tuple)):
            raw_text = "\n".join(str(x) for x in item)
        else:
            raw_text = str(item)
        parsed = parse_llm_response(raw_text)
        if job_id:
            path_metadata = extract_job_metadata(path, warn_on_missing_job_id=False)
            metadata = JobMetadata(job_id=job_id, cycle_id=path_metadata.cycle_id)
        else:
            metadata = extract_job_metadata(path)
        post_results(
            parsed,
            metadata,
            path,
            processing_time,
            user,
            fr_dump_path=fr_dump_path,
            fr_analysis=fr_analysis,
        )
