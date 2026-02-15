# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic result posting framework (no Slack).

This module provides the framework for posting analysis results to external systems.
The implementation is entirely injected via a single callback (post_fn). There is
no Slack or other side effects here; keep proprietary or optional integrations
(e.g. dataflow, Slack) in their own modules and inject a composed post_fn if needed.

Example:
    from nvidia_resiliency_ext.attribution.postprocessing import config, ResultPoster, post_results
    config.default_poster = ResultPoster(post_fn=my_post_fn)
    post_results(parsed, metadata, log_path, ...)
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from nvidia_resiliency_ext.attribution.log_analyzer.utils import (
    JobMetadata,
    ParsedLLMResponse,
    build_dataflow_record,
)

from .config import config
from .slack import maybe_send_slack_notification

logger = logging.getLogger(__name__)


@dataclass
class DataflowStats:
    """Statistics for dataflow/posting operations."""

    total_posts: int = 0
    successful_posts: int = 0
    failed_posts: int = 0


# Type alias for post function signature
PostFunction = Callable[[Dict[str, Any], str], bool]


class ResultPoster:
    """
    Handles only the custom post_fn: consumes pre-built data and calls post_fn(data, index).
    """

    def __init__(self, post_fn: Optional[PostFunction] = None):
        """
        Initialize the result poster.

        Args:
            post_fn: Function to post data. Signature: (data: dict, index: str) -> bool
                     If None, results are logged but not posted.
        """
        self._post_fn = post_fn
        self._stats = DataflowStats()

    @property
    def stats(self) -> DataflowStats:
        """Get current posting statistics."""
        return self._stats

    def post_results(self, data: Dict[str, Any], index: str) -> bool:
        """Run the custom post_fn with pre-built data. Caller builds data and handles Slack."""
        self._stats.total_posts += 1
        if self._post_fn is None:
            logger.debug("No post function configured, skipping post")
            return True
        success = self._post_fn(data, index)
        if success:
            self._stats.successful_posts += 1
        else:
            self._stats.failed_posts += 1
        return success


def get_default_poster() -> ResultPoster:
    """Return the default poster. Creates a no-op poster if none was set (e.g. lib used without service)."""
    if config.default_poster is None:
        config.default_poster = ResultPoster()
    return config.default_poster


def get_dataflow_stats() -> DataflowStats:
    """Get current dataflow statistics from default poster."""
    return get_default_poster().stats


def post_results(
    parsed: ParsedLLMResponse,
    metadata: JobMetadata,
    log_path: str,
    processing_time: float,
    user: str = "unknown",
) -> bool:
    """Build dataflow record once; pass to default poster (custom post_fn) and to Slack. Uses config.cluster_name, config.dataflow_index, config.slack_*."""
    data = build_dataflow_record(
        parsed=parsed,
        metadata=metadata,
        log_path=log_path,
        processing_time=processing_time,
        cluster_name=config.cluster_name,
        user=user,
    )

    logger.info("jobid: %s", metadata.job_id)
    logger.info("log_path: %s", log_path)
    logger.info("auto_resume: %s", parsed.auto_resume)
    logger.info("auto_resume_explanation: %s", parsed.auto_resume_explanation)
    logger.info("attribution_text: %s", parsed.attribution_text)

    poster = get_default_poster()
    success = True
    if config.dataflow_index:
        success = poster.post_results(data, config.dataflow_index)
    maybe_send_slack_notification(data)
    return success
