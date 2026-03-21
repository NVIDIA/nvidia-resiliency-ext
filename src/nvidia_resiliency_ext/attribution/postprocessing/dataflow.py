# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional nvdataflow integration for lib/mcp attribution posting.

When nvdataflow is installed, provides a post function for ResultPoster.
Otherwise get_nvdataflow_post_fn returns None and dataflow posting is skipped.
"""

import logging
import time
from typing import Any, Callable, Dict, Optional

try:
    from nvdataflow import post as nv_post

    logging.getLogger("nvdataflow").setLevel(logging.WARNING)
    logging.getLogger("nvdataflow.post").setLevel(logging.WARNING)
    logging.getLogger("nvdataflow.nvdataflowlog").setLevel(logging.WARNING)
    HAS_NVDATAFLOW = True
except ImportError:
    nv_post = None
    HAS_NVDATAFLOW = False

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 0.5


def _post_with_retry(data: Dict[str, Any], index: str) -> bool:
    """Post to nvdataflow with retry. Requires nvdataflow to be installed."""
    if nv_post is None:
        logger.error("nvdataflow not installed, cannot post")
        return False
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            nv_post(data=data, project=index)
            if attempt > 0:
                logger.info("dataflow post succeeded on attempt %d", attempt + 1)
            return True
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                backoff = INITIAL_BACKOFF_SECONDS * (2**attempt)
                logger.warning(
                    "dataflow post failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    MAX_RETRIES,
                    backoff,
                    e,
                )
                time.sleep(backoff)
    logger.error("failed to post to dataflow after %d attempts: %s", MAX_RETRIES, last_error)
    return False


def post(data: Dict[str, Any], index: str) -> bool:
    """
    Post data to nvdataflow/elasticsearch with retry logic.

    Callable directly for ResultPoster(post_fn=post). Returns False if nvdataflow not installed.
    """
    return _post_with_retry(data, index)


def get_nvdataflow_post_fn() -> Optional[Callable[[dict, str], bool]]:
    """Return post function for nvdataflow, or None if nvdataflow is not installed."""
    if not HAS_NVDATAFLOW:
        return None
    return _post_with_retry
