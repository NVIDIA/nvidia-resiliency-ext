#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""nvdataflow integration for nvrx_attrsvc."""

import logging
import time
from typing import Any

try:
    from nvdataflow import post as nv_post

    # Silence verbose nvdataflow INFO logs (they're debug-level messages)
    logging.getLogger("nvdataflow").setLevel(logging.WARNING)
    logging.getLogger("nvdataflow.post").setLevel(logging.WARNING)
    logging.getLogger("nvdataflow.nvdataflowlog").setLevel(logging.WARNING)
except ImportError:
    nv_post = None

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 0.5  # 0.5s, 1s, 2s


def post(data: dict[str, Any], index: str) -> bool:
    """
    Post data to nvdataflow/elasticsearch with retry logic.

    Uses exponential backoff: 0.5s, 1s, 2s between retries.

    Args:
        data: Data dictionary to post
        index: Dataflow/elasticsearch index name

    Returns:
        True if posted successfully, False otherwise
    """
    if nv_post is None:
        logger.error("can't import nvdataflow")
        return False

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            nv_post(data=data, project=index)
            if attempt > 0:
                logger.info(f"dataflow post succeeded on attempt {attempt + 1}")
            return True
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                backoff = INITIAL_BACKOFF_SECONDS * (2**attempt)
                logger.warning(
                    f"dataflow post failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {backoff}s: {e}"
                )
                time.sleep(backoff)

    logger.error(f"failed to post to dataflow after {MAX_RETRIES} attempts: {last_error}")
    return False
