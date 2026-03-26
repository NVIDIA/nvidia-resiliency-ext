# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable attribution post path with retries.

Each attempt uses the first available backend:

1. **Override** — :func:`set_post_override` (same signature as :class:`~.pipeline.ResultPoster` ``post_fn``).
2. **nvdataflow** — ``nvdataflow.post`` when that package is installed (typical Elasticsearch path).
   Its return value is interpreted by :func:`_nvdataflow_result_ok` (``bool``, ``None`` = success,
   ``int``: ``0`` or HTTP-style ``200`` / ``201`` = success). A ``False`` result (or other int mapped to
   failure) is **not** retried — it is treated as a definitive soft failure. Only **exceptions**
   from the backend trigger backoff retries.

If neither applies, :func:`post` returns ``False`` and :func:`get_retrying_post_fn` returns ``None``.

``nvrx_attrsvc`` imports :func:`post` through a small local shim for :class:`~.pipeline.ResultPoster`.
"""

import logging
import time
from typing import Any, Callable, Dict, Optional

try:
    from nvdataflow import post as _nvdataflow_post

    logging.getLogger("nvdataflow").setLevel(logging.WARNING)
    logging.getLogger("nvdataflow.post").setLevel(logging.WARNING)
    logging.getLogger("nvdataflow.nvdataflowlog").setLevel(logging.WARNING)
    HAS_NVDATAFLOW = True
except ImportError:
    _nvdataflow_post = None
    HAS_NVDATAFLOW = False

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 0.5

_post_override: Optional[Callable[[Dict[str, Any], str], bool]] = None

# nvdataflow may return an HTTP status (e.g. 201 Created) or 0 for legacy success.
_NVDATAFLOW_SUCCESS_INTS = frozenset({0, 200, 201})


def _nvdataflow_result_ok(result: Any) -> bool:
    """Map ``nvdataflow.post`` return value to success (exceptions are not passed here)."""
    if isinstance(result, bool):
        return result
    if result is None:
        return True
    if isinstance(result, int):
        return result in _NVDATAFLOW_SUCCESS_INTS
    logger.warning(
        "nvdataflow.post returned unexpected type %s (%r); treating as failure",
        type(result).__name__,
        result,
    )
    return False


def _nvdataflow_failure_reason(result: Any) -> str:
    """Human-readable reason for a failed ``nvdataflow.post`` return (after :func:`_nvdataflow_result_ok` is False)."""
    if result is False:
        return (
            "nvdataflow.post returned False — often ES/index/auth or payload rejection; "
            "enable DEBUG on loggers nvdataflow.post or nvdataflow for details"
        )
    if isinstance(result, int):
        return (
            f"nvdataflow.post returned code {result} "
            f"(success codes: {sorted(_NVDATAFLOW_SUCCESS_INTS)})"
        )
    return f"nvdataflow.post returned {type(result).__name__} = {result!r}"


def set_post_override(fn: Optional[Callable[[Dict[str, Any], str], bool]]) -> None:
    """Register a custom ``(data, index) -> bool`` poster, or ``None`` to use only nvdataflow when installed."""
    global _post_override
    _post_override = fn


def _attempt_once(data: Dict[str, Any], index: str) -> tuple[bool, Optional[str]]:
    """Return ``(success, None)`` or ``(False, detail)`` for logging on soft failure."""
    if _post_override is not None:
        ok = _post_override(data, index)
        if ok:
            return True, None
        return False, "custom post_fn returned False"
    if _nvdataflow_post is not None:
        raw = _nvdataflow_post(data=data, project=index)
        if _nvdataflow_result_ok(raw):
            return True, None
        return False, _nvdataflow_failure_reason(raw)
    return False, "no post backend reachable"


def _post_with_retries(data: Dict[str, Any], index: str) -> bool:
    if _post_override is None and _nvdataflow_post is None:
        logger.error("no attribution post backend (set_post_override or install nvdataflow)")
        return False
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            ok, soft_detail = _attempt_once(data, index)
            if ok:
                if attempt > 0:
                    logger.info("attribution post succeeded on attempt %d", attempt + 1)
                return True
            # nvdataflow/override returned False (or mapped a non-zero code to False): application-level
            # failure, not a transient error — do not retry (avoids triple-posting on soft failures).
            logger.warning(
                "attribution post failed without exception (not retried): %s (dataflow index=%r, attempt %d)",
                soft_detail or "unknown reason",
                index,
                attempt + 1,
            )
            return False
        except Exception as e:
            last_error = e
        if attempt < MAX_RETRIES - 1:
            backoff = INITIAL_BACKOFF_SECONDS * (2**attempt)
            logger.warning(
                "attribution post failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1,
                MAX_RETRIES,
                backoff,
                last_error,
            )
            time.sleep(backoff)
    if last_error is not None:
        logger.error(
            "failed to post after %d attempts: %s",
            MAX_RETRIES,
            last_error,
            exc_info=(type(last_error), last_error, last_error.__traceback__),
        )
    else:
        logger.error("failed to post after %d attempts (no exception captured)", MAX_RETRIES)
    return False


def post(data: Dict[str, Any], index: str) -> bool:
    """Post with retries (override, else nvdataflow). For ``ResultPoster(post_fn=post)``."""
    return _post_with_retries(data, index)


def get_retrying_post_fn() -> Optional[Callable[[dict, str], bool]]:
    """Return the shared retrying post callable, or ``None`` if no backend exists."""
    if _post_override is not None or HAS_NVDATAFLOW:
        return _post_with_retries
    return None
