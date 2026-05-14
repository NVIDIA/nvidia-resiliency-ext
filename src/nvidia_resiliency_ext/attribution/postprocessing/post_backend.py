# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable attribution post path with retries.

Each attempt uses either:

1. **Override** — :func:`set_post_override` (same signature as
   :class:`~.pipeline.ResultPoster` ``post_fn``).
2. **HTTP** — direct POST to the dataflow ingestion endpoint using ``httpx``.

``nvidia_resiliency_ext.services.attrsvc`` imports :func:`post` through a small local shim for :class:`~.pipeline.ResultPoster`.
"""

import logging
import os
import time
from typing import Any, Callable, Dict, Optional

import httpx

from .config import EXPORT_URL_ENV, export_url_from_env

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 0.5
DEFAULT_DATAFLOW_HTTP_TIMEOUT_SECONDS = 10.0

_post_override: Optional[Callable[[Dict[str, Any], str], bool]] = None
_default_http_post_fn: Optional[Callable[[Dict[str, Any], str], bool]] = None

_DATAFLOW_HTTP_RETRY_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


def _dataflow_http_timeout_seconds() -> float:
    raw = os.getenv("NVRX_ATTRSVC_DATAFLOW_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_DATAFLOW_HTTP_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except ValueError:
        logger.warning(
            "invalid dataflow HTTP timeout %r; using %.1fs",
            raw,
            DEFAULT_DATAFLOW_HTTP_TIMEOUT_SECONDS,
        )
        return DEFAULT_DATAFLOW_HTTP_TIMEOUT_SECONDS
    if timeout <= 0:
        logger.warning(
            "invalid dataflow HTTP timeout %.3f; using %.1fs",
            timeout,
            DEFAULT_DATAFLOW_HTTP_TIMEOUT_SECONDS,
        )
        return DEFAULT_DATAFLOW_HTTP_TIMEOUT_SECONDS
    return timeout


def _dataflow_http_endpoint() -> str:
    return export_url_from_env()


def _dataflow_http_queue() -> str:
    return os.getenv("NVRX_ATTRSVC_DATAFLOW_QUEUE", "").strip()


def _dataflow_http_url(endpoint: str) -> str:
    """Return the explicitly configured export URL for dataflow HTTP posts."""
    endpoint = endpoint.strip()
    if not endpoint:
        raise ValueError("NVRX_ATTRSVC_EXPORT_URL is required for dataflow HTTP posting")
    return endpoint


def make_dataflow_http_post_fn(
    *,
    endpoint: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    queue: Optional[str] = None,
) -> Callable[[Dict[str, Any], str], bool]:
    """Build a dependency-light dataflow HTTP poster.

    ``endpoint`` must be the full destination URI to post to.
    """
    resolved_endpoint = endpoint if endpoint is not None else _dataflow_http_endpoint()
    resolved_timeout = (
        timeout_seconds if timeout_seconds is not None else _dataflow_http_timeout_seconds()
    )
    resolved_queue = queue if queue is not None else _dataflow_http_queue()
    if not resolved_endpoint.strip():

        def _missing_endpoint_post(_data: Dict[str, Any], _unused: str) -> bool:
            logger.error("%s is required for dataflow HTTP posting", EXPORT_URL_ENV)
            return False

        return _missing_endpoint_post

    def _post(data: Dict[str, Any], _unused: str) -> bool:
        url = _dataflow_http_url(resolved_endpoint)
        payload = dict(data)
        params = {"queue": resolved_queue} if resolved_queue else None
        response = httpx.post(url, json=payload, params=params, timeout=resolved_timeout)
        if 200 <= response.status_code < 300:
            return True
        detail = response.text[:500]
        message = f"dataflow HTTP post returned {response.status_code} for {url}: {detail}"
        if response.status_code in _DATAFLOW_HTTP_RETRY_STATUS_CODES:
            raise RuntimeError(message)
        logger.warning(message)
        return False

    return _post


def _get_default_http_post_fn() -> Callable[[Dict[str, Any], str], bool]:
    global _default_http_post_fn
    if _default_http_post_fn is None:
        _default_http_post_fn = make_dataflow_http_post_fn()
    return _default_http_post_fn


def set_post_override(fn: Optional[Callable[[Dict[str, Any], str], bool]]) -> None:
    """Register a custom ``(data, label) -> bool`` poster, or ``None`` to use direct HTTP."""
    global _post_override
    _post_override = fn


def _attempt_once(data: Dict[str, Any], index: str) -> tuple[bool, Optional[str]]:
    """Return ``(success, None)`` or ``(False, detail)`` for logging on soft failure."""
    if _post_override is not None:
        ok = _post_override(data, index)
        if ok:
            return True, None
        return False, "custom post_fn returned False"
    ok = _get_default_http_post_fn()(data, index)
    if ok:
        return True, None
    return False, "dataflow HTTP post returned a non-success status"


def _post_with_retries(data: Dict[str, Any], index: str) -> bool:
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            ok, soft_detail = _attempt_once(data, index)
            if ok:
                if attempt > 0:
                    logger.info("attribution post succeeded on attempt %d", attempt + 1)
                return True
            # The poster returned False: application-level failure, not a transient error.
            # Do not retry; this avoids duplicate posts on definitive rejections.
            logger.warning(
                "attribution post failed without exception (not retried): %s (post label=%r, attempt %d)",
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
    """Post with retries (override, else direct HTTP). For ``ResultPoster(post_fn=post)``."""
    return _post_with_retries(data, index)


def get_retrying_post_fn() -> Optional[Callable[[dict, str], bool]]:
    """Return the shared retrying post callable."""
    return _post_with_retries
