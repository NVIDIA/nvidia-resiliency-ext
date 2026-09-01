# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural retry-lifecycle parsing for L0 failure observations."""

from __future__ import annotations

import re

from ..models import RetryLifecycle, RetryLifecycleState

_ATTEMPT_FRACTION_RE = re.compile(
    r"\battempt\s+(?P<attempt>\d+)\s*/\s*(?P<max_attempts>\d+)\b",
    re.IGNORECASE,
)
_RETRY_SUCCEEDED_RE = re.compile(
    r"\b(?:retry succeeded|successfully retried|re-?try succeeded|recovered after (?:a )?retry)\b",
    re.IGNORECASE,
)
_RETRY_EXHAUSTED_RE = re.compile(
    r"\b(?:retries exhausted|retry attempts exhausted|all retries failed|"
    r"giving up(?: after)?|failed after \d+ attempts?|max(?:imum)? retries exceeded)\b",
    re.IGNORECASE,
)
_RETRY_PENDING_RE = re.compile(
    r"\b(?:retrying|re-?trying|will re-?try|then re-?try|retry scheduled|"
    r"trying again|sleep(?:ing)?\b[^\n]{0,80}\bre-?try)\b",
    re.IGNORECASE,
)


def classify_retry_lifecycle(text: str) -> RetryLifecycle | None:
    """Return explicit operation-retry state without inferring recoverability."""

    lowered = text.lower()
    if not any(term in lowered for term in ("attempt", "retry", "re-try", "retried", "giving up")):
        return None

    attempt_match = _ATTEMPT_FRACTION_RE.search(text)
    attempt = int(attempt_match.group("attempt")) if attempt_match is not None else None
    max_attempts = int(attempt_match.group("max_attempts")) if attempt_match is not None else None

    if _RETRY_SUCCEEDED_RE.search(text):
        state = RetryLifecycleState.SUCCEEDED
    elif _RETRY_EXHAUSTED_RE.search(text) or (
        attempt is not None and max_attempts is not None and attempt >= max_attempts
    ):
        state = RetryLifecycleState.EXHAUSTED
    elif _RETRY_PENDING_RE.search(text) or (
        attempt is not None and max_attempts is not None and attempt < max_attempts
    ):
        state = RetryLifecycleState.PENDING
    else:
        return None

    return RetryLifecycle(
        state=state,
        attempt=attempt,
        max_attempts=max_attempts,
    )


def retry_lifecycle_blocks_primary(lifecycle: RetryLifecycle | None) -> bool:
    """Return whether an observed operation retry is not terminal-qualified."""

    return lifecycle is not None and lifecycle.state in {
        RetryLifecycleState.PENDING,
        RetryLifecycleState.SUCCEEDED,
    }
