# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small logging helpers for attribution payload previews."""

from __future__ import annotations

from typing import Any

LOG_VALUE_PREVIEW_CHARS = 16_384


def bounded_log_value(value: Any, *, limit: int = LOG_VALUE_PREVIEW_CHARS) -> str:
    """Return a bounded string preview for logs that may contain user/job data."""
    if limit <= 0:
        return ""
    try:
        text = str(value)
    except Exception as e:
        return f"<unprintable {type(value).__name__}: {e}>"
    if len(text) <= limit:
        return text
    suffix = f"... [truncated; original {len(text)} chars]"
    preview_len = max(0, limit - len(suffix))
    return f"{text[:preview_len]}{suffix}"
