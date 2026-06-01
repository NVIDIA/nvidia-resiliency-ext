# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Progressive log-analysis boundary for attribution orchestration.

This module defines the NVRx-owned request shape for starting progressive log
analysis. The concrete LogSage behavior is intentionally not implemented here
yet; until that shared contract exists, the MCP/tool entry point reports
``unsupported``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

ANALYSIS_INTENT_TRACK_ONLY = "track_only"
ANALYSIS_INTENT_PROGRESSIVE = "progressive"
ANALYSIS_INTENT_TERMINAL = "terminal"
ANALYSIS_INTENTS = (
    ANALYSIS_INTENT_TRACK_ONLY,
    ANALYSIS_INTENT_PROGRESSIVE,
    ANALYSIS_INTENT_TERMINAL,
)

MODULE_LOG_ANALYZER_PROGRESSIVE_START = "log_analyzer_progressive_start"

PROGRESSIVE_STATUS_UNSUPPORTED = "unsupported"
PROGRESSIVE_STATUS_FAILED = "failed"


def normalize_analysis_intent(value: str | None) -> str:
    """Normalize and validate a POST /logs analysis intent."""
    if value is None:
        return ANALYSIS_INTENT_TRACK_ONLY
    normalized = str(value).strip().lower()
    if not normalized:
        return ANALYSIS_INTENT_TRACK_ONLY
    if normalized not in ANALYSIS_INTENTS:
        raise ValueError("analysis_intent must be one of: " + ", ".join(sorted(ANALYSIS_INTENTS)))
    return normalized


@dataclass(frozen=True)
class ProgressiveStartResult:
    """Result of a non-result-producing progressive analysis start request."""

    status: str
    message: str = ""
    handle: str | None = None

    def as_payload(self) -> dict[str, str | None]:
        """Serialize status metadata for MCP/HTTP-adjacent boundaries."""
        return asdict(self)


def progressive_start_result_from_mcp_response(
    response: Mapping[str, Any],
) -> ProgressiveStartResult:
    """Parse the MCP progressive-start response envelope into status metadata."""
    result = response.get("result")
    payload: Mapping[str, Any]
    if isinstance(result, Mapping):
        payload = result
    else:
        payload = response

    status = str(payload.get("status") or PROGRESSIVE_STATUS_UNSUPPORTED)
    message = str(payload.get("message") or "")
    handle_value = payload.get("handle")
    handle = str(handle_value) if handle_value is not None else None
    return ProgressiveStartResult(status=status, message=message, handle=handle)


class ProgressiveLogAnalysisStartTool:
    """MCP tool stub for starting progressive log analysis.

    This is deliberately a non-result-producing tool. It advertises the
    NVRx-owned loganalysis boundary now, while the LogSage progressive API is
    still being defined.
    """

    def __init__(self, _args: Mapping[str, Any] | None = None):
        pass

    async def run(self, _arguments: Mapping[str, Any]) -> dict[str, str | None]:
        """Return status metadata without running terminal attribution."""
        payload = ProgressiveStartResult(
            status=PROGRESSIVE_STATUS_UNSUPPORTED,
            message="LogSage progressive start API is not configured",
        ).as_payload()
        payload["module"] = MODULE_LOG_ANALYZER_PROGRESSIVE_START
        return payload
