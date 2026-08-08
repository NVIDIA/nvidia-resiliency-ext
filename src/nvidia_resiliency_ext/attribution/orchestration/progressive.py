# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Progressive log-analysis boundary for attribution orchestration.

This module defines the NVRx-owned request shape for starting progressive log
analysis. When the MCP server binds a LogSage analyzer, the tool schedules the
poller in the background and returns status metadata immediately.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from typing import Any, Mapping

logger = logging.getLogger(__name__)

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
PROGRESSIVE_STATUS_STARTED = "started"


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
    """MCP tool for starting progressive log analysis.

    This is deliberately a non-result-producing tool. It starts LogSage
    progressive polling when the MCP server binds a shared analyzer, then
    returns status metadata without waiting for terminal attribution.
    """

    def __init__(
        self,
        _args: Mapping[str, Any] | None = None,
        *,
        analyzer: Any = None,
    ):
        self._analyzer = analyzer
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    async def run(self, _arguments: Mapping[str, Any]) -> dict[str, str | None] | None:
        """Run the progressive start phase.

        With no analyzer bound, returns the ``unsupported`` status payload.
        With an analyzer bound, schedules
        ``NVRxLogAnalyzer.analyze_logs_rt_start`` and returns immediately.
        """
        if self._analyzer is None:
            payload = ProgressiveStartResult(
                status=PROGRESSIVE_STATUS_UNSUPPORTED,
                message="LogSage progressive start API is not configured",
            ).as_payload()
        else:
            path = str(_arguments.get("log_path") or "")
            if not path:
                raise ValueError("log_path is required")

            task = self._tasks.get(path)
            if task is None or task.done():
                task = asyncio.create_task(
                    self._analyzer.analyze_logs_rt_start(dict(_arguments)),
                    name=f"nvrx-progressive-log-analysis:{path}",
                )
                self._tasks[path] = task
                task.add_done_callback(
                    lambda done_task, task_path=path: self._handle_task_done(
                        task_path,
                        done_task,
                    )
                )
                message = f"progressive analysis started for {path}"
            else:
                message = f"progressive analysis already running for {path}"

            payload = ProgressiveStartResult(
                status=PROGRESSIVE_STATUS_STARTED,
                message=message,
                handle=path,
            ).as_payload()

        payload["module"] = MODULE_LOG_ANALYZER_PROGRESSIVE_START
        return payload

    def _handle_task_done(self, path: str, task: asyncio.Task[Any]) -> None:
        if self._tasks.get(path) is task:
            self._tasks.pop(path, None)
        if task.cancelled():
            logger.debug("Progressive log analysis poller cancelled for %s", path)
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "Progressive log analysis poller failed for %s: %s",
                path,
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
