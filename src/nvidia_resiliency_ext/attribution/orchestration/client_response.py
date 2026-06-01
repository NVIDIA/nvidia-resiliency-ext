# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side helpers for attrsvc response payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import (
    RESP_LOG_FILE,
    RESP_MODE,
    RESP_MODULE,
    RESP_RESULT,
    RESP_RESULT_ID,
    RESP_SCHED_RESTARTS,
    RESP_STATUS,
    RESP_WL_RESTART,
)
from .types import (
    RECOMMENDATION_STOP,
    RECOMMENDATION_TIMEOUT,
    AttributionRecommendation,
    RawAnalysisResultItem,
)

_SPLITLOG_MODE = "splitlog"


@dataclass(frozen=True)
class AttrSvcResult:
    """Normalized view of an attrsvc ``GET /logs`` response."""

    result: Any
    status: str = "completed"
    log_path: str | None = None
    recommendation: AttributionRecommendation = field(default_factory=AttributionRecommendation)
    should_stop: bool = False
    mode: str = "single"
    analyzed_log_file: str = ""
    wl_restart: Any = None
    sched_restarts: Any = None

    @property
    def recommendation_reason(self) -> str:
        """Convenience alias for callers that only need the recommendation reason."""
        return self.recommendation.reason

    @property
    def module(self) -> str:
        """Raw backend module for the analyzed result, e.g. ``log_analyzer`` or ``fr_only``."""
        return _result_string(self.result, RESP_MODULE)

    @property
    def result_id(self) -> str:
        return _result_string(self.result, RESP_RESULT_ID)

    def result_id_preview(self, max_chars: int = 16) -> str:
        if len(self.result_id) > max_chars:
            return f"{self.result_id[:max_chars]}..."
        return self.result_id

    def attribution_preview(self, max_chars: int = 200) -> str:
        attribution_result = (
            self.result.get(RESP_RESULT, "") if isinstance(self.result, dict) else ""
        )
        if isinstance(attribution_result, list):
            text = " | ".join(_raw_result_item_preview(item) for item in attribution_result)
        else:
            text = str(attribution_result) if attribution_result else ""
        if len(text) > max_chars:
            return f"{text[:max_chars]}..."
        return text

    def format_summary(self, *, prefix: str = "", preview_chars: int = 200) -> str:
        """Return a multi-line summary suitable for smon logs/stdout."""
        action = self.recommendation.action
        reason = self.recommendation.reason
        if action == RECOMMENDATION_TIMEOUT and not reason:
            reason = "Attribution analysis timed out"

        title = "Attribution timeout" if action == RECOMMENDATION_TIMEOUT else "Attribution result"
        lines = [f"{prefix}{title}:"]
        if self.mode == _SPLITLOG_MODE:
            lines.append(
                f"  Mode: {_SPLITLOG_MODE} (wl_restart {self.wl_restart}/{self.sched_restarts})"
            )
            if self.log_path:
                lines.append(f"  Slurm output: {self.log_path}")
            if self.analyzed_log_file:
                lines.append(f"  Analyzed log: {self.analyzed_log_file}")
        elif self.log_path:
            lines.append(f"  Log: {self.log_path}")

        if self.module:
            lines.append(f"  Module: {self.module}")
        if self.result_id:
            lines.append(f"  Result ID: {self.result_id_preview()}")
        lines.append(f"  Recommendation: {action}")
        if reason:
            lines.append(f"  Reason: {reason}")
        if self.recommendation.source and self.recommendation.source != self.module:
            lines.append(f"  Recommendation source: {self.recommendation.source}")
        if action != RECOMMENDATION_TIMEOUT:
            lines.append(f"  Attribution: {self.attribution_preview(preview_chars)}")
        return "\n".join(lines)

    def format_log_message(self, *, preview_chars: int = 200) -> str:
        """Return a compact one-line summary suitable for launcher logs."""
        path = f" for {self.log_path}" if self.log_path else ""
        return (
            f"AttrSvcResult{path}: status={self.status} "
            f"recommendation={self.recommendation.action} "
            f"reason={self.recommendation_reason} "
            f"should_stop={self.should_stop} "
            f"result preview: {self.attribution_preview(preview_chars)}"
        )


def parse_attrsvc_response(payload: Any, *, log_path: str | None = None) -> AttrSvcResult:
    """Parse attrsvc response JSON into a stable client contract.

    Callers should use ``recommendation`` and ``should_stop`` instead of peeking at
    backend-specific fields under ``result``.
    """
    body = payload if isinstance(payload, dict) else {}
    result = body.get(RESP_RESULT, payload)
    status = _string_value(body.get(RESP_STATUS)) or "completed"
    recommendation = AttributionRecommendation.from_payload(body.get("recommendation"))
    if recommendation is None:
        recommendation = AttributionRecommendation()

    return AttrSvcResult(
        result=result,
        status=status,
        log_path=log_path,
        recommendation=recommendation,
        should_stop=recommendation_should_stop(recommendation),
        mode=_string_value(body.get(RESP_MODE)) or "single",
        analyzed_log_file=_string_value(body.get(RESP_LOG_FILE)),
        wl_restart=body.get(RESP_WL_RESTART),
        sched_restarts=body.get(RESP_SCHED_RESTARTS),
    )


def recommendation_should_stop(recommendation: AttributionRecommendation) -> bool:
    """Return whether a normalized recommendation is a stop signal."""
    return recommendation.action == RECOMMENDATION_STOP


def _string_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _result_string(result: Any, key: str) -> str:
    if not isinstance(result, dict):
        return ""
    return _string_value(result.get(key))


def _raw_result_item_preview(item: Any) -> str:
    try:
        return RawAnalysisResultItem.from_payload(item).raw_text
    except (TypeError, ValueError):
        return str(item)
