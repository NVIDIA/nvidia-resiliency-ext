# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LogSage recommendation and dataflow field shaping from structured output.

This module is separate from :mod:`nvidia_resiliency_ext.attribution.orchestration.utils` (path
validation and MCP/lib result shaping) because LogSage result shaping is a distinct concern.

Path-derived job/cycle ids live in :mod:`nvidia_resiliency_ext.attribution.orchestration.log_path_metadata`.

- :func:`logsage_recommendation` — decision helper for structured LogSage result items
- :func:`log_fields_for_dataflow_record` — keys merged in :mod:`~nvidia_resiliency_ext.attribution.postprocessing.pipeline`
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from .config import (
    MODULE_FR_ONLY,
    MODULE_LOG_ANALYZER,
    RESP_ERROR,
    RESP_MODULE,
    RESP_RESULT,
    RESP_STATE,
    STATE_NO_LOG,
    STATE_TIMEOUT,
)
from .log_path_metadata import JobMetadata
from .types import (
    RECOMMENDATION_CONTINUE,
    RECOMMENDATION_PAYLOAD_FIELDS,
    RECOMMENDATION_RESTART,
    RECOMMENDATION_STOP,
    RECOMMENDATION_TIMEOUT,
    RECOMMENDATION_UNKNOWN,
    AttributionRecommendation,
    RawAnalysisResultItem,
    normalize_recommendation_action,
)

_ACTION_ORDER = (
    RECOMMENDATION_STOP,
    RECOMMENDATION_RESTART,
    RECOMMENDATION_TIMEOUT,
    RECOMMENDATION_UNKNOWN,
    RECOMMENDATION_CONTINUE,
)


def logsage_recommendation(
    items: List[Any],
    *,
    source: str = "",
) -> AttributionRecommendation:
    """Derive the internal recommendation from structured LogSage result items.

    LogSage parses each cycle once and stores the derived action on
    :class:`RawAnalysisResultItem`. This helper only chooses the highest-priority
    action across cycles.
    """
    result_items = [RawAnalysisResultItem.from_payload(item) for item in items]
    selected = _selected_action_item(result_items)
    if selected is None:
        return AttributionRecommendation(source=source)
    return AttributionRecommendation(
        action=selected.action,
        reason=selected.raw_text,
        source=source,
    )


def logsage_recommendation_from_payload(
    payload: Optional[Dict[str, Any]],
) -> AttributionRecommendation:
    """Read or synthesize a recommendation from a wrapped LogSage payload.

    Normal success payloads should already contain ``recommendation``. Timeout,
    no-log, and error markers are handled explicitly; no generic result-list
    fallback is applied here.
    """
    if payload is None or not isinstance(payload, dict):
        return AttributionRecommendation(reason="missing attribution result")

    source = _payload_source(payload)
    state = _string_value(payload.get(RESP_STATE)).strip().lower()
    if state == STATE_TIMEOUT:
        return timeout_recommendation(
            source=source,
            reason=_string_value(payload.get(RESP_ERROR)),
        )
    if source == MODULE_FR_ONLY and state == STATE_NO_LOG:
        return no_log_recommendation(source=source)
    recommendation = AttributionRecommendation.from_payload(
        payload.get("recommendation"), source=source
    )
    if recommendation is not None:
        return recommendation

    return AttributionRecommendation(
        reason=_string_value(payload.get(RESP_ERROR)),
        source=source,
    )


def timeout_recommendation(*, source: str = "", reason: str = "") -> AttributionRecommendation:
    """Build the explicit recommendation used when analysis execution timed out."""
    return AttributionRecommendation(
        action=RECOMMENDATION_TIMEOUT,
        reason=reason,
        source=source,
    )


def no_log_recommendation(*, source: str = MODULE_FR_ONLY) -> AttributionRecommendation:
    """Build the explicit recommendation for FR-only entries with no LogSage result."""
    return AttributionRecommendation(
        action=RECOMMENDATION_UNKNOWN,
        reason=STATE_NO_LOG,
        source=source,
    )


def fr_only_no_log_payload() -> Dict[str, Any]:
    """Build the canonical FR-only result payload when no LogSage result exists."""
    return {
        RESP_MODULE: MODULE_FR_ONLY,
        RESP_STATE: STATE_NO_LOG,
        RESP_RESULT: [],
        "recommendation": recommendation_payload(no_log_recommendation()),
    }


def logsage_timeout_payload(reason: str) -> Dict[str, Any]:
    """Build the canonical LogSage timeout payload stored in the cache."""
    return {
        RESP_MODULE: MODULE_LOG_ANALYZER,
        RESP_STATE: STATE_TIMEOUT,
        RESP_RESULT: [],
        RESP_ERROR: reason,
        "recommendation": recommendation_payload(
            timeout_recommendation(source=MODULE_LOG_ANALYZER, reason=reason)
        ),
    }


def recommendation_payload(recommendation: AttributionRecommendation) -> Dict[str, str]:
    """Serialize the client-facing recommendation contract."""
    payload = {
        "action": normalize_recommendation_action(recommendation.action),
        "source": recommendation.source,
    }
    return {field_name: payload[field_name] for field_name in RECOMMENDATION_PAYLOAD_FIELDS}


def _string_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _payload_source(payload: Dict[str, Any]) -> str:
    module = payload.get(RESP_MODULE)
    return module.strip() if isinstance(module, str) and module.strip() else ""


def _selected_action_item(items: List[RawAnalysisResultItem]) -> Optional[RawAnalysisResultItem]:
    if not items:
        return None
    for action in _ACTION_ORDER:
        for item in items:
            if item.action == action:
                return item
    return None


def attribution_result_payload(item: RawAnalysisResultItem) -> Dict[str, Any]:
    """Complete normalized attribution result stored in ``s_attribution_result_json``."""
    return {
        "auto_resume_explanation": item.auto_resume_explanation,
        "primary_issues": item.primary_issues,
        "secondary_issues": item.secondary_issues,
        "checkpoint_saved": bool(item.checkpoint_saved_flag),
        "raw_auto_resume": item.auto_resume,
    }


def log_fields_for_dataflow_record(
    item: RawAnalysisResultItem,
    metadata: JobMetadata,
    log_path: str,
    attribution_analysis_duration_seconds: float,
    attribution_analysis_completed_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Build LogSage / parsed-LLM keys to merge into a dataflow posting record.

    Mirrors :func:`~nvidia_resiliency_ext.attribution.trace_analyzer.fr_support.fr_fields_for_dataflow_record`
    for the FR side; :func:`~nvidia_resiliency_ext.attribution.postprocessing.pipeline.build_dataflow_record`
    merges both with cluster and user.
    """
    result_payload = attribution_result_payload(item)
    return {
        "s_auto_resume": item.auto_resume,
        "s_auto_resume_explanation": item.auto_resume_explanation,
        "s_primary_issues": ", ".join(item.primary_issues),
        "s_attribution_result_json": json.dumps(result_payload, sort_keys=True),
        "s_job_id": metadata.job_id,
        "l_cycle_id": metadata.cycle_id,
        "s_log_path": log_path,
        "d_attribution_analysis_duration_seconds": round(attribution_analysis_duration_seconds, 2),
        "ts_attribution_analysis_completed_ms": (
            attribution_analysis_completed_ms
            if attribution_analysis_completed_ms is not None
            else round(time.time() * 1000)
        ),
    }
