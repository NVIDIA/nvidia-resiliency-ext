# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LogSage / LLM text parsing and dataflow field shaping from parsed output.

This module is separate from :mod:`nvidia_resiliency_ext.attribution.orchestration.utils` (path
validation and MCP/lib result shaping) because LLM output handling is a distinct concern.

Path-derived job/cycle ids live in :mod:`nvidia_resiliency_ext.attribution.orchestration.log_path_metadata`.

- :func:`parse_llm_response` / :class:`ParsedLLMResponse` — structured fields from raw LLM text
- :func:`attribution_recommendation` — decision helper
- :func:`log_fields_for_dataflow_record` — keys merged in :mod:`~nvidia_resiliency_ext.attribution.postprocessing.pipeline`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from .config import RESP_ERROR, RESP_MODULE, RESP_RESULT, RESP_STATE, STATE_TIMEOUT
from .log_path_metadata import JobMetadata
from .types import (
    RECOMMENDATION_CONTINUE,
    RECOMMENDATION_RESTART,
    RECOMMENDATION_STOP,
    RECOMMENDATION_TIMEOUT,
    RECOMMENDATION_UNKNOWN,
    AttributionRecommendation,
)

logger = logging.getLogger(__name__)


def attribution_recommendation(attr_result: Optional[Dict[str, Any]]) -> AttributionRecommendation:
    """Normalize backend-specific log-analysis output.

    This function expects the inner analysis result, e.g. ``{"state": ..., "result": ...}``.
    Full attrsvc HTTP response bodies are parsed by ``parse_attrsvc_response``.
    """
    if attr_result is None or not isinstance(attr_result, dict):
        return AttributionRecommendation(reason="missing attribution result")

    state = _extract_state(attr_result)
    source = _extract_source(attr_result)
    reason = _extract_reason(attr_result)
    state_action = _action_from_state(state)
    known_unknown_reason = _known_unknown_reason(state, source)

    if reason:
        text_action = _action_from_text(reason)
        action = text_action if text_action != RECOMMENDATION_UNKNOWN else state_action
    elif state_action != RECOMMENDATION_UNKNOWN:
        action = state_action
    elif known_unknown_reason:
        action = RECOMMENDATION_UNKNOWN
        reason = known_unknown_reason
    else:
        result_text = str(attr_result)
        logger.warning(
            "attribution_recommendation: falling through to string matching on result: %s",
            result_text[:200] + ("..." if len(result_text) > 200 else ""),
        )
        action = _action_from_text(result_text)

    if not reason and state:
        reason = f"state={state}"

    return AttributionRecommendation(
        action=action,
        reason=reason,
        source=source,
    )


def _known_unknown_reason(state: Optional[str], source: str) -> str:
    if source == "fr_only" and state == "no_log":
        return "no_log"
    return ""


def _extract_state(attr_result: Dict[str, Any]) -> Optional[str]:
    for candidate in (attr_result, attr_result.get(RESP_RESULT)):
        if isinstance(candidate, dict):
            state = candidate.get(RESP_STATE)
            if isinstance(state, str) and state.strip():
                return state.strip()
    return None


def _extract_source(attr_result: Dict[str, Any]) -> str:
    for candidate in (attr_result, attr_result.get(RESP_RESULT)):
        if isinstance(candidate, dict):
            module = candidate.get(RESP_MODULE)
            if isinstance(module, str) and module.strip():
                return module.strip()
    return ""


def _extract_reason(attr_result: Dict[str, Any]) -> str:
    error = attr_result.get(RESP_ERROR)
    if isinstance(error, str) and error:
        return error

    nested = attr_result.get(RESP_RESULT)
    if isinstance(nested, dict):
        nested_error = nested.get(RESP_ERROR)
        if isinstance(nested_error, str) and nested_error:
            return nested_error
        nested = nested.get(RESP_RESULT)

    return _reason_text(nested)


def _reason_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _reason_text(item)
            if text:
                return text
    return ""


def _action_from_state(state: Optional[str]) -> str:
    if not isinstance(state, str):
        return RECOMMENDATION_UNKNOWN
    normalized = _normalize_action(state)
    if normalized == RECOMMENDATION_STOP:
        return RECOMMENDATION_STOP
    if normalized == RECOMMENDATION_RESTART:
        return RECOMMENDATION_RESTART
    if normalized == RECOMMENDATION_CONTINUE:
        return RECOMMENDATION_CONTINUE
    if normalized == STATE_TIMEOUT.upper():
        return RECOMMENDATION_TIMEOUT
    return RECOMMENDATION_UNKNOWN


def _normalize_action(action: Any) -> str:
    return action.strip().upper() if isinstance(action, str) and action.strip() else ""


def _action_from_text(text: str) -> str:
    normalized = text.upper()
    if "STOP" in normalized and "RESTART" not in normalized.split("STOP", 1)[0]:
        return RECOMMENDATION_STOP
    if "RESTART" in normalized and "IMMEDIATE" in normalized:
        return RECOMMENDATION_RESTART
    if "ERRORS NOT FOUND" in normalized or "NO LOGS" in normalized:
        return RECOMMENDATION_CONTINUE
    return RECOMMENDATION_UNKNOWN


@dataclass
class ParsedLLMResponse:
    """Parsed fields from LLM response.

    The log_analyzer module returns structured text that includes:
    - auto_resume decision (first line)
    - auto_resume explanation (second line)
    - Attribution section with failure attribution
    - checkpoint_saved flag
    """

    auto_resume: str
    auto_resume_explanation: str
    attribution_text: str
    checkpoint_saved_flag: int


def parse_llm_response(raw_text: str) -> ParsedLLMResponse:
    """
    Parse raw LLM response text to extract structured fields.

    The expected format from log_analyzer is:
        <auto_resume_decision>
        <auto_resume_explanation>
        ...
        Attribution: <attribution_text>

        <checkpoint_saved>

    Args:
        raw_text: Raw text from LLM response

    Returns:
        ParsedLLMResponse with extracted fields
    """
    # Extract auto_resume (first line) and explanation (second line)
    lines = raw_text.split("\n")
    auto_resume = lines[0] if lines else ""
    if len(lines) > 1:
        auto_resume_explanation = lines[1]
    else:
        auto_resume_explanation = ""
        logger.warning("Failed to extract auto_resume_explanation: insufficient lines in response")

    # Extract text after 'Attribution:' marker
    attribution_parts = raw_text.split("Attribution:")
    if len(attribution_parts) > 1:
        attribution_section = attribution_parts[1].strip()
        parts = attribution_section.split("\n\n")
        attribution_text = parts[0].replace('"\\', "").replace('\\"', "")
        if len(parts) > 1:
            checkpoint_saved = parts[1]
        else:
            checkpoint_saved = "false"
            logger.debug("No checkpoint_saved field in attribution response")
    else:
        attribution_text = ""
        checkpoint_saved = "false"
        # For ERRORS NOT FOUND / NO LOGS, missing Attribution: marker is expected
        if "ERRORS NOT FOUND" in auto_resume or auto_resume.strip() == "NO LOGS":
            logger.debug(
                "No 'Attribution:' marker in LLM response (expected for %s)",
                auto_resume.strip() or "empty",
            )
        else:
            logger.warning("No 'Attribution:' marker found in LLM response")

    # Normalize checkpoint_saved to int flag
    checkpoint_saved_flag = 0
    if isinstance(checkpoint_saved, str) and checkpoint_saved.strip().lower() != "false":
        checkpoint_saved_flag = 1

    return ParsedLLMResponse(
        auto_resume=auto_resume,
        auto_resume_explanation=auto_resume_explanation,
        attribution_text=attribution_text,
        checkpoint_saved_flag=checkpoint_saved_flag,
    )


def log_fields_for_dataflow_record(
    parsed: ParsedLLMResponse,
    metadata: JobMetadata,
    log_path: str,
    processing_time: float,
) -> Dict[str, Any]:
    """Build LogSage / parsed-LLM keys to merge into a dataflow posting record.

    Mirrors :func:`~nvidia_resiliency_ext.attribution.trace_analyzer.fr_support.fr_fields_for_dataflow_record`
    for the FR side; :func:`~nvidia_resiliency_ext.attribution.postprocessing.pipeline.build_dataflow_record`
    merges both with cluster and user.
    """
    return {
        "s_attribution": parsed.attribution_text,
        "s_auto_resume": parsed.auto_resume,
        "s_auto_resume_explanation": parsed.auto_resume_explanation,
        "s_job_id": metadata.job_id,
        "l_cycle_id": metadata.cycle_id,
        "s_log_path": log_path,
        "l_checkpoint_saved": parsed.checkpoint_saved_flag,
        "d_processing_time": round(processing_time, 2),
        "ts_current_time": round(datetime.now().timestamp() * 1000),
    }
