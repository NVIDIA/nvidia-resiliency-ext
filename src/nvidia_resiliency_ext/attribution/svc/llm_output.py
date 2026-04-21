# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LogSage / LLM text parsing and dataflow field shaping from parsed output.

This module is separate from :mod:`nvidia_resiliency_ext.attribution.svc.utils` (path
validation and MCP/lib result shaping) because LLM output handling is a distinct concern.

Path-derived job/cycle ids live in :mod:`nvidia_resiliency_ext.attribution.svc.log_path_metadata`.

- :func:`parse_llm_response` / :class:`ParsedLLMResponse` — structured fields from raw LLM text
- :func:`attribution_no_restart` — decision helper on attribution dicts
- :func:`log_fields_for_dataflow_record` — keys merged in :mod:`~nvidia_resiliency_ext.attribution.postprocessing.pipeline`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from .log_path_metadata import JobMetadata

logger = logging.getLogger(__name__)


def attribution_no_restart(attr_result: Optional[Dict[str, Any]]) -> bool:
    """Whether attribution recommends do not restart (stop).

    Call on the raw result from log analysis or an attribution service (or None if unavailable).
    True = stop; False = restart or no usable result (skip).

    Handles result shapes: state STOP/CONTINUE/RESTART, or strings containing
    ``STOP - DONT RESTART`` / ``RESTART IMMEDIATE``.
    """
    if attr_result is None or not isinstance(attr_result, dict):
        return False
    state = attr_result.get("state")
    if state == "STOP":
        return True
    if state in ("CONTINUE", "RESTART"):
        return False
    nested = attr_result.get("result")
    if isinstance(nested, dict):
        nested_state = nested.get("state")
        if nested_state == "STOP":
            return True
        if nested_state in ("CONTINUE", "RESTART"):
            return False
    if isinstance(nested, (list, tuple)) and nested:
        first = nested[0]
        s = first if isinstance(first, str) else str(first)
        if "STOP" in s and "RESTART" not in s.split("STOP")[0]:
            return True
        if "RESTART" in s:
            return False
    s = str(attr_result)
    logger.warning(
        "attribution_no_restart: falling through to string matching on result: %s",
        s[:200] + ("..." if len(s) > 200 else ""),
    )
    if "STOP" in s and "DONT RESTART" in s:
        return True
    if "RESTART" in s and "IMMEDIATE" in s:
        return False
    return False


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
