# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP entrypoint for path-based log+FR collection and optional merge in the MCP process."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from nvidia_resiliency_ext.attribution.base import AttributionState
from nvidia_resiliency_ext.attribution.combined_log_fr.combined_log_fr import CombinedLogFR
from nvidia_resiliency_ext.attribution.combined_log_fr.llm_merge import unpack_run_result
from nvidia_resiliency_ext.attribution.log_analyzer.nvrx_logsage import NVRxLogAnalyzer
from nvidia_resiliency_ext.attribution.orchestration.config import (
    MODULE_LOG_ANALYZER,
    MODULE_LOG_FR_ANALYZER,
    resolved_llm_runtime_kwargs,
)
from nvidia_resiliency_ext.attribution.orchestration.types import LogSageAnalysisResult
from nvidia_resiliency_ext.attribution.orchestration.utils import log_analyzer_result_payload
from nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution import CollectiveAnalyzer
from nvidia_resiliency_ext.attribution.trace_analyzer.fr_support import (
    fr_path_resolvable_for_collective_analyzer,
)

logger = logging.getLogger(__name__)


def _missing_fr_dump_files_error(exc: ValueError) -> bool:
    msg = str(exc)
    return msg.startswith("No files at ") and "were processed successfully" in msg


async def _run_fr_or_skip_missing_dumps(
    fr_analyzer: CollectiveAnalyzer, fr_kw: dict[str, Any], fr_path: str
) -> Any:
    """Run FR analysis; empty dump dir is normal — return ``(None, CONTINUE)`` without merging."""
    try:
        return await fr_analyzer.run(fr_kw)
    except ValueError as e:
        if _missing_fr_dump_files_error(e):
            logger.info(
                "No FR dump files matched pattern %r under %s; continuing without merge LLM.",
                fr_kw.get("pattern", "_dump_*"),
                fr_path,
            )
            return (None, AttributionState.CONTINUE)
        raise


def _attribution_state_name(state: Any) -> str:
    return state.name if isinstance(state, AttributionState) else str(state)


def _log_section_from_run_result(
    log_raw: Any,
) -> tuple[dict[str, Any], LogSageAnalysisResult | list[Any], AttributionState]:
    """Return the canonical serialized LogSage section for combined MCP path mode."""
    log_actual, log_st = unpack_run_result(log_raw)
    wrapped_result = log_raw[0] if isinstance(log_raw, tuple) and len(log_raw) == 2 else log_raw
    if isinstance(wrapped_result, LogSageAnalysisResult):
        log_input = wrapped_result
    elif isinstance(wrapped_result, list):
        log_input = wrapped_result
    elif isinstance(log_actual, list):
        log_input = log_actual
    else:
        raise RuntimeError(
            "log_fr_analyzer LogSage result must be LogSageAnalysisResult or list, "
            f"got {type(log_actual).__name__}"
        )
    return log_analyzer_result_payload(log_input), log_input, log_st


def _log_section_from_input_data(log_input: Any) -> dict[str, Any]:
    """Normalize input-data LogSage input to the canonical payload section."""
    if isinstance(log_input, dict) and isinstance(log_input.get("result"), list):
        payload = {
            "module": str(log_input.get("module") or MODULE_LOG_ANALYZER),
            "result": log_input["result"],
        }
        recommendation = log_input.get("recommendation")
        if isinstance(recommendation, dict):
            payload["recommendation"] = recommendation
        else:
            payload["recommendation"] = log_analyzer_result_payload(
                log_input["result"],
                module=payload["module"],
            )["recommendation"]
        return payload
    if isinstance(log_input, (list, LogSageAnalysisResult)):
        return log_analyzer_result_payload(log_input)
    raise RuntimeError(
        "log_fr_analyzer input-data log input must be LogSageAnalysisResult, "
        "list, or canonical log result dict, "
        f"got {type(log_input).__name__}"
    )


def _log_input_for_merge(log_input: Any) -> Any:
    """Return the LogSage value shape that ``merge_log_fr_llm`` renders as raw text."""
    if isinstance(log_input, LogSageAnalysisResult):
        return log_input.items
    if isinstance(log_input, dict) and isinstance(log_input.get("result"), list):
        return log_input["result"]
    return log_input


def _log_fr_result_payload(
    log_payload_section: dict[str, Any],
    fr_actual: Any,
    fr_st: Any,
    merge_str: Any,
) -> dict[str, Any]:
    """Build the canonical top-level ``log_fr_analyzer`` MCP payload."""
    recommendation = log_payload_section.get("recommendation")
    if not isinstance(recommendation, dict):
        recommendation = {
            "action": "UNKNOWN",
            "source": MODULE_LOG_FR_ANALYZER,
        }

    result = log_payload_section.get("result")
    if not isinstance(result, list):
        result_type = type(result).__name__
        raise RuntimeError(
            f"log_fr_analyzer LogSage payload result must be list, got {result_type}"
        )

    payload: dict[str, Any] = {
        "module": MODULE_LOG_FR_ANALYZER,
        "result": result,
        "recommendation": recommendation,
        "fr": {
            "result": fr_actual,
            "state": _attribution_state_name(fr_st),
        },
    }
    if merge_str is not None:
        payload["llm_merged_summary"] = merge_str if isinstance(merge_str, str) else str(merge_str)
    return payload


async def _merge_log_fr_summary(
    log_actual: Any,
    fr_actual: Any,
    arguments: dict[str, Any],
    llm_kwargs: dict[str, Any],
) -> str:
    """Run the Log+FR merge LLM and return its text summary only.

    The merge runner may return an ``AttributionState`` for standalone CLI compatibility,
    but MCP callers must not treat merge output as stop/restart policy.
    """
    merge_kw: dict[str, Any] = {
        "input_data": [_log_input_for_merge(log_actual), fr_actual],
        "threshold": int(arguments.get("threshold", 0)),
        **llm_kwargs,
    }
    merger = CombinedLogFR(merge_kw)
    merge_raw = await merger.run(merge_kw)
    merge_str, _merge_st = unpack_run_result(merge_raw)
    return merge_str if isinstance(merge_str, str) else str(merge_str)


class CombinedLogFRMCPOrchestrator:
    """MCP tool ``log_fr_analyzer``.

    * **Path mode** (``log_path`` + ``fr_path``): run LogSage and FR in parallel.
      When ``merge_llm`` is true and FR data exists, also run the Log+FR merge LLM.
    * **Input-data mode** (``input_data`` only): wrap already-collected LogSage and FR
      outputs; when ``merge_llm`` is true and FR data exists, run the merge LLM.
    """

    def __init__(self, _args: Any = None) -> None:
        """Registry constructs one instance per process; per-call params come via ``run``."""

    async def run(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], AttributionState]:
        if arguments.get("log_path") and arguments.get("fr_path"):
            return await self._run_from_paths(arguments)
        if arguments.get("input_data") is not None:
            return await self._run_from_input_data(arguments)
        raise ValueError(
            "log_fr_analyzer requires either log_path and fr_path, or input_data "
            "(input-data MCP shape)"
        )

    async def _run_from_paths(
        self, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], AttributionState]:
        log_path = str(arguments["log_path"])
        fr_path = os.path.abspath(os.path.expanduser(str(arguments["fr_path"])))
        if not fr_path_resolvable_for_collective_analyzer(fr_path):
            raise ValueError(
                "fr_path must be an existing directory, an existing FR dump file, or a path prefix "
                "with at least one matching dump file (e.g. TORCH_FR_DUMP_TEMP_FILE=/shared/_dump_); "
                f"not usable: {fr_path!r}"
            )
        is_per_cycle = bool(arguments.get("is_per_cycle", False))
        llm_kwargs = resolved_llm_runtime_kwargs(arguments)

        log_kw: dict[str, Any] = {
            "log_path": log_path,
            "exclude_nvrx_logs": bool(arguments.get("exclude_nvrx_logs", False)),
            "is_per_cycle": is_per_cycle,
            **llm_kwargs,
        }
        fr_kw: dict[str, Any] = {
            "fr_path": fr_path,
            "pattern": str(arguments.get("pattern", "_dump_*")),
            "model": arguments.get("model"),
            "verbose": bool(arguments.get("verbose", False)),
            "health_check": bool(arguments.get("health_check", False)),
            "llm_analyze": bool(arguments.get("llm_analyze", False)),
            "threshold": arguments.get("threshold"),
        }

        log_analyzer = NVRxLogAnalyzer(log_kw)
        fr_analyzer = CollectiveAnalyzer(fr_kw)
        log_raw, fr_raw = await asyncio.gather(
            log_analyzer.run(log_kw),
            _run_fr_or_skip_missing_dumps(fr_analyzer, fr_kw, fr_path),
        )
        log_payload_section, log_actual, log_st = _log_section_from_run_result(log_raw)
        fr_actual, fr_st = unpack_run_result(fr_raw)
        merge_llm = bool(arguments.get("merge_llm", False))
        merge_str = None
        if merge_llm and fr_actual is not None:
            merge_str = await _merge_log_fr_summary(log_actual, fr_actual, arguments, llm_kwargs)

        payload = _log_fr_result_payload(log_payload_section, fr_actual, fr_st, merge_str)
        # FR and merge output are monitor/context signals here. The client-facing
        # stop/restart policy is the LogSage recommendation envelope in payload.
        combined_state = (
            AttributionState.STOP if log_st == AttributionState.STOP else AttributionState.CONTINUE
        )
        return payload, combined_state

    async def _run_from_input_data(
        self, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], AttributionState]:
        input_data = arguments["input_data"]
        if not isinstance(input_data, (list, tuple)) or len(input_data) < 2:
            raise ValueError("log_fr_analyzer input_data must be [log_result, fr_result]")

        llm_kwargs = resolved_llm_runtime_kwargs(arguments)
        fr_input = input_data[1]
        merge_llm = bool(arguments.get("merge_llm", False))
        merge_str = None
        if merge_llm and fr_input is not None:
            merge_str = await _merge_log_fr_summary(input_data[0], fr_input, arguments, llm_kwargs)
        return (
            _log_fr_result_payload(
                _log_section_from_input_data(input_data[0]),
                fr_input,
                AttributionState.CONTINUE,
                merge_str,
            ),
            AttributionState.CONTINUE,
        )
