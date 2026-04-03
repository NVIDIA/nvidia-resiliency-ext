# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP entrypoint for :class:`CombinedLogFR`: path-based parallel log+FR plus merge in the MCP process."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from nvidia_resiliency_ext.attribution.base import AttributionState
from nvidia_resiliency_ext.attribution.combined_log_fr.combined_log_fr import CombinedLogFR
from nvidia_resiliency_ext.attribution.combined_log_fr.llm_merge import unpack_run_result
from nvidia_resiliency_ext.attribution.log_analyzer.nvrx_logsage import NVRxLogAnalyzer
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
    """Run FR analysis; empty dump dir is normal — return ``(None, CONTINUE)`` for log-only merge."""
    try:
        return await fr_analyzer.run(fr_kw)
    except ValueError as e:
        if _missing_fr_dump_files_error(e):
            logger.info(
                "No FR dump files matched pattern %r under %s; continuing with log-only LLM merge.",
                fr_kw.get("pattern", "_dump_*"),
                fr_path,
            )
            return (None, AttributionState.CONTINUE)
        raise


class CombinedLogFRMCPOrchestrator:
    """MCP tool ``log_fr_analyzer``.

    * **Path mode** (``log_path`` + ``fr_path``): run LogSage and FR in parallel, then
      :class:`CombinedLogFR` / :func:`~nvidia_resiliency_ext.attribution.combined_log_fr.llm_merge.merge_log_fr_llm`
      inside this process.
    * **Legacy mode** (``input_data`` only): same as standalone :class:`CombinedLogFR`.
    """

    def __init__(self, _args: Any = None) -> None:
        """Registry constructs one instance per process; per-call params come via ``run``."""

    async def run(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], AttributionState]:
        if arguments.get("log_path") and arguments.get("fr_path"):
            return await self._run_from_paths(arguments)
        if arguments.get("input_data") is not None:
            return await self._run_from_input_data(arguments)
        raise ValueError(
            "log_fr_analyzer requires either log_path and fr_path, or input_data (legacy MCP shape)"
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

        log_kw: dict[str, Any] = {
            "log_path": log_path,
            "model": arguments.get("model", "nvidia/qwen/qwen-235b"),
            "base_url": arguments.get("base_url", "https://inference-api.nvidia.com/v1"),
            "temperature": float(arguments.get("temperature", 0.2)),
            "top_p": float(arguments.get("top_p", 0.7)),
            "max_tokens": int(arguments.get("max_tokens", 8192)),
            "exclude_nvrx_logs": bool(arguments.get("exclude_nvrx_logs", False)),
            "is_per_cycle": is_per_cycle,
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
        log_actual, log_st = unpack_run_result(log_raw)
        fr_actual, fr_st = unpack_run_result(fr_raw)

        merge_kw: dict[str, Any] = {
            "input_data": [log_actual, fr_actual],
            "model": arguments.get("model", "nvidia/qwen/qwen-235b"),
            "base_url": arguments.get("base_url", "https://inference-api.nvidia.com/v1"),
            "temperature": float(arguments.get("temperature", 0.2)),
            "top_p": float(arguments.get("top_p", 0.7)),
            "max_tokens": int(arguments.get("max_tokens", 8192)),
            "threshold": int(arguments.get("threshold", 0)),
        }
        merger = CombinedLogFR(merge_kw)
        merge_raw = await merger.run(merge_kw)
        merge_str, merge_st = unpack_run_result(merge_raw)

        payload: dict[str, Any] = {
            "log": {
                "result": log_actual,
                "state": log_st.name if isinstance(log_st, AttributionState) else str(log_st),
            },
            "fr": {
                "result": fr_actual,
                "state": fr_st.name if isinstance(fr_st, AttributionState) else str(fr_st),
            },
            "llm_merged_summary": merge_str,
        }
        combined_state = (
            AttributionState.STOP
            if (
                log_st == AttributionState.STOP
                or fr_st == AttributionState.STOP
                or merge_st == AttributionState.STOP
            )
            else AttributionState.CONTINUE
        )
        return payload, combined_state

    async def _run_from_input_data(self, arguments: dict[str, Any]) -> tuple[Any, AttributionState]:
        # Apply defaults here; pass this dict to both __init__ and run() — not raw
        # ``arguments`` — so the active run context (merged_attribution_config) always
        # includes model / temperature / top_p / max_tokens / threshold.
        run_kwargs: dict[str, Any] = {
            "input_data": arguments["input_data"],
            "model": arguments.get("model", "nvidia/qwen/qwen-235b"),
            "base_url": arguments.get("base_url", "https://inference-api.nvidia.com/v1"),
            "temperature": float(arguments.get("temperature", 0.2)),
            "top_p": float(arguments.get("top_p", 0.7)),
            "max_tokens": int(arguments.get("max_tokens", 8192)),
            "threshold": int(arguments.get("threshold", 0)),
        }
        merger = CombinedLogFR(run_kwargs)
        return await merger.run(run_kwargs)
