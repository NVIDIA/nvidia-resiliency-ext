# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate LogSage (log LLM) and optional NCCL flight-recorder (FR) analysis.

Lives under :mod:`nvidia_resiliency_ext.attribution.log_analyzer` so :class:`~nvidia_resiliency_ext.attribution.log_analyzer.log_analyzer.LogAnalyzer`
can run **log-only** or **log + FR** (via :class:`~nvidia_resiliency_ext.attribution.trace_analyzer.trace_analyzer.TraceAnalyzer`).

Callers inject LogSage via ``run_logsage`` when the mode needs it.
FR path discovery and dump analysis default to
:func:`~nvidia_resiliency_ext.attribution.trace_analyzer.fr_support.extract_fr_dump_path`
(``<run>/checkpoints`` when the analyzed log path lies under ``<run>/logs/``, else
``TORCH_FR_DUMP_TEMP_FILE=`` scanned from the log) and
:func:`~nvidia_resiliency_ext.attribution.trace_analyzer.fr_support.analyze_fr_dump`.

Modes:

- **LOG_ONLY** — LogSage only (no FR discovery or trace analysis).
- **TRACE_ONLY** — NCCL flight-recorder analysis only (no LogSage); requires a dump path from
  discovery or ``fr_dump_path_override``.
- **LOG_AND_TRACE** — LogSage plus FR when a dump path is found. Lib: both run in parallel via
  :func:`asyncio.gather` (no merge LLM). MCP: single ``log_fr_analyzer`` tool runs log + FR in
  parallel and runs the merge LLM inside the MCP process; host receives ``llm_merged_summary``.
- **LOG_AND_TRACE_WITH_LLM** — Lib: **LOG_AND_TRACE** then host :func:`merge_log_fr_llm`. MCP: same
  ``log_fr_analyzer`` tool as **LOG_AND_TRACE** (merge already in MCP); host merge is skipped when
  ``llm_merged_summary`` is already set.
"""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from nvidia_resiliency_ext.attribution.api_keys import load_nvidia_api_key
from nvidia_resiliency_ext.attribution.trace_analyzer.fr_support import (
    FRAnalysisResult,
    analyze_fr_dump,
    extract_fr_dump_path,
)


class FrDumpPathNotFoundError(Exception):
    """Raised when trace-only analysis cannot resolve an FR dump path from the log or overrides."""

    def __init__(self, message: Optional[str] = None) -> None:
        super().__init__(
            message
            or (
                "TRACE_ONLY mode requires an FR dump path (none found in log; "
                "set fr_dump_path_override or ensure the log references TORCH_FR_DUMP_TEMP_FILE)"
            )
        )


class AnalysisPipelineMode(str, enum.Enum):
    """How :func:`run_attribution_pipeline` combines LogSage and trace (FR) analysis."""

    LOG_ONLY = "log_only"
    TRACE_ONLY = "trace_only"
    LOG_AND_TRACE = "log_and_trace"
    LOG_AND_TRACE_WITH_LLM = "log_and_trace_with_llm"


@dataclass(frozen=True)
class CombinedAnalysisResult:
    """Outcome from :func:`run_attribution_pipeline` (LogSage-shaped dict plus optional FR + LLM)."""

    log_result: Optional[Dict[str, Any]]
    fr_dump_path: Optional[str]
    fr_analysis: Optional[FRAnalysisResult]
    processing_time: float
    llm_merged_summary: Optional[str] = None


async def run_attribution_pipeline(
    log_path: str,
    *,
    mode: AnalysisPipelineMode = AnalysisPipelineMode.LOG_AND_TRACE,
    run_logsage: Optional[Callable[[], Awaitable[Dict[str, Any]]]] = None,
    discover_fr_dump_path: Optional[Callable[[str], Optional[str]]] = None,
    run_fr_analysis: Optional[Callable[[str], Awaitable[Optional[FRAnalysisResult]]]] = None,
    run_log_fr_analyzer_mcp: Optional[
        Callable[
            [str, str],
            Awaitable[Tuple[Dict[str, Any], Optional[FRAnalysisResult], Optional[str]]],
        ]
    ] = None,
    fr_dump_path_override: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_temperature: float = 0.2,
    llm_top_p: float = 0.7,
    llm_max_tokens: int = 16384,
    nvidia_api_key: Optional[str] = None,
) -> CombinedAnalysisResult:
    """Run attribution according to ``mode``.

    Args:
        log_path: Path to the job log (for FR discovery when applicable).
        mode: :class:`AnalysisPipelineMode` selector.
        run_logsage: Async factory for the LogSage ``{"result": ..., "state": ...}`` dict. Required
            for modes that run LogSage (not **TRACE_ONLY**).
        discover_fr_dump_path: Override FR path scan; default uses ``extract_fr_dump_path``.
        run_fr_analysis: Override FR runner; default is :func:`analyze_fr_dump`.
        run_log_fr_analyzer_mcp: When set and a dump path exists, called as
            ``await run_log_fr_analyzer_mcp(log_path, fr_dump_path)`` returning
            ``(log_result, fr_analysis, llm_merged_summary)`` from MCP tool ``log_fr_analyzer`` (log + FR
            in parallel and merge LLM inside the MCP process).
        fr_dump_path_override: Explicit dump path for **TRACE_ONLY** (skips discovery when set).
        llm_model: Model id for **LOG_AND_TRACE_WITH_LLM** (required when that mode runs the merge).
        llm_base_url: Base url for **LOG_AND_TRACE_WITH_LLM** (required when that mode runs the merge).
        llm_temperature / llm_top_p / llm_max_tokens: Passed to the merge LLM when applicable.
        nvidia_api_key: NVIDIA API key for **LOG_AND_TRACE_WITH_LLM** host merge when MCP did not
            merge. If ``None``, resolved once per pipeline run via :func:`load_nvidia_api_key`.
    """
    discover = discover_fr_dump_path or extract_fr_dump_path
    run_fr = run_fr_analysis or analyze_fr_dump
    t0 = time.time()

    if mode == AnalysisPipelineMode.LOG_ONLY:
        if run_logsage is None:
            raise ValueError("run_logsage is required for LOG_ONLY mode")
        log_result = await run_logsage()
        return CombinedAnalysisResult(
            log_result=log_result,
            fr_dump_path=None,
            fr_analysis=None,
            processing_time=time.time() - t0,
            llm_merged_summary=None,
        )

    if mode == AnalysisPipelineMode.TRACE_ONLY:
        fr_dump_path = fr_dump_path_override or discover(log_path)
        if not fr_dump_path:
            raise FrDumpPathNotFoundError()
        fr_analysis = await run_fr(fr_dump_path)
        return CombinedAnalysisResult(
            log_result=None,
            fr_dump_path=fr_dump_path,
            fr_analysis=fr_analysis,
            processing_time=time.time() - t0,
            llm_merged_summary=None,
        )

    # LOG_AND_TRACE and LOG_AND_TRACE_WITH_LLM
    if run_logsage is None:
        raise ValueError("run_logsage is required for this mode")
    fr_dump_path = fr_dump_path_override or discover(log_path)
    llm_merged_summary: Optional[str] = None
    if fr_dump_path:
        if run_log_fr_analyzer_mcp is not None:
            log_result, fr_analysis, llm_merged_summary = await run_log_fr_analyzer_mcp(
                log_path, fr_dump_path
            )
        else:
            log_result, fr_analysis = await asyncio.gather(
                run_logsage(),
                run_fr(fr_dump_path),
            )
    else:
        log_result = await run_logsage()
        fr_analysis = None

    if (
        mode == AnalysisPipelineMode.LOG_AND_TRACE_WITH_LLM
        and llm_merged_summary is None
        and log_result is not None
        and fr_analysis is not None
    ):
        if not llm_model:
            raise ValueError("llm_model is required for LOG_AND_TRACE_WITH_LLM when merging")
        if not llm_base_url:
            raise ValueError("llm_base_url is required for LOG_AND_TRACE_WITH_LLM when merging")
        from nvidia_resiliency_ext.attribution.combined_log_fr.llm_merge import merge_log_fr_llm

        merge_key = nvidia_api_key if nvidia_api_key is not None else load_nvidia_api_key()
        llm_merged_summary = await merge_log_fr_llm(
            log_result,
            fr_analysis,
            nvidia_api_key=merge_key,
            model=llm_model,
            base_url=llm_base_url,
            temperature=llm_temperature,
            top_p=llm_top_p,
            max_tokens=llm_max_tokens,
        )

    return CombinedAnalysisResult(
        log_result=log_result,
        fr_dump_path=fr_dump_path,
        fr_analysis=fr_analysis,
        processing_time=time.time() - t0,
        llm_merged_summary=llm_merged_summary,
    )
