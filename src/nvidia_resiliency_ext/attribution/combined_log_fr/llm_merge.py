# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fuse LogSage output with NCCL flight-recorder analysis via an LLM (LangChain + ChatOpenAI).

Used by :class:`~nvidia_resiliency_ext.attribution.combined_log_fr.combined_log_fr.CombinedLogFR` and
by :func:`~nvidia_resiliency_ext.attribution.orchestration.analysis_pipeline.run_attribution_pipeline` (``LOG_AND_TRACE_WITH_LLM``).
Kept as a standalone function so library code does not need ``NVRxAttribution`` or CLI wiring.
"""

from __future__ import annotations

import asyncio
from typing import Any, Tuple

from nvidia_resiliency_ext.attribution.base import AttributionState
from nvidia_resiliency_ext.attribution.orchestration.config import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TOP_P,
)
from nvidia_resiliency_ext.attribution.orchestration.types import (
    RECOMMENDATION_STOP,
    LogSageAnalysisResult,
    RawAnalysisResultItem,
)
from nvidia_resiliency_ext.attribution.trace_analyzer.fr_support import FRAnalysisResult


def unpack_run_result(result: Any) -> Tuple[Any, AttributionState]:
    """Normalize :meth:`~nvidia_resiliency_ext.attribution.base.NVRxAttribution.run_sync` return values.

    Many analyzers (e.g. :class:`~nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution.CollectiveAnalyzer`)
    return ``(payload, AttributionState)``.

    :class:`~nvidia_resiliency_ext.attribution.log_analyzer.nvrx_logsage.NVRxLogAnalyzer` returns
    ``tuple[list[RawAnalysisResultItem], AttributionState]`` from
    :meth:`~nvidia_resiliency_ext.attribution.log_analyzer.nvrx_logsage.NVRxLogAnalyzer.print_output`.
    The outer 2-tuple is unpacked first (``payload = list[RawAnalysisResultItem]``, ``state``),
    then the inner list is unpacked on a second call to join per-cycle strings.
    """
    if isinstance(result, tuple) and len(result) == 2:
        payload, state = result
        if isinstance(payload, LogSageAnalysisResult):
            return payload.items, state
        return payload, state
    if isinstance(result, LogSageAnalysisResult):
        state = (
            AttributionState.STOP
            if result.recommendation.action == RECOMMENDATION_STOP
            else AttributionState.CONTINUE
        )
        return result.items, state
    if isinstance(result, list):
        if not result:
            return "", AttributionState.CONTINUE
        if all(isinstance(x, (RawAnalysisResultItem, dict)) for x in result):
            items = [RawAnalysisResultItem.from_payload(x) for x in result]
            texts = [item.raw_text for item in items]
            combined = "\n\n".join(t for t in texts if t)
            merged_state = (
                AttributionState.STOP
                if any(item.action == RECOMMENDATION_STOP for item in items)
                else AttributionState.CONTINUE
            )
            return combined, merged_state
    return result, AttributionState.CONTINUE


_MERGE_TEMPLATE = """
        You are a helpful assistant that analyzes the application logs and collective operations analysis.
        You are given the application logs and collective operations analysis.
        {log_result} includes attribution results based on application logs.
        Its attribution can be highly false positive. Use it to decide whether to restart the application.
        Even if the log results has some suggestion on ranks to be excluded, you should not use it.

        {fr_result} includes health check results per rank and collective analysis, which is more reliable.
        Use the hanging ranks it provides to isolate the ranks that are hanging.
        Even if the fr result has many ranks to be excluded, you can use them as they are to propose a solution.

        You need to analyze the application logs and collective operations analysis and return the proposed solution.
        Summary of the log result: <application log summary>
        Summary of the fr result: <collective operations analysis summary>

        The proposed solution should be in the following format: (one line only, if you have extra information, you can add it in the proposed solution with ranks)
        - List of ranks to be excluded: <identified ranks to be excluded, you can use comma to separate multiple ranks without space>
        - Proposed Solution with Ranks: <proposed solution with ranks>
        """


def _fr_side_to_prompt_text(fr: Any) -> str:
    """String fed into the ``fr_result`` slot: structured FR output or raw CLI string."""
    if fr is None:
        return (
            "No flight-recorder (NCCL) dump files were found in the configured directory. "
            "There is no collective-operations analysis; rely on the application log analysis only."
        )
    if isinstance(fr, FRAnalysisResult):
        return f"{fr.analysis_text}\n{fr.hanging_ranks}"
    if isinstance(fr, dict) and ("analysis_text" in fr or "hanging_ranks" in fr):
        text = str(fr.get("analysis_text") or "")
        hr = str(fr.get("hanging_ranks") or "")
        merged = f"{text}\n{hr}".strip()
        return merged if merged else str(fr)
    if isinstance(fr, str):
        return fr
    return str(fr)


async def merge_log_fr_llm(
    log_result: Any,
    fr_result: Any,
    *,
    llm_api_key: str,
    model: str = DEFAULT_LLM_MODEL,
    base_url: str = DEFAULT_LLM_BASE_URL,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    top_p: float = DEFAULT_LLM_TOP_P,
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
) -> str:
    """Run the Nemotron-style fusion prompt; ``fr_result`` may be :class:`FRAnalysisResult` or raw text.

    Callers should pass a key obtained once (e.g. from :func:`~nvidia_resiliency_ext.attribution.api_keys.load_llm_api_key`
    at startup or pipeline entry) so the merge step does not re-read env/files on every call.

    """
    if not (llm_api_key and llm_api_key.strip()):
        raise ValueError(
            "LLM API key is empty. Load it once via load_llm_api_key() and pass llm_api_key=... "
            "Required for log+FR LLM merge."
        )
    api_key = llm_api_key.strip()

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    log_payload, _ = unpack_run_result(log_result)
    fr_payload, _ = unpack_run_result(fr_result)

    log_str = log_payload if isinstance(log_payload, str) else str(log_payload)
    fr_str = _fr_side_to_prompt_text(fr_payload)

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    prompt = PromptTemplate(template=_MERGE_TEMPLATE, input_variables=["log_result", "fr_result"])
    chain = prompt | llm | StrOutputParser()

    def _invoke() -> str:
        return chain.invoke({"log_result": log_str, "fr_result": fr_str})

    return await asyncio.to_thread(_invoke)
