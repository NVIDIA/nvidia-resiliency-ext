# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RequestCoalescer cache payload: LogSage output plus optional NCCL flight-recorder fields.

Lives under ``attribution`` (not ``log_analyzer``) because it combines LogSage analysis with
optional FR attribution from ``trace_analyzer``.

An entry may be **FR-only** (``log_result`` is ``None``) when the cache stores flight-recorder
fields without a LogSage-shaped dict—for example imported or hand-built payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from nvidia_resiliency_ext.attribution.trace_analyzer.fr_support import FRAnalysisResult


def _normalize_fr_analysis(raw_fr: Any) -> Optional[FRAnalysisResult]:
    if isinstance(raw_fr, dict):
        return FRAnalysisResult(**raw_fr)
    if raw_fr is not None and isinstance(raw_fr, FRAnalysisResult):
        return raw_fr
    return None


@dataclass
class LogAnalysisCoalesced:
    """Cache value: joint LogSage result and optional FR attribution.

    ``log_result`` is the LogSage-shaped dict (``result`` plus ``recommendation``), or ``None``
    when the payload is FR-only (no LogSage output). FR fields are optional:
    ``fr_dump_path`` when :func:`~nvidia_resiliency_ext.attribution.trace_analyzer.extract_fr_dump_path`
    finds a path in the log; ``fr_analysis`` when :func:`~nvidia_resiliency_ext.attribution.trace_analyzer.analyze_fr_dump`
    succeeds. ``llm_merged_summary`` is set only when the merge LLM ran
    (**LOG_AND_TRACE_WITH_LLM** and FR data was present). Older cache entries may store only a bare
    ``log_result`` dict (no wrapper keys).
    """

    log_result: Optional[Dict[str, Any]] = None
    fr_dump_path: Optional[str] = None
    fr_analysis: Optional[FRAnalysisResult] = None
    llm_merged_summary: Optional[str] = None


def coalesced_from_cache(raw: Any) -> LogAnalysisCoalesced:
    """Normalize coalescer payload (``LogAnalysisCoalesced``, nested dict, or bare ``log_result`` dict)."""
    if isinstance(raw, LogAnalysisCoalesced):
        return raw
    if isinstance(raw, dict) and "log_result" in raw:
        return LogAnalysisCoalesced(
            log_result=raw["log_result"],
            fr_dump_path=raw.get("fr_dump_path"),
            fr_analysis=_normalize_fr_analysis(raw.get("fr_analysis")),
            llm_merged_summary=raw.get("llm_merged_summary"),
        )
    # FR-only serialized form: fr_dump_path / fr_analysis without log_result key
    if (
        isinstance(raw, dict)
        and "log_result" not in raw
        and ("fr_dump_path" in raw or "fr_analysis" in raw)
    ):
        return LogAnalysisCoalesced(
            log_result=None,
            fr_dump_path=raw.get("fr_dump_path"),
            fr_analysis=_normalize_fr_analysis(raw.get("fr_analysis")),
            llm_merged_summary=raw.get("llm_merged_summary"),
        )
    if isinstance(raw, dict):
        return LogAnalysisCoalesced(log_result=raw)
    raise TypeError(f"Unexpected coalescer value type: {type(raw).__name__}")
