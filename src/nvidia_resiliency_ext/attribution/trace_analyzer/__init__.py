# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NCCL flight recorder (FR) integration.

- :mod:`fr_support` — path from logs, :class:`FRAnalysisResult`, :func:`analyze_fr_dump`,
  dataflow fields, Markdown appendix
- :mod:`fr_attribution` — ``CollectiveAnalyzer`` (large CLI / core implementation)
"""

from nvidia_resiliency_ext.attribution.trace_analyzer.fr_support import (
    FR_DUMP_PATH_LOG_LINE_PATTERN,
    FR_DUMP_PATH_LOG_SCAN_LINES,
    FRAnalysisResult,
    analyze_fr_dump,
    extract_fr_dump_path,
    fr_fields_for_dataflow_record,
    fr_markdown_appendix,
    fr_markdown_appendix_from_result,
    fr_path_resolvable_for_collective_analyzer,
)
from nvidia_resiliency_ext.attribution.trace_analyzer.trace_analyzer import TraceAnalyzer

__all__ = [
    "FRAnalysisResult",
    "TraceAnalyzer",
    "FR_DUMP_PATH_LOG_LINE_PATTERN",
    "FR_DUMP_PATH_LOG_SCAN_LINES",
    "analyze_fr_dump",
    "extract_fr_dump_path",
    "fr_fields_for_dataflow_record",
    "fr_markdown_appendix",
    "fr_markdown_appendix_from_result",
    "fr_path_resolvable_for_collective_analyzer",
]
