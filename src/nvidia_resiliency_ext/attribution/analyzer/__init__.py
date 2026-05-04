# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attribution orchestration layer: :class:`Analyzer`, pipeline, wiring.

- :mod:`nvidia_resiliency_ext.attribution.analyzer.engine` — :class:`Analyzer`
- :mod:`nvidia_resiliency_ext.attribution.orchestration.log_analyzer` — LogSage :class:`~nvidia_resiliency_ext.attribution.orchestration.log_analyzer.LogAnalyzer`
- :mod:`nvidia_resiliency_ext.attribution.trace_analyzer.trace_analyzer` — :class:`~nvidia_resiliency_ext.attribution.trace_analyzer.trace_analyzer.TraceAnalyzer`
- :mod:`nvidia_resiliency_ext.attribution.orchestration.analysis_pipeline` — :func:`run_attribution_pipeline`,
  :class:`AnalysisPipelineMode`
- :mod:`nvidia_resiliency_ext.attribution.orchestration.types` — :class:`LogAnalyzerConfig`, result dataclasses

The sibling :mod:`nvidia_resiliency_ext.attribution.orchestration` package holds parsers,
splitlog, wire/error codes, and orchestration-facing dataclasses.
"""

from nvidia_resiliency_ext.attribution.analyzer.engine import Analyzer
from nvidia_resiliency_ext.attribution.orchestration.analysis_pipeline import (
    AnalysisPipelineMode,
    CombinedAnalysisResult,
    FrDumpPathNotFoundError,
    run_attribution_pipeline,
)
from nvidia_resiliency_ext.attribution.orchestration.types import (
    LogAnalysisCycleResult,
    LogAnalysisSplitlogResult,
    LogAnalyzerConfig,
    LogAnalyzerError,
    LogAnalyzerFilePreview,
    LogAnalyzerOutcome,
    LogAnalyzerSubmitResult,
)
from nvidia_resiliency_ext.attribution.trace_analyzer.trace_analyzer import TraceAnalyzer

__all__ = [
    "AnalysisPipelineMode",
    "CombinedAnalysisResult",
    "FrDumpPathNotFoundError",
    "Analyzer",
    "TraceAnalyzer",
    "LogAnalysisCycleResult",
    "LogAnalysisSplitlogResult",
    "LogAnalyzerConfig",
    "LogAnalyzerError",
    "LogAnalyzerFilePreview",
    "LogAnalyzerOutcome",
    "LogAnalyzerSubmitResult",
    "run_attribution_pipeline",
]
