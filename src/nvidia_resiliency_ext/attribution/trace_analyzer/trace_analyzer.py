# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NCCL flight-recorder discovery and dump analysis.

Used by :class:`~nvidia_resiliency_ext.attribution.svc.log_analyzer.LogAnalyzer` when the
attribution pipeline includes flight-recorder analysis (see :mod:`~nvidia_resiliency_ext.attribution.svc.analysis_pipeline`).
"""

from __future__ import annotations

from typing import Optional

from .fr_support import FRAnalysisResult, analyze_fr_dump, extract_fr_dump_path


class TraceAnalyzer:
    """Discovers FR dump paths from job logs and analyzes dumps (async)."""

    def __init__(self, allowed_root: Optional[str] = None) -> None:
        """``allowed_root``: when set, FR paths inferred from log layout or ``TORCH_FR_DUMP_TEMP_FILE`` must lie under it."""

        self._allowed_root = allowed_root

    def discover_fr_dump_path(self, log_path: str) -> Optional[str]:
        """Return path to NCCL FR dump referenced by the job log, or ``None``."""
        return extract_fr_dump_path(log_path, allowed_root=self._allowed_root)

    async def analyze_fr_dump(self, dump_path: str) -> Optional[FRAnalysisResult]:
        """Analyze a single FR dump file."""
        return await analyze_fr_dump(dump_path)
