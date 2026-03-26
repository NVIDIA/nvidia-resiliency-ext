# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVRX Attribution Service package."""

from nvidia_resiliency_ext.attribution import (
    CacheResult,
    InflightResult,
    StatsResult,
    SubmittedResult,
)

from .app import create_app
from .config import Settings, setup
from .service import (
    AttributionService,
    LogAnalysisCycleResult,
    LogAnalysisSplitlogResult,
    LogAnalyzerError,
    LogAnalyzerFilePreview,
    LogAnalyzerSubmitResult,
)

__all__ = [
    # App factory (for programmatic/ASGI use)
    "create_app",
    # Configuration
    "Settings",
    "setup",
    # Core service (for direct Python usage)
    "AttributionService",
    "LogAnalysisCycleResult",
    "LogAnalysisSplitlogResult",
    "LogAnalyzerError",
    "LogAnalyzerFilePreview",
    "LogAnalyzerSubmitResult",
    # TypedDicts for typed API access
    "StatsResult",
    "CacheResult",
    "InflightResult",
    "SubmittedResult",
]
