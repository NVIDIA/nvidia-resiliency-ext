#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

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
    AnalysisResult,
    AnalyzerError,
    AttributionService,
    FilePreviewResult,
    SubmitResult,
)

__all__ = [
    # App factory (for programmatic/ASGI use)
    "create_app",
    # Configuration
    "Settings",
    "setup",
    # Core service (for direct Python usage)
    "AttributionService",
    "AnalysisResult",
    "AnalyzerError",
    "FilePreviewResult",
    "SubmitResult",
    # TypedDicts for typed API access
    "StatsResult",
    "CacheResult",
    "InflightResult",
    "SubmittedResult",
]
