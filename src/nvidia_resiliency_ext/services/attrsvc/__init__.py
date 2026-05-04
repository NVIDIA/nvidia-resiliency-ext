# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVRX Attribution Service package.

The service depends on optional attribution extras. Keep this package import
lightweight so a base ``nvidia-resiliency-ext`` install can contain the service
code without forcing FastAPI, Pydantic, LogSage, MCP, or Slack dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from nvidia_resiliency_ext.attribution._optional import reraise_if_missing_attribution_dependency

if TYPE_CHECKING:
    from nvidia_resiliency_ext.attribution.coalescing import (
        CacheResult,
        InflightResult,
        StatsResult,
        SubmittedResult,
    )

    from .app import create_app
    from .config import Settings, setup
    from .service import (
        AttributionHttpAdapter,
        AttributionRecommendation,
        LogAnalysisCycleResult,
        LogAnalysisSplitlogResult,
        LogAnalyzerError,
        LogAnalyzerFilePreview,
        LogAnalyzerSubmitResult,
    )

_EXPORTS = {
    "create_app": ".app",
    "Settings": ".config",
    "setup": ".config",
    "AttributionHttpAdapter": ".service",
    "AttributionRecommendation": ".service",
    "LogAnalysisCycleResult": ".service",
    "LogAnalysisSplitlogResult": ".service",
    "LogAnalyzerError": ".service",
    "LogAnalyzerFilePreview": ".service",
    "LogAnalyzerSubmitResult": ".service",
    "StatsResult": "nvidia_resiliency_ext.attribution.coalescing",
    "CacheResult": "nvidia_resiliency_ext.attribution.coalescing",
    "InflightResult": "nvidia_resiliency_ext.attribution.coalescing",
    "SubmittedResult": "nvidia_resiliency_ext.attribution.coalescing",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        if module_name.startswith("."):
            module = import_module(module_name, __name__)
        else:
            module = import_module(module_name)
    except ModuleNotFoundError as exc:
        reraise_if_missing_attribution_dependency(exc, feature=f"{__name__}.{name}")
        raise

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))


__all__ = [
    # App factory (for programmatic/ASGI use)
    "create_app",
    # Configuration
    "Settings",
    "setup",
    # HTTP adapter (for direct Python usage)
    "AttributionHttpAdapter",
    "AttributionRecommendation",
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
