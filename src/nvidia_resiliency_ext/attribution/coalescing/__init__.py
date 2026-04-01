# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request coalescing, processed-files ledger, and cache payload types (LogSage + optional FR).

**Single-flight:** Only one LLM/analysis run per cache key at a time; concurrent callers share
the result via :class:`RequestCoalescer`.

**Ledger / cache:** Tracks analyzed paths with file (mtime, size) for invalidation, grace period,
timeout handling, eviction, and optional persistence import/export.

**Layout:** ``types`` (stats and entry records), ``coalesced_cache`` (cached value shape),
``coalescer`` (:class:`RequestCoalescer` and compute timeout default).
"""

from .coalesced_cache import LogAnalysisCoalesced, coalesced_from_cache
from .coalescer import DEFAULT_COMPUTE_TIMEOUT_SECONDS, RequestCoalescer
from .types import (
    CacheResult,
    CoalescerStats,
    ComputeStats,
    InflightResult,
    StatsResult,
    SubmittedResult,
)

__all__ = [
    "DEFAULT_COMPUTE_TIMEOUT_SECONDS",
    "LogAnalysisCoalesced",
    "coalesced_from_cache",
    "CacheResult",
    "CoalescerStats",
    "ComputeStats",
    "InflightResult",
    "RequestCoalescer",
    "StatsResult",
    "SubmittedResult",
]
