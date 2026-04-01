# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed results, stats, and in-memory entry records for the request coalescer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List, TypedDict


class _CacheStats(TypedDict):
    hits: int
    misses: int
    size: int


class _SubmissionsStats(TypedDict):
    total: int
    pending: int


class _InflightStats(TypedDict):
    total: int
    current: int


class _RequestStats(TypedDict):
    total: int
    coalesced: int


class _ComputeStats(TypedDict):
    total: int
    errors: int
    timeouts: int


@dataclass(frozen=True)
class ComputeStats:
    """Compute/LLM stats for health checks. Use this instead of raw dict keys."""

    total: int
    errors: int
    timeouts: int


class _CleanupStats(TypedDict):
    cache_expired: int
    submitted_expired: int
    in_flight_cleaned: int


class StatsResult(TypedDict):
    """Return type for :meth:`~nvidia_resiliency_ext.attribution.coalescing.coalescer.RequestCoalescer.get_stats`."""

    cache: _CacheStats
    submissions: _SubmissionsStats
    in_flight: _InflightStats
    requests: _RequestStats
    compute: _ComputeStats
    cleanup: _CleanupStats


class CacheEntryInfo(TypedDict):
    """Info about a single cache entry."""

    path: str
    age_seconds: float
    module: str
    result_id: str
    state: str


class CacheResult(TypedDict):
    """Return type for :meth:`~nvidia_resiliency_ext.attribution.coalescing.coalescer.RequestCoalescer.get_cache`."""

    count: int
    entries: List[CacheEntryInfo]


class InflightResult(TypedDict):
    """Return type for :meth:`~nvidia_resiliency_ext.attribution.coalescing.coalescer.RequestCoalescer.get_inflight`."""

    count: int
    paths: List[str]


class SubmittedEntryInfo(TypedDict):
    """Info about a single submitted entry."""

    path: str
    age_seconds: float
    status: str


class SubmittedResult(TypedDict):
    """Return type for :meth:`~nvidia_resiliency_ext.attribution.coalescing.coalescer.RequestCoalescer.get_submitted`."""

    count: int
    entries: List[SubmittedEntryInfo]


@dataclass
class CoalescerStats:
    """Statistics counters for :class:`~nvidia_resiliency_ext.attribution.coalescing.coalescer.RequestCoalescer`.

    Cache validation strategy:
    - Grace period: First N seconds after caching, serve without validation
    - After grace period: stat() file on each hit to validate (mtime, size)
    - If file changed: invalidate entry, treat as cache miss
    - Timeout entries (file_size=None): served as hit during grace; after grace
      invalidated so next request retries; evicted by same file-age rule
    - Eviction: file.mtime > max_file_age_days (default 14 days)
    """

    cache_hits: int = 0
    cache_misses: int = 0
    cache_invalidated: int = 0  # Cache entries invalidated (file changed)
    coalesced_requests: int = 0  # Requests that waited for in-flight
    total_computes: int = 0  # Total compute_fn invocations
    compute_errors: int = 0  # Compute failures
    compute_timeouts: int = 0  # Compute timeouts
    total_submitted: int = 0  # Total paths submitted via POST
    total_in_flight: int = 0  # Total paths that entered in_flight state
    cache_expired_cleaned: int = 0  # Cache entries removed by cleanup (file too old)
    submitted_expired_cleaned: int = 0  # Stale submitted entries removed
    in_flight_expired_cleaned: int = 0  # Stuck in-flight entries removed
    cache_imported: int = 0  # Entries restored from disk
    cache_import_skipped_changed: int = 0  # Skipped on import (file changed)
    cache_import_skipped_old_file: int = 0  # Skipped on import (file.mtime too old)


@dataclass
class CacheEntry:
    """Entry in the processed files ledger.

    Each entry represents a file that has been analyzed and posted to Elasticsearch.
    Storing (mtime, size) allows us to detect if the file has changed since processing.

    Validation behavior:
    - Grace period: First N seconds, serve without stat() check (absorbs straggling writes)
    - After grace period: stat() on each hit to detect file changes
    - If file changed: entry is invalidated, file will be re-analyzed
    - Eviction: file.mtime > max_file_age_days (default 14 days)

    Timeout entries store file_mtime only (file_size=None). They are served as hit
    during grace period; after grace they are invalidated so the next request retries.
    They are evicted by the same file-age rule (file_mtime used for cleanup).
    """

    result: Any  # Analysis result (also stored in ES, kept here for convenience)
    cached_at: float  # time.monotonic() when cached
    file_mtime: float | None = None  # File mtime at cache time (validation + eviction)
    file_size: int | None = (
        None  # File size at cache time (validation; None = timeout → retry after grace)
    )


@dataclass
class InFlightEntry:
    """Entry tracking an in-flight request using asyncio.Future."""

    future: asyncio.Future[Any]
    started_at: float  # time.monotonic() when started
