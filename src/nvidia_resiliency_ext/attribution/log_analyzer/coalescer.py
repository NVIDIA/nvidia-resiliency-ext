#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Request coalescing and processed files ledger.

This module provides two key functions:

1. **Single-flight pattern**: Ensures only one LLM call runs for a given file at a time.
   Concurrent requests for the same file share the result, avoiding duplicate API calls.

2. **Processed files ledger**: Tracks which files have been analyzed, preventing
   duplicate processing after service restarts. Results are posted to Elasticsearch,
   but the ledger allows the service to know "I've already processed this file"
   without querying ES.

Cache/Ledger behavior:
- Stores (path, mtime, size, result) for each processed file
- Grace period (default 10 min): Serve cached result without file validation
- After grace period: stat() file to check (mtime, size); invalidate if changed
- Timeout results: Stored with file_mtime only (file_size=None). Within grace period
  served as hit to avoid retry storm; after grace period invalidated so next request
  triggers a retry. Evicted by same file-age rule (14 days).
- Eviction: Remove entries where file.mtime > 14 days (safeguard against growth)
- Persistence: Save to disk on shutdown, restore on startup

The grace period absorbs straggling writes at end of files, preventing unnecessary
re-analysis while still detecting genuine file changes.

Future optimization: If stat() overhead becomes significant for immutable files
(older files that won't change), add mutable/immutable distinction to skip
validation for files where a newer file exists for the same job.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, TypedDict

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_COMPUTE_TIMEOUT_SECONDS = 300.0  # 5 minutes


# TypedDict definitions for structured return types
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
    """Return type for get_stats() from coalescer."""

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
    """Return type for get_cache()."""

    count: int
    entries: List[CacheEntryInfo]


class InflightResult(TypedDict):
    """Return type for get_inflight()."""

    count: int
    paths: List[str]


class SubmittedEntryInfo(TypedDict):
    """Info about a single submitted entry."""

    path: str
    age_seconds: float
    status: str


class SubmittedResult(TypedDict):
    """Return type for get_submitted()."""

    count: int
    entries: List[SubmittedEntryInfo]


@dataclass
class CoalescerStats:
    """Statistics counters for RequestCoalescer.

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

    future: "asyncio.Future[Any]"
    started_at: float  # time.monotonic() when started


class RequestCoalescer:
    """
    Ensures only one computation runs for a given key at a time.
    Subsequent requests for the same key wait for the result.

    This implements the "single-flight" pattern using asyncio.Future:
    - First request creates a Future and starts computing
    - Subsequent requests await the same Future
    - Result or exception is propagated to all waiters automatically

    Usage:
        coalescer = RequestCoalescer(compute_timeout=300.0)

        # All concurrent calls with same key share one computation
        result = await coalescer.get_or_compute(key, expensive_async_fn)
    """

    # Default configuration values
    DEFAULT_CLEANUP_CADENCE_MINUTES = 10.0  # Run cleanup every 10 minutes
    DEFAULT_GRACE_PERIOD_SECONDS = 600.0  # 10 min grace before stat() validation
    DEFAULT_SUBMITTED_TTL_HOURS = 168.0  # 1 week - keep submitted tracking longer
    DEFAULT_MAX_FILE_AGE_DAYS = 14.0  # Evict entries if file.mtime > 2 weeks
    SECONDS_PER_HOUR = 3600.0
    SECONDS_PER_MINUTE = 60.0
    SECONDS_PER_DAY = 86400.0

    @staticmethod
    def _get_file_metadata(path: str) -> tuple[float | None, int | None]:
        """Get file mtime and size for cache validation.

        Returns:
            Tuple of (mtime, size) or (None, None) if file doesn't exist
        """
        try:
            stat = os.stat(path)
            return stat.st_mtime, stat.st_size
        except OSError:
            return None, None

    def __init__(
        self,
        cleanup_cadence_minutes: float = DEFAULT_CLEANUP_CADENCE_MINUTES,
        grace_period_seconds: float = DEFAULT_GRACE_PERIOD_SECONDS,
        submitted_ttl_hours: float = DEFAULT_SUBMITTED_TTL_HOURS,
        compute_timeout: float = DEFAULT_COMPUTE_TIMEOUT_SECONDS,
        max_file_age_days: float = DEFAULT_MAX_FILE_AGE_DAYS,
    ):
        """
        Args:
            cleanup_cadence_minutes: How often to run cleanup (default 10 minutes).
            grace_period_seconds: Time before stat() validation on cache hits (default 5 min).
            submitted_ttl_hours: How long to keep submitted entries (default 1 week).
            compute_timeout: Timeout for compute_fn in seconds (default 5 minutes).
            max_file_age_days: Evict entries if file.mtime exceeds this (default 14 days).
        """
        self._cache: Dict[str, CacheEntry] = {}  # key -> cache entry
        self._in_flight: Dict[str, InFlightEntry] = {}  # key -> in-flight entry
        self._submitted: Dict[str, float] = {}  # key -> submit timestamp
        self._lock = asyncio.Lock()
        self._cleanup_cadence = cleanup_cadence_minutes * self.SECONDS_PER_MINUTE
        self._grace_period = grace_period_seconds
        self._submitted_ttl = submitted_ttl_hours * self.SECONDS_PER_HOUR
        self._max_file_age = max_file_age_days * self.SECONDS_PER_DAY
        self._last_cleanup = time.monotonic()
        self._compute_timeout = compute_timeout

        # Statistics counters
        self._stats = CoalescerStats()

    def _collect_expired_keys(
        self,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """
        Scan for expired keys WITHOUT holding lock.

        Safe because asyncio is single-threaded - no preemption during sync iteration.
        Returns lists of keys to delete; actual deletion happens under lock.

        Returns:
            Tuple of (expired_cache, expired_submitted, leaked_in_flight, stuck_in_flight)
        """
        now = time.monotonic()
        now_timestamp = time.time()  # For file mtime comparison
        stuck_threshold = self._compute_timeout * 2

        # Entries expire if file.mtime is too old (> max_file_age_days)
        expired_cache = []
        for key, entry in self._cache.items():
            if entry.file_mtime is not None:
                file_age = now_timestamp - entry.file_mtime
                if file_age >= self._max_file_age:
                    expired_cache.append(key)

        expired_submitted = [
            key
            for key, submit_time in self._submitted.items()
            if now - submit_time >= self._submitted_ttl
        ]
        # Leaked: future.done() but entry still in dict (bug - should have been removed)
        leaked_in_flight = [
            key
            for key, entry in self._in_flight.items()
            if entry.future.done() and now - entry.started_at >= stuck_threshold
        ]
        # Stuck: not future.done() but running too long (compute hung)
        stuck_in_flight = [
            key
            for key, entry in self._in_flight.items()
            if not entry.future.done() and now - entry.started_at >= stuck_threshold
        ]
        return (
            expired_cache,
            expired_submitted,
            leaked_in_flight,
            stuck_in_flight,
        )

    async def _delete_expired_keys(
        self,
        expired_cache: list[str],
        expired_submitted: list[str],
        leaked_in_flight: list[str],
        stuck_in_flight: list[str],
    ) -> int:
        """
        Delete expired keys while holding lock.

        Uses .pop(key, None) to handle keys that may have been removed
        between scan and delete (safe for race conditions).

        Returns:
            Number of entries removed
        """
        total_removed = 0

        # Track what was cleaned for logging outside lock
        cleaned_cache_count = 0
        cleaned_submitted_count = 0
        cleaned_leaked: list[str] = []
        cleaned_stuck: list[str] = []

        async with self._lock:
            # Delete expired cache entries (file.mtime too old)
            for key in expired_cache:
                if self._cache.pop(key, None) is not None:
                    total_removed += 1
                    cleaned_cache_count += 1
            if cleaned_cache_count:
                self._stats.cache_expired_cleaned += cleaned_cache_count

            # Delete stale submitted entries
            for key in expired_submitted:
                if self._submitted.pop(key, None) is not None:
                    total_removed += 1
                    cleaned_submitted_count += 1
            if cleaned_submitted_count:
                self._stats.submitted_expired_cleaned += cleaned_submitted_count

            # Delete leaked in-flight entries
            for key in leaked_in_flight:
                if self._in_flight.pop(key, None) is not None:
                    total_removed += 1
                    cleaned_leaked.append(key)

            # Delete stuck in-flight entries and set exception for waiters
            for key in stuck_in_flight:
                entry = self._in_flight.pop(key, None)
                if entry is not None:
                    total_removed += 1
                    if not entry.future.done():
                        entry.future.set_exception(
                            TimeoutError(
                                f"Compute exceeded stuck threshold ({self._compute_timeout * 2:.0f}s)"
                            )
                        )
                    cleaned_stuck.append(key)

            cleaned_in_flight = len(cleaned_leaked) + len(cleaned_stuck)
            if cleaned_in_flight:
                self._stats.in_flight_expired_cleaned += cleaned_in_flight

        # Log outside lock to avoid blocking other operations
        if cleaned_cache_count:
            logger.debug(
                f"Cleaned up {cleaned_cache_count} cache entries "
                f"(file.mtime > {self._max_file_age / self.SECONDS_PER_DAY:.0f} days)"
            )
        if cleaned_submitted_count:
            logger.debug(f"Cleaned up {cleaned_submitted_count} stale submitted entries")
        for key in cleaned_leaked:
            logger.warning(f"Cleaned up leaked in-flight entry (done but not removed): {key}")
        for key in cleaned_stuck:
            logger.warning(f"Cleaned up stuck in-flight entry (compute hung): {key}")

        return total_removed

    async def get_raw_stats(self) -> CoalescerStats:
        """
        Get raw statistics counters.

        Returns:
            CoalescerStats dataclass with raw counters
        """
        async with self._lock:
            return self._stats

    def _get_compute_stats_unsafe(self) -> ComputeStats:
        """Return compute stats; caller must hold self._lock."""
        s = self._stats
        return ComputeStats(
            total=s.total_computes,
            errors=s.compute_errors,
            timeouts=s.compute_timeouts,
        )

    async def get_compute_stats(self) -> ComputeStats:
        """Get compute/LLM stats for health checks. Prefer this over raw get_stats()['compute']."""
        async with self._lock:
            return self._get_compute_stats_unsafe()

    async def get_stats(self) -> StatsResult:
        """
        Get current statistics about the cache and request coalescing.

        Returns:
            StatsResult with statistics organized by category
        """
        async with self._lock:
            s = self._stats
            total_requests = s.cache_hits + s.cache_misses + s.coalesced_requests
            compute = self._get_compute_stats_unsafe()
            return {
                "cache": {
                    "size": len(self._cache),
                    "hits": s.cache_hits,
                    "misses": s.cache_misses,
                    "invalidated": s.cache_invalidated,
                },
                "submissions": {
                    "total": s.total_submitted,
                    "pending": len(self._submitted),
                },
                "in_flight": {
                    "total": s.total_in_flight,
                    "current": len(self._in_flight),
                },
                "requests": {
                    "total": total_requests,
                    "coalesced": s.coalesced_requests,
                },
                "compute": {
                    "total": compute.total,
                    "errors": compute.errors,
                    "timeouts": compute.timeouts,
                },
                "cleanup": {
                    "cache_expired": s.cache_expired_cleaned,
                    "submitted_expired": s.submitted_expired_cleaned,
                    "in_flight_cleaned": s.in_flight_expired_cleaned,
                },
                "persistence": {
                    "imported": s.cache_imported,
                    "import_skipped_changed": s.cache_import_skipped_changed,
                    "import_skipped_old_file": s.cache_import_skipped_old_file,
                },
            }

    async def get_cache(self) -> CacheResult:
        """
        Get current cache contents.

        Returns:
            CacheResult with cached keys and metadata
        """
        async with self._lock:
            now = time.monotonic()
            entries = []
            for key, cache_entry in self._cache.items():
                age_seconds = now - cache_entry.cached_at
                # Extract summary from result (avoid dumping full result)
                result = cache_entry.result
                if isinstance(result, dict):
                    module = str(result.get("module", "unknown"))
                    result_id = str(result.get("result_id", ""))[:16]
                    state = str(result.get("state", ""))
                else:
                    module = "unknown"
                    result_id = ""
                    state = ""
                entries.append(
                    {
                        "path": key,
                        "age_seconds": round(age_seconds, 1),
                        "module": module,
                        "result_id": result_id,
                        "state": state,
                    }
                )
            return {
                "count": len(entries),
                "entries": entries,
            }

    def export_cache(self) -> List[Dict[str, Any]]:
        """
        Export cache entries for persistence.

        Returns a list of dicts with key, result, and file metadata for validation.
        All entries include file_mtime and file_size for validation on import.

        Note: This is synchronous for use in shutdown handlers.
        """
        entries = []
        for key, cache_entry in self._cache.items():
            entry_data: Dict[str, Any] = {
                "key": key,
                "result": cache_entry.result,
            }
            # Include file metadata for validation
            if cache_entry.file_mtime is not None:
                entry_data["file_mtime"] = cache_entry.file_mtime
            if cache_entry.file_size is not None:
                entry_data["file_size"] = cache_entry.file_size
            entries.append(entry_data)
        logger.debug(f"Exported {len(entries)} cache entries")
        return entries

    def import_cache(self, entries: List[Dict[str, Any]]) -> int:
        """
        Import cache entries from persistence.

        All entries are validated by file (mtime, size). Entries are skipped if:
        - File no longer exists
        - File (mtime, size) has changed
        - File mtime is older than max_file_age_days

        Args:
            entries: List of dicts with key, result, and file metadata

        Returns:
            Number of entries imported

        Note: This is synchronous, should be called during startup before
        async processing begins.
        """
        now = time.monotonic()
        now_timestamp = time.time()  # For file mtime comparison
        imported = 0
        skipped_changed = 0
        skipped_old_file = 0

        for entry in entries:
            key = entry.get("key")
            result = entry.get("result")
            file_mtime = entry.get("file_mtime")
            file_size = entry.get("file_size")

            if not key or result is None:
                continue

            # Skip entries without file metadata (e.g., old cache format)
            if file_mtime is None or file_size is None:
                logger.debug(f"Skipping cache entry (no file metadata): {key}")
                skipped_changed += 1
                continue

            # Check if file.mtime is too old (> max_file_age_days)
            file_age = now_timestamp - file_mtime
            if file_age >= self._max_file_age:
                logger.debug(
                    f"Skipping cache entry (file too old): {key} "
                    f"(file.mtime age={file_age / self.SECONDS_PER_DAY:.1f} days)"
                )
                skipped_old_file += 1
                continue

            # Validate by file metadata
            current_mtime, current_size = self._get_file_metadata(key)
            if current_mtime is None or current_size is None:
                # File no longer exists
                logger.debug(f"Skipping cache entry (file gone): {key}")
                skipped_changed += 1
                continue
            if current_mtime != file_mtime or current_size != file_size:
                # File has changed
                logger.debug(
                    f"Skipping cache entry (file changed): {key} "
                    f"(mtime: {file_mtime}->{current_mtime}, size: {file_size}->{current_size})"
                )
                skipped_changed += 1
                continue

            # File unchanged - restore entry
            self._cache[key] = CacheEntry(
                result=result,
                cached_at=now,  # Treat as freshly cached for grace period
                file_mtime=file_mtime,
                file_size=file_size,
            )
            imported += 1

        # Update stats
        self._stats.cache_imported += imported
        self._stats.cache_import_skipped_changed += skipped_changed
        self._stats.cache_import_skipped_old_file += skipped_old_file

        logger.info(
            f"Imported {imported} cache entries "
            f"(skipped: {skipped_changed} changed/gone, {skipped_old_file} old file)"
        )
        return imported

    async def get_inflight(self) -> InflightResult:
        """
        Get currently in-flight requests.

        Returns:
            InflightResult with keys currently being processed
        """
        async with self._lock:
            paths = list(self._in_flight.keys())
            return {
                "count": len(paths),
                "paths": paths,
            }

    async def track_submission(self, key: str) -> None:
        """Track a submitted key."""
        async with self._lock:
            self._submitted[key] = time.monotonic()
            self._stats.total_submitted += 1

    async def get_submitted(self) -> SubmittedResult:
        """
        Get submitted keys (POST received but not yet processed).

        Returns:
            SubmittedResult with submitted keys and metadata
        """
        async with self._lock:
            now = time.monotonic()
            entries = []
            for key, submit_time in self._submitted.items():
                age_seconds = now - submit_time
                # Check status
                if key in self._cache:
                    status = "cached"
                elif key in self._in_flight:
                    status = "in_flight"
                else:
                    status = "pending"
                entries.append(
                    {
                        "path": key,
                        "age_seconds": round(age_seconds, 1),
                        "status": status,
                    }
                )
            return {
                "count": len(entries),
                "entries": entries,
            }

    def _is_cached_timeout(self, result: Any) -> bool:
        """Check if a cached result is a timeout marker."""
        return isinstance(result, dict) and result.get("state") == "timeout"

    async def get_or_compute(self, key: str, compute_fn: Callable[[], Awaitable[Any]]) -> Any:
        """
        Get cached result or compute it (only once per key).

        Cache validation strategy:
        - Grace period: First N seconds after caching, serve without stat() check
        - After grace period: stat() file to validate (mtime, size)
        - If file changed: invalidate entry, treat as cache miss
        - Timeout entries (file_mtime set, file_size None): served as hit during
          grace period; after grace, invalidated so next request triggers a retry

        Uses asyncio.Future for clean result/exception propagation to all waiters.

        Args:
            key: The cache key (e.g., normalized file path)
            compute_fn: Async function to call if not cached/in-flight

        Returns:
            The result (from cache or freshly computed)

        Raises:
            Any exception from compute_fn (propagated to all waiters)
        """
        future: "asyncio.Future[Any] | None" = None
        is_compute_owner = False

        cached_result = None
        is_waiting = False

        # Check if cleanup is needed (minimal lock time)
        needs_cleanup = False
        async with self._lock:
            now = time.monotonic()
            if now - self._last_cleanup >= self._cleanup_cadence:
                needs_cleanup = True
                self._last_cleanup = now  # Mark as done to prevent concurrent cleanups

        # Run cleanup outside main lock to avoid blocking other operations
        if needs_cleanup:
            expired = self._collect_expired_keys()  # Scan without lock
            await self._delete_expired_keys(*expired)  # Delete with lock

        async with self._lock:
            now = time.monotonic()
            # Check cache first
            if key in self._cache:
                cache_entry = self._cache[key]
                cache_age = now - cache_entry.cached_at

                # Within grace period: serve without validation
                if cache_age < self._grace_period:
                    self._stats.cache_hits += 1
                    self._submitted.pop(key, None)
                    cached_result = cache_entry.result
                else:
                    # After grace period: validate file (mtime, size)
                    if cache_entry.file_mtime is not None and cache_entry.file_size is not None:
                        current_mtime, current_size = self._get_file_metadata(key)
                        if (
                            current_mtime == cache_entry.file_mtime
                            and current_size == cache_entry.file_size
                        ):
                            # File unchanged - cache hit
                            self._stats.cache_hits += 1
                            self._submitted.pop(key, None)
                            cached_result = cache_entry.result
                        else:
                            # File changed - invalidate and treat as miss
                            self._cache.pop(key, None)
                            self._stats.cache_invalidated += 1
                            logger.debug(
                                f"Cache invalidated (file changed): {key} "
                                f"(mtime: {cache_entry.file_mtime}->{current_mtime}, "
                                f"size: {cache_entry.file_size}->{current_size})"
                            )
                    elif cache_entry.file_mtime is not None and cache_entry.file_size is None:
                        # Timeout entry: has mtime (for cleanup) but no size → always retry
                        self._cache.pop(key, None)
                        self._stats.cache_invalidated += 1
                        logger.debug(f"Cache invalidated (timeout entry, retry): {key}")
                    else:
                        # No file metadata - treat as hit
                        self._stats.cache_hits += 1
                        self._submitted.pop(key, None)
                        cached_result = cache_entry.result

            # Not in cache or invalidated - check in-flight or start compute
            if cached_result is None:
                if key in self._in_flight:
                    # Already in-flight, we'll wait
                    future = self._in_flight[key].future
                    self._stats.coalesced_requests += 1
                    is_waiting = True
                else:
                    # We're the first - create Future and mark as in-flight
                    future = asyncio.get_running_loop().create_future()
                    self._in_flight[key] = InFlightEntry(future=future, started_at=time.monotonic())
                    self._stats.cache_misses += 1
                    self._stats.total_in_flight += 1
                    self._submitted.pop(key, None)
                    is_compute_owner = True

        # Logging outside lock
        if cached_result is not None:
            if self._is_cached_timeout(cached_result):
                logger.info(f"Cache hit for {key} (timeout result)")
            else:
                logger.info(f"Cache hit for {key}")
            return cached_result
        if is_waiting:
            logger.info(f"Waiting for in-flight request for {key}")

        # Outside the lock: either compute or wait
        if is_compute_owner:
            return await self._do_compute(key, compute_fn, future)
        else:
            return await self._wait_for_result(key, future)

    async def _do_compute(
        self,
        key: str,
        compute_fn: Callable[[], Awaitable[Any]],
        future: "asyncio.Future[Any]",
    ) -> Any:
        """
        Execute compute_fn and propagate result/exception to Future.

        On timeout, caches a timeout result with file_mtime only (file_size=None).
        Within grace period the cached timeout is served as hit to avoid retry storm.
        After grace period the entry is invalidated so the next request retries.
        Timeout entries are evicted by the same file-age rule (file_mtime used).

        Args:
            key: The cache key
            compute_fn: Async function to compute the result
            future: Future to set with result or exception
        """
        async with self._lock:
            self._stats.total_computes += 1

        # Capture file metadata BEFORE compute to avoid race condition:
        # If file changes during compute, metadata represents what we analyzed
        file_mtime, file_size = self._get_file_metadata(key)

        logger.info(f"Computing result for {key} (timeout={self._compute_timeout}s)")

        try:
            result = await asyncio.wait_for(compute_fn(), timeout=self._compute_timeout)

            # Cache the result with pre-captured file metadata
            async with self._lock:
                self._cache[key] = CacheEntry(
                    result=result,
                    cached_at=time.monotonic(),
                    file_mtime=file_mtime,
                    file_size=file_size,
                )
                self._in_flight.pop(key, None)

            # Set result on future (propagates to all waiters)
            if not future.done():
                future.set_result(result)
            logger.debug(f"Cached result for {key}")
            return result

        except asyncio.TimeoutError:
            # Cache timeout result so subsequent GETs don't trigger new computes
            timeout_result = {
                "module": "log_analyzer",
                "state": "timeout",
                "result": [],
                "error": f"LLM analysis timed out after {self._compute_timeout}s",
            }

            # Store file_mtime only (not size) so: cleanup evicts after 14 days;
            # next request sees size=None and invalidates, triggering a retry.
            async with self._lock:
                self._stats.compute_timeouts += 1
                self._in_flight.pop(key, None)
                self._cache[key] = CacheEntry(
                    result=timeout_result,
                    cached_at=time.monotonic(),
                    file_mtime=file_mtime,
                    file_size=None,
                )

            logger.error(f"Compute timed out for {key} after {self._compute_timeout}s")

            if not future.done():
                future.set_result(timeout_result)
            return timeout_result

        except asyncio.CancelledError:
            # Handle cancellation specially (e.g., system shutdown)
            async with self._lock:
                self._stats.compute_errors += 1
                self._in_flight.pop(key, None)
            logger.warning(f"Compute cancelled for {key}")
            if not future.done():
                future.cancel()
            raise

        except BaseException as e:
            # Catch BaseException to handle other edge cases
            async with self._lock:
                self._stats.compute_errors += 1
                self._in_flight.pop(key, None)
            logger.error(f"Failed to compute result for {key}: {e}", exc_info=True)
            if not future.done():
                future.set_exception(e)
            raise

    async def _wait_for_result(self, key: str, future: "asyncio.Future[Any]") -> Any:
        """
        Wait for an in-flight computation to complete.

        Args:
            key: The cache key (for logging)
            future: Future to await

        Returns:
            The computed result

        Raises:
            Exception from compute_fn if it failed
            asyncio.TimeoutError if wait times out
        """
        # Use shield to prevent cancellation from affecting the shared future
        # Add safety margin to timeout in case of scheduling delays
        wait_timeout = self._compute_timeout + 60

        try:
            result = await asyncio.wait_for(asyncio.shield(future), timeout=wait_timeout)
            logger.debug(f"Got result from in-flight request for {key}")
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for in-flight request for {key}")
            raise
        except asyncio.CancelledError:
            # If our wait was cancelled, check if we can still get the result
            if future.done() and not future.cancelled():
                return future.result()
            raise
