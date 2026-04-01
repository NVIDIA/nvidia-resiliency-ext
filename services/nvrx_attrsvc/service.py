# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP service wrapper for :class:`~nvidia_resiliency_ext.attribution.analyzer.engine.Analyzer`.

This module provides AttributionService, a thin wrapper around the library
:class:`~nvidia_resiliency_ext.attribution.analyzer.engine.Analyzer` that adds HTTP-specific concerns like Settings and dataflow posting.

For direct Python usage without HTTP, use :class:`~nvidia_resiliency_ext.attribution.analyzer.engine.Analyzer` directly:
    from nvidia_resiliency_ext.attribution import Analyzer
"""

import json
import logging
import os
from typing import Any

from nvidia_resiliency_ext.attribution import (
    Analyzer,
    CacheResult,
    InflightResult,
    LogAnalysisCycleResult,
    LogAnalysisSplitlogResult,
    LogAnalyzerError,
    LogAnalyzerFilePreview,
    LogAnalyzerSubmitResult,
    SubmittedResult,
)
from nvidia_resiliency_ext.attribution.log_analyzer.config import LogSageExecutionConfig
from nvidia_resiliency_ext.attribution.postprocessing import get_posting_stats, get_slack_stats

from .config import PRINT_PREVIEW_MAX_BYTES, Settings

logger = logging.getLogger(__name__)


# Re-export result types for convenience
__all__ = [
    "AttributionService",
    "LogAnalyzerError",  # Use library error type directly
    "LogAnalysisCycleResult",
    "LogAnalyzerSubmitResult",
    "LogAnalysisSplitlogResult",
    "LogAnalyzerFilePreview",
]


class AttributionService:
    """
    HTTP service wrapper for :class:`~nvidia_resiliency_ext.attribution.analyzer.engine.Analyzer`.

    This class wraps the library :class:`~nvidia_resiliency_ext.attribution.analyzer.engine.Analyzer`
    and adds HTTP-specific concerns like Settings conversion and dataflow posting.

    For direct Python usage without HTTP, use :class:`~nvidia_resiliency_ext.attribution.analyzer.engine.Analyzer` directly:
        from nvidia_resiliency_ext.attribution import Analyzer

        analyzer = Analyzer(allowed_root="/logs", use_lib_log_analysis=False)
        result = await analyzer.analyze("/logs/slurm-12345.out")
    """

    def __init__(self, cfg: Settings):
        """
        Initialize the attribution service.

        Args:
            cfg: Application settings (ALLOWED_ROOT must be validated via setup())
        """
        self.cfg = cfg

        coalescing_kwargs: dict[str, Any] = {}
        if cfg.COMPUTE_TIMEOUT is not None:
            coalescing_kwargs["compute_timeout"] = cfg.COMPUTE_TIMEOUT
        if cfg.CACHE_GRACE_PERIOD_SECONDS:
            coalescing_kwargs["grace_period_seconds"] = cfg.CACHE_GRACE_PERIOD_SECONDS

        use_lib = cfg.ANALYSIS_BACKEND.lower() == "lib"
        log_sage_kwargs: dict[str, Any] = {
            "use_lib_log_analysis": use_lib,
            "mcp_server_log_level": cfg.LOG_LEVEL,
        }
        if cfg.LLM_MODEL is not None:
            log_sage_kwargs["llm_model"] = cfg.LLM_MODEL
        if cfg.LLM_TEMPERATURE is not None:
            log_sage_kwargs["llm_temperature"] = cfg.LLM_TEMPERATURE
        if cfg.LLM_TOP_P is not None:
            log_sage_kwargs["llm_top_p"] = cfg.LLM_TOP_P
        if cfg.LLM_MAX_TOKENS is not None:
            log_sage_kwargs["llm_max_tokens"] = cfg.LLM_MAX_TOKENS
        log_sage = LogSageExecutionConfig(**log_sage_kwargs)

        # Create library Analyzer (calls postprocessing.post_results when it has results)
        self._analyzer = Analyzer(
            allowed_root=cfg.ALLOWED_ROOT,
            log_sage=log_sage,
            **coalescing_kwargs,
        )

        timeout_str = f"{cfg.COMPUTE_TIMEOUT}s" if cfg.COMPUTE_TIMEOUT else "default"
        backend_str = cfg.ANALYSIS_BACKEND
        llm_overrides = [
            k
            for k, v in (
                ("llm_model", cfg.LLM_MODEL),
                ("llm_temperature", cfg.LLM_TEMPERATURE),
                ("llm_top_p", cfg.LLM_TOP_P),
                ("llm_max_tokens", cfg.LLM_MAX_TOKENS),
            )
            if v is not None
        ]
        llm_str = f", llm_overrides={llm_overrides}" if llm_overrides else ""
        logger.info(
            f"Initialized AttributionService with ALLOWED_ROOT={cfg.ALLOWED_ROOT}, "
            f"compute_timeout={timeout_str}, analysis_backend={backend_str}"
            f"{llm_str}"
        )
        logger.info(
            "Analyzer LLM wiring: model=%r temperature=%s top_p=%s max_tokens=%s (from LogSageExecutionConfig)",
            log_sage.llm_model,
            log_sage.llm_temperature,
            log_sage.llm_top_p,
            log_sage.llm_max_tokens,
        )

    def shutdown(self) -> None:
        """Shutdown the service and stop background threads.

        For full shutdown including MCP client cleanup, use shutdown_async() from
        an async context (e.g. app lifespan).
        """
        self._analyzer.shutdown()
        logger.info("AttributionService shutdown complete")

    async def shutdown_async(self) -> None:
        """Shutdown the service including MCP client. Call from async context (e.g. lifespan)."""
        await self._analyzer.shutdown_async()
        logger.info("AttributionService shutdown complete")

    async def save_cache(self, cache_file: str) -> bool:
        """
        Save cache to file for persistence across restarts.

        Must be awaited so export runs under the coalescer lock (safe during graceful
        shutdown while other tasks may still finish work).

        Args:
            cache_file: Path to save cache JSON

        Returns:
            True if successful, False otherwise
        """
        if not cache_file:
            return False

        try:
            entries = await self._analyzer.export_cache()
            if not entries:
                logger.debug("No cache entries to save")
                return True

            # Ensure parent directory exists
            cache_dir = os.path.dirname(cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

            # Write atomically using temp file
            temp_file = f"{cache_file}.tmp"
            with open(temp_file, "w") as f:
                json.dump({"version": 1, "entries": entries}, f)
            os.replace(temp_file, cache_file)

            logger.info(f"Saved {len(entries)} cache entries to {cache_file}")
            return True

        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_file}: {e}")
            return False

    async def load_cache(self, cache_file: str) -> int:
        """
        Load cache from file.

        All entries are validated by file (mtime, size) on import.
        Entries are skipped if file changed, gone, or mtime > 14 days.

        Must be awaited so import runs under the coalescer lock.

        Args:
            cache_file: Path to cache JSON file

        Returns:
            Number of entries loaded, or 0 if file doesn't exist or is invalid
        """
        if not cache_file or not os.path.exists(cache_file):
            return 0

        try:
            with open(cache_file) as f:
                data = json.load(f)

            # Check version
            version = data.get("version", 0)
            if version != 1:
                logger.warning(f"Unknown cache file version: {version}")
                return 0

            entries = data.get("entries", [])
            if not entries:
                return 0

            imported = await self._analyzer.import_cache(entries)
            logger.info(f"Loaded {imported} cache entries from {cache_file}")
            return imported

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in cache file {cache_file}: {e}")
            return 0
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_file}: {e}")
            return 0

    def set_event_loop(self, loop: Any) -> None:
        """Set the main event loop for background thread callbacks.

        Args:
            loop: asyncio.AbstractEventLoop from the main thread

        Call during app startup as soon as the running loop exists. Until then,
        splitlog fire-and-forget schedules are skipped (logged) rather than raising
        from the poll thread. See ``Analyzer.set_event_loop()`` for details.
        """
        self._analyzer.set_event_loop(loop)

    async def connect_mcp(self) -> None:
        """Connect the MCP client when using MCP analysis backend. No-op for lib backend."""
        await self._analyzer.connect_mcp()

    async def check_mcp_health(self, timeout_seconds: float = 5.0) -> tuple[str, str]:
        """Check MCP backend health. Returns (status, message). See Analyzer.check_mcp_health."""
        return await self._analyzer.check_mcp_health(timeout_seconds)

    # ─── Delegate to Analyzer ───

    def validate_path(
        self,
        user_path: str,
        *,
        require_regular_file: bool = True,
        reject_empty: bool = False,
    ) -> str | LogAnalyzerError:
        """Validate and normalize a path. Delegates to Analyzer."""
        return self._analyzer.validate_path(
            user_path,
            require_regular_file=require_regular_file,
            reject_empty=reject_empty,
        )

    async def submit_log(
        self,
        log_path: str,
        user: str = "unknown",
        job_id: str | None = None,
    ) -> LogAnalyzerSubmitResult | LogAnalyzerError:
        """Submit a log file for analysis tracking (POST /logs)."""
        return await self._analyzer.submit(log_path, user=user, job_id=job_id)

    async def analyze_log(
        self,
        log_path: str,
        file: str | None = None,
        wl_restart: int | None = None,
    ) -> LogAnalysisCycleResult | LogAnalysisSplitlogResult | LogAnalyzerError:
        """Analyze a log file using LLM (GET /logs).

        Caching: the analyzer's RequestCoalescer caches results per file path.
        Repeat requests for the same path (any wl_restart) get the cached full
        result; the analyzer returns the slice for the requested wl_restart.
        """
        return await self._analyzer.analyze(log_path, file=file, wl_restart=wl_restart)

    def read_file_preview(
        self, log_path: str, max_bytes: int = PRINT_PREVIEW_MAX_BYTES
    ) -> LogAnalyzerFilePreview | LogAnalyzerError:
        """Read the first N bytes of a file for preview (GET /print)."""
        return self._analyzer.read_file_preview(log_path, max_bytes=max_bytes)

    async def get_stats(self) -> dict[str, Any]:
        """Get coalescer, folder tracker, posting (ES/dataflow), and Slack statistics."""
        stats = await self._analyzer.get_stats()
        ps = get_posting_stats()
        posting = {
            "total_posts": ps.total_posts,
            "total_successful": ps.successful_posts,
            "total_failed": ps.failed_posts,
        }
        stats["posting"] = posting
        stats["dataflow"] = posting  # legacy alias; same counters as ``posting``
        # Add Slack stats
        slack_stats = get_slack_stats()
        stats["slack"] = {
            "total_attempts": slack_stats.total_attempts,
            "total_successful": slack_stats.total_successful,
            "total_failed": slack_stats.total_failed,
            "user_lookups": slack_stats.user_lookups,
            "user_not_found": slack_stats.user_not_found,
        }
        return stats

    async def get_cache(self) -> CacheResult:
        """Get current cache contents."""
        return await self._analyzer.get_cache()

    async def get_inflight(self) -> InflightResult:
        """Get currently in-flight requests."""
        return await self._analyzer.get_inflight()

    async def get_submitted(self) -> SubmittedResult:
        """Get submitted paths."""
        return await self._analyzer.get_submitted()

    def get_all_jobs(self) -> dict[str, Any]:
        """Get all tracked jobs."""
        return self._analyzer.get_all_jobs()

    async def get_health(self) -> dict[str, Any]:
        """Get health status based on recent statistics."""
        posting_stats = get_posting_stats()
        issues = []

        # MCP backend health (when using MCP)
        mcp_status, mcp_message = await self.check_mcp_health()
        if mcp_status == "disconnected":
            issues.append(f"MCP disconnected: {mcp_message}")

        # Use typed helper so we don't depend on raw stats dict keys
        compute = await self._analyzer.get_compute_health_metrics()
        if compute.total > 0:
            error_rate = (compute.errors + compute.timeouts) / compute.total
            if error_rate >= 0.5:
                issues.append(f"high compute error rate: {error_rate:.0%}")
            elif error_rate >= 0.2:
                issues.append(f"elevated compute error rate: {error_rate:.0%}")

        # Check dataflow health
        if posting_stats.total_posts > 0:
            df_error_rate = posting_stats.failed_posts / posting_stats.total_posts
            if df_error_rate >= 0.5:
                issues.append(f"high dataflow failure rate: {df_error_rate:.0%}")
            elif df_error_rate >= 0.2:
                issues.append(f"elevated dataflow failure rate: {df_error_rate:.0%}")

        if not issues:
            status = "ok"
        elif any("high" in issue for issue in issues):
            status = "fail"
        else:
            status = "degraded"

        result: dict[str, Any] = {"status": status, "issues": issues}
        if mcp_status != "unused":
            result["mcp"] = {"status": mcp_status, "message": mcp_message}
        return result
