#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""HTTP service wrapper for LogAnalyzer.

This module provides AttributionService, a thin wrapper around the library's
LogAnalyzer that adds HTTP-specific concerns like Settings and dataflow posting.

For direct Python usage without HTTP, use LogAnalyzer directly:
    from nvidia_resiliency_ext.attribution import LogAnalyzer, AnalyzerConfig
"""

import json
import logging
import os
from typing import Any

from nvidia_resiliency_ext.attribution import (
    AnalysisResult,
    AnalyzerConfig,
    AnalyzerError,
    CacheResult,
    FilePreviewResult,
    InflightResult,
    LogAnalyzer,
    SplitlogAnalysisResult,
    SubmitResult,
    SubmittedResult,
)
from nvidia_resiliency_ext.attribution.postprocessing import get_dataflow_stats, get_slack_stats

from .config import PRINT_PREVIEW_MAX_BYTES, Settings

logger = logging.getLogger(__name__)


# Re-export result types for convenience
__all__ = [
    "AttributionService",
    "AnalyzerError",  # Use library error type directly
    "AnalysisResult",
    "SubmitResult",
    "SplitlogAnalysisResult",
    "FilePreviewResult",
]


class AttributionService:
    """
    HTTP service wrapper for LogAnalyzer.

    This class wraps the library's LogAnalyzer and adds HTTP-specific
    concerns like Settings conversion and dataflow posting.

    For direct Python usage without HTTP, use LogAnalyzer directly:
        from nvidia_resiliency_ext.attribution import LogAnalyzer, AnalyzerConfig

        config = AnalyzerConfig(allowed_root="/logs")
        analyzer = LogAnalyzer(config)
        result = await analyzer.analyze("/logs/slurm-12345.out")
    """

    def __init__(self, cfg: Settings):
        """
        Initialize the attribution service.

        Args:
            cfg: Application settings (ALLOWED_ROOT must be validated via setup())
        """
        self.cfg = cfg

        # Convert HTTP Settings to library AnalyzerConfig
        # Only pass non-None values; library provides defaults for the rest
        analyzer_kwargs: dict[str, Any] = {
            "allowed_root": cfg.ALLOWED_ROOT,
        }
        # Optional overrides - only pass if explicitly set
        if cfg.COMPUTE_TIMEOUT is not None:
            analyzer_kwargs["compute_timeout"] = cfg.COMPUTE_TIMEOUT
        if cfg.CACHE_GRACE_PERIOD_SECONDS:
            analyzer_kwargs["grace_period_seconds"] = cfg.CACHE_GRACE_PERIOD_SECONDS
        if cfg.LLM_MODEL is not None:
            analyzer_kwargs["llm_model"] = cfg.LLM_MODEL
        if cfg.LLM_TEMPERATURE is not None:
            analyzer_kwargs["llm_temperature"] = cfg.LLM_TEMPERATURE
        if cfg.LLM_TOP_P is not None:
            analyzer_kwargs["llm_top_p"] = cfg.LLM_TOP_P
        if cfg.LLM_MAX_TOKENS is not None:
            analyzer_kwargs["llm_max_tokens"] = cfg.LLM_MAX_TOKENS

        analyzer_config = AnalyzerConfig(**analyzer_kwargs)

        # Create library LogAnalyzer (calls postprocessing.post_results when it has results)
        self._analyzer = LogAnalyzer(config=analyzer_config)

        timeout_str = f"{cfg.COMPUTE_TIMEOUT}s" if cfg.COMPUTE_TIMEOUT else "default"
        logger.info(
            f"Initialized AttributionService with ALLOWED_ROOT={cfg.ALLOWED_ROOT}, "
            f"compute_timeout={timeout_str}"
        )

    def shutdown(self) -> None:
        """Shutdown the service and stop background threads."""
        self._analyzer.shutdown()
        logger.info("AttributionService shutdown complete")

    def save_cache(self, cache_file: str) -> bool:
        """
        Save cache to file for persistence across restarts.

        Args:
            cache_file: Path to save cache JSON

        Returns:
            True if successful, False otherwise
        """
        if not cache_file:
            return False

        try:
            entries = self._analyzer.export_cache()
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

    def load_cache(self, cache_file: str) -> int:
        """
        Load cache from file.

        All entries are validated by file (mtime, size) on import.
        Entries are skipped if file changed, gone, or mtime > 14 days.

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

            imported = self._analyzer.import_cache(entries)
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

        Must be called during app startup before background polling makes
        analyze calls. See LogAnalyzer.set_event_loop() for details.
        """
        self._analyzer.set_event_loop(loop)

    # ─── Delegate to LogAnalyzer ───

    def validate_path(
        self,
        user_path: str,
        *,
        require_regular_file: bool = True,
        reject_empty: bool = False,
    ) -> str | AnalyzerError:
        """Validate and normalize a path. Delegates to LogAnalyzer."""
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
    ) -> SubmitResult | AnalyzerError:
        """Submit a log file for analysis tracking (POST /logs)."""
        return await self._analyzer.submit(log_path, user=user, job_id=job_id)

    async def analyze_log(
        self,
        log_path: str,
        file: str | None = None,
        wl_restart: int | None = None,
    ) -> AnalysisResult | SplitlogAnalysisResult | AnalyzerError:
        """Analyze a log file using LLM (GET /logs)."""
        return await self._analyzer.analyze(log_path, file=file, wl_restart=wl_restart)

    def read_file_preview(
        self, log_path: str, max_bytes: int = PRINT_PREVIEW_MAX_BYTES
    ) -> FilePreviewResult | AnalyzerError:
        """Read the first N bytes of a file for preview (GET /print)."""
        return self._analyzer.read_file_preview(log_path, max_bytes=max_bytes)

    async def get_stats(self) -> dict[str, Any]:
        """Get coalescer, folder tracker, dataflow, and Slack statistics."""
        stats = await self._analyzer.get_stats()
        # Add dataflow stats (service-specific)
        df_stats = get_dataflow_stats()
        stats["dataflow"] = {
            "total_posts": df_stats.total_posts,
            "total_successful": df_stats.successful_posts,
            "total_failed": df_stats.failed_posts,
        }
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
        df_stats = get_dataflow_stats()
        issues = []

        # Use typed helper so we don't depend on raw stats dict keys
        compute = await self._analyzer.get_compute_health_metrics()
        if compute.total > 0:
            error_rate = (compute.errors + compute.timeouts) / compute.total
            if error_rate >= 0.5:
                issues.append(f"high compute error rate: {error_rate:.0%}")
            elif error_rate >= 0.2:
                issues.append(f"elevated compute error rate: {error_rate:.0%}")

        # Check dataflow health
        if df_stats.total_posts > 0:
            df_error_rate = df_stats.failed_posts / df_stats.total_posts
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

        return {"status": status, "issues": issues}
