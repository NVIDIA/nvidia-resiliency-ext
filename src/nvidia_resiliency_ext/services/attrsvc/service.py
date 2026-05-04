# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP adapter for the attribution controller boundary.

This module keeps FastAPI-facing concerns in ``services/attrsvc`` while
delegating attribution ownership to
:class:`nvidia_resiliency_ext.attribution.controller.AttributionController`.
"""

import asyncio
import logging
from typing import Any

from nvidia_resiliency_ext.attribution.coalescing import (
    CacheResult,
    InflightResult,
    SubmittedResult,
)
from nvidia_resiliency_ext.attribution.controller import (
    AttributionAnalysisConfig,
    AttributionCacheConfig,
    AttributionController,
    AttributionControllerConfig,
    AttributionPostprocessingConfig,
)
from nvidia_resiliency_ext.attribution.orchestration.types import (
    AttributionRecommendation,
    LogAnalysisCycleResult,
    LogAnalysisSplitlogResult,
    LogAnalyzerError,
    LogAnalyzerFilePreview,
    LogAnalyzerSubmitResult,
)

from .config import PRINT_PREVIEW_MAX_BYTES, Settings

logger = logging.getLogger(__name__)


# Re-export result types for convenience
__all__ = [
    "AttributionHttpAdapter",
    "AttributionRecommendation",
    "LogAnalyzerError",
    "LogAnalysisCycleResult",
    "LogAnalyzerSubmitResult",
    "LogAnalysisSplitlogResult",
    "LogAnalyzerFilePreview",
]


def _controller_config_from_settings(cfg: Settings) -> AttributionControllerConfig:
    """Translate HTTP service settings into controller startup config."""
    return AttributionControllerConfig(
        allowed_root=cfg.ALLOWED_ROOT,
        analysis=AttributionAnalysisConfig(
            engine_backend=cfg.ANALYSIS_BACKEND,
            mcp_server_log_level=cfg.LOG_LEVEL,
            llm_model=cfg.LLM_MODEL,
            llm_base_url=cfg.LLM_BASE_URL,
            llm_temperature=cfg.LLM_TEMPERATURE,
            llm_top_p=cfg.LLM_TOP_P,
            llm_max_tokens=cfg.LLM_MAX_TOKENS,
        ),
        cache=AttributionCacheConfig(
            compute_timeout=cfg.COMPUTE_TIMEOUT,
            grace_period_seconds=(
                cfg.CACHE_GRACE_PERIOD_SECONDS if cfg.CACHE_GRACE_PERIOD_SECONDS else None
            ),
            cache_file=cfg.CACHE_FILE,
        ),
        postprocessing=AttributionPostprocessingConfig(
            cluster_name=cfg.CLUSTER_NAME,
            dataflow_index=cfg.DATAFLOW_INDEX,
            slack_bot_token=(cfg.SLACK_BOT_TOKEN or "").strip() or None,
            slack_channel=cfg.SLACK_CHANNEL,
        ),
    )


class AttributionHttpAdapter:
    """
    HTTP adapter facade for :class:`AttributionController`.

    ``AttributionHttpAdapter`` owns Settings conversion and keeps the public service
    method names stable for FastAPI routes. The controller owns attribution
    analysis, cache persistence, side-effect stats, and health/status policy.
    """

    def __init__(self, cfg: Settings):
        """
        Initialize the attribution HTTP adapter.

        Args:
            cfg: Application settings (ALLOWED_ROOT must be validated via setup())
        """
        self.cfg = cfg
        self._controller = AttributionController(_controller_config_from_settings(cfg))
        logger.info("Initialized AttributionHttpAdapter")

    def shutdown(self) -> None:
        """Shutdown the adapter and stop background threads."""
        self._controller.shutdown()
        logger.info("AttributionHttpAdapter shutdown complete")

    async def shutdown_async(self) -> None:
        """Shutdown the adapter including MCP client cleanup."""
        await self._controller.shutdown_async()
        logger.info("AttributionHttpAdapter shutdown complete")

    async def start(self, loop: asyncio.AbstractEventLoop | None = None) -> dict[str, Any]:
        """Start controller-owned runtime dependencies."""
        return await self._controller.start(loop)

    async def save_cache(self, cache_file: str | None = None) -> bool:
        """Save controller cache to file for persistence across restarts."""
        return await self._controller.save_cache(cache_file)

    async def load_cache(self, cache_file: str | None = None) -> int:
        """Load controller cache from file."""
        return await self._controller.load_cache(cache_file)

    async def check_mcp_health(self, timeout_seconds: float = 5.0) -> tuple[str, str]:
        """Check MCP backend health."""
        return await self._controller.check_mcp_health(timeout_seconds)

    def validate_path(
        self,
        user_path: str,
        *,
        require_regular_file: bool = True,
        reject_empty: bool = False,
    ) -> str | LogAnalyzerError:
        """Validate and normalize a path."""
        return self._controller.validate_path(
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
        """Submit a log file for analysis tracking."""
        return await self._controller.submit_log(log_path, user=user, job_id=job_id)

    async def analyze_log(
        self,
        log_path: str,
        file: str | None = None,
        wl_restart: int | None = None,
    ) -> LogAnalysisCycleResult | LogAnalysisSplitlogResult | LogAnalyzerError:
        """Analyze a log file using the configured attribution backend."""
        return await self._controller.analyze_log(
            log_path,
            file=file,
            wl_restart=wl_restart,
        )

    def read_file_preview(
        self, log_path: str, max_bytes: int = PRINT_PREVIEW_MAX_BYTES
    ) -> LogAnalyzerFilePreview | LogAnalyzerError:
        """Read the first N bytes of a file for preview."""
        return self._controller.read_file_preview(log_path, max_bytes=max_bytes)

    async def get_stats(self) -> dict[str, Any]:
        """Get controller, cache, posting/dataflow, and Slack statistics."""
        return await self._controller.get_stats()

    async def get_cache(self) -> CacheResult:
        """Get current cache contents."""
        return await self._controller.get_cache()

    async def get_inflight(self) -> InflightResult:
        """Get currently in-flight requests."""
        return await self._controller.get_inflight()

    async def get_submitted(self) -> SubmittedResult:
        """Get submitted paths."""
        return await self._controller.get_submitted()

    def get_all_jobs(self) -> dict[str, Any]:
        """Get all tracked jobs."""
        return self._controller.get_all_jobs()

    async def get_health(self) -> dict[str, Any]:
        """Get adapter health from controller status."""
        return await self._controller.status()
