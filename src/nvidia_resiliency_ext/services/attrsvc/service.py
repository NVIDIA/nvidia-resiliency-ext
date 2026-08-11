# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP adapter selecting the direct Restart Agent backend.

This module keeps FastAPI-facing concerns in ``services/attrsvc``. The default
``lib`` backend owns a direct Restart Agent runtime.
"""

import asyncio
import logging
from functools import partial
from typing import Any, Protocol

from nvidia_resiliency_ext.attribution.coalescing import (
    CacheResult,
    InflightResult,
    SubmittedResult,
)
from nvidia_resiliency_ext.attribution.orchestration.types import (
    AttributionRecommendation,
    LogAnalysisCycleResult,
    LogAnalysisSplitlogResult,
    LogAnalyzerError,
    LogAnalyzerFilePreview,
    LogAnalyzerSubmitResult,
)
from nvidia_resiliency_ext.attribution.restart_agent import (
    ProgressiveL0Accumulator,
    build_restart_agent_runtime,
)
from nvidia_resiliency_ext.attribution.restart_agent.infrastructure.log_source import (
    ChunkedLogReader,
)

from .config import PRINT_PREVIEW_MAX_BYTES, Settings
from .restart_agent_backend import (
    LogConvergencePolicy,
    ProgressiveAnalysisPolicy,
    RestartAgentServiceBackend,
)
from .restart_agent_config import restart_agent_config_from_settings

logger = logging.getLogger(__name__)

# Re-export result types for convenience
__all__ = [
    "AttributionHttpAdapter",
    "AttributionServiceBackend",
    "AttributionRecommendation",
    "LogAnalyzerError",
    "LogAnalysisCycleResult",
    "LogAnalyzerSubmitResult",
    "LogAnalysisSplitlogResult",
    "LogAnalyzerFilePreview",
]


class AttributionServiceBackend(Protocol):
    """Structural service contract implemented by direct and legacy backends."""

    def shutdown(self) -> None: ...

    async def shutdown_async(self) -> None: ...

    async def start(self, loop: asyncio.AbstractEventLoop | None = None) -> dict[str, Any]: ...

    async def save_cache(self, cache_file: str | None = None) -> bool: ...

    async def load_cache(self, cache_file: str | None = None) -> int: ...

    async def check_mcp_health(self, timeout_seconds: float = 5.0) -> tuple[str, str]: ...

    def validate_path(
        self,
        user_path: str,
        *,
        require_regular_file: bool = True,
        reject_empty: bool = False,
    ) -> str | LogAnalyzerError: ...

    async def submit_log(
        self,
        log_path: str,
        user: str = "unknown",
        job_id: str | None = None,
        cycle_id: int | None = None,
        analysis_intent: str | None = None,
    ) -> LogAnalyzerSubmitResult | LogAnalyzerError: ...

    async def analyze_log(
        self,
        log_path: str,
        file: str | None = None,
        wl_restart: int | None = None,
        wait: bool = True,
    ) -> LogAnalysisCycleResult | LogAnalysisSplitlogResult | LogAnalyzerError: ...

    def read_file_preview(
        self, log_path: str, max_bytes: int = PRINT_PREVIEW_MAX_BYTES
    ) -> LogAnalyzerFilePreview | LogAnalyzerError: ...

    async def get_stats(self) -> dict[str, Any]: ...

    async def get_cache(self) -> CacheResult: ...

    async def get_inflight(self) -> InflightResult: ...

    async def get_submitted(self) -> SubmittedResult: ...

    def get_all_jobs(self) -> dict[str, Any]: ...

    async def status(self) -> dict[str, Any]: ...


class AttributionHttpAdapter:
    """
    HTTP adapter facade over the selected attribution implementation.

    ``AttributionHttpAdapter`` owns Settings conversion and keeps the public service
    method names stable for FastAPI routes.
    """

    def __init__(self, cfg: Settings):
        """
        Initialize the attribution HTTP adapter.

        Args:
            cfg: Application settings (ALLOWED_ROOT must be validated via setup())
        """
        self.cfg = cfg
        restart_config = restart_agent_config_from_settings(cfg)
        self._backend: AttributionServiceBackend = RestartAgentServiceBackend(
            allowed_root=cfg.ALLOWED_ROOT,
            runtime=build_restart_agent_runtime(restart_config),
            config=restart_config,
            convergence=LogConvergencePolicy(
                quiet_seconds=cfg.RESTART_AGENT_LOG_QUIET_SECONDS,
                max_wait_seconds=cfg.RESTART_AGENT_LOG_MAX_WAIT_SECONDS,
                poll_seconds=cfg.RESTART_AGENT_LOG_POLL_SECONDS,
            ),
            progressive=ProgressiveAnalysisPolicy(
                enabled=(
                    cfg.RESTART_AGENT_PROGRESSIVE_ENABLED and cfg.PROGRESSIVE_ANALYSIS != "off"
                ),
                pre_end_poll_seconds=cfg.RESTART_AGENT_PRE_END_POLL_SECONDS,
                active_idle_seconds=cfg.RESTART_AGENT_ACTIVE_IDLE_SECONDS,
                max_active_states=cfg.RESTART_AGENT_MAX_ACTIVE_STATES,
                max_completed_results=cfg.RESTART_AGENT_MAX_COMPLETED_RESULTS,
            ),
            accumulator_factory=partial(
                ProgressiveL0Accumulator,
                reader=ChunkedLogReader(
                    chunk_bytes=restart_config.l0_source.chunk_size_bytes,
                    read_mode=restart_config.l0_source.read_mode,
                ),
            ),
        )
        logger.info("Initialized AttributionHttpAdapter backend=%s", cfg.ANALYSIS_BACKEND)

    def shutdown(self) -> None:
        """Shutdown the adapter and stop background threads."""
        self._backend.shutdown()
        logger.info("AttributionHttpAdapter shutdown complete")

    async def shutdown_async(self) -> None:
        """Shutdown the adapter and stop backend-owned async resources."""
        await self._backend.shutdown_async()
        logger.info("AttributionHttpAdapter shutdown complete")

    async def start(self, loop: asyncio.AbstractEventLoop | None = None) -> dict[str, Any]:
        """Start backend-owned runtime dependencies."""
        return await self._backend.start(loop)

    async def save_cache(self, cache_file: str | None = None) -> bool:
        """Save backend cache to file when the backend supports persistence."""
        return await self._backend.save_cache(cache_file)

    async def load_cache(self, cache_file: str | None = None) -> int:
        """Load backend cache from file when the backend supports persistence."""
        return await self._backend.load_cache(cache_file)

    async def check_mcp_health(self, timeout_seconds: float = 5.0) -> tuple[str, str]:
        """Return MCP health compatibility status."""
        return await self._backend.check_mcp_health(timeout_seconds)

    def validate_path(
        self,
        user_path: str,
        *,
        require_regular_file: bool = True,
        reject_empty: bool = False,
    ) -> str | LogAnalyzerError:
        """Validate and normalize a path."""
        return self._backend.validate_path(
            user_path,
            require_regular_file=require_regular_file,
            reject_empty=reject_empty,
        )

    async def submit_log(
        self,
        log_path: str,
        user: str = "unknown",
        job_id: str | None = None,
        cycle_id: int | None = None,
        analysis_intent: str | None = None,
    ) -> LogAnalyzerSubmitResult | LogAnalyzerError:
        """Submit a log file for analysis tracking."""
        return await self._backend.submit_log(
            log_path,
            user=user,
            job_id=job_id,
            cycle_id=cycle_id,
            analysis_intent=analysis_intent,
        )

    async def analyze_log(
        self,
        log_path: str,
        file: str | None = None,
        wl_restart: int | None = None,
        wait: bool = True,
    ) -> LogAnalysisCycleResult | LogAnalysisSplitlogResult | LogAnalyzerError:
        """Analyze a log file using the direct Restart Agent backend."""
        return await self._backend.analyze_log(
            log_path,
            file=file,
            wl_restart=wl_restart,
            wait=wait,
        )

    def read_file_preview(
        self, log_path: str, max_bytes: int = PRINT_PREVIEW_MAX_BYTES
    ) -> LogAnalyzerFilePreview | LogAnalyzerError:
        """Read the first N bytes of a file for preview."""
        return self._backend.read_file_preview(log_path, max_bytes=max_bytes)

    async def get_stats(self) -> dict[str, Any]:
        """Get direct Restart Agent backend statistics."""
        return await self._backend.get_stats()

    async def get_cache(self) -> CacheResult:
        """Get current cache contents."""
        return await self._backend.get_cache()

    async def get_inflight(self) -> InflightResult:
        """Get currently in-flight requests."""
        return await self._backend.get_inflight()

    async def get_submitted(self) -> SubmittedResult:
        """Get submitted paths."""
        return await self._backend.get_submitted()

    def get_all_jobs(self) -> dict[str, Any]:
        """Get all tracked jobs."""
        return self._backend.get_all_jobs()

    async def get_health(self) -> dict[str, Any]:
        """Get adapter health from controller status."""
        return await self._backend.status()
