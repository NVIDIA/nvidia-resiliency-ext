# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attribution controller boundary.

``AttributionController`` owns attribution lifecycle concerns above the low-level
``Analyzer``: config translation, cache persistence, service health/status, and
postprocessing stats. It can be embedded by a service process today and hosted as
its own process for ft_launcher integration later.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from nvidia_resiliency_ext.attribution.analyzer import Analyzer
from nvidia_resiliency_ext.attribution.api_keys import (
    llm_api_key_missing_message,
    load_llm_api_key,
    load_slack_bot_token,
)
from nvidia_resiliency_ext.attribution.coalescing import (
    CacheResult,
    InflightResult,
    SubmittedResult,
)
from nvidia_resiliency_ext.attribution.orchestration.config import LogSageExecutionConfig
from nvidia_resiliency_ext.attribution.orchestration.types import (
    LogAnalysisCycleResult,
    LogAnalysisSplitlogResult,
    LogAnalyzerError,
    LogAnalyzerFilePreview,
    LogAnalyzerSubmitResult,
)
from nvidia_resiliency_ext.attribution.postprocessing import ResultPoster
from nvidia_resiliency_ext.attribution.postprocessing import configure as configure_postprocessing
from nvidia_resiliency_ext.attribution.postprocessing import get_posting_stats, get_slack_stats
from nvidia_resiliency_ext.attribution.postprocessing.post_backend import post

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttributionAnalysisConfig:
    """Analysis engine configuration."""

    engine_backend: str = "mcp"
    mcp_server_log_level: str = "INFO"
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_temperature: float | None = None
    llm_top_p: float | None = None
    llm_max_tokens: int | None = None


@dataclass(frozen=True)
class AttributionCacheConfig:
    """Request coalescing and persistence configuration.

    ``cache_file`` is normalized to an absolute, user-expanded path when set so
    direct controller callers get the same path contract as service Settings.
    """

    compute_timeout: float | None = None
    grace_period_seconds: float | None = None
    cache_file: str = ""

    def __post_init__(self) -> None:
        cache_file = (self.cache_file or "").strip()
        if cache_file:
            cache_file = os.path.abspath(os.path.expanduser(cache_file))
        object.__setattr__(self, "cache_file", cache_file)


@dataclass(frozen=True)
class AttributionPostprocessingConfig:
    """Posting and notification configuration for attribution side effects."""

    cluster_name: str = ""
    dataflow_index: str = ""
    slack_bot_token: str | None = None
    slack_channel: str = ""
    enable_default_poster: bool = True


@dataclass(frozen=True)
class AttributionCredentialsConfig:
    """Credential policy for attribution engines.

    The current LogSage implementations load the LLM API key via
    :func:`load_llm_api_key`; the controller validates that policy at startup so
    callers do not need to know which lower layer will first need the key.
    """

    require_llm_api_key: bool = True


@dataclass(frozen=True)
class AttributionControllerConfig:
    """Startup configuration for :class:`AttributionController`.

    Runtime request metadata belongs in ``submit_log`` or ``analyze_log``; this
    config is the process-level policy and dependency wiring for the controller.
    """

    allowed_root: str
    analysis: AttributionAnalysisConfig = field(default_factory=AttributionAnalysisConfig)
    cache: AttributionCacheConfig = field(default_factory=AttributionCacheConfig)
    postprocessing: AttributionPostprocessingConfig = field(
        default_factory=AttributionPostprocessingConfig
    )
    credentials: AttributionCredentialsConfig = field(default_factory=AttributionCredentialsConfig)


class AttributionController:
    """Orchestrates attribution across analysis, cache, and side effects.

    The controller exposes the boundary we want callers to depend on. The
    current implementation delegates analysis to ``Analyzer`` but keeps cache
    persistence and health/status policy out of the HTTP service wrapper.
    """

    def __init__(self, config: AttributionControllerConfig):
        self.config = config
        self._engine_backend = self._normalize_engine_backend(config.analysis.engine_backend)
        self._llm_api_key_present = self._validate_llm_api_key()
        self._slack_configured = self._configure_postprocessing()

        coalescing_kwargs: dict[str, Any] = {}
        if config.cache.compute_timeout is not None:
            coalescing_kwargs["compute_timeout"] = config.cache.compute_timeout
        if config.cache.grace_period_seconds is not None:
            coalescing_kwargs["grace_period_seconds"] = config.cache.grace_period_seconds

        use_lib = self._engine_backend == "lib"
        analyzer_engine_kwargs: dict[str, Any] = {
            "use_lib_log_analysis": use_lib,
            "mcp_server_log_level": config.analysis.mcp_server_log_level,
        }
        if config.analysis.llm_model is not None:
            analyzer_engine_kwargs["llm_model"] = config.analysis.llm_model
        if config.analysis.llm_base_url is not None:
            analyzer_engine_kwargs["llm_base_url"] = config.analysis.llm_base_url
        if config.analysis.llm_temperature is not None:
            analyzer_engine_kwargs["llm_temperature"] = config.analysis.llm_temperature
        if config.analysis.llm_top_p is not None:
            analyzer_engine_kwargs["llm_top_p"] = config.analysis.llm_top_p
        if config.analysis.llm_max_tokens is not None:
            analyzer_engine_kwargs["llm_max_tokens"] = config.analysis.llm_max_tokens
        analyzer_engine = LogSageExecutionConfig(**analyzer_engine_kwargs)

        self._analyzer = Analyzer(
            allowed_root=config.allowed_root,
            log_sage=analyzer_engine,
            **coalescing_kwargs,
        )

        timeout_str = (
            f"{config.cache.compute_timeout}s" if config.cache.compute_timeout else "default"
        )
        llm_overrides = [
            k
            for k, v in (
                ("llm_model", config.analysis.llm_model),
                ("llm_base_url", config.analysis.llm_base_url),
                ("llm_temperature", config.analysis.llm_temperature),
                ("llm_top_p", config.analysis.llm_top_p),
                ("llm_max_tokens", config.analysis.llm_max_tokens),
            )
            if v is not None
        ]
        llm_str = f", llm_overrides={llm_overrides}" if llm_overrides else ""
        logger.info(
            "Initialized AttributionController with allowed_root=%s, compute_timeout=%s, "
            "analysis_engine_backend=%s%s",
            config.allowed_root,
            timeout_str,
            self._engine_backend,
            llm_str,
        )
        logger.info(
            "Analyzer engine LLM wiring: model=%r base_url=%s temperature=%s top_p=%s "
            "max_tokens=%s",
            analyzer_engine.llm_model,
            analyzer_engine.llm_base_url,
            analyzer_engine.llm_temperature,
            analyzer_engine.llm_top_p,
            analyzer_engine.llm_max_tokens,
        )

    def shutdown(self) -> None:
        """Shutdown the controller and stop background threads."""
        self._analyzer.shutdown()
        logger.info("AttributionController shutdown complete")

    async def shutdown_async(self) -> None:
        """Shutdown the controller including MCP client cleanup."""
        await self.save_cache()
        await self._analyzer.shutdown_async()
        logger.info("AttributionController shutdown complete")

    async def start(self, loop: asyncio.AbstractEventLoop | None = None) -> dict[str, Any]:
        """Start controller-owned runtime dependencies.

        The caller provides the process event loop when one exists; the
        controller decides which internal dependencies need startup work.
        """
        if loop is not None:
            self._analyzer.set_event_loop(loop)
        await self._connect_mcp()
        loaded = await self.load_cache()
        return {"cache_entries_loaded": loaded}

    async def save_cache(self, cache_file: str | None = None) -> bool:
        """Persist the analyzer cache using the controller cache-file format."""
        cache_file = cache_file if cache_file is not None else self.config.cache.cache_file
        if not cache_file:
            return False

        try:
            entries = await self._analyzer.export_cache()
            if not entries:
                logger.debug("No cache entries to save")
                return True

            cache_dir = os.path.dirname(cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

            temp_file = f"{cache_file}.tmp"
            with open(temp_file, "w") as f:
                json.dump({"version": 1, "entries": entries}, f)
            os.replace(temp_file, cache_file)

            logger.info("Saved %d cache entries to %s", len(entries), cache_file)
            return True

        except Exception as e:
            logger.warning("Failed to save cache to %s: %s", cache_file, e)
            return False

    async def load_cache(self, cache_file: str | None = None) -> int:
        """Load persisted cache entries into the analyzer cache."""
        cache_file = cache_file if cache_file is not None else self.config.cache.cache_file
        if not cache_file or not os.path.exists(cache_file):
            return 0

        try:
            with open(cache_file) as f:
                data = json.load(f)

            version = data.get("version", 0)
            if version != 1:
                logger.warning("Unknown cache file version: %s", version)
                return 0

            entries = data.get("entries", [])
            if not entries:
                return 0

            imported = await self._analyzer.import_cache(entries)
            logger.info("Loaded %d cache entries from %s", imported, cache_file)
            return imported

        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in cache file %s: %s", cache_file, e)
            return 0
        except Exception as e:
            logger.warning("Failed to load cache from %s: %s", cache_file, e)
            return 0

    async def _connect_mcp(self) -> None:
        """Connect internal MCP dependencies when the analysis engine needs them."""
        await self._analyzer.connect_mcp()

    async def check_mcp_health(self, timeout_seconds: float = 5.0) -> tuple[str, str]:
        """Check MCP backend health. Returns ``(status, message)``."""
        return await self._analyzer.check_mcp_health(timeout_seconds)

    def validate_path(
        self,
        user_path: str,
        *,
        require_regular_file: bool = True,
        reject_empty: bool = False,
    ) -> str | LogAnalyzerError:
        """Validate and normalize a path under the configured root."""
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
        """Submit a log file for attribution tracking."""
        return await self._analyzer.submit(log_path, user=user, job_id=job_id)

    async def analyze_log(
        self,
        log_path: str,
        file: str | None = None,
        wl_restart: int | None = None,
    ) -> LogAnalysisCycleResult | LogAnalysisSplitlogResult | LogAnalyzerError:
        """Analyze a log file and return the attribution result."""
        return await self._analyzer.analyze(log_path, file=file, wl_restart=wl_restart)

    def read_file_preview(
        self, log_path: str, max_bytes: int = 4096
    ) -> LogAnalyzerFilePreview | LogAnalyzerError:
        """Read the first ``max_bytes`` bytes of a log file."""
        return self._analyzer.read_file_preview(log_path, max_bytes=max_bytes)

    async def get_stats(self) -> dict[str, Any]:
        """Get analyzer, cache, dataflow, and Slack statistics."""
        stats = await self._analyzer.get_stats()
        posting_stats = get_posting_stats()
        stats["dataflow"] = {
            "total_posts": posting_stats.total_posts,
            "total_successful": posting_stats.successful_posts,
            "total_failed": posting_stats.failed_posts,
        }

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

    async def status(self) -> dict[str, Any]:
        """Get controller readiness/dependency health, not an attribution result."""
        posting_stats = get_posting_stats()
        issues: list[str] = []

        mcp_status, mcp_message = await self.check_mcp_health()
        if mcp_status == "disconnected":
            issues.append(f"MCP disconnected: {mcp_message}")

        compute = await self._analyzer.get_compute_health_metrics()
        if compute.total > 0:
            error_rate = (compute.errors + compute.timeouts) / compute.total
            if error_rate >= 0.5:
                issues.append(f"high compute error rate: {error_rate:.0%}")
            elif error_rate >= 0.2:
                issues.append(f"elevated compute error rate: {error_rate:.0%}")

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
        cache = await self.get_cache()
        in_flight = await self.get_inflight()
        submitted = await self.get_submitted()
        jobs = self.get_all_jobs()
        result["controller"] = {
            "ready": status != "fail",
            "config": self._status_config_summary(),
            "cache": {
                "count": cache.get("count", 0),
            },
            "in_flight": {
                "count": in_flight.get("count", 0),
            },
            "submitted": {
                "count": submitted.get("count", 0),
            },
            "jobs": {
                mode: payload.get("count", 0)
                for mode, payload in jobs.items()
                if isinstance(payload, dict)
            },
        }
        return result

    async def get_health(self) -> dict[str, Any]:
        """Compatibility alias for callers that use service-style health naming."""
        return await self.status()

    def _status_config_summary(self) -> dict[str, Any]:
        """Return non-secret controller config for status payloads."""
        llm_overrides = [
            k
            for k, v in (
                ("llm_model", self.config.analysis.llm_model),
                ("llm_base_url", self.config.analysis.llm_base_url),
                ("llm_temperature", self.config.analysis.llm_temperature),
                ("llm_top_p", self.config.analysis.llm_top_p),
                ("llm_max_tokens", self.config.analysis.llm_max_tokens),
            )
            if v is not None
        ]
        return {
            "analysis": {
                "engine_backend": self._engine_backend,
                "mcp_server_log_level": self.config.analysis.mcp_server_log_level,
                "llm_overrides": llm_overrides,
                "llm_api_key_present": self._llm_api_key_present,
            },
            "cache": {
                "compute_timeout": self.config.cache.compute_timeout,
                "grace_period_seconds": self.config.cache.grace_period_seconds,
                "persistence_enabled": bool(self.config.cache.cache_file),
            },
            "postprocessing": {
                "cluster_name": self.config.postprocessing.cluster_name,
                "dataflow_index": self.config.postprocessing.dataflow_index,
                "dataflow_enabled": bool(self.config.postprocessing.dataflow_index),
                "slack_channel": self.config.postprocessing.slack_channel,
                "slack_enabled": self._slack_configured,
            },
        }

    def _validate_llm_api_key(self) -> bool:
        if not self.config.credentials.require_llm_api_key:
            return bool(load_llm_api_key())
        llm_key = load_llm_api_key()
        if llm_key:
            return True
        logger.error(
            llm_api_key_missing_message(
                include_empty=True,
                context="Attribution requires a key.",
                suffix="Slack notifications remain optional.",
            )
        )
        raise RuntimeError("LLM API key not found or empty")

    def _configure_postprocessing(self) -> bool:
        cfg = self.config.postprocessing
        slack_token = cfg.slack_bot_token
        if slack_token is None:
            slack_token = load_slack_bot_token()
        slack_token = (slack_token or "").strip()
        slack_channel = (cfg.slack_channel or "").strip()

        poster = ResultPoster(post_fn=post) if cfg.enable_default_poster else ResultPoster()
        slack_enabled = bool(slack_token and slack_channel)
        configure_postprocessing(
            default_poster=poster,
            cluster_name=cfg.cluster_name or "",
            dataflow_index=cfg.dataflow_index or "",
            slack_bot_token=slack_token,
            slack_channel=slack_channel,
        )
        if slack_enabled:
            logger.info(
                "Slack notifications enabled for channel: %s",
                slack_channel,
            )
        return slack_enabled

    @staticmethod
    def _normalize_engine_backend(engine_backend: str) -> str:
        normalized = engine_backend.strip().lower()
        if normalized not in {"lib", "mcp"}:
            raise ValueError(
                "analysis.engine_backend must be 'lib' or 'mcp', " f"got {engine_backend!r}"
            )
        return normalized
