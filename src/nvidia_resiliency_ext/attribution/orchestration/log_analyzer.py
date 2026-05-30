# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Log-side subsystem: job tracking / splitlog (:class:`TrackedJobs`) and LogSage (:class:`LogSageRunner`).

:class:`LogAnalyzer` is the log-side facade (jobs, splitlog, LogSage). It can run **log-only** or
**log + flight-recorder** analysis via :class:`~nvidia_resiliency_ext.attribution.trace_analyzer.trace_analyzer.TraceAnalyzer`
and :func:`run_attribution_pipeline`. :class:`~nvidia_resiliency_ext.attribution.analyzer.engine.Analyzer`
composes it with request coalescing as the public entry point.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from nvidia_resiliency_ext.attribution.coalescing import LogAnalysisCoalesced

# svc is a connector layer; cross-package imports from attribution.* are intentional.
from nvidia_resiliency_ext.attribution.log_analyzer.nvrx_logsage import NVRxLogAnalyzer
from nvidia_resiliency_ext.attribution.mcp_integration import create_mcp_client
from nvidia_resiliency_ext.attribution.path_utils import path_is_under_allowed_root
from nvidia_resiliency_ext.attribution.postprocessing import post_analysis_items
from nvidia_resiliency_ext.attribution.trace_analyzer.fr_support import (
    FRAnalysisResult,
    analyze_fr_dump,
    fr_result_from_mcp_module_response,
)
from nvidia_resiliency_ext.attribution.trace_analyzer.trace_analyzer import TraceAnalyzer

from .analysis_pipeline import AnalysisPipelineMode, run_attribution_pipeline
from .config import MODULE_LOG_FR_ANALYZER, ErrorCode, LogSageExecutionConfig
from .job import Job
from .log_path_metadata import CYCLE_LOG_PATTERN
from .progressive import (
    MODULE_LOG_ANALYZER_PROGRESSIVE_START,
    PROGRESSIVE_STATUS_UNSUPPORTED,
    ProgressiveStartResult,
    progressive_start_result_from_mcp_response,
)
from .splitlog import SplitlogTracker
from .tracked_jobs import TrackedJobs
from .types import LogAnalyzerError, LogAnalyzerFilePreview, LogAnalyzerSubmitResult
from .utils import nvrx_run_result_to_log_dict, validate_log_path

logger = logging.getLogger(__name__)


class LogSageRunner:
    """Runs LogSage once per path (lib or MCP), returning the MCP-shaped ``log_result`` dict."""

    def __init__(self, config: LogSageExecutionConfig):
        self.config = config
        self._lib_log_analyzer: Any = None
        # Set on first failed NVRxLogAnalyzer() so we do not retry init every request.
        self._lib_log_analyzer_init_error: Optional[BaseException] = None
        self._log_analysis_lock = asyncio.Lock()
        self._mcp_client: Any = None
        if not config.use_lib_log_analysis:
            try:
                self._mcp_client = create_mcp_client(
                    mcp_server_log_level=config.mcp_server_log_level,
                )
            except Exception as e:
                raise RuntimeError(f"failed to create MCP client: {e}") from e

    async def shutdown_async(self) -> None:
        """Close MCP client when used."""
        if self._mcp_client is not None:
            try:
                await self._mcp_client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("MCP client shutdown error: %s", e)
            finally:
                self._mcp_client = None

    async def connect_mcp(self) -> None:
        """Connect the MCP client. No-op for lib backend."""
        if self._mcp_client is None:
            return
        await self._mcp_client.__aenter__()
        logger.info("MCP client connected")

    def _ensure_mcp_ready(self) -> None:
        if self._mcp_client is None:
            raise RuntimeError(
                "MCP client is not initialized. When using the MCP analysis backend "
                "(use_lib_log_analysis=False), client creation at init must succeed."
            )
        if not getattr(self._mcp_client, "session", None):
            raise RuntimeError(
                "MCP client is not connected. Call await connect_mcp() during async startup "
                "before the first analyze request (e.g. FastAPI lifespan)."
            )
        run_resilient = getattr(self._mcp_client, "run_module_resilient", None)
        if not callable(run_resilient):
            raise RuntimeError(
                "MCP client is invalid: expected a connected NVRxMCPClient with run_module_resilient()."
            )

    async def check_mcp_health(self, timeout_seconds: float = 5.0) -> tuple[str, str]:
        if self._mcp_client is None:
            return "unused", "MCP not used (analysis backend=lib)"
        if not getattr(self._mcp_client, "session", None):
            return "disconnected", "MCP client not connected"
        try:
            await asyncio.wait_for(
                self._mcp_client.get_status(),
                timeout=timeout_seconds,
            )
            return "ok", "MCP reachable"
        except asyncio.TimeoutError:
            return "disconnected", "MCP health check timed out"
        except Exception as e:
            return "disconnected", f"MCP unreachable: {e}"

    async def reconnect_mcp(self) -> bool:
        if self._mcp_client is None:
            return True
        return await self._mcp_client.reconnect()

    async def _get_lib_log_analyzer(self, run_kwargs: Dict[str, Any]) -> Any:
        """Return the cached in-process :class:`NVRxLogAnalyzer`, creating it on first success.

        **First call wins (LLM client is fixed):** ``NVRxLogAnalyzer`` is constructed at most once per
        :class:`LogSageRunner`. The **first** successful ``NVRxLogAnalyzer(dict(run_kwargs))`` binds
        the in-process LLM client and any other constructor-driven settings to that snapshot (model,
        temperature, top_p, max_tokens, etc.). Later calls pass different ``run_kwargs`` here, but
        ``__init__`` is **not** run again—the singleton keeps the original client. Per-request fields
        still go to :meth:`~NVRxLogAnalyzer.run` via ``run_kwargs``; whether that overrides the client
        is up to the library.

        To pick up new model or client settings after the first successful init, use a new
        :class:`LogSageRunner` (or a new process); this helper never re-instantiates after first success.
        """
        if self._lib_log_analyzer is not None:
            return self._lib_log_analyzer
        err = self._lib_log_analyzer_init_error
        if err is not None:
            raise err
        async with self._log_analysis_lock:
            if self._lib_log_analyzer is not None:
                return self._lib_log_analyzer
            if self._lib_log_analyzer_init_error is not None:
                raise self._lib_log_analyzer_init_error
            try:
                self._lib_log_analyzer = NVRxLogAnalyzer(dict(run_kwargs))
            except Exception as e:
                self._lib_log_analyzer_init_error = e
                logger.error("NVRxLogAnalyzer init failed; caching error (no retry): %s", e)
                raise
            return self._lib_log_analyzer

    def _streaming_kwargs(self) -> Dict[str, Any]:
        """Streaming-mode flags forwarded to NVRxLogAnalyzer init/run kwargs."""
        if not self.config.is_streaming_logs:
            return {}
        return {"is_streaming_logs": True}

    @property
    def is_streaming_logs(self) -> bool:
        """Whether streaming-mode LogSage is enabled for this runner."""
        return bool(self.config.is_streaming_logs)

    async def _fetch_log_result_lib(
        self, path: str, *, job_stage: Optional[str] = None
    ) -> Dict[str, Any]:
        is_per_cycle = bool(re.search(CYCLE_LOG_PATTERN, path))
        run_kwargs = {
            "log_path": path,
            "exclude_nvrx_logs": False,
            "is_per_cycle": is_per_cycle,
            **self._streaming_kwargs(),
            **self.config.llm_runtime_overrides(),
        }
        if job_stage is not None:
            run_kwargs["job_stage"] = job_stage
        analyzer = await self._get_lib_log_analyzer(run_kwargs)
        async with self._log_analysis_lock:
            result = await analyzer.run(run_kwargs)

        return nvrx_run_result_to_log_dict(result, path)

    async def _fetch_log_result_mcp(
        self, path: str, *, job_stage: Optional[str] = None
    ) -> Dict[str, Any]:
        self._ensure_mcp_ready()

        is_per_cycle = bool(re.search(CYCLE_LOG_PATTERN, path))
        run_kwargs = {
            "log_path": path,
            "exclude_nvrx_logs": False,
            "is_per_cycle": is_per_cycle,
            **self._streaming_kwargs(),
            **self.config.llm_runtime_overrides(),
        }
        if job_stage is not None:
            run_kwargs["job_stage"] = job_stage

        async with self._log_analysis_lock:
            log_result = await self._mcp_client.run_module_resilient(
                "log_analyzer", max_attempts=3, **run_kwargs
            )
        return log_result

    async def fetch_log_result(
        self, path: str, *, job_stage: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run LogSage for ``path``; return result items plus a derived recommendation.

        ``job_stage`` (``"start"`` / ``"end"``) is forwarded to NVRxLogAnalyzer's
        run kwargs and read by its streaming dispatcher; ignored when streaming
        mode is off.
        """
        if self.config.use_lib_log_analysis:
            return await self._fetch_log_result_lib(path, job_stage=job_stage)
        return await self._fetch_log_result_mcp(path, job_stage=job_stage)

    async def _start_progressive_analysis_lib(
        self,
        path: str,
        *,
        user: str = "unknown",
        job_id: str | None = None,
    ) -> ProgressiveStartResult:
        _ = (path, user, job_id)
        return ProgressiveStartResult(
            status=PROGRESSIVE_STATUS_UNSUPPORTED,
            message="LogSage in-process progressive start API is not configured",
        )

    async def _start_progressive_analysis_mcp(
        self,
        path: str,
        *,
        user: str = "unknown",
        job_id: str | None = None,
    ) -> ProgressiveStartResult:
        self._ensure_mcp_ready()
        is_per_cycle = bool(re.search(CYCLE_LOG_PATTERN, path))
        run_kwargs: Dict[str, Any] = {
            "log_path": path,
            "is_per_cycle": is_per_cycle,
            "user": user,
            **self.config.llm_runtime_overrides(),
        }
        if job_id:
            run_kwargs["job_id"] = job_id

        async with self._log_analysis_lock:
            response = await self._mcp_client.run_module_resilient(
                MODULE_LOG_ANALYZER_PROGRESSIVE_START,
                max_attempts=3,
                **run_kwargs,
            )
        return progressive_start_result_from_mcp_response(response)

    async def start_progressive_analysis(
        self,
        path: str,
        *,
        user: str = "unknown",
        job_id: str | None = None,
    ) -> ProgressiveStartResult:
        """Start progressive log analysis for ``path`` without producing final attribution."""
        if self.config.use_lib_log_analysis:
            return await self._start_progressive_analysis_lib(path, user=user, job_id=job_id)
        return await self._start_progressive_analysis_mcp(path, user=user, job_id=job_id)

    async def _fetch_fr_result_mcp(self, dump_path: str) -> Optional[FRAnalysisResult]:
        self._ensure_mcp_ready()
        async with self._log_analysis_lock:
            resp = await self._mcp_client.run_module_resilient(
                "fr_analyzer",
                max_attempts=3,
                fr_path=dump_path,
                pattern="_dump_*",
                verbose=False,
                health_check=False,
                llm_analyze=False,
                **self.config.llm_endpoint_overrides(),
            )
        return fr_result_from_mcp_module_response(resp)

    async def fetch_fr_result(self, dump_path: str) -> Optional[FRAnalysisResult]:
        """Run flight-recorder analysis (lib or MCP) for ``dump_path``."""
        if self.config.use_lib_log_analysis:
            return await analyze_fr_dump(dump_path)
        return await self._fetch_fr_result_mcp(dump_path)

    async def fetch_log_fr_analyzer_mcp(
        self, log_path: str, fr_dump_path: str, *, merge_llm: bool = False
    ) -> Tuple[Dict[str, Any], Optional[FRAnalysisResult], Optional[str]]:
        """Single MCP ``log_fr_analyzer`` call: collect log+FR, optionally merge with LLM."""
        self._ensure_mcp_ready()
        is_per_cycle = bool(re.search(CYCLE_LOG_PATTERN, log_path))
        kwargs: Dict[str, Any] = {
            "log_path": log_path,
            "fr_path": fr_dump_path,
            "exclude_nvrx_logs": False,
            "is_per_cycle": is_per_cycle,
            "pattern": "_dump_*",
            "verbose": False,
            "health_check": False,
            "llm_analyze": False,
            "merge_llm": merge_llm,
            "threshold": 0,
            **self.config.llm_runtime_overrides(),
        }

        async with self._log_analysis_lock:
            resp = await self._mcp_client.run_module_resilient(
                "log_fr_analyzer", max_attempts=3, **kwargs
            )

        if not isinstance(resp, dict):
            raise RuntimeError(f"log_fr_analyzer returned non-dict: {type(resp).__name__}")
        if resp.get("error"):
            raise RuntimeError(f"log_fr_analyzer MCP error: {resp.get('error')}")

        log_result = resp.get("result")
        if not isinstance(log_result, list):
            raise RuntimeError(
                "log_fr_analyzer result must be LogSage item list, "
                f"got {type(log_result).__name__}"
            )
        recommendation = resp.get("recommendation")
        if not isinstance(recommendation, dict):
            raise RuntimeError(
                "log_fr_analyzer recommendation must be dict, "
                f"got {type(recommendation).__name__}"
            )
        log_dict = {
            "module": resp.get("module", MODULE_LOG_FR_ANALYZER),
            "result": log_result,
            "recommendation": recommendation,
        }
        if "result_id" in resp:
            log_dict["result_id"] = resp["result_id"]
        if "resource_uri" in resp:
            log_dict["resource_uri"] = resp["resource_uri"]

        fr_part = resp.get("fr")
        fr_payload = fr_part.get("result") if isinstance(fr_part, dict) else None
        fr_analysis = fr_result_from_mcp_module_response({"result": fr_payload})
        summary = resp.get("llm_merged_summary") if merge_llm else None
        if summary is not None and not isinstance(summary, str):
            summary = str(summary)
        return log_dict, fr_analysis, summary


TrackSubmission = Callable[[str], Awaitable[None]]
FireAndForgetAnalyze = Callable[[str, str, Optional[str]], None]


class LogAnalyzer:
    """LogSage runner + tracked jobs / splitlog. Owns :meth:`submit` and coalescer submission hook."""

    def __init__(
        self,
        *,
        allowed_root: str,
        log_sage: LogSageExecutionConfig,
        track_submission: TrackSubmission,
        trace_analyzer: TraceAnalyzer | None = None,
        analysis_pipeline_mode: AnalysisPipelineMode = AnalysisPipelineMode.LOG_AND_TRACE,
        splitlog_tracker: SplitlogTracker | None = None,
        runner: LogSageRunner | None = None,
    ):
        self._allowed_root = allowed_root
        self._analysis_pipeline_mode = analysis_pipeline_mode
        if analysis_pipeline_mode == AnalysisPipelineMode.LOG_ONLY:
            self._trace_analyzer = None
        else:
            self._trace_analyzer = trace_analyzer or TraceAnalyzer(allowed_root=allowed_root)
        self._runner = runner or LogSageRunner(log_sage)
        self._tracked = TrackedJobs(
            track_submission=track_submission,
            splitlog_tracker=splitlog_tracker,
        )
        self._post_tasks: set[asyncio.Task[None]] = set()

    def validate_path(
        self,
        user_path: str,
        *,
        require_regular_file: bool = True,
        reject_empty: bool = False,
    ) -> str | LogAnalyzerError:
        """Validate and normalize a path under ``allowed_root``."""
        return validate_log_path(
            user_path,
            self._allowed_root,
            require_regular_file=require_regular_file,
            reject_empty=reject_empty,
        )

    def _discover_fr_dump_path(self, log_path: str) -> Optional[str]:
        """Resolve FR dump path via the trace analyzer; enforce ``allowed_root`` on the result."""
        if self._trace_analyzer is None:
            return None
        discovered = self._trace_analyzer.discover_fr_dump_path(log_path)
        if discovered is None:
            return None
        if not path_is_under_allowed_root(discovered, self._allowed_root):
            logger.warning(
                "Discovered FR dump path %r is outside allowed_root %r; skipping FR analysis",
                discovered,
                self._allowed_root,
            )
            return None
        return discovered

    def read_file_preview(
        self, log_path: str, max_bytes: int = 4096
    ) -> LogAnalyzerFilePreview | LogAnalyzerError:
        """Read the first N bytes of a file for preview."""
        validated = self.validate_path(log_path, require_regular_file=False, reject_empty=False)
        if isinstance(validated, LogAnalyzerError):
            return validated
        try:
            with open(validated, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(max_bytes)
            return LogAnalyzerFilePreview(content=content, path=validated)
        except Exception as e:
            return LogAnalyzerError(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"file read error: {e}",
            )

    @property
    def splitlog_tracker(self) -> SplitlogTracker:
        return self._tracked.splitlog_tracker

    def register_callbacks(self, fire_and_forget_analyze: FireAndForgetAnalyze) -> None:
        self._tracked.register_callbacks(fire_and_forget_analyze)

    def shutdown_tracked(self) -> None:
        """Stop splitlog polling (sync)."""
        self._tracked.shutdown()

    async def shutdown_async(self) -> None:
        await self._runner.shutdown_async()
        if self._post_tasks:
            tasks = tuple(self._post_tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def connect_mcp(self) -> None:
        await self._runner.connect_mcp()

    async def check_mcp_health(self, timeout_seconds: float = 5.0) -> tuple[str, str]:
        return await self._runner.check_mcp_health(timeout_seconds=timeout_seconds)

    async def reconnect_mcp(self) -> bool:
        return await self._runner.reconnect_mcp()

    async def fetch_log_result(
        self, path: str, *, job_stage: Optional[str] = None
    ) -> Dict[str, Any]:
        return await self._runner.fetch_log_result(path, job_stage=job_stage)

    @property
    def is_streaming_logs(self) -> bool:
        """True when the LogSage runner is configured for streaming start/end stages."""
        return self._runner.is_streaming_logs

    async def start_progressive_analysis(
        self,
        path: str,
        *,
        user: str = "unknown",
        job_id: str | None = None,
    ) -> ProgressiveStartResult:
        """Forward progressive start to the selected loganalysis tool adapter."""
        return await self._runner.start_progressive_analysis(path, user=user, job_id=job_id)

    def _post_analysis_results(
        self,
        result_items: List[Any],
        attribution_analysis_duration_seconds: float,
        attribution_analysis_completed_ms: int,
        path: str,
        user: str,
        job_id: Optional[str],
        fr_dump_path: Optional[str] = None,
        fr_analysis: Optional[FRAnalysisResult] = None,
        recommendation: Any = None,
    ) -> None:
        """Post each analysis result to dataflow/Slack. Shared by lib and MCP paths."""
        post_analysis_items(
            result_items,
            attribution_analysis_duration_seconds,
            path,
            user,
            job_id,
            fr_dump_path=fr_dump_path,
            fr_analysis=fr_analysis,
            attribution_analysis_completed_ms=attribution_analysis_completed_ms,
            recommendation=recommendation,
        )

    async def _run_post_analysis_results(
        self,
        result_items: List[Any],
        attribution_analysis_duration_seconds: float,
        attribution_analysis_completed_ms: int,
        path: str,
        user: str,
        job_id: Optional[str],
        fr_dump_path: Optional[str] = None,
        fr_analysis: Optional[FRAnalysisResult] = None,
        recommendation: Any = None,
    ) -> None:
        try:
            await asyncio.to_thread(
                self._post_analysis_results,
                result_items,
                attribution_analysis_duration_seconds,
                attribution_analysis_completed_ms,
                path,
                user,
                job_id,
                fr_dump_path=fr_dump_path,
                fr_analysis=fr_analysis,
                recommendation=recommendation,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Observability postprocessing failed for %s", path)

    def _schedule_post_analysis_results(
        self,
        result_items: List[Any],
        attribution_analysis_duration_seconds: float,
        attribution_analysis_completed_ms: int,
        path: str,
        user: str,
        job_id: Optional[str],
        fr_dump_path: Optional[str] = None,
        fr_analysis: Optional[FRAnalysisResult] = None,
        recommendation: Any = None,
    ) -> None:
        task = asyncio.create_task(
            self._run_post_analysis_results(
                list(result_items),
                attribution_analysis_duration_seconds,
                attribution_analysis_completed_ms,
                path,
                user,
                job_id,
                fr_dump_path=fr_dump_path,
                fr_analysis=fr_analysis,
                recommendation=recommendation,
            ),
            name=f"nvrx-observability-post:{os.path.basename(path)}",
        )
        self._post_tasks.add(task)
        task.add_done_callback(self._post_tasks.discard)

    async def run_attribution_for_path(
        self,
        path: str,
        user: str = "unknown",
        job_id: Optional[str] = None,
        *,
        job_stage: Optional[str] = None,
    ) -> LogAnalysisCoalesced:
        """Run LogSage and optional FR pipeline for ``path``; return coalescer payload."""
        if not os.access(path, os.R_OK):
            self.record_file_permission_error()
            raise PermissionError(f"Log file not readable: {path}")

        mode = self._analysis_pipeline_mode
        cfg = self._runner.config
        pipeline_kw: Dict[str, Any] = {}
        if self._trace_analyzer is not None:
            pipeline_kw["discover_fr_dump_path"] = self._discover_fr_dump_path
            pipeline_kw["run_fr_analysis"] = (
                self._trace_analyzer.analyze_fr_dump
                if cfg.use_lib_log_analysis
                else self._runner.fetch_fr_result
            )
            if not cfg.use_lib_log_analysis:

                async def _mcp_combined(
                    lp: str, fd: str
                ) -> tuple[Dict[str, Any], Optional[FRAnalysisResult], Optional[str]]:
                    return await self._runner.fetch_log_fr_analyzer_mcp(
                        lp,
                        fd,
                        merge_llm=(mode == AnalysisPipelineMode.LOG_AND_TRACE_WITH_LLM),
                    )

                pipeline_kw["run_log_fr_analyzer_mcp"] = _mcp_combined

        pipeline_kw.update(cfg.pipeline_llm_overrides())

        combined = await run_attribution_pipeline(
            path,
            mode=mode,
            run_logsage=(
                (lambda: self.fetch_log_result(path, job_stage=job_stage))
                if mode != AnalysisPipelineMode.TRACE_ONLY
                else None
            ),
            **pipeline_kw,
        )
        log_result = combined.log_result
        fr_dump_path = combined.fr_dump_path
        fr_analysis = combined.fr_analysis
        attribution_analysis_duration_seconds = combined.processing_time
        attribution_analysis_completed_ms = combined.analysis_completed_at_ms
        llm_merged_summary = combined.llm_merged_summary

        if mode == AnalysisPipelineMode.TRACE_ONLY:
            if fr_analysis is None:
                raise RuntimeError("TRACE_ONLY mode but FR analysis returned None")
        elif log_result is None:
            logger.error("Log analyzer returned None for path=%s", path)
            raise RuntimeError("Log analyzer returned None")
        if log_result is not None:
            if not isinstance(log_result, dict):
                logger.error(
                    "Log analyzer returned non-dict for path=%s: type=%s",
                    path,
                    type(log_result).__name__,
                )
                raise RuntimeError(
                    f"Log analyzer returned {type(log_result).__name__}, expected dict"
                )

            raw_result = log_result.get("result")
            if raw_result is not None and not isinstance(raw_result, list):
                logger.error(
                    "Log analyzer result has unexpected type for path=%s: "
                    "expected list or None, got %s",
                    path,
                    type(raw_result).__name__,
                )
                raise RuntimeError(
                    f"Log analyzer result must be list or None, got {type(raw_result).__name__}"
                )
            result_items = raw_result or []
        else:
            result_items = []
        recommendation = log_result.get("recommendation") if isinstance(log_result, dict) else None
        self._schedule_post_analysis_results(
            result_items,
            attribution_analysis_duration_seconds,
            attribution_analysis_completed_ms,
            path,
            user,
            job_id,
            fr_dump_path=fr_dump_path,
            fr_analysis=fr_analysis,
            recommendation=recommendation,
        )
        return LogAnalysisCoalesced(
            log_result=log_result,
            fr_dump_path=fr_dump_path,
            fr_analysis=fr_analysis,
            llm_merged_summary=llm_merged_summary,
        )

    def get_job(self, path: str) -> Optional[Job]:
        return self._tracked.get_job(path)

    def check_pending_jobs(self) -> None:
        self._tracked.check_pending_jobs()

    def record_deferred_single_demotion(self) -> None:
        self._tracked.record_deferred_single_demotion()

    def record_file_permission_error(self) -> None:
        self._tracked.record_file_permission_error()

    def detection_stats(self, pending_count: int) -> Dict[str, Any]:
        return self._tracked.detection_stats(pending_count)

    def deferred_stats(self) -> Dict[str, int]:
        return self._tracked.deferred_stats()

    def permission_error_stats(self) -> Dict[str, int]:
        return self._tracked.permission_error_stats()

    def pending_job_count(self) -> int:
        return self._tracked.pending_job_count()

    def get_all_jobs_payload(self) -> Dict[str, Any]:
        return self._tracked.get_all_jobs_payload()

    async def submit(
        self,
        log_path: str,
        user: str = "unknown",
        job_id: Optional[str] = None,
    ) -> LogAnalyzerSubmitResult | LogAnalyzerError:
        """Submit a log path for tracking (splitlog / pending / single)."""
        if not log_path:
            return LogAnalyzerError(
                error_code=ErrorCode.INVALID_PATH,
                message="log_path is required",
            )

        validated = validate_log_path(
            log_path,
            self._allowed_root,
            require_regular_file=True,
            reject_empty=False,
        )
        if isinstance(validated, LogAnalyzerError):
            return validated

        existing_job = self._tracked.get_job(validated)
        if existing_job:
            return self._tracked.handle_existing_job(existing_job, validated)

        return await self._tracked.create_new_job(validated, user, job_id)
