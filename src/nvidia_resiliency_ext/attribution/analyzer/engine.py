# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attribution orchestration: :class:`Analyzer` (jobs, coalescing, cache, pipelines).

This layer sits above :mod:`nvidia_resiliency_ext.attribution.orchestration` (LogSage, SLURM
parsers, splitlog, optional FR via :class:`~nvidia_resiliency_ext.attribution.trace_analyzer.trace_analyzer.TraceAnalyzer`).
:class:`Analyzer` adds request coalescing; on cache miss,
:meth:`~nvidia_resiliency_ext.attribution.orchestration.log_analyzer.LogAnalyzer.run_attribution_for_path`
runs LogSage and optional FR (see :mod:`~nvidia_resiliency_ext.attribution.orchestration.analysis_pipeline`).

Standalone API (no HTTP):

    from nvidia_resiliency_ext.attribution import Analyzer

    analyzer = Analyzer(allowed_root="/logs", use_lib_log_analysis=False)
    await analyzer.submit("/logs/slurm-12345.out", user="alice", job_id="12345")
    result = await analyzer.analyze("/logs/slurm-12345.out")
    analyzer.shutdown()

Architecture:
- :class:`Analyzer`: :class:`~nvidia_resiliency_ext.attribution.coalescing.RequestCoalescer` (cache) and
  :class:`~nvidia_resiliency_ext.attribution.orchestration.log_analyzer.LogAnalyzer` (jobs, LogSage, optional FR pipeline)

**Event loop (splitlog / thread callbacks):** :meth:`~Analyzer.set_event_loop` must be called with the
process main asyncio loop **as soon as it is available** (e.g. FastAPI ``startup``). Splitlog polling
starts in ``Analyzer.__init__``; if a poll schedules analysis before the loop is set, fire-and-forget
posting **logs and skips** that schedule rather than raising from the poll thread.

Job modes (tracking): PENDING, SINGLE, SPLITLOG — see class docstring.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from nvidia_resiliency_ext.attribution.coalescing import (
    CacheResult,
    ComputeStats,
    InflightResult,
    RequestCoalescer,
    SubmittedResult,
)
from nvidia_resiliency_ext.attribution.orchestration.analysis_pipeline import (
    AnalysisPipelineMode,
    FrDumpPathNotFoundError,
)
from nvidia_resiliency_ext.attribution.orchestration.config import ErrorCode, LogSageExecutionConfig
from nvidia_resiliency_ext.attribution.orchestration.job import Job, JobMode
from nvidia_resiliency_ext.attribution.orchestration.llm_output import attribution_recommendation
from nvidia_resiliency_ext.attribution.orchestration.log_analyzer import LogAnalyzer
from nvidia_resiliency_ext.attribution.orchestration.types import (
    LogAnalysisCycleResult,
    LogAnalysisSplitlogResult,
    LogAnalyzerError,
    LogAnalyzerFilePreview,
    LogAnalyzerOutcome,
    LogAnalyzerSubmitResult,
)
from nvidia_resiliency_ext.attribution.trace_analyzer.trace_analyzer import TraceAnalyzer

from ..coalescing import LogAnalysisCoalesced, coalesced_from_cache

logger = logging.getLogger(__name__)


class Analyzer:
    """
    Entry point: request coalescing plus :class:`~nvidia_resiliency_ext.attribution.orchestration.log_analyzer.LogAnalyzer`.

    On :meth:`analyze`, the coalescer returns a cached :class:`~nvidia_resiliency_ext.attribution.coalescing.LogAnalysisCoalesced`
    when possible; on a miss, :meth:`_run_llm_analysis` delegates to
    :meth:`~nvidia_resiliency_ext.attribution.orchestration.log_analyzer.LogAnalyzer.run_attribution_for_path`.

    Call :meth:`set_event_loop` during app startup so splitlog background threads can schedule work on the
    main loop (see module docstring).

    Example:
        from nvidia_resiliency_ext.attribution import Analyzer

        analyzer = Analyzer(allowed_root="/logs", use_lib_log_analysis=False)

        submit_result = await analyzer.submit("/logs/slurm-12345.out", user="alice")
        result = await analyzer.analyze("/logs/slurm-12345.out")
        analyzer.shutdown()
    """

    def __init__(
        self,
        allowed_root: str,
        use_lib_log_analysis: bool = False,
        coalescer: RequestCoalescer | None = None,
        log_analyzer: LogAnalyzer | None = None,
        trace_analyzer: TraceAnalyzer | None = None,
        *,
        log_sage: LogSageExecutionConfig | None = None,
        analysis_pipeline_mode: AnalysisPipelineMode = AnalysisPipelineMode.LOG_AND_TRACE,
        compute_timeout: float | None = None,
        grace_period_seconds: float | None = None,
    ):
        """
        Args:
            allowed_root: Absolute path prefix for log files (path policy).
            use_lib_log_analysis: If True, run LogSage in-process; if False, use MCP. Ignored when
                ``log_sage`` is provided (use ``log_sage.use_lib_log_analysis`` instead).
            coalescer: Optional RequestCoalescer (for testing/DI). When omitted, one is built
                using ``compute_timeout`` and ``grace_period_seconds``.
            log_analyzer: Optional log-side facade. When omitted, one is built from ``allowed_root``,
                ``use_lib_log_analysis`` / ``log_sage``, the coalescer, and optional ``trace_analyzer``.
            trace_analyzer: Optional FR :class:`~nvidia_resiliency_ext.attribution.trace_analyzer.trace_analyzer.TraceAnalyzer`
                passed into the default :class:`~nvidia_resiliency_ext.attribution.orchestration.log_analyzer.LogAnalyzer`
                (ignored when ``log_analyzer`` is provided).
            log_sage: Optional :class:`~nvidia_resiliency_ext.attribution.orchestration.config.LogSageExecutionConfig`
                (LLM model, temperature, lib vs MCP). When omitted, defaults are used with
                ``use_lib_log_analysis`` only.
            analysis_pipeline_mode: Passed to the default :class:`~nvidia_resiliency_ext.attribution.orchestration.log_analyzer.LogAnalyzer`
                (ignored when ``log_analyzer`` is provided). Default is :attr:`AnalysisPipelineMode.LOG_AND_TRACE`;
                set :attr:`AnalysisPipelineMode.LOG_ONLY`, :attr:`AnalysisPipelineMode.TRACE_ONLY`, or
                :attr:`AnalysisPipelineMode.LOG_AND_TRACE_WITH_LLM` without pre-building ``LogAnalyzer``.
            compute_timeout: Per-compute timeout for the coalescer (seconds). Default: coalescer default.
            grace_period_seconds: Cache grace before stat validation (seconds). Default: coalescer default.

        After construction, call :meth:`set_event_loop` when the asyncio loop exists (e.g. HTTP server
        startup) so splitlog fire-and-forget analysis can be scheduled; see module docstring.
        """
        if not allowed_root:
            raise ValueError("allowed_root is required")
        if not os.path.isabs(allowed_root):
            raise ValueError("allowed_root must be an absolute path")
        self.allowed_root = allowed_root
        self._log_sage = log_sage or LogSageExecutionConfig(
            use_lib_log_analysis=use_lib_log_analysis
        )
        if coalescer is None:
            ct = (
                compute_timeout
                if compute_timeout is not None
                else RequestCoalescer.DEFAULT_COMPUTE_TIMEOUT_SECONDS
            )
            gp = (
                grace_period_seconds
                if grace_period_seconds is not None
                else RequestCoalescer.DEFAULT_GRACE_PERIOD_SECONDS
            )
            self._coalescer = RequestCoalescer(
                compute_timeout=ct,
                grace_period_seconds=gp,
            )
            self._compute_timeout = ct
        else:
            self._coalescer = coalescer
            self._compute_timeout = coalescer.compute_timeout
        self._log = log_analyzer or LogAnalyzer(
            allowed_root=allowed_root,
            log_sage=self._log_sage,
            track_submission=self._coalescer.track_submission,
            trace_analyzer=trace_analyzer,
            analysis_pipeline_mode=analysis_pipeline_mode,
        )
        self._main_loop: asyncio.AbstractEventLoop | None = None  # For thread-safe callbacks

        self._log.register_callbacks(self._fire_and_forget_analyze)

        _mode_log = (
            f", analysis_pipeline_mode={analysis_pipeline_mode.value}"
            if log_analyzer is None
            else ""
        )
        logger.info(
            f"Initialized Analyzer with allowed_root={allowed_root}, "
            f"compute_timeout={self._compute_timeout}s{_mode_log}"
        )

    @property
    def compute_timeout(self) -> float:
        """Per-compute timeout for LLM/analysis (seconds), from the coalescer."""
        return self._compute_timeout

    def shutdown(self) -> None:
        """Shutdown the analyzer and stop background threads."""
        self._log.shutdown_tracked()
        logger.info("Analyzer shutdown complete")

    async def shutdown_async(self) -> None:
        """Shutdown the analyzer including MCP client cleanup. Call from async context (e.g. lifespan)."""
        await self._log.shutdown_async()
        self.shutdown()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the main event loop for background thread callbacks.

        **Ordering:** Splitlog polling starts in ``__init__``; call this **as soon as the running loop
        exists** (e.g. FastAPI/Starlette ``lifespan`` or ``startup``) so
        :meth:`_fire_and_forget_analyze` can use :func:`asyncio.run_coroutine_threadsafe`. Until this is
        called, fire-and-forget schedules from the poll thread are skipped (logged) rather than raising.

        Example:

            @app.on_event("startup")
            async def startup():
                analyzer.set_event_loop(asyncio.get_running_loop())
        """
        self._main_loop = loop
        logger.info("Event loop set for background thread callbacks")

    async def connect_mcp(self) -> None:
        """Connect the MCP client. Call during async startup when using MCP backend.

        Must be called before the first analyze request when use_lib_log_analysis is False.
        No-op when using lib (in-process) log analysis.
        """
        await self._log.connect_mcp()

    async def check_mcp_health(self, timeout_seconds: float = 5.0) -> tuple[str, str]:
        """Check if MCP backend is reachable.

        Returns:
            (status, message) where status is "ok" | "disconnected" | "unused".
            "unused" when using lib backend (no MCP). "disconnected" when MCP unreachable.
        """
        return await self._log.check_mcp_health(timeout_seconds=timeout_seconds)

    async def reconnect_mcp(self) -> bool:
        """Reconnect the MCP client after failure. Returns True on success."""
        return await self._log.reconnect_mcp()

    def validate_path(
        self,
        user_path: str,
        *,
        require_regular_file: bool = True,
        reject_empty: bool = False,
    ) -> str | LogAnalyzerError:
        """Validate and normalize a path under the analyzer ``allowed_root``."""
        return self._log.validate_path(
            user_path,
            require_regular_file=require_regular_file,
            reject_empty=reject_empty,
        )

    async def submit(
        self,
        log_path: str,
        user: str = "unknown",
        job_id: Optional[str] = None,
    ) -> LogAnalyzerSubmitResult | LogAnalyzerError:
        """
        Submit a log file for analysis tracking.

        Creates a Job for tracking. If job_id is provided and LOGS_DIR is found,
        enables split logging mode.

        Args:
            log_path: Path to the log file (or slurm output for splitlog mode)
            user: Job owner (for dataflow records)
            job_id: Job ID (required for split logging mode)

        Returns:
            LogAnalyzerSubmitResult on success, LogAnalyzerError on failure
        """
        return await self._log.submit(log_path, user=user, job_id=job_id)

    async def analyze(
        self,
        log_path: str,
        file: Optional[str] = None,
        wl_restart: Optional[int] = None,
    ) -> LogAnalyzerOutcome:
        """
        Analyze a log file using LLM.

        For split logging mode jobs, use file= to select a specific log file
        and wl_restart= to select a specific workload restart within that file.

        Args:
            log_path: Path to the log file (or slurm output for splitlog mode)
            file: Filename for splitlog mode (None = all files)
            wl_restart: Workload restart index within file (None = all)

        Returns:
            LogAnalysisCycleResult or LogAnalysisSplitlogResult on success, LogAnalyzerError on failure
        """
        # Validate wl_restart parameter
        if wl_restart is not None and wl_restart < 0:
            return LogAnalyzerError(
                error_code=ErrorCode.INVALID_PATH,
                message=f"wl_restart must be >= 0, got {wl_restart}",
            )

        validated = self.validate_path(log_path, require_regular_file=True, reject_empty=False)
        if isinstance(validated, LogAnalyzerError):
            return validated

        job = self._log.get_job(validated)
        mode = job.mode if job else JobMode.SINGLE
        is_pending = mode == JobMode.PENDING
        logger.debug(
            f"analyze() lookup: path={validated}, job_found={job is not None}, "
            f"mode={mode}, job_id={job.job_id if job else 'N/A'}"
        )

        # Handle pending jobs (may promote to SPLITLOG); re-read mode under lock
        # so we don't demote a job that was just promoted by this or another thread.
        if is_pending and job:
            self._log.check_pending_jobs()
            job = self._log.get_job(validated)
            mode = job.mode if job else JobMode.SINGLE
            if mode == JobMode.PENDING:
                job.demote_to_single()
                mode = JobMode.SINGLE
                self._log.record_deferred_single_demotion()

        if mode == JobMode.SPLITLOG and job:
            return await self._analyze_splitlog_mode(validated, job, file, wl_restart)

        # Single file mode
        validated = self.validate_path(log_path, require_regular_file=True, reject_empty=True)
        if isinstance(validated, LogAnalyzerError):
            return validated

        try:
            user = job.user if job else "unknown"
            job_id = job.job_id if job else None
            if not job_id:
                logger.debug(
                    f"No job_id for path {validated}: job_found={job is not None}, "
                    f"job.job_id={getattr(job, 'job_id', 'N/A') if job else 'N/A'}"
                )
            coalesced_raw = await self._coalescer.get_or_compute(
                validated, lambda: self._run_llm_analysis(validated, user=user, job_id=job_id)
            )
            bundle = coalesced_from_cache(coalesced_raw)
            log_result = bundle.log_result
            fr_dump = bundle.fr_dump_path
            fr_analysis = bundle.fr_analysis
            llm_merged = bundle.llm_merged_summary
            if log_result is None:
                if not (fr_dump or fr_analysis):
                    return LogAnalyzerError(
                        error_code=ErrorCode.INTERNAL_ERROR,
                        message="cached entry has no log analysis and no FR data",
                    )
                if wl_restart is not None:
                    return LogAnalyzerError(
                        error_code=ErrorCode.INVALID_PATH,
                        message="wl_restart is not applicable for FR-only cache entries",
                    )
                return LogAnalysisCycleResult(
                    result={
                        "state": "no_log",
                        "result": [],
                        "module": "fr_only",
                    },
                    status="completed",
                    wl_restart=0,
                    wl_restart_count=None,
                    fr_dump_path=fr_dump,
                    fr_analysis=fr_analysis,
                    llm_merged_summary=llm_merged,
                    recommendation=attribution_recommendation(
                        {
                            "state": "no_log",
                            "result": [],
                            "module": "fr_only",
                        }
                    ),
                )
            # LLM returns multiple results per file (one per workload cycle); support wl_restart to select one
            results_list = (
                log_result.get("result") if isinstance(log_result.get("result"), list) else []
            )
            wl_restart_count = len(results_list) if results_list else None

            if wl_restart is not None:
                if wl_restart_count is None or wl_restart >= wl_restart_count:
                    return LogAnalyzerError(
                        error_code=ErrorCode.INVALID_PATH,
                        message=f"wl_restart={wl_restart} out of range (file has {wl_restart_count or 0} workload cycle(s))",
                    )
                # Return single-cycle result (copy so we don't mutate cached log_result)
                single_result = {**log_result, "result": [results_list[wl_restart]]}
                return LogAnalysisCycleResult(
                    result=single_result,
                    status="completed",
                    wl_restart=wl_restart,
                    wl_restart_count=wl_restart_count,
                    fr_dump_path=fr_dump,
                    fr_analysis=fr_analysis,
                    llm_merged_summary=llm_merged,
                    recommendation=attribution_recommendation(single_result),
                )
            # No wl_restart: return full result (all cycles) with count so client can iterate
            return LogAnalysisCycleResult(
                result=log_result,
                status="completed",
                wl_restart=0,
                wl_restart_count=wl_restart_count,
                fr_dump_path=fr_dump,
                fr_analysis=fr_analysis,
                llm_merged_summary=llm_merged,
                recommendation=attribution_recommendation(log_result),
            )
        except FrDumpPathNotFoundError as e:
            return LogAnalyzerError(
                error_code=ErrorCode.FR_DUMP_NOT_FOUND,
                message=str(e),
            )
        except Exception as e:
            # Note: asyncio.CancelledError inherits from BaseException (Python 3.8+),
            # so it's NOT caught here and will propagate correctly for graceful shutdown.
            logger.error(f"Analysis error: {e}")
            return LogAnalyzerError(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=str(e),
            )

    def read_file_preview(
        self, log_path: str, max_bytes: int = 4096
    ) -> LogAnalyzerFilePreview | LogAnalyzerError:
        """
        Read the first N bytes of a file for preview.

        Args:
            log_path: Path to the file
            max_bytes: Maximum bytes to read

        Returns:
            LogAnalyzerFilePreview on success, LogAnalyzerError on failure
        """
        return self._log.read_file_preview(log_path, max_bytes=max_bytes)

    # ─── Internal methods ───

    async def _run_llm_analysis(
        self, path: str, user: str = "unknown", job_id: Optional[str] = None
    ) -> LogAnalysisCoalesced:
        """On cache miss: delegate to :meth:`LogAnalyzer.run_attribution_for_path`."""
        return await self._log.run_attribution_for_path(path, user=user, job_id=job_id)

    async def _analyze_splitlog_mode(
        self,
        slurm_output_path: str,
        job: Job,
        file: Optional[str],
        wl_restart: Optional[int],
    ) -> LogAnalysisSplitlogResult | LogAnalyzerError:
        """Handle analysis for split logging mode jobs."""
        tracker = self._log.splitlog_tracker
        tracker.refresh_job_from_disk(job)
        if file is None:
            job.mark_terminated()
            # poll_now skips terminated jobs; flush here so the last file is analyzed
            tracker.flush_pending_splitlog_files(job)
        tracker.poll_now()
        file_info = tracker.get_file_info(job, file)

        if not file_info:
            if len(job.file_info) == 0:
                return LogAnalyzerError(
                    error_code=ErrorCode.NOT_FOUND,
                    message="no log files detected yet",
                )
            return LogAnalyzerError(
                error_code=ErrorCode.NOT_FOUND,
                message=f"file '{file}' not found" if file else "no log files available",
            )

        try:
            coalesced_raw = await self._coalescer.get_or_compute(
                file_info.log_file,
                lambda: self._run_llm_analysis(
                    file_info.log_file, user=job.user, job_id=job.job_id
                ),
            )
        except FrDumpPathNotFoundError as e:
            return LogAnalyzerError(
                error_code=ErrorCode.FR_DUMP_NOT_FOUND,
                message=str(e),
            )
        except Exception as e:
            return LogAnalyzerError(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"analysis failed: {e}",
            )

        bundle = coalesced_from_cache(coalesced_raw)
        result = bundle.log_result
        fr_dump = bundle.fr_dump_path
        fr_analysis = bundle.fr_analysis
        if result is None:
            if not (fr_dump or fr_analysis):
                return LogAnalyzerError(
                    error_code=ErrorCode.INTERNAL_ERROR,
                    message="cached entry has no log analysis and no FR data",
                )
            result = {
                "state": "no_log",
                "result": [],
                "module": "fr_only",
            }
        return LogAnalysisSplitlogResult(
            result=result,
            status="completed",
            mode=JobMode.SPLITLOG.value,
            sched_restarts=job.sched_restarts,
            log_file=file_info.log_file,
            wl_restart=wl_restart or 0,
            fr_dump_path=fr_dump,
            fr_analysis=fr_analysis,
            llm_merged_summary=bundle.llm_merged_summary,
            recommendation=attribution_recommendation(result),
        )

    def _sync_analyze_via_coalescer(
        self, log_path: str, user: str, job_id: Optional[str] = None
    ) -> Dict[str, Any] | None:
        """Synchronous wrapper for background thread analysis.

        Uses asyncio.run_coroutine_threadsafe() to schedule work on the main
        event loop rather than creating a new loop. This ensures proper task
        lifecycle management and avoids orphaned tasks.

        If :meth:`set_event_loop` was not called, logs an error and returns ``None`` (does not raise),
        so callers from a thread without a surrounding try/except do not fail uncaught.
        """
        if self._main_loop is None:
            logger.error(
                "Event loop not set — cannot run sync analyze via coalescer (call "
                "set_event_loop() during startup). path=%s",
                log_path,
            )
            return None

        future = asyncio.run_coroutine_threadsafe(
            self._coalescer.get_or_compute(
                log_path, lambda: self._run_llm_analysis(log_path, user=user, job_id=job_id)
            ),
            self._main_loop,
        )
        # Block until complete (same behavior as before)
        return future.result(timeout=self._compute_timeout)

    def _fire_and_forget_analyze(
        self, log_path: str, user: str = "unknown", job_id: Optional[str] = None
    ) -> None:
        """
        Schedule analysis without waiting for completion.

        Used by splitlog tracker to trigger analysis from any context (async or sync)
        without blocking. The result will be available in the coalescer cache.

        If the event loop has not been set yet, logs and returns (no exception) so the poll thread
        does not raise uncaught :class:`RuntimeError`.
        """
        if self._main_loop is None:
            logger.error(
                "Event loop not set — skipping fire-and-forget analyze for %s (call "
                "set_event_loop() during startup so splitlog can schedule work)",
                log_path,
            )
            return

        # Schedule the coroutine but don't wait for it
        asyncio.run_coroutine_threadsafe(
            self._coalescer.get_or_compute(
                log_path, lambda: self._run_llm_analysis(log_path, user=user, job_id=job_id)
            ),
            self._main_loop,
        )
        logger.debug(f"Scheduled fire-and-forget analysis for {log_path}")

    # ─── Stats and introspection ───

    async def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        stats = await self._coalescer.get_stats()
        stats[JobMode.SPLITLOG.value] = self._log.splitlog_tracker.get_stats()
        stats["detection"] = self._log.detection_stats(self._log.pending_job_count())
        stats["deferred"] = self._log.deferred_stats()
        stats["permission_errors"] = self._log.permission_error_stats()

        return stats

    async def get_compute_health_metrics(self) -> ComputeStats:
        """Get compute/LLM stats for health checks. Prefer over parsing get_stats()['compute']."""
        return await self._coalescer.get_compute_stats()

    async def get_cache(self) -> CacheResult:
        """Get current cache contents."""
        return await self._coalescer.get_cache()

    async def export_cache(self) -> List[Dict[str, Any]]:
        """Export cache entries for persistence. See RequestCoalescer.export_cache()."""
        return await self._coalescer.export_cache()

    async def import_cache(self, entries: List[Dict[str, Any]]) -> int:
        """Import cache entries from persistence. See RequestCoalescer.import_cache()."""
        return await self._coalescer.import_cache(entries)

    async def get_inflight(self) -> InflightResult:
        """Get currently in-flight requests."""
        return await self._coalescer.get_inflight()

    async def get_submitted(self) -> SubmittedResult:
        """Get submitted paths."""
        return await self._coalescer.get_submitted()

    def get_all_jobs(self) -> Dict[str, Any]:
        """Get all tracked jobs."""
        return self._log.get_all_jobs_payload()
