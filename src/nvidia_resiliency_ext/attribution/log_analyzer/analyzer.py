#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Core log analyzer - usable as standalone Python API or via HTTP service.

This module provides the main LogAnalyzer class for analyzing log files
using LLM-based failure attribution. It can be used directly without any
HTTP dependencies:

    from nvidia_resiliency_ext.attribution import LogAnalyzer, AnalyzerConfig

    config = AnalyzerConfig(allowed_root="/logs")
    analyzer = LogAnalyzer(config)

    # Submit a job for tracking (enables splitlog mode if LOGS_DIR found)
    await analyzer.submit("/logs/slurm-12345.out", user="alice", job_id="12345")

    # Analyze the log
    result = await analyzer.analyze("/logs/slurm-12345.out")

    # Clean up background threads
    analyzer.shutdown()

Architecture:
- LogAnalyzer: Main API class, manages jobs and coordinates analysis
- RequestCoalescer: Deduplicates concurrent requests, caches results
- SplitlogTracker: Background polling for jobs with separate log directories

Job modes:
- PENDING: Job submitted but LOGS_DIR not found yet (deferred classification)
- SINGLE: Single-file mode (analyze slurm output directly)
- SPLITLOG: Split logging mode (analyze per-restart log files in LOGS_DIR)
"""

import argparse
import asyncio
import errno
import logging
import os
import re
import stat
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from nvidia_resiliency_ext.attribution.base import AttributionState
from nvidia_resiliency_ext.attribution.mcp_integration import create_mcp_client
from nvidia_resiliency_ext.attribution.postprocessing import post_results

from .coalescer import CacheResult, ComputeStats, InflightResult, RequestCoalescer, SubmittedResult
from .config import (
    DEFAULT_COMPUTE_TIMEOUT_SECONDS,
    RESP_FILES_ANALYZED,
    RESP_LOGS_DIR,
    RESP_MODE,
    RESP_SCHED_RESTARTS,
    STATS_JOB_ID,
    STATS_LOG_FILES,
    STATS_LOG_PATH,
    STATS_TERMINATED,
    STATS_USER,
    TTL_PENDING_SECONDS,
    ErrorCode,
)
from .job import Job, JobMode
from .nvrx_logsage import NVRxLogAnalyzer
from .slurm_parser import read_and_parse_slurm_output
from .splitlog import SplitlogTracker
from .utils import CYCLE_LOG_PATTERN, JobMetadata, extract_job_metadata, parse_llm_response

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerConfig:
    """Configuration for LogAnalyzer.

    This is a simple dataclass for library usage. HTTP services can
    convert their Settings to this format.

    Attributes:
        allowed_root: Base directory for path validation (required)
        compute_timeout: Timeout for LLM analysis in seconds
        grace_period_seconds: Grace period before stat() validation on cache hits
        llm_model: LLM model identifier for MCP client
        llm_temperature: Temperature for LLM (0.0 = deterministic)
        llm_top_p: Top-p for LLM nucleus sampling
        llm_max_tokens: Max tokens for LLM response
        use_lib_log_analysis: If True, run LogSage in-process (no MCP subprocess).
            Enables reusable cache and lower latency. Use in service or when subprocess
            overhead is undesirable.
    """

    allowed_root: str
    compute_timeout: float = DEFAULT_COMPUTE_TIMEOUT_SECONDS
    grace_period_seconds: float = 600.0  # 10 min default
    llm_model: str = "nvdev/nvidia/llama-3.3-nemotron-super-49b-v1"
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_max_tokens: int = 8192
    use_lib_log_analysis: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.allowed_root:
            raise ValueError("allowed_root is required")
        if not os.path.isabs(self.allowed_root):
            raise ValueError("allowed_root must be an absolute path")


@dataclass
class AnalyzerError:
    """Error result from analyzer operations."""

    error_code: ErrorCode
    message: str


@dataclass
class AnalysisResult:
    """Result from log analysis (single-file mode). One result per workload cycle when wl_restart is set."""

    result: Dict[str, Any]
    status: str = "completed"
    wl_restart: int = 0  # Workload restart index (which cycle this result is for)
    wl_restart_count: Optional[int] = (
        None  # Total workload cycles in the file (None if not applicable)
    )


@dataclass
class SubmitResult:
    """Result from log submission."""

    submitted: bool
    normalized_path: str
    mode: str = JobMode.SINGLE.value  # JobMode.PENDING, SINGLE, or SPLITLOG
    logs_dir: Optional[str] = None
    sched_restarts: int = 0
    files_analyzed: int = 0


@dataclass
class SplitlogAnalysisResult:
    """Result from split logging mode analysis."""

    result: Dict[str, Any]
    status: str = "completed"
    mode: str = JobMode.SPLITLOG.value
    sched_restarts: int = 0
    log_file: str = ""
    wl_restart: int = 0


@dataclass
class FilePreviewResult:
    """Result from file preview."""

    content: str
    path: str


# Type alias for result types
AnalyzerResult = AnalysisResult | SplitlogAnalysisResult | AnalyzerError


class LogAnalyzer:
    """
    Core log analyzer for LLM-based log analysis.

    This class provides the main API for analyzing log files. It can be
    used directly as a Python library or wrapped by an HTTP service.

    Example:
        from nvidia_resiliency_ext.attribution import LogAnalyzer, AnalyzerConfig

        config = AnalyzerConfig(allowed_root="/logs")
        analyzer = LogAnalyzer(config)

        # Submit a log for tracking
        submit_result = await analyzer.submit("/logs/slurm-12345.out", user="alice")

        # Analyze the log
        result = await analyzer.analyze("/logs/slurm-12345.out")

        # Clean up
        analyzer.shutdown()
    """

    def __init__(
        self,
        config: AnalyzerConfig,
        coalescer: RequestCoalescer | None = None,
        splitlog_tracker: SplitlogTracker | None = None,
    ):
        """
        Initialize the log analyzer.

        Args:
            config: Analyzer configuration
            coalescer: Optional RequestCoalescer (for testing/DI)
            splitlog_tracker: Optional SplitlogTracker (for testing/DI)
        """
        self.config = config
        self._coalescer = coalescer or RequestCoalescer(
            compute_timeout=config.compute_timeout,
            grace_period_seconds=config.grace_period_seconds,
        )
        self._splitlog_tracker = splitlog_tracker or SplitlogTracker()
        self._jobs: Dict[str, Job] = {}
        self._jobs_lock = threading.Lock()  # Protects _jobs from concurrent access
        self._main_loop: asyncio.AbstractEventLoop | None = None  # For thread-safe callbacks

        # Counters
        self._total_splitlog: int = 0
        self._total_single: int = 0
        self._deferred_splitlog: int = 0
        self._deferred_single: int = 0
        self._total_permission_errors: int = 0
        self._logs_dir_permission_errors: int = 0
        self._file_permission_errors: int = 0
        self._pending_expired: int = 0

        # Lib log analysis backend: one NVRxLogAnalyzer per process (reusable cache).
        # LogSage (both lib and MCP) is not reentrant: shared args/cache. _log_analysis_lock
        # serializes all log analysis execution (lib run() and MCP run_module()).
        self._lib_log_analyzer: Any = None
        self._log_analysis_lock = asyncio.Lock()

        # MCP backend: long-lived singleton client, created at init when use_lib_log_analysis
        # is False. Must call connect_mcp() during async startup before first use.
        if config.use_lib_log_analysis:
            self._mcp_client: Any = None
        else:
            try:
                self._mcp_client = create_mcp_client()
            except Exception as e:
                raise RuntimeError(f"failed to create MCP client: {e}") from e

        # Set up splitlog tracker callbacks
        # IMPORTANT: Use fire-and-forget callback to avoid blocking the async event loop.
        # The splitlog tracker runs in a background thread; calling blocking sync methods
        # from there would deadlock. fire-and-forget schedules work on the main loop.
        self._splitlog_tracker.set_analyze_callback(self._fire_and_forget_analyze)
        self._splitlog_tracker.set_pending_check_callback(self._check_pending_jobs)
        self._splitlog_tracker.set_get_splitlog_jobs_callback(self._get_splitlog_jobs)
        self._splitlog_tracker.set_cleanup_job_callback(self._cleanup_job)

        # Start background polling
        self._splitlog_tracker.start_polling()

        logger.info(
            f"Initialized LogAnalyzer with allowed_root={config.allowed_root}, "
            f"compute_timeout={config.compute_timeout}s"
        )

    def shutdown(self) -> None:
        """Shutdown the analyzer and stop background threads."""
        self._splitlog_tracker.stop_polling()
        # Note: MCP client cleanup requires async context; use shutdown_async() from app lifespan.
        logger.info("LogAnalyzer shutdown complete")

    async def shutdown_async(self) -> None:
        """Shutdown the analyzer including MCP client cleanup. Call from async context (e.g. lifespan)."""
        if self._mcp_client is not None:
            try:
                await self._mcp_client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("MCP client shutdown error: %s", e)
            finally:
                self._mcp_client = None
        self.shutdown()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the main event loop for background thread callbacks.

        Must be called from the main thread before background polling starts
        making analyze calls. Typically called during app startup:

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
        if self._mcp_client is None:
            return
        await self._mcp_client.__aenter__()
        logger.info("MCP client connected")

    async def check_mcp_health(self, timeout_seconds: float = 5.0) -> tuple[str, str]:
        """Check if MCP backend is reachable.

        Returns:
            (status, message) where status is "ok" | "disconnected" | "unused".
            "unused" when using lib backend (no MCP). "disconnected" when MCP unreachable.
        """
        if self._mcp_client is None:
            return "unused", "MCP not used (lib backend)"
        if not getattr(self._mcp_client, "session", None):
            return "disconnected", "MCP client not connected"
        try:
            # Invoke our MCP server's custom "status" tool as a lightweight health check
            # (returns server metadata; does not run LogSage).
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
        """Reconnect the MCP client after failure. Returns True on success.

        When called from _fetch_log_result_mcp (on connection error), the caller
        already holds _log_analysis_lock, so reconnects from that path are serialized.
        """
        if self._mcp_client is None:
            return True
        old = self._mcp_client
        self._mcp_client = None
        try:
            await old.__aexit__(None, None, None)
        except Exception as e:
            logger.debug("MCP cleanup during reconnect: %s", e)
        try:
            self._mcp_client = create_mcp_client()
            await self._mcp_client.__aenter__()
            logger.info("MCP client reconnected")
            return True
        except Exception as e:
            logger.error("MCP reconnect failed: %s", e)
            return False

    def validate_path(
        self,
        user_path: str,
        *,
        require_regular_file: bool = True,
        reject_empty: bool = False,
    ) -> str | AnalyzerError:
        """
        Validate and normalize a path.

        Symlinks are allowed if the resolved target is within allowed_root.

        Args:
            user_path: User-provided path
            require_regular_file: If True, path must be a regular file
            reject_empty: If True, reject empty files

        Returns:
            Normalized path string on success, AnalyzerError on failure
        """
        if not os.path.isabs(user_path):
            return AnalyzerError(
                error_code=ErrorCode.INVALID_PATH,
                message="path must be absolute",
            )

        try:
            real = os.path.realpath(user_path)
        except ValueError:
            return AnalyzerError(
                error_code=ErrorCode.INVALID_PATH,
                message="invalid path characters",
            )

        allowed_root = os.path.realpath(self.config.allowed_root)

        try:
            common = os.path.commonpath([real, allowed_root])
        except ValueError:
            return AnalyzerError(
                error_code=ErrorCode.OUTSIDE_ROOT,
                message="access outside allowed root is not permitted",
            )

        if common != allowed_root:
            return AnalyzerError(
                error_code=ErrorCode.OUTSIDE_ROOT,
                message="access outside allowed root is not permitted",
            )

        try:
            st = os.stat(real)
        except FileNotFoundError:
            return AnalyzerError(
                error_code=ErrorCode.NOT_FOUND,
                message="path not found",
            )
        except PermissionError:
            return AnalyzerError(
                error_code=ErrorCode.NOT_READABLE,
                message="permission denied",
            )
        except OSError as e:
            return AnalyzerError(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"filesystem error: {e}",
            )

        if require_regular_file and not stat.S_ISREG(st.st_mode):
            return AnalyzerError(
                error_code=ErrorCode.NOT_REGULAR,
                message="path must be a regular file",
            )

        if not os.access(real, os.R_OK):
            return AnalyzerError(
                error_code=ErrorCode.NOT_READABLE,
                message="path is not readable",
            )

        if reject_empty and require_regular_file and st.st_size == 0:
            return AnalyzerError(
                error_code=ErrorCode.EMPTY_FILE,
                message="file is empty",
            )

        return real

    async def submit(
        self,
        log_path: str,
        user: str = "unknown",
        job_id: Optional[str] = None,
    ) -> SubmitResult | AnalyzerError:
        """
        Submit a log file for analysis tracking.

        Creates a Job for tracking. If job_id is provided and LOGS_DIR is found,
        enables split logging mode.

        Args:
            log_path: Path to the log file (or slurm output for splitlog mode)
            user: Job owner (for dataflow records)
            job_id: Job ID (required for split logging mode)

        Returns:
            SubmitResult on success, AnalyzerError on failure
        """
        if not log_path:
            return AnalyzerError(
                error_code=ErrorCode.INVALID_PATH,
                message="log_path is required",
            )

        validated = self.validate_path(log_path, require_regular_file=True, reject_empty=False)
        if isinstance(validated, AnalyzerError):
            return validated

        # Check if already tracked
        with self._jobs_lock:
            existing_job = self._jobs.get(validated)
        if existing_job:
            return self._handle_existing_job(existing_job, validated)

        # First submission - create job and detect mode
        return await self._create_new_job(validated, user, job_id)

    async def analyze(
        self,
        log_path: str,
        file: Optional[str] = None,
        wl_restart: Optional[int] = None,
    ) -> AnalyzerResult:
        """
        Analyze a log file using LLM.

        For split logging mode jobs, use file= to select a specific log file
        and wl_restart= to select a specific workload restart within that file.

        Args:
            log_path: Path to the log file (or slurm output for splitlog mode)
            file: Filename for splitlog mode (None = all files)
            wl_restart: Workload restart index within file (None = all)

        Returns:
            AnalysisResult or SplitlogAnalysisResult on success, AnalyzerError on failure
        """
        # Validate wl_restart parameter
        if wl_restart is not None and wl_restart < 0:
            return AnalyzerError(
                error_code=ErrorCode.INVALID_PATH,
                message=f"wl_restart must be >= 0, got {wl_restart}",
            )

        validated = self.validate_path(log_path, require_regular_file=True, reject_empty=False)
        if isinstance(validated, AnalyzerError):
            return validated

        with self._jobs_lock:
            job = self._jobs.get(validated)
            mode = job.mode if job else JobMode.SINGLE
            is_pending = mode == JobMode.PENDING
        logger.debug(
            f"analyze() lookup: path={validated}, job_found={job is not None}, "
            f"mode={mode}, job_id={job.job_id if job else 'N/A'}"
        )

        # Handle pending jobs (may promote to SPLITLOG); re-read mode under lock
        # so we don't demote a job that was just promoted by this or another thread.
        if is_pending and job:
            self._check_pending_jobs()
            with self._jobs_lock:
                mode = job.mode if job else JobMode.SINGLE
            if mode == JobMode.PENDING:
                job.demote_to_single()
                mode = JobMode.SINGLE
                self._total_single += 1
                self._deferred_single += 1

        if mode == JobMode.SPLITLOG and job:
            return await self._analyze_splitlog_mode(validated, job, file, wl_restart)

        # Single file mode
        validated = self.validate_path(log_path, require_regular_file=True, reject_empty=True)
        if isinstance(validated, AnalyzerError):
            return validated

        try:
            user = job.user if job else "unknown"
            job_id = job.job_id if job else ""
            if not job_id:
                logger.debug(
                    f"No job_id for path {validated}: job_found={job is not None}, "
                    f"job.job_id={getattr(job, 'job_id', 'N/A') if job else 'N/A'}"
                )
            log_result = await self._coalescer.get_or_compute(
                validated, lambda: self._run_llm_analysis(validated, user=user, job_id=job_id)
            )
            # LLM returns multiple results per file (one per workload cycle); support wl_restart to select one
            results_list = (
                log_result.get("result") if isinstance(log_result.get("result"), list) else []
            )
            wl_restart_count = len(results_list) if results_list else None

            if wl_restart is not None:
                if wl_restart_count is None or wl_restart >= wl_restart_count:
                    return AnalyzerError(
                        error_code=ErrorCode.INVALID_PATH,
                        message=f"wl_restart={wl_restart} out of range (file has {wl_restart_count or 0} workload cycle(s))",
                    )
                # Return single-cycle result (copy so we don't mutate cached log_result)
                single_result = {**log_result, "result": [results_list[wl_restart]]}
                return AnalysisResult(
                    result=single_result,
                    status="completed",
                    wl_restart=wl_restart,
                    wl_restart_count=wl_restart_count,
                )
            # No wl_restart: return full result (all cycles) with count so client can iterate
            return AnalysisResult(
                result=log_result,
                status="completed",
                wl_restart=0,
                wl_restart_count=wl_restart_count,
            )
        except Exception as e:
            # Note: asyncio.CancelledError inherits from BaseException (Python 3.8+),
            # so it's NOT caught here and will propagate correctly for graceful shutdown.
            logger.error(f"Analysis error: {e}")
            return AnalyzerError(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=str(e),
            )

    def read_file_preview(
        self, log_path: str, max_bytes: int = 4096
    ) -> FilePreviewResult | AnalyzerError:
        """
        Read the first N bytes of a file for preview.

        Args:
            log_path: Path to the file
            max_bytes: Maximum bytes to read

        Returns:
            FilePreviewResult on success, AnalyzerError on failure
        """
        validated = self.validate_path(log_path, require_regular_file=False, reject_empty=False)
        if isinstance(validated, AnalyzerError):
            return validated

        try:
            with open(validated, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(max_bytes)
            return FilePreviewResult(content=content, path=validated)
        except Exception as e:
            return AnalyzerError(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"file read error: {e}",
            )

    # ─── Internal methods ───

    def _handle_existing_job(self, job: Job, validated: str) -> SubmitResult:
        """Handle re-submission of an existing job."""
        if job.is_splitlog():
            self._splitlog_tracker.poll_now()
            return SubmitResult(
                submitted=True,
                normalized_path=validated,
                mode=JobMode.SPLITLOG.value,
                logs_dir=job.logs_dir,
                sched_restarts=job.sched_restarts,
                files_analyzed=job.files_complete(),
            )
        elif job.is_pending():
            self._check_pending_jobs()
            if job.is_splitlog():
                return SubmitResult(
                    submitted=True,
                    normalized_path=validated,
                    mode=JobMode.SPLITLOG.value,
                    logs_dir=job.logs_dir,
                    sched_restarts=job.sched_restarts,
                    files_analyzed=job.files_complete(),
                )
            return SubmitResult(
                submitted=True,
                normalized_path=validated,
                mode=JobMode.PENDING.value,
            )
        else:
            return SubmitResult(
                submitted=True,
                normalized_path=validated,
                mode=JobMode.SINGLE.value,
            )

    async def _create_new_job(
        self, validated: str, user: str, job_id: Optional[str]
    ) -> SubmitResult | AnalyzerError:
        """Create a new job for tracking."""
        if job_id:
            logger.debug(f"_create_new_job: job_id={job_id}, checking for splitlog mode")
            info = read_and_parse_slurm_output(validated)
            logger.debug(
                f"_create_new_job: slurm parse result - "
                f"logs_dir={info.logs_dir if info else None}, "
                f"cycle_count={info.cycle_count if info else None}"
            )
            if info and info.logs_dir:
                if os.path.isdir(info.logs_dir):
                    if os.access(info.logs_dir, os.R_OK):
                        self._total_splitlog += 1
                        job = Job(
                            path=validated,
                            user=user,
                            mode=JobMode.SPLITLOG,
                            job_id=job_id,
                            logs_dir=info.logs_dir,
                        )
                        with self._jobs_lock:
                            self._jobs[validated] = job
                        logger.debug(
                            f"Created SPLITLOG job: path={validated}, job_id={job_id}, "
                            f"logs_dir={info.logs_dir}"
                        )
                        self._splitlog_tracker.initialize_job(job)
                        return SubmitResult(
                            submitted=True,
                            normalized_path=validated,
                            mode=JobMode.SPLITLOG.value,
                            logs_dir=info.logs_dir,
                            sched_restarts=job.sched_restarts,
                            files_analyzed=job.files_complete(),
                        )
                    else:
                        logger.debug(f"_create_new_job: logs_dir not readable: {info.logs_dir}")
                        self._total_permission_errors += 1
                        self._logs_dir_permission_errors += 1
                        return AnalyzerError(
                            error_code=ErrorCode.LOGS_DIR_NOT_READABLE,
                            message=f"LOGS_DIR not readable: {info.logs_dir}",
                        )
                else:
                    logger.debug(f"_create_new_job: logs_dir not a directory: {info.logs_dir}")
            else:
                logger.debug(
                    "_create_new_job: no logs_dir found in slurm output, deferring to PENDING"
                )

            # Defer to pending mode
            job = Job(
                path=validated,
                user=user,
                mode=JobMode.PENDING,
                job_id=job_id,
            )
            with self._jobs_lock:
                self._jobs[validated] = job
            logger.debug(f"Created PENDING job: path={validated}, job_id={job_id}")
            await self._coalescer.track_submission(validated)
            return SubmitResult(
                submitted=True,
                normalized_path=validated,
                mode=JobMode.PENDING.value,
            )

        # No job_id - single file mode
        self._total_single += 1
        job = Job(path=validated, user=user, mode=JobMode.SINGLE)
        with self._jobs_lock:
            self._jobs[validated] = job
        logger.debug(f"Created SINGLE job (no job_id provided): path={validated}")
        await self._coalescer.track_submission(validated)
        return SubmitResult(
            submitted=True,
            normalized_path=validated,
            mode=JobMode.SINGLE.value,
        )

    async def _get_lib_log_analyzer(self, run_kwargs: Dict[str, Any]) -> Any:
        """Get or create the single shared NVRxLogAnalyzer for lib (in-process) analysis.

        Reuses one instance per LogAnalyzer so the LogSage LRU cache is shared across
        requests (reduces LLM calls and improves latency). Creation is done inside
        _log_analysis_lock so only one instance is ever created. Does not use the MCP registry.
        """
        if self._lib_log_analyzer is not None:
            return self._lib_log_analyzer
        async with self._log_analysis_lock:
            if self._lib_log_analyzer is not None:
                return self._lib_log_analyzer
            args = argparse.Namespace(**run_kwargs)
            self._lib_log_analyzer = NVRxLogAnalyzer(args)
            return self._lib_log_analyzer

    def _post_analysis_results(
        self,
        result_items: List[Any],
        processing_time: float,
        path: str,
        user: str,
        job_id: str,
    ) -> None:
        """Post each analysis result to dataflow/Slack. Shared by lib and MCP paths."""
        if not result_items:
            return
        for item in result_items:
            raw_text = item[0] if isinstance(item, (list, tuple)) else item
            parsed = parse_llm_response(str(raw_text))
            if job_id:
                path_metadata = extract_job_metadata(path, warn_on_missing_job_id=False)
                metadata = JobMetadata(job_id=job_id, cycle_id=path_metadata.cycle_id)
            else:
                metadata = extract_job_metadata(path)
            post_results(parsed, metadata, path, processing_time, user)

    async def _fetch_log_result_lib(self, path: str) -> tuple[Dict[str, Any], float]:
        """Run LogSage in-process (lib backend); return (log_result dict, processing_time_seconds)."""
        is_per_cycle = bool(re.search(CYCLE_LOG_PATTERN, path))
        run_kwargs = {
            "log_path": path,
            "model": self.config.llm_model,
            "temperature": self.config.llm_temperature,
            "exclude_nvrx_logs": False,
            "is_per_cycle": is_per_cycle,
            "top_p": self.config.llm_top_p,
            "max_tokens": self.config.llm_max_tokens,
        }
        analyzer = await self._get_lib_log_analyzer(run_kwargs)
        s_time = time.time()
        async with self._log_analysis_lock:
            result = await analyzer.run(run_kwargs)
        processing_time = time.time() - s_time

        if result is None:
            logger.error("Lib log analyzer run returned None for path=%s", path)
            raise RuntimeError("LogSage run returned None")

        actual_result = result
        state = AttributionState.CONTINUE
        if isinstance(result, tuple) and len(result) == 2:
            try:
                actual_result, state = result
            except (ValueError, TypeError) as e:
                logger.error(
                    "Lib log analyzer result unpack failed for path=%s: %s",
                    path,
                    e,
                    exc_info=True,
                )
                raise RuntimeError(
                    f"LogSage result had unexpected shape for path={path}: {e}"
                ) from e

        log_result = {
            "result": actual_result,
            "state": state.name if isinstance(state, AttributionState) else str(state),
        }
        return log_result, processing_time

    def _is_mcp_connection_error(self, exc: BaseException) -> bool:
        """True if the exception suggests MCP connection/subprocess failure."""
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionError)):
            return True
        if isinstance(exc, OSError) and exc.errno in (errno.EPIPE, errno.ECONNRESET):
            return True
        if isinstance(exc, asyncio.IncompleteReadError):
            return True
        return False

    async def _fetch_log_result_mcp(self, path: str) -> tuple[Dict[str, Any], float]:
        """Run LogSage via MCP subprocess; return (log_result dict, processing_time_seconds).

        Uses the MCP client created at init. connect_mcp() must be called during startup.
        On connection failure, attempts reconnect and retries once.
        """
        if self._mcp_client is None:
            raise RuntimeError("MCP client not initialized (use_lib_log_analysis is True)")
        if not getattr(self._mcp_client, "session", None):
            raise RuntimeError("MCP client not connected; call connect_mcp() during startup")

        is_per_cycle = bool(re.search(CYCLE_LOG_PATTERN, path))
        run_kwargs = dict(
            module_name="log_analyzer",
            log_path=path,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            exclude_nvrx_logs=False,
            is_per_cycle=is_per_cycle,
            top_p=self.config.llm_top_p,
            max_tokens=self.config.llm_max_tokens,
        )

        s_time = time.time()
        async with self._log_analysis_lock:
            try:
                log_result = await self._mcp_client.run_module(**run_kwargs)
            except BaseException as e:
                if self._is_mcp_connection_error(e):
                    logger.warning("MCP connection error, attempting reconnect: %s", e)
                    if await self.reconnect_mcp():
                        log_result = await self._mcp_client.run_module(**run_kwargs)
                    else:
                        raise RuntimeError("MCP reconnect failed after connection error") from e
                else:
                    raise
        processing_time = time.time() - s_time
        return log_result, processing_time

    async def _run_llm_analysis(
        self, path: str, user: str = "unknown", job_id: str = ""
    ) -> Dict[str, Any]:
        """Run LLM analysis on a log file (lib or MCP backend).

        Args:
            path: Path to the log file
            user: User who submitted the job
            job_id: Job ID (from POST request); falls back to path extraction if empty
        """
        if not os.access(path, os.R_OK):
            self._total_permission_errors += 1
            self._file_permission_errors += 1
            raise PermissionError(f"Log file not readable: {path}")

        if self.config.use_lib_log_analysis:
            log_result, processing_time = await self._fetch_log_result_lib(path)
        else:
            log_result, processing_time = await self._fetch_log_result_mcp(path)

        if log_result is None:
            logger.error("Log analyzer returned None for path=%s", path)
            raise RuntimeError("Log analyzer returned None")
        if not isinstance(log_result, dict):
            logger.error(
                "Log analyzer returned non-dict for path=%s: type=%s",
                path,
                type(log_result).__name__,
            )
            raise RuntimeError(f"Log analyzer returned {type(log_result).__name__}, expected dict")

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
        self._post_analysis_results(result_items, processing_time, path, user, job_id)
        return log_result

    async def _analyze_splitlog_mode(
        self,
        slurm_output_path: str,
        job: Job,
        file: Optional[str],
        wl_restart: Optional[int],
    ) -> SplitlogAnalysisResult | AnalyzerError:
        """Handle analysis for split logging mode jobs."""
        if file is None:
            job.mark_terminated()

        self._splitlog_tracker.poll_now()
        file_info = self._splitlog_tracker.get_file_info(job, file)

        if not file_info:
            if len(job.file_info) == 0:
                return AnalyzerError(
                    error_code=ErrorCode.NOT_FOUND,
                    message="no log files detected yet",
                )
            return AnalyzerError(
                error_code=ErrorCode.NOT_FOUND,
                message=f"file '{file}' not found" if file else "no log files available",
            )

        try:
            result = await self._coalescer.get_or_compute(
                file_info.log_file,
                lambda: self._run_llm_analysis(
                    file_info.log_file, user=job.user, job_id=job.job_id
                ),
            )
        except Exception as e:
            return AnalyzerError(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"analysis failed: {e}",
            )

        return SplitlogAnalysisResult(
            result=result,
            status="completed",
            mode=JobMode.SPLITLOG.value,
            sched_restarts=job.sched_restarts,
            log_file=file_info.log_file,
            wl_restart=wl_restart or 0,
        )

    def _sync_analyze_via_coalescer(
        self, log_path: str, user: str, job_id: str = ""
    ) -> Dict[str, Any]:
        """Synchronous wrapper for background thread analysis.

        Uses asyncio.run_coroutine_threadsafe() to schedule work on the main
        event loop rather than creating a new loop. This ensures proper task
        lifecycle management and avoids orphaned tasks.

        Requires set_event_loop() to be called during app startup.
        """
        if self._main_loop is None:
            raise RuntimeError("Event loop not set - call set_event_loop() during app startup")

        future = asyncio.run_coroutine_threadsafe(
            self._coalescer.get_or_compute(
                log_path, lambda: self._run_llm_analysis(log_path, user=user, job_id=job_id)
            ),
            self._main_loop,
        )
        # Block until complete (same behavior as before)
        return future.result(timeout=self.config.compute_timeout)

    def _fire_and_forget_analyze(self, log_path: str, user: str = "", job_id: str = "") -> None:
        """
        Schedule analysis without waiting for completion.

        Used by splitlog tracker to trigger analysis from any context (async or sync)
        without blocking. The result will be available in the coalescer cache.
        """
        if self._main_loop is None:
            raise RuntimeError("Event loop not set - call set_event_loop() during app startup")

        # Schedule the coroutine but don't wait for it
        asyncio.run_coroutine_threadsafe(
            self._coalescer.get_or_compute(
                log_path, lambda: self._run_llm_analysis(log_path, user=user, job_id=job_id)
            ),
            self._main_loop,
        )
        logger.debug(f"Scheduled fire-and-forget analysis for {log_path}")

    def _check_pending_jobs(self) -> None:
        """Check pending jobs for LOGS_DIR and promote if found."""
        with self._jobs_lock:
            pending_jobs = [j for j in self._jobs.values() if j.is_pending()]
        if not pending_jobs:
            logger.debug("_check_pending_jobs: no pending jobs")
            return

        logger.debug(f"_check_pending_jobs: checking {len(pending_jobs)} pending jobs")
        now = time.monotonic()
        expired_count = 0
        promoted_count = 0

        for job in pending_jobs:
            age_seconds = now - job.created_at
            if age_seconds >= TTL_PENDING_SECONDS:
                logger.debug(
                    f"_check_pending_jobs: job {job.job_id} expired "
                    f"(age={age_seconds:.0f}s >= TTL={TTL_PENDING_SECONDS}s)"
                )
                with self._jobs_lock:
                    del self._jobs[job.path]
                expired_count += 1
                continue

            info = read_and_parse_slurm_output(job.path)
            logger.debug(
                f"_check_pending_jobs: job {job.job_id} slurm parse - "
                f"logs_dir={info.logs_dir if info else None}"
            )
            if info and info.logs_dir:
                is_dir = os.path.isdir(info.logs_dir)
                is_readable = os.access(info.logs_dir, os.R_OK) if is_dir else False
                logger.debug(
                    f"_check_pending_jobs: job {job.job_id} logs_dir check - "
                    f"is_dir={is_dir}, is_readable={is_readable}"
                )
                if is_dir and is_readable:
                    job.promote_to_splitlog(info.logs_dir)
                    self._splitlog_tracker.initialize_job(job)
                    self._total_splitlog += 1
                    self._deferred_splitlog += 1
                    promoted_count += 1
                    logger.debug(
                        f"_check_pending_jobs: job {job.job_id} promoted to SPLITLOG "
                        f"(logs_dir={info.logs_dir})"
                    )

        if expired_count > 0:
            self._pending_expired += expired_count

        logger.debug(
            f"_check_pending_jobs: complete - promoted={promoted_count}, expired={expired_count}"
        )

    def _get_splitlog_jobs(self) -> List[Job]:
        """Get all splitlog mode jobs."""
        with self._jobs_lock:
            return [j for j in self._jobs.values() if j.is_splitlog()]

    def _cleanup_job(self, path: str) -> None:
        """Remove a job from storage."""
        with self._jobs_lock:
            if path in self._jobs:
                del self._jobs[path]

    # ─── Stats and introspection ───

    async def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        stats = await self._coalescer.get_stats()
        folder_stats = self._splitlog_tracker.get_stats()

        stats[JobMode.SPLITLOG.value] = folder_stats
        with self._jobs_lock:
            pending_count = sum(1 for j in self._jobs.values() if j.is_pending())
        stats["detection"] = {
            "total_splitlog": self._total_splitlog,
            "total_single": self._total_single,
            JobMode.PENDING.value: pending_count,
            "jobs_expired": self._pending_expired,
        }
        stats["deferred"] = {
            "total_splitlog": self._deferred_splitlog,
            "total_single": self._deferred_single,
        }
        stats["permission_errors"] = {
            "total": self._total_permission_errors,
            "logs_dir": self._logs_dir_permission_errors,
            "file": self._file_permission_errors,
        }

        return stats

    async def get_compute_health_metrics(self) -> ComputeStats:
        """Get compute/LLM stats for health checks. Prefer over parsing get_stats()['compute']."""
        return await self._coalescer.get_compute_stats()

    async def get_cache(self) -> CacheResult:
        """Get current cache contents."""
        return await self._coalescer.get_cache()

    def export_cache(self) -> List[Dict[str, Any]]:
        """Export cache entries for persistence. See RequestCoalescer.export_cache()."""
        return self._coalescer.export_cache()

    def import_cache(self, entries: List[Dict[str, Any]]) -> int:
        """Import cache entries from persistence. See RequestCoalescer.import_cache()."""
        return self._coalescer.import_cache(entries)

    async def get_inflight(self) -> InflightResult:
        """Get currently in-flight requests."""
        return await self._coalescer.get_inflight()

    async def get_submitted(self) -> SubmittedResult:
        """Get submitted paths."""
        return await self._coalescer.get_submitted()

    def get_all_jobs(self) -> Dict[str, Any]:
        """Get all tracked jobs."""
        pending_jobs = []
        single_jobs = []
        splitlog_jobs = []

        with self._jobs_lock:
            jobs_snapshot = list(self._jobs.values())

        for job in jobs_snapshot:
            job_info = {
                STATS_JOB_ID: job.job_id or "unknown",
                STATS_LOG_PATH: job.path,
                STATS_USER: job.user,
                RESP_MODE: job.mode.value,  # JobMode enum
            }
            if job.is_pending():
                pending_jobs.append(job_info)
            elif job.is_single():
                single_jobs.append(job_info)
            elif job.is_splitlog():
                job_info.update(
                    {
                        RESP_LOGS_DIR: job.logs_dir,
                        RESP_SCHED_RESTARTS: job.sched_restarts,
                        RESP_FILES_ANALYZED: job.files_complete(),
                        STATS_TERMINATED: job.terminated,
                        STATS_LOG_FILES: job.known_log_files,
                    }
                )
                splitlog_jobs.append(job_info)

        return {
            JobMode.PENDING.value: {"count": len(pending_jobs), "jobs": pending_jobs},
            JobMode.SINGLE.value: {"count": len(single_jobs), "jobs": single_jobs},
            JobMode.SPLITLOG.value: {"count": len(splitlog_jobs), "jobs": splitlog_jobs},
        }
