#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Split logging mode tracking for jobs with separate log directories.

This module provides the SplitlogTracker class for managing jobs that write
separate log files for each scheduler restart (as opposed to appending to
a single slurm output file). It handles:

- Background polling of tracked jobs to detect new log files
- Scheduler restart detection via << START PATHS >> markers in slurm output
- Log file discovery in LOGS_DIR using configurable glob patterns
- Triggering analysis when files are complete (non-blocking fire-and-forget)
- Job lifecycle management with TTL-based cleanup

Terminology:
- sched_restarts: Count of scheduler restarts, detected by << START PATHS >> markers
- wl_restarts: Count of workload restarts within a file, detected by Cycle: N markers

Architecture notes:
- SplitlogTracker does NOT own job storage; it uses callbacks to access Job
  objects stored in LogAnalyzer._jobs
- Analysis is triggered via fire-and-forget callback to avoid blocking the
  async event loop from the background polling thread
- Results are stored in RequestCoalescer cache and retrieved via GET /logs
"""

import glob
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import (
    RESP_FILES_ANALYZED,
    RESP_LOGS_DIR,
    RESP_SCHED_RESTARTS,
    STATS_FILES,
    STATS_FILES_ANALYZED,
    STATS_FILES_TRACKED,
    STATS_JOB_ID,
    STATS_JOBS,
    STATS_JOBS_CLEANED,
    STATS_JOBS_TERMINATED,
    STATS_LOG_FILES,
    STATS_SCHED_RESTARTS,
    STATS_TERMINATED,
)
from .job import FileInfo, Job
from .slurm_parser import read_and_parse_slurm_output
from .utils import CYCLE_NUM_PATTERN, DATE_TIME_PATTERN

logger = logging.getLogger(__name__)


def _escape_glob(s: str) -> str:
    """Escape glob/fnmatch metacharacters so the string is matched literally."""
    # Order: [ and ] first so we don't escape the brackets we add for * and ?
    s = s.replace("[", "[[]")
    s = s.replace("]", "[]]")
    s = s.replace("*", "[*]")
    s = s.replace("?", "[?]")
    return s


# Default polling interval for splitlog mode jobs
DEFAULT_POLL_INTERVAL_SECONDS = 300.0  # 5 minutes

# TTL for terminated jobs (1 hour after termination)
DEFAULT_TERMINATED_JOB_TTL_SECONDS = 3600.0  # 1 hour

# TTL for non-terminated jobs (6 months max age)
DEFAULT_MAX_JOB_AGE_SECONDS = 180 * 24 * 3600.0  # 6 months


class SplitlogTracker:
    """
    Background tracker for split logging mode jobs.

    This class manages jobs that write separate log files to a LOGS_DIR for each
    scheduler restart. It does NOT own job storage - it uses callbacks to access
    Job objects stored in LogAnalyzer._jobs.

    Key responsibilities:
    - Background polling thread that runs every poll_interval seconds
    - Detects new scheduler restarts by parsing slurm output for << START PATHS >> markers
    - Discovers log files in LOGS_DIR using configurable glob patterns
    - Triggers analysis for complete files (all files except the last active one)
    - Cleans up terminated jobs after TTL expiration

    Thread safety:
    - All job state access is protected by self._lock
    - Analysis is triggered via ThreadPoolExecutor to avoid blocking async event loop

    Callbacks (set by LogAnalyzer during initialization):
    - set_analyze_callback: Called to trigger analysis (fire-and-forget)
    - set_pending_check_callback: Called each poll cycle to check pending jobs
    - set_get_splitlog_jobs_callback: Returns list of splitlog mode jobs
    - set_cleanup_job_callback: Removes job from storage after TTL expiration
    """

    def __init__(
        self,
        poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
        log_pattern: str = "*_{job_id}_*.log",
        terminated_job_ttl: float = DEFAULT_TERMINATED_JOB_TTL_SECONDS,
        max_job_age: float = DEFAULT_MAX_JOB_AGE_SECONDS,
    ):
        """
        Initialize the splitlog tracker.

        Args:
            poll_interval: How often to poll tracked jobs (seconds)
            log_pattern: Glob pattern for finding log files. {job_id} is replaced.
            terminated_job_ttl: How long to keep terminated jobs before cleanup (seconds)
            max_job_age: Maximum age for non-terminated jobs before cleanup (seconds)
        """
        self._lock = threading.Lock()
        self._poll_interval = poll_interval
        self._log_pattern = log_pattern
        self._terminated_job_ttl = terminated_job_ttl
        self._max_job_age = max_job_age
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._analyze_callback: Optional[Callable[[str, str, str], None]] = None
        self._pending_check_callback: Optional[Callable[[], None]] = None
        self._get_splitlog_jobs_callback: Optional[Callable[[], List[Job]]] = None
        self._cleanup_job_callback: Optional[Callable[[str], None]] = None
        self._jobs_cleaned: int = 0  # Counter for cleaned up jobs
        # Thread pool for non-blocking analysis (avoids blocking async event loop)
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="splitlog-analyze")

    def set_analyze_callback(self, callback: Callable[[str, str, str], None]) -> None:
        """
        Set the callback function for analyzing log files.

        The callback should schedule analysis without blocking (fire-and-forget).
        Results are stored in the coalescer cache and retrieved via GET /logs.

        Args:
            callback: Function that takes (log_file_path, user, job_id) and schedules analysis
        """
        self._analyze_callback = callback

    def set_pending_check_callback(self, callback: Callable[[], None]) -> None:
        """
        Set the callback function for checking pending jobs.

        Called during each poll cycle to check if pending jobs
        should be promoted to splitlog mode.

        Args:
            callback: Function that checks pending jobs for LOGS_DIR
        """
        self._pending_check_callback = callback

    def set_get_splitlog_jobs_callback(self, callback: Callable[[], List[Job]]) -> None:
        """
        Set the callback to get splitlog jobs from service.

        Args:
            callback: Function that returns list of splitlog mode jobs
        """
        self._get_splitlog_jobs_callback = callback

    def set_cleanup_job_callback(self, callback: Callable[[str], None]) -> None:
        """
        Set the callback to remove a job from service.

        Args:
            callback: Function that takes path and removes job from storage
        """
        self._cleanup_job_callback = callback

    def start_polling(self) -> None:
        """Start the background polling thread."""
        if self._poll_thread is not None and self._poll_thread.is_alive():
            logger.warning("Polling thread already running")
            return

        self._stop_event.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        logger.info(f"Started splitlog tracker polling (interval={self._poll_interval}s)")

    def stop_polling(self) -> None:
        """Stop the background polling thread and executor."""
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=5.0)
            self._poll_thread = None
        self._executor.shutdown(wait=False)
        logger.info("Stopped splitlog tracker polling")

    def initialize_job(self, job: Job) -> None:
        """
        Initialize a job for splitlog tracking (initial scan).

        Called by service when a job is promoted to splitlog mode.

        Args:
            job: Job object to initialize (must have logs_dir set)
        """
        with self._lock:
            self._update_job_state(job)
            logger.info(
                f"Initialized job {job.job_id} for splitlog tracking: "
                f"logs_dir={job.logs_dir}, sched_restarts={job.sched_restarts}"
            )
            # Analyze any files that are already complete (not the last file)
            self._analyze_pending_files(job)

    def _cleanup_expired_jobs(self) -> int:
        """
        Remove jobs that have exceeded their TTL.

        Two cleanup rules:
        1. Terminated jobs: removed after terminated_job_ttl (1 hour)
        2. Non-terminated jobs: removed after max_job_age (6 months)

        Returns:
            Number of jobs removed
        """
        if not self._get_splitlog_jobs_callback or not self._cleanup_job_callback:
            return 0

        now = time.monotonic()
        jobs = self._get_splitlog_jobs_callback()
        to_remove = []

        for job in jobs:
            if job.terminated and job.terminated_at is not None:
                # Terminated via GET: short TTL
                if now - job.terminated_at >= self._terminated_job_ttl:
                    to_remove.append((job.path, job.job_id, "terminated TTL"))
            else:
                # Not terminated: max age TTL
                if now - job.created_at >= self._max_job_age:
                    to_remove.append((job.path, job.job_id, "max age"))

        for path, job_id, reason in to_remove:
            self._cleanup_job_callback(path)
            logger.info(f"Cleaned up job {job_id} ({reason} expired)")

        self._jobs_cleaned += len(to_remove)
        return len(to_remove)

    def get_file_info(self, job: Job, filename: Optional[str] = None) -> Optional[FileInfo]:
        """
        Get info for a specific file or the latest file.

        Note: Results are stored in RequestCoalescer cache, not here.
        Use the returned FileInfo.log_file as the cache key.

        Args:
            job: Job object
            filename: Filename (basename) to look up, or None for latest

        Returns:
            FileInfo or None if not found
        """
        with self._lock:
            if filename is not None:
                return job.file_info.get(filename)

            # Return latest file where analysis was triggered
            if not job.file_info:
                return None

            # Find the most recently analyzed file (by analysis order)
            triggered_files = [info for info in job.file_info.values() if info.analysis_triggered]
            if triggered_files:
                # Return the last one in known_log_files order
                for log_file in reversed(job.known_log_files):
                    basename = os.path.basename(log_file)
                    if basename in job.file_info and job.file_info[basename].analysis_triggered:
                        return job.file_info[basename]

            # If no analysis triggered yet, return the first known file
            if job.known_log_files:
                first_file = os.path.basename(job.known_log_files[0])
                return job.file_info.get(first_file)

            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about tracked jobs."""
        jobs = self._get_splitlog_jobs_callback() if self._get_splitlog_jobs_callback else []
        return {
            STATS_JOBS: len(jobs),
            STATS_JOBS_TERMINATED: sum(1 for j in jobs if j.terminated),
            STATS_JOBS_CLEANED: self._jobs_cleaned,
            STATS_FILES: {
                STATS_FILES_TRACKED: sum(len(j.file_info) for j in jobs),
                STATS_FILES_ANALYZED: sum(j.files_complete() for j in jobs),
            },
            STATS_SCHED_RESTARTS: sum(j.sched_restarts for j in jobs),
        }

    def get_jobs_detail(self) -> List[Dict[str, Any]]:
        """Get per-job details for all tracked jobs."""
        jobs = self._get_splitlog_jobs_callback() if self._get_splitlog_jobs_callback else []
        return [
            {
                STATS_JOB_ID: job.job_id,
                RESP_LOGS_DIR: job.logs_dir,
                RESP_SCHED_RESTARTS: job.sched_restarts,
                RESP_FILES_ANALYZED: job.files_complete(),
                STATS_TERMINATED: job.terminated,
                STATS_LOG_FILES: job.known_log_files,
            }
            for job in jobs
        ]

    def poll_now(self) -> None:
        """Trigger an immediate poll of all tracked jobs."""
        jobs = self._get_splitlog_jobs_callback() if self._get_splitlog_jobs_callback else []
        for job in jobs:
            self._poll_job(job)

    def _poll_loop(self) -> None:
        """Background polling loop."""
        while not self._stop_event.is_set():
            try:
                # Get job count for logging
                jobs = (
                    self._get_splitlog_jobs_callback() if self._get_splitlog_jobs_callback else []
                )
                logger.debug(f"Poll cycle starting: {len(jobs)} splitlog jobs tracked")

                # Check pending jobs for LOGS_DIR (may promote to splitlog mode)
                if self._pending_check_callback:
                    self._pending_check_callback()

                # Poll tracked splitlog mode jobs
                self.poll_now()

                # Cleanup jobs that have exceeded TTL
                cleaned = self._cleanup_expired_jobs()
                if cleaned:
                    logger.debug(f"Poll cycle: cleaned {cleaned} expired jobs")

                logger.debug(f"Poll cycle complete, sleeping {self._poll_interval}s")
            except Exception as e:
                logger.error(f"Error in splitlog tracker poll loop: {e}", exc_info=True)

            # Wait for next poll interval or stop event
            self._stop_event.wait(timeout=self._poll_interval)

    def _poll_job(self, job: Job) -> None:
        """Poll a single job for updates."""
        if job.terminated:
            logger.debug(f"Job {job.job_id}: skipping poll (terminated)")
            return

        try:
            with self._lock:
                prev_restarts = job.sched_restarts
                prev_files = set(job.known_log_files)

            logger.debug(
                f"Job {job.job_id}: polling (logs_dir={job.logs_dir}, "
                f"prev_restarts={prev_restarts}, prev_files={len(prev_files)})"
            )

            self._update_job_state(job)

            with self._lock:
                job.last_poll_at = time.monotonic()

                logger.debug(
                    f"Job {job.job_id}: state after poll - "
                    f"sched_restarts={job.sched_restarts}, files={len(job.known_log_files)}, "
                    f"file_info={len(job.file_info)}"
                )

                # Check if new scheduler restart detected
                if job.sched_restarts > prev_restarts:
                    logger.info(
                        f"Job {job.job_id}: new scheduler restart detected "
                        f"({prev_restarts} -> {job.sched_restarts})"
                    )
                    # Analyze files from previous restart(s) if not already done
                    self._analyze_pending_files(job)

                # Check if new log files appeared
                new_files = set(job.known_log_files) - prev_files
                if new_files:
                    logger.info(f"Job {job.job_id}: new log files: {new_files}")
                    # New file means previous file is complete
                    self._analyze_pending_files(job)
                elif job.sched_restarts == prev_restarts:
                    logger.debug(f"Job {job.job_id}: no changes detected")

        except Exception as e:
            logger.error(f"Error polling job {job.job_id}: {e}", exc_info=True)

    def _update_job_state(self, job: Job) -> None:
        """Update job state by reading slurm output and LOGS_DIR."""
        # Read slurm output for scheduler restart count (<< START PATHS >> markers)
        info = read_and_parse_slurm_output(job.path)
        if info:
            old_restarts = job.sched_restarts
            job.sched_restarts = info.cycle_count  # cycle_count = sched_restart count
            if job.sched_restarts != old_restarts:
                logger.debug(
                    f"Job {job.job_id}: sched_restarts updated {old_restarts} -> {job.sched_restarts}"
                )
        else:
            logger.debug(f"Job {job.job_id}: no slurm info from path={job.path}")

        # Scan LOGS_DIR for log files
        log_files = self._find_log_files(job.logs_dir, job.job_id)
        old_file_count = len(job.known_log_files)
        job.known_log_files = log_files
        if len(log_files) != old_file_count:
            logger.debug(f"Job {job.job_id}: log files count {old_file_count} -> {len(log_files)}")

        # Create FileInfo entries for new files (keyed by filename/basename)
        new_file_infos = 0
        for log_file in log_files:
            filename = os.path.basename(log_file)
            if filename not in job.file_info:
                job.file_info[filename] = FileInfo(log_file=log_file)
                new_file_infos += 1
                logger.debug(f"Job {job.job_id}: added FileInfo for {filename}")
        if new_file_infos:
            logger.debug(f"Job {job.job_id}: created {new_file_infos} new FileInfo entries")

    def _find_log_files(self, logs_dir: Optional[str], job_id: Optional[str]) -> List[str]:
        """
        Find log files for a job in LOGS_DIR and sort by cycle order.

        Sorting priority:
        1. Explicit cycle number in filename: *_cycle<N>.log
        2. Date/time in filename: *_date_YY-MM-DD_time_HH-MM-SS.log
        3. Fallback to modification time

        Args:
            logs_dir: Path to logs directory
            job_id: Job ID to match

        Returns:
            List of log file paths, sorted by cycle order
        """
        if not logs_dir or not job_id:
            logger.debug(f"_find_log_files: skipped (logs_dir={logs_dir}, job_id={job_id})")
            return []

        if not os.path.isdir(logs_dir):
            logger.debug(f"_find_log_files: logs_dir not a directory: {logs_dir}")
            return []

        # Build pattern with job_id (escape so metacharacters don't inject)
        pattern = self._log_pattern.replace("{job_id}", _escape_glob(job_id))
        full_pattern = os.path.join(logs_dir, pattern)
        logger.debug(f"_find_log_files: searching pattern={full_pattern}")

        # Find matching files, exclude .env.log
        matches = [
            f for f in glob.glob(full_pattern) if not f.endswith(".env.log") and os.path.isfile(f)
        ]

        if not matches:
            logger.debug(f"_find_log_files: no matches for pattern {full_pattern}")
            return []

        logger.debug(f"_find_log_files: found {len(matches)} files matching pattern")

        # Try to extract cycle info using priority-based approach
        sorted_files = self._sort_log_files_by_cycle(matches)

        return sorted_files

    def _sort_log_files_by_cycle(self, files: List[str]) -> List[str]:
        """
        Sort log files by cycle order using priority-based matching.

        Priority:
        1. Explicit cycle number: *_cycle<N>.log
        2. Date/time in filename: *_date_YY-MM-DD_time_HH-MM-SS.log
        3. Fallback to modification time

        Args:
            files: List of file paths to sort

        Returns:
            Sorted list of file paths
        """
        # Strategy 1: Check for explicit cycle numbers
        cycle_numbered = self._extract_cycle_numbers(files)
        if cycle_numbered:
            # All files have cycle numbers - sort by cycle number
            logger.debug(f"Sorting {len(files)} files by explicit cycle number")
            return [f for _, f in sorted(cycle_numbered)]

        # Strategy 2: Check for date/time in filenames
        date_sorted = self._extract_datetime_from_filenames(files)
        if date_sorted:
            # All files have date/time - sort by datetime
            logger.debug(f"Sorting {len(files)} files by filename date/time")
            return [f for _, f in sorted(date_sorted)]

        # Strategy 3: Fall back to modification time
        logger.debug(f"Sorting {len(files)} files by modification time")
        files.sort(key=lambda f: os.path.getmtime(f))
        return files

    def _extract_cycle_numbers(self, files: List[str]) -> List[Tuple[int, str]]:
        """
        Extract cycle numbers from filenames.

        Args:
            files: List of file paths

        Returns:
            List of (cycle_number, file_path) tuples, or empty if not all files have cycle numbers
        """
        result = []
        for f in files:
            match = CYCLE_NUM_PATTERN.search(f)
            if match:
                cycle_num = int(match.group(1))
                result.append((cycle_num, f))
            else:
                # Not all files have cycle numbers, abort this strategy
                return []
        return result

    def _extract_datetime_from_filenames(self, files: List[str]) -> List[Tuple[str, str]]:
        """
        Extract date/time from filenames for sorting.

        Matches pattern: *_date_YY-MM-DD_time_HH-MM-SS.log

        Args:
            files: List of file paths

        Returns:
            List of (sortable_datetime_str, file_path) tuples, or empty if not all files have dates
        """
        result = []
        for f in files:
            match = DATE_TIME_PATTERN.search(f)
            if match:
                # Extract YY, MM, DD, HH, MM, SS
                yy, mo, dd, hh, mi, ss = match.groups()
                # Create sortable string: YYMMDDHHMMSS
                datetime_str = f"{yy}{mo}{dd}{hh}{mi}{ss}"
                result.append((datetime_str, f))
            else:
                # Not all files have date/time, abort this strategy
                return []
        return result

    def _analyze_pending_files(self, job: Job) -> None:
        """Trigger analysis for files that are complete but not yet triggered."""
        if self._analyze_callback is None:
            logger.warning("No analyze callback set, skipping analysis")
            return

        logger.debug(
            f"Job {job.job_id}: checking {len(job.known_log_files)} files for pending analysis "
            f"(terminated={job.terminated})"
        )

        # A file is complete if there's a newer file (or job terminated for last)
        files_analyzed = 0
        files_pending = 0
        for i, log_file in enumerate(job.known_log_files):
            filename = os.path.basename(log_file)
            file_info = job.file_info.get(filename)
            if not file_info:
                logger.debug(f"Job {job.job_id}: no file_info for {filename}, skipping")
                continue
            if file_info.analysis_triggered:
                logger.debug(f"Job {job.job_id}: {filename} already triggered, skipping")
                continue

            # Check if this file is complete:
            # - Not the last file (a newer file exists, so this one is done)
            # - OR job is terminated (all files are complete)
            is_last_file = i == len(job.known_log_files) - 1
            is_complete = not is_last_file or job.terminated

            logger.debug(
                f"Job {job.job_id}: {filename} - is_last={is_last_file}, "
                f"terminated={job.terminated}, is_complete={is_complete}"
            )

            if is_complete:
                self._trigger_analysis(job, file_info)
                files_analyzed += 1
            else:
                files_pending += 1

        if files_analyzed or files_pending:
            logger.debug(
                f"Job {job.job_id}: analyzed={files_analyzed}, still_pending={files_pending}"
            )

    def _trigger_analysis(self, job: Job, file_info: FileInfo) -> None:
        """
        Trigger analysis for a file (non-blocking).

        The callback is submitted to a thread pool executor to avoid blocking
        the async event loop. Results are stored in the coalescer cache.
        """
        filename = os.path.basename(file_info.log_file)
        logger.info(
            f"Triggering analysis for job {job.job_id} file {filename}: {file_info.log_file}"
        )

        # Mark as triggered immediately (non-blocking)
        file_info.analysis_triggered = True

        # Define the async analysis task
        def run_analysis() -> None:
            try:
                # Callback triggers analysis via coalescer - result goes to coalescer cache
                # Pass user and job_id so dataflow posting can attribute the analysis
                self._analyze_callback(file_info.log_file, job.user, job.job_id)
                file_info.analysis_complete = True
                file_info.analyzed_at = time.time()
                logger.info(f"Job {job.job_id} file {filename} analysis complete")
            except Exception as e:
                file_info.analysis_complete = True
                file_info.analyzed_at = time.time()
                logger.error(
                    f"Error analyzing job {job.job_id} file {filename}: {e}",
                    exc_info=True,
                )

        # Submit to thread pool (non-blocking)
        self._executor.submit(run_analysis)
