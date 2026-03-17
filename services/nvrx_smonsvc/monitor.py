#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""SLURM Job Monitor - main monitor class."""

import logging
import os
import re
import signal
import threading
import time
from types import FrameType

from .attrsvc_client import AttrsvcClient
from .job_handlers import fetch_results, submit_log
from .models import JobState, MonitorState, SlurmJob, copy_tracking_fields
from .slurm import SlurmClient, expand_slurm_patterns
from .stats import format_stats_summary, get_health_status, get_jobs_list, get_stats_dict
from .status_server import StatusServer

logger = logging.getLogger(__name__)


class SlurmJobMonitor:
    """
    Monitors SLURM jobs and integrates with the attribution service.
    """

    # Configuration constants
    DEFAULT_PARTITIONS = ["batch", "batch_long"]
    DEFAULT_POLL_INTERVAL = 180  # seconds (3 minutes)
    DEFAULT_HTTP_TIMEOUT = 60.0  # seconds
    DEFAULT_SLURM_TIMEOUT = 30  # seconds for squeue
    DEFAULT_SCONTROL_TIMEOUT = 45  # seconds for scontrol/sacct
    SCONTROL_BATCH_SIZE = 50  # jobs per scontrol call
    HTTP_MAX_ATTEMPTS = 3  # 1 initial + 2 retries
    HTTP_RETRY_DELAY = 5.0  # seconds between retries (server errors)
    HTTP_RATE_LIMIT_BASE_DELAY = 10.0  # base delay for 429 (exponential backoff)
    HTTP_RATE_LIMIT_MAX_DELAY = 60.0  # max delay for 429
    HTTP_REQUEST_THROTTLE = 0.25  # seconds between HTTP requests (rate limit prevention)
    MAX_COMPLETED_JOBS = 1000  # max completed jobs to keep in memory
    MAX_GET_ATTEMPTS = 10  # max GET attempts per job before giving up
    SQUEUE_JOB_STATES = "RUNNING,COMPLETING"  # job states to query

    def __init__(
        self,
        attrsvc_url: str,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        user: str | None = None,
        all_users: bool = True,
        partitions: list[str] | None = None,
        job_pattern: str | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
        port: int | None = None,
    ):
        """
        Initialize the SLURM job monitor.

        Args:
            attrsvc_url: URL of the attribution service (e.g., http://localhost:8000)
            poll_interval: How often to poll SLURM for job status (seconds)
            user: Filter jobs by user (default: current user, ignored if all_users=True)
            all_users: If True, monitor jobs from all users in specified partitions
            partitions: List of partitions to monitor (default: ["batch", "batch_long"])
            job_pattern: Regex pattern to match job names
            timeout: HTTP request timeout in seconds
            port: Port for HTTP server with stats/health/jobs endpoints (None to disable)
        """
        self.attrsvc_url = attrsvc_url.rstrip("/")
        self.poll_interval = poll_interval
        self.all_users = all_users
        self.user = (
            None
            if all_users
            else (user or os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown")
        )
        self.partitions = partitions if partitions is not None else self.DEFAULT_PARTITIONS
        self.job_pattern = re.compile(job_pattern) if job_pattern else None
        self.timeout = timeout
        self.state = MonitorState()
        self._shutdown_requested = False

        # Attribution service client
        self._attrsvc_client = AttrsvcClient(
            base_url=attrsvc_url,
            timeout=timeout,
            max_attempts=self.HTTP_MAX_ATTEMPTS,
            retry_delay=self.HTTP_RETRY_DELAY,
            rate_limit_base_delay=self.HTTP_RATE_LIMIT_BASE_DELAY,
            rate_limit_max_delay=self.HTTP_RATE_LIMIT_MAX_DELAY,
            request_throttle=self.HTTP_REQUEST_THROTTLE,
            on_rate_limited=self._on_rate_limited,
        )

        # SLURM client for subprocess calls
        self._slurm_client = SlurmClient(
            partitions=self.partitions,
            user=self.user,
            all_users=self.all_users,
            squeue_timeout=self.DEFAULT_SLURM_TIMEOUT,
            scontrol_timeout=self.DEFAULT_SCONTROL_TIMEOUT,
        )

        # Fail fast if SLURM is not available
        if not self._slurm_client.check_slurm_available():
            raise RuntimeError("SLURM is not available. Ensure squeue is installed and in PATH.")

        # Validate partitions exist
        valid, invalid = self._slurm_client.validate_partitions()
        if invalid:
            available = self._slurm_client.get_available_partitions()
            available_str = ", ".join(available) if available else "(could not retrieve)"
            raise RuntimeError(
                f"Invalid SLURM partitions: {', '.join(invalid)}. "
                f"Available partitions: {available_str}"
            )

        # Thread synchronization for state access
        self._state_lock = threading.Lock()

        # Status server
        self._port = port
        self._status_server: StatusServer | None = None

    def _on_rate_limited(self) -> None:
        """Callback when rate limited by attrsvc."""
        self.state.http_rate_limited += 1

    def __enter__(self) -> "SlurmJobMonitor":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures HTTP client is closed."""
        self.close()

    def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._attrsvc_client is not None:
            self._attrsvc_client.close()
            self._attrsvc_client = None

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)

    def _handle_shutdown_signal(self, signum: int, frame: FrameType | None = None) -> None:
        """Handle shutdown signals gracefully."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        self._shutdown_requested = True

    def _start_status_server(self) -> None:
        """Start the status HTTP server in a background thread."""
        if self._port is None:
            return

        self._status_server = StatusServer(
            port=self._port,
            get_stats=self._get_stats_for_http,
            get_jobs=self._get_jobs_for_http,
            get_health=self._get_health_for_http,
        )
        self._status_server.start()

    def _stop_status_server(self) -> None:
        """Stop the status HTTP server."""
        if self._status_server is not None:
            self._status_server.stop()
            self._status_server = None

    def _get_stats_for_http(self) -> dict:
        """Get monitor statistics for HTTP endpoint."""
        return get_stats_dict(self.state, self._state_lock)

    def _get_jobs_for_http(self) -> list:
        """Get jobs list for HTTP endpoint."""
        return get_jobs_list(self.state, self._state_lock)

    def _get_health_for_http(self) -> tuple:
        """Get health status for HTTP endpoint."""
        return get_health_status(self.state, self._state_lock, self._attrsvc_client)

    def run(self) -> None:
        """Main monitoring loop."""
        self._setup_signal_handlers()

        try:
            logger.info("=" * 60)
            logger.info("Starting SLURM Job Monitor")
            logger.info("=" * 60)
            logger.info(f"  Attribution service: {self.attrsvc_url}")
            logger.info(f"  Poll interval: {self.poll_interval}s ({self.poll_interval // 60}m)")
            if self.all_users:
                logger.info("  Monitoring: ALL users")
            else:
                logger.info(f"  Monitoring user: {self.user}")
            logger.info(f"  Partitions: {', '.join(self.partitions)}")
            if self.job_pattern:
                logger.info(f"  Job name pattern: {self.job_pattern.pattern}")

            self._start_status_server()
            logger.info("=" * 60)

            poll_cycle = 0
            while not self._shutdown_requested:
                try:
                    self._poll_and_process()
                    poll_cycle += 1
                    if poll_cycle % 20 == 0:
                        logger.info(f"Poll cycle {poll_cycle} - {self._get_stats_summary()}")
                except Exception as e:
                    logger.error(f"Error in poll cycle: {e}", exc_info=True)

                for _ in range(self.poll_interval):
                    if self._shutdown_requested:
                        break
                    time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            self._shutdown()

    def _get_stats_summary(self) -> str:
        """Get a one-line summary of current stats."""
        return format_stats_summary(self._get_stats_for_http())

    def _shutdown(self) -> None:
        """Clean shutdown of the monitor."""
        logger.info("Shutting down SLURM Job Monitor...")
        logger.info(f"Final stats - {self._get_stats_summary()}")
        self._stop_status_server()
        self.close()
        logger.info("Shutdown complete.")

    def _poll_and_process(self) -> None:
        """Poll SLURM for jobs and process state changes."""
        current_jobs = self._get_slurm_jobs()

        if current_jobs is None:
            logger.warning("Skipping state updates due to squeue failure")
            self._fetch_pending_results()
            return

        jobs_to_submit: list[tuple[SlurmJob, str]] = []
        jobs_to_fetch: list[tuple[SlurmJob, str]] = []

        with self._state_lock:
            self._mark_disappeared_jobs_finished(current_jobs)
            jobs_to_submit = self._sync_jobs_from_slurm(current_jobs)
            self._fetch_paths_for_terminal_jobs()
            jobs_to_fetch = self._collect_jobs_for_fetch()

        for job, log_path in jobs_to_submit:
            self._submit_log(job, log_path)

        for job, log_path in jobs_to_fetch:
            self._fetch_results(job, log_path)

        self._cleanup_old_jobs()

    def _mark_get_exhausted(self, job_id: str, job: SlurmJob) -> None:
        """Mark a job as exhausted after max GET attempts."""
        logger.warning(f"[{job_id}] Giving up on GET after {job.get_attempts} attempts")
        job.result_fetched = True  # Mark as done to allow cleanup

    def _fetch_pending_results(self) -> None:
        """Fetch results for terminal jobs when squeue is unavailable."""
        jobs_to_fetch: list[tuple[SlurmJob, str]] = []
        with self._state_lock:
            for job_id, tracked_job in self.state.jobs.items():
                if (
                    tracked_job.state.is_terminal()
                    and tracked_job.post_success
                    and not tracked_job.result_fetched
                    and tracked_job.get_attempts < self.MAX_GET_ATTEMPTS
                ):
                    log_path = self._get_log_path(tracked_job)
                    if log_path:
                        tracked_job.get_attempts += 1
                        jobs_to_fetch.append((tracked_job, log_path))
                elif tracked_job.get_attempts >= self.MAX_GET_ATTEMPTS:
                    self._mark_get_exhausted(job_id, tracked_job)

        for job, log_path in jobs_to_fetch:
            self._fetch_results(job, log_path)

    def _mark_disappeared_jobs_finished(self, current_jobs: dict[str, SlurmJob]) -> None:
        """Mark jobs that disappeared from squeue as FINISHED. Must hold _state_lock."""
        for job_id, tracked_job in list(self.state.jobs.items()):
            if job_id not in current_jobs and not tracked_job.state.is_terminal():
                logger.debug(
                    f"[{job_id}] Job no longer in queue: {tracked_job.state.value} -> FINISHED"
                )
                tracked_job.last_state = tracked_job.state
                tracked_job.state = JobState.FINISHED

    def _sync_jobs_from_slurm(
        self, current_jobs: dict[str, SlurmJob]
    ) -> list[tuple[SlurmJob, str]]:
        """
        Sync tracked jobs with current SLURM state. Must hold _state_lock.

        Returns list of (job, log_path) tuples for jobs needing submission.
        """
        jobs_to_submit: list[tuple[SlurmJob, str]] = []

        for job_id, job in current_jobs.items():
            if not self._matches_filters(job):
                continue

            prev_job = self.state.jobs.get(job_id)

            if prev_job is None:
                # New job
                logger.debug(f"[{job_id}] New job detected: {job.name} ({job.state.value})")
                self.state.jobs[job_id] = job
                self.state.jobs_seen += 1
                if job.stdout_path:
                    self.state.with_output_path += 1
            elif prev_job.state != job.state:
                # State changed - preserve tracking fields (single place: models.copy_tracking_fields)
                logger.debug(
                    f"[{job_id}] State change: {prev_job.state.value} -> {job.state.value}"
                )
                copy_tracking_fields(prev_job, job)
                if not job.stdout_path and prev_job.stdout_path:
                    job.stdout_path = prev_job.stdout_path
                if not job.stderr_path and prev_job.stderr_path:
                    job.stderr_path = prev_job.stderr_path
                self.state.jobs[job_id] = job

            tracked_job = self.state.jobs[job_id]

            # Check if job needs log submission
            if not tracked_job.log_submitted and not tracked_job.result_fetched:
                log_path = self._get_log_path(tracked_job)
                if log_path:
                    jobs_to_submit.append((tracked_job, log_path))

        return jobs_to_submit

    def _fetch_paths_for_terminal_jobs(self) -> None:
        """Fetch output paths for terminal jobs that don't have them. Must hold _state_lock."""
        terminal_jobs_needing_paths = [
            job_id
            for job_id, job in self.state.jobs.items()
            if job.state.is_terminal()
            and not job.result_fetched
            and not job.stdout_path
            and not job.path_fetch_attempted
            and "[" not in job_id
            and "+" not in job_id
        ]

        if not terminal_jobs_needing_paths:
            return

        logger.info(f"Fetching paths for {len(terminal_jobs_needing_paths)} terminal jobs")
        fetched_paths = self._get_job_output_paths_batch(terminal_jobs_needing_paths)

        for job_id in terminal_jobs_needing_paths:
            if job_id in self.state.jobs:
                self.state.jobs[job_id].path_fetch_attempted = True
                if job_id in fetched_paths:
                    stdout, stderr = fetched_paths[job_id]
                    if stdout:
                        if not self.state.jobs[job_id].stdout_path:
                            self.state.with_output_path += 1
                        self.state.jobs[job_id].stdout_path = stdout
                        self.state.jobs[job_id].stderr_path = stderr

    def _collect_jobs_for_fetch(self) -> list[tuple[SlurmJob, str]]:
        """Collect terminal jobs that need result fetching. Must hold _state_lock."""
        jobs_to_fetch: list[tuple[SlurmJob, str]] = []

        for job_id, tracked_job in self.state.jobs.items():
            if (
                tracked_job.state.is_terminal()
                and tracked_job.post_success
                and not tracked_job.result_fetched
            ):
                if tracked_job.get_attempts >= self.MAX_GET_ATTEMPTS:
                    self._mark_get_exhausted(job_id, tracked_job)
                    continue
                log_path = self._get_log_path(tracked_job)
                if log_path:
                    tracked_job.get_attempts += 1
                    jobs_to_fetch.append((tracked_job, log_path))

        return jobs_to_fetch

    def _get_slurm_jobs(self) -> dict[str, SlurmJob] | None:
        """Get currently queued/running jobs from squeue."""
        existing_paths: dict[str, tuple] = {
            job_id: (job.stdout_path, job.stderr_path)
            for job_id, job in self.state.jobs.items()
            if job.stdout_path
        }

        raw_jobs = self._slurm_client.get_running_jobs(existing_paths)
        self._sync_slurm_stats()

        if raw_jobs is None:
            return None

        jobs: dict[str, SlurmJob] = {}
        for job_id, info in raw_jobs.items():
            jobs[job_id] = SlurmJob(
                job_id=job_id,
                name=info["name"],
                user=info["user"],
                partition=info["partition"],
                state=JobState.from_str(info["state"]),
                stdout_path=info["stdout_path"],
                stderr_path=info["stderr_path"],
            )

        return jobs

    def _sync_slurm_stats(self) -> None:
        """Sync stats from SlurmClient to MonitorState."""
        slurm_stats = self._slurm_client.stats
        self.state.squeue_calls = slurm_stats.squeue_calls
        self.state.scontrol_calls = slurm_stats.scontrol_calls
        self.state.sacct_calls = slurm_stats.sacct_calls
        self.state.squeue_failures = slurm_stats.squeue_failures
        self.state.scontrol_failures = slurm_stats.scontrol_failures
        self.state.sacct_failures = slurm_stats.sacct_failures

    def _get_job_output_paths_batch(self, job_ids: list[str]) -> dict[str, tuple]:
        """Get stdout and stderr paths for multiple jobs."""
        result = self._slurm_client.get_job_output_paths(job_ids)
        self._sync_slurm_stats()
        return result

    def _matches_filters(self, job: SlurmJob) -> bool:
        """Check if job matches configured filters."""
        if self.job_pattern and not self.job_pattern.match(job.name):
            return False
        return True

    def _expand_slurm_patterns(self, path: str, job: SlurmJob) -> str:
        """Expand SLURM filename pattern placeholders in a path."""

        def on_unexpanded(result: str) -> None:
            self.state.path_errors_unexpanded += 1

        return expand_slurm_patterns(
            path=path,
            job_id=job.job_id,
            job_name=job.name,
            job_user=job.user,
            on_unexpanded=on_unexpanded,
        )

    def _get_log_path(self, job: SlurmJob) -> str | None:
        """Get the log file path for a job from SLURM StdOut."""
        if job.stdout_path:
            return self._expand_slurm_patterns(job.stdout_path, job)
        logger.debug(f"[{job.job_id}] No StdOut path available")
        return None

    def _submit_log(self, job: SlurmJob, log_path: str) -> None:
        """Submit a log file to the attribution service."""
        submit_log(job, log_path, self.state, self._attrsvc_client)

    def _fetch_results(self, job: SlurmJob, log_path: str) -> None:
        """Fetch attribution results for a completed job."""
        fetch_results(job, log_path, self.state, self._attrsvc_client)

    def _cleanup_old_jobs(self) -> None:
        """Remove old completed jobs from state to prevent memory growth."""
        with self._state_lock:
            max_completed = self.MAX_COMPLETED_JOBS

            done_jobs = [
                (job_id, job)
                for job_id, job in self.state.jobs.items()
                if job.state.is_terminal()
                and (job.result_fetched or (job.log_submitted and not job.post_success))
            ]

            if len(done_jobs) > max_completed:
                to_remove = len(done_jobs) - max_completed
                for job_id, _ in done_jobs[:to_remove]:
                    del self.state.jobs[job_id]
                logger.debug(f"Cleaned up {to_remove} old completed jobs from state")
