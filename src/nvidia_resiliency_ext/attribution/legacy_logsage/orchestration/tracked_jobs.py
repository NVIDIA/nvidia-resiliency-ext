# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracked job storage, splitlog polling, and detection counters for log analysis.

:mod:`~nvidia_resiliency_ext.attribution.legacy_logsage.analyzer.engine.Analyzer` composes this; attribution stays thin.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from nvidia_resiliency_ext.attribution.orchestration.config import (
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
from nvidia_resiliency_ext.attribution.orchestration.job import Job, JobMode
from nvidia_resiliency_ext.attribution.orchestration.types import (
    LogAnalyzerError,
    LogAnalyzerSubmitResult,
)
from nvidia_resiliency_ext.attribution.path_utils import path_is_under_allowed_root

from .slurm_parser import read_and_parse_slurm_output
from .splitlog import SplitlogTracker

logger = logging.getLogger(__name__)

# Fire-and-forget: (log_path, user, job_id)
FireAndForgetAnalyze = Callable[[str, str, Optional[str]], None]


class TrackedJobs:
    """Owns ``Job`` map, splitlog tracker, detection counters, and splitlog callback wiring."""

    def __init__(
        self,
        *,
        track_submission: Callable[[str], Awaitable[None]],
        splitlog_tracker: SplitlogTracker | None = None,
        allowed_root: Optional[str] = None,
    ):
        self._track_submission = track_submission
        self._allowed_root = os.path.realpath(allowed_root) if allowed_root else None
        self._splitlog_tracker = splitlog_tracker or SplitlogTracker(
            allowed_root=self._allowed_root
        )
        if splitlog_tracker is not None and self._allowed_root is not None:
            splitlog_tracker.set_allowed_root(self._allowed_root)
        self._jobs: Dict[str, Job] = {}
        self._jobs_lock = threading.Lock()

        self._total_splitlog: int = 0
        self._total_single: int = 0
        self._deferred_splitlog: int = 0
        self._deferred_single: int = 0
        self._total_permission_errors: int = 0
        self._logs_dir_permission_errors: int = 0
        self._file_permission_errors: int = 0
        self._pending_expired: int = 0

    @property
    def splitlog_tracker(self) -> SplitlogTracker:
        return self._splitlog_tracker

    def register_callbacks(self, fire_and_forget_analyze: FireAndForgetAnalyze) -> None:
        """Wire splitlog tracker and start background polling."""
        self._splitlog_tracker.set_analyze_callback(fire_and_forget_analyze)
        self._splitlog_tracker.set_pending_check_callback(self.check_pending_jobs)
        self._splitlog_tracker.set_get_splitlog_jobs_callback(self.get_splitlog_jobs)
        self._splitlog_tracker.set_cleanup_job_callback(self.cleanup_job)
        self._splitlog_tracker.start_polling()

    def shutdown(self) -> None:
        self._splitlog_tracker.stop_polling()

    def get_job(self, path: str) -> Optional[Job]:
        with self._jobs_lock:
            return self._jobs.get(path)

    def pending_job_count(self) -> int:
        with self._jobs_lock:
            return sum(1 for j in self._jobs.values() if j.is_pending())

    def record_deferred_single_demotion(self) -> None:
        self._total_single += 1
        self._deferred_single += 1

    def record_file_permission_error(self) -> None:
        self._total_permission_errors += 1
        self._file_permission_errors += 1

    def _record_logs_dir_policy_error(self, error: LogAnalyzerError) -> None:
        if error.error_code in {ErrorCode.LOGS_DIR_NOT_READABLE, ErrorCode.OUTSIDE_ROOT}:
            self._total_permission_errors += 1
            self._logs_dir_permission_errors += 1

    def _resolve_splitlog_dir(
        self,
        logs_dir: str,
    ) -> tuple[Optional[str], Optional[LogAnalyzerError]]:
        """Resolve and validate a parsed ``LOGS_DIR`` before splitlog promotion."""
        try:
            real_logs_dir = os.path.realpath(logs_dir)
        except (OSError, ValueError):
            return None, LogAnalyzerError(
                error_code=ErrorCode.INVALID_PATH,
                message="LOGS_DIR path is invalid",
            )

        if self._allowed_root is not None and not path_is_under_allowed_root(
            real_logs_dir,
            self._allowed_root,
        ):
            logger.warning(
                "LOGS_DIR outside allowed_root rejected: logs_dir=%r allowed_root=%r",
                logs_dir,
                self._allowed_root,
            )
            return None, LogAnalyzerError(
                error_code=ErrorCode.OUTSIDE_ROOT,
                message="LOGS_DIR outside allowed root is not permitted",
            )

        if not os.path.isdir(real_logs_dir):
            logger.debug("LOGS_DIR not a directory: %s", logs_dir)
            return None, None

        if not os.access(real_logs_dir, os.R_OK | os.X_OK):
            logger.debug("LOGS_DIR not readable/searchable: %s", real_logs_dir)
            return None, LogAnalyzerError(
                error_code=ErrorCode.LOGS_DIR_NOT_READABLE,
                message=f"LOGS_DIR not readable: {real_logs_dir}",
            )

        return real_logs_dir, None

    def handle_existing_job(self, job: Job, validated: str) -> LogAnalyzerSubmitResult:
        if job.is_splitlog():
            self._splitlog_tracker.poll_now()
            return LogAnalyzerSubmitResult(
                submitted=True,
                normalized_path=validated,
                mode=JobMode.SPLITLOG.value,
                logs_dir=job.logs_dir,
                sched_restarts=job.sched_restarts,
                files_analyzed=job.files_complete(),
            )
        if job.is_pending():
            self.check_pending_jobs()
            if job.is_splitlog():
                return LogAnalyzerSubmitResult(
                    submitted=True,
                    normalized_path=validated,
                    mode=JobMode.SPLITLOG.value,
                    logs_dir=job.logs_dir,
                    sched_restarts=job.sched_restarts,
                    files_analyzed=job.files_complete(),
                )
            return LogAnalyzerSubmitResult(
                submitted=True,
                normalized_path=validated,
                mode=JobMode.PENDING.value,
            )
        return LogAnalyzerSubmitResult(
            submitted=True,
            normalized_path=validated,
            mode=JobMode.SINGLE.value,
        )

    async def create_new_job(
        self,
        validated: str,
        user: str,
        job_id: Optional[str],
        *,
        detect_splitlog: bool = True,
    ) -> LogAnalyzerSubmitResult | LogAnalyzerError:
        if job_id and detect_splitlog:
            logger.debug("create_new_job: job_id=%s, checking for splitlog mode", job_id)
            info = read_and_parse_slurm_output(validated)
            logger.debug(
                "create_new_job: slurm parse - logs_dir=%s, cycle_count=%s",
                info.logs_dir if info else None,
                info.cycle_count if info else None,
            )
            if info and info.logs_dir:
                logs_dir, error = self._resolve_splitlog_dir(info.logs_dir)
                if error is not None:
                    self._record_logs_dir_policy_error(error)
                    return error
                if logs_dir is not None:
                    self._total_splitlog += 1
                    job = Job(
                        path=validated,
                        user=user,
                        mode=JobMode.SPLITLOG,
                        job_id=job_id,
                        logs_dir=logs_dir,
                    )
                    with self._jobs_lock:
                        self._jobs[validated] = job
                    logger.debug(
                        "Created SPLITLOG job: path=%s, job_id=%s, logs_dir=%s",
                        validated,
                        job_id,
                        logs_dir,
                    )
                    self._splitlog_tracker.initialize_job(job)
                    return LogAnalyzerSubmitResult(
                        submitted=True,
                        normalized_path=validated,
                        mode=JobMode.SPLITLOG.value,
                        logs_dir=logs_dir,
                        sched_restarts=job.sched_restarts,
                        files_analyzed=job.files_complete(),
                    )
            else:
                logger.debug("create_new_job: no logs_dir, deferring to PENDING")

            job = Job(
                path=validated,
                user=user,
                mode=JobMode.PENDING,
                job_id=job_id,
            )
            with self._jobs_lock:
                self._jobs[validated] = job
            logger.debug("Created PENDING job: path=%s, job_id=%s", validated, job_id)
            await self._track_submission(validated)
            return LogAnalyzerSubmitResult(
                submitted=True,
                normalized_path=validated,
                mode=JobMode.PENDING.value,
            )

        self._total_single += 1
        job = Job(
            path=validated,
            user=user,
            mode=JobMode.SINGLE,
            job_id=job_id,
        )
        with self._jobs_lock:
            self._jobs[validated] = job
        logger.debug(
            "Created SINGLE job: path=%s, job_id=%s, splitlog_detection=%s",
            validated,
            job_id,
            detect_splitlog,
        )
        await self._track_submission(validated)
        return LogAnalyzerSubmitResult(
            submitted=True,
            normalized_path=validated,
            mode=JobMode.SINGLE.value,
        )

    def check_pending_jobs(self) -> None:
        with self._jobs_lock:
            pending_jobs = [j for j in self._jobs.values() if j.is_pending()]
        if not pending_jobs:
            logger.debug("check_pending_jobs: no pending jobs")
            return

        logger.debug("check_pending_jobs: checking %d pending jobs", len(pending_jobs))
        now = time.monotonic()
        expired_count = 0
        promoted_count = 0

        for job in pending_jobs:
            age_seconds = now - job.created_at
            if age_seconds >= TTL_PENDING_SECONDS:
                logger.debug(
                    "check_pending_jobs: job %s expired (age=%.0fs >= TTL=%ss)",
                    job.job_id,
                    age_seconds,
                    TTL_PENDING_SECONDS,
                )
                with self._jobs_lock:
                    # Snapshot TOCTOU: same path may have been re-submitted; only remove if this
                    # instance is still the registered job.
                    if self._jobs.get(job.path) is job:
                        del self._jobs[job.path]
                        expired_count += 1
                continue

            info = read_and_parse_slurm_output(job.path)
            logger.debug(
                "check_pending_jobs: job %s slurm parse - logs_dir=%s",
                job.job_id,
                info.logs_dir if info else None,
            )
            if info and info.logs_dir:
                logs_dir, error = self._resolve_splitlog_dir(info.logs_dir)
                if error is not None:
                    self._record_logs_dir_policy_error(error)
                    with self._jobs_lock:
                        if self._jobs.get(job.path) is job:
                            job.demote_to_single()
                            self._total_single += 1
                            self._deferred_single += 1
                    logger.warning(
                        "check_pending_jobs: job %s demoted to SINGLE after rejecting "
                        "LOGS_DIR=%r: %s",
                        job.job_id,
                        info.logs_dir,
                        error.message,
                    )
                    continue
                logger.debug(
                    "check_pending_jobs: job %s logs_dir - valid=%s",
                    job.job_id,
                    logs_dir is not None,
                )
                if logs_dir is not None:
                    with self._jobs_lock:
                        if self._jobs.get(job.path) is not job:
                            continue
                        job.promote_to_splitlog(logs_dir)
                    self._splitlog_tracker.initialize_job(job)
                    self._total_splitlog += 1
                    self._deferred_splitlog += 1
                    promoted_count += 1
                    logger.debug(
                        "check_pending_jobs: job %s promoted to SPLITLOG (logs_dir=%s)",
                        job.job_id,
                        logs_dir,
                    )

        if expired_count > 0:
            self._pending_expired += expired_count

        logger.debug(
            "check_pending_jobs: complete - promoted=%s, expired=%s",
            promoted_count,
            expired_count,
        )

    def get_splitlog_jobs(self) -> List[Job]:
        with self._jobs_lock:
            return [j for j in self._jobs.values() if j.is_splitlog()]

    def cleanup_job(self, path: str) -> None:
        with self._jobs_lock:
            if path in self._jobs:
                del self._jobs[path]

    def detection_stats(self, pending_count: int) -> Dict[str, Any]:
        return {
            "total_splitlog": self._total_splitlog,
            "total_single": self._total_single,
            JobMode.PENDING.value: pending_count,
            "jobs_expired": self._pending_expired,
        }

    def deferred_stats(self) -> Dict[str, int]:
        return {
            "total_splitlog": self._deferred_splitlog,
            "total_single": self._deferred_single,
        }

    def permission_error_stats(self) -> Dict[str, int]:
        return {
            "total": self._total_permission_errors,
            "logs_dir": self._logs_dir_permission_errors,
            "file": self._file_permission_errors,
        }

    def get_all_jobs_payload(self) -> Dict[str, Any]:
        pending_jobs: List[Dict[str, Any]] = []
        single_jobs: List[Dict[str, Any]] = []
        splitlog_jobs: List[Dict[str, Any]] = []

        with self._jobs_lock:
            jobs_snapshot = list(self._jobs.values())

        for job in jobs_snapshot:
            job_info = {
                STATS_JOB_ID: job.job_id or "unknown",
                STATS_LOG_PATH: job.path,
                STATS_USER: job.user,
                RESP_MODE: job.mode.value,
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
