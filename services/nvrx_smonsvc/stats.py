#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Stats calculation and formatting for nvrx_smonsvc."""

import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .attrsvc_client import AttrsvcClient
    from .models import MonitorState


def get_stats_dict(
    state: "MonitorState",
    lock: threading.Lock,
) -> dict:
    """
    Build stats dictionary for HTTP endpoint.

    "jobs" is a snapshot of the current in-memory job set. All other sections
    (job_totals, slurm, path_errors, http_errors) are cumulative since process start.

    Args:
        state: MonitorState with job and counter data
        lock: Lock for thread-safe state access

    Returns:
        Stats dictionary with jobs, job_totals, slurm, path_errors, http_errors
    """
    with lock:
        jobs = state.jobs
        running = sum(1 for j in jobs.values() if j.state.is_running())
        terminal = sum(1 for j in jobs.values() if j.state.is_terminal())
        submitted = sum(1 for j in jobs.values() if j.log_submitted)
        post_success = sum(1 for j in jobs.values() if j.post_success)
        fetched = sum(1 for j in jobs.values() if j.result_fetched)
        has_path = sum(1 for j in jobs.values() if j.stdout_path)
        total = len(jobs)

        return {
            "jobs": {
                "total": total,
                "running": running,
                "terminal": terminal,
                "has_output_path": has_path,
                "logs_submitted": submitted,
                "logs_post_success": post_success,
                "results_fetched": fetched,
            },
            "job_totals": {
                "total": state.jobs_seen,
                "has_output_path": state.with_output_path,
                "logs_submitted": state.logs_submitted,
                "logs_post_success": state.post_success,
                "results_fetched": state.results_fetched,
            },
            "slurm": {
                "squeue_calls": state.squeue_calls,
                "scontrol_calls": state.scontrol_calls,
                "sacct_calls": state.sacct_calls,
                "squeue_failures": state.squeue_failures,
                "scontrol_failures": state.scontrol_failures,
                "sacct_failures": state.sacct_failures,
            },
            "path_errors": {
                "permission_denied": state.path_errors_permission,
                "not_found": state.path_errors_not_found,
                "file_empty": state.path_errors_empty,
                "unexpanded_patterns": state.path_errors_unexpanded,
                "other": state.path_errors_other,
            },
            "http_errors": {
                "rate_limited": state.http_rate_limited,
            },
        }


def get_jobs_list(
    state: "MonitorState",
    lock: threading.Lock,
) -> list[dict]:
    """
    Build jobs list for HTTP endpoint.

    Args:
        state: MonitorState with jobs
        lock: Lock for thread-safe state access

    Returns:
        List of job dictionaries
    """
    with lock:
        jobs_list = []
        for _job_id, job in state.jobs.items():
            job_dict = {
                "job_id": job.job_id,
                "name": job.name,
                "user": job.user,
                "partition": job.partition,
                "state": job.state.value,
                "stdout_path": job.stdout_path,
                "log_submitted": job.log_submitted,
                "post_success": job.post_success,
                "result_fetched": job.result_fetched,
            }
            if job.last_state:
                job_dict["last_state"] = job.last_state.value
            jobs_list.append(job_dict)
        return jobs_list


def get_health_status(
    state: "MonitorState",
    lock: threading.Lock,
    attrsvc_client: Optional["AttrsvcClient"],
) -> tuple:
    """
    Get health status for HTTP endpoint.

    Args:
        state: MonitorState with SLURM stats
        lock: Lock for thread-safe state access
        attrsvc_client: Client to check attrsvc connectivity (may be None)

    Returns:
        (is_healthy: bool, details: dict)
    """
    issues = []

    # Check SLURM connectivity
    with lock:
        squeue_calls = state.squeue_calls
        squeue_failures = state.squeue_failures

    slurm_healthy = True
    if squeue_calls > 0 and squeue_failures > 0:
        failure_rate = squeue_failures / squeue_calls
        if failure_rate >= 0.5:
            slurm_healthy = False
            issues.append(f"squeue_failure_rate={failure_rate:.1%}")

    # Check attrsvc connectivity (cached in client to avoid blocking HTTP calls)
    attrsvc_healthy = attrsvc_client.check_health_cached() if attrsvc_client else False

    if not attrsvc_healthy:
        issues.append("attrsvc_unreachable")

    is_healthy = slurm_healthy and attrsvc_healthy

    details = {
        "squeue_calls": squeue_calls,
        "squeue_failures": squeue_failures,
        "slurm_healthy": slurm_healthy,
        "attrsvc_healthy": attrsvc_healthy,
    }
    if issues:
        details["issues"] = issues

    return is_healthy, details


def format_stats_summary(stats: dict) -> str:
    """
    Format stats as a one-line summary for logging.

    Args:
        stats: Stats dictionary from get_stats_dict()

    Returns:
        One-line summary string
    """
    jobs = stats["jobs"]
    job_totals = stats["job_totals"]
    slurm = stats["slurm"]

    return (
        f"Jobs: {jobs['total']} tracked ({jobs['running']} running, {jobs['terminal']} terminal), "
        f"Totals: {job_totals['total']} seen, "
        f"{job_totals['has_output_path']} with path, "
        f"{job_totals['logs_submitted']} submitted, "
        f"{job_totals['logs_post_success']} post success, "
        f"{job_totals['results_fetched']} fetched, "
        f"SLURM: {slurm['squeue_calls']} squeue, "
        f"{slurm['scontrol_calls']} scontrol, "
        f"{slurm['sacct_calls']} sacct"
    )
