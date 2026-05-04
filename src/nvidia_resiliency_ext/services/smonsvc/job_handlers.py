# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job submission and result handling for nvidia_resiliency_ext.services.smonsvc."""

import logging
from typing import TYPE_CHECKING

from nvidia_resiliency_ext.attribution import (
    RESP_FILES_ANALYZED,
    RESP_LOGS_DIR,
    RESP_MODE,
    RESP_MODULE,
    RESP_RESULT,
    RESP_SCHED_RESTARTS,
    JobMode,
    parse_attrsvc_response,
)
from nvidia_resiliency_ext.attribution.orchestration.types import RECOMMENDATION_TIMEOUT

if TYPE_CHECKING:
    from .attrsvc_client import AttrsvcClient
    from .models import MonitorState, SlurmJob

logger = logging.getLogger(__name__)


def categorize_path_error(state: "MonitorState", error_msg: str) -> None:
    """
    Categorize a path error and increment the appropriate counter.

    Args:
        state: MonitorState to update counters
        error_msg: Error message to categorize
    """
    error_lower = error_msg.lower()
    if "permission denied" in error_lower:
        state.path_errors_permission += 1
    elif "not found" in error_lower or "path not found" in error_lower:
        state.path_errors_not_found += 1
    elif "empty" in error_lower or "file is empty" in error_lower:
        state.path_errors_empty += 1
    else:
        state.path_errors_other += 1


def submit_log(
    job: "SlurmJob",
    log_path: str,
    state: "MonitorState",
    attrsvc_client: "AttrsvcClient",
) -> None:
    """
    Submit a log file to the attribution service.

    Args:
        job: The SLURM job to submit log for
        log_path: Path to the log file
        state: MonitorState to update counters
        attrsvc_client: Client for attrsvc HTTP requests
    """

    def on_success(response):
        job.log_submitted = True
        state.logs_submitted += 1
        try:
            result = response.json()
        except Exception as e:
            logger.warning(f"[{job.job_id}] POST 2xx but JSON parse failed: {e}")
            return
        mode = result.get(RESP_MODE, JobMode.SINGLE.value)
        if mode == JobMode.SPLITLOG.value:
            logs_dir = result.get(RESP_LOGS_DIR, "")
            sched_restarts = result.get(RESP_SCHED_RESTARTS, 0)
            files_analyzed = result.get(RESP_FILES_ANALYZED, 0)
            logger.info(
                f"[{job.job_id}] POST submitted (splitlog mode): {log_path} "
                f"(logs_dir={logs_dir}, sched_restarts={sched_restarts}, files_analyzed={files_analyzed})"
            )
        else:
            logger.info(f"[{job.job_id}] POST submitted: {log_path}")
        job.post_success = True
        state.post_success += 1

    def on_client_error(error_msg: str):
        job.log_submitted = True
        state.logs_submitted += 1
        categorize_path_error(state, error_msg)

    def on_404():
        logger.debug(f"[{job.job_id}] POST 404 (file not found): {log_path}")
        job.log_submitted = True  # Don't retry - attrsvc received the path
        state.logs_submitted += 1
        state.path_errors_not_found += 1

    attrsvc_client.request_with_retry(
        method="POST",
        job_id=job.job_id,
        log_path=log_path,
        on_success=on_success,
        on_client_error=on_client_error,
        on_404=on_404,
        user=job.user,
    )


def fetch_results(
    job: "SlurmJob",
    log_path: str,
    state: "MonitorState",
    attrsvc_client: "AttrsvcClient",
) -> None:
    """
    Fetch attribution results for a completed job.

    Args:
        job: The SLURM job to fetch results for
        log_path: Path to the log file
        state: MonitorState to update counters
        attrsvc_client: Client for attrsvc HTTP requests
    """

    def on_success(response):
        try:
            result = response.json()
        except Exception as e:
            logger.warning(f"[{job.job_id}] GET JSON decode error: {e}")
            job.result_fetched = True
            state.results_fetched += 1
            return
        log_attribution_result(job, log_path, result)
        job.result_fetched = True
        state.results_fetched += 1

    def on_client_error(error_msg: str):
        job.result_fetched = True
        state.results_fetched += 1
        categorize_path_error(state, error_msg)

    attrsvc_client.request_with_retry(
        method="GET",
        job_id=job.job_id,
        log_path=log_path,
        on_success=on_success,
        on_client_error=on_client_error,
    )


def log_attribution_result(job: "SlurmJob", log_path: str, response: dict) -> None:
    """
    Log a summary of the attribution result to stdout.

    Args:
        job: The SLURM job
        log_path: Path to the log file
        response: Attribution result response dict (may be single-file or splitlog mode)
    """
    try:
        logger.debug(f"[{job.job_id}] Raw response: {response}")

        inner = response.get(RESP_RESULT, response)

        if not inner or not inner.get(RESP_MODULE):
            logger.warning(
                f"[{job.job_id}] Attribution result is empty or missing module: {response}"
            )
            return

        parsed = parse_attrsvc_response(response, log_path=log_path)
        action = parsed.recommendation.action

        if action == RECOMMENDATION_TIMEOUT:
            timeout_reason = parsed.recommendation_reason or "Attribution analysis timed out"
            logger.warning(f"[{job.job_id}] Attribution timeout: {timeout_reason}")
            print(parsed.format_summary(prefix=f"[{job.job_id}] "), flush=True)
            return

        print(parsed.format_summary(prefix=f"[{job.job_id}] "), flush=True)

    except Exception as e:
        logger.warning(f"[{job.job_id}] Could not parse attribution result: {e}")
