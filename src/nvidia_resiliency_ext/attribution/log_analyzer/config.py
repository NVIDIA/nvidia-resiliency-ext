#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Core configuration and constants for log analysis.

This module contains library-level constants and error codes used by
the log analyzer components. Service-specific configuration (HTTP settings,
pydantic Settings class) should remain in the service layer.

Constants overview:
- TTL_* : Time-to-live values for job cleanup
- POLL_INTERVAL_SECONDS: How often splitlog tracker polls for changes
- DEFAULT_COMPUTE_TIMEOUT_SECONDS: Timeout for LLM analysis
- MAX_JOBS: Maximum number of tracked jobs
- MIN_FILE_SIZE_KB: Minimum file size for analysis
"""

from enum import Enum

# TTL constants (see spec Section 3.2)
TTL_PENDING_SECONDS = 7 * 24 * 60 * 60  # 1 week - pending job expiry
TTL_TERMINATED_SECONDS = 60 * 60  # 1 hour - terminated job expiry (after GET)
TTL_MAX_JOB_AGE_SECONDS = 6 * 30 * 24 * 60 * 60  # 6 months - non-terminated safety net

# Poll/tracking constants
POLL_INTERVAL_SECONDS = 5 * 60  # 5 minutes - background poll interval
DEFAULT_COMPUTE_TIMEOUT_SECONDS = 300.0  # 5 minutes - compute function timeout

# Limits (see spec Section 3.2)
MAX_JOBS = 100_000  # Maximum tracked jobs
MIN_FILE_SIZE_KB = 4  # Minimum file size (KB) for classification

# Result/response keys (serialized shape of AnalysisResult, SplitlogAnalysisResult, SubmitResult)
# Used by library and HTTP layer for consistent parsing. Job mode values are JobMode enum.
RESP_MODE = "mode"
RESP_RESULT = "result"
RESP_STATUS = "status"
RESP_LOG_FILE = "log_file"
RESP_WL_RESTART = "wl_restart"
RESP_WL_RESTART_COUNT = "wl_restart_count"
RESP_SCHED_RESTARTS = "sched_restarts"
RESP_LOGS_DIR = "logs_dir"
RESP_FILES_ANALYZED = "files_analyzed"
# Inner result dict (RESP_RESULT value from analysis pipeline)
RESP_MODULE = "module"
RESP_STATE = "state"
RESP_ERROR = "error"
RESP_RESULT_ID = "result_id"
# Inner result RESP_STATE values
STATE_TIMEOUT = "timeout"

# Stats / job detail keys (get_stats, get_jobs_detail, get_all_jobs response shape)
STATS_JOBS = "jobs"
STATS_JOBS_TERMINATED = "jobs_terminated"
STATS_JOBS_CLEANED = "jobs_cleaned"
STATS_FILES = "files"
STATS_FILES_TRACKED = "tracked"
STATS_FILES_ANALYZED = "analyzed"  # key under STATS_FILES (count of analyzed files)
STATS_SCHED_RESTARTS = "sched_restarts"
STATS_JOB_ID = "job_id"
STATS_LOG_PATH = "log_path"
STATS_USER = "user"
STATS_TERMINATED = "terminated"
STATS_LOG_FILES = "log_files"


class ErrorCode(str, Enum):
    """Error codes for log analysis operations.

    See spec Section 7 for HTTP status mapping when used in HTTP context.
    """

    # Path validation errors (400 in HTTP)
    INVALID_PATH = "invalid_path"  # Path not absolute, null bytes, etc.
    NOT_REGULAR = "not_regular"  # Not a regular file (directory, device, etc.)
    EMPTY_FILE = "empty_file"  # File is empty (GET only)

    # Permission errors (403 in HTTP)
    OUTSIDE_ROOT = "outside_root"  # Path (or symlink target) outside allowed root
    NOT_READABLE = "not_readable"  # File permission denied
    LOGS_DIR_NOT_READABLE = "logs_dir_not_readable"  # LOGS_DIR permission denied

    # Not found (404 in HTTP)
    NOT_FOUND = "not_found"  # File doesn't exist

    # Server errors (5xx in HTTP)
    JOB_LIMIT_REACHED = "job_limit_reached"  # MAX_JOBS exceeded (503)
    INTERNAL_ERROR = "internal_error"  # Unexpected server error (500)
