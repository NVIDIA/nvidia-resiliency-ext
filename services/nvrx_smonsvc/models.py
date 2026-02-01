#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Data models for nvrx_smonsvc."""

from dataclasses import dataclass, field
from enum import Enum


class JobState(Enum):
    """SLURM job states."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    NODE_FAIL = "NODE_FAIL"
    PREEMPTED = "PREEMPTED"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    SUSPENDED = "SUSPENDED"
    UNKNOWN = "UNKNOWN"
    # Monitor-specific state: job left squeue but exact final state is unknown
    FINISHED = "FINISHED"

    @classmethod
    def from_str(cls, state_str: str) -> "JobState":
        """Parse SLURM state string to JobState."""
        # Handle state codes like "R", "PD", "CG", etc.
        state_map = {
            "R": cls.RUNNING,
            "PD": cls.PENDING,
            "CG": cls.COMPLETING,
            "CD": cls.COMPLETED,
            "F": cls.FAILED,
            "CA": cls.CANCELLED,
            "TO": cls.TIMEOUT,
            "NF": cls.NODE_FAIL,
            "PR": cls.PREEMPTED,
            "OOM": cls.OUT_OF_MEMORY,
            "S": cls.SUSPENDED,
        }
        # Try short code first
        if state_str in state_map:
            return state_map[state_str]
        # Try full name
        try:
            return cls(state_str.upper())
        except ValueError:
            return cls.UNKNOWN

    def is_terminal(self) -> bool:
        """Check if this is a terminal state (job has finished)."""
        return self in {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.TIMEOUT,
            JobState.NODE_FAIL,
            JobState.PREEMPTED,
            JobState.OUT_OF_MEMORY,
            JobState.FINISHED,  # Monitor-specific: left squeue, exact state unknown
        }

    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self in {JobState.RUNNING, JobState.COMPLETING}


@dataclass
class SlurmJob:
    """Represents a SLURM job."""

    job_id: str
    name: str
    user: str
    partition: str
    state: JobState
    stdout_path: str = ""
    stderr_path: str = ""

    # Tracking fields (must be preserved on state transition; see copy_tracking_fields)
    log_submitted: bool = False  # POST was attempted (success or client error)
    post_success: bool = False  # POST succeeded (200) - only fetch results if True
    result_fetched: bool = False
    path_fetch_attempted: bool = False  # True after attempting to fetch stdout path
    last_state: JobState | None = None
    get_attempts: int = 0  # Number of GET attempts (for giving up after max retries)


# When adding a new tracking field to SlurmJob, add its name here so it is preserved
# on state transitions (e.g. RUNNING -> COMPLETING). Prevents "forgot to copy" bugs.
SLURM_JOB_TRACKING_FIELDS = (
    "log_submitted",
    "post_success",
    "result_fetched",
    "path_fetch_attempted",
    "get_attempts",
)


def copy_tracking_fields(prev: SlurmJob, into: SlurmJob) -> None:
    """Copy tracking fields from prev to into. Use when replacing a job on state change."""
    for attr in SLURM_JOB_TRACKING_FIELDS:
        setattr(into, attr, getattr(prev, attr))
    into.last_state = prev.state  # state we're transitioning from


@dataclass
class MonitorState:
    """Tracks the state of monitored jobs.

    All counters are cumulative since process start (never reset).
    """

    jobs: dict[str, SlurmJob] = field(default_factory=dict)
    # Invocation counters for SLURM commands (to track controller pressure)
    squeue_calls: int = 0
    scontrol_calls: int = 0
    sacct_calls: int = 0
    # Failure counters for SLURM commands
    squeue_failures: int = 0
    scontrol_failures: int = 0
    sacct_failures: int = 0
    # Job counters (all cumulative since process start)
    jobs_seen: int = 0  # Unique jobs ever seen
    with_output_path: int = 0  # Jobs that had output path
    logs_submitted: int = 0  # POST attempts (success or client error)
    post_success: int = 0  # Successful POSTs
    results_fetched: int = 0  # Successful GETs
    # Log path error counters (from attrsvc responses and local checks)
    path_errors_permission: int = 0  # Permission denied
    path_errors_not_found: int = 0  # Path not found
    path_errors_empty: int = 0  # File is empty
    path_errors_unexpanded: int = 0  # Unexpanded SLURM patterns
    path_errors_other: int = 0  # Other validation errors
    # HTTP error counters
    http_rate_limited: int = 0  # 429 Too Many Requests
