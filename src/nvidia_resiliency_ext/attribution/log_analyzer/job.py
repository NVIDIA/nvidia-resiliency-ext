#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Unified job model for attribution service.

This module defines the Job and FileInfo dataclasses used to track submitted
log files through the analysis pipeline.

Job modes:
- PENDING: Job submitted but LOGS_DIR not yet found (deferred mode classification)
- SINGLE: Analyze the slurm output file directly (no separate log directory)
- SPLITLOG: Analyze per-restart log files in LOGS_DIR

Terminology:
- sched_restarts: Count of scheduler restarts, detected by << START PATHS >> markers
- wl_restarts: Count of workload restarts within a file, detected by Cycle: N markers
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class JobMode(Enum):
    """Job mode enum (see spec Section 3.4)."""

    PENDING = "pending"
    SINGLE = "single"
    SPLITLOG = "splitlog"


@dataclass(slots=True)
class FileInfo:
    """Tracks a single log file and its analysis results.

    See spec Section 5.2 for field definitions.
    Uses slots=True for memory efficiency with many instances.
    """

    log_file: str  # Absolute path to log file (basename used as dict key)
    analysis_triggered: bool = False  # Analysis started
    analysis_complete: bool = False  # Analysis finished
    analyzed_at: Optional[float] = None  # time.time() when analysis completed
    wl_restarts: int = 0  # Count of Cycle: N markers within this file
    wl_restart_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class Job:
    """
    Unified job model for all modes (single-file, splitlog, pending).

    Every POST creates a Job. The mode field indicates the job type:
    - PENDING: Awaiting mode classification (LOGS_DIR not found yet)
    - SINGLE: Single-file mode (no LOGS_DIR, analyze slurm output directly)
    - SPLITLOG: Split logging mode (LOGS_DIR found, analyze per-cycle log files)

    See spec Section 5 for full data model.
    Uses slots=True for memory efficiency (MAX_JOBS can be 100,000+).
    """

    # Common fields (all jobs)
    path: str  # slurm_output_path (client's original path, used as job key)
    user: str  # Job owner (for dataflow posting)
    mode: JobMode = JobMode.PENDING  # Job mode
    created_at: float = field(default_factory=time.monotonic)

    # Optional job_id (provided in POST for splitlog detection)
    job_id: Optional[str] = None

    # Splitlog-specific fields (populated when mode=SPLITLOG)
    logs_dir: Optional[str] = None
    sched_restarts: int = 0  # Count of << START PATHS >> markers (scheduler restarts)
    known_log_files: List[str] = field(default_factory=list)
    file_info: Dict[str, FileInfo] = field(default_factory=dict)  # Keyed by filename (basename)

    # Termination tracking (for cleanup)
    terminated: bool = False
    terminated_at: Optional[float] = None  # time.monotonic() when terminated

    # Polling state (for splitlog mode)
    last_poll_at: float = 0.0

    def files_triggered(self) -> int:
        """Count of files where analysis was triggered."""
        return sum(1 for f in self.file_info.values() if f.analysis_triggered)

    def files_complete(self) -> int:
        """Count of files where analysis is complete."""
        return sum(1 for f in self.file_info.values() if f.analysis_complete)

    def is_splitlog(self) -> bool:
        """Check if this is a splitlog mode job."""
        return self.mode == JobMode.SPLITLOG

    def is_pending(self) -> bool:
        """Check if this job is pending mode classification."""
        return self.mode == JobMode.PENDING

    def is_single(self) -> bool:
        """Check if this is a single-file mode job."""
        return self.mode == JobMode.SINGLE

    def promote_to_splitlog(self, logs_dir: str) -> None:
        """Promote a pending job to splitlog mode."""
        self.mode = JobMode.SPLITLOG
        self.logs_dir = logs_dir

    def demote_to_single(self) -> None:
        """Demote a pending job to single-file mode (fallback)."""
        self.mode = JobMode.SINGLE

    def mark_terminated(self) -> None:
        """Mark job as terminated and record timestamp."""
        if not self.terminated:
            self.terminated = True
            self.terminated_at = time.monotonic()
