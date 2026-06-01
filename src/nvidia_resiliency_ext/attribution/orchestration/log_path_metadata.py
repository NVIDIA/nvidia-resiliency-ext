# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regexes and helpers for job id / cycle id derived from log file paths.

Used by :mod:`~nvidia_resiliency_ext.attribution.orchestration.splitlog` (cycle and date ordering),
:mod:`~nvidia_resiliency_ext.attribution.analyzer.engine` (per-cycle path detection), and
:func:`extract_job_metadata` when building dataflow fields alongside parsed LLM output.

Job ID patterns are heuristics for paths where the id is not supplied out-of-band (e.g. MCP or HTTP).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Per-cycle log files (e.g., foo_cycle3.log); raw string for re.search(), compiled for .search()
CYCLE_LOG_PATTERN = r"_cycle(\d+)\.log$"
CYCLE_NUM_PATTERN = re.compile(r"_cycle(\d+)\.log$", re.IGNORECASE)

# Date/time in filename: *_date_YY-MM-DD_time_HH-MM-SS.log (for splitlog file sorting)
DATE_TIME_PATTERN = re.compile(
    r"_date_(\d{2})-(\d{2})-(\d{2})_time_(\d{2})-(\d{2})-(\d{2})\.log$", re.IGNORECASE
)

# Job ID extraction patterns (tried in order of specificity)
JOB_ID_PATTERNS = [
    r"_(\d+)_date_",  # foo_12345_date_2024... (most specific)
    r"[/\\]job_(\d+)[/\\]",  # job_12345/slurm.log (directory pattern)
    r"slurm-(\d+)\.(out|err|log)$",  # slurm-12345.out (SLURM default naming)
    r"[/\\](\d{6,})\.(out|err|log)$",  # /12345678.out (6+ digit job ID in filename)
    r"_(\d{6,})\.(out|err|log)$",  # prefix_12345678.out (6+ digit job ID after underscore)
]


@dataclass
class JobMetadata:
    """Metadata extracted from log file path."""

    job_id: str
    cycle_id: int


def extract_job_metadata(log_path: str, warn_on_missing_job_id: bool = True) -> JobMetadata:
    """
    Extract job ID and cycle ID from log file path.

    Job ID: tried in order of specificity; see JOB_ID_PATTERNS (inline comments).
    Cycle ID: from ..._cycle<N>.log (see CYCLE_LOG_PATTERN).

    Args:
        log_path: Path to the log file
        warn_on_missing_job_id: If True, log warning when job ID extraction fails.
            Set to False when job_id is provided externally (e.g., from POST request).

    Returns:
        JobMetadata with extracted fields (empty/zero if not found)
    """
    job_id = ""
    for pattern in JOB_ID_PATTERNS:
        match = re.search(pattern, log_path)
        if match:
            job_id = match.group(1)
            break

    if not job_id and warn_on_missing_job_id:
        logger.warning(f"Failed to extract job ID from path: {log_path}")

    match = re.search(CYCLE_LOG_PATTERN, log_path)
    if match:
        cycle_id = int(match.group(1))
    else:
        cycle_id = 0
        logger.debug(f"No cycle ID in path (not a per-cycle log): {log_path}")

    return JobMetadata(job_id=job_id, cycle_id=cycle_id)
