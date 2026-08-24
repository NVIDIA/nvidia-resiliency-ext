# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consume segment health decisions before joining rendezvous."""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path
from typing import Optional

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

log = logging.getLogger(LogConfig.name)


class SegmentHealthCheck:
    """Check whether a Slurm allocation unit remains eligible for rendezvous."""

    def __init__(self, directory: str, job_id: str, task_id: str) -> None:
        self.directory = directory
        self.job_id = job_id
        self.task_id = task_id
        self._path = Path(directory) / f"segment_health_check.{job_id}.{task_id}"

    def __call__(self) -> bool:
        return self._perform_health_check()

    def _perform_health_check(self) -> bool:
        """Evaluate the current segment health artifact, failing open."""
        try:
            artifact = self._path.stat()
        except FileNotFoundError:
            return True
        except OSError as exc:
            log.warning("Ignoring segment health decision: %s", exc)
            return True

        if not stat.S_ISREG(artifact.st_mode):
            log.warning("Ignoring segment health decision: not a regular file: %s", self._path)
            return True

        return artifact.st_size == 0


def get_segment_health_check(directory: Optional[str]) -> Optional[SegmentHealthCheck]:
    """Return the segment health check for this launcher, when eligible."""
    if not directory or os.environ.get("SLURM_PROCID") != "0":
        return None

    job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID") or job_id

    log.info(
        "Segment health check installed directory=%s job_id=%s task_id=%s",
        directory,
        job_id,
        task_id,
    )
    return SegmentHealthCheck(directory, job_id, task_id)
