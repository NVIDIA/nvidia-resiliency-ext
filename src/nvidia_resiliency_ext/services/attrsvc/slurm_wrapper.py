# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the scheduler's own output beside the application log.

An operator cancelling a run writes nothing to the application log. The ranks
stop, and whichever thread was mid-handoff reports its peer gone, so analysis
attributes the run to that surface - a dataloader queue read, a telemetry
socket, a collective. Every one of those is a real mechanism that did not
fail, and no amount of reading the application log distinguishes them, because
the deciding fact was never written to it.

``slurmstepd`` writes it to the batch output instead, beside the log and on
the same filesystem:

    *** JOB 2200100 ON node0001-T01 CANCELLED AT 2026-11-22T17:18:27 DUE to SIGNAL Terminated ***
    Cancelling all remaining tasks of 2200100.

Category 1 already describes this and already names that marker; it has simply
never been able to fire, because the marker is in a file the pipeline was not
given.

Granularity matters here. A single array task can be cancelled - a cold spare,
say - without the job stopping, so one task's marker is not enough. The second
line above is the job-wide statement, and agreement across the array is the
corroborating evidence.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: ``*** JOB 123 ON node CANCELLED AT 2026-11-22T17:18:27 DUE to SIGNAL Terminated ***``
_CANCELLED_RE = re.compile(
    r"\*\*\* (?:JOB|STEP) \S+ ON \S+ CANCELLED AT (?P<when>\S+)(?: DUE TO (?P<reason>.+?))? \*\*\*",
    re.IGNORECASE,
)
#: slurm's own closing statement that the whole array is going down.
_ALL_TASKS_RE = re.compile(r"Cancelling all remaining tasks of (\d+)", re.IGNORECASE)

#: Reasons this settles. A node failure is cat 2 and a genuine fault, so it is
#: left to ordinary analysis rather than claimed as a clean cancellation.
_OPERATOR_REASONS = ("signal terminated", "time limit", "preemption")

#: Only the tail is read. The marker is written as the step dies, and these
#: files can be tens of megabytes.
_TAIL_BYTES = 64 * 1024


@dataclass(frozen=True)
class WrapperCancellation:
    """A job the scheduler says was cancelled, as its own output records it."""

    job_id: str
    cancelled_at: str
    reason: str
    tasks_seen: int
    tasks_cancelled: int
    job_wide_marker: bool

    @property
    def summary(self) -> str:
        scope = "all remaining tasks" if self.job_wide_marker else "the array"
        return (
            f"slurm cancelled {scope} of job {self.job_id} at {self.cancelled_at}"
            f" ({self.reason}); {self.tasks_cancelled} of {self.tasks_seen} task outputs agree"
        )


def _tail(path: str) -> str:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            if size > _TAIL_BYTES:
                handle.seek(size - _TAIL_BYTES)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def wrapper_output_dir(log_path: str) -> str:
    """Where slurm writes batch output for a run whose app log is ``log_path``.

    By convention the experiment directory holds ``logs/`` beside
    ``slurm_out/``. That is a convention rather than a guarantee, which is why
    a miss falls through to ordinary analysis rather than being treated as an
    answer.
    """
    logs_dir = os.path.dirname(os.path.abspath(log_path))
    return os.path.join(os.path.dirname(logs_dir), "slurm_out")


def operator_cancellation(
    log_path: str,
    job_id: str | None,
    *,
    allowed_root: str | None = None,
    max_files: int = 256,
) -> WrapperCancellation | None:
    """Return the cancellation when slurm's output records one, else ``None``.

    ``None`` for every other outcome, including every way of failing to look:
    no such directory, no matching files, unreadable files, a reason this does
    not settle. The wrapper is a corroborating source, so not finding one must
    cost nothing.
    """

    job_id = (job_id or "").strip()
    if not job_id or not log_path:
        return None

    out_dir = wrapper_output_dir(log_path)
    if allowed_root and not os.path.realpath(out_dir).startswith(os.path.realpath(allowed_root)):
        logger.debug("wrapper directory outside the allowed root: %s", out_dir)
        return None
    try:
        names = sorted(
            name
            for name in os.listdir(out_dir)
            if name.startswith(f"slurm-{job_id}") and name.endswith(".out")
        )
    except OSError:
        return None
    if not names:
        return None

    seen = cancelled = 0
    when = reason = ""
    job_wide = False
    for name in names[:max_files]:
        text = _tail(os.path.join(out_dir, name))
        if not text:
            continue
        seen += 1
        match = _CANCELLED_RE.search(text)
        if match is None:
            continue
        cancelled += 1
        if not when:
            when = match.group("when") or ""
            reason = (match.group("reason") or "").strip().lower() or "unspecified"
        if _ALL_TASKS_RE.search(text):
            job_wide = True

    if not seen or not cancelled:
        return None
    # One cancelled task is not a cancelled job: a spare can go without the run
    # stopping. Either slurm says it is taking the rest down, or the array
    # agrees with itself.
    if not job_wide and cancelled * 2 <= seen:
        logger.debug("ignoring %s of %s cancelled tasks for %s", cancelled, seen, job_id)
        return None
    if not any(candidate in reason for candidate in _OPERATOR_REASONS):
        logger.debug("wrapper cancellation reason not settled here: %r", reason)
        return None

    return WrapperCancellation(
        job_id=job_id,
        cancelled_at=when,
        reason=reason,
        tasks_seen=seen,
        tasks_cancelled=cancelled,
        job_wide_marker=job_wide,
    )
