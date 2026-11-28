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

from nvidia_resiliency_ext.attribution.path_utils import path_is_under_allowed_root

logger = logging.getLogger(__name__)

#: ``*** JOB 123 ON node CANCELLED AT 2026-11-22T17:18:27 DUE to SIGNAL Terminated ***``
_CANCELLED_RE = re.compile(
    r"\*\*\* (?:JOB|STEP) \S+ ON \S+ CANCELLED AT (?P<when>\S+)(?: DUE TO (?P<reason>.+?))? \*\*\*",
    re.IGNORECASE,
)
#: ``slurm-<job>_<task>.out``. Anchored on both sides: a prefix match would
#: let job 70 and job 71 vote on job 7, and a shared directory holds every
#: job in the experiment.
_OUTPUT_NAME_RE_TEMPLATE = r"^slurm-{job}(?:_(?P<task>\d+))?\.out$"

#: slurm's own closing statement that the whole array is going down.
_ALL_TASKS_RE = re.compile(r"Cancelling all remaining tasks of (\d+)", re.IGNORECASE)

#: Task 0 runs an EXIT trap that cancels the rest of the array on *every*
#: exit, successful or not. So a cancellation marker on a peer proves only
#: that the array came down, not why. Task 0's own output carries the
#: ordering that does: cancelled before it exited means the stop came from
#: outside; exited first means the run ended on its own terms and the trap
#: took the peers with it.
_TASK0_EXIT_RE = re.compile(r"Task 0 exiting \(rc=(?P<rc>-?\d+)\)", re.IGNORECASE)

#: Reasons this settles. A node failure is cat 2 and a genuine fault, so it is
#: left to ordinary analysis rather than claimed as a clean cancellation.
_OPERATOR_REASONS = ("signal terminated", "time limit", "preemption")

#: Only the tail is read. The marker is written as the step dies, and these
#: files can be tens of megabytes.
_TAIL_BYTES = 64 * 1024


@dataclass(frozen=True)
class WrapperCancellation:
    """A cancellation the scheduler's own output attributes to outside the run."""

    job_id: str
    cancelled_at: str
    reason: str
    tasks_seen: int
    tasks_cancelled: int
    job_wide_marker: bool
    task0_cancelled_before_exit: bool

    @property
    def summary(self) -> str:
        scope = "all remaining tasks" if self.job_wide_marker else "the array"
        return (
            f"slurm cancelled {scope} of job {self.job_id} at {self.cancelled_at}"
            f" ({self.reason}); {self.tasks_cancelled} of {self.tasks_seen} task outputs agree,"
            " and task 0 was cancelled before it exited"
        )


def _read_tail(path: str) -> str | None:
    """The last chunk of a batch output, or ``None`` when it cannot be read.

    ``None`` and ``""`` are different answers: an unreadable file still counts
    as a task that exists, so it cannot be dropped from the denominator without
    inventing a majority out of the files that happened to open.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            if size > _TAIL_BYTES:
                handle.seek(size - _TAIL_BYTES)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None


def wrapper_output_dir(log_path: str) -> str:
    """Where slurm writes batch output for a run whose app log is ``log_path``.

    By convention the experiment directory holds ``logs/`` beside
    ``slurm_out/``. That is a convention rather than a guarantee, which is why
    a miss falls through to ordinary analysis rather than being treated as an
    answer.
    """
    logs_dir = os.path.dirname(os.path.abspath(log_path))
    return os.path.join(os.path.dirname(logs_dir), "slurm_out")


def _task0_was_cancelled_before_exiting(text: str) -> bool:
    """Did the stop come from outside, or did task 0 end and cancel its peers?

    Task 0's EXIT trap cancels the array on every exit, so the marker alone is
    worthless for intent. Order settles it: a cancellation recorded before
    "Task 0 exiting" means task 0 was stopped from outside.
    """
    cancelled = _CANCELLED_RE.search(text)
    if cancelled is None:
        return False
    exited = _TASK0_EXIT_RE.search(text)
    return exited is None or cancelled.start() < exited.start()


def operator_cancellation(
    log_path: str,
    job_id: str | None,
    *,
    allowed_root: str | None = None,
) -> WrapperCancellation | None:
    """Return the cancellation when slurm's output attributes one to outside.

    ``None`` for every other outcome, including every way of failing to look:
    no such directory, no matching files, unreadable files, a reason this does
    not settle, a path that leaves the allowed root. The wrapper is a
    corroborating source, so not finding one must cost nothing.
    """

    job_id = (job_id or "").strip()
    if not job_id or not log_path:
        return None

    out_dir = wrapper_output_dir(log_path)
    # A symlinked slurm_out, or a symlinked output inside it, would otherwise
    # read files the request was never allowed to reach.
    if allowed_root and not path_is_under_allowed_root(out_dir, allowed_root):
        logger.debug("wrapper directory outside the allowed root: %s", out_dir)
        return None

    name_re = re.compile(_OUTPUT_NAME_RE_TEMPLATE.format(job=re.escape(job_id)))
    try:
        names = sorted(name for name in os.listdir(out_dir) if name_re.match(name))
    except OSError:
        return None
    if not names:
        return None

    seen = cancelled = 0
    when = reason = ""
    job_wide = task0_external = False
    for name in names:
        # Every matching task counts, readable or not. Dropping the ones that
        # failed to open is what turns one cancelled spare into a majority.
        seen += 1
        path = os.path.join(out_dir, name)
        if allowed_root and not path_is_under_allowed_root(path, allowed_root):
            continue
        text = _read_tail(path)
        if not text:
            continue
        match = _CANCELLED_RE.search(text)
        if match is None:
            continue
        this_reason = (match.group("reason") or "").strip().lower() or "unspecified"
        # A node failure anywhere in the array is a real fault that keeps its
        # analysis, however the other tasks ended.
        if not any(candidate in this_reason for candidate in _OPERATOR_REASONS):
            logger.debug("wrapper reason not settled here: %r", this_reason)
            return None
        cancelled += 1
        if not when:
            when, reason = match.group("when") or "", this_reason
        if _ALL_TASKS_RE.search(text):
            job_wide = True
        if name_re.match(name).group("task") == "0":
            task0_external = _task0_was_cancelled_before_exiting(text)

    if not cancelled:
        return None
    # One cancelled task is not a cancelled job: a spare can go without the run
    # stopping. Either slurm says it is taking the rest down, or the array
    # agrees with itself, counted against every task that exists.
    if not job_wide and cancelled * 2 <= seen:
        logger.debug("ignoring %s of %s cancelled tasks for %s", cancelled, seen, job_id)
        return None
    # Task 0 cancels its peers on every exit, so without its own ordering this
    # cannot tell an external stop from the run ending on its own terms.
    if not task0_external:
        logger.debug("task 0 does not show an external stop for %s", job_id)
        return None

    return WrapperCancellation(
        job_id=job_id,
        cancelled_at=when,
        reason=reason,
        tasks_seen=seen,
        tasks_cancelled=cancelled,
        job_wide_marker=job_wide,
        task0_cancelled_before_exit=True,
    )
