# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ask the scheduler how a job ended, before reading what its log says.

An operator cancelling a run leaves nothing in the application log. The ranks
simply stop, and whichever thread was mid-handoff reports the peer vanishing -
a dataloader queue read, a collective, a checkpoint writer. Analysis then
attributes the run to whatever surface happened to notice first, which is the
wrong answer however carefully the rest of the pipeline reasons: the deciding
fact was never written to the file being read.

``sacct`` has it. A cancelled job records ``CANCELLED by <uid>`` against the
account that asked, and the record exists the moment the job ends, which is
necessarily before a terminal analysis of that job can start.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess  # nosec B404
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

#: ``CANCELLED by 102840``. A bare ``CANCELLED`` carries no uid and is not
#: matched: the scheduler also cancels for its own reasons, and only a named
#: account establishes that a person asked for this.
_CANCELLED_BY_RE = re.compile(r"^CANCELLED by (\d+)$")

_SACCT_FORMAT = "State%40,End"


@dataclass(frozen=True)
class SchedulerCancellation:
    """A job an operator stopped, and when."""

    job_id: str
    cancelled_by_uid: str
    ended_at: datetime | None

    @property
    def summary(self) -> str:
        who = f"uid {self.cancelled_by_uid}"
        when = f" at {self.ended_at.isoformat()}" if self.ended_at else ""
        return f"job {self.job_id} was cancelled by {who}{when}"


def _parse_end(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw.strip())
    except ValueError:
        # "Unknown" while a job is still winding down, among others.
        return None


def operator_cancellation(
    job_id: str | None,
    *,
    timeout_seconds: float = 10.0,
    runner=None,
) -> SchedulerCancellation | None:
    """Return the cancellation when an operator stopped this job, else ``None``.

    Returns ``None`` for every other outcome, including every way of failing to
    ask: no ``sacct`` on PATH, a non-zero exit, a timeout, an unparsable row.
    The scheduler is an optional corroborating source, so a deployment without
    one, or a scheduler that is briefly unreachable, must fall through to
    ordinary analysis rather than lose a verdict.
    """

    job_id = (job_id or "").strip()
    if not job_id:
        return None

    sacct = shutil.which("sacct")
    if sacct is None:
        return None

    cmd = [sacct, "-j", job_id, f"--format={_SACCT_FORMAT}", "-P", "-n"]
    try:
        # Fixed argv, shell=False, bounded timeout.
        completed = (runner or subprocess.run)(  # nosec B603
            cmd, capture_output=True, text=True, timeout=timeout_seconds
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("sacct lookup failed for %s: %s", job_id, exc)
        return None

    if completed.returncode != 0:
        logger.debug("sacct returned %s for %s", completed.returncode, job_id)
        return None

    # An array job reports one row per task plus its steps. Any task naming an
    # account is enough: a cancellation reaches the whole allocation at once,
    # which is why every row carries the same end time.
    for line in (completed.stdout or "").splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        match = _CANCELLED_BY_RE.match(parts[0].strip())
        if match is None:
            continue
        return SchedulerCancellation(
            job_id=job_id,
            cancelled_by_uid=match.group(1),
            ended_at=_parse_end(parts[1]),
        )
    return None
