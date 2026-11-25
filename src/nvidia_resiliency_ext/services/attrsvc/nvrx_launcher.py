# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the launcher's own log for a section the rank monitor timed out.

When fault tolerance decides a named section has run too long, it kills the
ranks. The application log records none of that - only the wake, as surviving
ranks notice the killed ones vanish - so analysing it alone yields "cause not
in log", which is honest and useless.

The launcher writes the decision down, per cycle, beside the application log:

    [Cycle 3] Section 'step' has been open for 121.30 seconds. Timeout is 120.00 seconds
    Rank monitor sent SIGCONT then SIGKILL to the rank (PID=...)

The first line carries the cycle, which matters: one launcher log covers every
cycle of a job, and a timeout in cycle 0 says nothing about cycle 3. The second
is not cycle-tagged, so it corroborates that the monitor acted without being
attributable on its own.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from nvidia_resiliency_ext.attribution.path_utils import path_is_under_allowed_root

logger = logging.getLogger(__name__)

_SECTION_TIMEOUT_TEMPLATE = (
    r"\[Cycle {cycle}\] Section '(?P<section>[^']+)' has been open for "
    r"(?P<open_for>[\d.]+) seconds\. Timeout is (?P<timeout>[\d.]+) seconds"
)
_KILLED_RE = re.compile(r"Rank monitor sent SIGCONT then SIGKILL to the rank", re.IGNORECASE)

#: A single warning is a section briefly over its limit and is not a kill. The
#: monitor firing shows up as thousands of ranks reporting within one window,
#: so a handful of lines is deliberately not enough.
_MIN_RANKS_REPORTING = 16

#: Launcher logs run to tens of megabytes. A full scan measured at 0.06s on an
#: 86MB file, so this is a guard against something pathological rather than a
#: tuned budget.
_MAX_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class SectionTimeout:
    """A section the rank monitor stopped waiting for, in one cycle."""

    job_id: str
    cycle_id: int
    section: str
    timeout_seconds: str
    ranks_reporting: int
    monitor_killed_ranks: bool

    @property
    def summary(self) -> str:
        return (
            f"nvrx stopped waiting on section '{self.section}' in cycle {self.cycle_id} "
            f"after {self.timeout_seconds}s; {self.ranks_reporting} ranks reported it"
            + (" and the rank monitor killed them" if self.monitor_killed_ranks else "")
        )


def launcher_log_path(log_path: str, job_id: str) -> str:
    """Where the launcher writes for a run whose application log is ``log_path``.

    The experiment directory holds ``logs/`` beside ``nvrx/<job>/logs/``, and
    the launcher log carries the same date and time stamp as the application
    log it accompanies. Both are conventions, so a miss falls through.
    """
    logs_dir = os.path.dirname(os.path.abspath(log_path))
    experiment = os.path.dirname(logs_dir)
    return os.path.join(experiment, "nvrx", job_id, "logs")


def section_timeout(
    log_path: str,
    job_id: str | None,
    cycle_id: int | None,
    allowed_root: str | None = None,
) -> SectionTimeout | None:
    """Return the section timeout for this cycle, when the launcher recorded one."""

    job_id = (job_id or "").strip()
    if not job_id or cycle_id is None or not log_path:
        return None

    directory = launcher_log_path(log_path, job_id)
    if allowed_root and not path_is_under_allowed_root(directory, allowed_root):
        return None
    try:
        names = [
            name
            for name in os.listdir(directory)
            # The grpc leaf and root logs are transport chatter, not decisions.
            if name.startswith(f"nvrx_{job_id}_") and name.endswith(".log") and "_grpc_" not in name
        ]
    except OSError:
        return None
    if not names:
        return None

    pattern = re.compile(_SECTION_TIMEOUT_TEMPLATE.format(cycle=int(cycle_id)))
    section = timeout_s = ""
    reporting = 0
    killed = False
    for name in sorted(names):
        path = os.path.join(directory, name)
        if allowed_root and not path_is_under_allowed_root(path, allowed_root):
            continue
        try:
            if os.path.getsize(path) > _MAX_BYTES:
                logger.debug("launcher log too large to scan: %s", path)
                continue
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    match = pattern.search(line)
                    if match is not None:
                        reporting += 1
                        if not section:
                            section = match.group("section")
                            timeout_s = match.group("timeout")
                        continue
                    if not killed and _KILLED_RE.search(line):
                        killed = True
        except OSError:
            continue

    if reporting < _MIN_RANKS_REPORTING:
        return None
    return SectionTimeout(
        job_id=job_id,
        cycle_id=int(cycle_id),
        section=section,
        timeout_seconds=timeout_s,
        ranks_reporting=reporting,
        monitor_killed_ranks=killed,
    )
