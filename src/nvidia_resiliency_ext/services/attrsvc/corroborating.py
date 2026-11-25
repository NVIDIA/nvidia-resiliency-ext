# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence that sits beside the application log rather than inside it.

Some runs end for reasons the application never records. An operator cancels
the allocation; the fault-tolerance layer decides a section is stuck and kills
the ranks. In both the application log holds only the wake - whichever thread
was mid-handoff reporting its peer gone - and analysing it attributes the run
to a mechanism that did not fail.

The deciding fact is usually written down, just somewhere else: the scheduler's
batch output, the launcher's own log. Both sit beside the log under the same
root the service already reads.

Each source answers one question and answers it narrowly. They are asked in
order of how much they explain: a source that names *why* the run ended beats
one that only says the run was stopped. The first finding wins, and finding
nothing is the normal case - these are corroborating sources, so a miss must
cost nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorroboratedFinding:
    """What a source established, and the category that describes it."""

    source: str
    category_id: int
    decision: str
    basis: str
    summary: str
    provenance: dict[str, Any] = field(default_factory=dict)


#: ``(log_path, job_id, cycle_id, allowed_root) -> finding or None``.
Probe = Callable[[str, Optional[str], Optional[int], Optional[str]], Optional[CorroboratedFinding]]


def first_finding(
    probes: Sequence[Probe],
    *,
    log_path: str,
    job_id: str | None,
    cycle_id: int | None,
    allowed_root: str | None,
) -> CorroboratedFinding | None:
    """Ask each source in turn and return the first that establishes something.

    A source raising is a source that failed to answer, not an answer. These
    read files on shared storage that may be absent, partial, or momentarily
    unreadable, and none of that is grounds for losing a verdict - so every
    probe is isolated and a failure falls through to the next.
    """

    for probe in probes:
        try:
            finding = probe(log_path, job_id, cycle_id, allowed_root)
        except Exception as exc:  # a corroborating source must never raise into the verdict
            logger.debug(
                "corroborating source failed: %s: %s", getattr(probe, "__name__", probe), exc
            )
            continue
        if finding is not None:
            logger.info("corroborated by %s: %s", finding.source, finding.summary)
            return finding
    return None


# ---------------------------------------------------------------------------
# The sources, ordered by how much they explain
# ---------------------------------------------------------------------------

#: Cat 1 already describes a clean external cancellation and already names the
#: slurmstepd marker; it has simply never been given the file that carries it.
SCHEDULER_CANCELLED_CATEGORY_ID = 1
SCHEDULER_CANCELLED_BASIS = "scheduler_cancelled"

#: The fault-tolerance layer killing a section it decided was stuck.
SECTION_TIMEOUT_CATEGORY_ID = 43
SECTION_TIMEOUT_BASIS = "nvrx_section_timeout"


def probe_section_timeout(
    log_path: str,
    job_id: str | None,
    cycle_id: int | None,
    allowed_root: str | None,
) -> CorroboratedFinding | None:
    """Asked first: it names why the run ended, not merely that it was stopped."""

    from .nvrx_launcher import section_timeout

    found = section_timeout(log_path, job_id, cycle_id, allowed_root=allowed_root)
    if found is None:
        return None
    return CorroboratedFinding(
        source="nvrx_launcher",
        category_id=SECTION_TIMEOUT_CATEGORY_ID,
        decision="RESTART",
        basis=SECTION_TIMEOUT_BASIS,
        summary=found.summary,
        provenance={
            "section": found.section,
            "timeout_seconds": found.timeout_seconds,
            "ranks_reporting": found.ranks_reporting,
            "monitor_killed_ranks": found.monitor_killed_ranks,
        },
    )


def probe_scheduler_cancellation(
    log_path: str,
    job_id: str | None,
    cycle_id: int | None,
    allowed_root: str | None,
) -> CorroboratedFinding | None:
    """A cancellation says the run was stopped, not why it stalled.

    A section timeout ends with the launcher exiting, which cancels the array,
    so both sources can fire on the same run. The timeout is the better answer
    because it names the mechanism; the cancellation is its consequence.
    """

    from .slurm_wrapper import operator_cancellation

    del cycle_id  # the wrapper is job-scoped; the cycle is established elsewhere
    found = operator_cancellation(log_path, job_id, allowed_root=allowed_root)
    if found is None:
        return None
    return CorroboratedFinding(
        source="slurm_wrapper",
        category_id=SCHEDULER_CANCELLED_CATEGORY_ID,
        decision="RESTART",
        basis=SCHEDULER_CANCELLED_BASIS,
        summary=found.summary,
        provenance={
            "cancelled_at": found.cancelled_at,
            "cancellation_reason": found.reason,
            "tasks_cancelled": found.tasks_cancelled,
            "tasks_seen": found.tasks_seen,
            "job_wide_marker": found.job_wide_marker,
        },
    )


#: Order is the contract: a source that names *why* a run ended is asked
#: before one that only says it was stopped.
DEFAULT_PROBES: tuple[Probe, ...] = (probe_section_timeout, probe_scheduler_cancellation)
