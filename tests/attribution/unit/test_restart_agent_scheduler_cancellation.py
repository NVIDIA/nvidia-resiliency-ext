# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading the scheduler before reading the log, for runs an operator stopped.

Across three weeks of alerts in one category, four of eight were jobs an
operator had cancelled. Cancellation writes nothing to the application log:
the ranks stop, and whichever thread was mid-handoff reports its peer gone, so
analysis attributes the run to that surface. In those four the surface was a
dataloader queue read, and each produced a RESTART on a job a person had
deliberately ended.

``sacct`` carries the deciding fact and carries it in time - the accounting
record exists when the job ends, which precedes any terminal analysis of it.
"""

from __future__ import annotations

import subprocess
from datetime import datetime

import pytest

from nvidia_resiliency_ext.services.attrsvc.scheduler_state import (
    SchedulerCancellation,
    operator_cancellation,
)


def _completed(stdout: str, returncode: int = 0):
    def runner(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    return runner


# The shape sacct returns for a cancelled array job: one row per task, plus
# steps whose own state is a bare CANCELLED with no account.
_ARRAY_ROWS = (
    "CANCELLED by 102840|2026-11-22T17:18:26\n"
    "CANCELLED|2026-11-22T17:18:28\n"
    "COMPLETED|2026-11-22T17:19:57\n"
    "CANCELLED by 102840|2026-11-22T17:18:26\n"
)


def test_an_operator_cancellation_is_reported_with_who_and_when(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/sacct")

    found = operator_cancellation("5785201", runner=_completed(_ARRAY_ROWS))

    assert isinstance(found, SchedulerCancellation)
    assert found.cancelled_by_uid == "102840"
    assert found.ended_at == datetime(2026, 11, 22, 17, 18, 26)
    assert "5785201" in found.summary and "102840" in found.summary


def test_a_bare_cancelled_is_not_an_operator(monkeypatch):
    """The scheduler cancels for its own reasons too - preemption, limits.

    Only a named account establishes that a person asked, and acting on an
    unattributed CANCELLED would skip analysis of failures nobody chose.
    """
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/sacct")

    assert (
        operator_cancellation("5785201", runner=_completed("CANCELLED|2026-11-22T17:18:28\n"))
        is None
    )


@pytest.mark.parametrize(
    "state",
    ["COMPLETED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "RUNNING"],
)
def test_every_other_outcome_falls_through_to_analysis(monkeypatch, state):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/sacct")

    assert operator_cancellation("1", runner=_completed(f"{state}|2026-11-22T17:18:26\n")) is None


# ---------------------------------------------------------------------------
# Failing to ask must never cost a verdict
# ---------------------------------------------------------------------------


def test_no_sacct_on_path_is_not_an_answer(monkeypatch):
    """attrsvc runs in containers with no scheduler. Absence of the tool says
    nothing about the job, so analysis must proceed."""
    monkeypatch.setattr("shutil.which", lambda _: None)

    assert operator_cancellation("5785201", runner=_completed(_ARRAY_ROWS)) is None


@pytest.mark.parametrize(
    "runner",
    [
        _completed("", returncode=1),
        _completed("not|a|known|shape\n"),
        _completed(""),
    ],
    ids=["nonzero_exit", "unparsable", "empty"],
)
def test_an_unusable_answer_falls_through(monkeypatch, runner):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/sacct")

    assert operator_cancellation("5785201", runner=runner) is None


@pytest.mark.parametrize(
    "exc", [subprocess.TimeoutExpired("sacct", 10), OSError("boom")], ids=["timeout", "oserror"]
)
def test_a_scheduler_that_will_not_answer_falls_through(monkeypatch, exc):
    """A briefly unreachable scheduler must degrade to ordinary analysis, not
    raise in the middle of a verdict."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/sacct")

    def runner(cmd, **kwargs):
        raise exc

    assert operator_cancellation("5785201", runner=runner) is None


def test_no_job_id_asks_nothing(monkeypatch):
    """Logs analysed without an id - a manual CLI run - cannot be looked up."""
    asked = []

    def runner(cmd, **kwargs):
        asked.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/sacct")
    assert operator_cancellation("", runner=runner) is None
    assert operator_cancellation(None, runner=runner) is None
    assert asked == [], "no id means no query"


def test_an_unparsable_end_still_reports_the_cancellation(monkeypatch):
    """End reads 'Unknown' while a job is still winding down. Who cancelled it
    is the part that decides the verdict; when is for the record."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/sacct")

    found = operator_cancellation("5785201", runner=_completed("CANCELLED by 39367|Unknown\n"))

    assert found is not None
    assert found.cancelled_by_uid == "39367"
    assert found.ended_at is None
