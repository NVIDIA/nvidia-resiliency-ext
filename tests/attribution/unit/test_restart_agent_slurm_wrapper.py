# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading slurm's own output for a cancellation the application log cannot show.

Five of nine alerts examined over three weeks were runs an operator had
cancelled. Each surfaced as whatever thread was mid-handoff when the ranks
stopped - a dataloader queue read, a telemetry socket - and each produced a
RESTART on a job a person had deliberately ended.

slurmstepd writes the fact beside the log, on the same filesystem the service
already reads, and writes it as the step dies. Measured on one of the five:
cancelled 17:18:26, wrapper on disk 17:18:27, alert published 17:19:24.
"""

from __future__ import annotations

import pytest

from nvidia_resiliency_ext.services.attrsvc.slurm_wrapper import (
    operator_cancellation,
    wrapper_output_dir,
)

# The real marker, verbatim from a production batch output.
CANCELLED = (
    "srun: Job step aborted: Waiting up to 122 seconds for job step to finish.\n"
    " 0: [Nov 22 17:18:27.75396  4127879 slurmstepd   0xffffde7cf120] error: "
    "*** STEP 2200100.1 ON node0001-T01 CANCELLED AT 2026-11-22T17:18:27 "
    "DUE to SIGNAL Terminated ***\n"
)
ALL_TASKS = "Cancelling all remaining tasks of 2200100.\n"
#: Task 0 cancels its peers on every exit, so its own ordering is what
#: separates an outside stop from the run ending on its own terms.
TASK0_EXTERNAL = CANCELLED + "Task 0 exiting (rc=0), creating RDZV_CLOSED\n" + ALL_TASKS
TASK0_SELF = "Task 0 exiting (rc=0), creating RDZV_CLOSED\n" + ALL_TASKS + CANCELLED
HEALTHY = "srun: launching StepId=2200100.0 on host node0001-T01\nTask 0 exiting (rc=0)\n"


def _layout(tmp_path, job_id, contents):
    """An experiment directory: logs/ beside slurm_out/, as production lays it out."""
    logs = tmp_path / "exp" / "logs"
    out = tmp_path / "exp" / "slurm_out"
    logs.mkdir(parents=True)
    out.mkdir(parents=True)
    log_path = logs / f"run_{job_id}_date_x_cycle1.log"
    log_path.write_text("0: iteration 10/100\n")
    for index, text in enumerate(contents):
        (out / f"slurm-{job_id}_{index}.out").write_text(text)
    return str(log_path)


def test_the_directory_is_found_beside_the_logs():
    assert wrapper_output_dir("/p/exp/logs/run_cycle1.log") == "/p/exp/slurm_out"


def test_a_job_wide_cancellation_is_reported(tmp_path):
    log = _layout(tmp_path, "2200100", [CANCELLED + ALL_TASKS] + [CANCELLED] * 3)

    found = operator_cancellation(log, "2200100")

    assert found is not None
    assert found.cancelled_at == "2026-11-22T17:18:27"
    assert found.reason == "signal terminated"
    assert found.job_wide_marker is True
    assert found.tasks_cancelled == 4 and found.tasks_seen == 4
    assert "2200100" in found.summary


def test_the_array_agreeing_is_enough_without_the_closing_line(tmp_path):
    """The 'all remaining tasks' line is written by one task. Agreement across
    the array carries the same meaning when that task's file is missing."""
    log = _layout(tmp_path, "7", [CANCELLED] * 4)

    found = operator_cancellation(log, "7")

    assert found is not None and found.job_wide_marker is False


# ---------------------------------------------------------------------------
# One cancelled task is not a cancelled job
# ---------------------------------------------------------------------------


def test_a_single_cancelled_spare_does_not_settle_the_job(tmp_path):
    """Raised in review: a cold spare can be cancelled without the run stopping.

    Treating that as a cancelled job would skip analysis of a job still
    running, which is the expensive direction to be wrong in.
    """
    log = _layout(tmp_path, "7", [CANCELLED] + [HEALTHY] * 5)

    assert operator_cancellation(log, "7") is None


def test_a_bare_majority_still_settles(tmp_path):
    """Agreement does not require unanimity: in production one task's output
    was missing the marker while 99 carried it."""
    log = _layout(tmp_path, "7", [CANCELLED] * 4 + [HEALTHY])

    assert operator_cancellation(log, "7") is not None


# ---------------------------------------------------------------------------
# Reasons this does and does not settle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason, settles",
    [
        ("SIGNAL Terminated", True),
        ("TIME LIMIT", True),
        ("PREEMPTION", True),
        ("NODE FAILURE", False),
    ],
)
def test_only_clean_cancellations_are_claimed(tmp_path, reason, settles):
    """A node failure is cat 2 and a real fault, so it keeps its analysis."""
    marker = f"error: *** JOB 7 ON node CANCELLED AT 2026-11-22T17:18:27 DUE TO {reason} ***\n"
    log = _layout(tmp_path, "7", [marker + ALL_TASKS] * 3)

    assert (operator_cancellation(log, "7") is not None) is settles


# ---------------------------------------------------------------------------
# Failing to look must never cost a verdict
# ---------------------------------------------------------------------------


def test_a_healthy_array_settles_nothing(tmp_path):
    log = _layout(tmp_path, "7", [HEALTHY] * 4)
    assert operator_cancellation(log, "7") is None


def test_a_missing_wrapper_directory_falls_through(tmp_path):
    """The layout is a convention, not a guarantee. Not finding one says
    nothing about the job."""
    logs = tmp_path / "exp" / "logs"
    logs.mkdir(parents=True)
    log = logs / "run_7_date_x_cycle1.log"
    log.write_text("0: iteration 10/100\n")

    assert operator_cancellation(str(log), "7") is None


def test_no_job_id_looks_at_nothing(tmp_path):
    log = _layout(tmp_path, "7", [CANCELLED + ALL_TASKS])
    assert operator_cancellation(log, "") is None
    assert operator_cancellation(log, None) is None


def test_a_directory_outside_the_allowed_root_is_refused(tmp_path):
    """The service reads only under its allowed root, and a derived sibling
    path is still a path the caller did not supply."""
    log = _layout(tmp_path, "7", [CANCELLED + ALL_TASKS])

    assert operator_cancellation(log, "7", allowed_root=str(tmp_path / "elsewhere")) is None
    assert operator_cancellation(log, "7", allowed_root=str(tmp_path)) is not None


def test_only_this_job_is_read(tmp_path):
    """A shared slurm_out holds every job in the experiment."""
    log = _layout(tmp_path, "7", [HEALTHY])
    out = tmp_path / "exp" / "slurm_out"
    (out / "slurm-9999_0.out").write_text(CANCELLED + ALL_TASKS)

    assert operator_cancellation(log, "7") is None


# ---------------------------------------------------------------------------
# Intent: task 0 cancels the array on every exit, so the marker alone is not it
# ---------------------------------------------------------------------------


def test_a_self_teardown_is_not_an_outside_stop(tmp_path):
    """task0_exit is an EXIT trap that runs on success and failure alike, and
    unconditionally scancels the array. Its peers' markers are identical either
    way; only task 0's ordering says which happened."""
    log = _layout(tmp_path, "7", [TASK0_SELF] + [CANCELLED] * 3)

    assert operator_cancellation(log, "7") is None


def test_an_outside_stop_is_reported(tmp_path):
    log = _layout(tmp_path, "7", [TASK0_EXTERNAL] + [CANCELLED] * 3)

    found = operator_cancellation(log, "7")

    assert found is not None and found.task0_cancelled_before_exit is True


def test_without_task_zero_nothing_is_claimed(tmp_path):
    """Peers agreeing says the array came down, not why."""
    log = _layout(tmp_path, "7", [CANCELLED] * 4)
    (tmp_path / "exp" / "slurm_out" / "slurm-7_0.out").unlink()

    assert operator_cancellation(log, "7") is None


# ---------------------------------------------------------------------------
# Findings from review
# ---------------------------------------------------------------------------


def test_a_longer_job_id_does_not_vote(tmp_path):
    """slurm-70_0.out and slurm-71_0.out must not decide job 7."""
    log = _layout(tmp_path, "7", [HEALTHY])
    out = tmp_path / "exp" / "slurm_out"
    for other in ("70", "71", "700"):
        (out / f"slurm-{other}_0.out").write_text(TASK0_EXTERNAL)

    assert operator_cancellation(log, "7") is None


def test_a_node_failure_anywhere_keeps_the_analysis(tmp_path):
    """Mixed reasons previously counted toward a clean cancellation because
    only the first reason was kept."""
    node_fail = (
        "error: *** JOB 7 ON node CANCELLED AT 2026-11-22T17:18:27 DUE TO NODE FAILURE ***\n"
    )
    log = _layout(tmp_path, "7", [TASK0_EXTERNAL, node_fail, node_fail])

    assert operator_cancellation(log, "7") is None


def test_unreadable_siblings_still_count_against_the_majority(tmp_path):
    """One cancelled spare beside five empty outputs is not a cancelled job;
    dropping the unreadable ones from the denominator invented a majority."""
    log = _layout(tmp_path, "7", [CANCELLED] + [""] * 5)

    assert operator_cancellation(log, "7") is None


def test_a_symlinked_output_directory_is_refused(tmp_path):
    """A sibling slurm_out pointing outside the root would otherwise be read."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "slurm-7_0.out").write_text(TASK0_EXTERNAL)
    root = tmp_path / "root"
    (root / "exp" / "logs").mkdir(parents=True)
    log = root / "exp" / "logs" / "run_7_date_x_cycle1.log"
    log.write_text("0: iteration 10/100\n")
    (root / "exp" / "slurm_out").symlink_to(outside)

    assert operator_cancellation(str(log), "7", allowed_root=str(root)) is None


# ---------------------------------------------------------------------------
# A cached shortcut must not outlive the setting that allowed it
# ---------------------------------------------------------------------------


def test_a_cached_cancellation_is_ignored_once_the_shortcut_is_off(tmp_path):
    """Giving these results a cache identity let them persist, which in turn
    let them survive the setting being turned off.

    A restored entry is already completed, so resubmitting the log could never
    reach the analysis the operator just asked for by disabling the shortcut.
    """
    from nvidia_resiliency_ext.services.attrsvc.restart_agent_backend import (
        _SCHEDULER_CANCELLED_BASIS,
        RestartAgentServiceBackend,
    )

    log = tmp_path / "run_7_date_x_cycle1.log"
    log.write_text("0: iteration 10/100\n")
    item = {
        "log_path": str(log),
        "source": _SCHEDULER_CANCELLED_BASIS,
        "result": {"decision": "RESTART", "decision_basis": _SCHEDULER_CANCELLED_BASIS},
        "size": log.stat().st_size,
        "mtime": log.stat().st_mtime_ns,
    }

    backend = RestartAgentServiceBackend.__new__(RestartAgentServiceBackend)
    backend._wrapper_cancellation_enabled = False

    assert backend._restore_entry(item) is False, "a disabled shortcut must not be restored"
