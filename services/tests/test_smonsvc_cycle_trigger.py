# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyzing a cycle as soon as a successor proves it complete.

A job that restarts in place keeps its SLURM allocation RUNNING across every
cycle, so SLURM's terminal state is the wrong trigger: run 4173891 produced
eight cycles over six hours and the allocation never left RUNNING until the
whole array was cancelled.
"""

from nvidia_resiliency_ext.services.smonsvc.log_resolver import AppLogResolution
from nvidia_resiliency_ext.services.smonsvc.models import JobState, MonitorState, SlurmJob
from nvidia_resiliency_ext.services.smonsvc.monitor import SlurmJobMonitor

LOG_DIR = "/logs"
BASE = f"{LOG_DIR}/nemotron4_ultra_60t_phase1_v1_4173891_date_26-10-02_time_08-18-26"


def _cycles(count):
    return [f"{BASE}_cycle{index}.log" for index in range(count)]


def _monitor(resolved, *, resolution_enabled=True):
    """A monitor whose log resolution is pinned to ``resolved``.

    Built without ``__init__`` on purpose: the constructor validates partitions
    against a live squeue, which these tests have no business needing.
    """
    monitor = object.__new__(SlurmJobMonitor)
    monitor.state = MonitorState()
    monitor.job_pattern = None
    monitor._app_log_resolution = AppLogResolution(enabled=resolution_enabled)
    monitor._get_log_paths = lambda job: list(resolved)
    return monitor


def _job(job_id="4173891_96", state=JobState.RUNNING):
    return SlurmJob(
        job_id=job_id,
        name="nemotron4_ultra_60t_phase1_v1",
        user="dnarayanan",
        partition="batch",
        state=state,
        stdout_path=f"{LOG_DIR}/slurm-4173891_96.out",
    )


def test_the_oldest_complete_cycle_goes_first_and_the_live_one_is_only_tracked():
    resolved = _cycles(4)
    monitor = _monitor(resolved)
    job = _job()

    submit, analyze = monitor._sync_jobs_from_slurm({job.job_id: job})

    # cycles 0-2 are complete, but only the oldest starts: L3 compares an
    # attempt against completed predecessors, so they must not run together.
    assert [path for _, path in analyze] == [resolved[0]]
    assert [path for _, path in submit] == [resolved[3]]
    assert monitor.state.cycle_inflight == {"4173891": resolved[0]}


def test_no_further_cycle_starts_while_one_is_in_flight():
    resolved = _cycles(4)
    monitor = _monitor(resolved)
    job = _job()
    monitor._sync_jobs_from_slurm({job.job_id: job})

    _, analyze = monitor._sync_jobs_from_slurm({job.job_id: job})

    assert analyze == []


def test_the_next_cycle_starts_once_the_previous_one_settles():
    resolved = _cycles(4)
    monitor = _monitor(resolved)
    job = _job()
    monitor._sync_jobs_from_slurm({job.job_id: job})

    monitor.state.cycle_inflight.clear()  # _release_finished_cycles on completion
    _, analyze = monitor._sync_jobs_from_slurm({job.job_id: job})

    assert [path for _, path in analyze] == [resolved[1]]


def test_cycles_are_worked_through_in_ascending_order():
    resolved = _cycles(4)
    monitor = _monitor(resolved)
    job = _job()

    started = []
    for _ in range(4):
        _, analyze = monitor._sync_jobs_from_slurm({job.job_id: job})
        started.extend(path for _, path in analyze)
        monitor.state.cycle_inflight.clear()

    # Ascending, and the live cycle is never among them.
    assert started == resolved[:3]


def test_a_quiet_poll_re_analyzes_nothing():
    monitor = _monitor(_cycles(4))
    job = _job()
    monitor._sync_jobs_from_slurm({job.job_id: job})

    submit, analyze = monitor._sync_jobs_from_slurm({job.job_id: job})

    assert analyze == []
    assert submit == []


def test_the_live_cycle_is_left_for_the_terminal_path():
    # Whatever is newest must stay unclaimed, or the terminal fetch that runs
    # when the allocation ends would find every path already taken and the
    # cycle that actually ended the run would never be analyzed.
    resolved = _cycles(4)
    monitor = _monitor(resolved)
    job = _job()

    for _ in range(4):
        monitor._sync_jobs_from_slurm({job.job_id: job})
        monitor.state.cycle_inflight.clear()

    assert resolved[3] not in monitor.state.analyzed_log_paths
    assert set(resolved[:3]) <= monitor.state.analyzed_log_paths


def test_sibling_array_tasks_do_not_start_different_cycles_at_once():
    # 149 concurrent tasks resolve to one shared application log, so a gate kept
    # per SLURM task would not hold: the second task would skip the cycle the
    # first claimed and start the next one alongside it. The gate is per run.
    resolved = _cycles(3)
    monitor = _monitor(resolved)
    tasks = {f"4173891_{index}": _job(f"4173891_{index}") for index in (96, 133, 141)}

    _, analyze = monitor._sync_jobs_from_slurm(tasks)

    # One cycle, claimed by exactly one of the three sibling tasks.
    assert [path for _, path in analyze] == [resolved[0]]


def test_analysis_identity_drops_the_array_task_for_a_shared_app_log():
    monitor = _monitor(_cycles(2))
    job = _job()

    assert monitor._analysis_job_id(job, f"{BASE}_cycle0.log") == "4173891"


def test_analysis_identity_keeps_the_task_for_its_own_wrapper():
    # A SLURM wrapper is written per task, so the task index identifies it.
    monitor = _monitor([])
    job = _job()

    assert monitor._analysis_job_id(job, job.stdout_path) == "4173891_96"


def test_analysis_identity_keeps_the_task_when_resolution_is_off():
    monitor = _monitor(_cycles(2), resolution_enabled=False)
    job = _job()

    assert monitor._analysis_job_id(job, f"{BASE}_cycle0.log") == "4173891_96"
