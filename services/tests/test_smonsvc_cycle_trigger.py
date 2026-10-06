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


def test_completed_cycles_are_analyzed_and_the_live_one_is_only_tracked():
    resolved = _cycles(4)
    monitor = _monitor(resolved)
    job = _job()

    submit, analyze = monitor._sync_jobs_from_slurm({job.job_id: job})

    # cycle3 is still being written; cycles 0-2 each have a successor.
    assert [path for _, path in analyze] == resolved[:3]
    assert [path for _, path in submit] == [resolved[3]]


def test_a_new_cycle_promotes_its_predecessor_without_redoing_the_rest():
    resolved = _cycles(4)
    monitor = _monitor(resolved)
    job = _job()
    monitor._sync_jobs_from_slurm({job.job_id: job})

    resolved.append(f"{BASE}_cycle4.log")
    _, analyze = monitor._sync_jobs_from_slurm({job.job_id: job})

    assert [path for _, path in analyze] == [f"{BASE}_cycle3.log"]


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

    monitor._sync_jobs_from_slurm({job.job_id: job})

    assert resolved[3] not in monitor.state.analyzed_log_paths
    assert set(resolved[:3]) <= monitor.state.analyzed_log_paths


def test_sibling_array_tasks_do_not_each_analyze_the_same_cycle():
    # 149 concurrent tasks resolve to one shared application log.
    resolved = _cycles(3)
    monitor = _monitor(resolved)
    tasks = {f"4173891_{index}": _job(f"4173891_{index}") for index in (96, 133, 141)}

    _, analyze = monitor._sync_jobs_from_slurm(tasks)

    assert [path for _, path in analyze] == resolved[:2]


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
