# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyzing a cycle as soon as a successor proves it complete.

A job that restarts in place keeps its SLURM allocation RUNNING across every
cycle, so SLURM's terminal state is the wrong trigger: run 2200200 produced
eight cycles over six hours and the allocation never left RUNNING until the
whole array was cancelled.
"""

import threading

from nvidia_resiliency_ext.services.smonsvc.log_resolver import AppLogResolution
from nvidia_resiliency_ext.services.smonsvc.models import JobState, MonitorState, SlurmJob
from nvidia_resiliency_ext.services.smonsvc.monitor import SlurmJobMonitor

LOG_DIR = "/logs"
BASE = f"{LOG_DIR}/trainjob_tierA_s4_phase1_v1_2200200_date_26-10-02_time_08-18-26"


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


def _job(job_id="2200200_96", state=JobState.RUNNING):
    return SlurmJob(
        job_id=job_id,
        name="trainjob_tierA_s4_phase1_v1",
        user="alice",
        partition="batch",
        state=state,
        stdout_path=f"{LOG_DIR}/slurm-2200200_96.out",
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
    run_id = monitor._analysis_job_id(job, resolved[0])
    assert monitor.state.cycle_inflight == {run_id: resolved[0]}


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
    tasks = {f"2200200_{index}": _job(f"2200200_{index}") for index in (96, 133, 141)}

    _, analyze = monitor._sync_jobs_from_slurm(tasks)

    # One cycle, claimed by exactly one of the three sibling tasks.
    assert [path for _, path in analyze] == [resolved[0]]


def test_analysis_identity_drops_the_array_task_for_a_shared_app_log():
    monitor = _monitor(_cycles(2))
    job = _job()

    identity = monitor._analysis_job_id(job, f"{BASE}_cycle0.log")

    # The task index is gone: concurrent tasks share one application log, so
    # whichever resolved it is incidental, and keeping it would split one
    # run's history into unrelated fragments.
    assert identity.startswith("2200200:")
    assert "_96" not in identity


def test_every_cycle_of_one_run_analyzes_under_the_same_identity():
    """L3 groups attempt history by exact job ID, so a run's cycles must agree."""
    monitor = _monitor(_cycles(3))
    job = _job()

    identities = {monitor._analysis_job_id(job, path) for path in _cycles(3)}

    assert len(identities) == 1, "one run is one population"


def test_a_second_run_of_the_same_array_job_gets_its_own_identity():
    """Each run numbers its cycles from zero, so sharing an identity made two
    runs collide on (job_id, cycle_id). attrsvc refuses the second path with a
    4xx, which the monitor treats as permanent - so a whole re-run was never
    analyzed."""
    monitor = _monitor([])
    job = _job()

    first = monitor._analysis_job_id(job, f"{BASE}_cycle0.log")
    other_run = "/logs/trainjob_tierA_s4_phase1_v1_2200200_date_26-10-09_time_11-02-44"
    second = monitor._analysis_job_id(job, f"{other_run}_cycle0.log")

    assert first != second, "two runs must not share one attempt identity"


def test_analysis_identity_keeps_the_task_for_its_own_wrapper():
    # A SLURM wrapper is written per task, so the task index identifies it.
    monitor = _monitor([])
    job = _job()

    assert monitor._analysis_job_id(job, job.stdout_path) == "2200200_96"


def test_analysis_identity_keeps_the_task_when_resolution_is_off():
    monitor = _monitor(_cycles(2), resolution_enabled=False)
    job = _job()

    assert monitor._analysis_job_id(job, f"{BASE}_cycle0.log") == "2200200_96"


# ─── a log that has not appeared yet is not a log that never will ───


def test_a_running_job_without_its_log_yet_is_not_retired():
    """The log appears after the first poll for nearly every job.

    Settling the job on that first empty poll left registration permanently
    blocked, so when the log did appear it was never submitted and the run was
    never analyzed at all.
    """
    monitor = _monitor([])
    job = _job(state=JobState.RUNNING)
    job.app_log_missing = True
    monitor.state.jobs[job.job_id] = job

    monitor._sync_jobs_from_slurm({job.job_id: job})

    assert job.log_submitted is False, "a running job must stay eligible"


def test_a_terminal_job_that_never_wrote_a_log_is_retired():
    monitor = _monitor([])
    job = _job(state=JobState.FAILED)
    job.app_log_missing = True
    monitor.state.jobs[job.job_id] = job

    monitor._sync_jobs_from_slurm({job.job_id: job})

    assert job.log_submitted is True


# ─── a claim that never settled must not retire the work ───


def test_a_cycle_whose_request_failed_is_retried():
    # The claim is taken before the POST. Holding it through a failure skips
    # that cycle forever and submits its successors without its history.
    monitor = _monitor(_cycles(3))
    job = _job()
    monitor.state.jobs[job.job_id] = job
    _, cycles = monitor._sync_jobs_from_slurm({job.job_id: job})
    assert len(cycles) == 1
    claimed = cycles[0][1]

    run_id = monitor._analysis_job_id(job, claimed)
    monitor.release_analysis_claim(claimed, run_id)

    _, retry = monitor._sync_jobs_from_slurm({job.job_id: job})
    assert [path for _job, path in retry] == [claimed], "the failed cycle must come back"


def test_a_submission_whose_request_failed_is_retried():
    monitor = _monitor(_cycles(1))
    job = _job()
    monitor.state.jobs[job.job_id] = job
    submit, _ = monitor._sync_jobs_from_slurm({job.job_id: job})
    assert len(submit) == 1
    claimed = submit[0][1]

    monitor.release_log_claim(claimed)

    submit_again, _ = monitor._sync_jobs_from_slurm({job.job_id: job})
    assert [path for _job, path in submit_again] == [claimed]


# ─── the terminal path obeys the same gate ───


def test_a_job_going_terminal_does_not_fan_its_cycles_out():
    """L3 needs an ordered sequence, including when the allocation ends.

    The terminal collector claimed every unanalyzed cycle in one pass, which
    sent them concurrently and defeated the serialization the in-flight path
    takes care to maintain.
    """
    monitor = _monitor(_cycles(4))
    job = _job(state=JobState.FAILED)
    job.post_success = True
    monitor.state.jobs[job.job_id] = job

    to_fetch = monitor._collect_jobs_for_fetch()

    assert len(to_fetch) == 1, f"expected one cycle at a time, got {len(to_fetch)}"


def test_the_terminal_path_waits_for_an_in_flight_cycle():
    monitor = _monitor(_cycles(4))
    job = _job(state=JobState.FAILED)
    job.post_success = True
    monitor.state.jobs[job.job_id] = job
    run_id = monitor._analysis_job_id(job, _cycles(4)[0])
    monitor.state.cycle_inflight[run_id] = _cycles(4)[0]

    assert monitor._collect_jobs_for_fetch() == []


def test_one_answered_cycle_does_not_retire_a_job_that_has_more():
    """``result_fetched`` retires the whole job, but one GET answers one cycle.

    Retiring on the first answer dropped every remaining cycle - including the
    final one, which is the attempt an alert is usually about - and made the
    job eligible for cleanup, which took the evidence with it.
    """
    cycles = _cycles(4)
    monitor = _monitor(cycles)
    monitor._state_lock = threading.RLock()
    job = _job(state=JobState.FAILED)

    for path in cycles[:-1]:
        monitor._settle_analyzed_cycle(job, path)
        assert not job.result_fetched, f"retired while {path} was still outstanding"

    monitor._settle_analyzed_cycle(job, cycles[-1])
    assert job.result_fetched, "every cycle answered, so the job is done"


def test_the_squeue_outage_path_claims_one_cycle_at_a_time():
    """Both fetch paths need the gate; only one of them had it.

    L3 compares an attempt against those that have completed, so fanning a
    job's remaining cycles out at once gives it the wrong attempt count. The
    squeue-outage path ran ungated, and an outage is exactly when a cluster is
    unhealthy and attribution matters most.
    """
    cycles = _cycles(4)
    monitor = _monitor(cycles)
    monitor._state_lock = threading.RLock()
    job = _job(state=JobState.FAILED)

    first = monitor._claim_next_cycle(job)
    assert first == cycles[0]
    # A second claim while one is in flight must yield nothing, not the rest.
    assert monitor._claim_next_cycle(job) is None


def test_an_squeue_outage_does_not_wedge_the_cycle_gate():
    """The outage path claims cycles but returns before the release probe.

    Without settlement clearing its own marker, one finished cycle left every
    later one blocked until squeue recovered - so an outage turned a fan-out
    into a stall, on the very path that only runs during an outage.
    """
    cycles = _cycles(4)
    monitor = _monitor(cycles)
    monitor._state_lock = threading.RLock()
    job = _job(state=JobState.FAILED)

    first = monitor._claim_next_cycle(job)
    assert first == cycles[0]
    assert monitor._claim_next_cycle(job) is None, "the gate must hold while one is in flight"

    # The GET answering is itself proof the analysis settled; no probe needed.
    monitor._settle_analyzed_cycle(job, first)
    assert not monitor.state.cycle_inflight, "settling releases its own gate"
    assert monitor._claim_next_cycle(job) == cycles[1], "the next cycle must not be blocked"
