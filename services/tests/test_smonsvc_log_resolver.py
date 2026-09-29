# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

import pytest

from nvidia_resiliency_ext.services.smonsvc.log_resolver import (
    AppLogResolution,
    base_job_id,
    declared_log_dir,
    logs_from_job_name,
    resolve_app_log,
    script_log_dir_pattern,
)

ON = AppLogResolution(enabled=True)
OFF = AppLogResolution(enabled=False)


def _nemotron_layout(tmp_path, job="3847837", cycles=(0,), name="n4_4k_egtp8_full_vlm_adam"):
    """The layout observed on the Nemotron clusters: slurm_out/ beside logs/."""
    run = tmp_path / "phase1"
    (run / "slurm_out").mkdir(parents=True)
    (run / "logs").mkdir(parents=True)
    stub = run / "slurm_out" / f"slurm-{job}_0.out"
    stub.write_text("<< START PATHS >>\nIMAGE_PATH=/x\n<< END PATHS >>\n")
    logs = []
    for c in cycles:
        p = run / "logs" / f"{name}_{job}_date_26-09-18_time_15-12-46_cycle{c}.log"
        p.write_text(f"cycle {c} training output\n")
        logs.append(p)
    return stub, logs


# ─── job id normalization ───


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("3847837", "3847837"),
        ("3847837_0", "3847837"),
        ("3847837_13", "3847837"),
        ("3847837+1", "3847837"),
        ("  3847837_2  ", "3847837"),
    ],
)
def test_base_job_id_strips_array_and_het_suffixes(raw, expected):
    assert base_job_id(raw) == expected


# ─── resolution ───


def test_resolves_stub_to_application_log(tmp_path):
    stub, logs = _nemotron_layout(tmp_path)
    assert resolve_app_log(str(stub), "3847837_0", ON) == str(logs[0])


def test_every_array_task_resolves_to_the_same_log(tmp_path):
    # The application log embeds the parent job ID, not the array task.
    stub, logs = _nemotron_layout(tmp_path)
    for task in ("3847837_0", "3847837_7", "3847837_13"):
        assert resolve_app_log(str(stub), task, ON) == str(logs[0])


def test_newest_cycle_wins(tmp_path):
    stub, logs = _nemotron_layout(tmp_path, cycles=(0, 1, 2))
    # cycle2 produced the terminal outcome; cycle10 must beat cycle2 numerically
    assert resolve_app_log(str(stub), "3847837", ON) == str(logs[-1])


def test_cycle_ordering_is_numeric_not_lexical(tmp_path):
    stub, logs = _nemotron_layout(tmp_path, cycles=(2, 10))
    assert resolve_app_log(str(stub), "3847837", ON).endswith("_cycle10.log")


def test_disabled_resolution_returns_none(tmp_path):
    stub, _ = _nemotron_layout(tmp_path)
    assert resolve_app_log(str(stub), "3847837_0", OFF) is None


def test_returns_none_when_no_log_matches_the_job(tmp_path):
    stub, _ = _nemotron_layout(tmp_path, job="3847837")
    assert resolve_app_log(str(stub), "9999999", ON) is None


def test_returns_none_without_a_logs_dir(tmp_path):
    stub = tmp_path / "slurm_out" / "slurm-1_0.out"
    stub.parent.mkdir(parents=True)
    stub.write_text("x")
    assert resolve_app_log(str(stub), "1", ON) is None


def test_does_not_match_another_jobs_log(tmp_path):
    # A prefix/suffix collision must not resolve: 384 should not match 3847837.
    stub, _ = _nemotron_layout(tmp_path, job="3847837")
    assert resolve_app_log(str(stub), "384", ON) is None


def test_layout_without_stdout_subdir_uses_sibling_logs_dir(tmp_path):
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    stub = run / "slurm-555.out"
    stub.write_text("x")
    log = run / "logs" / "job_555_date_x_cycle0.log"
    log.write_text("training")
    assert resolve_app_log(str(stub), "555", ON) == str(log)


def test_empty_inputs_are_safe():
    assert resolve_app_log("", "1", ON) is None
    assert resolve_app_log("/some/path", "", ON) is None


# ─── configuration ───


def test_from_env_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("NVRX_SMONSVC_APP_LOG_RESOLUTION", raising=False)
    cfg = AppLogResolution.from_env()
    assert cfg.enabled is False
    assert "disabled" in cfg.describe()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_from_env_enable_values(monkeypatch, value):
    monkeypatch.setenv("NVRX_SMONSVC_APP_LOG_RESOLUTION", value)
    assert AppLogResolution.from_env().enabled is True


def test_from_env_reads_subdir_overrides(monkeypatch):
    monkeypatch.setenv("NVRX_SMONSVC_APP_LOG_RESOLUTION", "1")
    monkeypatch.setenv("NVRX_SMONSVC_APP_LOG_STDOUT_SUBDIR", "batch_out")
    monkeypatch.setenv("NVRX_SMONSVC_APP_LOG_SUBDIR", "applogs")
    cfg = AppLogResolution.from_env()
    assert (cfg.stdout_subdir, cfg.log_subdir) == ("batch_out", "applogs")
    assert "batch_out/ -> applogs/" in cfg.describe()


def test_custom_subdirs_resolve(tmp_path):
    run = tmp_path / "run"
    (run / "batch_out").mkdir(parents=True)
    (run / "applogs").mkdir(parents=True)
    stub = run / "batch_out" / "slurm-77_0.out"
    stub.write_text("x")
    log = run / "applogs" / "j_77_date_x_cycle0.log"
    log.write_text("training")
    cfg = AppLogResolution(enabled=True, stdout_subdir="batch_out", log_subdir="applogs")
    assert resolve_app_log(str(stub), "77_0", cfg) == str(log)


# ─── sibling array tasks must not resubmit one shared log ───


def _claim(state, job_id, path):
    """Call the monitor's claim logic without constructing a live monitor."""
    from types import SimpleNamespace

    from nvidia_resiliency_ext.services.smonsvc.monitor import SlurmJobMonitor

    job = SimpleNamespace(job_id=job_id, log_submitted=False)
    granted = SlurmJobMonitor._claim_log_path(SimpleNamespace(state=state), job, path)
    return granted, job


def test_first_task_claims_the_log():
    from nvidia_resiliency_ext.services.smonsvc.models import MonitorState

    state = MonitorState()
    granted, job = _claim(state, "3847837_0", "/run/logs/a_3847837_cycle0.log")

    assert granted is True
    assert job.log_submitted is False
    assert state.duplicate_log_paths == 0


def test_sibling_tasks_are_skipped_and_counted():
    from nvidia_resiliency_ext.services.smonsvc.models import MonitorState

    state = MonitorState()
    path = "/run/logs/a_3847837_cycle0.log"
    _claim(state, "3847837_0", path)

    for task in ("3847837_1", "3847837_2", "3847837_13"):
        granted, job = _claim(state, task, path)
        assert granted is False
        # Marked settled so the task is not retried or fetched.
        assert job.log_submitted is True

    assert state.duplicate_log_paths == 3
    assert len(state.submitted_log_paths) == 1


def test_distinct_logs_are_each_claimed():
    from nvidia_resiliency_ext.services.smonsvc.models import MonitorState

    state = MonitorState()
    assert _claim(state, "1_0", "/run/logs/a_1_cycle0.log")[0] is True
    assert _claim(state, "2_0", "/run/logs/a_2_cycle0.log")[0] is True
    assert state.duplicate_log_paths == 0
    assert len(state.submitted_log_paths) == 2


# ─── single-cycle layout: no _cycle<N> suffix, plus metadata sidecars ───


def _single_cycle_layout(tmp_path, job="788958", name="nemotron4_derisking_super_2p6t_phase2"):
    """Observed on oci-aga: stdout sits in the run dir, log has no cycle suffix."""
    run = tmp_path / "phase2_2x_bs"
    (run / "logs").mkdir(parents=True)
    stub = run / f"slurm-{job}.out"
    stub.write_text("<< START PATHS >>\n")
    stamp = f"{name}_{job}_date_26-09-18_time_15-32-24"
    main = run / "logs" / f"{stamp}.log"
    main.write_text("training output\n")
    (run / "logs" / f"{stamp}.env.log").write_text("env dump\n")
    (run / "logs" / f"{stamp}.tasks.log").write_text("task list\n")
    return stub, main


def test_resolves_log_without_cycle_suffix(tmp_path):
    stub, main = _single_cycle_layout(tmp_path)
    assert resolve_app_log(str(stub), "788958", ON) == str(main)


def test_never_selects_env_or_tasks_sidecars(tmp_path):
    stub, main = _single_cycle_layout(tmp_path)
    resolved = resolve_app_log(str(stub), "788958", ON)
    assert not resolved.endswith(".env.log")
    assert not resolved.endswith(".tasks.log")
    assert resolved == str(main)


def test_cycle_log_outranks_a_plain_log_for_the_same_job(tmp_path):
    stub, main = _single_cycle_layout(tmp_path, job="900")
    cycled = main.parent / "run_900_date_x_cycle3.log"
    cycled.write_text("cycle 3\n")
    assert resolve_app_log(str(stub), "900", ON) == str(cycled)


def test_does_not_pick_another_jobs_log_in_a_shared_dir(tmp_path):
    stub, main = _single_cycle_layout(tmp_path, job="788958")
    other = main.parent / "nemotron4_other_788873_date_26-09-18_time_15-20-13.log"
    other.write_text("different job\n")
    assert resolve_app_log(str(stub), "788958", ON) == str(main)


# ─── one terminal analysis per log, even when siblings submitted wrappers ───


def _claim_analysis(state, job_id, path):
    from types import SimpleNamespace

    from nvidia_resiliency_ext.services.smonsvc.monitor import SlurmJobMonitor

    job = SimpleNamespace(job_id=job_id, result_fetched=False)
    granted = SlurmJobMonitor._claim_analysis_path(SimpleNamespace(state=state), job, path)
    return granted, job


def test_first_terminal_task_claims_the_analysis():
    from nvidia_resiliency_ext.services.smonsvc.models import MonitorState

    state = MonitorState()
    granted, job = _claim_analysis(state, "3848756_0", "/run/logs/a_3848756_cycle0.log")

    assert granted is True
    assert job.result_fetched is False
    assert state.duplicate_analyses == 0


def test_siblings_do_not_reanalyze_or_realert():
    from nvidia_resiliency_ext.services.smonsvc.models import MonitorState

    # Reproduces the observed failure: every array task submitted its own wrapper
    # before the application log existed, so all were fetch-eligible and each one
    # ran a terminal analysis and sent an alert for the same log.
    state = MonitorState()
    log = "/run/logs/n4_3848756_date_x_cycle0.log"
    _claim_analysis(state, "3848756_0", log)

    for task in ("3848756_17", "3848756_27", "3848756_31"):
        granted, job = _claim_analysis(state, task, log)
        assert granted is False
        assert job.result_fetched is True  # settled, so cleanup can reap it

    assert state.duplicate_analyses == 3
    assert len(state.analyzed_log_paths) == 1


def test_analysis_claims_are_independent_of_submit_claims():
    from nvidia_resiliency_ext.services.smonsvc.models import MonitorState

    # A task submits its wrapper, then later resolves to the application log.
    # Claiming the wrapper must not block analyzing the real log.
    state = MonitorState()
    _claim(state, "3848756_5", "/run/slurm_out/slurm-3848756_5.out")
    granted, _ = _claim_analysis(state, "3848756_5", "/run/logs/n4_3848756_cycle0.log")

    assert granted is True


def test_distinct_logs_are_each_analyzed():
    from nvidia_resiliency_ext.services.smonsvc.models import MonitorState

    state = MonitorState()
    assert _claim_analysis(state, "1_0", "/run/logs/a_1_cycle0.log")[0] is True
    assert _claim_analysis(state, "2_0", "/run/logs/a_2_cycle0.log")[0] is True
    assert state.duplicate_analyses == 0


# ─── a job that wrote no application log has nothing to analyze ───


def _monitor(enabled=True):
    from types import SimpleNamespace

    from nvidia_resiliency_ext.services.smonsvc.log_resolver import AppLogResolution
    from nvidia_resiliency_ext.services.smonsvc.models import MonitorState
    from nvidia_resiliency_ext.services.smonsvc.monitor import SlurmJobMonitor

    return SimpleNamespace(
        state=MonitorState(),
        _app_log_resolution=AppLogResolution(enabled=enabled),
        _expand_slurm_patterns=lambda path, job: path,
        _script_path=SlurmJobMonitor._script_path,
    )


def _get_path(monitor, job):
    from nvidia_resiliency_ext.services.smonsvc.monitor import SlurmJobMonitor

    return SlurmJobMonitor._get_log_path(monitor, job)


def _sidecars_only(tmp_path, job="3911280"):
    """Observed shape: setup wrote env/tasks metadata but training never started."""
    run = tmp_path / "12544g_forcedlb"
    (run / "slurm_out").mkdir(parents=True)
    (run / "logs").mkdir(parents=True)
    stub = run / "slurm_out" / f"slurm-{job}_100.out"
    stub.write_text("<< START PATHS >>\n")
    stamp = f"run_{job}_date_26-09-21_time_17-13-17"
    (run / "logs" / f"{stamp}.env.log").write_text("env")
    (run / "logs" / f"{stamp}.tasks.log").write_text("tasks")
    return stub


def test_job_without_application_log_is_skipped(tmp_path):
    from types import SimpleNamespace

    stub = _sidecars_only(tmp_path)
    monitor = _monitor()
    job = SimpleNamespace(
        job_id="3911280_100",
        stdout_path=str(stub),
        app_log_missing=False,
        name='',
        script='',
        work_dir='',
    )

    # The wrapper is a launcher banner; analyzing it attributes nothing.
    assert _get_path(monitor, job) is None
    assert job.app_log_missing is True
    assert monitor.state.jobs_without_app_log == 1


def test_missing_application_log_is_counted_once_per_job(tmp_path):
    from types import SimpleNamespace

    stub = _sidecars_only(tmp_path)
    monitor = _monitor()
    job = SimpleNamespace(
        job_id="3911280_100",
        stdout_path=str(stub),
        app_log_missing=False,
        name='',
        script='',
        work_dir='',
    )

    for _ in range(4):  # polled every cycle until cleanup
        _get_path(monitor, job)

    assert monitor.state.jobs_without_app_log == 1


def test_sibling_array_tasks_are_each_skipped_not_analyzed(tmp_path):
    from types import SimpleNamespace

    # 1002 tasks resolving to 1002 distinct wrappers defeated path-keyed dedup;
    # skipping removes the analyses entirely rather than deduplicating them.
    stub_dir = _sidecars_only(tmp_path).parent
    monitor = _monitor()
    for task in (0, 100, 500, 1001):
        stub = stub_dir / f"slurm-3911280_{task}.out"
        stub.write_text("<< START PATHS >>\n")
        job = SimpleNamespace(
            job_id=f"3911280_{task}",
            stdout_path=str(stub),
            app_log_missing=False,
            name='',
            script='',
            work_dir='',
        )
        assert _get_path(monitor, job) is None

    assert monitor.state.jobs_without_app_log == 4
    assert monitor.state.submitted_log_paths == set()


def test_resolution_disabled_still_submits_the_stdout_path(tmp_path):
    from types import SimpleNamespace

    stub = _sidecars_only(tmp_path)
    monitor = _monitor(enabled=False)
    job = SimpleNamespace(
        job_id="3911280_100",
        stdout_path=str(stub),
        app_log_missing=False,
        name='',
        script='',
        work_dir='',
    )

    # Deployments whose StdOut is the training log must be unaffected.
    assert _get_path(monitor, job) == str(stub)
    assert monitor.state.jobs_without_app_log == 0


def test_job_with_an_application_log_is_still_analyzed(tmp_path):
    from types import SimpleNamespace

    stub, logs = _nemotron_layout(tmp_path, job="3906471", cycles=(0, 1))
    monitor = _monitor()
    job = SimpleNamespace(
        job_id="3906471_0",
        stdout_path=str(stub),
        app_log_missing=False,
        name='',
        script='',
        work_dir='',
    )

    assert _get_path(monitor, job) == str(logs[-1])
    assert monitor.state.jobs_without_app_log == 0


# ─── wrapper written above the run directory ───


def _nested_layout(tmp_path, job="4075258"):
    """Observed shape: <parent>/slurm-<jobid>.out beside <parent>/phase1/logs/."""
    parent = tmp_path / "super_3t_data_smoke"
    logs = parent / "phase1" / "logs"
    logs.mkdir(parents=True)
    stub = parent / f"slurm-{job}.out"
    stub.write_text("launcher banner\n")
    return stub, logs


def test_resolves_when_logs_live_one_level_below_the_wrapper(tmp_path):
    stub, logs = _nested_layout(tmp_path)
    log = logs / "nemotron4_derisking_super_3t_data_smoke_4075258_date_26-09-28_time_13-42-39.log"
    log.write_text("training output\n")

    assert resolve_app_log(str(stub), "4075258", ON) == str(log)


def test_empty_log_never_outranks_a_populated_one(tmp_path):
    stub, logs = _nested_layout(tmp_path)
    stamp = "nemotron4_derisking_super_3t_data_smoke_4075258_date_26-09-28_time"
    populated = logs / f"{stamp}_13-42-39.log"
    populated.write_text("training output\n")
    # A launcher can open a log and die before writing; make the empty one newer.
    empty = logs / f"{stamp}_13-38-23.log"
    empty.write_text("")
    os.utime(empty, (empty.stat().st_atime + 60, empty.stat().st_mtime + 60))

    assert resolve_app_log(str(stub), "4075258", ON) == str(populated)


def test_all_logs_empty_resolves_to_nothing(tmp_path):
    stub, logs = _nested_layout(tmp_path)
    (logs / "run_4075258_date_x.log").write_text("")

    assert resolve_app_log(str(stub), "4075258", ON) is None


def test_nested_search_does_not_match_another_jobs_log(tmp_path):
    stub, logs = _nested_layout(tmp_path, job="4075258")
    (logs / "run_4074334_date_x.log").write_text("a different job")

    assert resolve_app_log(str(stub), "4075258", ON) is None


def test_sibling_run_directories_do_not_confuse_resolution(tmp_path):
    stub, logs = _nested_layout(tmp_path, job="4075258")
    other = logs.parent.parent / "phase2" / "logs"
    other.mkdir(parents=True)
    (other / "run_4075258_date_x_cycle3.log").write_text("same job, later phase")
    (logs / "run_4075258_date_x.log").write_text("same job, phase1")

    # Both belong to this job ID; the cycle log ranks higher.
    assert resolve_app_log(str(stub), "4075258", ON).endswith("_cycle3.log")


def test_unreadable_run_directory_is_handled(tmp_path):
    stub = tmp_path / "gone" / "slurm-1.out"
    assert resolve_app_log(str(stub), "1", ON) is None


# ─── the launcher's own LOGS_DIR declaration ───


def test_declared_logs_dir_reaches_an_unrelated_sibling(tmp_path):
    # Observed: submitted from phase1_tp1, logs written to phase1_tp1_test.
    submit = tmp_path / "ultra_2t" / "phase1_tp1"
    submit.mkdir(parents=True)
    elsewhere = tmp_path / "ultra_2t" / "phase1_tp1_test" / "logs"
    elsewhere.mkdir(parents=True)
    log = elsewhere / "run_3988470_date_26-09-24_time_14-31-43.log"
    log.write_text("training output\n")

    stub = submit / "slurm-3988470.out"
    stub.write_text(f"<< START PATHS >>\nLOGS_DIR={elsewhere}\n<< END PATHS >>\n")

    # No containment relationship exists, so only the declaration can find it.
    assert resolve_app_log(str(stub), "3988470", ON) == str(log)


def test_structural_match_wins_without_reading_the_wrapper(tmp_path):
    stub, logs = _nemotron_layout(tmp_path, job="777")
    # A bogus declaration must not be consulted when the layout already resolves.
    stub.write_text("<< START PATHS >>\nLOGS_DIR=/nonexistent\n<< END PATHS >>\n")

    assert resolve_app_log(str(stub), "777", ON) == str(logs[0])


def test_declared_logs_dir_is_ignored_when_it_does_not_exist(tmp_path):
    submit = tmp_path / "run"
    submit.mkdir()
    stub = submit / "slurm-42.out"
    stub.write_text("LOGS_DIR=/no/such/place\n")

    assert resolve_app_log(str(stub), "42", ON) is None


def test_declared_log_dir_parses_the_banner():
    import pathlib as _p

    assert declared_log_dir(_p.Path("/no/such/file")) is None


def test_declared_log_dir_reads_the_value(tmp_path):
    stub = tmp_path / "slurm-1.out"
    stub.write_text("IMAGE_PATH=/x\nLOGS_DIR=/some/logs\nOTHER=y\n")

    assert str(declared_log_dir(stub)) == "/some/logs"


def test_wrapper_without_a_banner_yields_no_declaration(tmp_path):
    stub = tmp_path / "slurm-1.out"
    stub.write_text("just some launcher output\nno paths block here\n")

    assert declared_log_dir(stub) is None


def test_unreadable_sibling_directory_does_not_fail_resolution(tmp_path, monkeypatch):
    # Shared run trees contain directories this account cannot stat, including
    # ones created by unexpanded shell variables such as a literal "${HOME}".
    stub, logs = _nemotron_layout(tmp_path, job="555")
    blocked = stub.parent.parent / "${HOME}"
    blocked.mkdir()

    real_is_dir = pathlib.Path.is_dir

    def guarded(self):
        if "${HOME}" in str(self):
            raise PermissionError(13, "Permission denied", str(self))
        return real_is_dir(self)

    monkeypatch.setattr(pathlib.Path, "is_dir", guarded)

    assert resolve_app_log(str(stub), "555", ON) == str(logs[0])


# ─── tier: the submit script states where logs go ───


def _script(tmp_path, body, name="run.sh"):
    p = tmp_path / name
    p.write_text(body)
    return p


def test_script_pattern_resolves_a_fully_static_logs_dir(tmp_path):
    s = _script(
        tmp_path,
        (
            'ROOT_DIR="/scratch/n4"\n'
            'NAME="derisking/ultra_smoke/full/llm_adam_forcedlb_mock"\n'
            'RUN_DIR="${ROOT_DIR}/${NAME}"\n'
            'LOGS_DIR="${RUN_DIR}/logs"\n'
        ),
    )
    assert (
        script_log_dir_pattern(s)
        == "/scratch/n4/derisking/ultra_smoke/full/llm_adam_forcedlb_mock/logs"
    )


def test_script_pattern_wildcards_only_the_unknown_component(tmp_path):
    # TAG comes from the submitting environment, which SLURM does not record.
    # Everything above it is still pinned, so the search stays one level wide.
    s = _script(
        tmp_path,
        (
            'ROOT_DIR="/scratch/n4"\n'
            'FINAL_DIR="${ROOT_DIR}/derisking/super_3t/phase2"\n'
            'SMOKE_ROOT="${FINAL_DIR}/smoke_64n"\n'
            'TAG=${TAG:-aa}\n'
            'RUN_DIR="${SMOKE_ROOT}/${TAG}"\n'
            'LOGS_DIR="${RUN_DIR}/logs"\n'
        ),
    )
    assert script_log_dir_pattern(s) == "/scratch/n4/derisking/super_3t/phase2/smoke_64n/*/logs"


def test_script_pattern_refuses_command_substitution(tmp_path):
    s = _script(tmp_path, 'RUN_DIR="$(pwd)/run"\nLOGS_DIR="${RUN_DIR}/logs"\n')
    # RUN_DIR is unknowable, so the pattern would be rooted at a wildcard.
    assert script_log_dir_pattern(s) is None


def test_script_pattern_without_logs_dir(tmp_path):
    assert script_log_dir_pattern(_script(tmp_path, 'FOO="bar"\n')) is None


def test_script_pattern_survives_a_cycle(tmp_path):
    s = _script(tmp_path, 'A="${B}"\nB="${A}"\nLOGS_DIR="/x/${A}/logs"\n')
    assert script_log_dir_pattern(s) == "/x/*/logs"


def test_missing_script_yields_no_pattern(tmp_path):
    assert script_log_dir_pattern(tmp_path / "absent.sh") is None


def test_resolution_uses_the_script_when_layout_fails(tmp_path):
    # Wrapper in one directory, logs written to a sibling the layout cannot reach.
    submit = tmp_path / "phase1_tp1"
    submit.mkdir()
    stub = submit / "slurm-3988470.out"
    stub.write_text("banner\n")
    logs = tmp_path / "phase1_tp1_test" / "logs"
    logs.mkdir(parents=True)
    log = logs / "run_3988470_date_x.log"
    log.write_text("training\n")

    s = _script(submit, f'LOGS_DIR="{logs}"\n', name="run_test.sh")
    assert resolve_app_log(str(stub), "3988470", ON, script=str(s)) == str(log)


def test_resolution_uses_a_wildcarded_script_pattern(tmp_path):
    submit = tmp_path / "phase2"
    submit.mkdir()
    stub = submit / "slurm-3982013.out"
    stub.write_text("banner\n")
    logs = submit / "smoke_64n" / "cont" / "logs"
    logs.mkdir(parents=True)
    log = logs / "run_3982013_date_x.log"
    log.write_text("training\n")

    s = _script(
        submit,
        (
            f'SMOKE_ROOT="{submit}/smoke_64n"\n'
            'TAG=${TAG:-aa}\n'
            'RUN_DIR="${SMOKE_ROOT}/${TAG}"\n'
            'LOGS_DIR="${RUN_DIR}/logs"\n'
        ),
        name="run_smoke.sh",
    )
    assert resolve_app_log(str(stub), "3982013", ON, script=str(s)) == str(log)


# ─── tier: the job name mirrors the run path ───


def test_job_name_recovers_a_sibling_run_directory(tmp_path):
    root = tmp_path / "derisking"
    submit = root / "ultra_smoke" / "half" / "llm_adam_forcedlb"
    submit.mkdir(parents=True)
    stub = submit / "slurm-3776237_0.out"
    stub.write_text("banner\n")
    logs = root / "ultra_smoke" / "half" / "llm_adam_forcedlb_mock" / "logs"
    logs.mkdir(parents=True)
    log = logs / "run_3776237_date_x.log"
    log.write_text("training\n")

    hits = logs_from_job_name(root, "ultra_smoke_half_llm_adam_forcedlb_mock", "3776237")
    assert [str(h) for h in hits] == [str(log)]


def test_job_name_tolerates_a_product_prefix(tmp_path):
    root = tmp_path / "derisking"
    logs = root / "nano_21t" / "phase1" / "logs"
    logs.mkdir(parents=True)
    log = logs / "run_555_date_x.log"
    log.write_text("training\n")

    # "nemotron4_derisking_..." carries tokens the path does not.
    hits = logs_from_job_name(root, "nemotron4_nano_21t_phase1", "555")
    assert [str(h) for h in hits] == [str(log)]


def test_job_name_finds_nothing_for_an_unrelated_name(tmp_path):
    root = tmp_path / "derisking"
    (root / "a" / "logs").mkdir(parents=True)
    assert logs_from_job_name(root, "totally_other_run", "999") == []


def test_job_name_is_used_when_script_resolution_fails(tmp_path):
    # The stale-script case: the script now points somewhere this job never wrote.
    root = tmp_path / "derisking"
    submit = root / "half" / "llm_adam_forcedlb"
    submit.mkdir(parents=True)
    stub = submit / "slurm-3776237_0.out"
    stub.write_text("banner\n")
    logs = root / "half" / "llm_adam_forcedlb_mock" / "logs"
    logs.mkdir(parents=True)
    log = logs / "run_3776237_date_x.log"
    log.write_text("training\n")

    stale = _script(
        submit, f'LOGS_DIR="{root}/half/llm_adam_forcedlb_mock_gtpoptin/logs"\n', name="run_mock.sh"
    )
    resolved = resolve_app_log(
        str(stub),
        "3776237",
        ON,
        script=str(stale),
        job_name="half_llm_adam_forcedlb_mock",
        search_root=str(root),
    )
    assert resolved == str(log)


def test_job_name_matches_at_an_offset_below_the_search_root(tmp_path):
    # Real shape: the search root is already <...>/ultra_smoke/half, so the name's
    # matching suffix begins five tokens in. Trimmed names hid this.
    root = tmp_path / "ultra_smoke" / "half"
    submit = root / "llm_adam_forcedlb"
    submit.mkdir(parents=True)
    logs = root / "llm_adam_forcedlb_mock" / "logs"
    logs.mkdir(parents=True)
    log = logs / "run_3776237_date_x_cycle0.log"
    log.write_text("training\n")

    hits = logs_from_job_name(
        root, "nemotron4_derisking_ultra_smoke_half_llm_adam_forcedlb_mock", "3776237"
    )
    assert [str(h) for h in hits] == [str(log)]
