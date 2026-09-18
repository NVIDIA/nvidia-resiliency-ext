# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nvidia_resiliency_ext.services.smonsvc.log_resolver import (
    AppLogResolution,
    base_job_id,
    resolve_app_log,
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
