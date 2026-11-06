# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from nvidia_resiliency_ext.attribution.scheduler_exclusions import PublisherConfig
from nvidia_resiliency_ext.attribution.scheduler_exclusions import manager as module
from nvidia_resiliency_ext.fault_tolerance import launcher
from nvidia_resiliency_ext.fault_tolerance.config import FaultToleranceConfig


@pytest.mark.parametrize("style", ["-", "_"])
@pytest.mark.parametrize("enabled", [False, True])
def test_presence_only_flags(style, enabled):
    flag = (
        "ft-enable-segment-health-check-publisher"
        if enabled
        else "ft-disable-segment-health-check-publisher"
    )
    flag = "--" + flag.replace("-", style)
    parser = launcher.get_args_parser()
    args = parser.parse_args(
        [flag, "--ft-segment-health-check-dir", "/shared/decisions", "train.py"]
    )
    config = FaultToleranceConfig.from_args(args)
    assert args.training_script == "train.py"
    assert config.enable_segment_health_check_publisher is enabled
    assert config.segment_health_check_dir == "/shared/decisions"


def test_default_disabled_and_explicit_disable_overrides_parser_default():
    parser = launcher.get_args_parser()
    args = parser.parse_args(["train.py"])
    assert args.ft_enable_segment_health_check_publisher is None
    assert FaultToleranceConfig.from_args(args).enable_segment_health_check_publisher is False
    parser.set_defaults(
        ft_enable_segment_health_check_publisher=True,
        ft_segment_health_check_dir="/shared/decisions",
    )
    assert (
        FaultToleranceConfig.from_args(
            parser.parse_args(["train.py"])
        ).enable_segment_health_check_publisher
        is True
    )
    config = FaultToleranceConfig.from_args(
        parser.parse_args(["--ft-disable-segment-health-check-publisher", "train.py"])
    )
    assert config.enable_segment_health_check_publisher is False
    assert config.segment_health_check_dir == "/shared/decisions"


@pytest.mark.parametrize(
    "options",
    [
        [
            "--ft-enable-segment-health-check-publisher",
            "--ft-disable-segment-health-check-publisher",
        ],
        ["--ft-enable-segment-health-check-publisher=true"],
        ["--ft-disable-segment-health-check-publisher=false"],
    ],
)
def test_invalid_flag_combinations(options):
    with pytest.raises(SystemExit):
        launcher.get_args_parser().parse_args([*options, "train.py"])


@pytest.mark.parametrize("directory", [None, "", "relative"])
def test_enabled_publisher_requires_absolute_directory(directory):
    with pytest.raises(ValueError, match="requires an absolute"):
        FaultToleranceConfig.from_kwargs(
            enable_segment_health_check_publisher=True, segment_health_check_dir=directory
        )
    args = launcher.get_args_parser().parse_args(
        ["--ft-enable-segment-health-check-publisher", "train.py"]
    )
    args.ft_segment_health_check_dir = directory
    with pytest.raises(ValueError, match="requires an absolute"):
        FaultToleranceConfig.from_args(args)


def test_publisher_config_requires_real_boolean():
    with pytest.raises(ValueError, match="must be a boolean"):
        FaultToleranceConfig(enable_segment_health_check_publisher="false")


@pytest.fixture
def owner_env(monkeypatch):
    env = {
        "SLURM_ARRAY_JOB_ID": "123",
        "SLURM_JOB_PARTITION": "batch",
        "NVRX_SEGMENT_HEALTH_CHECK_DIR": "/old/path",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return env


@pytest.fixture
def manager(owner_env):
    instance = module.SegmentHealthPublisherManager(
        PublisherConfig("123", "batch", Path("/shared/decisions")), env=owner_env
    )
    yield instance
    instance.stop()


@pytest.mark.parametrize("is_store_host", [False, True])
@pytest.mark.parametrize(
    "slurm_identity",
    [
        {"SLURM_ARRAY_TASK_ID": "0", "SLURM_PROCID": "0"},
        {"SLURM_ARRAY_TASK_ID": "7", "SLURM_PROCID": "2"},
        {},
    ],
)
def test_only_store_host_owns_publisher(owner_env, slurm_identity, is_store_host, monkeypatch):
    for name in ("SLURM_ARRAY_TASK_ID", "SLURM_PROCID"):
        monkeypatch.delenv(name, raising=False)
    for name, value in slurm_identity.items():
        monkeypatch.setenv(name, value)
    factory = Mock()
    monkeypatch.setattr(launcher, "SegmentHealthPublisherManager", factory)
    monkeypatch.setattr(launcher, "_SEGMENT_HEALTH_PUBLISHER", None)
    config = FaultToleranceConfig(
        enable_segment_health_check_publisher=True, segment_health_check_dir="/shared/decisions"
    )
    try:
        launcher._start_segment_health_publisher(config, is_store_host=is_store_host)
        if is_store_host:
            factory.assert_called_once()
            factory.return_value.start.assert_called_once_with()
            assert launcher._SEGMENT_HEALTH_PUBLISHER is factory.return_value
        else:
            factory.assert_not_called()
            assert launcher._SEGMENT_HEALTH_PUBLISHER is None
    finally:
        launcher._stop_segment_health_publisher()


@pytest.mark.parametrize("missing", ["SLURM_ARRAY_JOB_ID", "SLURM_JOB_PARTITION"])
def test_owner_requires_slurm_array_context(owner_env, missing, monkeypatch):
    monkeypatch.delenv(missing)
    factory = Mock()
    monkeypatch.setattr(launcher, "SegmentHealthPublisherManager", factory)
    monkeypatch.setattr(launcher, "_SEGMENT_HEALTH_PUBLISHER", None)
    config = FaultToleranceConfig(
        enable_segment_health_check_publisher=True, segment_health_check_dir="/shared/decisions"
    )
    launcher._start_segment_health_publisher(config, is_store_host=True)
    assert launcher._SEGMENT_HEALTH_PUBLISHER is None
    factory.assert_not_called()


def test_start_is_idempotent_isolates_child_and_does_not_wait(manager, monkeypatch):
    process = Mock(pid=12345, returncode=0)
    process.poll.return_value = 0
    popen = Mock(return_value=process)
    killpg = Mock()
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.os, "killpg", killpg)
    assert manager.start()
    assert manager.start()
    assert popen.call_count == 1
    process.poll.assert_not_called()
    process.wait.assert_not_called()
    args, kwargs = popen.call_args
    assert args[0] == [
        sys.executable,
        "-m",
        "nvidia_resiliency_ext.services.scheduler_exclusions",
        "--parent-pid",
        str(os.getpid()),
    ]
    assert kwargs["env"]["NVRX_SEGMENT_HEALTH_CHECK_DIR"] == "/shared/decisions"
    assert kwargs["start_new_session"] is True
    assert kwargs["close_fds"] is True
    assert kwargs["stdin"] == kwargs["stdout"] == subprocess.DEVNULL
    assert kwargs["stderr"] is None
    manager.stop()
    manager.stop()
    killpg.assert_called_once_with(process.pid, signal.SIGKILL)
    assert manager.process is None


def test_managed_child_polls_and_is_reaped_on_shutdown(tmp_path, owner_env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    called = tmp_path / "polled"
    sinfo = bin_dir / "sinfo"
    sinfo.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\n" f"Path({str(called)!r}).touch()\n"
    )
    sinfo.chmod(0o755)
    directory = tmp_path / "decisions"
    env = dict(os.environ, **owner_env)
    env["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
    manager = module.SegmentHealthPublisherManager(
        PublisherConfig("123", "batch", directory), env=env
    )
    pidfile = directory / "segment_health_producer.123.pid"
    try:
        assert manager.start()
        deadline = time.monotonic() + 5
        while not called.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert called.exists()
        process = manager.process
        assert process is not None and process.poll() is None
        assert os.getpgid(process.pid) == process.pid
        assert pidfile.read_text().split()[1] == str(process.pid)
    finally:
        manager.stop()
    assert process.returncode == 0
    assert not pidfile.exists()
    assert manager.process is None


@pytest.mark.parametrize("returncode", [0, 1])
def test_exited_publisher_is_not_restarted(manager, monkeypatch, returncode):
    process = Mock(pid=12345, returncode=returncode)
    process.poll.return_value = returncode
    popen = Mock(return_value=process)
    killpg = Mock()
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.os, "killpg", killpg)
    assert manager.start()
    assert manager.process.poll() == returncode
    assert manager.start()
    popen.assert_called_once()
    killpg.assert_not_called()
    manager.stop()
    killpg.assert_called_once_with(process.pid, signal.SIGKILL)
    assert manager.process is None


def test_spawn_failure_is_not_retried(manager, monkeypatch):
    popen = Mock(side_effect=OSError("spawn failed"))
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    with pytest.raises(OSError, match="spawn failed"):
        manager.start()
    manager.stop()
    popen.assert_called_once()
    assert manager.process is None


def test_shutdown_escalates_to_kill_with_bounded_waits(manager, monkeypatch):
    process = Mock(pid=12345)
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("publisher", 2), -9]
    killpg = Mock()
    monkeypatch.setattr(module.os, "killpg", killpg)
    manager.process = process
    manager.stop()
    assert killpg.call_args_list == [call(12345, signal.SIGTERM), call(12345, signal.SIGKILL)]
    assert process.wait.call_args_list == [
        call(timeout=module._TERMINATE_TIMEOUT),
        call(timeout=module._KILL_TIMEOUT),
    ]
    assert manager.process is None


def test_unreaped_child_prevents_replacement(manager, monkeypatch):
    process = Mock(pid=12345)
    process.poll.return_value = None
    process.wait.side_effect = subprocess.TimeoutExpired("publisher", 2)
    popen = Mock(return_value=process)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.os, "killpg", Mock())
    manager.start()
    manager.stop()
    assert manager.process is process
    assert manager.start()
    popen.assert_called_once()


@pytest.mark.parametrize("failure", ["constructor", "start", "stop", None])
def test_launcher_lifecycle_errors_do_not_propagate(monkeypatch, failure, owner_env):
    instance = Mock()
    factory = Mock(return_value=instance)
    if failure == "constructor":
        factory.side_effect = RuntimeError("publisher failed")
    elif failure:
        getattr(instance, failure).side_effect = RuntimeError("publisher failed")
    monkeypatch.setattr(launcher, "SegmentHealthPublisherManager", factory)
    monkeypatch.setattr(launcher, "_SEGMENT_HEALTH_PUBLISHER", None)
    config = FaultToleranceConfig(
        enable_segment_health_check_publisher=True, segment_health_check_dir="/shared/decisions"
    )
    launcher._start_segment_health_publisher(config, is_store_host=True)
    if failure in ("stop", None):
        launcher._start_segment_health_publisher(config, is_store_host=True)
        factory.assert_called_once_with(
            PublisherConfig.from_env(
                {**os.environ, "NVRX_SEGMENT_HEALTH_CHECK_DIR": "/shared/decisions"}
            )
        )
    launcher._stop_segment_health_publisher()
    launcher._stop_segment_health_publisher()
    assert launcher._SEGMENT_HEALTH_PUBLISHER is None


def test_launcher_does_not_create_disabled_publisher(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(launcher, "SegmentHealthPublisherManager", factory)
    monkeypatch.setattr(launcher, "_SEGMENT_HEALTH_PUBLISHER", None)
    launcher._start_segment_health_publisher(FaultToleranceConfig(), is_store_host=True)
    factory.assert_not_called()
