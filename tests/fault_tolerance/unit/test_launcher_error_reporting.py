# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import errno
import os
import select
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvidia_resiliency_ext.fault_tolerance import launcher
from nvidia_resiliency_ext.fault_tolerance.config import FaultToleranceConfig
from nvidia_resiliency_ext.fault_tolerance.launcher_error_reporting import LauncherErrorReporter
from nvidia_resiliency_ext.fault_tolerance.timeout_dump import DiagnosticCollectorProcess
from nvidia_resiliency_ext.fault_tolerance.timeout_dump.config import TimeoutDumpConfig
from nvidia_resiliency_ext.fault_tolerance.timeout_dump.pyspy import PySpy


@pytest.fixture
def log_server_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize(
    ("failure", "redirect"),
    [
        ("missing_cuda_command", True),
        ("unreadable_cuda_command", True),
        ("invalid_nproc", True),
        ("invalid_nproc", False),
    ],
)
def test_startup_failure_reaches_original_stderr(tmp_path, failure, redirect):
    args = ["--nnodes=1", "--nproc-per-node=1"]
    if redirect:
        args += [
            f"--ft-nvrx-logfile={tmp_path}/nvrx/logs/launcher.log",
            f"--ft-per-cycle-applog-prefix={tmp_path}/train.log",
        ]
    if failure == "invalid_nproc":
        args += ["--nproc-per-node=invalid"]
        expected = "Unsupported nproc_per_node value: invalid"
    elif failure in ("missing_cuda_command", "unreadable_cuda_command"):
        # Use a known executable; validation fails before any debugger is invoked.
        args += [
            "--ft-timeout-dump-enabled=true",
            "--ft-timeout-dump-tools=cuda-gdb",
            "--ft-timeout-dump-cuda-gdb-mode=stdin",
            f"--ft-timeout-dump-cuda-gdb={sys.executable}",
        ]
        if failure == "unreadable_cuda_command":
            args += [f"--ft-timeout-dump-cuda-gdb-command-file={tmp_path}/missing.gdb"]
            expected = "Timeout dump cuda-gdb command file is not readable"
        else:
            expected = "Timeout dump cuda-gdb command file is required"
    else:
        raise ValueError(f"Unsupported failure type: {failure}")

    result = subprocess.run(
        [sys.executable, "-m", launcher.__name__, *args, "train.py"],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 1
    assert expected in result.stderr
    assert "Traceback (most recent call last)" in result.stderr
    if failure == "invalid_nproc":
        # The int conversion is the chained cause, not just the final error summary.
        assert "invalid literal for int()" in result.stderr
        assert "direct cause" in result.stderr


def test_successful_launch_keeps_centralized_logging(tmp_path, log_server_port):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import sys\nprint('worker stdout marker')\n"
        "print('worker stderr marker', file=sys.stderr)\n"
    )
    log = tmp_path / "nvrx/logs/launcher.log"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            launcher.__name__,
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=1",
            "--max-restarts=0",
            f"--ft-nvrx-logfile={log}",
            f"--ft-per-cycle-applog-prefix={tmp_path}/train.log",
            f"--ft-log-server-port={log_server_port}",
            "--ft-log-server-graceful-shutdown-timeout=1",
            "--ft-gpu-memory-reclaim-timeout=0",
            str(worker),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "ft_launcher failed on" not in result.stderr
    app_log = (tmp_path / "train_cycle0.log").read_text()
    assert "worker stdout marker" in app_log
    assert "worker stderr marker" in app_log
    assert "worker_start_completed" in log.read_text()


@pytest.mark.parametrize("failure", [False, True])
def test_main_closes_saved_stderr_after_service_cleanup(tmp_path, monkeypatch, failure):
    saved = []

    def run(args):
        saved.append(launcher.launcher_error_reporter._stderr_fd)
        assert not os.get_inheritable(saved[-1])
        if failure:
            raise ValueError("startup failure")

    def cleanup():
        # Keep the fallback available throughout service cleanup.
        os.fstat(saved[0])

    monkeypatch.setattr(launcher, "run", run)
    monkeypatch.setattr(launcher, "_stop_diagnostic_collector", cleanup)
    monkeypatch.setattr(launcher, "_GRPC_SERVER_PROCESSES", [])
    monkeypatch.setattr(launcher, "_ATTRIBUTION_MANAGER", None)

    args = [f"--ft-nvrx-logfile={tmp_path}/launcher.log", "train.py"]
    if failure:
        with pytest.raises(SystemExit) as exc:
            launcher.main(args)
        assert exc.value.code == 1
    else:
        # PyTorch's @record wrapper consumes a successful SystemExit(0).
        assert launcher.main(args) is None
    with pytest.raises(OSError) as closed:
        os.fstat(saved[0])
    assert closed.value.errno == errno.EBADF


def test_main_closes_saved_stderr_even_if_cleanup_raises(tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr(
        launcher, "run", lambda args: saved.append(launcher.launcher_error_reporter._stderr_fd)
    )
    monkeypatch.setattr(
        launcher, "_stop_diagnostic_collector", Mock(side_effect=RuntimeError("cleanup failed"))
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        launcher.main([f"--ft-nvrx-logfile={tmp_path}/launcher.log", "train.py"])

    with pytest.raises(OSError) as closed:
        os.fstat(saved[0])
    assert closed.value.errno == errno.EBADF


def test_broken_original_stderr_does_not_mask_failure(tmp_path, monkeypatch):
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    real_dup = os.dup
    saved = []

    def dup(fd):
        saved.append(real_dup(write_fd))
        return saved[-1]

    monkeypatch.setattr(launcher.os, "dup", dup)
    monkeypatch.setattr(launcher, "run", Mock(side_effect=ValueError("original failure")))
    cleanup = Mock()
    monkeypatch.setattr(launcher, "_stop_diagnostic_collector", cleanup)
    monkeypatch.setattr(launcher, "_GRPC_SERVER_PROCESSES", [])
    monkeypatch.setattr(launcher, "_ATTRIBUTION_MANAGER", None)
    try:
        with pytest.raises(SystemExit) as exc:
            launcher.main([f"--ft-nvrx-logfile={tmp_path}/launcher.log", "train.py"])
        assert exc.value.code == 1
        cleanup.assert_called_once_with()
        with pytest.raises(OSError) as closed:
            os.fstat(saved[0])
        assert closed.value.errno == errno.EBADF
    finally:
        os.close(write_fd)


@pytest.mark.parametrize("helper", ["rank_monitor", "collector"])
def test_forked_helpers_release_saved_stderr(tmp_path, helper, monkeypatch):
    read_fd, write_fd = os.pipe()
    saved_fd = os.dup(write_fd)
    monkeypatch.setattr(launcher.launcher_error_reporter, "_stderr_fd", saved_fd)
    os.close(write_fd)
    process = None
    collector = None
    try:
        if helper == "rank_monitor":
            cfg = FaultToleranceConfig()
            monitors = launcher.LocalElasticAgent.setup_rank_monitors_early(
                nproc_per_node=1,
                ft_cfg=cfg,
                launcher_pid=os.getpid(),
            )
            process = monitors[0].process
        elif helper == "collector":
            cfg = TimeoutDumpConfig(
                enabled=True,
                tools=[PySpy.Config()],
                local_root=str(tmp_path / "local"),
                poll_interval=0.02,
                shutdown_grace=1,
            )
            collector = DiagnosticCollectorProcess(cfg, tmp_path / "diags")
            process = collector.process
        else:
            raise ValueError(f"Unknown helper type: {helper}")

        # Child cleanup must leave the parent's reporting stream usable.
        os.write(saved_fd, b"parent can still report\n")
        os.close(saved_fd)
        launcher.launcher_error_reporter._stderr_fd = None
        saved_fd = None
        assert select.select([read_fd], [], [], 5)[0]
        assert os.read(read_fd, 1024) == b"parent can still report\n"
        # EOF while the helper is alive proves it retained no extra pipe writer.
        assert select.select([read_fd], [], [], 5)[0]
        assert os.read(read_fd, 1024) == b""
        assert process.is_alive()
    finally:
        if saved_fd is not None:
            launcher.launcher_error_reporter.close()
        os.close(read_fd)
        if collector is not None:
            collector.shutdown()
        elif process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


@pytest.mark.parametrize("forwarding_available", [False, True])
def test_stderr_fallback_depends_on_current_forwarding_state(tmp_path, forwarding_available):
    reporter = LauncherErrorReporter()
    output = tmp_path / "original-stderr.txt"
    reporter._stderr_fd = os.open(output, os.O_WRONLY | os.O_CREAT, 0o600)
    reporter.logs_specs = SimpleNamespace(launcher_forwarding_available=forwarding_available)
    try:
        try:
            raise ValueError("fatal error marker")
        except ValueError as exc:
            reporter.report_exception(exc)
    finally:
        reporter.close()

    text = output.read_text()
    if forwarding_available:
        assert text == ""
    else:
        assert "Traceback (most recent call last)" in text
        assert "ValueError: fatal error marker" in text


def test_fatal_error_with_live_forwarding_reaches_central_log_only(tmp_path, log_server_port):
    log = tmp_path / "nvrx/logs/launcher.log"
    code = """
import os
from nvidia_resiliency_ext.fault_tolerance import launcher

# Fail after the actual launcher logging setup, before the agent starts.
def fail_before_agent(config, entrypoint):
    specs = config.logs_specs
    assert specs._reader_thread._grpc_ready.wait(10), "gRPC writer never became ready"
    assert specs.launcher_forwarding_available
    raise ValueError("fatal forwarded error marker")

os.environ["NVRX_DISABLE_RANK_MONITORS"] = "1"
launcher.elastic_launch = fail_before_agent
launcher.main()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=1",
            f"--ft-nvrx-logfile={log}",
            f"--ft-per-cycle-applog-prefix={tmp_path}/train.log",
            f"--ft-log-server-port={log_server_port}",
            "--ft-log-server-graceful-shutdown-timeout=1",
            "train.py",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 1, result.stderr
    assert "fatal forwarded error marker" not in result.stderr
    assert "ft_launcher failed on" not in result.stderr
    text = log.read_text()
    assert "Traceback (most recent call last)" in text
    assert "ValueError: fatal forwarded error marker" in text
