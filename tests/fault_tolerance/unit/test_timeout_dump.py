# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nvidia_resiliency_ext.fault_tolerance.config import FaultToleranceConfig
from nvidia_resiliency_ext.fault_tolerance.data import InitMsg, RankInfo
from nvidia_resiliency_ext.fault_tolerance.rank_monitor_server import RankMonitorServer
from nvidia_resiliency_ext.fault_tolerance.timeout_dump import (
    DiagnosticCollectorProcess,
    ToolCapture,
    authorize_monitor_ptrace,
    parse_capture_sections,
    parse_capture_tools,
    staging_dir,
    validate_capture_dependencies,
    validate_config,
)
from nvidia_resiliency_ext.fault_tolerance.timeout_dump.collector import DiagnosticCollector
from nvidia_resiliency_ext.fault_tolerance.timeout_dump.config import debugger_command_file


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    # Most capture tests use synthetic worker PIDs and mock termination.
    monkeypatch.setattr("nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.Process", Mock())
    config = FaultToleranceConfig(
        timeout_dump_enabled=True,
        timeout_dump="pyspy",
        timeout_dump_pyspy=sys.executable,
        timeout_dump_local_root=str(tmp_path / "local"),
        timeout_dump_capture_timeout=0.2,
        timeout_dump_max_capture_bytes=2048,
        timeout_dump_poll_interval=0.02,
        timeout_dump_shutdown_grace=0.5,
    )
    for tool in ("gdb", "cuda-gdb"):
        path = tmp_path / f"{tool}.gdb"
        path.write_text("thread apply all bt\n")
        setattr(config, "timeout_dump_" + tool.replace("-", "_") + "_command_file", str(path))
    staging_dir(config).mkdir(parents=True)
    return config


def ready(cfg, rank=0, cycle=0, payload=None, tool="pyspy"):
    directory = staging_dir(cfg) / f"cycle{cycle}"
    directory.mkdir(exist_ok=True)
    path = directory / f"rank{rank}_pid{100 + rank}_{tool}.txt.ready"
    path.write_text(payload or f"rank {rank}\n")
    return path


def destination(cfg):
    return Path(cfg.timeout_dump_local_root).parent / "nvrx" / "diags"


def final_path(cfg, cycle=0):
    return destination(cfg) / f"cycle{cycle}" / "node42_pyspy.txt"


def fake_tool(cfg, tmp_path, body, tool="pyspy"):
    executable = tmp_path / tool
    executable.write_text(f"#!{sys.executable}\n" + body)
    executable.chmod(0o700)
    cfg.timeout_dump = tool
    setattr(cfg, "timeout_dump_" + tool.replace("-", "_"), str(executable))
    if tool == "cuda-gdb":
        cfg.timeout_dump_cuda_gdb_mode = "stdin"


def monitor(cfg):
    server = RankMonitorServer(cfg, "/unused", Mock(), Mock(level=20), False)
    server.rank_info = RankInfo(global_rank=42, local_rank=0, pid=12345, host="node42")
    server._shutdown_rank_if_alive = Mock()
    return server


def test_publish_then_late_capture(cfg):
    collector = DiagnosticCollector(cfg, destination(cfg), "node42")
    first = ready(cfg)
    collector.tick()
    assert final_path(cfg).read_text() == "rank 0\n"
    assert not first.exists()
    second = ready(cfg, rank=1)
    collector.tick()
    assert final_path(cfg).read_text() == "rank 0\nrank 1\n"
    assert not second.exists()
    assert not final_path(cfg).with_suffix(".txt.partial").exists()


@pytest.mark.parametrize("local_world_size", [2, 4, 8])
def test_incomplete_batch_collects_late_ranks_on_next_poll(cfg, local_world_size):
    collector = DiagnosticCollector(
        cfg, destination(cfg), "node42", local_world_size=local_world_size
    )
    first = ready(cfg)
    with patch.object(collector, "publish", wraps=collector.publish) as publish:
        collector.tick()
        publish.assert_not_called()
        assert first.exists()
        assert collector.batches[("cycle0", "pyspy")].attempts == 0
        for rank in range(1, local_world_size):
            ready(cfg, rank=rank)
        collector.tick()
        publish.assert_called_once()
        assert len(publish.call_args.args[1].paths) == local_world_size
        collector.tick()
        publish.assert_called_once()
    assert final_path(cfg).read_text() == "".join(
        f"rank {rank}\n" for rank in range(local_world_size)
    )
    assert not list(staging_dir(cfg).glob("cycle0/*.ready"))


def test_incomplete_batch_publishes_next_poll_even_when_more_arrive(cfg):
    collector = DiagnosticCollector(cfg, destination(cfg), "node42", local_world_size=4)
    ready(cfg)
    collector.tick()
    assert not final_path(cfg).exists()
    ready(cfg, rank=1)
    collector.tick()  # New arrivals must not restart the wait for missing ranks.
    assert final_path(cfg).read_text() == "rank 0\nrank 1\n"
    late = ready(cfg, rank=2)
    collector.tick()
    assert late.exists()
    collector.tick()
    assert final_path(cfg).read_text() == "rank 0\nrank 1\nrank 2\n"
    assert not late.exists()


def test_complete_tool_does_not_wait_for_incomplete_tool_or_cycle(cfg):
    collector = DiagnosticCollector(cfg, destination(cfg), "node42", local_world_size=4)
    for rank in range(4):
        ready(cfg, rank=rank, tool="gdb")
    ready(cfg)
    ready(cfg, cycle=1)
    collector.tick()
    gdb = destination(cfg) / "cycle0/node42_gdb.txt"
    assert gdb.read_text() == "rank 0\nrank 1\nrank 2\nrank 3\n"
    assert not final_path(cfg).exists()
    assert not final_path(cfg, cycle=1).exists()
    collector.tick()
    assert final_path(cfg).read_text() == "rank 0\n"
    assert final_path(cfg, cycle=1).read_text() == "rank 0\n"


def test_multiple_pids_of_one_rank_do_not_make_a_complete_batch(cfg):
    collector = DiagnosticCollector(cfg, destination(cfg), "node42", local_world_size=2)
    ready(cfg)
    (staging_dir(cfg) / "cycle0/rank0_pid200_pyspy.txt.ready").write_text("replacement rank 0\n")
    collector.tick()
    assert not final_path(cfg).exists()
    collector.tick()
    assert final_path(cfg).read_text() == "rank 0\nreplacement rank 0\n"


@pytest.mark.parametrize("already_deferred", [False, True])
def test_shutdown_flush_publishes_incomplete_batches_immediately(cfg, already_deferred):
    collector = DiagnosticCollector(cfg, destination(cfg), "node42", local_world_size=4)
    source = ready(cfg)
    if already_deferred:
        collector.tick()
    collector.tick(flush=True)
    assert final_path(cfg).read_text() == "rank 0\n"
    assert not source.exists()


def test_deferred_batch_freezes_inputs_only_after_first_upload_attempt(cfg):
    collector = DiagnosticCollector(cfg, destination(cfg), "node42", local_world_size=4)
    first = ready(cfg)
    collector.tick()
    second = ready(cfg, rank=1)
    with patch("os.replace", side_effect=OSError("Lustre unavailable")):
        collector.tick()
    batch = collector.batches[("cycle0", "pyspy")]
    assert batch.attempts == 1
    assert batch.paths == (first, second)
    third = ready(cfg, rank=2)
    collector.tick()
    assert final_path(cfg).read_text() == "rank 0\nrank 1\n"
    assert third.exists()
    collector.tick()
    collector.tick()
    assert final_path(cfg).read_text() == "rank 0\nrank 1\nrank 2\n"
    assert not third.exists()


def test_publication_does_not_remove_directory_under_producer(cfg):
    collector = DiagnosticCollector(cfg, destination(cfg), "node42")
    ready(cfg)
    capture = ToolCapture(cfg, 12345, "pyspy")
    mkdir = Path.mkdir

    def publish_after_mkdir(path, *args, **kwargs):
        mkdir(path, *args, **kwargs)
        # Complete the previous upload after the producer's mkdir, before open.
        with patch.object(Path, "mkdir", mkdir):
            collector.tick()

    with patch.object(Path, "mkdir", publish_after_mkdir):
        path = capture.publish_local(monitor(cfg).rank_info, 0, "step")
    assert path.exists()
    collector.tick()
    assert "pid=12345" in final_path(cfg).read_text()
    assert not path.exists()


def test_late_capture_gets_new_budget_after_batch_exhaustion(cfg):
    cfg.timeout_dump_upload_attempts = 2
    collector = DiagnosticCollector(cfg, destination(cfg), "node42")
    exhausted = ready(cfg, rank=0)
    with patch.object(collector, "publish", side_effect=OSError("storage unavailable")) as publish:
        for _ in range(3):
            collector.tick()
        assert publish.call_count == 2
    assert exhausted.exists()

    late = ready(cfg, rank=1)
    # New arrivals get the full retry budget; old files are never retried.
    with patch.object(collector, "publish", side_effect=OSError("still unavailable")) as publish:
        collector.tick()
        assert publish.call_args.args[1].paths == (late,)
        assert publish.call_args.args[1].attempts == 1
    collector.tick()
    assert final_path(cfg).read_text() == "rank 1\n"
    assert exhausted.exists()
    assert not late.exists()
    with patch.object(collector, "publish") as publish:
        collector.tick()
        publish.assert_not_called()


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("tool", ["pyspy", "cuda-gdb"])
def test_blocked_launch_cannot_delay_worker_termination(cfg, cancel, tool):
    cfg.timeout_dump = f"{tool},gdb"
    cfg.timeout_dump_cuda_gdb_mode = "stdin"
    server = monitor(cfg)
    server.rank_info.pid = os.getpid()
    server._timed_out_sections = {"step"}
    entered = threading.Event()
    release = threading.Event()
    popen = subprocess.Popen
    captures = []
    inputs = []

    def blocked_launch(*args, **kwargs):
        if tool == "cuda-gdb":
            inputs.append(kwargs["stdin"])
        entered.set()
        assert release.wait(5), "test did not release debugger startup"
        return popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)

    async def run():
        task = asyncio.create_task(server._handle_timeout())
        while not entered.is_set():
            await asyncio.sleep(0.001)
        captures.append(server._diagnostic_child)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        server._shutdown_rank_if_alive.assert_called_once()
        assert captures[0].is_running()
        assert not release.is_set()
        # Periodic housekeeping must retain the pending launch even without a PID.
        rank_info, server.rank_info = server.rank_info, None
        periodic = asyncio.create_task(server._periodic_rank_check())
        await asyncio.sleep(0)
        assert server._diagnostic_child is captures[0]
        periodic.cancel()
        with pytest.raises(asyncio.CancelledError):
            await periodic
        server.rank_info = rank_info
        # A replacement incarnation must not start another debugger either.
        server._diagnostic_attempted = False
        await server._handle_timeout()
        assert server._diagnostic_child is captures[0]

    try:
        with patch(
            "nvidia_resiliency_ext.fault_tolerance.timeout_dump.capture.subprocess.Popen",
            side_effect=blocked_launch,
        ) as launch:
            started = time.monotonic()
            asyncio.run(run())
            assert time.monotonic() - started < cfg.timeout_dump_capture_timeout + 0.5
            launch.assert_called_once()
    finally:
        release.set()
        for capture in captures:
            capture._launch_thread.join(timeout=3)
            assert not capture.is_running()
            assert capture.process.returncode == -signal.SIGKILL
        assert all(stream.closed for stream in inputs)


def test_retry_rebuilds_partial_keeps_old_final_and_fixed_inputs(cfg):
    collector = DiagnosticCollector(cfg, destination(cfg), "node42")
    ready(cfg)
    collector.tick()
    first = ready(cfg, rank=1)
    with patch("os.replace", side_effect=OSError("Lustre unavailable")):
        collector.tick()
    assert first.exists()
    assert final_path(cfg).read_text() == "rank 0\n"
    partial = final_path(cfg).with_suffix(".txt.partial")
    partial.write_text("interrupted garbage")
    late = ready(cfg, rank=2)
    collector.tick()
    assert final_path(cfg).read_text() == "rank 0\nrank 1\n"
    assert late.exists()  # not part of the selected upload
    collector.tick()
    assert final_path(cfg).read_text() == "rank 0\nrank 1\nrank 2\n"


def test_retry_after_publish_does_not_reappend(cfg):
    collector = DiagnosticCollector(cfg, destination(cfg), "node42")
    source = ready(cfg)
    unlink = Path.unlink

    def fail_cleanup(path, *args, **kwargs):
        if path == source:
            raise OSError("local cleanup failed")
        return unlink(path, *args, **kwargs)

    with patch.object(Path, "unlink", fail_cleanup):
        collector.tick()
    assert final_path(cfg).read_text() == "rank 0\n"
    batch = collector.batches[("cycle0", "pyspy")]
    assert batch.published
    collector.tick()
    assert final_path(cfg).read_text() == "rank 0\n"
    assert not source.exists()


def test_retry_budget_does_not_reset_on_poll(cfg):
    cfg.timeout_dump_upload_attempts = 2
    collector = DiagnosticCollector(cfg, destination(cfg), "node42")
    source = ready(cfg)
    with patch.object(collector, "publish", side_effect=OSError("offline")) as publish:
        for _ in range(10):
            collector.tick()
        assert publish.call_count == 2
    assert source.exists()
    assert collector.batches[("cycle0", "pyspy")].exhausted


def test_each_poll_batches_all_ready_files_and_cycles(cfg):
    collector = DiagnosticCollector(cfg, destination(cfg), "node42")
    ready(cfg)
    ready(cfg, rank=1)
    ready(cfg, rank=2, cycle=1)
    collector.tick()
    assert final_path(cfg).read_text() == "rank 0\nrank 1\n"
    assert final_path(cfg, cycle=1).read_text() == "rank 2\n"


def test_ignore_incomplete_and_unknown_files(cfg):
    source = ready(cfg)
    source.rename(source.with_suffix(".tmp"))
    (source.parent / "other.txt.ready").write_text("unknown")
    collector = DiagnosticCollector(cfg, destination(cfg), "node42")
    collector.tick()
    assert not final_path(cfg).exists()


def test_capture_only_once_on_step_timeout(cfg):
    server = monitor(cfg)
    server._timed_out_sections = {"step"}
    with patch("nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.ToolCapture") as cls:
        capture = cls.return_value
        capture.run = AsyncMock()
        asyncio.run(server._handle_timeout())
        asyncio.run(server._handle_timeout())
        assert capture.run.await_count == 1
        capture.publish_local.assert_called_once()
        assert server._shutdown_rank_if_alive.call_count == 2


@pytest.mark.parametrize("sections", [set(), {"setup"}, {"checkpoint"}, {"checkpointing"}])
def test_other_timeouts_kill_without_capture(cfg, sections):
    server = monitor(cfg)
    server._timed_out_sections = sections
    with patch("nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.ToolCapture") as cls:
        asyncio.run(server._handle_timeout())
        cls.assert_not_called()
    server._shutdown_rank_if_alive.assert_called_once()
    assert not server._diagnostic_attempted
    assert not list(staging_dir(cfg).glob("cycle*/*.ready"))


@pytest.mark.parametrize("section", ["checkpoint", "checkpointing", "setup", "custom-section"])
def test_selected_section_timeout_captures(cfg, section):
    cfg.timeout_dump_sections = f"step, {section}"
    cfg.rank_section_timeouts = {section: 1}
    server = monitor(cfg)
    server.open_sections = {section: 0}
    assert server._is_section_timeout_elapsed(2)
    with patch("nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.ToolCapture") as cls:
        cls.return_value.run = AsyncMock()
        cls.return_value.is_running.return_value = False
        asyncio.run(server._handle_timeout())
        cls.return_value.run.assert_awaited_once()
        assert cls.return_value.publish_local.call_args.args[2] == section
    server._shutdown_rank_if_alive.assert_called_once()


def test_selection_requires_exact_expired_section(cfg):
    cfg.timeout_dump_sections = "checkpoint"
    cfg.rank_section_timeouts = {"step": 1, "checkpoint": 10, "checkpointing": 1}
    server = monitor(cfg)
    server.open_sections = {"step": 0, "checkpoint": 0, "checkpointing": 0}
    assert server._is_section_timeout_elapsed(2)
    with patch("nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.ToolCapture") as cls:
        asyncio.run(server._handle_timeout())
        cls.assert_not_called()
    server._shutdown_rank_if_alive.assert_called_once()


@pytest.mark.parametrize("sections", [None, [], "", " ", "step,", ",step", "step, step"])
def test_invalid_capture_sections_rejected(cfg, sections):
    cfg.timeout_dump_sections = sections
    with pytest.raises(ValueError, match="timeout_dump_sections"):
        validate_config(cfg)


def test_section_selection_cli():
    from nvidia_resiliency_ext.fault_tolerance import launcher

    parser = launcher.get_args_parser()
    defaults = FaultToleranceConfig.from_args(parser.parse_args(["train.py"]))
    assert defaults.timeout_dump_sections == "step"
    config = FaultToleranceConfig.from_args(
        parser.parse_args(["--ft-timeout-dump-sections=setup,custom", "train.py"])
    )
    assert parse_capture_sections(config.timeout_dump_sections) == {"setup", "custom"}
    assert not config.timeout_dump_enabled


@pytest.mark.parametrize("timeout_type", ["heartbeat", "out_of_section"])
def test_section_names_do_not_enable_non_section_timeouts(cfg, timeout_type):
    cfg.timeout_dump_sections = "heartbeat,out_of_section"
    cfg.initial_rank_heartbeat_timeout = 1
    cfg.rank_out_of_section_timeout = 1
    server = monitor(cfg)
    server.start_time = server.out_of_section_time = 0
    if timeout_type == "heartbeat":
        assert server._is_hb_timeout_elapsed(2)
        assert not server._is_section_timeout_elapsed(0.5)
    else:
        assert server._is_section_timeout_elapsed(2)
    with patch("nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.ToolCapture") as cls:
        asyncio.run(server._handle_timeout())
        cls.assert_not_called()
    server._shutdown_rank_if_alive.assert_called_once()


def test_nested_nonexpired_section_cannot_hide_timeout(cfg):
    server = monitor(cfg)
    cfg.rank_section_timeouts = {"step": 1, "checkpointing": 100}
    server.open_sections = {"step": 0, "checkpointing": 9}
    assert server._is_section_timeout_elapsed(10)
    assert server._timed_out_sections == {"step"}


@pytest.mark.parametrize("section", ["checkpoint", "checkpointing", None])
def test_non_step_detection_path_terminates_without_dump(cfg, section):
    server = monitor(cfg)
    server._step_section_seen_this_cycle = True
    cfg.rank_section_timeouts = {"step": 1, "checkpoint": 1, "checkpointing": 1}
    cfg.rank_out_of_section_timeout = 1
    server.open_sections = {section: 0} if section is not None else {}
    server.out_of_section_time = 0
    # Clear any reason retained from an earlier check before classifying timeout.
    server._timed_out_sections = {"step"}
    assert server._is_section_timeout_elapsed(10)
    with patch("nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.ToolCapture") as capture:
        asyncio.run(server._handle_timeout())
        capture.assert_not_called()
    server._shutdown_rank_if_alive.assert_called_once()


def test_replacement_worker_not_killed(cfg):
    server = monitor(cfg)
    old = server.rank_info
    server._timed_out_sections = {"step"}

    async def replace(**kwargs):
        server.rank_info = RankInfo(global_rank=42, local_rank=0, pid=67890, host="node42")

    with patch("nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.ToolCapture") as cls:
        cls.return_value.run = replace
        asyncio.run(server._handle_timeout())
        assert cls.return_value.publish_local.call_args.args[0] is old
    server._shutdown_rank_if_alive.assert_called_once()


def test_new_worker_can_capture(cfg):
    server = monitor(cfg)
    server._diagnostic_attempted = True
    msg = InitMsg(rank_info=server.rank_info)
    with patch(
        "nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.write_obj_to_ipc_stream",
        new=AsyncMock(),
    ):
        asyncio.run(server._handle_init_msg(msg, Mock()))
    assert not server._diagnostic_attempted


@pytest.mark.parametrize("connection", ["disconnect", "reconnect", "replacement"])
@pytest.mark.parametrize("finish_capture", [False, True])
def test_original_worker_terminated_across_connection_changes(
    cfg, monkeypatch, connection, finish_capture
):
    from psutil import Process

    monkeypatch.setattr(
        "nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.Process", Process
    )
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    replacement = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    server = monitor(cfg)
    server.rank_info.pid = worker.pid
    server._timed_out_sections = {"step"}
    server._shutdown_rank_if_alive = RankMonitorServer._shutdown_rank_if_alive.__get__(server)

    async def change_connection(**kwargs):
        server._handle_ipc_connection_lost()
        if connection != "disconnect":
            pid = worker.pid if connection == "reconnect" else replacement.pid
            with patch(
                "nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.write_obj_to_ipc_stream",
                new=AsyncMock(),
            ):
                await server._handle_init_msg(
                    InitMsg(rank_info=RankInfo(42, 0, "node42", pid)), Mock()
                )
        if not finish_capture:
            # The timer must terminate the original even before capture returns.
            await asyncio.sleep(cfg.timeout_dump_capture_timeout + 0.05)
            assert worker.poll() == -signal.SIGKILL

    try:
        with patch("nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.ToolCapture") as cls:
            cls.return_value.run = change_connection
            cls.return_value.is_running.return_value = False
            asyncio.run(server._handle_timeout())
        assert worker.wait(timeout=2) == -signal.SIGKILL
        assert replacement.poll() is None
    finally:
        for process in (worker, replacement):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)


def test_reused_pid_is_not_signalled(cfg):
    from psutil import Process

    # Simulate a saved identity differing from the process now holding this PID.
    process = Process(os.getpid())
    process._ident = (process.pid, process.create_time() - 1)
    server = monitor(cfg)
    with patch("psutil.os.kill") as kill:
        RankMonitorServer._shutdown_rank_if_alive(server, process)
    kill.assert_not_called()


@pytest.mark.parametrize("tool", ["pyspy", "cuda-gdb"])
def test_hanging_capture_does_not_delay_termination_or_publish_writer(cfg, tmp_path, tool):
    fake_tool(
        cfg, tmp_path, "import time\nprint('partial stack', flush=True)\ntime.sleep(30)\n", tool
    )
    server = monitor(cfg)
    server.rank_info.pid = os.getpid()
    server._timed_out_sections = {"step"}
    kill_times = []
    server._shutdown_rank_if_alive.side_effect = lambda *args: kill_times.append(time.monotonic())
    started = time.monotonic()
    asyncio.run(server._handle_timeout())
    assert kill_times[0] - started < cfg.timeout_dump_capture_timeout + 0.3
    output = next(staging_dir(cfg).glob("cycle0/*.ready")).read_text()
    assert "partial stack" in output
    assert "status=timeout" in output
    proc = server._diagnostic_child.process
    proc.wait(timeout=2)  # test cleanup only; production never waits unbounded
    assert proc.returncode == -signal.SIGKILL


@pytest.mark.parametrize("tool", ["pyspy", "cuda-gdb"])
def test_output_cap(cfg, tmp_path, tool):
    fake_tool(cfg, tmp_path, "import os\nwhile True: os.write(1, b'x' * 65536)\n", tool)
    capture = asyncio.run(ToolCapture(cfg, os.getpid(), tool).run())
    assert capture.status == "output_limit"
    path = capture.publish_local(RankInfo(0, 0, "node42", os.getpid()), 0, "step")
    assert path.stat().st_size <= cfg.timeout_dump_max_capture_bytes
    capture.process.wait(timeout=2)


def test_missing_pyspy_still_terminates(cfg):
    cfg.timeout_dump_pyspy = "/does-not-exist/py-spy"
    server = monitor(cfg)
    server._timed_out_sections = {"step"}
    asyncio.run(server._handle_timeout())
    server._shutdown_rank_if_alive.assert_called_once()
    assert "status=error:" in next(staging_dir(cfg).glob("cycle0/*.ready")).read_text()


def test_authorization_uses_peer_pid():
    left, right = socket.socketpair()
    try:
        with patch("ctypes.CDLL") as libc:
            libc.return_value.prctl.return_value = 0
            authorize_monitor_ptrace(left)
            args = libc.return_value.prctl.call_args.args
            assert args[0] == 0x59616D61
            assert args[1].value == os.getpid()
    finally:
        left.close()
        right.close()


def test_fork_collector_process_uploads_and_releases_ownership(cfg):
    collector_process = DiagnosticCollectorProcess(cfg, destination(cfg))
    try:
        assert collector_process.process._start_method == "fork"
        with pytest.raises(BlockingIOError):
            DiagnosticCollectorProcess(cfg, destination(cfg))
        source = ready(cfg)
        final = destination(cfg) / "cycle0" / f"{socket.gethostname()}_pyspy.txt"
        deadline = time.monotonic() + 5
        while not final.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert final.read_text() == "rank 0\n"
        while source.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not source.exists()
    finally:
        collector_process.shutdown()
    replacement = DiagnosticCollectorProcess(cfg, destination(cfg))
    replacement.shutdown()


def test_stale_captures_preserved_and_rejected(cfg):
    source = ready(cfg)
    with pytest.raises(RuntimeError, match="Stale captures"):
        DiagnosticCollectorProcess(cfg, destination(cfg))
    assert source.exists()


@pytest.mark.parametrize(
    "name,value",
    [
        ("capture_timeout", 0),
        ("capture_timeout", float("inf")),
        ("upload_attempts", 0),
        ("max_capture_bytes", 1),
        ("local_root", "relative"),
        ("poll_interval", 0),
    ],
)
def test_config_limits(cfg, name, value):
    setattr(cfg, "timeout_dump_" + name, value)
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_reconnect_does_not_allow_second_capture(cfg):
    server = monitor(cfg)
    msg = InitMsg(rank_info=server.rank_info)
    with patch(
        "nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.write_obj_to_ipc_stream",
        new=AsyncMock(),
    ):
        asyncio.run(server._handle_init_msg(msg, Mock()))
        server._diagnostic_attempted = True
        asyncio.run(server._handle_init_msg(msg, Mock()))
        assert server._diagnostic_attempted
        msg.rank_info = RankInfo(42, 0, "node42", 67890)
        asyncio.run(server._handle_init_msg(msg, Mock()))
        assert not server._diagnostic_attempted


@pytest.mark.parametrize("tool", ["pyspy", "cuda-gdb"])
def test_cancellation_still_kills_worker(cfg, tmp_path, tool):
    fake_tool(cfg, tmp_path, "import time\ntime.sleep(30)\n", tool)
    server = monitor(cfg)
    server.rank_info.pid = os.getpid()
    server._timed_out_sections = {"step"}

    async def cancel():
        task = asyncio.create_task(server._handle_timeout())
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())
    server._shutdown_rank_if_alive.assert_called_once()
    output = next(staging_dir(cfg).glob("cycle0/*.ready")).read_text()
    assert "status=cancelled" in output
    proc = server._diagnostic_child.process
    proc.wait(timeout=2)
    assert proc.returncode == -signal.SIGKILL


def test_monitor_shutdown_during_capture_preserves_cancellation(cfg, tmp_path):
    fake_tool(cfg, tmp_path, "import time\nprint('partial stack', flush=True)\ntime.sleep(30)\n")
    cfg.timeout_dump_capture_timeout = 5
    server = monitor(cfg)
    server.ipc_socket_path = str(tmp_path / "worker.sock")
    server.launcher_ipc_socket_path = str(tmp_path / "launcher.sock")
    server._timed_out_sections = {"step"}
    server._is_hb_timeout_elapsed = Mock(return_value=False)
    server._is_section_timeout_elapsed = Mock(return_value=True)
    errors = []
    interrupted_capture = []

    async def idle_health_check():
        await asyncio.Future()

    async def main():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        deadline = loop.time() + 3

        def shutdown_when_capturing():
            child = server._diagnostic_child
            if child is not None and b"partial stack" in child.output:
                interrupted_capture.append(child)
                server._handle_signal(signal.SIGTERM)
            elif loop.time() >= deadline:
                # Bound a failed test without leaving the IPC servers running.
                server._handle_signal(signal.SIGTERM)
            else:
                loop.call_later(0.01, shutdown_when_capturing)

        loop.call_later(0.01, shutdown_when_capturing)
        with (
            patch.object(loop, "add_signal_handler"),
            patch.object(server, "_periodic_node_health_check", idle_health_check),
        ):
            await server._rank_monitor_loop()
        # Returning exercises asyncio.run's cancellation of unfinished tasks.

    asyncio.run(main())
    assert len(interrupted_capture) == 1
    assert server._periodic_tasks[0].cancelled()
    assert not errors
    server._shutdown_rank_if_alive.assert_called_once()
    output = next(staging_dir(cfg).glob("cycle0/*.ready")).read_text()
    assert "partial stack" in output
    assert "status=cancelled" in output
    proc = interrupted_capture[0].process
    proc.wait(timeout=2)
    assert proc.returncode == -signal.SIGKILL
    assert not Path(server.ipc_socket_path).exists()
    assert not Path(server.launcher_ipc_socket_path).exists()


def test_failed_read_of_previous_snapshot_cannot_replace_it(cfg):
    collector = DiagnosticCollector(cfg, destination(cfg), "node42")
    ready(cfg)
    collector.tick()
    source = ready(cfg, rank=1)
    original_open = Path.open

    def fail_read(path, *args, **kwargs):
        if path == final_path(cfg):
            raise OSError("read failed")
        return original_open(path, *args, **kwargs)

    with patch.object(Path, "open", fail_read):
        collector.tick()
    assert final_path(cfg).read_text() == "rank 0\n"
    assert source.exists()


def test_collector_process_shutdown_does_not_join_uninterruptible_child(cfg):
    collector_process = object.__new__(DiagnosticCollectorProcess)
    collector_process.cfg = cfg
    collector_process.process = Mock()
    collector_process.process.is_alive.return_value = True
    with patch("multiprocessing.process._children") as children:
        collector_process.shutdown()
        children.discard.assert_called_once_with(collector_process.process)
    collector_process.process.kill.assert_called_once()
    assert [call.kwargs['timeout'] for call in collector_process.process.join.call_args_list] == [
        cfg.timeout_dump_shutdown_grace,
        0.1,
    ]


def test_launcher_cli_and_root_inference(cfg):
    from nvidia_resiliency_ext.fault_tolerance import launcher

    args = launcher.get_args_parser().parse_args(
        [
            "--ft-timeout-dump-enabled=true",
            "--ft-timeout-dump=pyspy",
            f"--ft-timeout-dump-pyspy={sys.executable}",
            f"--ft-timeout-dump-local-root={cfg.timeout_dump_local_root}",
            "--ft-timeout-dump-capture-timeout",
            "7",
            "train.py",
        ]
    )
    merged = FaultToleranceConfig.from_args(args)
    assert merged.timeout_dump == "pyspy"
    assert merged.timeout_dump_capture_timeout == 7
    with patch(
        "nvidia_resiliency_ext.fault_tolerance.timeout_dump.DiagnosticCollectorProcess"
    ) as collector_process:
        try:
            launcher._start_diagnostic_collector(
                merged, "/shared/nvrx/123/logs/launcher.log", local_world_size=4
            )
            collector_process.assert_called_once_with(
                merged, Path("/shared/nvrx/123/diags"), local_world_size=4
            )
        finally:
            launcher._stop_diagnostic_collector()
        collector_process.return_value.shutdown.assert_called_once()


@pytest.mark.parametrize(
    "rank_env",
    [
        {
            "SLURM_JOB_ID": "123",
            "NVRX_INFRA_RANK_FROM_NODENAME": "1",
            "SLURMD_NODENAME": "node42",
        },
        {"CROSS_SLURM_PROCID": "invalid"},
    ],
)
@pytest.mark.parametrize("missing_tool", [False, True])
def test_optional_startup_rank_failure_does_not_skip_dependency_validation(
    cfg, rank_env, missing_tool
):
    from nvidia_resiliency_ext.fault_tolerance import launcher

    if missing_tool:
        cfg.timeout_dump_pyspy = "/missing/py-spy"
    with (
        patch.dict(os.environ, rank_env, clear=True),
        patch.object(launcher.logger, "isEnabledFor", return_value=True),
        patch(
            "nvidia_resiliency_ext.fault_tolerance.timeout_dump.validate_capture_dependencies",
            wraps=validate_capture_dependencies,
        ) as validate,
        patch(
            "nvidia_resiliency_ext.fault_tolerance.timeout_dump.DiagnosticCollectorProcess"
        ) as process,
    ):
        try:
            if missing_tool:
                with pytest.raises(ValueError, match="'pyspy' is missing or not executable"):
                    launcher._start_diagnostic_collector(cfg, "/shared/nvrx/123/logs/launcher.log")
                process.assert_not_called()
            else:
                launcher._start_diagnostic_collector(
                    cfg, "/shared/nvrx/123/logs/launcher.log", local_world_size=4
                )
                process.assert_called_once_with(
                    cfg, Path("/shared/nvrx/123/diags"), local_world_size=4
                )
            validate.assert_called_once_with(cfg, log_resolved_paths=False)
        finally:
            launcher._stop_diagnostic_collector()


def test_long_poll_is_interrupted_for_shutdown_upload(cfg):
    cfg.timeout_dump_poll_interval = 30
    cfg.timeout_dump_shutdown_grace = 2
    collector_process = DiagnosticCollectorProcess(cfg, destination(cfg), local_world_size=4)
    try:
        # Let the fork child initialize and enter its idle wait.
        time.sleep(0.2)
        source = ready(cfg)
        started = time.monotonic()
        collector_process.shutdown()
        collector_process = None
        assert time.monotonic() - started < 2.5
        final = destination(cfg) / "cycle0" / f"{socket.gethostname()}_pyspy.txt"
        assert final.read_text() == "rank 0\n"
        assert not source.exists()
    finally:
        if collector_process is not None:
            collector_process.shutdown()


def test_cli_can_disable_and_has_no_removed_options():
    from dataclasses import fields

    from nvidia_resiliency_ext.fault_tolerance import launcher

    parser = launcher.get_args_parser()
    args = parser.parse_args(
        ["--ft-timeout-dump=pyspy", "--ft-timeout-dump-enabled=false", "train.py"]
    )
    merged = FaultToleranceConfig.from_args(args)
    assert not merged.timeout_dump_enabled
    with patch(
        "nvidia_resiliency_ext.fault_tolerance.timeout_dump.DiagnosticCollectorProcess"
    ) as process:
        launcher._start_diagnostic_collector(merged, None)
        process.assert_not_called()
    names = {field.name for field in fields(FaultToleranceConfig)}
    for removed in (
        "nvrx_root",
        "quiet_interval",
        "max_batch_delay",
        "upload_jitter",
        "retry_delay",
    ):
        assert "timeout_dump_" + removed not in names
        assert "--ft-timeout-dump-" + removed.replace("_", "-") not in parser._option_string_actions
    assert not any(name.startswith("--ft-diagnostics") for name in parser._option_string_actions)


@pytest.mark.parametrize("mode", ["unknown", "none", "", "pyspy,gdb,", "pyspy,pyspy", None, True])
def test_invalid_dump_mode_rejected(cfg, mode):
    cfg.timeout_dump = mode
    with pytest.raises(ValueError, match="timeout_dump must"):
        validate_config(cfg)


@pytest.mark.parametrize("name", ["capture_timeout", "poll_interval", "shutdown_grace"])
def test_boolean_duration_is_rejected(cfg, name):
    setattr(cfg, "timeout_dump_" + name, True)
    with pytest.raises(ValueError, match="finite and positive"):
        validate_config(cfg)


@pytest.mark.parametrize("logfile", [None, "relative/logs/nvrx.log", "/shared/nvrx.log"])
def test_dump_requires_logfile_in_nvrx_layout(cfg, logfile):
    from nvidia_resiliency_ext.fault_tolerance import launcher

    with pytest.raises(ValueError, match="--ft-nvrx-logfile"):
        launcher._start_diagnostic_collector(cfg, logfile)
    args = launcher.get_args_parser().parse_args(
        ["--ft-per-cycle-applog-prefix=/shared/train.log", "train.py"]
    )
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="--ft-nvrx-logfile"):
            launcher._validate_args(args, cfg, logfile)


def test_collector_initial_offset_and_retry_cadence(cfg):
    from nvidia_resiliency_ext.fault_tolerance.timeout_dump.collector import _collector_main

    handlers = {}
    sleeps = []

    def sleep(delay):
        sleeps.append(delay)
        if len(sleeps) == 3:
            handlers[signal.SIGTERM](signal.SIGTERM, None)

    with (
        patch("os.setsid"),
        patch("os.close"),
        patch("os.getppid", return_value=123),
        patch("signal.signal", side_effect=lambda sig, handler: handlers.__setitem__(sig, handler)),
        patch("time.sleep", side_effect=sleep),
        patch("random.uniform", return_value=0.007) as jitter,
        patch(
            "nvidia_resiliency_ext.fault_tolerance.timeout_dump.collector.DiagnosticCollector"
        ) as collector,
        patch("nvidia_resiliency_ext.shared_utils.log_manager.setup_logger"),
    ):
        _collector_main(cfg, destination(cfg), 999, [], 123, 4)
        collector.assert_called_once_with(cfg, destination(cfg), local_world_size=4)
        jitter.assert_called_once_with(0, cfg.timeout_dump_poll_interval)
        # Two normal polls, then exactly one final shutdown collection pass.
        assert collector.return_value.tick.call_count == 3
        assert collector.return_value.tick.call_args.kwargs == {"flush": True}
    assert sleeps == [0.007, cfg.timeout_dump_poll_interval, cfg.timeout_dump_poll_interval]


def test_defaults_and_list_order():
    cfg = FaultToleranceConfig()
    assert not cfg.timeout_dump_enabled
    assert cfg.timeout_dump_capture_timeout == 30
    assert parse_capture_tools(cfg.timeout_dump) == ("pyspy", "gdb", "cuda-gdb")
    cfg.timeout_dump = "cuda-gdb, pyspy, gdb"
    assert parse_capture_tools(cfg.timeout_dump) == ("cuda-gdb", "pyspy", "gdb")


def test_missing_requested_tool_fails_before_collector_starts(cfg):
    from nvidia_resiliency_ext.fault_tolerance import launcher

    cfg.timeout_dump = "pyspy,gdb"
    cfg.timeout_dump_gdb = "/missing/gdb"
    with patch(
        "nvidia_resiliency_ext.fault_tolerance.timeout_dump.DiagnosticCollectorProcess"
    ) as process:
        with pytest.raises(ValueError, match="gdb.*missing or not executable"):
            launcher._start_diagnostic_collector(cfg, "/shared/nvrx/123/logs/launcher.log")
        process.assert_not_called()


def test_validate_all_executables_and_resolve_paths(cfg):
    cfg.timeout_dump = "pyspy,gdb,cuda-gdb"
    cfg.timeout_dump_pyspy = cfg.timeout_dump_gdb = cfg.timeout_dump_cuda_gdb = sys.executable
    validate_capture_dependencies(cfg)
    assert cfg.timeout_dump_gdb == sys.executable


@pytest.mark.parametrize("log_resolved_paths", [False, True])
@pytest.mark.parametrize("debug_enabled", [False, True])
def test_dependency_summaries_require_requested_debug_logging(
    cfg, caplog, capture_nvrx_logs, log_resolved_paths, debug_enabled
):
    import logging

    cfg.timeout_dump = "pyspy,gdb,cuda-gdb"
    cfg.timeout_dump_pyspy = cfg.timeout_dump_gdb = cfg.timeout_dump_cuda_gdb = sys.executable
    with caplog.at_level(logging.DEBUG if debug_enabled else logging.INFO, logger="nvrx"):
        validate_capture_dependencies(cfg, log_resolved_paths=log_resolved_paths)
    summaries = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Timeout dump ")
    ]
    if log_resolved_paths and debug_enabled:
        assert len(summaries) == 2
        assert f"cuda-gdb={sys.executable} (mode=batch)" in summaries[0]
        assert f"gdb={cfg.timeout_dump_gdb_command_file}" in summaries[1]
        assert f"cuda-gdb={cfg.timeout_dump_cuda_gdb_command_file}" in summaries[1]
    else:
        assert not summaries


def test_disabled_configuration_does_not_validate_tools(cfg):
    from nvidia_resiliency_ext.fault_tolerance import launcher

    cfg.timeout_dump_enabled = False
    cfg.timeout_dump = "invalid"
    cfg.timeout_dump_pyspy = "/missing/pyspy"
    with patch(
        "nvidia_resiliency_ext.fault_tolerance.timeout_dump.validate_capture_dependencies"
    ) as validate:
        launcher._start_diagnostic_collector(cfg, None)
        validate.assert_not_called()


@pytest.mark.parametrize("value", ["yes", "0", "typo"])
def test_enable_cli_requires_explicit_boolean(value):
    from nvidia_resiliency_ext.fault_tolerance import launcher

    with pytest.raises(SystemExit):
        launcher.get_args_parser().parse_args(["--ft-timeout-dump-enabled=" + value, "train.py"])


def test_tool_list_does_not_enable():
    from nvidia_resiliency_ext.fault_tolerance import launcher

    parser = launcher.get_args_parser()
    args = parser.parse_args(["--ft-timeout-dump=gdb", "train.py"])
    config = FaultToleranceConfig.from_args(args)
    assert config.timeout_dump == "gdb"
    assert not config.timeout_dump_enabled
    args = parser.parse_args(
        ["--ft-timeout-dump-enabled=true", "--ft-timeout-dump=cuda-gdb,gdb", "train.py"]
    )
    config = FaultToleranceConfig.from_args(args)
    assert config.timeout_dump_enabled
    assert parse_capture_tools(config.timeout_dump) == ("cuda-gdb", "gdb")


def test_sequence_shares_remaining_budget_and_stages_after_termination(cfg):
    from types import SimpleNamespace

    cfg.timeout_dump = "pyspy,gdb,cuda-gdb"
    cfg.timeout_dump_capture_timeout = 30
    server = monitor(cfg)
    server._timed_out_sections = {"step"}
    now = [100.0]
    captures = []
    budgets = []

    def capture_factory(config, pid, tool):
        capture = Mock()
        capture.process = None
        capture.is_running.return_value = False

        async def run(timeout):
            budgets.append(timeout)
            now[0] += [2, 8, 1][len(budgets) - 1]

        capture.run = run
        capture.publish_local.side_effect = (
            lambda *args: server._shutdown_rank_if_alive.assert_called_once()
        )
        captures.append((tool, capture))
        return capture

    with (
        patch(
            "nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.ToolCapture", capture_factory
        ),
        patch(
            "nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.time",
            SimpleNamespace(monotonic=lambda: now[0]),
        ),
    ):
        asyncio.run(server._handle_timeout())
    assert [tool for tool, _ in captures] == ["pyspy", "gdb", "cuda-gdb"]
    assert budgets == [10, 14, 20]
    for _, capture in captures:
        capture.publish_local.assert_called_once()


def test_unreaped_diagnostic_prevents_next_attachment(cfg):
    cfg.timeout_dump = "gdb,cuda-gdb"
    server = monitor(cfg)
    server._timed_out_sections = {"step"}
    with patch("nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.ToolCapture") as cls:
        capture = cls.return_value
        capture.run = AsyncMock()
        capture.process.poll.return_value = None
        asyncio.run(server._handle_timeout())
        cls.assert_called_once_with(cfg, 12345, "gdb")
    server._shutdown_rank_if_alive.assert_called_once()


def test_previous_worker_live_debugger_blocks_capture_until_exit(cfg):
    server = monitor(cfg)
    server._timed_out_sections = {"step"}
    old_child = Mock()
    old_child.process.poll.return_value = None
    server._diagnostic_child = old_child
    with patch("nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.ToolCapture") as cls:
        asyncio.run(server._handle_timeout())
        cls.assert_not_called()
        server._shutdown_rank_if_alive.assert_called_once()
        assert server._diagnostic_child is old_child

        # A replacement worker gets an attempt after the old debugger exits.
        old_child.process.poll.return_value = -signal.SIGKILL
        old_child.is_running.return_value = False
        msg = InitMsg(rank_info=RankInfo(42, 0, "node42", 67890))
        with patch(
            "nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.write_obj_to_ipc_stream",
            new=AsyncMock(),
        ):
            asyncio.run(server._handle_init_msg(msg, Mock()))
        cls.return_value.run = AsyncMock()
        cls.return_value.process.poll.return_value = 0
        asyncio.run(server._handle_timeout())
        cls.assert_called_once_with(cfg, 67890, "pyspy")
        assert server._diagnostic_child is cls.return_value


def test_three_hanging_tools_are_sequential_and_bounded(cfg, tmp_path):
    cfg.timeout_dump = "pyspy,gdb,cuda-gdb"
    cfg.timeout_dump_capture_timeout = 0.9
    for tool in parse_capture_tools(cfg.timeout_dump):
        executable = tmp_path / tool
        executable.write_text(
            f"#!{sys.executable}\nimport os, time\n"
            f"print('{tool} pid=' + str(os.getpid()), flush=True)\ntime.sleep(30)\n"
        )
        executable.chmod(0o700)
        setattr(cfg, "timeout_dump_" + tool.replace("-", "_"), str(executable))
    server = monitor(cfg)
    server.rank_info.pid = os.getpid()
    server._timed_out_sections = {"step"}
    captures = []

    def make_capture(*args):
        # The previous tool must exit before the next subprocess can start.
        assert all(capture.process.poll() is not None for capture in captures)
        capture = ToolCapture(*args)
        captures.append(capture)
        return capture

    started = time.monotonic()
    with patch(
        "nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.ToolCapture",
        side_effect=make_capture,
    ):
        asyncio.run(server._handle_timeout())
    assert time.monotonic() - started < cfg.timeout_dump_capture_timeout + 0.3
    assert len(captures) == 3
    assert server._diagnostic_child is captures[-1]
    assert all(capture.process.poll() is not None for capture in captures)
    outputs = list(staging_dir(cfg).glob("cycle0/*.ready"))
    assert len(outputs) == 3
    for path in outputs:
        assert "status=timeout" in path.read_text()
    assert len({capture.process.pid for capture in captures}) == 3


def test_runtime_failure_continues_to_next_tool(cfg):
    cfg.timeout_dump = "pyspy,gdb"
    cfg.timeout_dump_pyspy = "/missing/py-spy"
    cfg.timeout_dump_gdb = "/missing/gdb"
    server = monitor(cfg)
    server._timed_out_sections = {"step"}
    asyncio.run(server._handle_timeout())
    assert server._diagnostic_child.tool == "gdb"
    assert server._diagnostic_child.process is None
    assert len(list(staging_dir(cfg).glob("cycle0/*.ready"))) == 2
    server._shutdown_rank_if_alive.assert_called_once()


@pytest.mark.parametrize("tool", ["gdb", "cuda-gdb"])
def test_debugger_command_wraps_user_file(cfg, tool):
    command = ToolCapture(cfg, 123, tool).command()
    assert "--nx" in command
    assert "--batch" in command
    assert "attach 123" in command
    body = Path(command[command.index("-x") + 1]).read_text()
    assert "thread apply all bt" in body
    assert command.index("attach 123") < command.index("-x") < command.index("detach")
    assert command[-1] == "detach"
    assert not any(
        "gcore" in arg or "generate-core-file" in arg or "SIGSEGV" in arg
        for arg in command + [body]
    )
    assert command.index("set debuginfod enabled off") < command.index("attach 123")


@pytest.mark.parametrize("mode", ["batch", "stdin"])
def test_cuda_gdb_mode_does_not_change_other_tools(cfg, mode):
    cfg.timeout_dump_cuda_gdb_mode = mode
    assert "--batch" in ToolCapture(cfg, 123, "gdb").command()
    assert ToolCapture(cfg, 123, "pyspy").command() == [sys.executable, "dump", "--pid", "123"]


@pytest.mark.parametrize("mode", [None, "", "interactive", "STDIN", True, []])
def test_invalid_cuda_gdb_mode_rejected(cfg, mode):
    cfg.timeout_dump_cuda_gdb_mode = mode
    with pytest.raises(ValueError, match="timeout_dump_cuda_gdb_mode"):
        validate_config(cfg)


def test_cuda_gdb_mode_cli(cfg):
    from nvidia_resiliency_ext.fault_tolerance import launcher

    parser = launcher.get_args_parser()
    defaults = FaultToleranceConfig.from_args(parser.parse_args(["train.py"]))
    assert defaults.timeout_dump_cuda_gdb_mode == "batch"
    config = FaultToleranceConfig.from_args(
        parser.parse_args(["--ft-timeout-dump-cuda-gdb-mode=stdin", "train.py"])
    )
    assert config.timeout_dump_cuda_gdb_mode == "stdin"
    assert not config.timeout_dump_enabled
    with pytest.raises(SystemExit):
        parser.parse_args(["--ft-timeout-dump-cuda-gdb-mode=interactive", "train.py"])
    cfg.timeout_dump_cuda_gdb_mode = "stdin"
    validate_config(cfg)
    cfg.timeout_dump_cuda_gdb_mode = "invalid"
    with pytest.raises(ValueError, match="timeout_dump_cuda_gdb_mode"):
        validate_config(cfg)


@pytest.mark.parametrize("character", ["\n", "\r"])
@pytest.mark.parametrize("symlink", [False, True])
@pytest.mark.parametrize("mode", ["batch", "stdin"])
def test_command_path_newlines_are_validated_at_startup(cfg, tmp_path, character, symlink, mode):
    cfg.timeout_dump = "cuda-gdb"
    cfg.timeout_dump_cuda_gdb = sys.executable
    cfg.timeout_dump_cuda_gdb_mode = mode
    path = tmp_path / f"commands{character}quit.gdb"
    path.write_text("info cuda kernels\n")
    if symlink:
        link = tmp_path / "commands.gdb"
        link.symlink_to(path)
        path = link
    cfg.timeout_dump_cuda_gdb_command_file = str(path)
    if mode == "stdin":
        with pytest.raises(ValueError, match="single line"):
            validate_capture_dependencies(cfg)
    else:
        validate_capture_dependencies(cfg)
        command = ToolCapture(cfg, 123, "cuda-gdb").command()
        assert command[command.index("-x") + 1] == str(path.resolve())


def test_launcher_validates_dump_settings_before_startup():
    from nvidia_resiliency_ext.fault_tolerance import launcher

    parser = launcher.get_args_parser()
    options = [
        "--ft-nvrx-logfile=/shared/nvrx/123/logs/launcher.log",
        "--ft-per-cycle-applog-prefix=/shared/train.log",
        "--ft-timeout-dump-enabled=true",
        "--ft-timeout-dump=pyspy",
        "--ft-timeout-dump-capture-timeout=0",
    ]
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(launcher, "determine_local_world_size") as local_world_size,
        patch.object(launcher, "_start_diagnostic_collector") as collector,
    ):
        with pytest.raises(ValueError, match="timeout_dump_capture_timeout"):
            launcher.config_from_args(parser.parse_args(options + ["train.py"]))
        local_world_size.assert_not_called()
        collector.assert_not_called()
        # Disabled dumps are not validated.
        launcher._validate_args(
            parser.parse_args(options + ["--ft-timeout-dump-enabled=false", "train.py"])
        )


@pytest.mark.parametrize("exit_code", [0, 7])
def test_stdin_capture_preserves_commands_errors_and_exit_code(cfg, tmp_path, exit_code):
    fake_tool(
        cfg,
        tmp_path,
        "import os, stat, sys\n"
        "assert '--batch' not in sys.argv and '-x' not in sys.argv\n"
        "assert stat.S_ISREG(os.fstat(0).st_mode)\n"
        "print(sys.stdin.read(), end='')\n"
        "print('example dump command error', file=sys.stderr)\n"
        f"sys.exit({exit_code})\n",
        "cuda-gdb",
    )
    commands = tmp_path / 'custom "quoted" \\ commands.gdb'
    commands.write_text("info cuda kernels\n")
    cfg.timeout_dump_cuda_gdb_command_file = str(commands)
    cfg.timeout_dump_capture_timeout = 2
    capture = ToolCapture(cfg, os.getpid(), "cuda-gdb")
    inputs = []
    popen = subprocess.Popen

    def launch(*args, **kwargs):
        inputs.append(kwargs["stdin"])
        return popen(*args, **kwargs)

    with patch(
        "nvidia_resiliency_ext.fault_tolerance.timeout_dump.capture.subprocess.Popen", launch
    ):
        asyncio.run(capture.run())
    assert inputs[0].closed
    assert capture.process.returncode == exit_code
    assert capture.status == ("completed" if exit_code == 0 else "exit_7")
    assert capture.stdin_commands().encode() in capture.output
    assert f"source {commands}\ndetach\nquit\n".encode() in capture.output
    assert b"example dump command error" in capture.output
    path = capture.publish_local(RankInfo(42, 0, "node42", os.getpid()), 0, "step")
    result = path.read_text()
    assert "mode=stdin" in result
    assert f"exit_code={exit_code}" in result
    DiagnosticCollector(cfg, destination(cfg), "node42").tick()
    assert (destination(cfg) / "cycle0/node42_cuda-gdb.txt").read_text() == result


@pytest.mark.parametrize("failure", ["popen", "commands"])
def test_stdin_launch_failure_closes_memfd(cfg, failure):
    cfg.timeout_dump_cuda_gdb_mode = "stdin"
    cfg.timeout_dump_cuda_gdb = "/missing/cuda-gdb"
    capture = ToolCapture(cfg, os.getpid(), "cuda-gdb")
    descriptors = []
    create = os.memfd_create

    def memfd(*args, **kwargs):
        fd = create(*args, **kwargs)
        descriptors.append(fd)
        return fd

    with patch("os.memfd_create", memfd), contextlib.ExitStack() as stack:
        if failure == "commands":
            stack.enter_context(
                patch.object(
                    capture, "stdin_commands", side_effect=OSError("cannot prepare commands")
                )
            )
        asyncio.run(capture.run())
    assert capture.status.startswith("error:")
    assert capture.process is None
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


@pytest.mark.parametrize(
    "tool,mode", [("gdb", "batch"), ("cuda-gdb", "batch"), ("cuda-gdb", "stdin")]
)
def test_capture_command_does_not_access_shared_command_file(cfg, tool, mode):
    cfg.timeout_dump = tool
    cfg.timeout_dump_cuda_gdb_mode = mode
    setattr(cfg, "timeout_dump_" + tool.replace("-", "_"), sys.executable)
    validate_capture_dependencies(cfg)
    expected = getattr(cfg, "timeout_dump_" + tool.replace("-", "_") + "_command_file")
    # Shared storage may be unavailable after startup. Command construction in
    # the monitor must leave filesystem access to the diagnostic subprocess.
    with (
        patch.object(Path, "resolve", side_effect=AssertionError("capture resolved shared path")),
        patch.object(Path, "stat", side_effect=AssertionError("capture stat'ed shared path")),
        patch.object(Path, "open", side_effect=AssertionError("capture opened shared path")),
    ):
        capture = ToolCapture(cfg, 123, tool)
        command = capture.command()
        if mode == "stdin":
            assert f"source {expected}\n" in capture.stdin_commands()
    if mode == "batch":
        assert command[command.index("-x") + 1] == expected


def test_cuda_environment_matches_worker_only_selected_fields(cfg):
    capture = ToolCapture(cfg, os.getpid(), "cuda-gdb")
    from unittest.mock import mock_open

    with (
        patch.dict(os.environ, {"TMPDIR": "/monitor", "CUDA_VISIBLE_DEVICES": "7"}),
        patch("builtins.open", mock_open(read_data=b"TMPDIR=/worker\0SECRET=value\0")),
    ):
        env = capture.environment()
    assert env["TMPDIR"] == "/worker"
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "SECRET" not in env


def test_collector_groups_tool_outputs_separately(cfg):
    directory = staging_dir(cfg) / "cycle0"
    directory.mkdir()
    for tool in ("pyspy", "gdb", "cuda-gdb"):
        (directory / f"rank0_pid100_{tool}.txt.ready").write_text(tool)
    collector = DiagnosticCollector(cfg, destination(cfg), "node42")
    collector.tick()
    for tool in ("pyspy", "gdb", "cuda-gdb"):
        assert (destination(cfg) / "cycle0" / f"node42_{tool}.txt").read_text() == tool
    assert not list(directory.glob("*.ready"))


@pytest.mark.parametrize("tool", ["gdb", "cuda-gdb"])
def test_command_files_are_independent(cfg, tmp_path, tool):
    path = tmp_path / "custom commands.gdb"
    path.write_text("info sharedlibrary\ninfo proc mappings\n")
    setattr(cfg, "timeout_dump_" + tool.replace("-", "_") + "_command_file", str(path))
    command = ToolCapture(cfg, 123, tool).command()
    assert command[command.index("-x") + 1] == str(path)
    assert command.index("attach 123") < command.index("-x") < command.index("detach")
    other = "cuda-gdb" if tool == "gdb" else "gdb"
    assert debugger_command_file(cfg, other).name == other + ".gdb"
    assert ToolCapture(cfg, 123, "pyspy").command() == [sys.executable, "dump", "--pid", "123"]


@pytest.mark.parametrize("tool", ["gdb", "cuda-gdb"])
@pytest.mark.parametrize("bad_path", [None, "missing", "directory", "", True])
def test_bad_command_file_fails_before_collector_starts(cfg, tmp_path, tool, bad_path):
    from nvidia_resiliency_ext.fault_tolerance import launcher

    cfg.timeout_dump = tool
    setattr(cfg, "timeout_dump_" + tool.replace("-", "_"), sys.executable)
    if bad_path == "missing":
        bad_path = str(tmp_path / "missing.gdb")
    elif bad_path == "directory":
        bad_path = str(tmp_path)
    setattr(cfg, "timeout_dump_" + tool.replace("-", "_") + "_command_file", bad_path)
    with patch(
        "nvidia_resiliency_ext.fault_tolerance.timeout_dump.DiagnosticCollectorProcess"
    ) as process:
        with pytest.raises(ValueError, match="command file"):
            launcher._start_diagnostic_collector(cfg, "/shared/nvrx/123/logs/launcher.log")
        process.assert_not_called()


def test_unused_command_file_is_not_validated(cfg):
    cfg.timeout_dump_gdb_command_file = "/missing.gdb"
    validate_capture_dependencies(cfg)  # pyspy only


def test_command_files_cli_path_resolution(tmp_path, monkeypatch):
    from nvidia_resiliency_ext.fault_tolerance import launcher

    cuda_commands = tmp_path / "cuda.gdb"
    cuda_commands.write_text("info cuda kernels\n")
    (tmp_path / "cpu.gdb").write_text("info threads\n")
    monkeypatch.chdir(tmp_path)
    args = launcher.get_args_parser().parse_args(
        [
            "--ft-timeout-dump-enabled=true",
            "--ft-timeout-dump=gdb,cuda-gdb",
            f"--ft-timeout-dump-gdb={sys.executable}",
            f"--ft-timeout-dump-cuda-gdb={sys.executable}",
            "--ft-timeout-dump-gdb-command-file",
            "cpu.gdb",
            f"--ft-timeout-dump-cuda-gdb-command-file={cuda_commands}",
            "train.py",
        ]
    )
    merged = FaultToleranceConfig.from_args(args)
    validate_capture_dependencies(merged)
    assert merged.timeout_dump_gdb_command_file == str(tmp_path / "cpu.gdb")
    assert merged.timeout_dump_cuda_gdb_command_file == str(cuda_commands)


def test_custom_file_output_uses_existing_capture_and_upload(cfg, tmp_path):
    # A subprocess stand-in verifies argv and file/output plumbing without ptrace or a GPU.
    executable = tmp_path / "gdb"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "assert sys.argv[-2:] == ['-ex', 'detach']\n"
        "path = pathlib.Path(sys.argv[sys.argv.index('-x') + 1])\n"
        "print(path.read_text(), end='')\n"
    )
    executable.chmod(0o700)
    commands = tmp_path / "custom commands.gdb"
    commands.write_text("info sharedlibrary\ninfo proc mappings\n")
    cfg.timeout_dump = "gdb"
    cfg.timeout_dump_gdb = str(executable)
    cfg.timeout_dump_gdb_command_file = str(commands)
    cfg.timeout_dump_capture_timeout = 2
    validate_capture_dependencies(cfg)
    capture = asyncio.run(ToolCapture(cfg, 123, "gdb").run())
    assert capture.status == "success"
    capture.publish_local(RankInfo(42, 0, "node42", 123), 0, "step_timeout")
    DiagnosticCollector(cfg, destination(cfg), "node42").tick()
    result = (destination(cfg) / "cycle0/node42_gdb.txt").read_text()
    assert "info sharedlibrary\ninfo proc mappings" in result
    assert "global_rank=42 local_rank=0 pid=123" in result
    commands.write_text("info registers\n")
    capture = asyncio.run(ToolCapture(cfg, 124, "gdb").run())
    assert capture.status == "success"
    assert capture.output == b"info registers\n"
