# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle, ownership and logging contracts for the colocated control supervisor."""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvidia_resiliency_ext.fault_tolerance import (
    child_process,
    control_supervisor,
    job_controller,
    launcher,
)
from nvidia_resiliency_ext.fault_tolerance.child_process import spawn_child
from nvidia_resiliency_ext.fault_tolerance.config import FaultToleranceConfig
from nvidia_resiliency_ext.fault_tolerance.control_logging import ControlLogCapture, agent_log_path
from nvidia_resiliency_ext.fault_tolerance.utils import (
    RDZV_SHUTDOWN_REASON_FAILURE,
    RDZV_SHUTDOWN_REASON_GRACEFUL,
    RDZV_SHUTDOWN_REASON_NO_PROGRESS,
)


def _args(**values):
    args = argparse.Namespace(rdzv_conf="", ft_no_restart_exit_code=None, term_timeout=0.2)
    vars(args).update(values)
    return args


@pytest.mark.parametrize(
    "path,expected",
    [
        (None, None),
        ("/logs/nvrx.log", "/logs/nvrx_agent.log"),
        ("/logs/nvrx", "/logs/nvrx_agent.log"),
    ],
)
def test_agent_log_path(path, expected):
    assert agent_log_path(path) == expected


def test_capture_merges_children_and_flushes_partial_lines(tmp_path):
    path = tmp_path / "control.log"
    capture = ControlLogCapture(str(path))
    children = []
    try:
        for name in ("control", "root", "leaf"):
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import os; os.write(1, b'%s stdout\\n'); os.write(2, b'%s final')"
                    % (name, name),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            children.append(child)
            capture.attach(child.stdout)
        for child in children:
            assert child.wait(timeout=10) == 0
    finally:
        capture.close()
    content = path.read_text()
    for name in ("control", "root", "leaf"):
        assert content.count(f"{name} stdout\n") == 1
        assert content.count(f"{name} final") == 1
    assert not capture._reader.is_alive()
    assert not capture._writer.is_alive()


def test_capture_closes_even_if_a_producer_keeps_its_pipe_open(tmp_path):
    path = tmp_path / "control.log"
    capture = ControlLogCapture(str(path))
    read_fd, write_fd = os.pipe()
    try:
        capture.attach(os.fdopen(read_fd, "rb", buffering=0))
        os.write(write_fd, b"final partial line")
        capture.close(timeout=1)
        assert path.read_bytes() == b"final partial line"
        assert not capture._reader.is_alive()
        assert not capture._writer.is_alive()
    finally:
        os.close(write_fd)


def test_failed_control_log_uses_original_stderr(capfd):
    capture = ControlLogCapture("/dev/full")
    read_fd, write_fd = os.pipe()
    capture.attach(os.fdopen(read_fd, "rb", buffering=0))
    os.write(write_fd, b"retained failure details\n")
    os.close(write_fd)
    capture.close()
    assert "retained failure details" in capfd.readouterr().err


def test_stalled_writer_and_full_console_do_not_block_child_output(tmp_path):
    capture = ControlLogCapture(str(tmp_path / "control.log"), buffer_chunks=2)
    console_read, console_write = os.pipe()
    os.set_blocking(console_write, False)
    os.close(capture._fallback_fd)
    capture._fallback_fd = console_write
    capture._fallback_nonblocking = True
    entered, release = threading.Event(), threading.Event()
    original_file = capture._file

    class SlowFile:
        def write(self, data):
            entered.set()
            assert release.wait(10)
            return original_file.write(data)

        def close(self):
            original_file.close()

    capture._file = SlowFile()
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os; os.write(1, b'x'*262144)",
        ],
        stdout=subprocess.PIPE,
    )
    try:
        capture.attach(child.stdout)
        assert entered.wait(5)
        # More than the file buffer and kernel pipe combined, while disk I/O is stalled.
        assert child.wait(timeout=5) == 0
    finally:
        release.set()
        child.wait(timeout=5)
        capture.close()
    assert b"using original stderr" in os.read(console_read, 65536)
    os.close(console_read)


@pytest.mark.parametrize("mode", ["stdin", "batch"])
def test_private_child_configuration_preserves_typed_args_and_literal_argv(
    tmp_path, monkeypatch, mode
):
    (tmp_path / "config_probe.py").write_text(
        "import json, os\n"
        "from nvidia_resiliency_ext.fault_tolerance.child_process import read_child_args\n"
        "from nvidia_resiliency_ext.fault_tolerance.timeout_dump.cuda_gdb import Mode\n"
        "args = vars(read_child_args())\n"
        "mode = args.pop('timeout_dump.cuda-gdb.mode')\n"
        "assert type(mode) is Mode\n"
        "extra_tuple = args.pop('extra_tuple')\n"
        "assert isinstance(extra_tuple, tuple)\n"
        "print(json.dumps({'args': args, 'mode': mode.value, 'extra_tuple': extra_tuple, "
        "'resource': os.environ.get('OTEL_RESOURCE_ATTRIBUTES')}))\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    resource = "nv.dl.job.uuid=job-test,example.attribute=preserved"
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", resource)
    args = launcher.parse_args(
        [
            f"--ft-timeout-dump-cuda-gdb-mode={mode}",
            "train.py",
            "a b",
            "$(touch /should-not-exist)",
            "--x='quoted'",
            "α",
        ]
    )
    args.extra_tuple = ("first", "second")
    child = spawn_child("config_probe", args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = child.communicate(timeout=20)
    assert child.returncode == 0, stderr.decode()
    expected_args = vars(args).copy()
    expected_args.pop("timeout_dump.cuda-gdb.mode")
    expected_args.pop("extra_tuple")
    assert json.loads(stdout) == {
        "args": expected_args,
        "mode": mode,
        "extra_tuple": list(args.extra_tuple),
        "resource": resource,
    }
    assert os.environ["OTEL_RESOURCE_ATTRIBUTES"] == resource


@pytest.mark.parametrize(
    "agent_code,control_code,expected",
    [
        (0, 0, 0),
        (0, 42, 42),
        (23, None, 23),
        (None, 77, 77),
        (-signal.SIGTERM, None, 128 + signal.SIGTERM),
        (None, -signal.SIGKILL, 128 + signal.SIGKILL),
    ],
)
def test_generation_exit_code_comes_from_children(agent_code, control_code, expected):
    owner = control_supervisor.ControlSupervisor(_args())
    owner.node_agent = Mock(poll=Mock(return_value=agent_code))
    owner.job_controller = Mock(poll=Mock(return_value=control_code))
    assert owner._monitor() == expected


@pytest.mark.parametrize("kind", ["grpc", "attribution"])
@pytest.mark.parametrize("code,expected", [(0, 1), (37, 37), (-signal.SIGKILL, 137)])
def test_auxiliary_service_exit_is_fatal(kind, code, expected):
    owner = control_supervisor.ControlSupervisor(_args())
    owner.node_agent = Mock(poll=Mock(return_value=None))
    owner.job_controller = Mock(poll=Mock(return_value=None))
    child = Mock(poll=Mock(return_value=code))
    if kind == "grpc":
        owner.grpc_processes = [child]
    else:
        owner.attribution = Mock(process=child)
    assert owner._monitor() == expected


@pytest.mark.parametrize("first", ["node_agent", "job_controller"])
def test_completion_waits_for_both_critical_children(first, monkeypatch):
    owner = control_supervisor.ControlSupervisor(_args())
    first_child = Mock(poll=Mock(return_value=0))
    other = "job_controller" if first == "node_agent" else "node_agent"
    second_child = Mock(poll=Mock(return_value=None))
    setattr(owner, first, first_child)
    setattr(owner, other, second_child)
    sleep = Mock(side_effect=lambda _: setattr(second_child.poll, "return_value", 0))
    monkeypatch.setattr(control_supervisor.time, "sleep", sleep)
    assert owner._monitor() == 0
    sleep.assert_called_once()


@pytest.mark.parametrize("first", ["node_agent", "job_controller"])
def test_monitor_remains_cancellable_after_one_child_finishes(first, monkeypatch):
    owner = control_supervisor.ControlSupervisor(_args())
    owner.node_agent = Mock(poll=Mock(return_value=0 if first == "node_agent" else None))
    owner.job_controller = Mock(poll=Mock(return_value=0 if first == "job_controller" else None))
    sleep = Mock(side_effect=lambda _: setattr(owner, "signum", signal.SIGTERM))
    monkeypatch.setattr(control_supervisor.time, "sleep", sleep)
    with pytest.raises(control_supervisor._ShutdownRequested):
        owner._monitor()
    sleep.assert_called_once()


def test_cleanup_preserves_first_failure_and_disables_restarts(monkeypatch):
    owner = control_supervisor.ControlSupervisor(_args())
    monkeypatch.setattr(owner, "_start", Mock())
    monkeypatch.setattr(owner, "_monitor", Mock(return_value=42))
    monkeypatch.setattr(control_supervisor, "_set_subreaper", Mock(return_value=False))

    def cleanup():
        assert owner.stopping
        owner.signum = signal.SIGTERM

    monkeypatch.setattr(owner, "_cleanup", cleanup)
    assert owner.run() == 42


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT])
def test_termination_signal_disables_restarts_and_restores_handlers(monkeypatch, signum):
    owner = control_supervisor.ControlSupervisor(_args())
    previous = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)
    }
    monkeypatch.setattr(owner, "_start", lambda: signal.getsignal(signum)(signum, None))
    cleanup = Mock(side_effect=lambda: owner.stopping or pytest.fail("restarts still enabled"))
    monkeypatch.setattr(owner, "_cleanup", cleanup)
    monkeypatch.setattr(control_supervisor, "_set_subreaper", Mock(return_value=False))
    assert owner.run() == 128 + signum
    cleanup.assert_called_once()
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


def test_cleanup_drains_agent_before_stopping_attribution_and_log_servers(monkeypatch):
    owner = control_supervisor.ControlSupervisor(_args(ft_log_server_graceful_shutdown_timeout=0.1))
    events = []
    owner.node_agent = Mock(poll=Mock(return_value=None))
    owner.node_agent.terminate.side_effect = lambda: events.append("agent-term")
    owner.job_controller = Mock(poll=Mock(return_value=None))
    owner.job_controller.terminate.side_effect = lambda: events.append("controller-term")
    owner.job_controller.wait.side_effect = lambda **_: events.append("controller-exit")
    owner.attribution = Mock()
    owner.attribution.stop.side_effect = lambda: events.append("attribution-stop")
    owner.grpc_processes = [Mock()]
    monkeypatch.setattr(owner, "_stop_node_agent", lambda **_: events.append("agent-drained"))
    monkeypatch.setattr(
        control_supervisor, "stop_grpc_log_servers", lambda *_: events.append("grpc-stop")
    )
    owner._cleanup()
    assert events == [
        "agent-term",
        "controller-term",
        "agent-drained",
        "controller-exit",
        "attribution-stop",
        "grpc-stop",
    ]


def test_control_and_agent_receive_same_timeout_defaults():
    args = launcher.parse_args(["--rdzv-conf", "read_timeout=21,is_host=true", "train.py"])
    control = control_supervisor._job_controller_args(args)
    agent = control_supervisor._node_agent_args(args, "host:123")
    assert control_supervisor._rdzv_config(control)["join_timeout"] == "300"
    assert control_supervisor._rdzv_config(control)["read_timeout"] == "21"
    assert control_supervisor._rdzv_config(agent)["is_host"] == "false"
    assert agent.rdzv_endpoint == "host:123"
    assert args.rdzv_conf == "read_timeout=21,is_host=true"


def test_standalone_keeps_embedded_control_without_a_supervisor(monkeypatch):
    args = launcher.parse_args(
        [
            "--standalone",
            "--rdzv-backend=etcd",
            "--rdzv-endpoint=unused.example:1234",
            "--rdzv-id=unused-run",
            "train.py",
        ]
    )
    new_run_id = Mock(return_value="standalone-run")
    monkeypatch.setattr(launcher.uuid, "uuid4", new_run_id)
    monkeypatch.setattr(launcher, "setup_logger", Mock())
    monkeypatch.setattr(launcher, "init_node_health_check", Mock())
    monkeypatch.setattr(launcher, "_register_ft_rdzv_handler", Mock())
    configuration = Mock(return_value=("config", "train.py", []))
    monkeypatch.setattr(launcher, "config_from_args", configuration)
    launch = Mock()
    monkeypatch.setattr(launcher, "elastic_launch", launch)
    supervisor = Mock()
    monkeypatch.setattr(control_supervisor, "run_control_host", supervisor)
    launcher.run(args)
    assert args.rdzv_backend == "c10d"
    assert args.rdzv_endpoint == "localhost:0"
    assert args.rdzv_id == "standalone-run"
    new_run_id.assert_called_once_with()
    supervisor.assert_not_called()
    configuration.assert_called_once()
    launch.assert_called_once_with(config="config", entrypoint="train.py")


@pytest.mark.parametrize(
    "endpoint,expected",
    [
        ("localhost", "localhost:29400"),
        ("[::1]", "[::1]:29400"),
    ],
)
def test_managed_control_preserves_c10d_endpoint_defaults(endpoint, expected):
    args = launcher.parse_args(["--rdzv-endpoint", endpoint, "train.py"])
    assert control_supervisor._job_controller_args(args).rdzv_endpoint == expected


def test_supervised_control_rejects_an_ephemeral_endpoint():
    args = launcher.parse_args(["--rdzv-endpoint=localhost:0", "train.py"])
    with pytest.raises(ValueError, match="fixed rendezvous port"):
        control_supervisor._job_controller_args(args)


@pytest.mark.parametrize("use_libuv", [True, False])
def test_control_tcp_store_honors_configured_backend(use_libuv):
    args = launcher.parse_args(
        [
            "--nnodes=1",
            "--rdzv-endpoint",
            "localhost:0",
            "--rdzv-conf",
            f"use_libuv={str(use_libuv).lower()}",
            "train.py",
        ]
    )
    store = job_controller._create_tcp_store(args, 5)
    assert store.libuvBackend is use_libuv


@pytest.mark.parametrize(
    "is_control_host_agent,rdzv_conf",
    [(False, "is_host=false"), (True, "is_host=true")],
)
def test_agent_launch_does_not_start_supervisor(monkeypatch, is_control_host_agent, rdzv_conf):
    args = launcher.parse_args(["--rdzv-conf", rdzv_conf, "train.py"])
    args._is_control_host_agent = is_control_host_agent
    launch = Mock()
    monkeypatch.setattr(launcher, "config_from_args", Mock(return_value=("config", "train.py", [])))
    monkeypatch.setattr(launcher, "elastic_launch", launch)
    monkeypatch.setattr(launcher, "setup_logger", Mock())
    monkeypatch.setattr(launcher, "init_node_health_check", Mock())
    monkeypatch.setattr(launcher, "_register_ft_rdzv_handler", Mock())
    owner = Mock()
    monkeypatch.setattr(control_supervisor, "run_control_host", owner)
    launcher.run(args)
    launch.assert_called_once_with(config="config", entrypoint="train.py")
    owner.assert_not_called()


def test_managed_control_does_not_spawn_duplicate_services(monkeypatch, tmp_path):
    args = launcher.parse_args(
        [
            "--ft-per-cycle-applog-prefix",
            str(tmp_path / "train.log"),
            "--ft-attribution-endpoint",
            "localhost",
            "train.py",
        ]
    )
    service = Mock()
    monkeypatch.setattr(job_controller, "AttributionService", Mock(return_value=service))
    result = job_controller._start_controller_resources(
        args, FaultToleranceConfig.from_args(args), node_id="control.example"
    )
    assert result.attribution_service is service
    service.start_poller.assert_called_once_with(node_id="control.example")


@pytest.mark.parametrize(
    "reason,expected",
    [
        (RDZV_SHUTDOWN_REASON_GRACEFUL, 0),
        (RDZV_SHUTDOWN_REASON_NO_PROGRESS, 77),
        (RDZV_SHUTDOWN_REASON_FAILURE, 1),
        (None, 1),
    ],
)
def test_managed_controller_classifies_exit_status(
    monkeypatch,
    reason,
    expected,
):
    args = launcher.parse_args(
        [
            "--rdzv-endpoint",
            "127.0.0.1:0",
            "--rdzv-conf",
            "exit_timeout=0,replacement_group_size=2",
            "train.py",
        ]
    )
    args.ft_no_restart_exit_code = 77
    state = Mock()
    state.close_current_round_as_host.side_effect = job_controller.RendezvousClosedError("done")
    state.get_shutdown_reason.return_value = reason
    state_class = Mock(return_value=state)
    monkeypatch.setattr(job_controller, "_RendezvousBarrierState", state_class)
    result = job_controller._run_control_rendezvous_loop(
        args,
        FaultToleranceConfig.from_args(args),
        SimpleNamespace(port=12345),
        job_controller.ControllerResources(),
        threading.Event(),
    )
    assert result == expected
    assert state_class.call_args.kwargs["replacement_group_size"] == 2


def test_supervisor_reaps_session_detached_workers_after_agent_crash():
    # Isolate Linux subreaper state from pytest and any unrelated test subprocesses.
    script = r'''
import os, subprocess, sys
import psutil
from argparse import Namespace
from nvidia_resiliency_ext.fault_tolerance.control_supervisor import ControlSupervisor, _set_subreaper
previous = _set_subreaper(True)
owner = ControlSupervisor(Namespace(rdzv_conf="", ft_no_restart_exit_code=None, term_timeout=.2))
sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
owner.job_controller = sibling
try:
    source = "import subprocess,sys,os; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); print(p.pid,flush=True); os._exit(2)"
    owner.node_agent = subprocess.Popen([sys.executable, "-c", source], stdout=subprocess.PIPE, text=True)
    worker_pid = int(owner.node_agent.stdout.readline())
    assert owner.node_agent.wait(timeout=10) == 2
    owner._stop_node_agent()
    assert not psutil.pid_exists(worker_pid), worker_pid
    assert sibling.poll() is None, "agent cleanup killed control"
finally:
    sibling.terminate()
    sibling.wait(timeout=10)
    _set_subreaper(previous)
'''
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=25
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("code,expected", [(0, 0), (42, 42), (77, 77), (-signal.SIGKILL, 137)])
def test_control_only_exit_status_without_a_node_agent(code, expected):
    owner = control_supervisor.ControlSupervisor(_args(ft_control_only=True))
    owner.job_controller = Mock(poll=Mock(return_value=code))
    assert owner.node_agent is None
    assert owner._monitor() == expected


def test_control_only_waits_for_the_controller(monkeypatch):
    owner = control_supervisor.ControlSupervisor(_args(ft_control_only=True))
    owner.job_controller = Mock(poll=Mock(return_value=None))
    monkeypatch.setattr(
        control_supervisor.time,
        "sleep",
        lambda _: setattr(owner.job_controller.poll, "return_value", 0),
    )
    assert owner._monitor() == 0


@pytest.mark.parametrize("control_only", [False, True])
def test_control_host_dispatch_never_initializes_workers(monkeypatch, control_only):
    options = ["--rdzv-endpoint=localhost:29500", "--nproc-per-node=gpu"]
    options += ["--ft-control-only"] if control_only else ["train.py"]
    args = launcher.parse_args(options)
    owner = Mock(return_value=0)
    monkeypatch.setattr(control_supervisor, "run_control_host", owner)
    configure_workers = Mock(side_effect=AssertionError("worker setup in supervisor dispatch"))
    monkeypatch.setattr(launcher, "config_from_args", configure_workers)
    monkeypatch.setattr(launcher, "determine_local_world_size", configure_workers)
    with pytest.raises(SystemExit) as exited:
        launcher.run(args)
    assert exited.value.code == 0
    assert args.training_script == (None if control_only else "train.py")
    owner.assert_called_once_with(args)
    configure_workers.assert_not_called()


def test_control_only_requires_ownership_of_the_endpoint():
    args = launcher.parse_args(
        ["--ft-control-only", "--rdzv-endpoint=localhost:29500", "--rdzv-conf=is_host=false"]
    )
    with pytest.raises(ValueError, match="rendezvous host"):
        launcher.run(args)


@pytest.mark.parametrize(
    "options,expected",
    [
        ([], "training command is required"),
        (
            ["--ft-control-only", "--rdzv-endpoint=localhost:29500", "train.py"],
            "does not accept a training command",
        ),
        (
            ["--ft-control-only", "--standalone", "--rdzv-endpoint=localhost:29500"],
            "cannot be combined",
        ),
        (["--ft-control-only"], "requires --rdzv-endpoint"),
        (
            [
                "--ft-control-only",
                "--rdzv-endpoint=localhost:29500",
                "--ft-cycle-info-dir=/tmp/cycles",
            ],
            "requires --ft-cycle-info-job-id",
        ),
        (["--ft-cycle-info-job-id=123", "train.py"], "only supported with --ft-control-only"),
        (
            ["--ft-control-only", "--rdzv-endpoint=localhost:29500", "--ft-cycle-info-job-id= "],
            "must be non-empty",
        ),
    ],
)
def test_launch_mode_validation(options, expected, capsys):
    with pytest.raises(SystemExit) as exited:
        launcher.parse_args(options)
    assert exited.value.code == 2
    assert expected in capsys.readouterr().err


def test_control_only_start_spawns_the_controller_without_an_agent(monkeypatch):
    args = launcher.parse_args(["--ft-control-only", "--rdzv-endpoint=localhost:29500"])
    owner = control_supervisor.ControlSupervisor(args)
    capture = Mock()
    controller = Mock()
    spawn = Mock(return_value=controller)
    monkeypatch.setattr(control_supervisor, "ControlLogCapture", Mock(return_value=capture))
    monkeypatch.setattr(control_supervisor, "setup_logger", Mock())
    monkeypatch.setattr(control_supervisor, "spawn_child", spawn)
    owner._start()
    assert spawn.call_count == 1
    assert spawn.call_args.args[0].endswith(".job_controller")
    assert owner.job_controller is controller
    assert owner.node_agent is None
    capture.attach.assert_called_once_with(controller.stdout)


def test_child_shutdown_waits_without_killing_a_finished_child():
    child = Mock(wait=Mock(return_value=0))
    assert child_process.wait_or_kill_child(child, 3)
    child.wait.assert_called_once_with(timeout=3)
    child.kill.assert_not_called()


def test_child_shutdown_escalates_and_bounds_the_final_wait():
    child = Mock(wait=Mock(side_effect=[subprocess.TimeoutExpired("child", 3), -9]))
    assert child_process.wait_or_kill_child(child, 3, kill_timeout=0.5)
    child.kill.assert_called_once()
    assert [call.kwargs["timeout"] for call in child.wait.call_args_list] == [3, 0.5]


def test_child_shutdown_survivor_is_reported_without_blocking(monkeypatch):
    child = Mock(wait=Mock(side_effect=subprocess.TimeoutExpired("child", 1)))
    error = Mock()
    monkeypatch.setattr(child_process.logger, "error", error)
    assert not child_process.wait_or_kill_child(child, 1, kill_timeout=0.25)
    assert [call.kwargs["timeout"] for call in child.wait.call_args_list] == [1, 0.25]
    error.assert_called_once()
    assert "still alive" in error.call_args.args[0]


def test_child_shutdown_can_kill_the_agent_process_group(monkeypatch):
    child = Mock(pid=12345, wait=Mock(side_effect=[subprocess.TimeoutExpired("child", 1), -9]))
    killpg = Mock()
    monkeypatch.setattr(child_process.os, "killpg", killpg)
    assert child_process.wait_or_kill_child(child, 1, process_group=True)
    killpg.assert_called_once_with(12345, signal.SIGKILL)
    child.kill.assert_not_called()


@pytest.mark.parametrize("finishes_cleanup", [True, False])
def test_agent_shutdown_allows_cleanup_and_still_bounds_escalation(monkeypatch, finishes_cleanup):
    owner = control_supervisor.ControlSupervisor(_args(term_timeout=2))
    child = Mock(pid=12345, poll=Mock(return_value=None))
    child.wait.side_effect = (
        [0] if finishes_cleanup else [subprocess.TimeoutExpired("agent", 17), -9]
    )
    owner.node_agent = child
    monkeypatch.setattr(owner, "_agent_descendants", lambda _: [])
    killpg = Mock()
    monkeypatch.setattr(child_process.os, "killpg", killpg)

    owner._stop_node_agent()

    child.terminate.assert_called_once()
    timeouts = [call.kwargs["timeout"] for call in child.wait.call_args_list]
    assert timeouts == ([17] if finishes_cleanup else [17, child_process.KILL_WAIT_TIMEOUT])
    if finishes_cleanup:
        killpg.assert_not_called()
    else:
        killpg.assert_called_once_with(child.pid, signal.SIGKILL)
    assert owner.args.term_timeout == 2


def test_agent_descendant_discovery_takes_one_snapshot_and_protects_services(monkeypatch):
    owner = control_supervisor.ControlSupervisor(_args())
    owner.node_agent = Mock(pid=10001, poll=Mock(return_value=None))
    owner.job_controller = Mock(pid=10002)
    owner.grpc_processes = [Mock(pid=10003)]

    def process(pid, parent):
        return SimpleNamespace(pid=pid, info={"pid": pid, "ppid": parent})

    root = os.getpid()
    snapshot = [
        process(10001, root),
        process(10002, root),
        process(10003, root),
        process(10004, 10001),
        process(10005, 10004),
        process(10006, root),
        process(10007, 10006),
        process(10008, 10002),
        process(10009, 10003),
        process(10010, 1),
    ]
    scan = Mock(return_value=iter(snapshot))
    monkeypatch.setattr(control_supervisor.psutil, "process_iter", scan)
    assert {p.pid for p in owner._agent_descendants(float("inf"))} == {10004, 10005, 10006, 10007}
    scan.assert_called_once_with(["pid", "ppid"])


def test_agent_descendant_discovery_observes_its_budget(monkeypatch):
    owner = control_supervisor.ControlSupervisor(_args())
    owner.node_agent = Mock(pid=10001, poll=Mock(return_value=2))
    monkeypatch.setattr(control_supervisor.psutil, "process_iter", lambda _: iter([Mock()]))
    monkeypatch.setattr(control_supervisor.time, "monotonic", lambda: 10)
    error = Mock()
    monkeypatch.setattr(control_supervisor.logger, "error", error)
    assert owner._agent_descendants(0) == []
    assert "snapshot is incomplete" in error.call_args.args[0]


def test_agent_cleanup_reports_surviving_descendants(monkeypatch):
    owner = control_supervisor.ControlSupervisor(_args())
    owner.node_agent = Mock(poll=Mock(return_value=2))
    survivor = Mock(pid=10004)
    monkeypatch.setattr(owner, "_agent_descendants", lambda _: [survivor])
    monkeypatch.setattr(control_supervisor.time, "monotonic", lambda: 10)
    wait = Mock(return_value=([], [survivor]))
    monkeypatch.setattr(control_supervisor.psutil, "wait_procs", wait)
    error = Mock()
    monkeypatch.setattr(control_supervisor.logger, "error", error)
    owner._stop_node_agent()
    survivor.kill.assert_called_once()
    assert wait.call_args.kwargs["timeout"] == 2
    error.assert_called_once_with("Descendants survived cleanup: PIDs=%s", [10004])
