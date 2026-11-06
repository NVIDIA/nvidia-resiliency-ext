# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay the same scheduler snapshots against the deployed Bash and Python publishers."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from nvidia_resiliency_ext.attribution.scheduler_exclusions import publisher as module
from nvidia_resiliency_ext.attribution.scheduler_exclusions.publisher import (
    PollError,
    PublisherConfig,
    SchedulerExclusionPublisher,
    run_slurm,
)

ROOT = Path(__file__).resolve().parents[2]
BASH_PRODUCER = ROOT / "examples/fault_tolerance/deployment/slurm/scheduler_segment_health.sh"
PACKAGE = "nvidia_resiliency_ext.services.scheduler_exclusions"


@dataclass
class Step:
    sinfo: str = "node03\n"
    squeue: str = "7|group7\n"
    hostlists: dict[str, str] = field(default_factory=lambda: {"group7": "node01\nnode03\n"})
    failed: str = ""
    age: int = 0
    restart: bool = False

    def output(self, argv):
        if argv[0] == self.failed:
            raise PollError("injected scheduler failure")
        if argv[0] == "scontrol":
            return self.hostlists.get(argv[-1], "")
        return getattr(self, argv[0])


def marker(directory, task="7"):
    return directory / f"segment_health_check.123.{task}"


def history(directory):
    path = directory / "segment_health_check_history.123.log"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def snapshot(directory):
    records = history(directory)
    for record in records:
        timestamp = record.pop("observed_at")
        assert timestamp is None or timestamp.endswith("Z")
    files = {
        path.name: path.read_bytes()
        for path in directory.glob("segment_health_check.*")
        if path.is_file()
    }
    return files, records


BASH_MOCKS = r'''
timeout() {
    printf '%s %s\n' "$2" "${*:3}" >> "$CALLS"
    shift; shift
    "$@"
}
sinfo() { [[ "$FAILED" != sinfo ]] || return 1; cat "$STEP/sinfo"; }
squeue() { [[ "$FAILED" != squeue ]] || return 1; cat "$STEP/squeue"; }
scontrol() { [[ "$FAILED" != scontrol ]] || return 1; cat "$STEP/$3"; }
'''


def replay_bash(root, steps, seed):
    directory = root / "decisions"
    directory.mkdir(parents=True)
    for name, value in seed.items():
        (directory / name).write_bytes(value)
    calls = root / "calls"
    calls.touch()
    env = dict(
        os.environ,
        SLURM_ARRAY_JOB_ID="123",
        SLURM_JOB_PARTITION="batch",
        NVRX_SEGMENT_HEALTH_CHECK_DIR=str(directory),
        NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS="1800",
        NVRX_SEGMENT_HEALTH_SLURM_CMD_TIMEOUT="30",
        CALLS=str(calls),
    )
    script = [f"source {shlex.quote(str(BASH_PRODUCER))}", BASH_MOCKS]
    for i, step in enumerate(steps):
        inputs = root / f"input{i}"
        inputs.mkdir()
        for name, value in {"sinfo": step.sinfo, "squeue": step.squeue, **step.hostlists}.items():
            (inputs / name).write_text(value)
        if step.restart:
            script.append("scheduler_segment_health_cleanup")
        script.extend(
            [
                "scheduler_segment_health_configure || exit 91",
                f"STEP={shlex.quote(str(inputs))}",
                f"FAILED={shlex.quote(step.failed)}",
                f"_SCHEDULER_SEGMENT_HEALTH_LAST_COMPLETE_POLL_AT=$((SECONDS-{step.age}))",
                "scheduler_segment_health_poll_once",
                "printf '%s\\n' \"$?\"",
                f'cp -R "$NVRX_SEGMENT_HEALTH_CHECK_DIR" {shlex.quote(str(root / f"snapshot{i}"))}',
            ]
        )
    script.append("scheduler_segment_health_cleanup")
    result = subprocess.run(
        ["/bin/bash", "-c", "\n".join(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    statuses = [int(line) for line in result.stdout.splitlines()]
    assert len(statuses) == len(steps), result.stderr
    return (
        statuses,
        [snapshot(root / f"snapshot{i}") for i in range(len(steps))],
        calls.read_text().splitlines(),
    )


def replay_python(root, steps, seed):
    directory = root / "decisions"
    directory.mkdir(parents=True)
    for name, value in seed.items():
        (directory / name).write_bytes(value)
    calls = []
    now = [0.0]
    current = [steps[0]]

    def runner(argv, timeout):
        calls.append(f"{timeout} {' '.join(argv)}")
        return current[0].output(argv)

    config = PublisherConfig("123", "batch", directory)
    publisher = SchedulerExclusionPublisher(config, runner=runner, clock=lambda: now[0])
    statuses, snapshots = [], []
    for step in steps:
        current[0] = step
        if step.restart:
            publisher = SchedulerExclusionPublisher(config, runner=runner, clock=lambda: now[0])
        now[0] = publisher._last_complete_poll + step.age
        statuses.append(0 if publisher.poll_once() else 1)
        snapshots.append(snapshot(directory))
    return statuses, snapshots, calls


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param([Step(sinfo="")], id="healthy-fast-path"),
        pytest.param(
            [Step(), Step(), Step(sinfo="node01\nnode03\n"), Step(sinfo=""), Step()],
            id="publish-update-clear-reexclude",
        ),
        pytest.param(
            [Step(), Step(sinfo="", squeue=""), Step(sinfo="", squeue="")], id="two-misses-retire"
        ),
        pytest.param([Step(), Step(squeue=""), Step()], id="transient-missing-task"),
        pytest.param(
            [Step(), Step(squeue=""), Step(failed="sinfo"), Step(squeue="")],
            id="failed-poll-does-not-reset-miss",
        ),
        pytest.param(
            [Step(), Step(restart=True), Step(sinfo="", restart=True)],
            id="restart-reconciles-existing-markers",
        ),
        pytest.param(
            [Step(), Step(squeue=""), Step(squeue="", restart=True)], id="restart-resets-miss-count"
        ),
        pytest.param(
            [Step(), Step(failed="sinfo", age=1799), Step(failed="sinfo", age=1800), Step()],
            id="stale-deadline-and-recovery",
        ),
        pytest.param([Step(), Step(failed="squeue")], id="squeue-failure"),
        pytest.param([Step(), Step(failed="scontrol")], id="hostlist-failure"),
        pytest.param([Step(sinfo="unrelated-node\n")], id="outside-allocation"),
        pytest.param(
            [Step(sinfo=" node03 \n", squeue=" 7 | group7 \n", hostlists={"group7": " node03 \n"})],
            id="field-padding",
        ),
        pytest.param(
            [Step(sinfo="node01\nnode03\nnode03\n", squeue="12|group7\n7|group7\n")],
            id="ordered-tasks-and-deduplicated-nodes",
        ),
        pytest.param([Step(), Step(sinfo="node03")], id="incomplete-sinfo"),
        pytest.param([Step(), Step(squeue="7|group7")], id="incomplete-squeue"),
        pytest.param([Step(), Step(hostlists={"group7": "node03"})], id="incomplete-hostlist"),
        pytest.param([Step(), Step(sinfo="node03|bad\n")], id="invalid-node"),
        pytest.param([Step(), Step(squeue="7|group7\n7|group7\n")], id="duplicate-task"),
        pytest.param([Step(), Step(squeue="bad-task|group7\n")], id="invalid-task"),
        pytest.param([Step(), Step(squeue="7|group7|extra\n")], id="invalid-row"),
        pytest.param([Step(), Step(squeue="7|node{1,2}\n")], id="invalid-hostlist"),
        pytest.param([Step(), Step(hostlists={"group7": ""})], id="empty-hostlist"),
        pytest.param(
            [
                Step(
                    sinfo="".join(f"node{i}\n" for i in range(17)),
                    hostlists={"group7": "".join(f"node{i}\n" for i in range(17))},
                ),
                Step(sinfo="", hostlists={"group7": "".join(f"node{i}\n" for i in range(17))}),
            ],
            id="mass-recovery-clears-without-verification",
        ),
    ],
)
def test_bash_python_parity(tmp_path, steps):
    assert replay_python(tmp_path / "python", steps, {}) == replay_bash(
        tmp_path / "bash", steps, {}
    )


@pytest.mark.parametrize("value", [b"node03", b"", b"node03\n", b"node03\n999|K|rogue", b"\xff"])
@pytest.mark.parametrize("sinfo", ["", "node03\n"])
def test_persisted_state_parity(tmp_path, value, sinfo):
    seed = {"segment_health_check.123.7": value, "segment_health_check.123.12.inactive": b"node12"}
    steps = [Step(sinfo=sinfo), Step(sinfo=sinfo)]
    assert replay_python(tmp_path / "python", steps, seed) == replay_bash(
        tmp_path / "bash", steps, seed
    )


@pytest.fixture
def publisher(tmp_path):
    step = Step()
    instance = SchedulerExclusionPublisher(
        PublisherConfig("123", "batch", tmp_path / "decisions"),
        runner=lambda argv, timeout: step.output(argv),
    )
    return instance


def test_publication_failure_does_not_block_other_tasks(publisher, monkeypatch):
    publisher._runner = lambda argv, timeout: Step(squeue="7|group7\n12|group7\n").output(argv)
    original = publisher._write_current

    def fail_one(task, value):
        if task == "7":
            raise OSError("write failed")
        original(task, value)

    monkeypatch.setattr(publisher, "_write_current", fail_one)
    assert not publisher.poll_once()
    assert marker(publisher.config.directory, "12").read_text() == "node03"
    assert [record["task_id"] for record in history(publisher.config.directory)] == [12]
    monkeypatch.setattr(publisher, "_write_current", original)
    assert publisher.poll_once()
    assert [record["task_id"] for record in history(publisher.config.directory)] == [12, 7]


def test_completed_write_with_error_is_reconciled(publisher, monkeypatch):
    original = publisher._write_current

    def write_then_fail(task, value):
        original(task, value)
        raise OSError("late write failure")

    monkeypatch.setattr(publisher, "_write_current", write_then_fail)
    assert publisher.poll_once()
    assert len(history(publisher.config.directory)) == 1


def test_inactive_rename_failure_retries(publisher, monkeypatch):
    assert publisher.poll_once()
    publisher._runner = lambda argv, timeout: ""
    assert publisher.poll_once()
    original = Path.replace
    monkeypatch.setattr(
        Path, "replace", lambda *args: (_ for _ in ()).throw(OSError("rename failed"))
    )
    assert not publisher.poll_once()
    assert marker(publisher.config.directory).read_text() == "node03"
    monkeypatch.setattr(Path, "replace", original)
    assert publisher.poll_once()
    assert not marker(publisher.config.directory).exists()
    assert history(publisher.config.directory)[-1]["reason"] == "task_inactive"


def test_history_failure_does_not_fail_publication(publisher):
    publisher.history_path.mkdir()
    assert publisher.poll_once()
    assert marker(publisher.config.directory).read_text() == "node03"


def test_missing_directory_does_not_query_or_recreate(publisher):
    publisher.config.directory.rmdir()
    publisher._runner = lambda *args: pytest.fail("must not query scheduler")
    assert not publisher.poll_once()
    assert not publisher.config.directory.exists()


def test_timestamp_failure_still_records_event(publisher, monkeypatch):
    monkeypatch.setattr(publisher, "_timestamp", lambda: None)
    assert publisher.poll_once()
    assert history(publisher.config.directory)[0]["observed_at"] is None


def test_existing_marker_is_never_opened_with_truncate(publisher, monkeypatch):
    path = marker(publisher.config.directory)
    path.write_text("node03,node04")
    original = os.open
    flags_seen = []

    def observe(path, flags, *args):
        flags_seen.append(flags)
        return original(path, flags, *args)

    monkeypatch.setattr(os, "open", observe)
    publisher._write_current("7", "node03")
    assert path.read_bytes() == b"node03"
    assert flags_seen and all(not flags & os.O_TRUNC for flags in flags_seen)


def test_fifo_marker_cannot_block_publisher(publisher):
    os.mkfifo(marker(publisher.config.directory))
    assert not publisher.poll_once()


def test_stale_unknown_markers_are_cleared(publisher):
    marker(publisher.config.directory).write_bytes(b"node03\n999|K|rogue")
    publisher._clock = lambda: publisher._last_complete_poll + publisher.config.stale_seconds

    def fail(*args):
        raise PollError("unavailable")

    publisher._runner = fail
    assert not publisher.poll_once()
    assert marker(publisher.config.directory).read_bytes() == b""
    assert history(publisher.config.directory)[-1]["reason"] == "stale_decision_expired"


@pytest.mark.parametrize(
    "name,value",
    [
        ("SLURM_ARRAY_JOB_ID", "../wrong"),
        ("SLURM_ARRAY_JOB_ID", ""),
        ("SLURM_JOB_PARTITION", ""),
        ("NVRX_SEGMENT_HEALTH_CHECK_DIR", "relative"),
        ("NVRX_SEGMENT_HEALTH_CHECK_DIR", ""),
        ("NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS", "0"),
        ("NVRX_SEGMENT_HEALTH_SLURM_CMD_TIMEOUT", "-1"),
        ("NVRX_SEGMENT_HEALTH_SLURM_CMD_TIMEOUT", "1.5"),
    ],
)
def test_invalid_configuration(tmp_path, name, value):
    env = {
        "SLURM_ARRAY_JOB_ID": "123",
        "SLURM_JOB_PARTITION": "batch",
        "NVRX_SEGMENT_HEALTH_CHECK_DIR": str(tmp_path),
    }
    env[name] = value
    with pytest.raises(ValueError):
        PublisherConfig.from_env(env)


def test_slurm_runner_resolves_and_bounds_commands(monkeypatch):
    calls = []
    monkeypatch.setattr(module.shutil, "which", lambda command: f"/slurm/bin/{command}")

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="node03\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)
    assert run_slurm(["sinfo", "--Node"], 30) == "node03\n"
    assert calls[0][0] == ["/slurm/bin/sinfo", "--Node"]
    assert calls[0][1]["timeout"] == 30
    assert not calls[0][1].get("shell", False)


@pytest.mark.parametrize("failure", ["missing", "timeout", "exit", "oserror"])
def test_slurm_runner_reports_failures(monkeypatch, failure):
    monkeypatch.setattr(
        module.shutil, "which", lambda command: None if failure == "missing" else "/bin/sinfo"
    )

    def run(argv, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if failure == "oserror":
            raise OSError("could not start")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="failed")

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(PollError):
        run_slurm(["sinfo"], 30)


@pytest.fixture
def cli_environment(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sinfo = bin_dir / "sinfo"
    sinfo.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, time\n"
        "pathlib.Path(os.environ['CHILD_PID']).write_text(str(os.getpid()))\n"
        "with open(os.environ['CALLS'], 'a') as out: out.write('sinfo\\n')\n"
        "while pathlib.Path(os.environ['BLOCK']).exists(): time.sleep(0.02)\n"
    )
    sinfo.chmod(0o755)
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        SLURM_ARRAY_JOB_ID="123",
        SLURM_JOB_PARTITION="batch",
        NVRX_SEGMENT_HEALTH_CHECK_DIR=str(tmp_path / "decisions"),
        SEGMENT_HEALTH_POLL_INTERVAL="300",
        CALLS=str(tmp_path / "calls"),
        BLOCK=str(tmp_path / "block"),
        CHILD_PID=str(tmp_path / "child.pid"),
    )
    return env


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_stop_reaps_blocked_slurm_child(cli_environment, sig):
    Path(cli_environment["BLOCK"]).touch()
    child_file = Path(cli_environment["CHILD_PID"])
    directory = Path(cli_environment["NVRX_SEGMENT_HEALTH_CHECK_DIR"])
    directory.mkdir()
    marker(directory).write_text("node03")
    process = subprocess.Popen(
        [sys.executable, "-S", "-m", PACKAGE],
        env=cli_environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for(lambda: child_file.exists() and bool(child_file.read_text()), process)
        child_pid = int(child_file.read_text())
        process.send_signal(sig)
        assert process.wait(timeout=5) == 0
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert marker(directory).read_text() == "node03"
        assert not (directory / "segment_health_producer.123.pid").exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def test_cli_invalid_interval_does_not_create_artifacts(cli_environment):
    cli_environment["SEGMENT_HEALTH_POLL_INTERVAL"] = "0"
    result = subprocess.run(
        [sys.executable, "-S", "-m", PACKAGE],
        env=cli_environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 2
    assert not Path(cli_environment["NVRX_SEGMENT_HEALTH_CHECK_DIR"]).exists()


def wait_for(predicate, process):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        assert process.poll() is None, "publisher exited early"
        time.sleep(0.02)
    pytest.fail("publisher did not reach expected state")


def test_cli_once(cli_environment):
    result = subprocess.run(
        [sys.executable, "-S", "-m", PACKAGE, "--once"],
        env=cli_environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert Path(cli_environment["CALLS"]).read_text() == "sinfo\n"


def test_parent_death_stops_publisher_and_blocked_slurm_child(cli_environment, tmp_path):
    Path(cli_environment["BLOCK"]).touch()
    cli_environment["PUBLISHER_PID"] = str(tmp_path / "publisher.pid")
    publisher_file = Path(cli_environment["PUBLISHER_PID"])
    child_file = Path(cli_environment["CHILD_PID"])
    pidfile = (
        Path(cli_environment["NVRX_SEGMENT_HEALTH_CHECK_DIR"]) / "segment_health_producer.123.pid"
    )
    parent = subprocess.Popen(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import os, pathlib, subprocess, sys, time\n"
                f"p = subprocess.Popen([sys.executable, '-S', '-m', {PACKAGE!r}, "
                "'--parent-pid', str(os.getpid())], start_new_session=True)\n"
                "pathlib.Path(os.environ['PUBLISHER_PID']).write_text(str(p.pid))\n"
                "time.sleep(60)\n"
            ),
        ],
        env=cli_environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for(lambda: child_file.exists() and bool(child_file.read_text()), parent)
        assert pidfile.exists()
        child_pid = int(child_file.read_text())
        # The watchdog must tolerate a healthy parent through multiple checks.
        time.sleep(1.1)
        assert pidfile.exists()
        parent.kill()
        parent.wait(timeout=5)
        deadline = time.monotonic() + 5
        while pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not pidfile.exists()
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)
        if publisher_file.exists():
            try:
                os.killpg(int(publisher_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("successor", [False, True])
def test_serve_signals_during_poll_and_wait(cli_environment, successor):
    block = Path(cli_environment["BLOCK"])
    block.touch()
    calls = Path(cli_environment["CALLS"])
    pidfile = (
        Path(cli_environment["NVRX_SEGMENT_HEALTH_CHECK_DIR"]) / "segment_health_producer.123.pid"
    )
    process = subprocess.Popen(
        [sys.executable, "-S", "-m", PACKAGE],
        env=cli_environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for(lambda: calls.exists() and pidfile.exists(), process)
        assert pidfile.read_text().split()[1] == str(process.pid)
        process.send_signal(signal.SIGUSR1)
        time.sleep(0.1)
        block.unlink()
        wait_for(lambda: len(calls.read_text().splitlines()) >= 2, process)
        time.sleep(0.1)
        process.send_signal(signal.SIGUSR1)
        wait_for(lambda: len(calls.read_text().splitlines()) >= 3, process)
        if successor:
            pidfile.write_text("successor-host 999999\n")
        process.terminate()
        assert process.wait(timeout=5) == 0
        assert pidfile.exists() == successor
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
