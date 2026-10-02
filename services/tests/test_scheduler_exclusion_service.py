# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import http.client
import json
import logging
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from unittest.mock import MagicMock

import pytest

from nvidia_resiliency_ext.services.scheduler_exclusions.config import (
    SchedulerExclusionServiceSettings,
)
from nvidia_resiliency_ext.services.scheduler_exclusions.decision_file import (
    DecisionFileWriteError,
    DecisionFileWriter,
    DecisionObservation,
    decision_file_path,
)
from nvidia_resiliency_ext.services.scheduler_exclusions.monitor import (
    ArrayTaskGeneration,
    JobIdentity,
    LocalSlurmCommandRunner,
    MalformedSchedulerResponse,
    SchedulerExclusionConfig,
    SchedulerExclusionError,
    SchedulerExclusionMonitor,
    discover_allocation,
    job_identity_from_env,
    query_scheduler_exclusions,
)
from nvidia_resiliency_ext.services.scheduler_exclusions.server import (
    SchedulerExclusionHttpServer,
    SchedulerExclusionRequestHandler,
)


class CallbackRunner:
    def __init__(self, callback: Callable[[list[str]], str]):
        self.callback = callback
        self.calls: list[list[str]] = []

    def run(self, argv: Sequence[str]) -> str:
        call = list(argv)
        self.calls.append(call)
        return self.callback(call)


def test_job_identity_prefers_array_parent():
    identity = job_identity_from_env(
        {
            "SLURM_ARRAY_JOB_ID": "123",
            "SLURM_JOB_ID": "123_7",
        }
    )

    assert identity == JobIdentity(job_id="123", is_array=True)


def test_job_identity_uses_regular_job_id():
    assert job_identity_from_env({"SLURM_JOB_ID": "456"}) == JobIdentity(
        job_id="456",
        is_array=False,
    )
    assert job_identity_from_env({}) is None


def test_decision_file_path_is_owned_by_component(tmp_path):
    assert decision_file_path(tmp_path, "123") == tmp_path / "segment_health_check.123.state"

    with pytest.raises(ValueError, match="invalid Slurm job ID"):
        decision_file_path(tmp_path, "../123")


def test_discover_array_allocation_groups_nodes_by_partition():
    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            return "7|0|batch|node[01-02]\n" "8|2|batch|node[02-03]\n" "9|1|spare|node04\n"
        if argv[-1] == "node[01-02]":
            return "node01\nnode02\n"
        if argv[-1] == "node[02-03]":
            return "node02\nnode03\n"
        if argv[-1] == "node04":
            return "node04\n"
        raise AssertionError(argv)

    runner = CallbackRunner(callback)
    allocation = discover_allocation(runner, JobIdentity(job_id="123", is_array=True))

    assert allocation.nodes_by_partition == {
        "batch": ("node01", "node02", "node03"),
        "spare": ("node04",),
    }
    assert allocation.array_task_generations_by_node == {
        "node01": (ArrayTaskGeneration("7", 0),),
        "node02": (
            ArrayTaskGeneration("7", 0),
            ArrayTaskGeneration("8", 2),
        ),
        "node03": (ArrayTaskGeneration("8", 2),),
        "node04": (ArrayTaskGeneration("9", 1),),
    }
    assert runner.calls[0] == [
        "squeue",
        "--noheader",
        "--array",
        "--jobs",
        "123",
        "--states=RUNNING",
        "--Format=ArrayTaskID:64|,RestartCnt:16|,Partition:128|,NodeList:1024",
    ]


def test_discover_regular_allocation_has_no_array_task_ids():
    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            return "batch|node01\n"
        return "node01\n"

    allocation = discover_allocation(
        CallbackRunner(callback),
        JobIdentity(job_id="123", is_array=False),
    )

    assert allocation.array_task_generations_by_node == {}


def test_discover_allocation_rejects_empty_squeue():
    runner = CallbackRunner(lambda _: "")

    with pytest.raises(SchedulerExclusionError, match="no running allocations"):
        discover_allocation(runner, JobIdentity(job_id="123", is_array=False))


def test_discover_array_allocation_rejects_invalid_restart_count():
    runner = CallbackRunner(lambda _: "7|invalid|batch|node01\n")

    with pytest.raises(MalformedSchedulerResponse, match="invalid restart count"):
        discover_allocation(runner, JobIdentity(job_id="123", is_array=True))


@pytest.mark.parametrize(
    "row, message",
    [
        ("7|0||node01\n", "no partition"),
        ("7|0|batch|\n", "no nodelist"),
    ],
)
def test_discover_array_allocation_rejects_incomplete_rows(row, message):
    runner = CallbackRunner(lambda _: row)

    with pytest.raises(MalformedSchedulerResponse, match=message):
        discover_allocation(runner, JobIdentity(job_id="123", is_array=True))


def test_discover_allocation_rejects_empty_nodelist_expansion():
    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            return "7|0|batch|node01\n"
        return ""

    with pytest.raises(MalformedSchedulerResponse, match="returned no nodes"):
        discover_allocation(
            CallbackRunner(callback),
            JobIdentity(job_id="123", is_array=True),
        )


def test_discover_allocation_stops_between_nodelist_expansions():
    stop_requested = threading.Event()

    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            return "7|0|batch|node01\n8|0|batch|node02\n"
        stop_requested.set()
        return "node01\n"

    runner = CallbackRunner(callback)

    with pytest.raises(SchedulerExclusionError, match="monitor is stopping"):
        discover_allocation(
            runner,
            JobIdentity(job_id="123", is_array=True),
            stop_requested=stop_requested.is_set,
        )

    assert [call[0] for call in runner.calls] == ["squeue", "scontrol"]


def test_query_scheduler_exclusions_uses_one_filtered_sinfo_call():
    runner = CallbackRunner(lambda _: "node01|drained|test drain\nnode02|down*|not responding\n")

    records = query_scheduler_exclusions(runner)

    assert records["node01"].excluded
    assert records["node01"].state == "DRAINED"
    assert records["node02"].excluded
    assert records["node02"].state == "NO_RESPOND"
    assert runner.calls == [
        [
            "sinfo",
            "--noheader",
            "--Node",
            "--states=drain,down,fail,no_respond",
            "--format=%N|%T|%E",
        ]
    ]


def test_query_scheduler_exclusions_accepts_no_unavailable_nodes():
    assert query_scheduler_exclusions(CallbackRunner(lambda _: "")) == {}


def test_query_scheduler_exclusions_rejects_allocatable_rows():
    runner = CallbackRunner(lambda _: "node01|idle|none\n")

    with pytest.raises(MalformedSchedulerResponse, match="allocatable nodes: node01"):
        query_scheduler_exclusions(runner)


def test_query_scheduler_exclusions_treats_slurm_no_response_suffix_as_excluded():
    runner = CallbackRunner(lambda _: "node01|idle*|Not responding\n")

    records = query_scheduler_exclusions(runner)

    assert records["node01"].excluded
    assert records["node01"].state == "NO_RESPOND"


@pytest.mark.parametrize("state", ["", "unknown", "UNKNOWN"])
def test_query_scheduler_exclusions_rejects_unknown_state(state):
    runner = CallbackRunner(lambda _: f"node01|{state}|no reliable state\n")

    with pytest.raises(MalformedSchedulerResponse, match="(no|unknown) state"):
        query_scheduler_exclusions(runner)


def test_local_runner_uses_resolved_binary_and_inherited_environment(monkeypatch):
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="ok\n",
        stderr="",
    )
    run = MagicMock(return_value=completed)
    monkeypatch.setattr(
        "nvidia_resiliency_ext.services.scheduler_exclusions.monitor.shutil.which",
        lambda _: "/usr/bin/squeue",
    )
    monkeypatch.setattr(
        "nvidia_resiliency_ext.services.scheduler_exclusions.monitor.subprocess.run",
        run,
    )
    runner = LocalSlurmCommandRunner(timeout_seconds=12)

    assert runner.run(["squeue", "--jobs", "123"]) == "ok\n"
    assert run.call_args.args[0] == ["/usr/bin/squeue", "--jobs", "123"]
    assert run.call_args.kwargs["timeout"] == 12
    assert run.call_args.kwargs["env"]["PATH"]
    assert "shell" not in run.call_args.kwargs


def test_local_runner_applies_slurm_paths(monkeypatch):
    run = MagicMock(
        return_value=subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="ok\n",
            stderr="",
        )
    )
    monkeypatch.setattr(
        "nvidia_resiliency_ext.services.scheduler_exclusions.monitor.subprocess.run",
        run,
    )
    runner = LocalSlurmCommandRunner(
        slurm_bin_dir="/cm/shared/apps/slurm/current/bin",
        slurm_conf="/cm/shared/apps/slurm/var/etc/slurm/slurm.conf",
    )

    assert runner.run(["squeue", "--format=%P|%N"]) == "ok\n"
    assert run.call_args.args[0] == [
        "/cm/shared/apps/slurm/current/bin/squeue",
        "--format=%P|%N",
    ]
    assert (
        run.call_args.kwargs["env"]["SLURM_CONF"]
        == "/cm/shared/apps/slurm/var/etc/slurm/slurm.conf"
    )


def test_decision_file_starts_with_compact_task_ids_then_detailed_decisions(tmp_path):
    path = tmp_path / "decision.jsonl"
    writer = DecisionFileWriter(path)

    assert writer.publish(
        job_id="123",
        generated_at=250.0,
        cache_ttl_seconds=1800.0,
        observations=[
            DecisionObservation(
                node="node01",
                state="DRAINED",
                reason="first",
                observed_at=100.0,
                array_tasks=(("7", 0),),
            ),
            DecisionObservation(
                node="node02",
                state="DOWN",
                reason="second",
                observed_at=200.0,
                array_tasks=(("7", 0),),
            ),
            DecisionObservation(
                node="node03",
                state="NO_RESPOND",
                reason="third",
                observed_at=150.0,
                array_tasks=(("8", 1),),
            ),
        ],
    ) == (2, 3)

    lines = path.read_text().splitlines()
    assert lines[0] == '["7","8"]'
    records = [json.loads(line) for line in lines]
    assert records[0] == ["7", "8"]
    assert [record["type"] for record in records[1:]] == [
        "decision",
        "decision",
        "observation",
        "observation",
        "observation",
    ]
    assert records[1] == {
        "type": "decision",
        "schema_version": 1,
        "job_id": "123",
        "generated_at": "1970-01-01T00:04:10Z",
        "scope": "array_task",
        "excluded_array_tasks": [
            {
                "task_id": "7",
                "restart_count": 0,
                "valid_until": "1970-01-01T00:33:20Z",
            },
            {
                "task_id": "8",
                "restart_count": 1,
                "valid_until": "1970-01-01T00:32:30Z",
            },
        ],
    }
    assert records[2] == {
        "type": "decision",
        "schema_version": 1,
        "job_id": "123",
        "generated_at": "1970-01-01T00:04:10Z",
        "scope": "node",
        "excluded_nodes": [
            {"node": "node01", "valid_until": "1970-01-01T00:31:40Z"},
            {"node": "node02", "valid_until": "1970-01-01T00:33:20Z"},
            {"node": "node03", "valid_until": "1970-01-01T00:32:30Z"},
        ],
    }


def test_decision_file_replace_failure_preserves_previous_artifact(tmp_path, monkeypatch):
    path = tmp_path / "decision.jsonl"
    path.write_text("previous\n", encoding="utf-8")
    writer = DecisionFileWriter(path)
    monkeypatch.setattr(
        "nvidia_resiliency_ext.services.scheduler_exclusions.decision_file.os.replace",
        MagicMock(side_effect=OSError("filesystem busy")),
    )

    with pytest.raises(DecisionFileWriteError, match="filesystem busy"):
        writer.publish(
            job_id="123",
            generated_at=100.0,
            cache_ttl_seconds=1800.0,
            observations=[],
        )

    assert path.read_text(encoding="utf-8") == "previous\n"
    assert list(tmp_path.glob(".decision.jsonl.*.tmp")) == []


def test_monitor_refreshes_preserves_and_expires_unavailable_observations():
    now = [100.0]
    fail_sinfo = [False]

    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            return "batch|node[01-02]\n"
        if argv[0] == "scontrol":
            return "node01\nnode02\n"
        if fail_sinfo[0]:
            raise SchedulerExclusionError("controller busy")
        return "node01|drained|test drain\n"

    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(
            cache_ttl_seconds=1800,
        ),
        env={"SLURM_JOB_ID": "123"},
        runner=CallbackRunner(callback),
        clock=lambda: now[0],
    )

    assert monitor.poll_once()
    snapshot = monitor.snapshot()
    assert snapshot["excluded_nodes"] == ["node01"]
    assert snapshot["observations"]["node01"]["state"] == "DRAINED"
    assert snapshot["observations"]["node01"]["array_tasks"] == []
    assert snapshot["stats"]["cache_quality"] == "complete"

    now[0] = 200.0
    fail_sinfo[0] = True
    assert not monitor.poll_once()
    assert monitor.snapshot()["excluded_nodes"] == ["node01"]

    now[0] = 1901.0
    snapshot = monitor.snapshot()
    assert snapshot["excluded_nodes"] == []
    assert snapshot["stats"]["cache_quality"] == "unavailable"


def test_monitor_snapshot_does_not_wait_for_mutable_state_lock():
    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(),
        env={"SLURM_JOB_ID": "123"},
        runner=CallbackRunner(lambda argv: pytest.fail(f"unexpected scheduler call: {argv}")),
        clock=lambda: 100.0,
    )
    snapshot_ready = threading.Event()
    result: list[dict] = []

    def read_snapshot() -> None:
        result.append(monitor.snapshot())
        snapshot_ready.set()

    reader = threading.Thread(target=read_snapshot)
    monitor._lock.acquire()
    try:
        reader.start()
        assert snapshot_ready.wait(timeout=1)
    finally:
        monitor._lock.release()
        reader.join(timeout=1)

    assert result[0]["job_id"] == "123"
    assert result[0]["excluded_nodes"] == []


def test_monitor_reports_current_array_task_generation_for_excluded_nodes():
    array_task_id = ["7"]
    restart_count = [0]

    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            return f"{array_task_id[0]}|{restart_count[0]}|batch|node01\n"
        if argv[0] == "scontrol":
            return "node01\n"
        return "node01|drained|test drain\n"

    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(),
        env={"SLURM_ARRAY_JOB_ID": "123"},
        runner=CallbackRunner(callback),
        clock=lambda: 100.0,
    )

    assert monitor.poll_once()
    assert monitor.snapshot()["observations"]["node01"]["array_tasks"] == [
        {"task_id": "7", "restart_count": 0}
    ]

    restart_count[0] = 1
    assert monitor.poll_once()
    assert monitor.snapshot()["observations"]["node01"]["array_tasks"] == [
        {"task_id": "7", "restart_count": 1}
    ]


def test_monitor_replaces_decision_for_new_task_generation_and_clears_it(tmp_path):
    restart_count = [0]
    state = ["drained"]
    fail_allocation = [False]
    decision_path = decision_file_path(tmp_path, "123")

    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            if fail_allocation[0]:
                raise SchedulerExclusionError("controller busy")
            return f"7|{restart_count[0]}|batch|node01\n"
        if argv[0] == "scontrol":
            return "node01\n"
        if any(arg.startswith("--nodes=") for arg in argv):
            return "node01|allocated|none\n" if state[0] == "idle" else ""
        return "node01|drained|reason\n" if state[0] == "drained" else ""

    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(scheduler_exclusion_dir=str(tmp_path)),
        env={"SLURM_ARRAY_JOB_ID": "123"},
        runner=CallbackRunner(callback),
        clock=lambda: 100.0,
    )

    assert monitor.poll_once()
    first = decision_path.read_text(encoding="utf-8")
    first_records = [json.loads(line) for line in first.splitlines()]
    assert first_records[0] == ["7"]
    first_decision = first_records[1]
    node_decision = first_records[2]
    response = monitor.scheduler_exclusions()
    assert response is not None
    assert response["excluded_array_tasks"] == first_decision["excluded_array_tasks"]
    assert response["excluded_nodes"] == node_decision["excluded_nodes"]
    assert first_decision["excluded_array_tasks"] == [
        {
            "task_id": "7",
            "restart_count": 0,
            "valid_until": "1970-01-01T00:31:40Z",
        }
    ]
    assert node_decision["excluded_nodes"] == [
        {"node": "node01", "valid_until": "1970-01-01T00:31:40Z"}
    ]

    fail_allocation[0] = True
    assert not monitor.poll_once()
    assert decision_path.read_text(encoding="utf-8") == first

    fail_allocation[0] = False
    restart_count[0] = 1
    assert monitor.poll_once()
    records = [json.loads(line) for line in decision_path.read_text().splitlines()]
    assert records[0] == ["7"]
    current_decision = records[1]
    response = monitor.scheduler_exclusions()
    assert response is not None
    assert response["excluded_array_tasks"] == current_decision["excluded_array_tasks"]
    assert response["excluded_nodes"] == records[2]["excluded_nodes"]
    assert current_decision["excluded_array_tasks"] == [
        {
            "task_id": "7",
            "restart_count": 1,
            "valid_until": "1970-01-01T00:31:40Z",
        }
    ]

    state[0] = "idle"
    assert monitor.poll_once()
    records = [json.loads(line) for line in decision_path.read_text().splitlines()]
    assert records[0] == []
    current_decision = records[1]
    assert current_decision["excluded_array_tasks"] == []
    assert records[2]["excluded_nodes"] == []
    assert monitor.scheduler_exclusions() == {
        "type": "decision",
        "schema_version": 1,
        "job_id": "123",
        "generated_at": "1970-01-01T00:01:40Z",
        "excluded_array_tasks": [],
        "excluded_nodes": [],
    }
    assert monitor.snapshot()["last_decision_write"] == "1970-01-01T00:01:40Z"
    assert monitor.snapshot()["stats"]["decision_write_failures"] == 0


def test_monitor_publishes_node_decision_for_regular_job(tmp_path):
    decision_path = decision_file_path(tmp_path, "123")

    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            return "batch|node01\n"
        if argv[0] == "scontrol":
            return "node01\n"
        return "node01|drained|reason\n"

    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(scheduler_exclusion_dir=str(tmp_path)),
        env={"SLURM_JOB_ID": "123"},
        runner=CallbackRunner(callback),
        clock=lambda: 100.0,
    )

    assert monitor.poll_once()
    records = [json.loads(line) for line in decision_path.read_text().splitlines()]
    assert records[0] == []
    assert records[1]["scope"] == "array_task"
    assert records[1]["excluded_array_tasks"] == []
    assert records[2] == {
        "type": "decision",
        "schema_version": 1,
        "job_id": "123",
        "generated_at": "1970-01-01T00:01:40Z",
        "scope": "node",
        "excluded_nodes": [{"node": "node01", "valid_until": "1970-01-01T00:31:40Z"}],
    }
    assert records[3]["node"] == "node01"
    assert records[3]["array_tasks"] == []
    response = monitor.scheduler_exclusions()
    assert response is not None
    assert response["excluded_array_tasks"] == []
    assert response["excluded_nodes"] == records[2]["excluded_nodes"]


def test_failed_filtered_query_does_not_remap_stale_evidence_to_requeued_task(tmp_path):
    restart_count = [0]
    fail_node01 = [False]
    decision_path = decision_file_path(tmp_path, "123")

    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            return f"7|{restart_count[0]}|batch|node01\n" "8|0|batch|node02\n"
        if argv[0] == "scontrol":
            return f"{argv[-1]}\n"
        if fail_node01[0]:
            raise SchedulerExclusionError("controller busy")
        return "node01|drained|reason\n"

    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(scheduler_exclusion_dir=str(tmp_path)),
        env={"SLURM_ARRAY_JOB_ID": "123"},
        runner=CallbackRunner(callback),
        clock=lambda: 100.0,
    )

    assert monitor.poll_once()
    restart_count[0] = 1
    fail_node01[0] = True
    assert not monitor.poll_once()

    records = [json.loads(line) for line in decision_path.read_text().splitlines()]
    assert records[0] == ["7"]
    decision = records[1]
    assert decision["excluded_array_tasks"] == [
        {
            "task_id": "7",
            "restart_count": 0,
            "valid_until": "1970-01-01T00:31:40Z",
        }
    ]


def test_monitor_records_decision_publication_failure(tmp_path):
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("blocked", encoding="utf-8")

    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            return "7|0|batch|node01\n"
        if argv[0] == "scontrol":
            return "node01\n"
        return "node01|drained|reason\n"

    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(scheduler_exclusion_dir=str(blocked_parent)),
        env={"SLURM_ARRAY_JOB_ID": "123"},
        runner=CallbackRunner(callback),
        clock=lambda: 100.0,
    )

    assert monitor.poll_once()
    snapshot = monitor.snapshot()
    assert snapshot["last_decision_write"] is None
    assert snapshot["last_decision_error"]
    assert snapshot["stats"]["decision_write_failures"] == 1


def test_monitor_removes_node_after_explicit_good_observation(caplog):
    now = [100.0]
    state = ["drained"]
    caplog.set_level(logging.INFO)

    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            return "batch|node01\n"
        if argv[0] == "scontrol":
            return "node01\n"
        if any(arg.startswith("--nodes=") for arg in argv):
            return "node01|allocated|none\n"
        return "node01|drained|reason\n" if state[0] == "drained" else ""

    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(),
        env={"SLURM_JOB_ID": "123"},
        runner=CallbackRunner(callback),
        clock=lambda: now[0],
    )

    assert monitor.poll_once()
    assert monitor.snapshot()["excluded_nodes"] == ["node01"]

    now[0] = 200.0
    state[0] = "idle"
    assert monitor.poll_once()
    assert monitor.snapshot()["excluded_nodes"] == []
    assert "excluded=1" in caplog.text
    assert "newly_excluded=['node01']" in caplog.text
    assert "became_allocatable=['node01']" in caplog.text


def test_monitor_distinguishes_excluded_node_leaving_allocation(caplog):
    allocated_node = ["node01"]
    caplog.set_level(logging.INFO)

    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            return f"batch|{allocated_node[0]}\n"
        if argv[0] == "scontrol":
            return f"{allocated_node[0]}\n"
        if allocated_node[0] == "node01":
            return "node01|drained|reason\n"
        return ""

    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(),
        env={"SLURM_JOB_ID": "123"},
        runner=CallbackRunner(callback),
        clock=lambda: 100.0,
    )

    assert monitor.poll_once()
    allocated_node[0] = "node02"
    assert monitor.poll_once()

    assert "left_allocation=['node01']" in caplog.text
    assert "became_allocatable=['node01']" not in caplog.text


def test_monitor_preserves_complete_cache_when_filtered_query_fails():
    fail_sinfo = [False]

    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            return "batch|node[01-02]\n"
        if argv[0] == "scontrol":
            return "node01\nnode02\n"
        if fail_sinfo[0]:
            raise SchedulerExclusionError("controller busy")
        return "node01|drained|reason\n"

    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(),
        env={"SLURM_JOB_ID": "123"},
        runner=CallbackRunner(callback),
        clock=lambda: 100.0,
    )

    assert monitor.poll_once()
    fail_sinfo[0] = True
    assert not monitor.poll_once()
    snapshot = monitor.snapshot()
    assert snapshot["excluded_nodes"] == ["node01"]
    assert snapshot["stats"]["cache_quality"] == "complete"
    assert "controller busy" in snapshot["last_error"]


def test_monitor_rejects_unexpected_allocatable_filtered_row():
    sinfo_calls = 0

    def callback(argv: list[str]) -> str:
        nonlocal sinfo_calls
        if argv[0] == "squeue":
            return "batch|node[01-02]\n"
        if argv[0] == "scontrol":
            return "node01\nnode02\n"
        sinfo_calls += 1
        return "node01|idle|none\n"

    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(),
        env={"SLURM_JOB_ID": "123"},
        runner=CallbackRunner(callback),
        clock=lambda: 100.0,
    )

    assert not monitor.poll_once()
    assert sinfo_calls == 1
    snapshot = monitor.snapshot()
    assert snapshot["excluded_nodes"] == []
    assert snapshot["stats"]["cache_quality"] == "unavailable"
    assert "allocatable nodes" in snapshot["last_error"]


def test_monitor_replaces_complete_unavailable_snapshot():
    excluded_nodes = ["node01", "node02"]

    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            return "batch|node[01-02]\n"
        if argv[0] == "scontrol":
            return "node01\nnode02\n"
        if any(arg.startswith("--nodes=") for arg in argv):
            return "node01|allocated|none\n"
        return "".join(f"{node}|drained|reason\n" for node in excluded_nodes)

    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(),
        env={"SLURM_JOB_ID": "123"},
        runner=CallbackRunner(callback),
        clock=lambda: 100.0,
    )

    assert monitor.poll_once()
    assert monitor.snapshot()["excluded_nodes"] == ["node01", "node02"]

    excluded_nodes[:] = ["node02"]
    assert monitor.poll_once()
    assert monitor.snapshot()["excluded_nodes"] == ["node02"]


def test_monitor_clears_explicitly_recovered_node_and_retains_missing_node():
    now = [100.0]
    first_poll = [True]

    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            return "batch|node[01-02]\n"
        if argv[0] == "scontrol":
            return "node01\nnode02\n"
        if any(arg.startswith("--nodes=") for arg in argv):
            return "node01|allocated|none\n"
        if first_poll[0]:
            return "node01|drained|reason\nnode02|drained|reason\n"
        return ""

    runner = CallbackRunner(callback)
    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(),
        env={"SLURM_JOB_ID": "123"},
        runner=runner,
        clock=lambda: now[0],
    )

    assert monitor.poll_once()
    first_poll[0] = False
    now[0] = 200.0
    assert monitor.poll_once()

    snapshot = monitor.snapshot()
    assert snapshot["excluded_nodes"] == ["node02"]
    assert snapshot["observations"]["node02"]["observed_at"] == "1970-01-01T00:01:40Z"
    assert [call for call in runner.calls if call[0] == "sinfo"][-1] == [
        "sinfo",
        "--noheader",
        "--Node",
        "--nodes=node01,node02",
        "--format=%N|%T|%E",
    ]


def test_monitor_retains_recovery_candidates_when_verification_fails():
    filtered_poll = [0]

    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            return "batch|node01\n"
        if argv[0] == "scontrol":
            return "node01\n"
        if any(arg.startswith("--nodes=") for arg in argv):
            raise SchedulerExclusionError("controller busy")
        filtered_poll[0] += 1
        return "node01|drained|reason\n" if filtered_poll[0] == 1 else ""

    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(),
        env={"SLURM_JOB_ID": "123"},
        runner=CallbackRunner(callback),
        clock=lambda: 100.0,
    )

    assert monitor.poll_once()
    assert monitor.poll_once()
    assert monitor.snapshot()["excluded_nodes"] == ["node01"]


def test_monitor_refreshes_candidate_still_unavailable_during_verification():
    now = [100.0]
    filtered_poll = [0]

    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            return "batch|node01\n"
        if argv[0] == "scontrol":
            return "node01\n"
        if any(arg.startswith("--nodes=") for arg in argv):
            return "node01|down|still unavailable\n"
        filtered_poll[0] += 1
        return "node01|drained|initial\n" if filtered_poll[0] == 1 else ""

    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(),
        env={"SLURM_JOB_ID": "123"},
        runner=CallbackRunner(callback),
        clock=lambda: now[0],
    )

    assert monitor.poll_once()
    now[0] = 200.0
    assert monitor.poll_once()
    observation = monitor.snapshot()["observations"]["node01"]
    assert observation["state"] == "DOWN"
    assert observation["reason"] == "still unavailable"
    assert observation["observed_at"] == "1970-01-01T00:03:20Z"


def test_monitor_skips_recovery_verification_above_candidate_limit(caplog):
    filtered_poll = [0]
    nodes = [f"node{index:02d}" for index in range(1, 18)]
    caplog.set_level(logging.WARNING)

    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            return "batch|node[01-17]\n"
        if argv[0] == "scontrol":
            return "".join(f"{node}\n" for node in nodes)
        if any(arg.startswith("--nodes=") for arg in argv):
            raise AssertionError("large candidate set must not be queried")
        filtered_poll[0] += 1
        if filtered_poll[0] == 1:
            return "".join(f"{node}|drained|reason\n" for node in nodes)
        return ""

    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(),
        env={"SLURM_JOB_ID": "123"},
        runner=CallbackRunner(callback),
        clock=lambda: 100.0,
    )

    assert monitor.poll_once()
    assert monitor.poll_once()
    assert monitor.snapshot()["excluded_nodes"] == nodes
    assert "recovery verification skipped" in caplog.text
    assert "candidates=17 limit=16" in caplog.text


def test_monitor_stop_during_filtered_query_returns_without_retry():
    monitor = None
    sinfo_calls = 0

    def callback(argv: list[str]) -> str:
        nonlocal sinfo_calls
        if argv[0] == "squeue":
            return "batch|node01\n"
        if argv[0] == "scontrol":
            return "node01\n"
        sinfo_calls += 1
        assert monitor is not None
        monitor._stop_event.set()
        raise SchedulerExclusionError("controller busy")

    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(),
        env={"SLURM_JOB_ID": "123"},
        runner=CallbackRunner(callback),
        clock=lambda: 100.0,
    )

    assert not monitor.poll_once()
    assert sinfo_calls == 1


def test_monitor_stop_retains_a_live_worker_reference():
    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(),
        env={"SLURM_JOB_ID": "123"},
        runner=CallbackRunner(lambda _: ""),
    )
    worker = MagicMock()
    worker.is_alive.return_value = True
    monitor._thread = worker

    monitor.stop()
    monitor.start()

    assert monitor._thread is worker
    worker.join.assert_called_once()
    assert monitor._stop_event.is_set()


def test_refresh_hint_wakes_worker_and_coalesces_while_polling():
    squeue_calls = 0
    second_poll_started = threading.Event()
    release_second_poll = threading.Event()

    def callback(argv: list[str]) -> str:
        nonlocal squeue_calls
        if argv[0] == "squeue":
            squeue_calls += 1
            if squeue_calls == 2:
                second_poll_started.set()
                assert release_second_poll.wait(timeout=2)
            return "batch|node01\n"
        if argv[0] == "scontrol":
            return "node01\n"
        return ""

    runner = CallbackRunner(callback)
    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(
            refresh_interval_seconds=3600,
            jitter_fraction=0,
        ),
        env={"SLURM_JOB_ID": "123"},
        runner=runner,
    )
    monitor.start()
    try:
        deadline = time.monotonic() + 2
        while monitor.snapshot()["stats"]["polls_completed"] < 1:
            assert time.monotonic() < deadline
            time.sleep(0.01)

        deadline = time.monotonic() + 2
        while not monitor.request_refresh():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert second_poll_started.wait(timeout=2)
        assert not monitor.request_refresh()
    finally:
        release_second_poll.set()
        monitor.stop()

    assert squeue_calls == 2


def test_refresh_hint_does_not_run_scheduler_io():
    runner = CallbackRunner(lambda argv: pytest.fail(f"unexpected scheduler call: {argv}"))
    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(),
        env={"SLURM_JOB_ID": "123"},
        runner=runner,
    )

    assert not monitor.request_refresh()
    assert runner.calls == []


def test_monitor_uses_one_filtered_sinfo_query_for_5000_allocated_nodes():
    nodes = [f"node{i:04d}" for i in range(5000)]
    sinfo_calls: list[list[str]] = []

    def callback(argv: list[str]) -> str:
        if argv[0] == "squeue":
            return "7|0|batch|all-nodes\n"
        if argv[0] == "scontrol":
            return "\n".join(nodes)
        sinfo_calls.append(argv)
        return "node4999|drained|test drain\nnode-outside|down|other job\n"

    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(),
        env={"SLURM_ARRAY_JOB_ID": "123"},
        runner=CallbackRunner(callback),
        clock=lambda: 100.0,
    )

    assert monitor.poll_once()
    assert len(sinfo_calls) == 1
    assert "--nodes" not in sinfo_calls[0]
    assert monitor.snapshot()["excluded_nodes"] == ["node4999"]


def test_monitor_without_job_identity_is_unavailable():
    monitor = SchedulerExclusionMonitor(
        SchedulerExclusionConfig(),
        env={},
        clock=lambda: 100.0,
    )

    assert monitor.snapshot() == {
        "job_id": None,
        "last_complete_poll": None,
        "last_poll_attempt": None,
        "last_decision_write": None,
        "excluded_nodes": [],
        "observations": {},
        "last_error": "Slurm job identity is unavailable",
        "last_decision_error": None,
        "stats": {
            "polls_attempted": 0,
            "polls_completed": 0,
            "decision_write_failures": 0,
            "current_nodes": 0,
            "current_excluded_nodes": 0,
            "cache_quality": "unavailable",
        },
    }


def test_settings_use_scheduler_exclusion_environment_prefix():
    settings = SchedulerExclusionServiceSettings.from_env(
        {
            "NVRX_SCHEDULER_EXCLUSION_HOST": "0.0.0.0",
            "NVRX_SCHEDULER_EXCLUSION_PORT": "19090",
            "NVRX_SCHEDULER_EXCLUSION_SLURM_BIN_DIR": "/opt/slurm/bin",
            "NVRX_SCHEDULER_EXCLUSION_SLURM_CONF": "/etc/slurm/slurm.conf",
            "NVRX_SCHEDULER_EXCLUSION_DIR": "/shared/scheduler-exclusions",
            "NVRX_SCHEDULER_EXCLUSION_REFRESH_INTERVAL_SECONDS": "30",
            "NVRX_SCHEDULER_EXCLUSION_CACHE_TTL_SECONDS": "90",
            "NVRX_SCHEDULER_EXCLUSION_QUERY_TIMEOUT_SECONDS": "4",
        }
    )

    assert settings.host == "0.0.0.0"
    assert settings.port == 19090
    assert settings.monitor_config() == SchedulerExclusionConfig(
        slurm_bin_dir="/opt/slurm/bin",
        slurm_conf="/etc/slurm/slurm.conf",
        scheduler_exclusion_dir="/shared/scheduler-exclusions",
        refresh_interval_seconds=30,
        cache_ttl_seconds=90,
        query_timeout_seconds=4,
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PORT", "not-a-port"),
        ("REFRESH_INTERVAL_SECONDS", "often"),
        ("CACHE_TTL_SECONDS", "later"),
        ("QUERY_TIMEOUT_SECONDS", "slow"),
    ],
)
def test_settings_name_invalid_numeric_environment_values(name, value):
    variable = f"NVRX_SCHEDULER_EXCLUSION_{name}"

    with pytest.raises(ValueError, match=rf"{variable}: '{value}'"):
        SchedulerExclusionServiceSettings.from_env({variable: value})


@pytest.mark.parametrize("field", ["slurm_bin_dir", "slurm_conf", "scheduler_exclusion_dir"])
def test_local_slurm_paths_must_be_absolute(field):
    with pytest.raises(ValueError, match="absolute path"):
        SchedulerExclusionServiceSettings(**{field: "relative/path"})


@pytest.mark.parametrize(
    "field",
    [
        "refresh_interval_seconds",
        "cache_ttl_seconds",
        "query_timeout_seconds",
    ],
)
def test_numeric_settings_must_be_positive(field):
    with pytest.raises(ValueError, match="positive"):
        SchedulerExclusionServiceSettings(**{field: 0})


def _http_json(method: str, url: str) -> tuple[int, dict]:
    request = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(request, timeout=2) as response:  # nosec B310
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_http_server_exposes_cache_and_nonblocking_refresh():
    monitor = MagicMock()
    monitor.snapshot.return_value = {
        "job_id": "123",
        "last_complete_poll": "2026-07-30T12:40:00Z",
        "last_poll_attempt": "2026-07-30T12:41:00Z",
        "last_decision_write": "2026-07-30T12:40:31Z",
        "excluded_nodes": ["node01"],
        "observations": {
            "node01": {
                "state": "DRAIN",
                "reason": "test",
                "observed_at": "2026-07-30T12:40:00Z",
                "array_tasks": [{"task_id": "7", "restart_count": 0}],
            }
        },
        "last_error": None,
        "last_decision_error": None,
        "stats": {
            "polls_attempted": 4,
            "polls_completed": 3,
            "decision_write_failures": 0,
            "current_nodes": 5000,
            "current_excluded_nodes": 1,
            "cache_quality": "complete",
        },
    }
    monitor.scheduler_exclusions.return_value = {
        "type": "decision",
        "schema_version": 1,
        "job_id": "123",
        "generated_at": "2026-07-30T12:40:30Z",
        "excluded_array_tasks": [
            {
                "task_id": "7",
                "restart_count": 0,
                "valid_until": "2026-07-30T13:10:00Z",
            }
        ],
        "excluded_nodes": [
            {
                "node": "node01",
                "valid_until": "2026-07-30T13:10:00Z",
            }
        ],
    }
    monitor.request_refresh.return_value = True
    server = SchedulerExclusionHttpServer(("127.0.0.1", 0), monitor)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    try:
        status, health = _http_json("GET", f"{endpoint}/healthz")
        assert status == 200
        assert health["service"] == "nvrx-scheduler-exclusion-service"

        status, body = _http_json("GET", f"{endpoint}/scheduler-exclusions")
        assert status == 200
        assert body == monitor.scheduler_exclusions.return_value

        status, stats = _http_json("GET", f"{endpoint}/stats")
        assert status == 200
        assert stats == {
            "job_id": "123",
            "last_complete_poll": "2026-07-30T12:40:00Z",
            "last_poll_attempt": "2026-07-30T12:41:00Z",
            "last_decision_write": "2026-07-30T12:40:31Z",
            "last_error": None,
            "last_decision_error": None,
            "polls_attempted": 4,
            "polls_completed": 3,
            "decision_write_failures": 0,
            "current_nodes": 5000,
            "current_excluded_nodes": 1,
            "cache_quality": "complete",
        }

        status, body = _http_json("GET", f"{endpoint}/allocation-state")
        assert status == 404
        assert body == {"error": "not_found"}

        status, body = _http_json("GET", f"{endpoint}/allocation-state/stats")
        assert status == 404
        assert body == {"error": "not_found"}

        status, refresh = _http_json("POST", f"{endpoint}/refresh")
        assert status == 202
        assert refresh == {"accepted": True}
        monitor.request_refresh.assert_called_once_with()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_server_reports_scheduler_exclusions_unavailable_before_first_poll():
    monitor = MagicMock()
    monitor.scheduler_exclusions.return_value = None
    server = SchedulerExclusionHttpServer(("127.0.0.1", 0), monitor)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    try:
        status, body = _http_json("GET", f"{endpoint}/scheduler-exclusions")
        assert status == 503
        assert body == {"error": "scheduler_exclusions_unavailable"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_server_consumes_post_body_on_persistent_connection(monkeypatch):
    monitor = MagicMock()
    monitor.snapshot.return_value = {
        "job_id": "123",
        "last_complete_poll": None,
        "last_poll_attempt": None,
        "last_decision_write": None,
        "excluded_nodes": [],
        "observations": {},
        "last_error": None,
        "last_decision_error": None,
        "stats": {
            "polls_attempted": 0,
            "polls_completed": 0,
            "decision_write_failures": 0,
            "current_nodes": 0,
            "current_excluded_nodes": 0,
            "cache_quality": "unavailable",
        },
    }
    monitor.request_refresh.return_value = True
    monkeypatch.setattr(SchedulerExclusionRequestHandler, "protocol_version", "HTTP/1.1")
    server = SchedulerExclusionHttpServer(("127.0.0.1", 0), monitor)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.request(
            "POST",
            "/missing",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 404
        assert json.loads(response.read()) == {"error": "not_found"}

        connection.request(
            "POST",
            "/refresh",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 202
        assert json.loads(response.read()) == {"accepted": True}

        connection.request("GET", "/stats")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["cache_quality"] == "unavailable"
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
