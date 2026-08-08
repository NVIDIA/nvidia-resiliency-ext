# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
from unittest.mock import patch

import pytest

from nvidia_resiliency_ext.fault_tolerance.scheduler_exclusion import (
    SlurmArrayTaskGeneration,
    current_slurm_array_replacement_group_id,
    current_slurm_array_task_generation,
    is_current_array_task_node0,
    scheduler_exclusion_segment_is_healthy,
)


def _environment(**overrides):
    values = {
        "SLURM_ARRAY_JOB_ID": "123",
        "SLURM_ARRAY_TASK_ID": "7",
        "SLURM_RESTART_COUNT": "2",
        "SLURM_NODEID": "0",
        "SLURM_PROCID": "0",
    }
    values.update(overrides)
    return values


def _write_decision(tmp_path, tasks=("7",), *, modified_at=None):
    path = tmp_path / "scheduler_exclusion.123.jsonl"
    path.write_text(json.dumps(list(tasks), separators=(",", ":")) + "\n", encoding="utf-8")
    if modified_at is not None:
        os.utime(path, (modified_at, modified_at))
    return path


def _scheduler_check_message(caplog):
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Scheduler Exclusion FT check ")
    ]
    assert len(messages) == 1
    return messages[0]


def _is_segment_healthy(directory, **kwargs):
    return scheduler_exclusion_segment_is_healthy(str(directory), **kwargs)


def test_current_generation_and_replacement_group_are_restart_aware():
    env = _environment()

    assert current_slurm_array_task_generation(env) == SlurmArrayTaskGeneration("123", "7", 2)
    assert current_slurm_array_replacement_group_id(env) == "7:2"
    assert current_slurm_array_replacement_group_id({"SLURM_ARRAY_TASK_ID": "7"}) == "7:0"


@pytest.mark.parametrize("value", ["invalid", "-1"])
def test_current_generation_rejects_invalid_restart_count(value):
    with pytest.raises(ValueError, match="SLURM_RESTART_COUNT"):
        current_slurm_array_task_generation(_environment(SLURM_RESTART_COUNT=value))


def test_node0_prefers_node_id_and_supports_process_id_fallback():
    assert is_current_array_task_node0(_environment())
    assert not is_current_array_task_node0(_environment(SLURM_NODEID="1", SLURM_PROCID="0"))

    env = _environment()
    del env["SLURM_NODEID"]
    assert is_current_array_task_node0(env)
    env["SLURM_PROCID"] = "1"
    assert not is_current_array_task_node0(env)


def test_matching_task_is_excluded(tmp_path, caplog):
    _write_decision(tmp_path, modified_at=100.0)

    with (
        caplog.at_level(logging.INFO),
        patch(
            "nvidia_resiliency_ext.fault_tolerance.scheduler_exclusion.time.monotonic",
            side_effect=[100.0, 100.00125],
        ),
    ):
        assert not _is_segment_healthy(
            tmp_path,
            env=_environment(),
            now=200.0,
            round_id=4,
        )

    message = _scheduler_check_message(caplog)
    assert "job_id=123 round=4 task_id=7 restart_count=2" in message
    assert "outcome=excluded" in message
    assert "elapsed_ms=1.250" in message


def test_short_decision_uses_only_the_initial_read(tmp_path):
    path = _write_decision(tmp_path)
    with path.open("ab") as stream:
        stream.write(b"x" * (128 * 1024))

    real_read = os.read
    read_sizes = []

    def recording_read(descriptor, size):
        read_sizes.append(size)
        return real_read(descriptor, size)

    with patch(
        "nvidia_resiliency_ext.fault_tolerance.scheduler_exclusion.os.read",
        side_effect=recording_read,
    ):
        assert not _is_segment_healthy(
            tmp_path,
            env=_environment(),
        )

    assert read_sizes == [8 * 1024]


def test_large_decision_uses_continuation_reads(tmp_path):
    _write_decision(tmp_path, tasks=(str(task_id) for task_id in range(20_000)))

    real_read = os.read
    read_sizes = []

    def recording_read(descriptor, size):
        read_sizes.append(size)
        return real_read(descriptor, size)

    with patch(
        "nvidia_resiliency_ext.fault_tolerance.scheduler_exclusion.os.read",
        side_effect=recording_read,
    ):
        assert not _is_segment_healthy(
            tmp_path,
            env=_environment(),
        )

    assert read_sizes[0] == 8 * 1024
    assert read_sizes[1:] == [(1024 * 1024 + 1) - 8 * 1024]


def test_quoted_task_id_does_not_match_a_longer_id(tmp_path):
    _write_decision(tmp_path, tasks=("17",))

    assert _is_segment_healthy(tmp_path, env=_environment(SLURM_ARRAY_TASK_ID="7"))


def test_restart_count_does_not_change_task_exclusion(tmp_path):
    _write_decision(tmp_path, tasks=("7",))

    assert not _is_segment_healthy(tmp_path, env=_environment(SLURM_RESTART_COUNT="3"))


@pytest.mark.parametrize(
    ("tasks", "env_overrides", "modified_at", "now", "outcome"),
    [
        (("7",), {"SLURM_ARRAY_TASK_ID": "8"}, 100.0, 200.0, "not_excluded"),
        (("7",), {}, 100.0, 1900.0, "expired"),
        ((), {}, 100.0, 200.0, "not_excluded"),
    ],
)
def test_clean_or_stale_decision_is_healthy(
    tmp_path, tasks, env_overrides, modified_at, now, outcome, caplog
):
    _write_decision(tmp_path, tasks=tasks, modified_at=modified_at)

    with caplog.at_level(logging.INFO):
        assert _is_segment_healthy(
            tmp_path,
            env=_environment(**env_overrides),
            now=now,
        )

    assert f"outcome={outcome}" in _scheduler_check_message(caplog)


def test_non_node0_does_not_open_artifact_or_log_check(tmp_path, caplog):
    with (
        caplog.at_level(logging.INFO),
        patch("nvidia_resiliency_ext.fault_tolerance.scheduler_exclusion.os.open") as open_file,
    ):
        assert _is_segment_healthy(tmp_path, env=_environment(SLURM_NODEID="1"))
    open_file.assert_not_called()
    assert not any(
        "Scheduler Exclusion FT check" in record.getMessage() for record in caplog.records
    )


@pytest.mark.parametrize(
    "writer",
    [
        lambda path: path.write_text("not-json\n", encoding="utf-8"),
        lambda path: path.write_text(json.dumps({"type": "decision"}) + "\n", encoding="utf-8"),
        lambda path: path.write_text('["7",garbage]\n', encoding="utf-8"),
        lambda path: path.write_text('["7", "8"]\n', encoding="utf-8"),
        lambda path: path.write_bytes(b"{" + b"x" * (1024 * 1024)),
    ],
)
def test_malformed_decision_fails_open(tmp_path, writer):
    path = tmp_path / "scheduler_exclusion.123.jsonl"
    writer(path)

    assert _is_segment_healthy(tmp_path, env=_environment())


def test_missing_decision_and_relative_directory_fail_open(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        assert _is_segment_healthy(tmp_path, env=_environment())
    assert "outcome=missing" in _scheduler_check_message(caplog)

    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert _is_segment_healthy("relative", env=_environment())
    assert "outcome=invalid" in _scheduler_check_message(caplog)


def test_io_failure_is_observable_and_fails_open(tmp_path, caplog):
    with (
        caplog.at_level(logging.INFO),
        patch(
            "nvidia_resiliency_ext.fault_tolerance.scheduler_exclusion.os.open",
            side_effect=OSError("storage unavailable"),
        ),
    ):
        assert _is_segment_healthy(tmp_path, env=_environment())

    assert "outcome=io_error" in _scheduler_check_message(caplog)
