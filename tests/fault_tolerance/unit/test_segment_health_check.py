# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from unittest.mock import patch

import pytest

from nvidia_resiliency_ext.fault_tolerance.segment_health_check import (
    SegmentHealthCheck,
    get_segment_health_check,
)


def _environment(**overrides):
    values = {
        "SLURM_ARRAY_JOB_ID": "123",
        "SLURM_ARRAY_TASK_ID": "7",
        "SLURM_NODEID": "0",
        "SLURM_PROCID": "0",
    }
    values.update(overrides)
    return values


def _write_decision(tmp_path, task_id="7", *, nodes="node03", modified_at=None):
    path = tmp_path / f"segment_health_check.123.{task_id}"
    path.write_text(nodes, encoding="utf-8")
    if modified_at is not None:
        os.utime(path, (modified_at, modified_at))
    return path


def _is_segment_healthy(directory, *, task_id="7"):
    return SegmentHealthCheck(str(directory), "123", task_id)()


def test_segment_health_check_is_installed_only_on_array_task_process_zero():
    with patch.dict(os.environ, _environment(SLURM_JOB_ID="123_7"), clear=True):
        check = get_segment_health_check("/shared/nvrx")
    assert check is not None
    assert check.job_id == "123"
    assert check.task_id == "7"

    with patch.dict(os.environ, _environment(SLURM_PROCID="1"), clear=True):
        assert get_segment_health_check("/shared/nvrx") is None

    with patch.dict(os.environ, _environment(SLURM_NODEID="1"), clear=True):
        assert get_segment_health_check("/shared/nvrx") is not None

    env = _environment()
    del env["SLURM_PROCID"]
    with patch.dict(os.environ, env, clear=True):
        assert get_segment_health_check("/shared/nvrx") is None


def test_segment_health_check_is_installed_for_regular_job_process_zero(tmp_path):
    path = tmp_path / "segment_health_check.456.456"
    path.write_text("node03", encoding="utf-8")

    with patch.dict(
        os.environ,
        {"SLURM_JOB_ID": "456", "SLURM_PROCID": "0"},
        clear=True,
    ):
        check = get_segment_health_check(str(tmp_path))

    assert check is not None
    assert check.job_id == "456"
    assert check.task_id == "456"
    assert not check()

    path.write_text("", encoding="utf-8")
    assert check()


def test_segment_health_check_is_not_installed_without_job_metadata():
    env = _environment()
    del env["SLURM_ARRAY_JOB_ID"]
    del env["SLURM_ARRAY_TASK_ID"]

    with patch.dict(os.environ, env, clear=True):
        assert get_segment_health_check("/shared/nvrx") is None


def test_nonempty_task_decision_is_excluded(tmp_path):
    _write_decision(tmp_path)

    assert not _is_segment_healthy(tmp_path)


def test_consumer_checks_only_its_task_file(tmp_path):
    _write_decision(tmp_path, task_id="17")

    assert _is_segment_healthy(tmp_path, task_id="7")


def test_zero_byte_decision_is_healthy(tmp_path):
    _write_decision(tmp_path, nodes="")

    assert _is_segment_healthy(tmp_path)


@pytest.mark.parametrize("nodes", ["node03", "not,csv", "x" * (128 * 1024)])
def test_nonempty_content_is_excluded_without_parsing(tmp_path, nodes):
    _write_decision(tmp_path, nodes=nodes)

    with patch(
        "nvidia_resiliency_ext.fault_tolerance.segment_health_check.Path.open",
        side_effect=AssertionError("consumer must not read the artifact"),
    ):
        assert not _is_segment_healthy(tmp_path)


def test_old_decision_remains_authoritative(tmp_path):
    _write_decision(tmp_path, modified_at=100.0)

    assert not _is_segment_healthy(tmp_path)


def test_missing_decision_fails_open_quietly(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert _is_segment_healthy(tmp_path)
    assert not caplog.text


def test_io_failure_is_observable_and_fails_open(tmp_path, caplog):
    with (
        caplog.at_level(logging.INFO),
        patch(
            "nvidia_resiliency_ext.fault_tolerance.segment_health_check.Path.stat",
            side_effect=OSError("storage unavailable"),
        ),
    ):
        assert _is_segment_healthy(tmp_path)

    assert "Ignoring segment health decision: storage unavailable" in caplog.text


def test_non_regular_decision_is_observable_and_fails_open(tmp_path, caplog):
    path = tmp_path / "segment_health_check.123.7"
    path.mkdir()

    with caplog.at_level(logging.WARNING):
        assert _is_segment_healthy(tmp_path)

    assert f"Ignoring segment health decision: not a regular file: {path}" in caplog.text
