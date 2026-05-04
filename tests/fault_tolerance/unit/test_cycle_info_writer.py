# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for cycle_info_writer module."""

import json
import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from nvidia_resiliency_ext.fault_tolerance.cycle_info_writer import (
    CycleInfoReporter,
    CycleInfoRoundSnapshot,
    CycleInfoWriter,
    utc_iso_now,
)


@pytest.fixture
def tmp_dir():
    """Temporary directory for cycle info files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_cycle_info_writer_init_empty_dir_raises():
    """CycleInfoWriter raises ValueError for empty or whitespace cycle_info_dir."""
    with pytest.raises(ValueError, match="cycle_info_dir must be non-empty"):
        CycleInfoWriter("")
    with pytest.raises(ValueError, match="cycle_info_dir must be non-empty"):
        CycleInfoWriter("   ")


def test_cycle_info_writer_get_cycle_info_dir(tmp_dir):
    """get_cycle_info_dir returns the normalized absolute path."""
    writer = CycleInfoWriter(tmp_dir)
    assert writer.get_cycle_info_dir() == os.path.abspath(tmp_dir.rstrip("/"))
    writer.shutdown()


def test_cycle_info_writer_get_current_cycle_info_path(tmp_dir):
    """get_current_cycle_info_path returns path to cycle_info.<job_id>.current."""
    writer = CycleInfoWriter(tmp_dir)
    path = writer.get_current_cycle_info_path("job123")
    assert path == os.path.join(os.path.abspath(tmp_dir.rstrip("/")), "cycle_info.job123.current")
    writer.shutdown()


def test_cycle_info_writer_write_cycle_start_creates_file_and_symlink(tmp_dir):
    """write_cycle_start creates JSON file and .current symlink."""
    writer = CycleInfoWriter(tmp_dir)
    job_id = "job1"
    attempt_index = 0
    cycle_number = 0
    writer.write_cycle_start(
        job_id=job_id,
        attempt_index=attempt_index,
        cycle_number=cycle_number,
        cycle_start_time="2024-01-01T00:00:00Z",
        cycle_log_file="/path/to/log",
        active_nodes="node[001-002]",
        standby_nodes="node003",
    )
    writer.shutdown()

    filename = f"cycle_info.{job_id}.{attempt_index}.{cycle_number}"
    path = os.path.join(writer.get_cycle_info_dir(), filename)
    # Wait for worker to finish writing (avoids race with thread + FS sync)
    for _ in range(50):
        if os.path.isfile(path):
            break
        time.sleep(0.02)
    assert os.path.isfile(path), f"cycle info file not found at {path}"
    with open(path) as f:
        data = json.load(f)
    assert data["job_id"] == job_id
    assert data.get("attempt_index", 0) == attempt_index
    assert data.get("cycle_number", 0) == cycle_number
    assert data["cycle_start_time"] == "2024-01-01T00:00:00Z"
    # shutdown() may set cycle_end_time for the current cycle; allow "" or ISO timestamp
    cycle_end = data.get("cycle_end_time", "")
    assert isinstance(cycle_end, str)
    if cycle_end:
        assert "T" in cycle_end and (cycle_end.endswith("Z") or "+00:00" in cycle_end)
    assert data["cycle_log_file"] == "/path/to/log"
    assert data["active_nodes"] == "node[001-002]"
    assert data["standby_nodes"] == "node003"
    # generation 0 = no update yet; 1 = shutdown() applied cycle_end update
    assert int(data.get("generation", 0)) in (0, 1)

    symlink_path = os.path.join(writer.get_cycle_info_dir(), f"cycle_info.{job_id}.current")
    assert os.path.islink(symlink_path)
    assert os.path.realpath(symlink_path) == os.path.realpath(path)


def test_cycle_info_writer_update_cycle_end(tmp_dir):
    """update_cycle_end updates existing file with cycle_end_time."""
    writer = CycleInfoWriter(tmp_dir)
    job_id = "job2"
    attempt_index = 0
    cycle_number = 0
    writer.write_cycle_start(
        job_id=job_id,
        attempt_index=attempt_index,
        cycle_number=cycle_number,
        cycle_start_time="2024-01-01T00:00:00Z",
        cycle_log_file="/log",
        active_nodes="node001",
        standby_nodes="",
    )
    end_time = "2024-01-01T01:00:00Z"
    writer.update_cycle_end(
        job_id=job_id,
        attempt_index=attempt_index,
        cycle_number=cycle_number,
        cycle_end_time=end_time,
    )
    writer.shutdown()

    filename = f"cycle_info.{job_id}.{attempt_index}.{cycle_number}"
    path = os.path.join(tmp_dir, filename)
    with open(path) as f:
        data = json.load(f)
    assert data["cycle_end_time"] == end_time
    assert data["cycle_start_time"] == "2024-01-01T00:00:00Z"
    assert int(data["generation"]) == 1


def test_cycle_info_writer_update_cycle_end_missing_file_no_crash(tmp_dir):
    """update_cycle_end on non-existent file does not raise (logs warning)."""
    writer = CycleInfoWriter(tmp_dir)
    writer.update_cycle_end(
        job_id="nonexistent",
        attempt_index=0,
        cycle_number=99,
        cycle_end_time="2024-01-01T00:00:00Z",
    )
    writer.shutdown()
    # No file created for the update target
    path = os.path.join(tmp_dir, "cycle_info.nonexistent.0.99")
    assert not os.path.isfile(path)


def test_cycle_info_reporter_formats_and_starts_cycle(tmp_dir):
    """CycleInfoReporter formats round snapshots and writes cycle start."""
    writer = MagicMock()
    reporter = CycleInfoReporter(
        tmp_dir,
        cycle_log_prefix="/logs/train.log",
        cycle_info_job_id="job1",
        attempt_index=7,
        writer=writer,
    )
    snapshot = CycleInfoRoundSnapshot(
        cycle_number=0,
        active_node_addrs=["node002", "node001"],
        standby_node_addrs=["node003"],
        active_ranks=[1, 0],
    )

    with patch(
        "nvidia_resiliency_ext.fault_tolerance.cycle_info_writer.utc_iso_now",
        return_value="2024-01-01T00:00:00Z",
    ):
        assert reporter.report_cycle_start(snapshot)

    writer.write_cycle_start.assert_called_once()
    call_kw = writer.write_cycle_start.call_args.kwargs
    assert call_kw["job_id"] == "job1"
    assert call_kw["attempt_index"] == 7
    assert call_kw["cycle_number"] == 0
    assert call_kw["cycle_start_time"] == "2024-01-01T00:00:00Z"
    assert call_kw["cycle_log_file"] == "/logs/train_cycle0.log"
    assert call_kw["active_nodes"] == "node[001-002]"
    assert call_kw["standby_nodes"] == "node[003]"
    assert call_kw["active_ranks"] == "0-1"


def test_cycle_info_reporter_derives_file_namespace_from_env(tmp_dir):
    """Colocated reporter derives cycle-info file naming from its SLURM environment."""
    writer = MagicMock()
    reporter = CycleInfoReporter(tmp_dir, writer=writer)
    snapshot = CycleInfoRoundSnapshot(cycle_number=0, active_node_addrs=["node001"])

    with (
        patch.dict(
            os.environ,
            {
                "SLURM_ARRAY_JOB_ID": "array-job",
                "SLURM_JOB_ID": "job",
                "SLURM_RESTART_CNT": "4",
            },
            clear=False,
        ),
        patch(
            "nvidia_resiliency_ext.fault_tolerance.cycle_info_writer.utc_iso_now",
            return_value="2024-01-01T00:00:00Z",
        ),
    ):
        assert reporter.report_cycle_start(snapshot)

    call_kw = writer.write_cycle_start.call_args.kwargs
    assert call_kw["job_id"] == "array-job"
    assert call_kw["attempt_index"] == 4


def test_cycle_info_reporter_closes_cycle_on_shutdown(tmp_dir):
    """CycleInfoReporter shutdown finalizes the active cycle and drains the writer."""
    writer = MagicMock()
    reporter = CycleInfoReporter(
        tmp_dir,
        cycle_info_job_id="job1",
        attempt_index=2,
        writer=writer,
    )
    snapshot = CycleInfoRoundSnapshot(cycle_number=3, active_node_addrs=["node001"])

    with patch(
        "nvidia_resiliency_ext.fault_tolerance.cycle_info_writer.utc_iso_now",
        side_effect=[
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:01:00Z",
        ],
    ):
        reporter.report_cycle_start(snapshot)
        reporter.shutdown()

    writer.update_cycle_end.assert_called_once_with(
        job_id="job1",
        attempt_index=2,
        cycle_number=3,
        cycle_end_time="2024-01-01T00:01:00Z",
    )
    writer.shutdown.assert_called_once_with()


def test_cycle_info_reporter_closes_previous_cycle_on_next_start(tmp_dir):
    """Starting cycle N+1 finalizes cycle N before writing the new start file."""
    writer = MagicMock()
    reporter = CycleInfoReporter(
        tmp_dir,
        cycle_info_job_id="job1",
        attempt_index=2,
        writer=writer,
    )
    snapshot0 = CycleInfoRoundSnapshot(cycle_number=0, active_node_addrs=["node001"])
    snapshot1 = CycleInfoRoundSnapshot(cycle_number=1, active_node_addrs=["node002"])

    with patch(
        "nvidia_resiliency_ext.fault_tolerance.cycle_info_writer.utc_iso_now",
        side_effect=[
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:01:00Z",
            "2024-01-01T00:02:00Z",
        ],
    ):
        reporter.report_cycle_start(snapshot0)
        reporter.report_cycle_start(snapshot1)

    writer.update_cycle_end.assert_called_once_with(
        job_id="job1",
        attempt_index=2,
        cycle_number=0,
        cycle_end_time="2024-01-01T00:01:00Z",
    )
    assert writer.write_cycle_start.call_count == 2
    assert writer.write_cycle_start.call_args_list[1].kwargs["cycle_number"] == 1


def test_utc_iso_now_format():
    """utc_iso_now returns ISO 8601-like string ending with Z."""
    s = utc_iso_now()
    assert isinstance(s, str)
    assert "T" in s
    assert s.endswith("Z") or "+00:00" in s or "Z" in s
