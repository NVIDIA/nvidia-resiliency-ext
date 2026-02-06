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

import pytest

from nvidia_resiliency_ext.fault_tolerance.cycle_info_writer import CycleInfoWriter, utc_iso_now


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
    assert os.path.isfile(path)
    with open(path) as f:
        data = json.load(f)
    assert data["job_id"] == job_id
    assert data.get("attempt_index", 0) == attempt_index
    assert data.get("cycle_number", 0) == cycle_number
    assert data["cycle_start_time"] == "2024-01-01T00:00:00Z"
    assert data.get("cycle_end_time", "") == ""
    assert data["cycle_log_file"] == "/path/to/log"
    assert data["active_nodes"] == "node[001-002]"
    assert data["standby_nodes"] == "node003"

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


def test_utc_iso_now_format():
    """utc_iso_now returns ISO 8601-like string ending with Z."""
    s = utc_iso_now()
    assert isinstance(s, str)
    assert "T" in s
    assert s.endswith("Z") or "+00:00" in s or "Z" in s
