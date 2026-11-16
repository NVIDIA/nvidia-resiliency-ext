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
import logging
import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from nvidia_resiliency_ext.fault_tolerance.cycle_info.messages import build_cycle_info
from nvidia_resiliency_ext.fault_tolerance.cycle_info_writer import (
    CycleInfoReporter,
    CycleInfoWriter,
    utc_iso_ms,
    utc_iso_now,
)
from nvidia_resiliency_ext.shared_utils.proto import nvrx_interface_pb2 as pb


@pytest.fixture
def tmp_dir():
    """Temporary directory for cycle info files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


def _file_info(job_id, cycle_number, **fields):
    """A CycleInfo as the writer gets it: in its file format, named by its own fields."""
    return pb.CycleInfo(
        job_id=job_id,
        attempt_index=0,
        cycle_number=cycle_number,
        cycle_start_time="2024-01-01T00:00:00Z",
        **fields,
    )


def _store_host_info(cycle_number=0, **kwargs):
    """A CycleInfo as the store host publishes it, in its own (environment) namespace."""
    kwargs.setdefault("active_nodes", ["node002", "node001"])
    kwargs.setdefault("active_ranks", [1, 0])
    return build_cycle_info(
        job_id="store-host-job",
        attempt_index=9,
        cycle_number=cycle_number,
        started_at_ms=10_250,
        **kwargs,
    )


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
    """write_cycle_start writes the CycleInfo as JSON and points the .current symlink at it."""
    writer = CycleInfoWriter(tmp_dir)
    job_id = "job1"
    attempt_index = 0
    cycle_number = 0
    writer.write_cycle_start(
        _file_info(
            job_id,
            cycle_number,
            cycle_log_file="/path/to/log",
            active_nodes="node[001-002]",
            standby_nodes="node003",
            generation=5,  # reset to 0 by the writer
        )
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
    writer.write_cycle_start(_file_info(job_id, cycle_number, active_nodes="node001"))
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


def _launcher_end_report(node, detected_timestamp):
    return pb.NodeCycleEndReport(
        job_id="job1",
        cycle_number=3,
        node_name=node,
        end_reason=pb.CycleEndReasonWorkerFailed,
        detected_timestamp=detected_timestamp,
    )


def _job_end_report(node, detected_timestamp):
    return pb.JobCycleEndReport(
        end_reason=pb.CycleEndReasonWorkerFailed,
        first_detected_timestamp="2024-01-01T00:00:59.000Z",
        first_detected_nodes=[node],
        launcher_end_reports=[_launcher_end_report(node, detected_timestamp)],
    )


def test_cycle_info_writer_update_cycle_end_report(tmp_dir):
    """The end report is added to the existing file; a second update replaces it."""
    writer = CycleInfoWriter(tmp_dir)
    writer.write_cycle_start(_file_info("job1", 3, active_nodes="node[001-002]"))
    writer.update_cycle_end("job1", 0, 3, "2024-01-01T00:01:00Z")

    def update(node, detected_timestamp):
        return lambda: {
            "end_report": _job_end_report(node, detected_timestamp),
            "not_a_field": "ignored",
        }

    writer.update_cycle_later("job1", 0, 3, update("node001", 1))
    writer.update_cycle_later("job1", 0, 3, update("node002", 1704067199000))
    writer.update_cycle_later("job1", 0, 3, lambda: None)  # nothing to add: file unchanged
    writer.shutdown()

    with open(os.path.join(tmp_dir, "cycle_info.job1.0.3")) as f:
        data = json.load(f)
    assert data["cycle_end_time"] == "2024-01-01T00:01:00Z"
    assert "not_a_field" not in data
    end_report = data["end_report"]
    assert end_report["end_reason"] == "CycleEndReasonWorkerFailed"
    assert end_report["first_detected_nodes"] == ["node002"]
    # Enum values by name; 64-bit integers are strings in proto3 JSON.
    assert end_report["launcher_end_reports"] == [
        {
            "job_id": "job1",
            "attempt_index": 0,
            "cycle_number": 3,
            "node_name": "node002",
            "end_reason": "CycleEndReasonWorkerFailed",
            "detected_timestamp": "1704067199000",
            "exit_details": "",
        }
    ]
    assert data["generation"] == "3"  # create, end, and two end report updates


def test_cycle_info_reporter_writes_the_store_host_start_in_its_namespace(tmp_dir):
    """The start is written as the store host built it, with the reporter's file namespace and
    the cycle log file of its prefix; the received message is left unmodified."""
    writer = MagicMock()
    reporter = CycleInfoReporter(
        tmp_dir,
        cycle_log_prefix="/logs/train.log",
        cycle_info_job_id="job1",
        attempt_index=7,
        writer=writer,
    )
    info = _store_host_info(standby_nodes=["node003"])

    assert reporter.report_cycle_start(info)

    (written,) = writer.write_cycle_start.call_args.args
    assert written is not info
    assert (written.job_id, written.attempt_index, written.cycle_number) == ("job1", 7, 0)
    assert written.cycle_start_time == info.cycle_start_time
    assert written.cycle_log_file == "/logs/train_cycle0.log"
    # Compressed once, by the store host.
    assert (written.active_nodes, written.standby_nodes) == ("node[001-002]", "node[003]")
    assert written.active_ranks == "0-1"
    assert (info.job_id, info.attempt_index, info.cycle_log_file) == ("store-host-job", 9, "")


def test_cycle_info_reporter_keeps_the_store_host_cycle_log_file(tmp_dir):
    writer = MagicMock()
    reporter = CycleInfoReporter(
        tmp_dir, cycle_log_prefix="/logs/train.log", cycle_info_job_id="job1", writer=writer
    )
    assert reporter.report_cycle_start(_store_host_info(cycle_log_file="/other/log_cycle0.log"))
    assert writer.write_cycle_start.call_args.args[0].cycle_log_file == "/other/log_cycle0.log"


def test_cycle_info_reporter_derives_file_namespace_from_env(tmp_dir):
    """Colocated reporter derives cycle-info file naming from its SLURM environment."""
    writer = MagicMock()
    reporter = CycleInfoReporter(tmp_dir, writer=writer)

    with patch.dict(
        os.environ,
        {
            "SLURM_ARRAY_JOB_ID": "array-job",
            "SLURM_JOB_ID": "job",
            "SLURM_RESTART_COUNT": "4",
        },
        clear=False,
    ):
        assert reporter.report_cycle_start(_store_host_info())

    written = writer.write_cycle_start.call_args.args[0]
    assert written.job_id == "array-job"
    assert written.attempt_index == 4


def test_cycle_info_reporter_defaults_missing_restart_count_to_zero(tmp_dir):
    writer = MagicMock()
    reporter = CycleInfoReporter(tmp_dir, writer=writer)

    with patch.dict(
        os.environ,
        {
            "SLURM_ARRAY_JOB_ID": "array-job",
            "SLURM_JOB_ID": "job",
        },
        clear=True,
    ):
        assert reporter.report_cycle_start(_store_host_info())

    written = writer.write_cycle_start.call_args.args[0]
    assert written.job_id == "array-job"
    assert written.attempt_index == 0


def test_cycle_info_reporter_invalid_restart_count_falls_back_to_zero(
    tmp_dir, caplog, capture_nvrx_logs
):
    writer = MagicMock()
    reporter = CycleInfoReporter(tmp_dir, writer=writer)

    with (
        patch.dict(
            os.environ,
            {
                "SLURM_ARRAY_JOB_ID": "array-job",
                "SLURM_JOB_ID": "job",
                "SLURM_RESTART_COUNT": "invalid",
            },
            clear=True,
        ),
        caplog.at_level(logging.WARNING),
    ):
        assert reporter.report_cycle_start(_store_host_info())

    written = writer.write_cycle_start.call_args.args[0]
    assert written.job_id == "array-job"
    assert written.attempt_index == 0
    assert "Invalid SLURM_RESTART_COUNT='invalid' for cycle info; using 0" in caplog.text


def test_cycle_info_reporter_ignores_a_second_start_of_a_cycle(tmp_dir):
    writer = MagicMock()
    reporter = CycleInfoReporter(tmp_dir, cycle_info_job_id="job1", writer=writer)
    assert reporter.report_cycle_start(_store_host_info(3))
    assert not reporter.report_cycle_start(_store_host_info(3))
    writer.write_cycle_start.assert_called_once()


def test_cycle_info_reporter_closes_cycle_on_shutdown(tmp_dir):
    """CycleInfoReporter shutdown finalizes the active cycle and drains the writer."""
    writer = MagicMock()
    reporter = CycleInfoReporter(
        tmp_dir,
        cycle_info_job_id="job1",
        attempt_index=2,
        writer=writer,
    )

    with patch(
        "nvidia_resiliency_ext.fault_tolerance.cycle_info_writer.utc_iso_now",
        return_value="2024-01-01T00:01:00Z",
    ):
        reporter.report_cycle_start(_store_host_info(3))
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

    with patch(
        "nvidia_resiliency_ext.fault_tolerance.cycle_info_writer.utc_iso_now",
        return_value="2024-01-01T00:01:00Z",
    ):
        reporter.report_cycle_start(_store_host_info(0, active_nodes=["node001"], active_ranks=[0]))
        reporter.report_cycle_start(_store_host_info(1, active_nodes=["node002"], active_ranks=[0]))

    writer.update_cycle_end.assert_called_once_with(
        job_id="job1",
        attempt_index=2,
        cycle_number=0,
        cycle_end_time="2024-01-01T00:01:00Z",
    )
    assert writer.write_cycle_start.call_count == 2
    assert writer.write_cycle_start.call_args_list[1].args[0].cycle_number == 1


def _reporter_with_cycle(tmp_dir, **kwargs):
    writer = MagicMock()
    reporter = CycleInfoReporter(
        tmp_dir, cycle_info_job_id="job1", attempt_index=0, writer=writer, **kwargs
    )
    reporter.report_cycle_start(_store_host_info(3))
    return reporter, writer


def test_cycle_info_reporter_reports_the_job_end_report(tmp_dir):
    """One update: the end report only, not the cycle end time."""
    reporter, writer = _reporter_with_cycle(tmp_dir)
    end_report = _job_end_report("node001", 10_250)

    assert reporter.report_cycle_end_report(3, end_report)

    job_id, attempt_index, cycle_number, build_update = writer.update_cycle_later.call_args.args
    assert (job_id, attempt_index, cycle_number) == ("job1", 0, 3)
    payload = build_update()
    assert payload == {"end_report": end_report}  # the store host's end is not in it
    # Its end reports are written, so shutdown does not stamp the cycle.
    reporter.shutdown()
    writer.update_cycle_end.assert_not_called()


def test_cycle_info_reporter_ignores_the_end_report_of_an_unstarted_cycle(tmp_dir):
    writer = MagicMock()
    reporter = CycleInfoReporter(tmp_dir, cycle_info_job_id="job1", attempt_index=0, writer=writer)
    assert not reporter.report_cycle_end_report(3, pb.JobCycleEndReport())
    writer.update_cycle_later.assert_not_called()


def test_cycle_info_reporter_ends_a_given_cycle_at_a_given_time(tmp_dir):
    reporter, writer = _reporter_with_cycle(tmp_dir, end_previous_on_start=False)
    reporter.report_cycle_start(_store_host_info(4))
    writer.update_cycle_end.assert_not_called()  # starting 4 did not end 3

    assert reporter.report_cycle_end(3, "1970-01-01T00:00:10.250000Z")
    assert not reporter.report_cycle_end(7)  # never started

    writer.update_cycle_end.assert_called_once_with(
        job_id="job1",
        attempt_index=0,
        cycle_number=3,
        cycle_end_time="1970-01-01T00:00:10.250000Z",
    )
    # Cycle 4 is still open: shutdown ends it.
    reporter.shutdown()
    assert writer.update_cycle_end.call_args.kwargs["cycle_number"] == 4


def test_utc_iso_ms_is_exact():
    assert utc_iso_ms(0) == "1970-01-01T00:00:00.000Z"
    assert utc_iso_ms(1_700_000_000_123) == "2023-11-14T22:13:20.123Z"


def test_utc_iso_now_format():
    """utc_iso_now returns ISO 8601-like string ending with Z."""
    s = utc_iso_now()
    assert isinstance(s, str)
    assert "T" in s
    assert s.endswith("Z") or "+00:00" in s or "Z" in s
