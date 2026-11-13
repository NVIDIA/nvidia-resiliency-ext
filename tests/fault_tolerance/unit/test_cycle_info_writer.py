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

from nvidia_resiliency_ext.fault_tolerance.cycle_info.records import (
    CycleSummary,
    EndReason,
    JobEndReason,
    NodeCycleEndReport,
)
from nvidia_resiliency_ext.fault_tolerance.cycle_info_writer import (
    CycleInfoReporter,
    CycleInfoRoundSnapshot,
    CycleInfoWriter,
    utc_iso_ms,
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
                "SLURM_RESTART_COUNT": "4",
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
        assert reporter.report_cycle_start(CycleInfoRoundSnapshot(cycle_number=0))

    call_kw = writer.write_cycle_start.call_args.kwargs
    assert call_kw["job_id"] == "array-job"
    assert call_kw["attempt_index"] == 0


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
        assert reporter.report_cycle_start(CycleInfoRoundSnapshot(cycle_number=0))

    call_kw = writer.write_cycle_start.call_args.kwargs
    assert call_kw["job_id"] == "array-job"
    assert call_kw["attempt_index"] == 0
    assert "Invalid SLURM_RESTART_COUNT='invalid' for cycle info; using 0" in caplog.text


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


def _report(node, end_reason, detected_timestamp, exit_details=""):
    return NodeCycleEndReport("job1", 0, 3, node, end_reason, detected_timestamp, exit_details)


def _end_report_dict(node, detected_timestamp):
    return {
        "job_id": "job1",
        "attempt_index": 0,
        "cycle_number": 3,
        "node_name": node,
        "end_reason": "CycleEndReasonWorkerFailed",
        "detected_timestamp": detected_timestamp,
        "exit_details": "",
    }


def test_cycle_info_writer_update_cycle_end_report(tmp_dir):
    """The end report is added to the existing file; a second update replaces it."""
    writer = CycleInfoWriter(tmp_dir)
    writer.write_cycle_start("job1", 0, 3, "2024-01-01T00:00:00Z", "/log", "node[001-002]")
    writer.update_cycle_end("job1", 0, 3, "2024-01-01T00:01:00Z")

    def update(reports):
        return lambda: {
            "end_report": {
                "end_reason": "CycleEndReasonWorkerFailed",
                "ended_timestamp": "2024-01-01T00:00:59.000Z",
                "first_detected_nodes": [reports[0]["node_name"]],
                "launcher_end_reports": reports,
            },
            "not_a_field": "ignored",
        }

    writer.update_cycle_later("job1", 0, 3, update([_end_report_dict("node001", 1)]))
    writer.update_cycle_later("job1", 0, 3, update([_end_report_dict("node002", 1704067199000)]))
    writer.update_cycle_later("job1", 0, 3, lambda: None)  # nothing to add: file unchanged
    writer.shutdown()

    with open(os.path.join(tmp_dir, "cycle_info.job1.0.3")) as f:
        data = json.load(f)
    assert data["cycle_end_time"] == "2024-01-01T00:01:00Z"
    assert "not_a_field" not in data
    end_report = data["end_report"]
    assert end_report["end_reason"] == "CycleEndReasonWorkerFailed"
    assert end_report["first_detected_nodes"] == ["node002"]
    # 64-bit integers are strings in proto3 JSON.
    expected = dict(_end_report_dict("node002", "1704067199000"))
    assert end_report["launcher_end_reports"] == [expected]
    assert data["generation"] == "3"  # create, end, and two end report updates


def _reporter_with_cycle(tmp_dir, **kwargs):
    writer = MagicMock()
    reporter = CycleInfoReporter(
        tmp_dir, cycle_info_job_id="job1", attempt_index=0, writer=writer, **kwargs
    )
    reporter.report_cycle_start(
        CycleInfoRoundSnapshot(
            cycle_number=3,
            active_node_addrs=["node001", "node002", "node003"],
            active_ranks=[0, 1, 2],
        )
    )
    return reporter, writer


def _summary(
    outcomes,
    end_reason=JobEndReason.RANK_FAILURE,
    first_detected=("node001",),
    ended_at=10_250,
):
    return CycleSummary(
        cycle=3,
        outcomes=tuple(outcomes),
        launchers_expected=3,
        missing=("node004",),
        end_reason=end_reason,
        first_detected=first_detected,
        ended_at=ended_at,
    )


def test_cycle_info_reporter_reports_cycle_summary(tmp_dir):
    """One update: the end time and the end report; launchers that observed the end are
    listed by detection time, peers left out."""
    reporter, writer = _reporter_with_cycle(tmp_dir)
    late = _report("node003", EndReason.TERMINATED_BY_RANK_MONITOR, 20_500, '[{"rank":9}]')
    early = _report("node001", EndReason.FAILED, 10_250, '[{"rank":0}]')
    peer = _report("node002", EndReason.TERMINATED_DUE_TO_PEER, 11_000)

    assert reporter.report_cycle_summary(_summary([late, peer, early]))

    job_id, attempt_index, cycle_number, build_update = writer.update_cycle_later.call_args.args
    assert (job_id, attempt_index, cycle_number) == ("job1", 0, 3)
    payload = build_update()
    assert payload["cycle_end_time"] == "1970-01-01T00:00:10.250Z"
    assert payload["end_report"] == {
        # The first detector's reason.
        "end_reason": "CycleEndReasonWorkerFailed",
        "ended_timestamp": "1970-01-01T00:00:10.250Z",
        "first_detected_nodes": ["node001"],
        "launcher_end_reports": [
            {
                "job_id": "job1",
                "attempt_index": 0,
                "cycle_number": 3,
                "node_name": "node001",
                "end_reason": "CycleEndReasonWorkerFailed",
                "detected_timestamp": 10_250,
                "exit_details": '[{"rank":0}]',
            },
            {
                "job_id": "job1",
                "attempt_index": 0,
                "cycle_number": 3,
                "node_name": "node003",
                "end_reason": "CycleEndReasonWorkerTerminatedByRankMonitor",
                "detected_timestamp": 20_500,
                "exit_details": '[{"rank":9}]',
            },
        ],
    }
    # Ended by the summary, so shutdown does not stamp the cycle again.
    reporter.shutdown()
    writer.update_cycle_end.assert_not_called()


def test_cycle_info_reporter_completed_summary(tmp_dir):
    reporter, writer = _reporter_with_cycle(tmp_dir)
    reports = [_report(f"node00{i}", EndReason.COMPLETED, 10_000 + i) for i in (1, 2, 3)]

    reporter.report_cycle_summary(
        _summary(reports, JobEndReason.COMPLETED, first_detected=(), ended_at=10_003)
    )

    end_report = writer.update_cycle_later.call_args.args[3]()["end_report"]
    assert end_report["end_reason"] == "CycleEndReasonWorkerCompleted"
    assert [r["node_name"] for r in end_report["launcher_end_reports"]] == [
        "node001",
        "node002",
        "node003",
    ]


def test_cycle_info_reporter_summary_without_first_detector(tmp_dir):
    """Only peer restarts: the reason is unspecified and no launcher is listed."""
    reporter, writer = _reporter_with_cycle(tmp_dir)
    peer = _report("node002", EndReason.TERMINATED_DUE_TO_PEER, 11_000)

    reporter.report_cycle_summary(_summary([peer], JobEndReason.UNKNOWN, first_detected=()))

    end_report = writer.update_cycle_later.call_args.args[3]()["end_report"]
    assert end_report["end_reason"] == "CycleEndReasonUnspecified"
    assert end_report["launcher_end_reports"] == []


def test_cycle_info_reporter_summary_without_end_time_keeps_the_cycle_end(tmp_dir):
    reporter, writer = _reporter_with_cycle(tmp_dir)

    reporter.report_cycle_summary(_summary([], JobEndReason.UNKNOWN, (), ended_at=None))

    payload = writer.update_cycle_later.call_args.args[3]()
    assert "cycle_end_time" not in payload
    assert payload["end_report"]["ended_timestamp"] == ""


def test_utc_iso_ms_is_exact():
    assert utc_iso_ms(0) == "1970-01-01T00:00:00.000Z"
    assert utc_iso_ms(1_700_000_000_123) == "2023-11-14T22:13:20.123Z"


def test_cycle_info_reporter_ignores_summary_of_unstarted_cycle(tmp_dir):
    writer = MagicMock()
    reporter = CycleInfoReporter(tmp_dir, cycle_info_job_id="job1", attempt_index=0, writer=writer)
    assert not reporter.report_cycle_summary(_summary([]))
    writer.update_cycle_later.assert_not_called()


def test_cycle_info_reporter_uses_the_snapshot_start_time(tmp_dir):
    writer = MagicMock()
    reporter = CycleInfoReporter(tmp_dir, cycle_info_job_id="job1", attempt_index=0, writer=writer)
    reporter.report_cycle_start(CycleInfoRoundSnapshot(cycle_number=3, started_at=10.25))
    assert (
        writer.write_cycle_start.call_args.kwargs["cycle_start_time"]
        == "1970-01-01T00:00:10.250000Z"
    )


def test_cycle_info_reporter_ends_a_given_cycle_at_a_given_time(tmp_dir):
    reporter, writer = _reporter_with_cycle(tmp_dir, end_previous_on_start=False)
    reporter.report_cycle_start(CycleInfoRoundSnapshot(cycle_number=4))
    writer.update_cycle_end.assert_not_called()  # starting 4 did not end 3

    assert reporter.report_cycle_end(3, ended_at=10.25)
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


def test_utc_iso_now_format():
    """utc_iso_now returns ISO 8601-like string ending with Z."""
    s = utc_iso_now()
    assert isinstance(s, str)
    assert "T" in s
    assert s.endswith("Z") or "+00:00" in s or "Z" in s
