# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import time
from types import SimpleNamespace

import pytest

from nvidia_resiliency_ext.fault_tolerance.cycle_info.messages import (
    MAX_EXIT_DETAILS_WORKERS,
    MAX_SECTION_NAME_LEN,
    MAX_TIMEOUT_SECTIONS,
    active_hosts,
    build_cycle_info,
    build_node_cycle_end_report,
    check_cycle_info,
    check_node_cycle_end_report,
    ended_cycle_info,
    iso_to_seconds,
    now_ms,
)
from nvidia_resiliency_ext.fault_tolerance.data import RankTimeout, TimeoutKind
from nvidia_resiliency_ext.fault_tolerance.utils import range_str_to_ranks
from nvidia_resiliency_ext.shared_utils.proto import nvrx_interface_pb2 as pb

# Nodes as the store host knows them (FQDN, rank order); ranks aligned with them.
START = build_cycle_info(
    job_id="job1",
    attempt_index=2,
    cycle_number=3,
    started_at_ms=1_700_000_000_500,
    cycle_log_file="/logs/train_cycle3.log",
    active_nodes=("node002.cluster", "node001.cluster"),
    active_ranks=(0, 1),
    standby_nodes=("node003.cluster",),
)


class TestCycleInfo:
    def test_fields_are_in_the_wire_and_file_format(self):
        assert (START.job_id, START.attempt_index, START.cycle_number) == ("job1", 2, 3)
        assert START.cycle_start_time == "2023-11-14T22:13:20.500000Z"
        assert START.cycle_end_time == ""
        assert START.cycle_log_file == "/logs/train_cycle3.log"
        # Compressed: hosts sorted and without their domain.
        assert START.active_nodes == "node[001-002]"
        assert START.standby_nodes == "node[003]"
        assert START.timestamp == 1_700_000_000_500

    def test_ranks_follow_the_sorted_hosts(self):
        # node001 has rank 1, node002 rank 0.
        assert range_str_to_ranks(START.active_ranks) == [1, 0]

    def test_active_hosts_are_expanded(self):
        assert active_hosts(START) == ["node001", "node002"]

    def test_without_nodes_the_lists_are_empty(self):
        info = build_cycle_info(job_id="j", attempt_index=0, cycle_number=0, started_at_ms=1)
        assert (info.active_nodes, info.active_ranks, info.standby_nodes) == ("", "", "")

    def test_ended_cycle_info_is_a_copy_with_the_end(self):
        end = ended_cycle_info(START, 1_700_000_100_250)

        assert end.cycle_end_time == "2023-11-14T22:15:00.250000Z"
        assert end.timestamp == 1_700_000_100_250
        assert (end.cycle_start_time, end.active_nodes) == (
            START.cycle_start_time,
            START.active_nodes,
        )
        # The published start is left as it was.
        assert (START.cycle_end_time, START.timestamp) == ("", 1_700_000_000_500)

    def test_iso_times_parse_back(self):
        assert iso_to_seconds(START.cycle_start_time) == 1_700_000_000.5
        assert iso_to_seconds("") is None
        with pytest.raises(ValueError):
            iso_to_seconds("not a time")


class TestChecks:
    def test_a_built_cycle_info_passes(self):
        check_cycle_info(START)
        check_cycle_info(ended_cycle_info(START, 1_700_000_100_250))

    @pytest.mark.parametrize(
        "field, value",
        [
            ("cycle_start_time", ""),
            ("cycle_start_time", "not a time"),
            ("cycle_end_time", "not a time"),
            ("active_nodes", "node[a-b]"),
            ("active_ranks", "x-1"),
        ],
    )
    def test_a_malformed_cycle_info_is_rejected(self, field, value):
        info = pb.CycleInfo()
        info.CopyFrom(START)
        setattr(info, field, value)
        with pytest.raises(ValueError):
            check_cycle_info(info)

    def test_a_built_end_report_passes(self):
        check_node_cycle_end_report(_build(pb.CycleEndReasonWorkerFailed))

    def test_an_end_report_without_detection_time_is_rejected(self):
        report = _build(pb.CycleEndReasonWorkerFailed)
        report.detected_timestamp = 0
        with pytest.raises(ValueError):
            check_node_cycle_end_report(report)

    def test_an_end_report_with_an_unknown_end_reason_is_rejected(self):
        report = _build(pb.CycleEndReasonWorkerFailed)
        report.end_reason = 99  # a value from a newer version
        with pytest.raises(ValueError):
            check_node_cycle_end_report(report)


# Two local workers: global ranks 4 and 5, PIDs 100 and 101.
WORKERS = [SimpleNamespace(global_rank=4 + i, local_rank=i, id=100 + i) for i in range(2)]
SECTION_TIMEOUT = RankTimeout(TimeoutKind.SECTION, ("step",))
DETECTED = 1_700_000_000_123


def _build(end_reason, failures=None, timeouts=None, workers=WORKERS):
    return build_node_cycle_end_report(
        job_id="job1",
        attempt_index=2,
        cycle=3,
        node="node0",
        end_reason=end_reason,
        workers=workers,
        failures=failures,
        detected_timestamp=DETECTED,
        timeouts=timeouts,
    )


def _exit_details(report):
    return json.loads(report.exit_details) if report.exit_details else []


class TestNodeCycleEndReport:
    def test_report_fields(self):
        report = _build(pb.CycleEndReasonWorkerFailed, {5: SimpleNamespace(exitcode=1)})

        assert report == pb.NodeCycleEndReport(
            job_id="job1",
            attempt_index=2,
            cycle_number=3,
            node_name="node0",
            end_reason=pb.CycleEndReasonWorkerFailed,
            detected_timestamp=DETECTED,
            exit_details='[{"rank":5,"local_rank":1,"pid":101,"end":"failed","exitcode":1}]',
        )

    @pytest.mark.parametrize(
        "end_reason",
        [
            pb.CycleEndReasonWorkerCompleted,
            pb.CycleEndReasonWorkerTerminatedDueToPeer,
            pb.CycleEndReasonWorkerTerminatedByAttribution,
        ],
    )
    def test_no_failures_means_no_exit_details(self, end_reason):
        report = _build(end_reason)
        assert (report.end_reason, report.exit_details) == (end_reason, "")

    def test_only_failed_workers_are_listed(self):
        report = _build(pb.CycleEndReasonWorkerFailed, {4: SimpleNamespace(exitcode=1)})
        assert [entry["rank"] for entry in _exit_details(report)] == [4]

    def test_rank_monitor_stop_upgrades_failed(self):
        report = _build(
            pb.CycleEndReasonWorkerFailed, {5: SimpleNamespace(exitcode=-9)}, {5: SECTION_TIMEOUT}
        )

        assert report.end_reason == pb.CycleEndReasonWorkerTerminatedByRankMonitor
        assert _exit_details(report) == [
            {
                "rank": 5,
                "local_rank": 1,
                "pid": 101,
                "end": "stopped_by_rank_monitor",
                "exitcode": -9,
                "timeout": "section",
                "sections": ["step"],
            }
        ]

    def test_timeout_of_a_worker_that_did_not_exit_is_ignored(self):
        report = _build(
            pb.CycleEndReasonWorkerFailed, {5: SimpleNamespace(exitcode=1)}, {4: SECTION_TIMEOUT}
        )

        assert report.end_reason == pb.CycleEndReasonWorkerFailed
        assert [entry["end"] for entry in _exit_details(report)] == ["failed"]

    @pytest.mark.parametrize(
        "end_reason",
        [
            pb.CycleEndReasonWorkerTerminatedDueToPeer,
            pb.CycleEndReasonWorkerTerminatedByAttribution,
        ],
    )
    def test_only_failed_is_upgraded(self, end_reason):
        report = _build(end_reason, {5: SimpleNamespace(exitcode=-9)}, {5: SECTION_TIMEOUT})
        assert report.end_reason == end_reason

    def test_timeout_without_sections_has_no_sections_entry(self):
        timeout = RankTimeout(TimeoutKind.HEARTBEAT)
        report = _build(
            pb.CycleEndReasonWorkerFailed, {4: SimpleNamespace(exitcode=-9)}, {4: timeout}
        )

        (entry,) = _exit_details(report)
        assert entry["timeout"] == "heartbeat"
        assert "sections" not in entry

    def test_sections_are_bounded(self):
        sections = tuple("s" * (MAX_SECTION_NAME_LEN + 10) + str(i) for i in range(10))
        timeout = RankTimeout(TimeoutKind.SECTION, sections)
        report = _build(
            pb.CycleEndReasonWorkerFailed, {4: SimpleNamespace(exitcode=-9)}, {4: timeout}
        )

        (entry,) = _exit_details(report)
        assert len(entry["sections"]) == MAX_TIMEOUT_SECTIONS
        assert all(len(name) == MAX_SECTION_NAME_LEN for name in entry["sections"])

    def test_exit_details_are_bounded(self):
        count = MAX_EXIT_DETAILS_WORKERS + 3
        workers = [SimpleNamespace(global_rank=i, local_rank=i, id=1000 + i) for i in range(count)]
        failures = {i: SimpleNamespace(exitcode=1) for i in range(count)}

        entries = _exit_details(_build(pb.CycleEndReasonWorkerFailed, failures, workers=workers))

        assert [entry["rank"] for entry in entries[:-1]] == list(range(MAX_EXIT_DETAILS_WORKERS))
        assert entries[-1] == {"omitted": 3}


def test_now_ms_is_integer_milliseconds():
    before = time.time()
    value = now_ms()
    after = time.time()

    assert isinstance(value, int)
    assert int(before * 1000) - 1 <= value <= int(after * 1000) + 1
