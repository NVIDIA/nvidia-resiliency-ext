# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import time
from types import SimpleNamespace

import pytest

from nvidia_resiliency_ext.fault_tolerance.cycle_info.records import (
    MAX_EXIT_DETAILS_WORKERS,
    MAX_SECTION_NAME_LEN,
    MAX_TIMEOUT_SECTIONS,
    EndReason,
    NodeCycleEndReport,
    RankTimeout,
    TimeoutKind,
    build_node_cycle_end_report,
    now_ms,
)

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


def test_report_fields():
    report = _build(EndReason.FAILED, {5: SimpleNamespace(exitcode=1)})

    assert report == NodeCycleEndReport(
        job_id="job1",
        attempt_index=2,
        cycle_number=3,
        node_name="node0",
        end_reason=EndReason.FAILED,
        detected_timestamp=DETECTED,
        exit_details='[{"rank":5,"local_rank":1,"pid":101,"end":"failed","exitcode":1}]',
    )


@pytest.mark.parametrize(
    "end_reason",
    [
        EndReason.COMPLETED,
        EndReason.TERMINATED_DUE_TO_PEER,
        EndReason.TERMINATED_BY_ATTRIBUTION,
    ],
)
def test_no_failures_means_no_exit_details(end_reason):
    report = _build(end_reason)
    assert (report.end_reason, report.exit_details) == (end_reason, "")


def test_only_failed_workers_are_listed():
    report = _build(EndReason.FAILED, {4: SimpleNamespace(exitcode=1)})
    assert [entry["rank"] for entry in _exit_details(report)] == [4]


def test_rank_monitor_stop_upgrades_failed():
    report = _build(EndReason.FAILED, {5: SimpleNamespace(exitcode=-9)}, {5: SECTION_TIMEOUT})

    assert report.end_reason is EndReason.TERMINATED_BY_RANK_MONITOR
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


def test_timeout_of_a_worker_that_did_not_exit_is_ignored():
    report = _build(EndReason.FAILED, {5: SimpleNamespace(exitcode=1)}, {4: SECTION_TIMEOUT})

    assert report.end_reason is EndReason.FAILED
    assert [entry["end"] for entry in _exit_details(report)] == ["failed"]


@pytest.mark.parametrize(
    "end_reason", [EndReason.TERMINATED_DUE_TO_PEER, EndReason.TERMINATED_BY_ATTRIBUTION]
)
def test_only_failed_is_upgraded(end_reason):
    report = _build(end_reason, {5: SimpleNamespace(exitcode=-9)}, {5: SECTION_TIMEOUT})
    assert report.end_reason is end_reason


def test_timeout_without_sections_has_no_sections_entry():
    timeout = RankTimeout(TimeoutKind.HEARTBEAT)
    report = _build(EndReason.FAILED, {4: SimpleNamespace(exitcode=-9)}, {4: timeout})

    (entry,) = _exit_details(report)
    assert entry["timeout"] == "heartbeat"
    assert "sections" not in entry


def test_sections_are_bounded():
    sections = tuple("s" * (MAX_SECTION_NAME_LEN + 10) + str(i) for i in range(10))
    timeout = RankTimeout(TimeoutKind.SECTION, sections)
    report = _build(EndReason.FAILED, {4: SimpleNamespace(exitcode=-9)}, {4: timeout})

    (entry,) = _exit_details(report)
    assert len(entry["sections"]) == MAX_TIMEOUT_SECTIONS
    assert all(len(name) == MAX_SECTION_NAME_LEN for name in entry["sections"])


def test_exit_details_are_bounded():
    count = MAX_EXIT_DETAILS_WORKERS + 3
    workers = [SimpleNamespace(global_rank=i, local_rank=i, id=1000 + i) for i in range(count)]
    failures = {i: SimpleNamespace(exitcode=1) for i in range(count)}

    entries = _exit_details(_build(EndReason.FAILED, failures, workers=workers))

    assert [entry["rank"] for entry in entries[:-1]] == list(range(MAX_EXIT_DETAILS_WORKERS))
    assert entries[-1] == {"omitted": 3}


def test_now_ms_is_integer_milliseconds():
    before = time.time()
    value = now_ms()
    after = time.time()

    assert isinstance(value, int)
    assert int(before * 1000) - 1 <= value <= int(after * 1000) + 1
