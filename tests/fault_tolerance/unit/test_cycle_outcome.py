# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from nvidia_resiliency_ext.fault_tolerance.cycle_outcome import (
    MAX_EXIT_DETAILS_LEN,
    MAX_SECTION_NAME_LEN,
    MAX_TIMEOUT_SECTIONS,
    CycleEndReason,
    LocalCycleOutcome,
    RankEndReason,
    RankOutcome,
    RankTimeout,
    TimeoutKind,
    build_local_cycle_outcome,
    decode_outcome,
    encode_outcome,
)

# Two local workers: global ranks 4 and 5, PIDs 100 and 101.
WORKERS = [SimpleNamespace(global_rank=4 + i, local_rank=i, id=100 + i) for i in range(2)]
SECTION_TIMEOUT = RankTimeout(TimeoutKind.SECTION, ("step",))


def _build(node_end_reason, failures=None, timeouts=None):
    return build_local_cycle_outcome(
        cycle=3,
        node="node0",
        node_end_reason=node_end_reason,
        workers=WORKERS,
        failures=failures,
        detected_at=12.5,
        timeouts=timeouts,
    )


@pytest.mark.parametrize(
    ("node_end_reason", "failures", "timeouts", "expected"),
    [
        pytest.param(
            CycleEndReason.COMPLETED,
            None,
            None,
            [(RankEndReason.COMPLETED, 0), (RankEndReason.COMPLETED, 0)],
        ),
        pytest.param(
            CycleEndReason.FAILED,
            {5: SimpleNamespace(exitcode=1)},
            None,
            [(RankEndReason.STOPPED_BY_LAUNCHER, None), (RankEndReason.FAILED, 1)],
        ),
        pytest.param(
            CycleEndReason.FAILED,
            {5: SimpleNamespace(exitcode=-9)},
            {5: SECTION_TIMEOUT},
            [(RankEndReason.STOPPED_BY_LAUNCHER, None), (RankEndReason.STOPPED_BY_RMON, -9)],
        ),
        pytest.param(
            CycleEndReason.FAILED,
            {5: SimpleNamespace(exitcode=1)},
            {4: SECTION_TIMEOUT},
            [(RankEndReason.STOPPED_BY_LAUNCHER, None), (RankEndReason.FAILED, 1)],
        ),
        pytest.param(
            CycleEndReason.PEER_RESTART,
            None,
            None,
            [(RankEndReason.STOPPED_BY_LAUNCHER, None), (RankEndReason.STOPPED_BY_LAUNCHER, None)],
        ),
        pytest.param(
            CycleEndReason.TERMINATED,
            {4: SimpleNamespace(exitcode=1)},
            None,
            [(RankEndReason.FAILED, 1), (RankEndReason.STOPPED_BY_LAUNCHER, None)],
        ),
    ],
)
def test_ranks_are_classified(node_end_reason, failures, timeouts, expected):
    outcome = _build(node_end_reason, failures, timeouts)
    assert [(rank.end, rank.exitcode) for rank in outcome.ranks] == expected


def test_outcome_fields():
    outcome = _build(CycleEndReason.FAILED, {5: SimpleNamespace(exitcode=-9)}, {5: SECTION_TIMEOUT})

    assert (outcome.cycle, outcome.node, outcome.node_end, outcome.detected_at) == (
        3,
        "node0",
        CycleEndReason.FAILED,
        12.5,
    )
    assert [(r.global_rank, r.local_rank, r.pid) for r in outcome.ranks] == [
        (4, 0, 100),
        (5, 1, 101),
    ]
    stopped, timed_out = outcome.ranks
    assert (stopped.timeout_kind, stopped.timeout_sections) == (None, ())
    assert (timed_out.timeout_kind, timed_out.timeout_sections) == (TimeoutKind.SECTION, ("step",))


@pytest.mark.parametrize(
    "node_end_reason",
    [CycleEndReason.COMPLETED, CycleEndReason.FAILED, CycleEndReason.PEER_RESTART],
)
def test_encoded_outcome_round_trips(node_end_reason):
    failures = (
        {5: SimpleNamespace(exitcode=-9)} if node_end_reason is CycleEndReason.FAILED else None
    )
    outcome = _build(node_end_reason, failures, {5: SECTION_TIMEOUT})

    record = encode_outcome(outcome)

    assert record.endswith(b"\n") and record.count(b"\n") == 1
    assert decode_outcome(record) == outcome


def test_appended_records_split_on_newlines():
    """Records appended to one store key are recovered line by line."""
    first = _build(CycleEndReason.FAILED, {5: SimpleNamespace(exitcode=1)})
    second = _build(CycleEndReason.PEER_RESTART)
    # A newline in a section name must not split the record.
    third = _build(
        CycleEndReason.FAILED,
        {5: SimpleNamespace(exitcode=-9)},
        {5: RankTimeout(TimeoutKind.SECTION, ("a\nb",))},
    )
    data = encode_outcome(first) + encode_outcome(second) + encode_outcome(third)

    lines = data.splitlines()
    assert [decode_outcome(line) for line in lines] == [first, second, third]


def test_timeout_sections_are_bounded():
    sections = tuple("s" * 100 + str(i) for i in range(MAX_TIMEOUT_SECTIONS + 2))
    outcome = _build(
        CycleEndReason.FAILED,
        {5: SimpleNamespace(exitcode=-9)},
        {5: RankTimeout(TimeoutKind.SECTION, sections)},
    )

    decoded = decode_outcome(encode_outcome(outcome))

    assert decoded.ranks[1].timeout_sections == tuple(
        name[:MAX_SECTION_NAME_LEN] for name in sections[:MAX_TIMEOUT_SECTIONS]
    )


def test_detection_time_keeps_milliseconds():
    outcome = build_local_cycle_outcome(
        3, "node0", CycleEndReason.PEER_RESTART, WORKERS, None, 1.23456
    )
    assert decode_outcome(encode_outcome(outcome)).detected_at == 1.235


@pytest.mark.parametrize(
    "line",
    [
        b"not json",
        b'{"v":1,"c":3,"n":"node0","e":"failed","t":1.0,"r":[]}',
        b'{"v":3,"c":3,"n":"node0","e":"failed","t":1.0,"r":[]}',
        b'{"v":2,"c":3,"n":"node0","e":"exploded","t":1.0,"r":[]}',
        b'{"v":2,"c":3,"n":"node0","e":"failed","t":1.0}',
        b'{"v":2,"c":3,"n":"node0","e":"failed","t":1.0,"r":[[4,0]]}',
        b'{"v":2,"c":3,"n":"node0","e":"failed","t":1.0,"r":[4]}',
    ],
    ids=[
        "not_json",
        "older_version",
        "future_version",
        "unknown_reason",
        "no_ranks",
        "short_rank",
        "bad_rank",
    ],
)
def test_malformed_records_raise_value_error(line):
    with pytest.raises(ValueError):
        decode_outcome(line)


def test_exit_details_are_empty_until_captured():
    outcome = _build(CycleEndReason.FAILED, {5: SimpleNamespace(exitcode=1)})
    assert [rank.exitdetails for rank in outcome.ranks] == ["", ""]


def test_exit_details_round_trip_and_are_bounded():
    rank = RankOutcome(5, 1, 101, RankEndReason.FAILED, 1, exitdetails="x" * 1000)
    outcome = LocalCycleOutcome(3, "node0", CycleEndReason.FAILED, 12.5, (rank,))

    decoded = decode_outcome(encode_outcome(outcome))

    assert decoded.ranks[0].exitdetails == "x" * MAX_EXIT_DETAILS_LEN
