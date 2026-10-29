# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from nvidia_resiliency_ext.fault_tolerance.cycle_outcome import (
    CycleEndReason,
    RankEndReason,
    RankTimeout,
    TimeoutKind,
    build_local_cycle_outcome,
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
