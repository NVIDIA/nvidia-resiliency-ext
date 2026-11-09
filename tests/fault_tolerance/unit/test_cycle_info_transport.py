# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nvidia_resiliency_ext.fault_tolerance.cycle_info import (
    CycleEndReason,
    CycleInfoPublisher,
    CycleInfoReader,
    CycleStartInfo,
    LocalCycleOutcome,
)


class _ListTransport(CycleInfoPublisher, CycleInfoReader):
    """Minimal in-memory implementation of both interfaces."""

    def __init__(self):
        self.starts = []
        self.outcomes = []

    def publish_cycle_start(self, info):
        self.starts.append(info)

    def publish_outcome(self, outcome):
        self.outcomes.append(outcome)

    def read_cycle_starts(self):
        starts, self.starts = self.starts, []
        return starts

    def read_outcomes(self):
        outcomes, self.outcomes = self.outcomes, []
        return outcomes


@pytest.mark.parametrize("interface", [CycleInfoPublisher, CycleInfoReader])
def test_interfaces_are_abstract(interface):
    with pytest.raises(TypeError):
        interface()


def test_each_kind_is_read_separately_once():
    transport = _ListTransport()
    start = CycleStartInfo(3, 10.0, ("node0", "node1"), ("node2",), (0, 1))
    outcomes = [
        LocalCycleOutcome(3, node, CycleEndReason.COMPLETED, 20.0, ()) for node in ("n0", "n1")
    ]

    for outcome in outcomes:
        transport.publish_outcome(outcome)
    transport.publish_cycle_start(start)

    assert transport.read_outcomes() == outcomes
    assert transport.read_cycle_starts() == [start]
    assert transport.read_outcomes() == []
    assert transport.read_cycle_starts() == []
    transport.close()  # the default close is a no-op
