# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from nvidia_resiliency_ext.fault_tolerance.cycle_info import (
    CycleInfo,
    CycleInfoPublisher,
    CycleInfoReader,
    EndReason,
    NodeCycleEndReport,
)


class _ListTransport(CycleInfoPublisher, CycleInfoReader):
    """Minimal in-memory implementation of both interfaces."""

    def __init__(self):
        self.infos = []
        self.updates = []
        self.reports = []

    def publish_cycle_info(self, info):
        self.infos.append(info)
        return True

    def update_cycle_info(self, info):
        self.updates.append(info)
        return True

    def publish_end_report(self, report):
        self.reports.append(report)
        return True

    def read_cycle_infos(self):
        infos, self.infos = self.infos, []
        return infos

    def read_cycle_info_updates(self):
        updates, self.updates = self.updates, []
        return updates

    def read_end_reports(self):
        reports, self.reports = self.reports, []
        return reports


@pytest.mark.parametrize("interface", [CycleInfoPublisher, CycleInfoReader])
def test_interfaces_are_abstract(interface):
    with pytest.raises(TypeError):
        interface()


def test_each_kind_is_read_separately_once():
    transport = _ListTransport()
    start = CycleInfo("job1", 0, 3, 10.0, active_nodes=("node0", "node1"), active_ranks=(0, 1))
    end = replace(start, ended_at=20.0, timestamp=20_000)
    reports = [
        NodeCycleEndReport("job1", 0, 3, node, EndReason.COMPLETED, 20_000) for node in ("n0", "n1")
    ]

    for report in reports:
        assert transport.publish_end_report(report)
    assert transport.update_cycle_info(end)
    assert transport.publish_cycle_info(start)

    assert transport.read_end_reports() == reports
    assert transport.read_cycle_infos() == [start]
    assert transport.read_cycle_info_updates() == [end]
    assert transport.read_end_reports() == []
    assert transport.read_cycle_infos() == []
    assert transport.read_cycle_info_updates() == []
    transport.close()  # the default close is a no-op
