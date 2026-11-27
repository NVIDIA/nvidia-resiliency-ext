# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from unittest.mock import MagicMock

from nvidia_resiliency_ext.fault_tolerance.cycle_info import build_cycle_info, ended_cycle_info
from nvidia_resiliency_ext.fault_tolerance.cycle_info.queued_publisher import (
    QueuedCycleInfoPublisher,
)
from nvidia_resiliency_ext.shared_utils.proto import nvrx_interface_pb2 as pb

START = build_cycle_info(
    job_id="job1", attempt_index=0, cycle_number=3, started_at_ms=10_000, active_nodes=("n0",)
)
END = ended_cycle_info(START, 20_000)


class TestQueuedCycleInfoPublisher:
    """The store host only queues its cycle infos; a background thread sends them."""

    def test_publishing_only_queues_and_the_thread_sends_in_order(self):
        sending = threading.Event()  # the inner publisher is blocked until set
        sent = []
        inner = MagicMock()
        inner.publish_cycle_info.side_effect = lambda info: sending.wait(5) and sent.append(
            ("start", info)
        )
        inner.update_cycle_info.side_effect = lambda info: sent.append(("end", info))
        publisher = QueuedCycleInfoPublisher(inner)

        # Both return at once, although the inner publisher is still blocked.
        assert publisher.publish_cycle_info(START) is True
        assert publisher.update_cycle_info(END) is True
        assert not publisher.flush(0.05)
        assert sent == []

        sending.set()
        assert publisher.flush(5)
        assert sent == [("start", START), ("end", END)]
        publisher.close()

    def test_close_sends_what_is_queued_then_stops_and_closes_the_inner_publisher(self):
        inner = MagicMock()
        publisher = QueuedCycleInfoPublisher(inner)
        publisher.update_cycle_info(END)

        publisher.close(timeout=5)

        inner.update_cycle_info.assert_called_once_with(END)
        inner.close.assert_called_once_with()
        assert not publisher._thread.is_alive()

    def test_a_failing_publish_is_logged_and_the_next_still_sent(self):
        inner = MagicMock()
        inner.publish_cycle_info.side_effect = RuntimeError("boom")
        publisher = QueuedCycleInfoPublisher(inner)

        publisher.publish_cycle_info(START)
        publisher.update_cycle_info(END)

        assert publisher.flush(5)
        inner.update_cycle_info.assert_called_once_with(END)
        publisher.close()

    def test_end_reports_go_to_the_inner_publisher_directly(self):
        inner = MagicMock()
        inner.publish_end_report.return_value = False
        publisher = QueuedCycleInfoPublisher(inner)
        report = pb.NodeCycleEndReport(cycle_number=3, node_id="n0")

        assert publisher.publish_end_report(report) is False
        inner.publish_end_report.assert_called_once_with(report)
        publisher.close()
