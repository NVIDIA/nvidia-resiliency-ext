# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publishing of the cycle infos on a background thread, over any CycleInfoPublisher."""

import logging
import queue
import threading

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig
from nvidia_resiliency_ext.shared_utils.proto.nvrx_interface_pb2 import (
    NodeCycleEndReport,
    NVRxCycleInfo,
)

from .transport import CycleInfoPublisher

logger = logging.getLogger(LogConfig.name)


class QueuedCycleInfoPublisher(CycleInfoPublisher):
    """Publishes the cycle infos through `publisher` on a background thread, so the caller,
    the store host's rendezvous loop, only queues them and never waits for delivery.

    One thread sends the records in the order they were queued, so a cycle's start is sent
    before its end. The thread is idle, blocked on the queue, except while publishing. End
    reports are not queued: they are passed to `publisher` as they come.

    Args:
        publisher: Sends the records; it drops a record it cannot deliver.
    """

    _STOP = object()

    def __init__(self, publisher: CycleInfoPublisher):
        self._publisher = publisher
        self._queue: "queue.SimpleQueue" = queue.SimpleQueue()
        # Records queued and not yet sent (or dropped), to wait for them in flush().
        self._pending = 0
        self._sent = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="CycleInfoPublisher", daemon=True)
        self._thread.start()

    def publish_cycle_info(self, info: NVRxCycleInfo) -> bool:
        """Queue the start of a cycle; returns True once queued."""
        return self._put(self._publisher.publish_cycle_info, info)

    def update_cycle_info(self, info: NVRxCycleInfo) -> bool:
        """Queue the end of a cycle; returns True once queued."""
        return self._put(self._publisher.update_cycle_info, info)

    def publish_end_report(self, report: NodeCycleEndReport) -> bool:
        return self._publisher.publish_end_report(report)

    def flush(self, timeout: float) -> bool:
        with self._sent:
            return self._sent.wait_for(lambda: self._pending == 0, timeout=timeout)

    def close(self, timeout: float = 5.0) -> None:
        """Send the queued records, waiting up to `timeout` seconds, then stop the thread
        and close `publisher`. Records still queued then are dropped."""
        if not self.flush(timeout):
            logger.warning("Cycle infos not sent within %.1fs are dropped", timeout)
        self._queue.put(self._STOP)
        self._thread.join(timeout=1.0)
        self._publisher.close()

    def _put(self, send, record) -> bool:
        with self._sent:
            self._pending += 1
        self._queue.put((send, record))
        return True

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._STOP:
                return
            send, record = item
            try:
                send(record)  # publishers drop what they cannot deliver; guard anyway
            except Exception as e:
                logger.warning(f"Failed to publish cycle {record.cycle_number}: {e}")
            finally:
                with self._sent:
                    self._pending -= 1
                    self._sent.notify_all()
