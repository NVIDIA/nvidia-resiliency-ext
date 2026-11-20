# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport of the cycle records to the CycleAnalyzer.

The store host publishes an `NVRxCycleInfo` per cycle when it starts, and updates it when the
cycle ends; each launcher publishes its `NodeCycleEndReport` per cycle. They go through a
`CycleInfoPublisher`; the CycleAnalyzer, the only writer of the cycle info files,
receives them through a `CycleInfoReader`. The records are the messages of
nvrx_interface.proto (see messages.py); implementations own how they carry them (e.g.
gRPC, or the serialized bytes in a key-value store) and where they send to or read from.

Delivery is best effort: a record can be lost, e.g. when its node fails, and publishing
must never delay the rendezvous or a launcher's restart. Records of one cycle can arrive
in any order, e.g. end reports before the cycle info, so the reader keys them by cycle.
"""

from abc import ABC, abstractmethod
from typing import List

from nvidia_resiliency_ext.shared_utils.proto.nvrx_interface_pb2 import (
    NodeCycleEndReport,
    NVRxCycleInfo,
)


class CycleInfoPublisher(ABC):
    """Sends cycle records towards the CycleAnalyzer.

    The publish methods may wait for delivery, but only for a bounded time, and must not
    raise on transport failures: a record that cannot be delivered is dropped. They do not
    modify the record; the caller must not modify it after publishing either. A publisher
    may only queue the records, sending them in the background: its publish methods then
    return True once queued, and flush() waits for them to be sent.
    """

    @abstractmethod
    def publish_cycle_info(self, info: NVRxCycleInfo) -> bool:
        """Publish a cycle that started; called by the store host only.

        Returns:
            True if the record was delivered.
        """

    @abstractmethod
    def update_cycle_info(self, info: NVRxCycleInfo) -> bool:
        """Publish a cycle that ended (`info.cycle_end_time` set); called by the store host
        only.

        Returns:
            True if the record was delivered.
        """

    @abstractmethod
    def publish_end_report(self, report: NodeCycleEndReport) -> bool:
        """Publish how a cycle ended on this launcher.

        Returns:
            True if the record was delivered.
        """

    def flush(self, timeout: float) -> bool:
        """Wait up to `timeout` seconds until the records queued so far were sent. Returns
        True if none is left pending; at once for a publisher that does not queue."""
        return True

    def close(self) -> None:
        """Stop publishing; records not yet delivered may be dropped."""


class CycleInfoReader(ABC):
    """Receives the cycle records on the CycleAnalyzer side.

    The read methods do not wait: each returns the records of its kind received since its
    previous call, in arrival order, or an empty list if there are none.
    """

    @abstractmethod
    def read_cycle_infos(self) -> List[NVRxCycleInfo]:
        """Return the cycle infos received since the previous call: the published ones
        (cycle starts) and the updated ones (cycle ends, with cycle_end_time), together in
        arrival order, so a cycle's start always comes before its end."""

    @abstractmethod
    def read_end_reports(self) -> List[NodeCycleEndReport]:
        """Return the launchers' end reports received since the previous call."""

    def close(self) -> None:
        """Stop receiving; records not yet read may be dropped."""
