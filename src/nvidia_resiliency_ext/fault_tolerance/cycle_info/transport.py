# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport of the cycle records to the CycleAnalyzer.

The store host publishes a `CycleStartInfo` per cycle and each launcher publishes its
`LocalCycleOutcome` per cycle, through a `CycleInfoPublisher`; the CycleAnalyzer, the only
writer of the cycle info files, receives them through a `CycleInfoReader`. There is no
cycle end record: the end is concluded from the outcomes. Implementations own their wire
format and where they send to or read from; the interfaces exchange only the records.

Delivery is best effort: a record can be lost, e.g. when its node fails, and publishing
must never delay the rendezvous or a launcher's restart. Records of one cycle can arrive
in any order, e.g. outcomes before the start info, so the reader keys them by cycle.
"""

from abc import ABC, abstractmethod
from typing import List

from .records import CycleStartInfo, LocalCycleOutcome


class CycleInfoPublisher(ABC):
    """Sends cycle records towards the CycleAnalyzer.

    The publish methods may wait for delivery, but only for a bounded time, and must not
    raise on transport failures: a record that cannot be delivered is dropped.
    """

    @abstractmethod
    def publish_cycle_start(self, info: CycleStartInfo) -> bool:
        """Publish how the rendezvous started a cycle; called by the store host only.

        Returns:
            True if the record was delivered.
        """

    @abstractmethod
    def publish_outcome(self, outcome: LocalCycleOutcome) -> bool:
        """Publish how a cycle ended on this launcher.

        Returns:
            True if the record was delivered.
        """

    def close(self) -> None:
        """Stop publishing; records not yet delivered may be dropped."""


class CycleInfoReader(ABC):
    """Receives the cycle records on the CycleAnalyzer side.

    The read methods do not wait: each returns the records of its kind received since its
    previous call, in arrival order, or an empty list if there are none.
    """

    @abstractmethod
    def read_cycle_starts(self) -> List[CycleStartInfo]:
        """Return the cycle start infos received since the previous call."""

    @abstractmethod
    def read_outcomes(self) -> List[LocalCycleOutcome]:
        """Return the launchers' outcomes received since the previous call."""

    def close(self) -> None:
        """Stop receiving; records not yet read may be dropped."""
