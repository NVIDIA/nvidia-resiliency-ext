# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file adds time profiling capabilities for fault tolerance (cycle and event logging).

import logging
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from ..shared_utils.log_manager import LogConfig


class ProfilingEvent(Enum):
    """Enumeration of profiling events for fault tolerance metrics."""

    FAILURE_DETECTED = "failure_detected"
    WORKER_TERMINATED = "worker_terminated"
    RENDEZVOUS_STARTED = "rendezvous_started"
    RENDEZVOUS_COMPLETED = "rendezvous_completed"
    WORKER_START_STARTED = "worker_start_started"
    WORKER_START_COMPLETED = "worker_start_completed"


class FaultToleranceProfiler:
    """Profiler for measuring fault tolerance timing metrics (cycle and event logging)."""

    def __init__(self):
        self._current_cycle = 0
        self._logger = logging.getLogger(LogConfig.name)

    def _timestamp_to_utc_datetime(self, timestamp: float) -> str:
        """Convert timestamp to UTC datetime string."""
        utc_datetime = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return utc_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ]  # Remove last 3 digits for milliseconds

    def set_cycle(self, cycle: int) -> None:
        """Set the current cycle number.

        Called by the rendezvous handler when a newly joining node syncs its cycle number
        from the global_cycle_key in the store. This ensures newly joining nodes (e.g.,
        replacement array tasks) continue with the correct cycle number instead of starting from 0.

        Args:
            cycle: The cycle number to set. Only sets if >= current cycle to prevent backward jumps.
        """
        if cycle >= self._current_cycle:
            self._current_cycle = cycle
        else:
            self._logger.warning(
                f"Attempted to set profiler cycle to {cycle}, which is less than "
                f"current cycle {self._current_cycle}. Ignoring to prevent backward cycle jumps."
            )

    def record_event(
        self,
        event: ProfilingEvent,
        node_id: Optional[Any] = None,
        rank: Optional[int] = None,
    ) -> str:
        """Record a profiling event and return a unique event ID."""
        timestamp = time.time()
        # Convert node_id to string for event ID and logging
        node_id_str = str(node_id) if node_id is not None else 'unknown'
        event_id = f"{event.value}_{timestamp}_{node_id_str}_{rank or 'unknown'}"

        # Increment cycle count for failure detection events
        if event == ProfilingEvent.FAILURE_DETECTED:
            self._current_cycle += 1

        # Format log message with cycle count and UTC time
        utc_time = self._timestamp_to_utc_datetime(timestamp)
        self._logger.info(
            f"  - Cycle: {self._current_cycle} Event: {event.value} Node: {node_id_str} Rank: {rank} "
            f"Time: {utc_time} UTC"
        )
        return event_id


# Global profiler instance (lazy-initialized to avoid stdout output at import time)
_global_profiler: Optional[FaultToleranceProfiler] = None
_global_profiler_lock = threading.Lock()


def _get_global_profiler() -> FaultToleranceProfiler:
    """Get or create the global profiler instance (thread-safe)."""
    global _global_profiler
    if _global_profiler is None:
        with _global_profiler_lock:
            # Double-check pattern to avoid race conditions
            if _global_profiler is None:
                _global_profiler = FaultToleranceProfiler()
    return _global_profiler


def record_profiling_event(
    event: ProfilingEvent,
    node_id: Optional[Any] = None,
    rank: Optional[int] = None,
) -> str:
    """Convenience function to record a profiling event.

    Args:
        event: The profiling event to record
        node_id: Node identifier (can be any type, will be converted to string)
        rank: Rank identifier

    Returns:
        Event ID string
    """
    return _get_global_profiler().record_event(event, node_id, rank)


def set_profiling_cycle(cycle: int) -> None:
    """Set the current cycle number in the global profiler.

    Called by the rendezvous handler when a newly joining node syncs its cycle number
    from the global_cycle_key in the store. This ensures newly joining nodes (e.g.,
    replacement array tasks) continue with the correct cycle number instead of starting from 0.

    Args:
        cycle: The cycle number to set. Only sets if >= current cycle to prevent backward jumps.
    """
    _get_global_profiler().set_cycle(cycle)
