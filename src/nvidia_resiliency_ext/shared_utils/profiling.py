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

# This file adds time profiling capabilities using nv one logger

import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.core.attributes import Attributes
from nv_one_logger.core.event import Event

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
    """Profiler for measuring fault tolerance timing metrics using nv one logger."""

    def __init__(self):
        self._current_cycle = 0
        # Initialize logger as a member to avoid module-level logger issues
        self._logger = logging.getLogger(LogConfig.name)

    def _timestamp_to_utc_datetime(self, timestamp: float) -> str:
        """Convert timestamp to UTC datetime string."""
        utc_datetime = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return utc_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ]  # Remove last 3 digits for milliseconds

    def _publish_metrics(
        self, event: ProfilingEvent, timestamp: float, node_id: Optional[str], rank: Optional[int]
    ) -> None:
        """Publish metrics using nv one logger."""
        try:
            # Check if nv one logger is available and enabled
            if OneLoggerProvider.instance().one_logger_enabled:
                # Create attributes for the event
                attributes = Attributes()
                attributes.add("event_type", event.value)
                attributes.add("timestamp_ms", int(timestamp * 1000))
                attributes.add("cycle", self._current_cycle)
                if node_id:
                    attributes.add("node_id", node_id)
                if rank is not None:
                    attributes.add("rank", rank)

                # Create and record the event
                event_obj = Event.create(f"ft.{event.value}", attributes)
                OneLoggerProvider.instance().recorder.event(None, event_obj)
        except Exception as e:
            # If nv one logger fails, just log a warning and continue
            self._logger.warning(f"Failed to publish metrics to nv one logger: {e}")

    def record_event(
        self,
        event: ProfilingEvent,
        node_id: Optional[str] = None,
        rank: Optional[int] = None,
    ) -> str:
        """Record a profiling event and return a unique event ID."""
        timestamp = time.time()
        event_id = f"{event.value}_{timestamp}_{node_id or 'unknown'}_{rank or 'unknown'}"

        # Increment cycle count for failure detection events
        if event == ProfilingEvent.FAILURE_DETECTED:
            self._current_cycle += 1

        # Publish metrics using nv one logger
        self._publish_metrics(event, timestamp, node_id, rank)

        # Format log message with cycle count and UTC time
        utc_time = self._timestamp_to_utc_datetime(timestamp)
        self._logger.info(
            f"  - Cycle: {self._current_cycle} Event: {event.value} Node: {node_id} Rank: {rank} "
            f"Time: {utc_time} UTC"
        )
        return event_id


# Global profiler instance
_global_profiler = FaultToleranceProfiler()


def record_profiling_event(
    event: ProfilingEvent,
    node_id: Optional[str] = None,
    rank: Optional[int] = None,
) -> str:
    """Convenience function to record a profiling event."""
    return _global_profiler.record_event(event, node_id, rank)
