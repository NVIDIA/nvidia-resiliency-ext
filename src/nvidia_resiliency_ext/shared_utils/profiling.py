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

# This file adds time profiling capabilities using PyTorch's metrics system

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from torch.distributed.elastic.metrics.api import put_metric

from ..shared_utils.log_manager import LogConfig


class ProfilingEvent(Enum):
    """Enumeration of profiling events for fault tolerance metrics."""

    FAILURE_DETECTED = "failure_detected"
    WORKER_TERMINATED = "worker_terminated"
    RENDEZVOUS_STARTED = "rendezvous_started"
    RENDEZVOUS_COMPLETED = "rendezvous_completed"
    WORKER_START_STARTED = "worker_start_started"
    WORKER_START_COMPLETED = "worker_start_completed"


@dataclass
class ProfilingMeasurement:
    """Represents a single profiling measurement."""

    event: ProfilingEvent
    timestamp: float
    node_id: Optional[str] = None
    rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FaultToleranceProfiler:
    """Profiler for measuring fault tolerance timing metrics using PyTorch metrics."""

    def __init__(self):
        self._measurements: Dict[str, ProfilingMeasurement] = {}
        self._lock = threading.Lock()
        self._enabled = True
        self._metric_group = "fault_tolerance"
        # Initialize logger as a member to avoid module-level logger issues
        self._logger = logging.getLogger(LogConfig.name)

    def enable(self):
        """Enable profiling."""
        with self._lock:
            self._enabled = True

    def disable(self):
        """Disable profiling."""
        with self._lock:
            self._enabled = False

    def is_enabled(self) -> bool:
        """Check if profiling is enabled."""
        with self._lock:
            return self._enabled

    def _timestamp_to_utc_datetime(self, timestamp: float) -> str:
        """Convert timestamp to UTC datetime string."""
        utc_datetime = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return utc_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ]  # Remove last 3 digits for milliseconds

    def record_event(
        self,
        event: ProfilingEvent,
        node_id: Optional[str] = None,
        rank: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a profiling event and return a unique event ID."""
        if not self.is_enabled():
            return ""

        timestamp = time.time()
        event_id = f"{event.value}_{timestamp}_{node_id or 'unknown'}_{rank or 'unknown'}"

        measurement = ProfilingMeasurement(
            event=event, timestamp=timestamp, node_id=node_id, rank=rank, metadata=metadata or {}
        )

        with self._lock:
            self._measurements[event_id] = measurement

        # Publish metric to PyTorch metrics system
        put_metric(f"ft.{event.value}.count", 1, self._metric_group)
        put_metric(f"ft.{event.value}.timestamp", int(timestamp * 1000), self._metric_group)

        self._logger.debug(
            f"Recorded profiling event: {event.value} at {timestamp} for node {node_id}, rank {rank}"
        )
        return event_id

    def get_measurement(self, event_id: str) -> Optional[ProfilingMeasurement]:
        """Get a measurement by event ID."""
        with self._lock:
            return self._measurements.get(event_id)

    def get_all_measurements(self) -> Dict[str, ProfilingMeasurement]:
        """Get all recorded measurements."""
        with self._lock:
            return self._measurements.copy()

    def clear_measurements(self):
        """Clear all recorded measurements."""
        with self._lock:
            self._measurements.clear()

    def log_metrics_summary(self):
        """Log a summary of all recorded metrics."""
        if not self.is_enabled():
            return

        measurements = self.get_all_measurements()
        if not measurements:
            self._logger.info("No profiling measurements recorded.")
            return

        # Group measurements by event type
        events_by_type = self._group_events_by_type(list(measurements.values()))

        # Create mapping from event keys to cycle numbers
        event_to_cycle = self._create_event_to_cycle_mapping(measurements)

        # Log the summary
        self._logger.info("=== Fault Tolerance Profiling Summary ===")
        for event_type, event_measurements in events_by_type.items():
            self._logger.info(f"{event_type.value}: {len(event_measurements)} events recorded")
            for measurement in event_measurements:
                utc_time = self._timestamp_to_utc_datetime(measurement.timestamp)
                event_key = self._create_event_key(measurement)
                cycle_num = event_to_cycle.get(event_key, "Unknown")
                self._logger.info(
                    f"  - Cycle: {cycle_num} Event: {measurement.event.value} Node: {measurement.node_id} Rank: {measurement.rank} "
                    f"Time: {utc_time} UTC"
                )

        # Calculate and log timing metrics
        cycles = self._group_events_by_restart_cycles(measurements)
        self._log_timing_metrics(measurements, cycles)

    def log_cycle_summary(self, cycle_events: List[ProfilingMeasurement], cycle_num: int):
        """Log a summary for a specific restart cycle."""
        if not self.is_enabled() or not cycle_events:
            return

        # Determine cycle type
        cycle_type = self._determine_cycle_type(cycle_events)

        self._logger.info(f"=== {cycle_type} Cycle {cycle_num} Profiling Summary ===")

        # Group events by type for this cycle
        events_by_type = self._group_events_by_type(cycle_events)

        # Log events in this cycle
        for event_type, event_measurements in events_by_type.items():
            self._logger.info(f"{event_type.value}: {len(event_measurements)} events")
            for measurement in event_measurements:
                utc_time = self._timestamp_to_utc_datetime(measurement.timestamp)
                self._logger.info(
                    f"  - Cycle: {cycle_num} Event: {measurement.event.value} Node: {measurement.node_id} Rank: {measurement.rank} "
                    f"Time: {utc_time} UTC"
                )

        # Log timing metrics for this cycle
        self._log_cycle_timing_metrics(cycle_events, cycle_num, cycle_type)

    def _log_timing_metrics(
        self,
        measurements: Dict[str, ProfilingMeasurement],
        cycles: Optional[List[List[ProfilingMeasurement]]] = None,
    ):
        """Log calculated timing metrics grouped by cycles."""
        # Group events by cycles using temporal proximity (only if not provided)
        if cycles is None:
            cycles = self._group_events_by_restart_cycles(measurements)

        self._logger.info("=== Timing Metrics by Cycle ===")
        self._logger.info(f"Found {len(cycles)} cycle(s)")

        for cycle_idx, cycle_events in enumerate(cycles):
            # Determine if this is a startup or restart cycle
            cycle_type = self._determine_cycle_type(cycle_events)

            self._logger.info(f"--- {cycle_type} Cycle {cycle_idx} ---")
            self._log_cycle_timing_metrics(cycle_events, cycle_idx, cycle_type)

    def _group_events_by_restart_cycles(
        self, measurements: Dict[str, ProfilingMeasurement]
    ) -> List[List[ProfilingMeasurement]]:
        """Group events into restart cycles based on temporal proximity and event sequence."""
        # Sort all events by timestamp
        all_events = sorted(measurements.values(), key=lambda m: m.timestamp)

        cycles = []
        current_cycle = []

        for event in all_events:
            # Start a new cycle when we see a failure detection
            if event.event == ProfilingEvent.FAILURE_DETECTED:
                if current_cycle:
                    cycles.append(current_cycle)
                current_cycle = [event]
            else:
                # Add event to current cycle if it's part of a sequence
                if current_cycle:
                    current_cycle.append(event)
                elif event.event == ProfilingEvent.RENDEZVOUS_STARTED:
                    # Start a new cycle for initial startup (no failure detected)
                    current_cycle = [event]

        # Add the last cycle if it exists
        if current_cycle:
            cycles.append(current_cycle)

        return cycles

    def _group_events_by_type(self, events: List[ProfilingMeasurement]) -> Dict[ProfilingEvent, List[ProfilingMeasurement]]:
        """Group events by their type for easier lookup."""
        events_by_type = {}
        for event in events:
            if event.event not in events_by_type:
                events_by_type[event.event] = []
            events_by_type[event.event].append(event)
        return events_by_type

    def _determine_cycle_type(self, cycle_events: List[ProfilingMeasurement]) -> str:
        """Determine if this is a startup or restart cycle."""
        has_failure = any(event.event == ProfilingEvent.FAILURE_DETECTED for event in cycle_events)
        return "Restart" if has_failure else "Startup"

    def _create_event_key(self, measurement: ProfilingMeasurement) -> str:
        """Create a unique key for an event measurement."""
        return f"{measurement.event.value}_{measurement.timestamp}_{measurement.node_id}_{measurement.rank}"

    def _create_event_to_cycle_mapping(self, measurements: Dict[str, ProfilingMeasurement]) -> Dict[str, int]:
        """Create a mapping from event keys to cycle numbers."""
        cycles = self._group_events_by_restart_cycles(measurements)
        event_to_cycle = {}
        for cycle_idx, cycle_events in enumerate(cycles):
            for event in cycle_events:
                event_key = self._create_event_key(event)
                event_to_cycle[event_key] = cycle_idx
        return event_to_cycle

    def _log_cycle_timing_metrics(
        self, cycle_events: List[ProfilingMeasurement], cycle_num: int, cycle_type: str = "Cycle"
    ):
        """Log timing metrics for a single cycle."""
        # Group events by type for easier lookup
        events_by_type = self._group_events_by_type(cycle_events)

        # Calculate failure to termination time
        failure_events = events_by_type.get(ProfilingEvent.FAILURE_DETECTED, [])
        termination_events = events_by_type.get(ProfilingEvent.WORKER_TERMINATED, [])

        if failure_events and termination_events:
            # Match by node/rank within this cycle
            for failure_event in failure_events:
                matching_termination = next(
                    (
                        t
                        for t in termination_events
                        if t.node_id == failure_event.node_id and t.rank == failure_event.rank
                    ),
                    None,
                )
                if matching_termination:
                    duration = matching_termination.timestamp - failure_event.timestamp
                    self._logger.info(
                        f"  Cycle: {cycle_num} failure_to_termination: {duration:.3f}s "
                        f"(GroupRank: {failure_event.rank})"
                    )

        # Calculate rendezvous duration
        rendezvous_start = events_by_type.get(ProfilingEvent.RENDEZVOUS_STARTED, [])
        rendezvous_complete = events_by_type.get(ProfilingEvent.RENDEZVOUS_COMPLETED, [])

        if rendezvous_start and rendezvous_complete:
            # Match by node within this cycle
            for start_event in rendezvous_start:
                matching_complete = next(
                    (c for c in rendezvous_complete if c.node_id == start_event.node_id), None
                )
                if matching_complete:
                    duration = matching_complete.timestamp - start_event.timestamp
                    self._logger.info(f"  Cycle: {cycle_num} rendezvous_duration: {duration:.3f}s")

        # Calculate rendezvous to worker start time
        if rendezvous_complete:
            start_start = events_by_type.get(ProfilingEvent.WORKER_START_STARTED, [])
            for complete_event in rendezvous_complete:
                matching_start = next(
                    (s for s in start_start if s.node_id == complete_event.node_id), None
                )
                if matching_start:
                    duration = matching_start.timestamp - complete_event.timestamp
                    self._logger.info(
                        f"  Cycle: {cycle_num} rendezvous_complete_to_worker_start: {duration:.3f}s"
                    )

        # Calculate worker start time
        start_start = events_by_type.get(ProfilingEvent.WORKER_START_STARTED, [])
        start_complete = events_by_type.get(ProfilingEvent.WORKER_START_COMPLETED, [])

        if start_start and start_complete:
            for start_event in start_start:
                matching_complete = next(
                    (
                        c
                        for c in start_complete
                        if c.node_id == start_event.node_id and c.rank == start_event.rank
                    ),
                    None,
                )
                if matching_complete:
                    duration = matching_complete.timestamp - start_event.timestamp
                    self._logger.info(
                        f"  Cycle: {cycle_num} worker_start_time: {duration:.3f}s "
                        f"(GroupRank: {start_event.rank})"
                    )

        # Calculate total cycle time
        start_complete = events_by_type.get(ProfilingEvent.WORKER_START_COMPLETED, [])

        if failure_events:
            # Restart cycle: from failure detection to worker start completion
            first_failure = min(failure_events, key=lambda e: e.timestamp)
            if start_complete:
                last_start = max(start_complete, key=lambda e: e.timestamp)
                total_duration = last_start.timestamp - first_failure.timestamp
                self._logger.info(f"  Cycle: {cycle_num} total_restart_time: {total_duration:.3f}s")
            else:
                # Fallback to worker start if completed events not available
                start_start = events_by_type.get(ProfilingEvent.WORKER_START_STARTED, [])
                if start_start:
                    last_start = max(start_start, key=lambda e: e.timestamp)
                    total_duration = last_start.timestamp - first_failure.timestamp
                    self._logger.info(
                        f"  Cycle: {cycle_num} total_restart_time (to worker start): {total_duration:.3f}s"
                    )
        elif start_complete:
            # Startup cycle: from first event to worker start completion
            first_event = min(cycle_events, key=lambda e: e.timestamp)
            last_start = max(start_complete, key=lambda e: e.timestamp)
            total_duration = last_start.timestamp - first_event.timestamp
            self._logger.info(f"  Cycle: {cycle_num} total_startup_time: {total_duration:.3f}s")


# Global profiler instance
_global_profiler = FaultToleranceProfiler()


def get_profiler() -> FaultToleranceProfiler:
    """Get the global profiler instance."""
    return _global_profiler


def record_profiling_event(
    event: ProfilingEvent,
    node_id: Optional[str] = None,
    rank: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Convenience function to record a profiling event."""
    return _global_profiler.record_event(event, node_id, rank, metadata)


def log_profiling_summary():
    """Convenience function to log profiling summary."""
    _global_profiler.log_metrics_summary()

def log_current_cycle_summary():
    """Log profiling summary for the current restart cycle."""
    profiler = get_profiler()
    
    # Get all measurements
    all_measurements = profiler.get_all_measurements()
    if not all_measurements:
        return
        
    # Group events by cycles
    cycles = profiler._group_events_by_restart_cycles(all_measurements)
    if not cycles:
        return
        
    # Get the most recent cycle (last one)
    current_cycle = cycles[-1]
    cycle_num = len(cycles) - 1
    
    # Log the cycle summary
    profiler.log_cycle_summary(current_cycle, cycle_num)
