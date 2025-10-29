#!/usr/bin/env python3

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Cycle management for fault tolerance launcher.

Maintains a cache of cycle objects that can be updated by external modules
(e.g., failure attribution) and queried by the launcher to make exit decisions.
"""

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

logger = logging.getLogger(LogConfig.name)


class Cycle:
    """
    Thread-safe representation of a fault tolerance cycle.

    A cycle represents one iteration of the fault tolerance loop. External modules
    can update cycle properties (e.g., failure_reason) and the launcher can query
    cycles to make decisions.
    """

    # Policy: Which failure reasons should trigger early exit
    NON_RECOVERABLE_FAILURES = {
        'GPU_HW_FAILURE',  # GPU hardware failure (XID errors, fallen off bus)
        'GPU_FATAL_ERROR',  # Fatal GPU errors
        'MEMORY_HW_FAILURE',  # Memory hardware errors (uncorrectable ECC)
        'NETWORK_HW_FAILURE',  # Network hardware failure (IB errors)
        'DISK_FAILURE',  # Storage failures
        'POWER_FAILURE',  # Power supply issues
    }

    def __init__(self, cycle_number: int):
        """
        Initialize a cycle object.

        Args:
            cycle_number: The cycle number
        """
        self._lock = threading.Lock()
        self.cycle_number = cycle_number
        self.failure_reason: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self.events: list = []  # List of profiling events for this cycle
        self.created_at = time.time()
        self.updated_at = time.time()

    def add_event(
        self,
        event: str,
        timestamp: float,
        node_id: Optional[Any] = None,
        rank: Optional[int] = None,
    ) -> None:
        """
        Thread-safe addition of a profiling event.

        Args:
            event: Event type (e.g., 'WORKER_START_COMPLETED')
            timestamp: Event timestamp
            node_id: Node identifier
            rank: Rank identifier
        """
        with self._lock:
            event_data = {
                'event': event,
                'timestamp': timestamp,
                'node_id': str(node_id) if node_id is not None else None,
                'rank': rank,
            }
            self.events.append(event_data)
            self.updated_at = time.time()
            logger.debug(f"Cycle {self.cycle_number}: added event '{event}'")

    def update(
        self, failure_reason: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Thread-safe update of cycle properties.

        Args:
            failure_reason: Attributed failure reason (e.g., 'GPU_HW_FAILURE')
            metadata: Additional metadata to store with this cycle
        """
        with self._lock:
            if failure_reason is not None:
                self.failure_reason = failure_reason
                logger.info(
                    f"Cycle {self.cycle_number}: failure_reason updated to '{failure_reason}'"
                )

            if metadata is not None:
                self.metadata.update(metadata)
                logger.debug(f"Cycle {self.cycle_number}: metadata updated")

            self.updated_at = time.time()

    def is_non_recoverable(self) -> bool:
        """
        Check if this cycle has a non-recoverable failure.

        Returns:
            True if failure_reason is in NON_RECOVERABLE_FAILURES, False otherwise
        """
        with self._lock:
            return self.failure_reason in self.NON_RECOVERABLE_FAILURES

    def to_dict(self) -> Dict[str, Any]:
        """
        Get a dictionary representation of this cycle.

        Returns:
            Dictionary with cycle information including all events
        """
        with self._lock:
            return {
                'cycle_number': self.cycle_number,
                'failure_reason': self.failure_reason,
                'is_non_recoverable': self.is_non_recoverable(),
                'metadata': self.metadata.copy(),
                'events': self.events.copy(),  # List of all profiling events
                'created_at': self.created_at,
                'updated_at': self.updated_at,
            }


class CycleManager:
    """
    Thread-safe manager for maintaining an LRU cache of recent cycles.

    This is a singleton accessed through the launcher agent. External modules
    can access the cycle manager through the agent instance to update cycle
    information (e.g., failure attribution).

    The manager maintains an LRU cache using OrderedDict where:
    - New cycles are added to the end
    - When max size is reached, oldest cycles are removed
    - Accessing a cycle does NOT move it (no LRU promotion, just FIFO eviction)
    """

    def __init__(self, max_cache_size: int = 10):
        """
        Initialize the cycle manager.

        Args:
            max_cache_size: Maximum number of cycles to keep in LRU cache
        """
        self._lock = threading.Lock()
        # OrderedDict maintains insertion order, making it perfect for FIFO/LRU cache
        self._cycles: OrderedDict[int, Cycle] = OrderedDict()
        self._max_cache_size = max_cache_size

    def get_or_create_cycle(self, cycle_number: int) -> Cycle:
        """
        Get an existing cycle or create a new one.

        Uses LRU eviction: when cache is full, removes oldest cycle.

        Args:
            cycle_number: The cycle number

        Returns:
            Cycle object
        """
        with self._lock:
            if cycle_number in self._cycles:
                # Cycle exists, return it (no LRU promotion)
                return self._cycles[cycle_number]

            # Create new cycle
            cycle = Cycle(cycle_number)
            self._cycles[cycle_number] = cycle

            # Evict oldest cycle if cache is full (LRU eviction)
            if len(self._cycles) > self._max_cache_size:
                oldest_cycle_num, _ = self._cycles.popitem(last=False)
                logger.debug(f"Evicted cycle {oldest_cycle_num} from cache (LRU)")

            logger.debug(f"Created cycle {cycle_number} in cache")
            return cycle

    def get_cycles(self, cycle_number: Optional[int] = None) -> Optional[Dict[int, Dict[str, Any]]]:
        """
        Get cycle(s) from cache.

        Args:
            cycle_number: Optional cycle number. If None, returns all cycles.
                         If provided, returns just that cycle.
                         Supports negative indexing: -1 for last cycle, -2 for second-to-last, etc.

        Returns:
            If cycle_number is None: Dict mapping cycle_number to cycle info dict
            If cycle_number is provided: Dict with single entry {cycle_number: cycle_info}
                                        or None if cycle not found in cache
        """
        with self._lock:
            if cycle_number is None:
                # Return all cycles
                return {num: cycle.to_dict() for num, cycle in self._cycles.items()}
            else:
                # Handle negative indexing: convert to actual cycle number
                if cycle_number < 0:
                    cycle_nums = list(self._cycles.keys())
                    if not cycle_nums:
                        return None
                    try:
                        cycle_number = cycle_nums[cycle_number]
                    except IndexError:
                        return None

                # Return specific cycle by number
                cycle = self._cycles.get(cycle_number)
                if cycle:
                    return {cycle_number: cycle.to_dict()}
                return None

    def check_recent_cycles_for_exit(
        self, lookback: int = 2
    ) -> Optional[tuple[int, str]]:
        """
        Check recent cycles for non-recoverable failures.

        Looks at the current cycle (highest in cache) and up to 'lookback' previous cycles.

        Args:
            lookback: Number of previous cycles to check (default: 2)

        Returns:
            Tuple of (cycle_number, failure_reason) if non-recoverable failure found, None otherwise
        """
        with self._lock:
            if not self._cycles:
                return None
            
            # Get current cycle (highest cycle number in cache)
            current_cycle = max(self._cycles.keys())
            
            # Check current cycle and recent cycles
            for cycle_num in range(max(0, current_cycle - lookback), current_cycle + 1):
                cycle = self._cycles.get(cycle_num)
                if cycle and cycle.is_non_recoverable():
                    logger.warning(
                        f"Non-recoverable failure found in cycle {cycle_num}: {cycle.failure_reason}"
                    )
                    return (cycle_num, cycle.failure_reason)

            return None
