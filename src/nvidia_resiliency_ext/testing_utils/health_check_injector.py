# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Node Failure Injection Utility

This module provides utilities to simulate node failures during specific
training cycles for specific physical nodes. Useful for testing fault tolerance
mechanisms such as spare-node replacement.

Injection fires inside ensure_node_is_healthy() (before rendezvous), keyed by
infra_rank (physical node rank derived from SLURM env). A node targeted by
infra_rank may be active or standby in a given cycle — the failure is only
triggered if that physical node is selected.

Environment Variables:
    NVRX_INJECT_GPU_FAILURE: Format "cycle:infra_rank,cycle:infra_rank,..."
        - cycle: profiling cycle number shown in logs ("Cycle: N Event: ...")
        - infra_rank: physical node rank (from SLURMD_NODENAME or SLURM_PROCID)
        - Multiple entries can target the same cycle by repeating the cycle number
        - Examples:
          * "4:0,8:1" means:
            - Cycle 4: physical node with infra_rank 0 fails
            - Cycle 8: physical node with infra_rank 1 fails
          * "1:0,1:1,3:0" means:
            - Cycle 1: infra_rank 0 and infra_rank 1 both fail
            - Cycle 3: infra_rank 0 fails

Note:
    - Injection fires before rendezvous (in ensure_node_is_healthy)
    - The targeted node may be active or standby in that cycle
    - Using infra_rank is deterministic and suitable for small-scale HW failure testing

Known limitations (cycle counter on replacement/standby nodes):
    The cycle counter is driven by FAILURE_DETECTED events in the launcher. This event
    only fires on nodes whose workers are actively monitored (i.e. active training nodes).

    - Replacement nodes (new SLURM tasks joining after a failure): their launcher
      process starts fresh with cycle=0. FAILURE_DETECTED never fires, so
      get_profiling_cycle() returns 0 regardless of the actual training round.
      A "1:33" injection targeting a replacement node at infra_rank=33 will NOT fire
      because the cycle never reaches 1 on that process.

    - Standby nodes that are promoted to active: FAILURE_DETECTED fires on the
      surviving active nodes' launchers, not on the standby's launcher. If the
      targeted infra_rank happens to be the standby node in that cycle, the cycle
      counter on its process may not match the expected injection cycle, and the
      fault will silently not fire.

    In both cases the injection is a no-op for that node. To reliably test HW failure,
    target an infra_rank that is guaranteed to be a persistent active training node
    (not a candidate standby or replacement) across the target cycles.
"""

import logging
import os
import socket
from typing import Dict, Optional, Set

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

logger = logging.getLogger(LogConfig.name)


class HealthCheckInjector:
    """Manages node failure injection for testing fault tolerance."""

    def __init__(self):
        self._failure_map: Dict[int, Set[int]] = self._parse_failure_spec()
        self._current_node = socket.gethostname()

        if self._failure_map:
            failure_summary = ", ".join(
                f"cycle {cycle} -> infra_ranks {sorted(ranks)}"
                for cycle, ranks in sorted(self._failure_map.items())
            )
            logger.warning(
                f"Node failure injection ENABLED on node {self._current_node}:\n"
                f"  Failure map: {failure_summary}"
            )

    def _parse_failure_spec(self) -> Dict[int, Set[int]]:
        """Parse NVRX_INJECT_GPU_FAILURE into {cycle: set(infra_ranks)}."""
        spec = os.environ.get("NVRX_INJECT_GPU_FAILURE", "").strip()
        if not spec:
            return {}

        failure_map: Dict[int, Set[int]] = {}

        try:
            for entry in spec.split(","):
                entry = entry.strip()
                if not entry:
                    continue

                parts = entry.split(":")
                if len(parts) != 2:
                    logger.error(
                        f"Invalid failure spec entry '{entry}'. Expected format 'cycle:infra_rank'"
                    )
                    continue

                cycle = int(parts[0].strip())
                infra_rank = int(parts[1].strip())

                if cycle not in failure_map:
                    failure_map[cycle] = set()
                failure_map[cycle].add(infra_rank)

        except ValueError as e:
            logger.error(
                f"Invalid failure specification format in NVRX_INJECT_GPU_FAILURE='{spec}': {e}"
            )
            return {}

        return failure_map

    def should_inject_failure(self, cycle: int, infra_rank: int) -> bool:
        """Return True if a failure should be injected for this node at (cycle, infra_rank)."""
        if not self._failure_map:
            return False

        if cycle not in self._failure_map:
            return False

        should_inject = infra_rank in self._failure_map[cycle]

        if should_inject:
            logger.warning(
                f"INJECTING FAILURE: "
                f"node={self._current_node}, infra_rank={infra_rank}, cycle={cycle}"
            )

        return should_inject


# Global singleton instance
_injector: Optional[HealthCheckInjector] = None


def _get_injector() -> Optional[HealthCheckInjector]:
    """Get the global injector instance, creating it on first call if enabled."""
    global _injector
    if _injector is None and os.environ.get("NVRX_INJECT_GPU_FAILURE"):
        _injector = HealthCheckInjector()
    return _injector


def should_inject_failure(cycle: int, infra_rank: int) -> bool:
    """Return True if a failure should be injected at (cycle, infra_rank)."""
    injector = _get_injector()
    if injector is None:
        return False
    return injector.should_inject_failure(cycle, infra_rank)
