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
GPU Health Check Failure Injection Utility

This module provides utilities to simulate GPU health check failures
during specific restart cycles for specific nodes. Useful for testing fault
tolerance mechanisms.

This module monkey-patches GPUHealthCheck when NVRX_INJECT_GPU_FAILURE is set.

Environment Variables:
    NVRX_INJECT_GPU_FAILURE: Format "cycle:infra_rank,cycle:infra_rank,..."
        - Specify which infrastructure group rank(s) should fail at which cycle
        - Multiple ranks can fail in the same cycle by repeating the cycle number
        - Examples: 
          * "1:1,3:17,5:33" means:
            - Cycle 1: infrastructure rank 1 fails
            - Cycle 3: infrastructure rank 17 fails  
            - Cycle 5: infrastructure rank 33 fails
          * "1:1,1:2,1:5,3:17,3:18" means:
            - Cycle 1: infrastructure ranks 1, 2, and 5 fail
            - Cycle 3: infrastructure ranks 17 and 18 fail
    
    SLURM_PROCID: SLURM process ID within the job (required)
        - Used to determine infrastructure rank
        - Automatically set by SLURM (0, 1, 2, ... for each node)

Note:
    - Infrastructure rank comes from SLURM_PROCID environment variable
    - Cycle number is passed from the rendezvous round counter
    - The rendezvous runs once per node, not per rank
    - This module automatically activates when imported if NVRX_INJECT_GPU_FAILURE is set
"""

import logging
import os
import socket
from typing import Dict, Optional, Set

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

logger = logging.getLogger(LogConfig.name)

# Global variable to store the current cycle for injection checking
_current_cycle: Optional[int] = None


def set_current_cycle(cycle: int) -> None:
    """
    Set the current rendezvous cycle for injection checking.

    This should be called by the rendezvous handler before health checks.

    Args:
        cycle: The current rendezvous round/cycle number.
    """
    global _current_cycle
    _current_cycle = cycle


class HealthCheckInjector:
    """Manages GPU health check failure injection for testing purposes."""

    def __init__(self):
        """Initialize the health check injector."""
        # Parse the failure specification: "cycle:infra_rank,cycle:infra_rank,..."
        self._failure_map: Dict[int, Set[int]] = self._parse_failure_spec()
        self._current_node = socket.gethostname()

        if self._failure_map:
            failure_summary = ", ".join(
                f"cycle {cycle} -> ranks {sorted(ranks)}"
                for cycle, ranks in sorted(self._failure_map.items())
            )
            logger.warning(
                f"GPU health check failure injection ENABLED on node {self._current_node}:\n"
                f"  Failure map: {failure_summary}"
            )

    def _parse_failure_spec(self) -> Dict[int, Set[int]]:
        """
        Parse the failure specification from environment variable.

        Format: "cycle:infra_rank,cycle:infra_rank,..."
        Multiple ranks can fail in the same cycle by repeating the cycle number.

        Examples:
            "1:1,3:17,5:33" - Single rank per cycle
            "1:1,1:2,1:5,3:17,3:18" - Multiple ranks in cycles 1 and 3

        Returns:
            Dict mapping cycle number to set of infrastructure ranks that should fail
        """
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

    def _get_infra_rank(self) -> Optional[int]:
        """
        Get the infrastructure rank of this node from SLURM_PROCID.

        Infrastructure rank represents the node's position in the cluster.
        SLURM_PROCID is the process ID within the SLURM job allocation.

        Returns:
            Infrastructure rank (0, 1, 2, ...) or None if SLURM_PROCID is not available.
        """
        infra_rank_str = os.environ.get("SLURM_PROCID")
        if infra_rank_str is None:
            logger.warning("SLURM_PROCID not found. Cannot determine infrastructure rank.")
            return None

        try:
            return int(infra_rank_str)
        except ValueError:
            logger.error(f"Invalid SLURM_PROCID value: {infra_rank_str}")
            return None

    def should_inject_gpu_failure(self, cycle: int) -> bool:
        """
        Check if GPU failure should be injected for this node in the given cycle.

        Args:
            cycle: The current rendezvous cycle/round number.

        Returns:
            bool: True if GPU failure should be injected, False otherwise.
        """
        if not self._failure_map:
            return False

        # Check if this cycle has any failures configured
        if cycle not in self._failure_map:
            return False

        target_ranks = self._failure_map[cycle]
        infra_rank = self._get_infra_rank()

        # If we can't determine infra rank, don't inject
        if infra_rank is None:
            logger.warning(
                f"Cannot inject failure: unable to determine infrastructure rank for node {self._current_node}"
            )
            return False

        should_inject = infra_rank in target_ranks

        if should_inject:
            logger.warning(
                f"INJECTING GPU HEALTH CHECK FAILURE: "
                f"node={self._current_node}, infra_rank={infra_rank}, cycle={cycle}"
            )

        return should_inject


# Global singleton instance
_injector: Optional[HealthCheckInjector] = None


def _get_injector() -> Optional[HealthCheckInjector]:
    """Get the global health check injector instance, or None if not enabled."""
    global _injector
    if _injector is None and os.environ.get("NVRX_INJECT_GPU_FAILURE"):
        _injector = HealthCheckInjector()
    return _injector


def _monkey_patch_gpu_health_check():
    """
    Monkey-patch GPUHealthCheck to inject failures when configured.

    This wraps the GPUHealthCheck.__call__ method to check for injection before
    performing the actual health check.
    """
    from nvidia_resiliency_ext.shared_utils.health_check import GPUHealthCheck

    # Store the original __call__ method
    _original_call = GPUHealthCheck.__call__

    def _wrapped_call(self):
        """Wrapped __call__ that checks for injection first."""
        injector = _get_injector()
        if injector and _current_cycle is not None:
            if injector.should_inject_gpu_failure(_current_cycle):
                # Inject failure - return False to indicate unhealthy
                return False

        # Otherwise, call the original health check
        return _original_call(self)

    # Apply the monkey patch
    GPUHealthCheck.__call__ = _wrapped_call
    logger.debug("GPU health check has been monkey-patched for failure injection")


# Automatically apply monkey patch if injection is enabled
if os.environ.get("NVRX_INJECT_GPU_FAILURE"):
    _monkey_patch_gpu_health_check()
