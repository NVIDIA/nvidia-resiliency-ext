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

"""
NUMA binding utilities for optimizing CPU and memory affinity.
"""

import os
import subprocess
import sys

import numa


def _parse_cpu_list(cpu_str):
    """
    Parse CPU list string format into a set of CPU IDs.

    Examples:
        "0-31" -> {0, 1, 2, ..., 31}
        "0,2,4-7" -> {0, 2, 4, 5, 6, 7}

    Args:
        cpu_str: CPU list string in taskset format

    Returns:
        Set of CPU IDs
    """
    cpus = set()
    for part in cpu_str.split(','):
        if '-' in part:
            start, end = part.split('-')
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(part))
    return cpus


def numa_bind():
    """
    Bind the current process to a NUMA node based on SLURM_LOCALID.

    This function is equivalent to running:
        numactl --cpunodebind=$((SLURM_LOCALID/GPUS_PER_NUMA)) --membind=$((SLURM_LOCALID/GPUS_PER_NUMA))

    Environment Variables:
        SLURM_LOCALID (int): Local rank of this process (set by Slurm). Default: 0
        GPUS_PER_NUMA (int): Number of GPUs mapped to one NUMA node.
                             If not set, NUMA binding is skipped (no-op).
                             Example: 4 for H100, 2 for GB200

    The function performs the following operations:
        1. Determines the NUMA node ID based on SLURM_LOCALID and GPUS_PER_NUMA
        2. Binds CPU affinity to the CPUs of the target NUMA node
        3. Binds memory allocation to the target NUMA node
        4. Prints verification information

    Raises:
        RuntimeError: If NUMA binding operations fail

    Example:
        >>> from nvidia_resiliency_ext.shared_utils.numa_bind import numa_bind
        >>> numa_bind()
        [NUMA] PID=12345 SLURM_LOCALID=0 → node=0
        [NUMA] CPU affinity: pid 12345's current affinity list: 0-31
        [NUMA] Memory node bound: 0
    """
    # Skip NUMA binding if GPUS_PER_NUMA is not set
    gpus_per_numa_str = os.getenv("GPUS_PER_NUMA")
    if gpus_per_numa_str is None:
        return

    pid = os.getpid()
    local_id = int(os.getenv("SLURM_LOCALID", 0))
    gpus_per_numa = int(gpus_per_numa_str)
    node_id = local_id // gpus_per_numa

    sys.stderr.write(f"[NUMA] PID={pid} SLURM_LOCALID={local_id} → node={node_id}\n")

    try:
        # --- CPU binding ---
        cpus = numa.node_to_cpus(node_id)
        os.sched_setaffinity(0, cpus)

        # --- Memory binding ---
        numa.run_on_node(node_id)
        numa.set_membind(str(node_id))
        numa.set_preferred(node_id)

        # --- Verification ---
        try:
            result = subprocess.check_output(
                ["taskset", "-cp", str(pid)], text=True, stderr=subprocess.DEVNULL
            ).strip()
            # Parse the output: "pid 12345's current affinity list: 0-31"
            if ":" in result:
                actual_affinity_str = result.split(":")[-1].strip()
                actual_cpus = _parse_cpu_list(actual_affinity_str)
                expected_cpus = set(cpus)

                if actual_cpus != expected_cpus:
                    sys.stderr.write(
                        f"[NUMA] Warning: CPU affinity mismatch for node {node_id}. "
                        f"Expected {len(expected_cpus)} CPUs, got {len(actual_cpus)} CPUs\n"
                    )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # taskset might not be available
            sys.stderr.write(f"[NUMA] Warning: Could not verify CPU affinity for node {node_id}\n")
    except Exception as e:
        raise RuntimeError(f"Failed to bind to NUMA node {node_id}: {e}") from e
