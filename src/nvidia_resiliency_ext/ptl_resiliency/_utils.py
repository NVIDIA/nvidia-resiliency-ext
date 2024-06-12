# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import os
import random
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import torch


def is_module_available(module: str) -> bool:
    import importlib

    return importlib.util.find_spec(module) is not None


@dataclass
class SimulatedFaultParams:
    """
    Description of a simulated rank fault, used for FT testing and debugging.

    Simulated fault types are:
    - 'rank_killed' a rank is killed with SIGKILL
    - 'rank_hung' a rank is stopped with SIGSTOP
    - 'random' randomly selects one of the above faults.

    Fault delay is computed as:
    - `base_delay` + RAND_FLOAT_FROM_0.0_to_1.0 * `rand_delay`

    Attributes:
        fault_type (str): The type of fault, one of: ['random', 'rank_killed', 'rank_hung'].
        base_delay (float): The base (minimum) delay [seconds] for the fault.
        rand_delay (float, optional): The max additional random delay for the fault. Defaults to 0.0.
        rank_to_fail (int, optional): The rank to fail. Defaults to None - random rank will be picked.
    """

    fault_type: str
    base_delay: float
    rand_delay: float = 0.0
    rank_to_fail: Optional[int] = None


def parse_simulated_fault_params(simulated_fault_params) -> Optional[SimulatedFaultParams]:

    if simulated_fault_params is None:
        return None
    if isinstance(simulated_fault_params, SimulatedFaultParams):
        return simulated_fault_params
    try:
        return SimulatedFaultParams(**simulated_fault_params)
    except Exception as e:
        raise ValueError(
            f"Failed to parse simulated fault params, "
            "it should be SimulatedFaultParams instance or "
            "an object that can be unpacked with '**' and passed to the "
            f"SimulatedFaultParams.__init__ Got: {simulated_fault_params}"
        ) from e


def setup_simulated_fault(fault_desc: SimulatedFaultParams):

    rng = random.Random()

    rank = torch.distributed.get_rank()

    if rank == 0:
        print(f"Initializing simulated fault: {fault_desc}")

    # rank that simulates a fault can be explicitly specified in the `rank_to_fail` field
    # if not specified, it just picks a random rank

    rand_rank = rng.randint(0, torch.distributed.get_world_size() - 1)
    rank_to_fail = fault_desc.rank_to_fail if fault_desc.rank_to_fail is not None else rand_rank
    rank_to_fail = torch.tensor([rank_to_fail], device=torch.cuda.current_device())
    torch.distributed.broadcast(rank_to_fail, 0)
    rank_to_fail = int(rank_to_fail.item())

    if rank != rank_to_fail:
        # this rank is not going to simulate a fault, nothing more to do
        return

    fault_type = fault_desc.fault_type
    if fault_type == 'random':
        fault_type = rng.choice(['rank_killed', 'rank_hung'])

    if fault_type == 'rank_killed':
        target_pid = os.getpid()
    elif fault_type == 'rank_hung':
        target_pid = os.getpid()
    else:
        raise Exception(f"Unknown fault type {fault_type}")

    delay = fault_desc.base_delay + fault_desc.rand_delay * rng.random()

    if rank == 0:
        print(f"Selected fault={fault_type}; target rank={rank_to_fail}; delay={delay}")

    def __fault_thread():
        time.sleep(delay)
        for of in [sys.stdout, sys.stderr]:
            print(
                f"\n####\nSimulating fault: {fault_type}; rank to fail: {rank_to_fail}\n#####\n",
                file=of,
                flush=True,
            )
        if fault_type == 'rank_hung':
            os.kill(target_pid, signal.SIGSTOP)
        else:
            os.kill(target_pid, signal.SIGKILL)

    fault_sim_thread = threading.Thread(target=__fault_thread)
    fault_sim_thread.daemon = True
    fault_sim_thread.start()
