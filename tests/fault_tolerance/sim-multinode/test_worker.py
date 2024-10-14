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

import json
import os
import random
import signal
import socket
import sys
import threading
import time

import torch
import torch.distributed as dist

from nvidia_resiliency_ext import fault_tolerance

# Tester script for multi-node FT test
# Initializes fault tolerance, periodically sends heartbeats
# Crashes or hangs random rank after some delay
# Checks if workload is restarted after a failure and the same ranks land on the same nodes

SIM_FAULT_BASE_DELAY = 2
SIM_FAULT_MAX_RAND_DELAY = 3
MAX_RUN_INTERVAL = 60
STATE_FILE_PATH_PATT = "/workspace/_rank{rank}-assignment-test-state.json"
NUM_RUNS = 16


def _print_on_rank0(msg):
    if dist.get_rank() == 0:
        print(msg)


def _setup_simulated_fault():
    rng = random.Random()

    fault_type = rng.choice(['rank_killed', 'rank_hanged'])

    rank_to_fail = rng.randint(0, dist.get_world_size() - 1)
    rank_to_fail = torch.tensor([rank_to_fail])
    dist.broadcast(rank_to_fail, 0)
    rank_to_fail = int(rank_to_fail.item())

    rank = torch.distributed.get_rank()
    if rank != rank_to_fail:
        return

    if fault_type == 'rank_killed':
        target_pid = os.getpid()
        target_sig = signal.SIGKILL
    elif fault_type == 'rank_hanged':
        target_pid = os.getpid()
        target_sig = signal.SIGSTOP
    else:
        raise Exception(f"Unknown fault type {fault_type}")

    delay = SIM_FAULT_BASE_DELAY + SIM_FAULT_MAX_RAND_DELAY * random.random()

    def __fault_thread():
        time.sleep(delay)
        print(
            f"\n####\nSimulating fault: {fault_type}; rank to fail: {rank_to_fail}\n#####\n",
            file=sys.stderr,
        )
        os.kill(target_pid, target_sig)

    fault_sim_thread = threading.Thread(target=__fault_thread)
    fault_sim_thread.daemon = True
    fault_sim_thread.start()


def _load_state(rank):
    fn = STATE_FILE_PATH_PATT.format(rank=rank)
    if not os.path.exists(fn):
        return {}
    with open(fn, "r") as f:
        return json.load(f)


def _get_new_state(prev_state):
    curr_state = {
        'is_failed': False,
        'is_finished': False,
        'node': socket.gethostname(),
        'run_idx': prev_state['run_idx'] + 1 if prev_state else 0,
        'start_time': time.monotonic(),
    }
    if prev_state:
        if prev_state['node'] != curr_state['node']:
            curr_state['is_finished'] = True
            curr_state['is_failed'] = True
            curr_state['failure_reason'] = "Ranks assignment changed."
        elif (curr_state['start_time'] - prev_state['start_time']) > MAX_RUN_INTERVAL:
            curr_state['is_finished'] = True
            curr_state['is_failed'] = True
            curr_state['failure_reason'] = "Interval between runs exceeded limit."

    if curr_state['run_idx'] == NUM_RUNS:
        curr_state['is_finished'] = True

    return curr_state


def _save_state(rank, state):
    fn = STATE_FILE_PATH_PATT.format(rank=rank)
    with open(fn, "w") as f:
        json.dump(state, f)


if __name__ == '__main__':
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()

    print(f"Running rank {rank} on node {socket.gethostname()}.")

    loaded_state = _load_state(rank)
    curr_state = _get_new_state(loaded_state)
    _save_state(rank, curr_state)

    dist.barrier()
    _print_on_rank0(f"### RUN {curr_state['run_idx']}/{NUM_RUNS} ###")

    if curr_state['is_finished']:
        dist.barrier()
        if curr_state['is_failed']:
            print(f"TEST FAILED rank={rank} final state={curr_state}")
        else:
            assert curr_state['run_idx'] == NUM_RUNS
            _print_on_rank0("TEST SUCCEEDED")
        sys.exit(0)  # return 0 so launcher wont respawn the workload

    ft_client = fault_tolerance.RankMonitorClient()
    ft_client.init_workload_monitoring()

    _setup_simulated_fault()

    for i in range(1000000):
        ft_client.send_heartbeat()
        time.sleep(0.5)
