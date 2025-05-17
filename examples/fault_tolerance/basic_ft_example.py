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

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


# FT: import NVRx
import nvidia_resiliency_ext.fault_tolerance as ft

# Simple example of using the FT library with PyTorch DDP.
# This script trains a dummy model on dummy data. CPU is used for training.
# After each epoch, FT timeouts are calculated and saved to the file "./ft_state.json".
#
# You can run it using:
# `ft_launcher --nproc-per-node=4 --max-restarts=3 --ft-initial-rank-heartbeat-timeout=30 --ft-rank-heartbeat-timeout=15 examples/fault_tolerance/basic_ft_example.py`
# In this example configuration, at most 3 training restarts are allowed.
#
# To find rank PIDs, use:
# `ps aux | grep basic_ft_example.py | grep -v grep`
#
# Examples:
#
# 1. Hang detection using predefined timeouts:
#    - Remove `ft_state.json` if it exists (`rm ft_state.json`).
#    - During the 0th epoch, stop a rank using `kill -SIGSTOP <rank_pid>`.
#    - After approximately 15 seconds, a "Did not get subsequent heartbeat." error should be raised.
#    - All ranks will be restarted.
#
# 2. Hang detection using computed timeouts:
#    - Run the example for more than 1 epoch to allow FT timeouts to be calculated.
#    - Stop a rank using `kill -SIGSTOP <rank_pid>`.
#    - After the computed timeout elapses, a "Did not get subsequent heartbeat." error should be raised.
#    - All ranks will be restarted.
#
# 3. Rank error handling:
#    - Kill a rank using `kill -SIGKILL <rank_pid>`.
#    - All ranks will be restarted.


FEAT_SIZE = 4096
DNN_OUT_SIZE = 128
BATCH_SIZE = 100
NUM_EPOCHS = 10
DATASET_LEN = 100000


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.rand((FEAT_SIZE,), dtype=torch.float32, device='cpu')
        y = torch.rand((DNN_OUT_SIZE,), dtype=torch.float32, device='cpu')
        return x, y


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(FEAT_SIZE, FEAT_SIZE)
        self.fc2 = nn.Linear(FEAT_SIZE, DNN_OUT_SIZE)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


def print_on_rank0(msg):
    if dist.get_rank() == 0:
        print(msg)


def main(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print_on_rank0(f"Starting new training run... World size={dist.get_world_size()}")

    # FT: initialize the client
    ft_client = ft.RankMonitorClient()
    ft_client.init_workload_monitoring()
    print_on_rank0(f"FT initialized. Timeouts: {ft_client.hb_timeouts}")
    # FT: load state (calculated timeouts)
    if os.path.exists("ft_state.json"):
        with open("ft_state.json", "r") as f:
            ft_state = json.load(f)
            ft_client.load_state_dict(ft_state)
        print_on_rank0(f"FT timeouts {ft_client.hb_timeouts} loaded from ft_state.json")

    # Dataset and DataLoader with DistributedSampler
    dataset = SimpleDataset(size=DATASET_LEN)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    # Model, optimizer, and DDP
    model = SimpleModel()
    ddp_model = DDP(model)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    num_iters_in_epoch = len(dataloader)
    num_iters_for_10pct = num_iters_in_epoch // 10  # iters for 1/10 of epoch

    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(dataloader):
            if (batch_idx % num_iters_for_10pct) == 0 and rank == 0:
                print(f"Epoch {epoch} progress: {100 * batch_idx / num_iters_in_epoch:.2f}%")
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            # FT: send heartbeat to the server
            ft_client.send_heartbeat()
        print_on_rank0(f"Epoch {epoch} complete. Loss: {loss.item()}")
        # FT: calculate and set new timeouts
        ft_client.calculate_and_set_hb_timeouts()
        # FT: save the state (calculated timeouts)
        with open("ft_state.json", "w") as f:
            json.dump(ft_client.state_dict(), f)
        print_on_rank0(f"FT timeouts {ft_client.hb_timeouts} saved to ft_state.json")

    # FT: shutdown the client
    ft_client.shutdown_workload_monitoring()
    dist.destroy_process_group()


if __name__ == "__main__":
    import os

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    main(rank, world_size)
