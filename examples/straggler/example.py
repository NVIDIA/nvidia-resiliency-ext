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

#
# This is a basic example of straggler detection usage with a simple DDP workload
# It uses straggler detection API to wrap the forward pass and measure GPU performance
# GPU performance scores are printed at regular intervals
# You can try "nvidia-smi -i <GPU idx> -lgc 800" to slow down some GPUs and see the effect.
#

import argparse
import os
import time
import uuid

import torch
import torch.distributed as dist
import torch.distributed.launcher as pet
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from nvidia_resiliency_ext import attribution.straggler as straggler


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.layers(x)


def train(args) -> None:
    print(args)

    straggler.Detector.initialize(gather_on_rank0=True)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    torch.manual_seed(42)

    print(f"Running basic straggler det. DDP example on device {device}.")
    model = Model().to(device)

    ddp_model = DDP(model, device_ids=[local_rank])

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    optim = torch.optim.SGD(ddp_model.parameters(), lr=0.01, momentum=0.5)
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch_num = 0

    ddp_model.train()
    total_iters_made = 0
    training_start_time = time.monotonic()

    while epoch_num < args.num_epochs:
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            with straggler.Detector.detection_section("fwd", profile_cuda=True):
                output = ddp_model(data)

            loss = loss_fn(output, target)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if (batch_idx % args.log_interval) == 0 and rank == 0:
                print(
                    f"Rank {local_rank}, Epoch {epoch_num}, Batch {batch_idx}, Loss {loss.item()}"
                )

            if (batch_idx % args.report_interval) == 0:
                report = straggler.Detector.generate_report()
                if rank == 0:
                    print(
                        f"Rank {local_rank} GPUs relative perf: {report.gpu_relative_perf_scores}"
                    )
                    print(
                        f"Rank {local_rank} GPUs individual perf: {report.gpu_individual_perf_scores}"
                    )

            total_iters_made += 1
        epoch_num += 1

    training_stop_time = time.monotonic()
    time_per_iter = (training_stop_time - training_start_time) / total_iters_made
    print(f"Time per iteration [sec]: {time_per_iter:.5f}")

    straggler.Detector.shutdown()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--report-interval", type=int, default=300)

    args: argparse.Namespace = parser.parse_args()

    lc = pet.LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=args.num_processes,
        run_id=str(uuid.uuid4()),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        max_restarts=0,
        monitor_interval=1,
    )

    pet.elastic_launch(lc, entrypoint=train)(args)


if __name__ == "__main__":
    main()
