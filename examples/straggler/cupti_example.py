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

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

from nvidia_resiliency_ext.straggler import cupti_module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=10)
    # training time in minutes
    parser.add_argument("--train_time", type=float, default=1)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--iters", type=int, default=100_000)

    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--use_cupti", action="store_true")
    parser.add_argument("--use_prof", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--local-rank", default=int(os.getenv("LOCAL_RANK", 0)), type=int)

    args = parser.parse_args()

    return args


class Layer(nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        x = self.layer(x)
        return x


def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))


def benchmark(args):
    if args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    device = torch.device("cuda")
    model = nn.Sequential(
        *[Layer(args.hidden, args.hidden, bias=False) for _ in range(args.layers)]
    ).to(device, dtype)
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model)

    data = torch.rand(args.batch, args.hidden, device=device, dtype=dtype)
    iters_finished = args.iters  # if it is not changed further, then all the iters finished
    torch.cuda.synchronize()
    t_end = time.perf_counter() + 60 * args.train_time  # for args.train_time mins

    if args.use_prof:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for i in range(args.iters):
                torch.cuda.synchronize()
                if time.perf_counter() > t_end:
                    iters_finished = i
                    break
                out = model(data)
                if args.backward:
                    out.backward(torch.ones_like(out))
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    else:
        for i in range(args.iters):
            torch.cuda.synchronize()
            if time.perf_counter() > t_end:
                iters_finished = i
                break
            out = model(data)
            if args.backward:
                out.backward(torch.ones_like(out))

    return iters_finished


def main():
    start_time = time.perf_counter()
    args = parse_args()
    print(args)

    iters_finished_list = []

    print("world size", int(os.getenv("WORLD_SIZE", "1")))
    torch.cuda.set_device(args.local_rank)
    if int(os.getenv("WORLD_SIZE", "1")) > 1:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # warmup:
    iters_finished_list.append(benchmark(args))

    if args.use_cupti:
        profiler = cupti_module.CuptiProfiler(bufferSize=1024 * 1024 * 8, numBuffers=10)
        profiler.initialize()

    iters_finished_list.append(benchmark(args))

    if args.use_cupti:
        profiler.stop()

    iters_finished_list.append(benchmark(args))

    if args.use_cupti:
        profiler.start()

    iters_finished_list.append(benchmark(args))

    if args.use_cupti:
        profiler.stop()

    iters_finished_list.append(benchmark(args))

    if args.use_cupti:
        profiler.start()

    iters_finished_list.append(benchmark(args))

    if args.use_cupti:
        stats = cupti_module.get_stats(profiler)  # Retrieve and print statistics
        for kernel_name, stats in stats.items():
            print(f"{kernel_name}: {stats}")
        profiler.shutdown()  # Clean up and flush CUPTI activities

    print(f"Used CUPTI API: {args.use_cupti}")
    print(f"Used PyTorch Profiler: {args.use_prof}")
    print(f"Iterations for {args.train_time} min: {iters_finished_list}")

    elapsed_time = time.perf_counter() - start_time
    print(f"Whole script execution time: {elapsed_time} seconds")


if __name__ == "__main__":
    main()
