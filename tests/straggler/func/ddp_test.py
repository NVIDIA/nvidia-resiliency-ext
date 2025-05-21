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
import random
import time
from typing import Dict

import torch
import torch.nn as nn

from nvidia_resiliency_ext import attribution.straggler as straggler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--iters', type=int, default=1000000)
    parser.add_argument('--max_runtime', type=int, default=300)
    parser.add_argument('--local-rank', default=int(os.getenv('LOCAL_RANK', 0)), type=int)

    # Straggler detection arguments
    parser.add_argument(
        '--no_rel_scores',
        action='store_true',
        help='Do not compute relative performance scores',
    )
    parser.add_argument(
        '--no_indiv_scores',
        action='store_true',
        help='Do not compute individual performance scores',
    )
    parser.add_argument(
        '--no_gather_on_rank0',
        action='store_true',
        help='Set gather_on_rank0 to False',
    )
    parser.add_argument(
        '--report_iter_interval',
        type=int,
        default=1000,
        help='Interval for .generate_report in iterations',
    )
    parser.add_argument(
        '--generate_if_elapsed',
        action='store_true',
        default=False,
        help='Use .generate_report_if_interval_elapsed',
    )
    parser.add_argument(
        '--report_time_interval',
        type=float,
        default=60,
        help='Time interval for .generate_report_if_interval_elapsed in seconds',
    )
    parser.add_argument(
        '--straggler_gpu_rel_threshold',
        type=float,
        default=0.7,
        help='Threshold for identify_stragglers',
    )
    parser.add_argument(
        '--straggler_gpu_indiv_threshold',
        type=float,
        default=0.7,
        help='Threshold for identify_stragglers',
    )
    parser.add_argument(
        '--straggler_section_rel_threshold',
        type=float,
        default=0.7,
        help='Threshold for identify_stragglers',
    )
    parser.add_argument(
        '--straggler_section_indiv_threshold',
        type=float,
        default=0.7,
        help='Threshold for identify_stragglers',
    )

    args = parser.parse_args()

    args.gather_on_rank0 = not args.no_gather_on_rank0
    args.scores_to_compute = []

    if not args.no_rel_scores:
        args.scores_to_compute.append("relative_perf_scores")

    if not args.no_indiv_scores:
        args.scores_to_compute.append("individual_perf_scores")

    del args.no_gather_on_rank0, args.no_indiv_scores, args.no_rel_scores

    return args


class Layer(nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        x = self.layer(x)
        return x


def round_float_values(d: Dict) -> Dict:
    return {k: round(v, 2) for k, v in d.items()}


def print_gpu_scores(report, rank):
    print(f"=== GPUs perf scores. Report from rank {rank} ===")
    rel_scores = round_float_values(report.gpu_relative_perf_scores)
    print("GPU relative perf scores:", rel_scores)
    indiv_scores = round_float_values(report.gpu_individual_perf_scores)
    print("GPU individual perf scores:", indiv_scores)


def print_section_scores(report, rank):
    print(f"=== Sections perf scores. Report from rank {rank} ===")
    rel_scores = {}
    for section in report.section_relative_perf_scores:
        rel_scores[section] = round_float_values(report.section_relative_perf_scores[section])
    print("Sections relative perf scores:", rel_scores)
    indiv_scores = {}
    for section in report.section_individual_perf_scores:
        indiv_scores[section] = round_float_values(report.section_individual_perf_scores[section])
    print("Sections individual perf scores:", indiv_scores)


def print_stragglers(stragglers):
    # Print stragglers in easy to parse format
    for s in stragglers['straggler_gpus_relative']:
        print(f"DETECTED RELATIVE STRAGGLER GPU RANK={s.rank} NODE={s.node}")
    for s in stragglers['straggler_gpus_individual']:
        print(f"DETECTED INDIVIDUAL STRAGGLER GPU RANK={s.rank} NODE={s.node}")
    for section in stragglers['straggler_sections_relative']:
        for s in stragglers['straggler_sections_relative'][section]:
            print(f"DETECTED RELATIVE STRAGGLER SECTION={section} RANK={s.rank} NODE={s.node}")
    for section in stragglers['straggler_sections_individual']:
        for s in stragglers['straggler_sections_individual'][section]:
            print(f"DETECTED INDIVIDUAL STRAGGLER SECTION={section} RANK={s.rank} NODE={s.node}")


def _all_reduce_bool_flag(flag):
    flag = torch.tensor([flag], dtype=torch.float, device='cuda')
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item() > 0)


def print_rank0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)


def main():
    args = parse_args()
    print(args)

    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    rank = torch.distributed.get_rank()

    random.seed(rank)

    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda')

    model = nn.Sequential(
        *[Layer(args.hidden, args.hidden, bias=False) for _ in range(args.layers)]
    ).to(device)

    model = torch.nn.parallel.DistributedDataParallel(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

    straggler.Detector.initialize(
        report_time_interval=args.report_time_interval,
        scores_to_compute=args.scores_to_compute,
        gather_on_rank0=args.gather_on_rank0,
    )

    straggler.Detector.wrap_callables(callable_ids=[straggler.CallableId(model, "forward")])

    t0 = time.monotonic()
    report_idx = 1

    for i in range(args.iters):

        data = torch.rand(args.batch, args.hidden, device=device)
        target_tensor = torch.rand(args.batch, args.hidden, device=device)

        out = model(data)

        optimizer.zero_grad()

        loss = criterion(out, target_tensor)
        loss.backward()

        optimizer.step()

        report = None
        if args.generate_if_elapsed:
            report = straggler.Detector.generate_report_if_interval_elapsed()

        elif i > 0 and (i % args.report_iter_interval) == 0:
            report = straggler.Detector.generate_report()

        if report:
            print(f"STRAGGLER REPORT #{report_idx}")
            print_gpu_scores(report, rank)
            print_section_scores(report, rank)
            stragglers = report.identify_stragglers(
                gpu_rel_threshold=args.straggler_gpu_rel_threshold,
                section_rel_threshold=args.straggler_section_rel_threshold,
                gpu_indiv_threshold=args.straggler_gpu_indiv_threshold,
                section_indiv_threshold=args.straggler_section_indiv_threshold,
            )
            print_stragglers(stragglers)
            report_idx += 1

        elapsed = time.monotonic() - t0
        stop_flag = args.max_runtime and elapsed > args.max_runtime
        if _all_reduce_bool_flag(stop_flag):
            print_rank0("Time limit reached. Stopping.")
            break

    straggler.Detector.shutdown()
    print_rank0("DONE")


if __name__ == '__main__':
    main()
