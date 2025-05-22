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
import sys

import pytest
import torch
import torch.nn as nn

from nvidia_resiliency_ext.attribution import straggler


class Layer(nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        x = self.layer(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--report_interval', type=int, default=100)
    parser.add_argument('--local-rank', default=int(os.getenv('LOCAL_RANK', 0)), type=int)

    # Filter out pytest arguments
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def setup_distributed():
    if int(os.getenv('WORLD_SIZE', '1')) > 1:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')


def skip_if_condition(test_scenario):
    return int(os.getenv('WORLD_SIZE', '1')) == 1 and test_scenario["gather_on_rank0"] is True


@pytest.mark.parametrize(
    "test_scenario",
    [
        {"report_time_interval": 5, "gather_on_rank0": True},
        {"report_time_interval": 0, "gather_on_rank0": True},
        {"report_time_interval": 5, "gather_on_rank0": False},
        {"report_time_interval": 0, "gather_on_rank0": False},
    ],
)
def test_report_elapsed_wrap_callables(test_scenario):
    if skip_if_condition(test_scenario):
        pytest.skip(
            "Testing with gather_on_rank0=True option on multi GPU, but only 1 GPU is available"
        )

    args = parse_args()
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    if world_size > 1:
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda')

    model = nn.Sequential(
        *[Layer(args.hidden, args.hidden, bias=False) for _ in range(args.layers)]
    ).to(device)

    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model)

    random.seed(args.local_rank)

    report_time_interval = test_scenario["report_time_interval"]
    gather_on_rank0 = test_scenario["gather_on_rank0"]

    straggler.Detector.initialize(
        report_time_interval=report_time_interval,
        gather_on_rank0=gather_on_rank0,
    )

    straggler.Detector.wrap_callables(callable_ids=[straggler.CallableId(model, "forward")])

    for i in range(args.iters):
        data = torch.rand(args.batch, args.hidden, device=device)
        model(data)

        assert i == straggler.Detector.report_interval_tracker.current_iter

        if i > straggler.Detector.report_interval_tracker.INTERVAL_ESTIMATION_ITERS:
            assert straggler.Detector.report_interval_tracker.iter_interval is not None

        report = straggler.Detector.generate_report_if_interval_elapsed()

        if gather_on_rank0:
            if rank == 0:
                assert straggler.Detector.report_interval_tracker.is_interval_elapsed() == bool(
                    report is not None
                )
        else:
            assert straggler.Detector.report_interval_tracker.is_interval_elapsed() == bool(
                report is not None
            )

        # check `is_interval_elapsed` public API.
        assert (
            straggler.Detector.report_interval_tracker.is_interval_elapsed()
            == straggler.Detector.is_interval_elapsed()
        )

    straggler.Detector.shutdown()

    if world_size > 1:
        torch.distributed.barrier()


@pytest.mark.parametrize(
    "test_scenario",
    [
        {"report_time_interval": 5, "gather_on_rank0": True},
        {"report_time_interval": 0, "gather_on_rank0": True},
        {"report_time_interval": 5, "gather_on_rank0": False},
        {"report_time_interval": 0, "gather_on_rank0": False},
    ],
)
def test_report_elapsed_det_section(test_scenario):
    if skip_if_condition(test_scenario):
        pytest.skip(
            "Testing with gather_on_rank0=True option on multi GPU, but only 1 GPU is available"
        )

    args = parse_args()
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    if world_size > 1:
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda')

    model = nn.Sequential(
        *[Layer(args.hidden, args.hidden, bias=False) for _ in range(args.layers)]
    ).to(device)

    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model)

    random.seed(args.local_rank)

    report_time_interval = test_scenario["report_time_interval"]
    gather_on_rank0 = test_scenario["gather_on_rank0"]

    straggler.Detector.initialize(
        report_time_interval=report_time_interval,
        gather_on_rank0=gather_on_rank0,
    )

    for i in range(args.iters):
        data = torch.rand(args.batch, args.hidden, device=device)

        with straggler.Detector.detection_section("fwd", profile_cuda=True):
            model(data)

        assert i == straggler.Detector.report_interval_tracker.current_iter

        if i > straggler.Detector.report_interval_tracker.INTERVAL_ESTIMATION_ITERS:
            assert straggler.Detector.report_interval_tracker.iter_interval is not None

        report = straggler.Detector.generate_report_if_interval_elapsed()

        if gather_on_rank0:
            if rank == 0:
                assert straggler.Detector.report_interval_tracker.is_interval_elapsed() == bool(
                    report is not None
                )
        else:
            assert straggler.Detector.report_interval_tracker.is_interval_elapsed() == bool(
                report is not None
            )

    straggler.Detector.shutdown()

    if world_size > 1:
        torch.distributed.barrier()


def test_report_min_interval_is_profiling_interval():

    # Ensure that estimated reporting interval is at least as large as the profiling interval.
    # It makes no sense to report more frequently than the profiling interval (some reports would be empty).

    args = parse_args()
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda')

    model = nn.Sequential(
        *[Layer(args.hidden, args.hidden, bias=False) for _ in range(args.layers)]
    ).to(device)

    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model)

    random.seed(args.local_rank)

    profiling_interval = 1000  # large profiling interval, only 1th per 1e6 will be profiled
    straggler.Detector.initialize(
        profiling_interval=profiling_interval,
        report_time_interval=0.01,  # computed reporting interval should be small
        gather_on_rank0=True,
    )

    for _ in range(args.iters):
        data = torch.rand(args.batch, args.hidden, device=device)

        with straggler.Detector.detection_section("fwd", profile_cuda=True):
            _ = model(data)

        report = straggler.Detector.generate_report_if_interval_elapsed()
        if report:
            assert straggler.Detector.report_interval_tracker.is_interval_elapsed()
            break

    assert straggler.Detector.report_interval_tracker.iter_interval == profiling_interval

    straggler.Detector.shutdown()

    if world_size > 1:
        torch.distributed.barrier()


if __name__ == '__main__':
    setup_distributed()
    if torch.distributed.get_rank() == 0:
        pytest.main(["tests/unit/test_reporting_elapsed.py"])
    else:
        pytest.main(["-q", "tests/unit/test_reporting_elapsed.py"])
