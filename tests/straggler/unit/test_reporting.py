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
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from nvidia_resiliency_ext import attribution.straggler as straggler


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
    parser.add_argument('--backward', action='store_true')
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
        {"scores_to_compute": "all", "gather_on_rank0": True},
        {"scores_to_compute": "all", "gather_on_rank0": False},
        {
            "scores_to_compute": ["relative_perf_scores"],
            "gather_on_rank0": True,
        },
        {
            "scores_to_compute": ["relative_perf_scores"],
            "gather_on_rank0": False,
        },
        {
            "scores_to_compute": ["individual_perf_scores"],
            "gather_on_rank0": True,
        },
        {
            "scores_to_compute": ["individual_perf_scores"],
            "gather_on_rank0": False,
        },
    ],
)
def test_reporting_options(test_scenario, monkeypatch):
    if skip_if_condition(test_scenario):
        pytest.skip("Testing gather_on_rank0 option on multi GPU, but only 1 GPU is available")

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

    scores_to_compute = test_scenario["scores_to_compute"]
    gather_on_rank0 = test_scenario["gather_on_rank0"]

    straggler.Detector.initialize(
        scores_to_compute=scores_to_compute, gather_on_rank0=gather_on_rank0
    )

    straggler.Detector.wrap_callables(callable_ids=[straggler.CallableId(model, "forward")])

    for i in range(args.iters):
        data = torch.rand(args.batch, args.hidden, device=device)
        out = model(data)
        if args.backward:
            out.backward(torch.ones_like(out))

        if (i % args.report_interval) == 0:
            report = straggler.Detector.generate_report()
            if gather_on_rank0:
                if rank == 0:
                    report_len = world_size
                    stragglers = report.identify_stragglers()
                else:
                    assert report is None
                    continue
            else:
                report_len = 1
                stragglers = report.identify_stragglers()

            if "relative_perf_scores" in scores_to_compute:
                assert len(report.gpu_relative_perf_scores) == report_len
                for (
                    section_name,
                    section_scores,
                ) in report.section_relative_perf_scores.items():
                    assert len(section_scores) == report_len

            if "individual_perf_scores" in scores_to_compute:
                assert len(report.gpu_individual_perf_scores) == report_len
                for (
                    section_name,
                    section_scores,
                ) in report.section_individual_perf_scores.items():
                    assert len(section_scores) == report_len

            for straggler_type in [
                "straggler_gpus_relative",
                "straggler_gpus_individual",
                "straggler_sections_relative",
                "straggler_sections_individual",
            ]:
                assert straggler_type in stragglers

            if scores_to_compute == "all":
                scores_to_compute = [
                    "relative_perf_scores",
                    "individual_perf_scores",
                ]

            assert bool(report.gpu_relative_perf_scores) == (
                "relative_perf_scores" in scores_to_compute
            )
            assert bool(report.section_relative_perf_scores) == (
                "relative_perf_scores" in scores_to_compute
            )
            assert bool(report.gpu_individual_perf_scores) == (
                "individual_perf_scores" in scores_to_compute
            )
            assert bool(report.section_individual_perf_scores) == (
                "individual_perf_scores" in scores_to_compute
            )

    straggler.Detector.shutdown()

    if world_size > 1:
        torch.distributed.barrier()


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1,
    reason="Testing no gather happens with individual_perf_scores and gather_on_rank0=False on multi GPU, only 1 GPU is available",
)
def test_no_gather_called(monkeypatch):
    with patch('torch.distributed.all_gather_object') as mock_all_gather:
        mock_all_gather.side_effect = RuntimeError("Distributed communication should not be used")
        test_scenario = {
            "scores_to_compute": "individual_perf_scores",
            "gather_on_rank0": False,
        }
        test_reporting_options(test_scenario=test_scenario, monkeypatch=monkeypatch)
        mock_all_gather.assert_not_called()


if __name__ == '__main__':
    setup_distributed()
    if torch.distributed.get_rank() == 0:
        pytest.main(["tests/unit/test_reporting.py"])
    else:
        pytest.main(["-q", "tests/unit/test_reporting.py"])
