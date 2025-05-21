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

import random
from unittest.mock import patch

import numpy as np
import torch
import torch.multiprocessing as mp

from nvidia_resiliency_ext import attribution.straggler as straggler

from ._utils import multiprocessing_execute_join, multiprocessing_execute_start

#
# Goal of this test module is to check the number of `all_gather_object` calls
# these can be costly, so we want to be sure that there are no unnecessary calls.
#

RANK_DONE_TIMEOUT = 30


def _get_summary(timings):
    stats = {
        straggler.Statistic.MIN: np.min(timings),
        straggler.Statistic.MAX: np.max(timings),
        straggler.Statistic.MED: np.median(timings),
        straggler.Statistic.AVG: np.mean(timings),
        straggler.Statistic.STD: (np.std(timings).item() if len(timings) > 1 else float("nan")),
        straggler.Statistic.NUM: len(timings),
    }
    return stats


def _test_num_of_all_gather_object_calls(*args, **kwargs):

    orig_all_gather_obj = torch.distributed.all_gather_object

    rank = torch.distributed.get_rank()
    random.seed(rank)

    scores_to_compute = ['relative_perf_scores']
    report_gen = straggler.reporting.ReportGenerator(
        scores_to_compute,
        gather_on_rank0=True,
        node_name=f'testnode{rank}',
    )

    # this is just to make the dummy kernel names longer
    KERNEL_POSTFIX = '_' * 1024

    # 1. report 4096 common kernels; each kernel has its dummy summary
    kernel_summaries = {
        f'kernel{k}_{KERNEL_POSTFIX}': _get_summary(np.ones(8)) for k in range(4096)
    }

    with patch('torch.distributed.all_gather_object', wraps=orig_all_gather_obj) as mock_all_gather:
        _ = report_gen.generate_report({}, kernel_summaries=kernel_summaries)
        assert mock_all_gather.call_count == 2  # initial sync: kernel names, nodenames

    # 2. next report with same 4096 common kernels
    with patch('torch.distributed.all_gather_object', wraps=orig_all_gather_obj) as mock_all_gather:
        _ = report_gen.generate_report({}, kernel_summaries=kernel_summaries)
        assert mock_all_gather.call_count == 0  # no all gather object calls this time

    # 3. report with 4096 old and 4096 new kernels
    new_kernel_summaries = {
        f'new_kernel{k}_{KERNEL_POSTFIX}': _get_summary(np.ones(8)) for k in range(4096)
    }
    new_kernel_summaries.update(kernel_summaries)

    with patch('torch.distributed.all_gather_object', wraps=orig_all_gather_obj) as mock_all_gather:
        _ = report_gen.generate_report({}, kernel_summaries=new_kernel_summaries)
        assert mock_all_gather.call_count == 1  #  sync: kernel names

    # 4. next report with same 4096 old and 4096 new kernels
    with patch('torch.distributed.all_gather_object', wraps=orig_all_gather_obj) as mock_all_gather:
        _ = report_gen.generate_report({}, kernel_summaries=new_kernel_summaries)
        assert mock_all_gather.call_count == 0  #  there are no new kernel names to synchronize

    # 5. old kernels for ranks != 0, new kernel for rank 0
    if rank == 0:
        kernel_summaries5_rank0 = {'the_latest_kernel_rank0': _get_summary(np.ones(8))}
    else:
        kernel_summaries5_rank0 = kernel_summaries
    with patch('torch.distributed.all_gather_object', wraps=orig_all_gather_obj) as mock_all_gather:
        _ = report_gen.generate_report({}, kernel_summaries=kernel_summaries5_rank0)
        assert mock_all_gather.call_count == 1  #  new kernel on rank0, so another sync was needed


def test_all_gather_object_calls_num():

    mp_ctx = mp.get_context("spawn")

    rank_processes = multiprocessing_execute_start(
        worker_fn=_test_num_of_all_gather_object_calls,
        mp_ctx=mp_ctx,
        world_size=4,
        backend="gloo",
        dist_store_type="file",
    )

    ret_codes = multiprocessing_execute_join(rank_processes, timeout=RANK_DONE_TIMEOUT)
    assert all(ret_code == 0 for ret_code in ret_codes)
