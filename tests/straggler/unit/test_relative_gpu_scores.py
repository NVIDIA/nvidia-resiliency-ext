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

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from nvidia_resiliency_ext import attribution.straggler as straggler

from ._utils import multiprocessing_execute_join, multiprocessing_execute_start

#
# Tests of `straggler.reporting.ReportGenerator` relative GPU scores computation
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


def test_relative_gpu_scores_one_rank():
    # Relative score is always 1.0 when there is just one rank
    scores_to_compute = ['relative_perf_scores']
    report_gen = straggler.reporting.ReportGenerator(
        scores_to_compute, gather_on_rank0=False, node_name='testnode'
    )

    kernel_summaries = {
        'kernel0': _get_summary(np.array([1.0, 1.0, 2.0])),
    }
    report = report_gen.generate_report({}, kernel_summaries=kernel_summaries)
    assert report.gpu_relative_perf_scores[0] == pytest.approx(1.0)
    kernel_summaries = {
        'kernel0': _get_summary(1.25 * np.array([1.0, 1.0, 2.0])),
    }
    report = report_gen.generate_report({}, kernel_summaries=kernel_summaries)
    assert report.gpu_relative_perf_scores[0] == pytest.approx(1.0)


def _rank_main_gpu_relative_test(*args, gather_on_rank0, ret_queue, **kwargs):

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    random.seed(rank)

    scores_to_compute = ['relative_perf_scores']
    report_gen = straggler.reporting.ReportGenerator(
        scores_to_compute,
        gather_on_rank0=gather_on_rank0,
        node_name=f'testnode{rank}',
    )

    # multiply base timing by rank+1, just to obtain different scores on different ranks
    # nccl* kernel should be ignored
    kernel_summaries = {
        'kernel0': _get_summary((rank + 1) * np.array([1.0, 1.0, 2.0])),
        'kernel1': _get_summary((rank + 1) * np.array([2.0, 2.0, 3.0])),
        'ncclDevKernel_AllReduce_Sum': _get_summary(
            (world_size - rank) * np.array([10.0, 20.0, 30.0])
        ),
    }

    # make it trickier by randomizing the summaries order
    shuffled_kernel_names = list(kernel_summaries.keys())
    random.shuffle(shuffled_kernel_names)
    kernel_summaries = {n: kernel_summaries[n] for n in shuffled_kernel_names}

    report = report_gen.generate_report({}, kernel_summaries=kernel_summaries)
    ret_queue.put((rank, report))


def test_relative_gpu_scores_gather_on_rank0():

    # Check that all scores are gathered on rank0 if gather_on_rank0=True

    mp_ctx = mp.get_context("spawn")
    ret_queue = mp_ctx.Queue()

    rank_processes = multiprocessing_execute_start(
        worker_fn=_rank_main_gpu_relative_test,
        mp_ctx=mp_ctx,
        world_size=4,
        backend="gloo",
        dist_store_type="file",
        gather_on_rank0=True,
        ret_queue=ret_queue,
    )

    reports = {}
    for _ in range(4):
        rank, report = ret_queue.get(timeout=RANK_DONE_TIMEOUT)
        reports[rank] = report

    ret_codes = multiprocessing_execute_join(rank_processes, timeout=RANK_DONE_TIMEOUT)
    assert all(ret_code == 0 for ret_code in ret_codes)

    assert len(reports) == 4
    # results for all ranks should be included in the report generated on rank0
    assert len(reports[0].gpu_relative_perf_scores) == 4
    assert reports[0].gpu_relative_perf_scores[0] == pytest.approx(1.0)
    assert reports[0].gpu_relative_perf_scores[1] == pytest.approx(1.0 / 2.0)
    assert reports[0].gpu_relative_perf_scores[2] == pytest.approx(1.0 / 3.0)
    assert reports[0].gpu_relative_perf_scores[3] == pytest.approx(1.0 / 4.0)
    # reports from other ranks should be None or empty
    assert not reports[1]
    assert not reports[2]
    assert not reports[3]
    # check rank to node mapping in the report
    assert reports[0].rank_to_node[0] == 'testnode0'
    assert reports[0].rank_to_node[1] == 'testnode1'
    assert reports[0].rank_to_node[2] == 'testnode2'
    assert reports[0].rank_to_node[3] == 'testnode3'
    # there should be no individual scores
    assert not reports[0].gpu_individual_perf_scores


def test_relative_gpu_scores_no_gather():

    # Check that each rank has its own scores if gather_on_rank0=False

    mp_ctx = mp.get_context("spawn")
    ret_queue = mp_ctx.Queue()

    rank_processes = multiprocessing_execute_start(
        worker_fn=_rank_main_gpu_relative_test,
        mp_ctx=mp_ctx,
        world_size=4,
        backend="gloo",
        dist_store_type="file",
        gather_on_rank0=False,
        ret_queue=ret_queue,
    )

    reports = {}
    for _ in range(4):
        rank, report = ret_queue.get(timeout=RANK_DONE_TIMEOUT)
        reports[rank] = report

    ret_codes = multiprocessing_execute_join(rank_processes, timeout=RANK_DONE_TIMEOUT)
    assert all(ret_code == 0 for ret_code in ret_codes)

    assert len(reports) == 4
    # check if each rank returned its own report only
    assert len(reports[0].gpu_relative_perf_scores) == 1
    assert reports[0].gpu_relative_perf_scores[0] == pytest.approx(1.0)
    assert len(reports[1].gpu_relative_perf_scores) == 1
    assert reports[1].gpu_relative_perf_scores[1] == pytest.approx(1.0 / 2.0)
    assert len(reports[2].gpu_relative_perf_scores) == 1
    assert reports[2].gpu_relative_perf_scores[2] == pytest.approx(1.0 / 3.0)
    assert len(reports[3].gpu_relative_perf_scores) == 1
    assert reports[3].gpu_relative_perf_scores[3] == pytest.approx(1.0 / 4.0)
    # check rank to node mapping in the reports
    # each report should contain only its own rank to node mapping
    assert len(reports[0].rank_to_node) == 1
    assert len(reports[1].rank_to_node) == 1
    assert len(reports[2].rank_to_node) == 1
    assert len(reports[3].rank_to_node) == 1
    assert reports[0].rank_to_node[0] == 'testnode0'
    assert reports[1].rank_to_node[1] == 'testnode1'
    assert reports[2].rank_to_node[2] == 'testnode2'
    assert reports[3].rank_to_node[3] == 'testnode3'


def _rank_main_gpu_relative_test_some_common_kernels(*args, gather_on_rank0, ret_queue, **kwargs):

    rank = torch.distributed.get_rank()
    random.seed(rank)

    scores_to_compute = ['relative_perf_scores']
    report_gen = straggler.reporting.ReportGenerator(
        scores_to_compute,
        gather_on_rank0=gather_on_rank0,
        node_name=f'testnode{rank}',
    )

    # there is one common kernel and one unique kernel on each rank
    kernel_summaries = {
        'kernel_common': _get_summary((rank + 1) * np.array([1.0, 1.0, 2.0])),
        f'kernel_only_on_rank{rank}': _get_summary((rank + 1) * np.array([2.0, 2.0, 3.0])),
    }
    report = report_gen.generate_report({}, kernel_summaries=kernel_summaries)
    ret_queue.put((rank, report))


def test_relative_gpu_scores_some_common_kernels():

    # Check that scores are computed when some kernels are common across ranks,
    # but at the same time other kernels are unique to each rank.

    mp_ctx = mp.get_context("spawn")
    ret_queue = mp_ctx.Queue()

    rank_processes = multiprocessing_execute_start(
        worker_fn=_rank_main_gpu_relative_test_some_common_kernels,
        mp_ctx=mp_ctx,
        world_size=4,
        backend="gloo",
        dist_store_type="file",
        gather_on_rank0=True,
        ret_queue=ret_queue,
    )

    reports = {}
    for _ in range(4):
        rank, report = ret_queue.get(timeout=RANK_DONE_TIMEOUT)
        reports[rank] = report

    ret_codes = multiprocessing_execute_join(rank_processes, timeout=RANK_DONE_TIMEOUT)
    assert all(ret_code == 0 for ret_code in ret_codes)

    assert len(reports) == 4
    # results for all ranks should be included in the report generated on rank0
    assert len(reports[0].gpu_relative_perf_scores) == 4
    assert reports[0].gpu_relative_perf_scores[0] == pytest.approx(1.0)
    assert reports[0].gpu_relative_perf_scores[1] == pytest.approx(1.0 / 2.0)
    assert reports[0].gpu_relative_perf_scores[2] == pytest.approx(1.0 / 3.0)
    assert reports[0].gpu_relative_perf_scores[3] == pytest.approx(1.0 / 4.0)


def _rank_main_gpu_relative_test_no_common_kernels_in_rank(
    *args,
    ranks_with_unique_kernels=(),
    ranks_without_kernels=(),
    gather_on_rank0=True,
    ret_queue,
    **kwargs,
):

    rank = torch.distributed.get_rank()
    random.seed(rank)

    scores_to_compute = ['relative_perf_scores']
    report_gen = straggler.reporting.ReportGenerator(
        scores_to_compute,
        gather_on_rank0=gather_on_rank0,
        node_name=f'testnode{rank}',
    )

    # put unique kernels in some ranks OR no kernels at all
    if rank in ranks_with_unique_kernels:
        kernel_summaries = {
            f'rank_specific_kernel{rank}': _get_summary(np.array([99.0, 99.0, 99.0])),
        }
    elif rank in ranks_without_kernels:
        kernel_summaries = {}
    else:
        kernel_summaries = {
            'kernel_common0': _get_summary((rank + 1) * np.array([1.0, 1.0, 2.0])),
            'kernel_common1': _get_summary((rank + 1) * np.array([2.0, 2.0, 3.0])),
        }

    report = report_gen.generate_report({}, kernel_summaries=kernel_summaries)
    ret_queue.put((rank, report))


def test_relative_gpu_scores_ranks_with_unique_kernels():

    # Check that (all) scores are NaN when there are no common kernels for all ranks.
    # TODO: should we compute the results for a subset of ranks with common kernels?

    mp_ctx = mp.get_context("spawn")
    ret_queue = mp_ctx.Queue()

    rank_processes = multiprocessing_execute_start(
        worker_fn=_rank_main_gpu_relative_test_no_common_kernels_in_rank,
        mp_ctx=mp_ctx,
        world_size=4,
        backend="gloo",
        dist_store_type="file",
        ranks_with_unique_kernels=(
            1,
            2,
        ),
        gather_on_rank0=True,
        ret_queue=ret_queue,
    )

    reports = {}
    for _ in range(4):
        rank, report = ret_queue.get(timeout=RANK_DONE_TIMEOUT)
        reports[rank] = report

    ret_codes = multiprocessing_execute_join(rank_processes, timeout=RANK_DONE_TIMEOUT)
    assert all(ret_code == 0 for ret_code in ret_codes)

    assert len(reports) == 4
    # results for all ranks should be included in the report generated on rank0
    assert len(reports[0].gpu_relative_perf_scores) == 4
    assert np.isnan(reports[0].gpu_relative_perf_scores[0])
    assert np.isnan(reports[0].gpu_relative_perf_scores[1])
    assert np.isnan(reports[0].gpu_relative_perf_scores[2])
    assert np.isnan(reports[0].gpu_relative_perf_scores[3])


def test_relative_gpu_scores_ranks_without_kernels():

    # Check that (all) scores are NaN when there are no common kernels for all ranks.
    # TODO: should we compute the results for a subset of ranks with common kernels?

    mp_ctx = mp.get_context("spawn")
    ret_queue = mp_ctx.Queue()

    rank_processes = multiprocessing_execute_start(
        worker_fn=_rank_main_gpu_relative_test_no_common_kernels_in_rank,
        mp_ctx=mp_ctx,
        world_size=4,
        backend="gloo",
        dist_store_type="file",
        ranks_without_kernels=(
            1,
            3,
        ),
        gather_on_rank0=True,
        ret_queue=ret_queue,
    )

    reports = {}
    for _ in range(4):
        rank, report = ret_queue.get(timeout=RANK_DONE_TIMEOUT)
        reports[rank] = report

    ret_codes = multiprocessing_execute_join(rank_processes, timeout=RANK_DONE_TIMEOUT)
    assert all(ret_code == 0 for ret_code in ret_codes)

    assert len(reports) == 4
    # results for all ranks should be included in the report generated on rank0
    assert len(reports[0].gpu_relative_perf_scores) == 4
    assert np.isnan(reports[0].gpu_relative_perf_scores[0])
    assert np.isnan(reports[0].gpu_relative_perf_scores[1])
    assert np.isnan(reports[0].gpu_relative_perf_scores[2])
    assert np.isnan(reports[0].gpu_relative_perf_scores[3])
