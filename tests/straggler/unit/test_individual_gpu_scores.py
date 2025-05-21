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
# Tests of `straggler.reporting.ReportGenerator` individual GPU scores computation
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


def test_individual_gpu_scores_one_rank():
    scores_to_compute = ['individual_perf_scores']
    report_gen = straggler.reporting.ReportGenerator(
        scores_to_compute, gather_on_rank0=False, node_name='testnode'
    )

    # a few basic cases
    kernel_summaries = {
        'kernel0': _get_summary(np.array([1.0, 1.0, 2.0])),
        'kernel1': _get_summary(np.array([2.0, 2.0, 3.0])),
    }
    report = report_gen.generate_report({}, kernel_summaries=kernel_summaries)
    assert report.gpu_individual_perf_scores[0] == pytest.approx(1.0)

    kernel_summaries = {
        'kernel0': _get_summary(1.25 * np.array([1.0, 1.0, 2.0])),
        'kernel1': _get_summary(1.25 * np.array([2.0, 2.0, 3.0])),
    }
    report = report_gen.generate_report({}, kernel_summaries=kernel_summaries)
    assert report.gpu_individual_perf_scores[0] == pytest.approx(1 / 1.25)

    kernel_summaries = {
        'kernel0': _get_summary(2.0 * np.array([1.0, 1.0, 2.0])),
        'kernel1': _get_summary(2.0 * np.array([2.0, 2.0, 3.0])),
    }
    report = report_gen.generate_report({}, kernel_summaries=kernel_summaries)
    assert report.gpu_individual_perf_scores[0] == pytest.approx(0.5)

    kernel_summaries = {
        'kernel0': _get_summary(3.0 * np.array([1.0, 1.0, 2.0])),
        'kernel1': _get_summary(3.0 * np.array([2.0, 2.0, 3.0])),
    }
    report = report_gen.generate_report({}, kernel_summaries=kernel_summaries)
    assert report.gpu_individual_perf_scores[0] == pytest.approx(0.333, abs=0.001)

    # missing kernel1
    kernel_summaries = {
        'kernel0': _get_summary(4.0 * np.array([1.0, 1.0, 2.0])),
    }
    report = report_gen.generate_report({}, kernel_summaries=kernel_summaries)
    assert report.gpu_individual_perf_scores[0] == pytest.approx(0.25)

    # missing kernel0
    kernel_summaries = {
        'kernel1': _get_summary(5.0 * np.array([2.0, 2.0, 3.0])),
    }
    report = report_gen.generate_report({}, kernel_summaries=kernel_summaries)
    assert report.gpu_individual_perf_scores[0] == pytest.approx(0.2)

    # no kernels - should return NaN
    kernel_summaries = {}
    report = report_gen.generate_report({}, kernel_summaries=kernel_summaries)
    assert np.isnan(report.gpu_individual_perf_scores[0])

    # suddenly new kernels appear!
    kernel_summaries = {
        'new_kernel': _get_summary(np.array([1.0, 1.0, 2.0])),
        'another_new_kernel': _get_summary(np.array([2.0, 2.0, 3.0])),
    }
    report = report_gen.generate_report({}, kernel_summaries=kernel_summaries)
    assert report.gpu_individual_perf_scores[0] == pytest.approx(1.0)

    # should update the reference with the new minimum
    kernel_summaries = {
        'kernel0': _get_summary(0.5 * np.array([1.0, 1.0, 2.0])),
        'kernel1': _get_summary(0.5 * np.array([2.0, 2.0, 3.0])),
    }
    report = report_gen.generate_report({}, kernel_summaries=kernel_summaries)
    assert report.gpu_individual_perf_scores[0] == pytest.approx(1.0)

    kernel_summaries = {
        'kernel0': _get_summary(np.array([1.0, 1.0, 2.0])),
        'kernel1': _get_summary(np.array([2.0, 2.0, 3.0])),
    }
    report = report_gen.generate_report({}, kernel_summaries=kernel_summaries)
    assert report.gpu_individual_perf_scores[0] == pytest.approx(0.5)


def _rank_main_gpu_indiv_test(*args, gather_on_rank0, ret_queue, **kwargs):

    rank = torch.distributed.get_rank()
    random.seed(rank)

    scores_to_compute = ['individual_perf_scores']
    report_gen = straggler.reporting.ReportGenerator(
        scores_to_compute,
        gather_on_rank0=gather_on_rank0,
        node_name=f'testnode{rank}',
    )

    # initial values just for a reference
    kernel_summaries = {
        'kernel0': _get_summary(np.array([1.0, 1.0, 2.0])),
        'kernel1': _get_summary(np.array([2.0, 2.0, 3.0])),
    }
    report = report_gen.generate_report({}, kernel_summaries=kernel_summaries)

    # multiply base timing by rank+1, just to obtain different scores on different ranks
    kernel_summaries = {
        'kernel0': _get_summary((rank + 1) * np.array([1.0, 1.0, 2.0])),
        'kernel1': _get_summary((rank + 1) * np.array([2.0, 2.0, 3.0])),
    }
    report = report_gen.generate_report({}, kernel_summaries=kernel_summaries)
    ret_queue.put((rank, report))


def test_individual_gpu_scores_gather_on_rank0():

    # Check with gather_on_rank0=True it gathers all results on rank0

    mp_ctx = mp.get_context("spawn")
    ret_queue = mp_ctx.Queue()

    rank_processes = multiprocessing_execute_start(
        worker_fn=_rank_main_gpu_indiv_test,
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
    assert reports[0].gpu_individual_perf_scores[0] == pytest.approx(1.0)
    assert reports[0].gpu_individual_perf_scores[1] == pytest.approx(1.0 / 2.0)
    assert reports[0].gpu_individual_perf_scores[2] == pytest.approx(1.0 / 3.0)
    assert reports[0].gpu_individual_perf_scores[3] == pytest.approx(1.0 / 4.0)
    # reports from other ranks should be None or empty
    assert not reports[1]
    assert not reports[2]
    assert not reports[3]
    # check rank to node mapping in the report
    assert reports[0].rank_to_node[0] == 'testnode0'
    assert reports[0].rank_to_node[1] == 'testnode1'
    assert reports[0].rank_to_node[2] == 'testnode2'
    assert reports[0].rank_to_node[3] == 'testnode3'
    # there should be no relative scores
    assert not reports[0].gpu_relative_perf_scores


def test_individual_gpu_scores_no_gather():

    # Check with gather_on_rank0=False each rank has its own results only

    mp_ctx = mp.get_context("spawn")
    ret_queue = mp_ctx.Queue()

    rank_processes = multiprocessing_execute_start(
        worker_fn=_rank_main_gpu_indiv_test,
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
    assert len(reports[0].gpu_individual_perf_scores) == 1
    assert reports[0].gpu_individual_perf_scores[0] == pytest.approx(1.0)
    assert len(reports[1].gpu_individual_perf_scores) == 1
    assert reports[1].gpu_individual_perf_scores[1] == pytest.approx(1.0 / 2.0)
    assert len(reports[2].gpu_individual_perf_scores) == 1
    assert reports[2].gpu_individual_perf_scores[2] == pytest.approx(1.0 / 3.0)
    assert len(reports[3].gpu_individual_perf_scores) == 1
    assert reports[3].gpu_individual_perf_scores[3] == pytest.approx(1.0 / 4.0)
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
    # there should be no relative scores
    assert not reports[0].gpu_relative_perf_scores
