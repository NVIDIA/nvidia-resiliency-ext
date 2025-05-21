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

import math
import random

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from nvidia_resiliency_ext import attribution.straggler as straggler

from ._utils import multiprocessing_execute_join, multiprocessing_execute_start

RANK_DONE_TIMEOUT = 30


#
# Tests of `straggler.name_mapper.NameMapper` class and usage
#


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


def test_name_mapper_gather():
    report_gen = straggler.reporting.ReportGenerator(
        scores_to_compute=['relative_perf_scores'], gather_on_rank0=True
    )

    kernel_summaries = {
        'kernel0': _get_summary(np.array([1.0, 1.0, 2.0])),
        'kernel1': _get_summary(np.array([2.0, 3.0, 3.0])),
    }

    section_summaries = {
        'section0': _get_summary(np.array([1.0, 1.0, 2.0])),
    }

    # gather and assign of ids
    report_gen.generate_report(
        kernel_summaries=kernel_summaries, section_summaries=section_summaries
    )

    assert report_gen.name_mapper.kernel_counter == 2
    assert report_gen.name_mapper.get_kernel_name(0) == 'kernel0'
    assert report_gen.name_mapper.get_kernel_id('kernel0') == 0
    assert report_gen.name_mapper.get_kernel_name(1) == 'kernel1'
    assert report_gen.name_mapper.get_kernel_id('kernel1') == 1

    assert report_gen.name_mapper.section_counter == 1
    assert report_gen.name_mapper.get_section_name(0) == 'section0'
    assert report_gen.name_mapper.get_section_id('section0') == 0

    kernel_summaries = {
        'kernel2': _get_summary(np.array([1.0, 2.0, 4.0])),
    }
    section_summaries = {
        'section0': _get_summary(np.array([1.0, 1.0, 2.0])),
        'section1': _get_summary(np.array([1.0, 2.0, 4.0])),
        'section2': _get_summary(np.array([3.0, 4.0, 4.0])),
    }

    report_gen.generate_report(
        kernel_summaries=kernel_summaries, section_summaries=section_summaries
    )

    # new names get new ids, old keep their ids
    assert report_gen.name_mapper.kernel_counter == 3
    assert report_gen.name_mapper.get_kernel_id('kernel0') == 0
    assert report_gen.name_mapper.get_kernel_id('kernel1') == 1
    assert report_gen.name_mapper.get_kernel_id('kernel2') == 2
    assert report_gen.name_mapper.section_counter == 3
    assert report_gen.name_mapper.get_section_id('section0') == 0
    assert report_gen.name_mapper.get_section_id('section1') == 1
    assert report_gen.name_mapper.get_section_id('section2') == 2


def test_name_mapper_no_gather():
    report_gen = straggler.reporting.ReportGenerator(
        scores_to_compute=['individual_perf_scores'], gather_on_rank0=False
    )

    kernel_summaries = {
        'kernel0': _get_summary(np.array([1.0, 1.0, 2.0])),
    }

    section_summaries = {
        'section0': _get_summary(np.array([1.0, 1.0, 2.0])),
    }

    report_gen.generate_report(
        kernel_summaries=kernel_summaries, section_summaries=section_summaries
    )

    # mapper should not be initialized, when its init params were:
    # scores_to_compute=['individual_perf_scores'], gather_on_rank0=False
    with pytest.raises(KeyError):
        assert report_gen.name_mapper.get_kernel_id('kernel0')
    with pytest.raises(KeyError):
        assert report_gen.name_mapper.get_section_id('section0')


def _test_mapping_consitiency_no_gather_on_rank0(ret_queue):

    rank = torch.distributed.get_rank()
    random.seed(rank)

    scores_to_compute = ['relative_perf_scores']
    report_gen = straggler.reporting.ReportGenerator(
        scores_to_compute,
        gather_on_rank0=False,
        node_name=f'testnode{rank}',
    )

    # feed the report generator with some initial data,
    # without any common kernels or sections
    kernel_summaries = {f'initial_kernel_rank{rank}': _get_summary([1.0])}
    section_summaries = {f'initial_section_rank{rank}': _get_summary([1.0])}

    _ = report_gen.generate_report(
        section_summaries=section_summaries, kernel_summaries=kernel_summaries
    )

    # now, the main test part:
    # rank0 has its own kernel and section, rank1 also has its own kernel and section
    # common kernels (k1,k2) and sections (s1,s2) are in different order on ranks.
    # simulate that common kernels and sections on rank0 "took" 2x more time
    if rank == 0:
        kernel_summaries = {
            'rank0_only': _get_summary([1.0]),
            'k1': _get_summary([4.0]),
            'k2': _get_summary([6.0]),
        }
        section_summaries = {
            'rank0_only': _get_summary([1.0]),
            's1': _get_summary([4.0]),
            's2': _get_summary([6.0]),
        }
    elif rank == 1:
        kernel_summaries = {
            'k2': _get_summary([3.0]),
            'k1': _get_summary([2.0]),
            'rank1_only': _get_summary([1.0]),
        }
        section_summaries = {
            's2': _get_summary([3.0]),
            's1': _get_summary([2.0]),
            'rank1_only': _get_summary([1.0]),
        }
    else:
        assert "Only 2 ranks are expected for this test"

    report = report_gen.generate_report(
        section_summaries=section_summaries, kernel_summaries=kernel_summaries
    )

    # scores sanity check, gather_on_rank0=False, so each rank has its own scores only
    assert set(report.gpu_relative_perf_scores.keys()) == {rank}
    assert set(report.section_relative_perf_scores['s1'].keys()) == {rank}
    assert set(report.section_relative_perf_scores['s2'].keys()) == {rank}

    if rank == 0:
        assert report.gpu_relative_perf_scores[0] == 0.5
        assert set(report.section_relative_perf_scores.keys()) == {
            's1',
            's2',
            'rank0_only',
        }
        assert report.section_relative_perf_scores['s1'][0] == 0.5
        assert report.section_relative_perf_scores['s2'][0] == 0.5
        assert math.isnan(report.section_relative_perf_scores['rank0_only'][0])
    elif rank == 1:
        assert report.gpu_relative_perf_scores[1] == 1.0
        assert set(report.section_relative_perf_scores.keys()) == {
            's1',
            's2',
            'rank1_only',
        }
        assert report.section_relative_perf_scores['s1'][1] == 1.0
        assert report.section_relative_perf_scores['s2'][1] == 1.0
        assert math.isnan(report.section_relative_perf_scores['rank1_only'][1])

    ret_queue.put(report_gen.name_mapper)


def test_mapping_consitiency_no_gather_on_rank0():

    # Goal of this test is to check if name mapping is consistent between ranks

    mp_ctx = mp.get_context("spawn")
    ret_queue = mp_ctx.Queue()

    rank_processes = multiprocessing_execute_start(
        worker_fn=_test_mapping_consitiency_no_gather_on_rank0,
        mp_ctx=mp_ctx,
        world_size=2,
        backend="gloo",
        dist_store_type="file",
        ret_queue=ret_queue,
    )

    mapper1 = ret_queue.get(timeout=RANK_DONE_TIMEOUT)
    mapper2 = ret_queue.get(timeout=RANK_DONE_TIMEOUT)

    ret_codes = multiprocessing_execute_join(rank_processes, timeout=RANK_DONE_TIMEOUT)
    assert all(ret_code == 0 for ret_code in ret_codes)

    # there were 6 different kernel and section names passed to the report generator
    assert mapper1.kernel_counter == 6
    assert mapper1.section_counter == 6
    # check if both ranks ended up with the same mapping
    assert mapper1.kernel_name_to_id == mapper2.kernel_name_to_id
    assert mapper1.section_name_to_id == mapper2.section_name_to_id


def _test_mapping_consitiency_with_gather_on_rank0(ret_queue):

    rank = torch.distributed.get_rank()
    random.seed(rank)

    scores_to_compute = ['relative_perf_scores']
    report_gen = straggler.reporting.ReportGenerator(
        scores_to_compute,
        gather_on_rank0=True,
        node_name=f'testnode{rank}',
    )

    # feed the report generator with some initial data,
    # without any common kernels or sections
    kernel_summaries = {f'initial_kernel_rank{rank}': _get_summary([1.0])}
    section_summaries = {f'initial_section_rank{rank}': _get_summary([1.0])}

    _ = report_gen.generate_report(
        section_summaries=section_summaries, kernel_summaries=kernel_summaries
    )

    # now, the main test part:
    # rank0 has its own kernel and section, rank1 also has its own kernel and section
    # common kernels (k1,k2) and sections (s1,s2) are in different order on ranks.
    # simulate that common kernels and sections on rank0 "took" 2x more time
    if rank == 0:
        kernel_summaries = {
            'rank0_only': _get_summary([1.0]),
            'k1': _get_summary([4.0]),
            'k2': _get_summary([6.0]),
        }
        section_summaries = {
            'rank0_only': _get_summary([1.0]),
            's1': _get_summary([4.0]),
            's2': _get_summary([6.0]),
        }
    elif rank == 1:
        kernel_summaries = {
            'k2': _get_summary([3.0]),
            'k1': _get_summary([2.0]),
            'rank1_only': _get_summary([1.0]),
        }
        section_summaries = {
            's2': _get_summary([3.0]),
            's1': _get_summary([2.0]),
            'rank1_only': _get_summary([1.0]),
        }
    else:
        assert "Only 2 ranks are expected for this test"

    report = report_gen.generate_report(
        section_summaries=section_summaries, kernel_summaries=kernel_summaries
    )

    # scores sanity check, gather_on_rank0=True, so rank0 has all scores
    if rank == 0:
        # check GPU scores
        assert set(report.gpu_relative_perf_scores.keys()) == {0, 1}
        assert report.gpu_relative_perf_scores[0] == 0.5
        assert report.gpu_relative_perf_scores[1] == 1.0
        # check sections scores
        assert set(report.section_relative_perf_scores.keys()) == {
            's1',
            's2',
            'initial_section_rank0',
            'initial_section_rank1',
            'rank0_only',
            'rank1_only',
        }
        assert set(report.section_relative_perf_scores['s1'].keys()) == {0, 1}
        assert list(report.section_relative_perf_scores['s1'].values()) == [
            0.5,
            1.0,
        ]
        assert set(report.section_relative_perf_scores['s2'].keys()) == {0, 1}
        assert list(report.section_relative_perf_scores['s2'].values()) == [
            0.5,
            1.0,
        ]
        assert set(report.section_relative_perf_scores['initial_section_rank0'].keys()) == {0, 1}
        assert all(
            math.isnan(v)
            for v in report.section_relative_perf_scores['initial_section_rank0'].values()
        )
        assert set(report.section_relative_perf_scores['initial_section_rank1'].keys()) == {0, 1}
        assert all(
            math.isnan(v)
            for v in report.section_relative_perf_scores['initial_section_rank1'].values()
        )
        assert set(report.section_relative_perf_scores['rank0_only'].keys()) == {0, 1}
        assert all(
            math.isnan(v) for v in report.section_relative_perf_scores['rank0_only'].values()
        )
        assert set(report.section_relative_perf_scores['rank1_only'].keys()) == {0, 1}
        assert all(
            math.isnan(v) for v in report.section_relative_perf_scores['rank1_only'].values()
        )
    else:
        assert not report

    ret_queue.put(report_gen.name_mapper)


def test_mapping_consitiency_with_gather_on_rank0():

    # Goal of this test is to check if name mapping is consistent between ranks

    mp_ctx = mp.get_context("spawn")
    ret_queue = mp_ctx.Queue()

    rank_processes = multiprocessing_execute_start(
        worker_fn=_test_mapping_consitiency_with_gather_on_rank0,
        mp_ctx=mp_ctx,
        world_size=2,
        backend="gloo",
        dist_store_type="file",
        ret_queue=ret_queue,
    )

    mapper1 = ret_queue.get(timeout=RANK_DONE_TIMEOUT)
    mapper2 = ret_queue.get(timeout=RANK_DONE_TIMEOUT)

    ret_codes = multiprocessing_execute_join(rank_processes, timeout=RANK_DONE_TIMEOUT)
    assert all(ret_code == 0 for ret_code in ret_codes)

    # there were 6 different kernel and section names passed to the report generator
    assert mapper1.kernel_counter == 6
    assert mapper1.section_counter == 6
    # check if both ranks ended up with the same mapping
    assert mapper1.kernel_name_to_id == mapper2.kernel_name_to_id
    assert mapper1.section_name_to_id == mapper2.section_name_to_id
