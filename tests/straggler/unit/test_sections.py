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
import time

import pytest
import torch
import torch.multiprocessing as mp

from nvidia_resiliency_ext.attribution import straggler

from ._utils import multiprocessing_execute_join, multiprocessing_execute_start

# This is a basic test of custom sections:
# - Run distributed dummy workload, where each rank has 3 custom sections
# - Each rank executes the sections in a loop
# - Each section simulates some work with `sleep`
# - Depending of the test parameters, some sections in some ranks can be stragglers (longer sleep)
# - After a few iters, get the report, verify that stragglers are detected correctly

TEST_WORLD_SIZE = 4
ALL_RANK_IDS = set(range(TEST_WORLD_SIZE))
RANK_DONE_TIMEOUT = 30


def _dummy_section_work(section_name, rank, test_scenario, is_slow_iter=False):
    is_straggler = ('stragglers' in test_scenario.keys()) and (
        (section_name, rank) in test_scenario['stragglers']
    )
    is_indiv_straggler = ('indiv_stragglers' in test_scenario.keys()) and (
        (section_name, rank) in test_scenario['indiv_stragglers']
    )
    if is_straggler:
        mu = test_scenario['avg_section_time_straggler']
        sigma = test_scenario['stdev_section_time_straggler']
    else:
        mu = test_scenario['avg_section_time']
        sigma = test_scenario['stdev_section_time']

    if is_indiv_straggler and is_slow_iter:
        mu = test_scenario['avg_section_time_straggler']
        sigma = test_scenario['stdev_section_time_straggler']
    sleep_time = max(0.0, random.gauss(mu=mu, sigma=sigma))
    time.sleep(sleep_time)


def _rank_main(*args, test_scenario, ret_queue, **kwargs):

    rank = torch.distributed.get_rank()
    random.seed(rank)

    training_half = test_scenario['iters'] // 2

    straggler.Detector.initialize(node_name='dummy_node_name')

    for i in range(test_scenario['iters']):
        for section in ["section00", "section01", "section02"]:
            with straggler.Detector.detection_section(section):
                _dummy_section_work(
                    section,
                    rank,
                    test_scenario,
                    is_slow_iter=(i > training_half),
                )

        if i == training_half:
            report = straggler.Detector.generate_report()

    report = straggler.Detector.generate_report()

    if rank == 0:
        ret_queue.put(report)

    straggler.Detector.shutdown()


test_scenarios = [
    {
        "stragglers": [("section00", 0)],
        "avg_section_time": 0.01,
        "stdev_section_time": 0.003,
        "avg_section_time_straggler": 0.015,
        "stdev_section_time_straggler": 0.003,
        "iters": 100,
    },
    {
        "stragglers": [("section00", 0), ("section00", 1)],
        "avg_section_time": 0.01,
        "stdev_section_time": 0.003,
        "avg_section_time_straggler": 0.015,
        "stdev_section_time_straggler": 0.003,
        "iters": 100,
    },
    {
        "stragglers": [("section00", 0), ("section01", 1)],
        "avg_section_time": 0.01,
        "stdev_section_time": 0.003,
        "avg_section_time_straggler": 0.015,
        "stdev_section_time_straggler": 0.003,
        "iters": 100,
    },
    {
        "stragglers": [],
        "avg_section_time": 0.01,
        "stdev_section_time": 0.003,
        "iters": 100,
    },
    {
        "indiv_stragglers": [("section00", 0)],
        "avg_section_time": 0.01,
        "stdev_section_time": 0.003,
        "avg_section_time_straggler": 0.015,
        "stdev_section_time_straggler": 0.003,
        "iters": 100,
    },
    {
        "indiv_stragglers": [("section00", 0), ("section00", 1)],
        "avg_section_time": 0.01,
        "stdev_section_time": 0.003,
        "avg_section_time_straggler": 0.015,
        "stdev_section_time_straggler": 0.003,
        "iters": 100,
    },
    {
        "indiv_stragglers": [("section00", 0), ("section01", 1)],
        "avg_section_time": 0.01,
        "stdev_section_time": 0.003,
        "avg_section_time_straggler": 0.015,
        "stdev_section_time_straggler": 0.003,
        "iters": 100,
    },
    {
        "indiv_stragglers": [],
        "avg_section_time": 0.01,
        "stdev_section_time": 0.003,
        "iters": 100,
    },
]


@pytest.mark.parametrize("test_scenario", test_scenarios)
def test_straggler_sections_detected(test_scenario):

    mp_ctx = mp.get_context("spawn")
    ret_queue = mp_ctx.Queue()

    rank_processes = multiprocessing_execute_start(
        worker_fn=_rank_main,
        mp_ctx=mp_ctx,
        world_size=TEST_WORLD_SIZE,
        backend="gloo",
        dist_store_type="file",
        test_scenario=test_scenario,
        ret_queue=ret_queue,
    )

    report = ret_queue.get(timeout=RANK_DONE_TIMEOUT)

    ret_codes = multiprocessing_execute_join(rank_processes, timeout=RANK_DONE_TIMEOUT)
    assert all(ret_code == 0 for ret_code in ret_codes)

    found_stragglers = report.identify_stragglers()

    assert not found_stragglers['straggler_gpus_relative']
    assert not found_stragglers['straggler_gpus_individual']

    if 'stragglers' in test_scenario.keys():
        if test_scenario['stragglers']:
            # verify that all stragglers are detected
            for test_straggler_section, test_straggler_rank in test_scenario['stragglers']:
                assert test_straggler_section in found_stragglers['straggler_sections_relative']
                assert (
                    straggler.StragglerId(test_straggler_rank, 'dummy_node_name')
                    in found_stragglers['straggler_sections_relative'][test_straggler_section]
                )
        else:
            # there should be no stragglers detected if there were no stragglers in the test
            assert not found_stragglers['straggler_sections_relative']

    if 'indiv_stragglers' in test_scenario.keys():
        if test_scenario['indiv_stragglers']:
            # verify that all stragglers are detected
            for test_straggler_section, test_straggler_rank in test_scenario['indiv_stragglers']:
                assert test_straggler_section in found_stragglers['straggler_sections_individual']
                assert (
                    straggler.StragglerId(test_straggler_rank, 'dummy_node_name')
                    in found_stragglers['straggler_sections_individual'][test_straggler_section]
                )
        else:
            # there should be no stragglers detected if there were no stragglers in the test
            assert not found_stragglers['straggler_sections_individual']
