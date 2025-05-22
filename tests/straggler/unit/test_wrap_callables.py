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

TEST_WORLD_SIZE = 4
ALL_RANK_IDS = set(range(TEST_WORLD_SIZE))
RANK_DONE_TIMEOUT = 30


class CallableModule:
    def callable1(self, test_scenario=None):
        pass

    @staticmethod
    def callable2(test_scenario=None):
        pass

    @classmethod
    def callable3(cls, test_scenario=None):
        sleep_time = test_scenario['sleep_time']
        time.sleep(sleep_time)

    def callable4(self, test_scenario=None):
        pass


def _rank_main(*args, test_scenario, ret_queue, **kwargs):

    rank = torch.distributed.get_rank()
    random.seed(rank)

    callable_module_instance = CallableModule()

    straggler.Detector.initialize(gather_on_rank0=False)
    straggler.Detector.wrap_callables(
        callable_ids=[
            straggler.CallableId(callable_module_instance, "callable1"),
            straggler.CallableId(CallableModule, "callable2"),
            straggler.CallableId(CallableModule, "callable3"),
            straggler.CallableId(CallableModule, "callable4", ignored_args=("self",)),
        ]
    )

    for i in range(test_scenario['iters']):
        callable_module_instance.callable1()
        CallableModule.callable2()
        CallableModule.callable3(test_scenario)
        callable_module_instance.callable4()

    # test that after restore_original_callables call callables are not recorded
    straggler.Detector.restore_original_callables()

    for i in range(test_scenario['unrecorded_iters']):
        callable_module_instance.callable1()
        CallableModule.callable2()
        CallableModule.callable3(test_scenario)
        callable_module_instance.callable4()

    report = straggler.Detector.generate_report()
    int_to_section_mapper = straggler.Detector.reporter.name_mapper.id_to_section_name.copy()

    ret_queue.put((rank, report, int_to_section_mapper))

    straggler.Detector.shutdown()


test_scenarios = [
    {
        "sleep_time": 0.01,
        "iters": 10,
        "unrecorded_iters": 0,
    },
    {
        "sleep_time": 0.01,
        "iters": 100,
        "unrecorded_iters": 10,
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

    reports = {}
    for _ in ALL_RANK_IDS:
        rank, report, int_to_section_mapper = ret_queue.get(timeout=RANK_DONE_TIMEOUT)
        reports[rank] = report

    ret_codes = multiprocessing_execute_join(rank_processes, timeout=RANK_DONE_TIMEOUT)
    assert all(ret_code == 0 for ret_code in ret_codes)

    expected_callables = {
        "CallableModule.callable1",
        f"{CallableModule.__module__}.CallableModule.callable2",
        f"{CallableModule.__module__}.CallableModule.callable3",
        f"{CallableModule.__module__}.CallableModule.callable4",
    }

    for rank_id in ALL_RANK_IDS:
        local_sections = reports[rank_id].local_section_summaries
        assert expected_callables == local_sections.keys()
        for section_name, stats in local_sections.items():
            assert stats[straggler.Statistic.NUM] == test_scenario["iters"]
