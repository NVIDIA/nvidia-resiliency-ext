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

import faulthandler
import shutil
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist

from nvidia_resiliency_ext.fault_tolerance.timeouts_calc import TimeoutsCalc, TimeoutsCalcError

from .utils import multiprocessing_execute_join, multiprocessing_execute_start

TEST_WORLD_SIZE = 4
WORKLOAD_TIMEOUT = 180


@pytest.fixture
def tmp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_basic():
    tc = TimeoutsCalc(start_time=0, safety_factor=2.0)
    assert tc.can_get_timeouts() is False
    with pytest.raises(TimeoutsCalcError):
        tc.get_timeouts()
    tc.update(hb_time=10)
    tc.update(hb_time=11)
    assert tc.can_get_timeouts()
    tc.update(hb_time=18)
    tc.update(hb_time=20)
    assert tc.get_timeouts() == (20, 14)


def _rank_main(*args, tmp_dir, **kwargs):
    faulthandler.enable(file=sys.stderr)
    tc = TimeoutsCalc(start_time=0, safety_factor=2.0)
    rank = dist.get_rank()
    if rank in [1, 2]:
        # update calculators in ranks 1,2
        # instances in other ranks are not initialized
        tc.update(hb_time=10 * rank)
        # rank 2 will get largest times:
        # 20 for initial heartbeat, 2 for subsequent
        tc.update(hb_time=10 * rank + rank)
    tc.synchronize_all()
    # after synchronization, all ranks should have the same values
    assert tc.get_timeouts() == (40, 4)
    sys.exit(0)


def test_distributed(tmp_dir):
    mp_ctx = torch.multiprocessing.get_context("spawn")

    local_tc = TimeoutsCalc()
    with pytest.raises(TimeoutsCalcError):
        # no process group initialized
        local_tc.synchronize_all()

    rank_processes = multiprocessing_execute_start(
        worker_fn=_rank_main,
        world_size=TEST_WORLD_SIZE,
        mp_ctx=mp_ctx,
        backend="gloo",
        dist_store_type="file",
        tmp_dir=tmp_dir,
    )

    ret_codes = multiprocessing_execute_join(rank_processes, timeout=WORKLOAD_TIMEOUT)

    assert ret_codes == [0] * TEST_WORLD_SIZE
