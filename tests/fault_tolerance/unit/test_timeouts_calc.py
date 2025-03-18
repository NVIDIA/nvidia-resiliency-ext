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

import dataclasses
import shutil
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist

from nvidia_resiliency_ext.fault_tolerance.data import (
    HeartbeatTimeouts,
    SectionAction,
    SectionTimeouts,
)
from nvidia_resiliency_ext.fault_tolerance.timeouts_calc import TimeoutsCalc, TimeoutsCalcError

from .utils import multiprocessing_execute_join, multiprocessing_execute_start

TEST_WORLD_SIZE = 4
WORKLOAD_TIMEOUT = 60


@pytest.fixture
def tmp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_with_heartbeats():
    tc = TimeoutsCalc(start_time=0, safety_factor=2.0)
    assert tc.can_get_hb_timeouts() is False
    with pytest.raises(TimeoutsCalcError):
        tc.get_hb_timeouts()
    tc.update_on_heartbeat(hb_time=10)
    tc.update_on_heartbeat(hb_time=11)
    assert tc.can_get_hb_timeouts()
    tc.update_on_heartbeat(hb_time=18)
    tc.update_on_heartbeat(hb_time=20)
    assert dataclasses.astuple(tc.get_hb_timeouts()) == (20, 14, True)
    # check merging with not-calculated timeouts - should use the calculated ones
    other = HeartbeatTimeouts(99, 99, were_calculated=False)
    assert dataclasses.astuple(tc.get_hb_timeouts(current=other)) == (20, 14, True)
    # check merging with calculated timeouts, should compute EMA(old,new,alpha=0.5)
    other = HeartbeatTimeouts(21, 15, were_calculated=True)
    assert dataclasses.astuple(tc.get_hb_timeouts(current=other)) == (20.5, 14.5, True)
    # check merging with invalid timeouts - should use the calculated ones
    other = HeartbeatTimeouts(None, None, were_calculated=False)
    assert other.are_valid is False
    assert dataclasses.astuple(tc.get_hb_timeouts(current=other)) == (20, 14, True)
    # check merging with invalid and calculated - that should not happen, so error is expected
    with pytest.raises(Exception):
        other = HeartbeatTimeouts(None, None, were_calculated=True)
        tc.get_hb_timeouts(current=other)


def test_with_custom_sections():
    valid_sections = ['one', 'two']
    tc = TimeoutsCalc(sections=valid_sections, start_time=5, safety_factor=1.0)
    assert tc.can_get_section_timeouts() is False
    with pytest.raises(TimeoutsCalcError):
        tc.get_section_timeouts()
    with pytest.raises(TimeoutsCalcError):
        tc.update_on_section_event(
            section='unknown_section', action=SectionAction.OPEN, event_time=1
        )
    # to be able to compute timeouts,
    # we need to see heartbeats from all defined sections
    # implements following scenario:
    # (init)       (time=5) (implicit section is opened)
    #  ->"one"     (time=10) (implicit section is closed)
    #    ->"two"   (time=13)
    #    <-"two"   (time=23)
    #  <-"one"     (time=40) (implicit section is opened)
    #  ->"one"     (time=100) (implicit section is closed)
    #  ->"one"     (time=120)
    tc.update_on_section_event(section='one', action=SectionAction.OPEN, event_time=10)
    tc.update_on_section_event(section='two', action=SectionAction.OPEN, event_time=13)
    with pytest.raises(TimeoutsCalcError):
        # cant open the same section twice
        tc.update_on_section_event(section='one', action=SectionAction.OPEN, event_time=14)
    with pytest.raises(TimeoutsCalcError):
        # cant open the same section twice
        tc.update_on_section_event(section='two', action=SectionAction.OPEN, event_time=14)
    tc.update_on_section_event(section='two', action=SectionAction.CLOSE, event_time=23)
    # Still not enough data to compute the timeouts
    tc.update_on_section_event(section='one', action=SectionAction.CLOSE, event_time=40)
    with pytest.raises(TimeoutsCalcError):
        # cant close if already closed
        tc.update_on_section_event(section='one', action=SectionAction.CLOSE, event_time=111)
    with pytest.raises(TimeoutsCalcError):
        # cant close if already closed
        tc.update_on_section_event(section='two', action=SectionAction.CLOSE, event_time=111)
    # Now has enough data to compute the timeouts
    assert tc.can_get_section_timeouts() is True
    computed = tc.get_section_timeouts()
    assert computed.section == {'one': 30, 'two': 10}
    assert computed.out_of_section == 5
    assert computed.calculated_sections == {'one', 'two'}
    assert computed.is_out_of_section_calculated is True
    assert computed.were_calculated is True
    # implicit section is open at this moment, as all other sections are closed
    tc.update_on_section_event(section='one', action=SectionAction.OPEN, event_time=100)
    tc.update_on_section_event(section='one', action=SectionAction.CLOSE, event_time=120)
    assert tc.can_get_section_timeouts() is True
    computed = tc.get_section_timeouts()
    assert computed.section == {'one': 30, 'two': 10}
    assert computed.out_of_section == 60
    assert computed.calculated_sections == {'one', 'two'}
    assert computed.is_out_of_section_calculated is True
    assert computed.were_calculated is True
    # try to bump out-of-section time
    tc.maybe_bump_oos_time(curr_time=220)
    computed = tc.get_section_timeouts()
    assert computed.section == {'one': 30, 'two': 10}
    assert computed.out_of_section == 100
    assert computed.calculated_sections == {'one', 'two'}
    assert computed.is_out_of_section_calculated is True
    assert computed.were_calculated is True


def test_with_sections_subset():
    valid_sections = ['one', 'two']
    tc = TimeoutsCalc(sections=valid_sections, start_time=1, safety_factor=1.0)

    # check API for computing timeouts for selected sections only

    assert tc.can_get_section_timeouts() is False
    assert tc.can_get_section_timeouts(selected_sections=['one'], calc_out_of_section=True) is False
    assert (
        tc.can_get_section_timeouts(selected_sections=['one'], calc_out_of_section=False) is False
    )
    assert tc.can_get_section_timeouts(selected_sections=['two'], calc_out_of_section=True) is False
    assert (
        tc.can_get_section_timeouts(selected_sections=['two'], calc_out_of_section=False) is False
    )

    tc.update_on_section_event(section='one', action=SectionAction.OPEN, event_time=10)
    tc.update_on_section_event(section='one', action=SectionAction.CLOSE, event_time=20)
    assert tc.can_get_section_timeouts() is False
    assert tc.can_get_section_timeouts(selected_sections=['one'], calc_out_of_section=True) is True
    assert tc.can_get_section_timeouts(selected_sections=['one'], calc_out_of_section=False) is True
    assert tc.can_get_section_timeouts(selected_sections=['two'], calc_out_of_section=True) is False
    assert (
        tc.can_get_section_timeouts(selected_sections=['two'], calc_out_of_section=False) is False
    )

    tc.update_on_section_event(section='two', action=SectionAction.OPEN, event_time=25)
    tc.update_on_section_event(section='two', action=SectionAction.CLOSE, event_time=36)
    assert tc.can_get_section_timeouts() is True  # now it has all timeouts
    assert tc.can_get_section_timeouts(selected_sections=['one'], calc_out_of_section=True) is True
    assert tc.can_get_section_timeouts(selected_sections=['one'], calc_out_of_section=False) is True
    assert tc.can_get_section_timeouts(selected_sections=['two'], calc_out_of_section=True) is True
    assert tc.can_get_section_timeouts(selected_sections=['two'], calc_out_of_section=False) is True

    computed = tc.get_section_timeouts()
    assert computed.section == {'one': 10, 'two': 11}
    assert computed.out_of_section == 9
    assert computed.calculated_sections == {'one', 'two'}
    assert computed.is_out_of_section_calculated is True
    assert computed.were_calculated is True

    computed = tc.get_section_timeouts(selected_sections=['one'])
    assert computed.section == {'one': 10, 'two': None}
    assert computed.out_of_section == 9
    assert computed.calculated_sections == {'one'}
    assert computed.is_out_of_section_calculated is True
    assert computed.were_calculated is False

    computed = tc.get_section_timeouts(selected_sections=['two'])
    assert computed.section == {'one': None, 'two': 11}
    assert computed.out_of_section == 9
    assert computed.calculated_sections == {'two'}
    assert computed.is_out_of_section_calculated is True
    assert computed.were_calculated is False

    computed = tc.get_section_timeouts(selected_sections=None, calc_out_of_section=False)
    assert computed.section == {'one': 10, 'two': 11}
    assert computed.out_of_section is None
    assert computed.calculated_sections == {'one', 'two'}
    assert computed.is_out_of_section_calculated is False
    assert computed.were_calculated is False

    computed = tc.get_section_timeouts(selected_sections=[], calc_out_of_section=False)
    assert computed.section == {'one': None, 'two': None}
    assert computed.out_of_section is None
    assert computed.calculated_sections == set()
    assert computed.is_out_of_section_calculated is False
    assert computed.were_calculated is False


def test_section_timeouts_merging():

    # NOTE: timeouts are merged with EMA

    valid_sections = ['one', 'two']
    tc = TimeoutsCalc(sections=valid_sections, start_time=1, safety_factor=1.0)
    tc.update_on_section_event(section='one', action=SectionAction.OPEN, event_time=10)
    tc.update_on_section_event(section='one', action=SectionAction.CLOSE, event_time=111)

    computed = tc.get_section_timeouts(selected_sections=['one'], calc_out_of_section=True)
    assert computed.section == {'one': 101, 'two': None}
    assert computed.out_of_section == 9
    assert computed.calculated_sections == {'one'}
    assert computed.is_out_of_section_calculated is True
    assert computed.were_calculated is False

    # case1: merge with not calculated current timeouts
    current = SectionTimeouts(
        section={'one': 102, 'two': 222},
        out_of_section=123,
        calculated_sections={},
        is_out_of_section_calculated=False,
    )
    assert current.are_valid is True
    assert current.were_calculated is False
    computed = tc.get_section_timeouts(
        selected_sections=['one'], calc_out_of_section=True, current=current
    )
    assert computed.section == {'one': 101, 'two': 222}
    assert computed.out_of_section == 9
    assert computed.calculated_sections == {'one'}
    assert computed.is_out_of_section_calculated is True
    assert computed.were_calculated is False

    # case2: merge with parially calculated current timeouts
    current = SectionTimeouts(
        section={'one': 102, 'two': 222},
        out_of_section=10,
        calculated_sections={'two'},
        is_out_of_section_calculated=True,
    )
    assert current.are_valid is True
    assert current.were_calculated is False
    computed = tc.get_section_timeouts(
        selected_sections=['one'], calc_out_of_section=True, current=current
    )
    assert computed.section == {'one': 101, 'two': 222}
    assert computed.out_of_section == 9.5
    assert computed.calculated_sections == {'one', 'two'}
    assert computed.is_out_of_section_calculated is True
    assert computed.were_calculated is True

    # case3: merge with fully calculated current timeouts
    current = SectionTimeouts(
        section={'one': 102, 'two': 223},
        out_of_section=10,
        calculated_sections={'one', 'two'},
        is_out_of_section_calculated=True,
    )
    assert current.are_valid is True
    assert current.were_calculated is True
    computed = tc.get_section_timeouts(
        selected_sections=['one'], calc_out_of_section=True, current=current
    )
    assert computed.section == {'one': 101.5, 'two': 223}
    assert computed.out_of_section == 9.5
    assert computed.calculated_sections == {'one', 'two'}
    assert computed.is_out_of_section_calculated is True
    assert computed.were_calculated is True


def test_get_merged_section_timeouts():
    valid_sections = ['one', 'two']
    tc = TimeoutsCalc(sections=valid_sections, start_time=1, safety_factor=1.0)

    # should update "one" and out-of-section (aka "oos") with the new, calculated
    current = SectionTimeouts(
        section={'one': 111, 'two': None},
        out_of_section=222,
        calculated_sections={},
        is_out_of_section_calculated=False,
    )
    new = SectionTimeouts(
        section={'one': 101, 'two': 103},
        out_of_section=202,
        calculated_sections={'one'},
        is_out_of_section_calculated=True,
    )
    computed = tc._get_merged_section_timeouts(new, current)
    assert computed.section == {'one': 101, 'two': None}
    assert computed.out_of_section == 202
    assert computed.calculated_sections == {'one'}
    assert computed.is_out_of_section_calculated is True

    # should update "two" with the new, calculated
    current = SectionTimeouts(
        section={'one': 101, 'two': None},
        out_of_section=123,
        calculated_sections={},
        is_out_of_section_calculated=False,
    )
    new = SectionTimeouts(
        section={'one': 102, 'two': 103},
        out_of_section=124,
        calculated_sections={'two'},
        is_out_of_section_calculated=False,
    )
    computed = tc._get_merged_section_timeouts(new, current)
    assert computed.section == {'one': 101, 'two': 103}
    assert computed.out_of_section == 123
    assert computed.calculated_sections == {'two'}
    assert computed.is_out_of_section_calculated is False

    # current was calculated, new is not calculated, current should be intact
    current = SectionTimeouts(
        section={'one': 101, 'two': 102},
        out_of_section=103,
        calculated_sections={'one', 'two'},
        is_out_of_section_calculated=True,
    )
    new = SectionTimeouts(
        section={'one': 999, 'two': 999},
        out_of_section=999,
        calculated_sections={},
        is_out_of_section_calculated=False,
    )
    computed = tc._get_merged_section_timeouts(new, current)
    assert computed.section == {'one': 101, 'two': 102}
    assert computed.out_of_section == 103
    assert computed.calculated_sections == {'one', 'two'}
    assert computed.is_out_of_section_calculated is True

    # merging calculated
    current = SectionTimeouts(
        section={'one': 101, 'two': 999},
        out_of_section=999,
        calculated_sections={'one'},
        is_out_of_section_calculated=False,
    )
    new = SectionTimeouts(
        section={'one': 999, 'two': 102},
        out_of_section=103,
        calculated_sections={'two'},
        is_out_of_section_calculated=True,
    )
    computed = tc._get_merged_section_timeouts(new, current)
    assert computed.section == {'one': 101, 'two': 102}
    assert computed.out_of_section == 103
    assert computed.calculated_sections == {'one', 'two'}
    assert computed.is_out_of_section_calculated is True


def test_end_all_sections():
    valid_sections = ['one', 'two', 'three']
    tc = TimeoutsCalc(sections=valid_sections, start_time=0, safety_factor=1.0)

    tc.update_on_section_event(section='one', action=SectionAction.OPEN, event_time=1)
    tc.update_on_section_event(section='two', action=SectionAction.OPEN, event_time=3)
    tc.update_on_section_event(section='three', action=SectionAction.OPEN, event_time=6)
    tc.update_on_section_event(section=None, action=SectionAction.CLOSE_ALL, event_time=10)
    computed = tc.get_section_timeouts()
    assert computed.section == {'one': 9, 'two': 7, 'three': 4}
    assert computed.out_of_section == 1
    assert computed.were_calculated is True

    tc.update_on_section_event(section='one', action=SectionAction.OPEN, event_time=12)
    tc.update_on_section_event(section='two', action=SectionAction.OPEN, event_time=21)
    tc.update_on_section_event(section='three', action=SectionAction.OPEN, event_time=23)
    tc.update_on_section_event(section=None, action=SectionAction.CLOSE_ALL, event_time=30)
    computed = tc.get_section_timeouts()
    assert computed.section == {'one': 18, 'two': 9, 'three': 7}
    assert computed.out_of_section == 2
    assert computed.were_calculated is True


def _rank_main(*args, tmp_dir, **kwargs):
    valid_sections = ['one', 'two']
    tc = TimeoutsCalc(sections=valid_sections, start_time=0, safety_factor=1.0)
    rank = dist.get_rank()
    if rank in [1, 2]:
        # update calculators in ranks 1,2 only. instances in other ranks are not updated.
        # - should obtain the values from ranks 1,2 during synchronization
        # rank 2 should get the largest times, that should be propagated to all ranks
        # Rank1:             Rank2
        # ->one (time=10)    ->one (time=20)
        # <-one (time=11)    <-one (time=22)
        # ->two (time=22)    ->two (time=33)
        # <-two (time=26)    <-two (time=41)
        start_time = 10 * rank
        tc.update_on_section_event(section='one', action=SectionAction.OPEN, event_time=start_time)
        tc.update_on_section_event(
            section='one', action=SectionAction.CLOSE, event_time=start_time + rank
        )
        tc.update_on_section_event(
            section='two',
            action=SectionAction.OPEN,
            event_time=start_time + rank + 11,
        )
        tc.update_on_section_event(
            section='two',
            action=SectionAction.CLOSE,
            event_time=start_time + rank + 11 + 4 * rank,
        )
        assert tc.can_get_section_timeouts() is True
    else:
        assert tc.can_get_section_timeouts() is False
    tc.synchronize_all()
    # after synchronization, all ranks should have the same values
    # safety factor is 1.0, so timeouts are the same as observed intervals
    computed = tc.get_section_timeouts()
    assert computed.section == {'one': 2, 'two': 8}
    assert computed.out_of_section == 20
    assert computed.were_calculated is True
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
