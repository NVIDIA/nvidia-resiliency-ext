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

import contextlib
import json
import os
import shutil
import signal
import sys
import tempfile
import time

import pytest
import torch

import nvidia_resiliency_ext.fault_tolerance as fault_tolerance

from .utils import multiprocessing_execute_join, multiprocessing_execute_start

TEST_WORLD_SIZE = 4
ALL_RANK_IDS = set(range(TEST_WORLD_SIZE))
WORKLOAD_TIMEOUT = 60
TERM_BY_FT_EXIT_CODE = 123
FT_TERM_SIGNAL = signal.SIGUSR1
TIMEOUTS_FILENAME = "_ft_timeouts.json"


def _install_signal_handler():
    def __sighandler(*args, **kwargs):
        sys.exit(TERM_BY_FT_EXIT_CODE)

    signal.signal(FT_TERM_SIGNAL, __sighandler)


@pytest.fixture
def tmp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@contextlib.contextmanager
def rank_monitors_running(ft_cfg, mp_ctx):
    rank_monitors = []

    try:
        for rank in range(TEST_WORLD_SIZE):
            os.environ["RANK"] = str(rank)
            p = fault_tolerance.RankMonitorServer.run_in_subprocess(ft_cfg, rank, mp_ctx)
            rank_monitors.append(p)
            os.environ["RANK"] = ""

        yield

    finally:
        for p in rank_monitors:
            with contextlib.suppress(Exception):
                p.terminate()
                p.join(timeout=180)


def _rank_main_1st_run(*args, tmp_dir, **kwargs):
    """
    Send a few heartbeats and then calculate timeouts.
    Check if the calculated timeouts are correct.
    Timeouts should be synchronized among all `TEST_WORLD_SIZE` ranks.
    """

    _install_signal_handler()

    rank_mon_cli = fault_tolerance.RankMonitorClient()

    # timeouts info is not availabe until initialized
    assert rank_mon_cli.timeouts is None

    rank_mon_cli.init_workload_monitoring()

    assert rank_mon_cli.timeouts.are_valid is False
    assert rank_mon_cli.timeouts.were_calculated is None

    rank = torch.distributed.get_rank()
    time.sleep(rank * 0.4)
    rank_mon_cli.send_heartbeat()  # 1st heartbeat

    # not enough data to calculate timeouts
    with pytest.raises(fault_tolerance.RankMonitorClientError):
        rank_mon_cli.calculate_and_set_timeouts()
    # non-throwing version, should return False if not enough data
    assert rank_mon_cli.calculate_and_set_timeouts(skip_if_not_ready=True) is False

    # timeouts can be calculated after 2nd timeout
    rank_mon_cli.send_heartbeat()
    # timeouts were not re-calculated yet
    assert rank_mon_cli.timeouts.are_valid is False
    assert rank_mon_cli.timeouts.were_calculated is None

    # calculate initial timeouts, that should be overwritten with the next estimate at the end
    rank_mon_cli.calculate_and_set_timeouts()
    assert rank_mon_cli.timeouts.are_valid is True
    assert rank_mon_cli.timeouts.were_calculated is True

    for i in range(4):
        rank_mon_cli.send_heartbeat()
        time.sleep(rank * 0.2)

    # final timeouts estimate, should overwrite the initial one
    rank_mon_cli.calculate_and_set_timeouts()
    assert rank_mon_cli.timeouts.are_valid is True
    assert rank_mon_cli.timeouts.were_calculated is True

    # NOTE: do not check calculated timeout values,
    # because initial iteration(s) take longer than the rest
    # and it's hard to predict the exact values.

    # Dump timeouts to a file, so that it can be checked in the 2nd run.
    state = {
        "rmon_cli_state": rank_mon_cli.state_dict(),
        "timeouts1_initial": rank_mon_cli.timeouts.initial,
        "timeouts1_subsequent": rank_mon_cli.timeouts.subsequent,
    }

    dest_file = os.path.join(tmp_dir, TIMEOUTS_FILENAME)
    with open(dest_file, "w") as f:
        json.dump(state, f)

    sys.exit(0)


def _rank_main_2nd_run(*args, tmp_dir, **kwargs):
    """
    Check if timeouts calculated during 1st run are available.
    """

    _install_signal_handler()

    rank_mon_cli = fault_tolerance.RankMonitorClient()

    # timeouts info is not availabe until initialized
    assert rank_mon_cli.timeouts is None

    # load state from 1st run
    src_file = os.path.join(tmp_dir, TIMEOUTS_FILENAME)
    with open(src_file, "r") as f:
        state = json.load(f)

    rank_mon_cli.load_state_dict(state["rmon_cli_state"])
    rank_mon_cli.init_workload_monitoring()
    rank_mon_cli.send_heartbeat()

    # check if the timeouts are restored correctly
    assert rank_mon_cli.timeouts.are_valid is True
    assert rank_mon_cli.timeouts.were_calculated is True

    assert rank_mon_cli.timeouts.initial == pytest.approx(state["timeouts1_initial"], rel=0.01)
    assert rank_mon_cli.timeouts.subsequent == pytest.approx(
        state["timeouts1_subsequent"], rel=0.01
    )

    # send a few heartbeats one after another and re-calculate the timeouts
    # timeouts should be reduced, as EMA is used to merge old and new values
    rank_mon_cli.send_heartbeat()
    rank_mon_cli.send_heartbeat()
    rank_mon_cli.send_heartbeat()
    rank_mon_cli.calculate_and_set_timeouts()

    assert rank_mon_cli.timeouts.initial < state["timeouts1_initial"]
    assert rank_mon_cli.timeouts.subsequent < state["timeouts1_subsequent"]

    sys.exit(0)


def _rank_main_3nd_run(*args, tmp_dir, **kwargs):
    """
    Check if timeouts can be set after RankMonitorClient is initialized.
    """

    _install_signal_handler()

    rank_mon_cli = fault_tolerance.RankMonitorClient()

    # timeouts info is not availabe until initialized
    assert rank_mon_cli.timeouts is None

    # load state from 1st run
    src_file = os.path.join(tmp_dir, TIMEOUTS_FILENAME)
    with open(src_file, "r") as f:
        state = json.load(f)

    rank_mon_cli.init_workload_monitoring()
    rank_mon_cli.load_state_dict(state["rmon_cli_state"])

    # check if the timeouts are restored correctly
    assert rank_mon_cli.timeouts.are_valid is True
    assert rank_mon_cli.timeouts.were_calculated is True

    assert rank_mon_cli.timeouts.initial == pytest.approx(state["timeouts1_initial"], rel=0.01)
    assert rank_mon_cli.timeouts.subsequent == pytest.approx(
        state["timeouts1_subsequent"], rel=0.01
    )

    sys.exit(0)


def test_timeouts(tmp_dir):
    # 1st run calculates timeouts and exits.
    # 2nd run checks if the calculated timeouts are available.

    mp_ctx = torch.multiprocessing.get_context("spawn")

    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = None
    ft_cfg.rank_heartbeat_timeout = None
    ft_cfg.workload_check_interval = 1.0
    ft_cfg.rank_termination_signal = FT_TERM_SIGNAL

    with rank_monitors_running(ft_cfg=ft_cfg, mp_ctx=mp_ctx):
        rank_processes = multiprocessing_execute_start(
            worker_fn=_rank_main_1st_run,
            world_size=TEST_WORLD_SIZE,
            mp_ctx=mp_ctx,
            backend="gloo",
            dist_store_type="file",
            tmp_dir=tmp_dir,
        )

        ret_codes = multiprocessing_execute_join(rank_processes, timeout=WORKLOAD_TIMEOUT)

        assert ret_codes == [0] * TEST_WORLD_SIZE

    # We set some arbitrary values for timeouts for the 2nd run.
    # These should be ignored and calculated timeouts should take precedence.
    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = 999.0
    ft_cfg.rank_heartbeat_timeout = 999.0
    ft_cfg.workload_check_interval = 1.0
    ft_cfg.rank_termination_signal = FT_TERM_SIGNAL

    with rank_monitors_running(ft_cfg=ft_cfg, mp_ctx=mp_ctx):
        rank_processes = multiprocessing_execute_start(
            worker_fn=_rank_main_2nd_run,
            world_size=TEST_WORLD_SIZE,
            mp_ctx=mp_ctx,
            backend="gloo",
            dist_store_type="file",
            tmp_dir=tmp_dir,
        )

        ret_codes = multiprocessing_execute_join(rank_processes, timeout=WORKLOAD_TIMEOUT)

        assert ret_codes == [0] * TEST_WORLD_SIZE

    # We set "None"" for timeouts for the 3nd run.
    # These should be replaced by the calculated timeouts
    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = None
    ft_cfg.rank_heartbeat_timeout = None
    ft_cfg.workload_check_interval = 1.0
    ft_cfg.rank_termination_signal = FT_TERM_SIGNAL

    with rank_monitors_running(ft_cfg=ft_cfg, mp_ctx=mp_ctx):
        rank_processes = multiprocessing_execute_start(
            worker_fn=_rank_main_3nd_run,
            world_size=TEST_WORLD_SIZE,
            mp_ctx=mp_ctx,
            backend="gloo",
            dist_store_type="file",
            tmp_dir=tmp_dir,
        )

        ret_codes = multiprocessing_execute_join(rank_processes, timeout=WORKLOAD_TIMEOUT)

        assert ret_codes == [0] * TEST_WORLD_SIZE
