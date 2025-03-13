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
import os
import signal
import sys
import time

import pytest
import torch
import torch.multiprocessing as mp

from nvidia_resiliency_ext import fault_tolerance

from .utils import multiprocessing_execute_join, multiprocessing_execute_start

TEST_WORLD_SIZE = 4
ALL_RANK_IDS = set(range(TEST_WORLD_SIZE))
CHKPT_PATH = "/tmp/_ft_test_init_dummy_chkpt.txt"
WORKLOAD_TIMEOUT = 60
TERM_BY_FT_EXIT_CODE = 123
FT_TERM_SIGNAL = signal.SIGUSR1


def _get_ft_test_config():
    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = 8.0
    ft_cfg.rank_heartbeat_timeout = 12.0
    ft_cfg.workload_check_interval = 1.0
    ft_cfg.rank_termination_signal = FT_TERM_SIGNAL
    return ft_cfg


def _install_signal_handler():
    def __sighandler(*args, **kwargs):
        sys.exit(TERM_BY_FT_EXIT_CODE)

    signal.signal(FT_TERM_SIGNAL, __sighandler)


@pytest.fixture(autouse=True)
def _run_rank_monitors_fixture():
    ft_cfg = _get_ft_test_config()
    mp_ctx_spawn = mp.get_context("spawn")
    rank_monitors = []

    try:
        for rank in range(TEST_WORLD_SIZE):
            os.environ["RANK"] = str(rank)
            p = fault_tolerance.RankMonitorServer.run_in_subprocess(
                ft_cfg, rank, mp_ctx_spawn
            )
            rank_monitors.append(p)
            os.environ["RANK"] = ""

        yield

    finally:
        for p in rank_monitors:
            with contextlib.suppress(Exception):
                p.terminate()
                p.join(timeout=180)


def _rank_main_all_ranks_initialized(*args, **kwargs):
    # Test scenario:
    # - initialize fault tolerance
    # - all ranks exit before any heartbeats are sent
    # Expected result:
    # - clean exit, no failure detected
    _install_signal_handler()
    rank_mon_cli = fault_tolerance.RankMonitorClient()
    rank_mon_cli.init_workload_monitoring()
    sys.exit(0)


def _rank_main_no_initial_heartbeat_from_rank_0(*args, **kwargs):
    # Test scenario:
    # - rank 0 fails to send initial hearbeat
    # - all other ranks send initial hearbeat and exit with 0 exit code
    # Expected result:
    # - workload failure detected, rank 0 is terminated by rank monitor
    _install_signal_handler()
    rank_mon_cli = fault_tolerance.RankMonitorClient()
    rank_mon_cli.init_workload_monitoring()

    rank = torch.distributed.get_rank()
    if rank == 0:
        time.sleep(WORKLOAD_TIMEOUT + 1)
    else:
        rank_mon_cli.send_heartbeat()
    sys.exit(0)


def _rank_main_no_initial_heartbeats(*args, **kwargs):
    # Test scenario:
    # - all ranks fail to send initial hearbeat
    # Expected result:
    # - workload failure detected, all ranks terminated by rank monitors
    _install_signal_handler()
    rank_mon_cli = fault_tolerance.RankMonitorClient()
    rank_mon_cli.init_workload_monitoring()
    time.sleep(WORKLOAD_TIMEOUT + 1)
    sys.exit(0)


test_scenarios = [
    (_rank_main_all_ranks_initialized, False, TEST_WORLD_SIZE * [0]),
    (
        _rank_main_no_initial_heartbeat_from_rank_0,
        True,
        [TERM_BY_FT_EXIT_CODE] + (TEST_WORLD_SIZE - 1) * [0],
    ),
    (
        _rank_main_no_initial_heartbeats,
        True,
        TEST_WORLD_SIZE * [TERM_BY_FT_EXIT_CODE],
    ),
]


@pytest.mark.parametrize(
    "rank_main, is_chkpt_expected, expected_ret_codes", test_scenarios
)
def test_init(rank_main, is_chkpt_expected, expected_ret_codes):
    with contextlib.suppress(FileNotFoundError):
        os.remove(CHKPT_PATH)

    mp_ctx = torch.multiprocessing.get_context("spawn")

    rank_processes = multiprocessing_execute_start(
        worker_fn=rank_main,
        world_size=TEST_WORLD_SIZE,
        mp_ctx=mp_ctx,
        backend="gloo",
        dist_store_type="file",
    )

    ret_codes = multiprocessing_execute_join(rank_processes, timeout=WORKLOAD_TIMEOUT)

    assert ret_codes == expected_ret_codes
