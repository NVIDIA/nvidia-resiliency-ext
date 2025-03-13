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

import nvidia_resiliency_ext.fault_tolerance as fault_tolerance
from nvidia_resiliency_ext.fault_tolerance.utils import is_process_alive, wait_for_mp_events

from .utils import multiprocessing_execute_join, multiprocessing_execute_start

TEST_WORLD_SIZE = 4
ALL_RANK_IDS = set(range(TEST_WORLD_SIZE))
CHKPT_PATH = "/tmp/_ft_test_shutdown_dummy_chkpt.txt"
WORKLOAD_SHUTDOWN_TIMEOUT = 12
TERM_BY_FT_EXIT_CODE = 123
FT_TERM_SIGNAL = signal.SIGUSR1
INITIAL_HB_TIMEOUT = 2.0
SUBSEQUENT_HB_TIMEOUT = 1.0


def _get_ft_test_config():
    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = INITIAL_HB_TIMEOUT
    ft_cfg.rank_heartbeat_timeout = SUBSEQUENT_HB_TIMEOUT
    ft_cfg.workload_check_interval = 1.0
    ft_cfg.rank_termination_signal = FT_TERM_SIGNAL
    return ft_cfg


@pytest.fixture(autouse=True)
def _run_rank_monitors_fixture():
    ft_cfg = _get_ft_test_config()
    mp_ctx_spawn = mp.get_context("spawn")
    rank_monitors = []

    try:
        for rank in range(TEST_WORLD_SIZE):
            os.environ["RANK"] = str(rank)
            p = fault_tolerance.RankMonitorServer.run_in_subprocess(ft_cfg, rank, mp_ctx_spawn)
            rank_monitors.append(p)
            os.environ["RANK"] = ""

        yield

    finally:
        for p in rank_monitors:
            with contextlib.suppress(Exception):
                p.terminate()
                p.join(timeout=180)


def _rank_main(*args, rank_ready_events, **kwargs):
    rank_mon_cli = fault_tolerance.RankMonitorClient()
    rank_mon_cli.init_workload_monitoring()

    # Capture FT termination signal, and exit with custom code
    def _sig_handler(*args, **kwargs):
        sys.exit(TERM_BY_FT_EXIT_CODE)

    signal.signal(FT_TERM_SIGNAL, _sig_handler)

    # Notify main process that worker is initialized
    rank_ready_events[torch.distributed.get_rank()].set()

    # Infinite loop to simulate training
    while True:
        time.sleep(0.1)
        rank_mon_cli.send_heartbeat()
        time.sleep(0.1)
        try:
            torch.distributed.barrier()
        except Exception:
            # GLOO throws exception if distributed pg member is terminated
            # use sleep to simulate hang after some rank(s) are gone
            time.sleep(600)


def wait_for_process(pid, timeout):
    sleep_time = 0.1
    rem_time = timeout
    while is_process_alive(pid) and rem_time > 0:
        time.sleep(sleep_time)
        rem_time -= sleep_time


def _send_sig(sig, pids):
    for pid in pids:
        os.kill(pid, sig)


test_scenarios = [
    # When all ranks get SIGTERM,
    # tests end before any missing heartbeats are detected
    {
        "action": "term_ranks",
        "delay": 2.0,
        "sig": signal.SIGTERM,
        "target_ranks": ALL_RANK_IDS,
        "should_write_chkpt": False,
        "expected_ret_codes": TEST_WORLD_SIZE * [-signal.SIGTERM],
    },
    # Just one rank get SIGTERM
    # remaning ranks should be terminated due to missing heartbeats
    {
        "action": "term_ranks",
        "delay": 2.0,
        "sig": signal.SIGTERM,
        "target_ranks": [0],
        "should_write_chkpt": True,
        "expected_ret_codes": [-signal.SIGTERM] + (TEST_WORLD_SIZE - 1) * [TERM_BY_FT_EXIT_CODE],
    },
    # When rank 0 get SIGKILL,
    # remaning ranks should be terminated due to missing heartbeats
    {
        "action": "term_ranks",
        "delay": 2.0,
        "sig": signal.SIGKILL,
        "target_ranks": [0],
        "should_write_chkpt": True,
        "expected_ret_codes": [-signal.SIGKILL] + (TEST_WORLD_SIZE - 1) * [TERM_BY_FT_EXIT_CODE],
    },
    # Ranks 1,2 killed, other should be terminated due to missing heartbeats
    {
        "action": "term_ranks",
        "delay": 2.0,
        "sig": signal.SIGKILL,
        "target_ranks": [1, 2],
        "should_write_chkpt": True,
        "expected_ret_codes": [TERM_BY_FT_EXIT_CODE]
        + 2 * [-signal.SIGKILL]
        + (TEST_WORLD_SIZE - 3) * [TERM_BY_FT_EXIT_CODE],
    },
    # All ranks killed,
    # tests end before any missing heartbeats are detected
    {
        "action": "term_ranks",
        "delay": 2.0,
        "sig": signal.SIGKILL,
        "target_ranks": ALL_RANK_IDS,
        "should_write_chkpt": False,
        "expected_ret_codes": TEST_WORLD_SIZE * [-signal.SIGKILL],
    },
]


@pytest.mark.parametrize("test_scenario", test_scenarios)
def test_shutdown(test_scenario):
    with contextlib.suppress(FileNotFoundError):
        os.remove(CHKPT_PATH)

    mp_ctx = torch.multiprocessing.get_context("spawn")
    rank_ready_events = [mp_ctx.Event() for _ in range(TEST_WORLD_SIZE)]

    rank_processes = multiprocessing_execute_start(
        worker_fn=_rank_main,
        world_size=TEST_WORLD_SIZE,
        mp_ctx=mp_ctx,
        backend="gloo",
        dist_store_type="file",
        test_scenario=test_scenario,
        rank_ready_events=rank_ready_events,
    )

    wait_for_mp_events(rank_ready_events, timeout=60)

    rank_pids = {r: p.pid for r, p in enumerate(rank_processes)}

    if test_scenario["action"] == "term_ranks":
        target_pids = [rank_pids[r] for r in test_scenario["target_ranks"]]
        time.sleep(test_scenario["delay"])
        _send_sig(test_scenario["sig"], target_pids)
    else:
        raise Exception(f"Unrecognized action {test_scenario}")

    ret_codes = multiprocessing_execute_join(rank_processes, timeout=60)

    assert ret_codes == test_scenario["expected_ret_codes"]


def _rank_main_explicit_shutdown(*args, rank_ready_events, **kwargs):
    rank_mon_cli = fault_tolerance.RankMonitorClient()
    rank_mon_cli.init_workload_monitoring()

    rank_ready_events[torch.distributed.get_rank()].set()

    rank_mon_cli.send_heartbeat()
    rank_mon_cli.send_heartbeat()

    rank_mon_cli.shutdown_workload_monitoring()

    # there should be not timeout after explicit FT shutdown
    time.sleep(2.0 * SUBSEQUENT_HB_TIMEOUT)


def test_explicit_shutdown():
    mp_ctx = torch.multiprocessing.get_context("spawn")
    rank_ready_events = [mp_ctx.Event() for _ in range(TEST_WORLD_SIZE)]

    rank_processes = multiprocessing_execute_start(
        worker_fn=_rank_main_explicit_shutdown,
        world_size=TEST_WORLD_SIZE,
        mp_ctx=mp_ctx,
        backend="gloo",
        dist_store_type="file",
        rank_ready_events=rank_ready_events,
    )

    wait_for_mp_events(rank_ready_events, timeout=60)

    ret_codes = multiprocessing_execute_join(rank_processes, timeout=60)

    assert all([r == 0 for r in ret_codes])
