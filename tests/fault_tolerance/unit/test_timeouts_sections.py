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
import torch.distributed

import nvidia_resiliency_ext.fault_tolerance as fault_tolerance
from nvidia_resiliency_ext.fault_tolerance.data import FT_RANK_MONITOR_IPC_SOCKET_ENV_VAR

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


def _set_rmon_socket_env_var_for_this_rank():
    rank = os.environ["RANK"]
    ipc_sock_path = f"{tempfile.gettempdir()}/_rmon_r{rank}.socket"
    os.environ[FT_RANK_MONITOR_IPC_SOCKET_ENV_VAR] = ipc_sock_path


@contextlib.contextmanager
def rank_monitors_running(ft_cfg, mp_ctx):
    rank_monitors = []

    try:
        for rank in range(TEST_WORLD_SIZE):
            os.environ["RANK"] = str(rank)
            ipc_sock_path = f"{tempfile.gettempdir()}/_rmon_r{rank}.socket"
            p = fault_tolerance.RankMonitorServer.run_in_subprocess(
                cfg=ft_cfg, ipc_socket_path=ipc_sock_path, is_restarter_logger=False, mp_ctx=mp_ctx
            )
            rank_monitors.append(p)
            os.environ["RANK"] = ''

        yield

    finally:
        for p in rank_monitors:
            with contextlib.suppress(Exception):
                p.terminate()
                p.join(timeout=180)


def _rank_main_1st_run(*args, tmp_dir, **kwargs):
    """
    Timeouts should be synchronized among all `TEST_WORLD_SIZE` ranks.
    In the subsequent run, we will check if the timeouts were correctly loaded
    """

    _install_signal_handler()
    _set_rmon_socket_env_var_for_this_rank()

    rank = torch.distributed.get_rank()

    rank_mon_cli = fault_tolerance.RankMonitorClient()

    # timeouts info is not availabe until initialized
    assert rank_mon_cli.section_timeouts is None
    assert rank_mon_cli.hb_timeouts is None

    rank_mon_cli.init_workload_monitoring()

    assert rank_mon_cli.hb_timeouts.are_valid is False
    assert rank_mon_cli.hb_timeouts.were_calculated is False
    assert rank_mon_cli.section_timeouts.are_valid is False
    assert rank_mon_cli.section_timeouts.were_calculated is False

    # not enough data to calculate timeouts
    with pytest.raises(fault_tolerance.RankMonitorClientError):
        rank_mon_cli.calculate_and_set_section_timeouts()

    rank_mon_cli.start_section(section="one")

    # not enough data to calculate timeouts
    with pytest.raises(fault_tolerance.RankMonitorClientError):
        rank_mon_cli.calculate_and_set_section_timeouts()

    # sections "two" and "three" are nested in "one"
    rank_mon_cli.start_section(section="two")
    time.sleep(0.1)
    rank_mon_cli.end_section(section="two")

    rank_mon_cli.start_section(section="three")
    time.sleep(0.2)
    rank_mon_cli.end_section(section="three")

    # not enough data to calculate all timeouts - section "one" is still open
    with pytest.raises(fault_tolerance.RankMonitorClientError):
        rank_mon_cli.calculate_and_set_section_timeouts()

    # all closed now
    rank_mon_cli.end_section(section="one")

    # open section "one" again, but just for a short time
    # should not affect the timeouts
    time.sleep(0.51)
    rank_mon_cli.start_section(section="one")
    rank_mon_cli.end_section(section="one")

    assert rank_mon_cli.section_timeouts.are_valid is False
    assert rank_mon_cli.section_timeouts.were_calculated is False

    # calculate timeouts
    rank_mon_cli.calculate_and_set_section_timeouts()
    assert rank_mon_cli.section_timeouts.are_valid is True
    assert rank_mon_cli.section_timeouts.were_calculated is True
    assert rank_mon_cli.section_timeouts.out_of_section > 0.0
    assert rank_mon_cli.section_timeouts.section['one'] >= (
        rank_mon_cli.section_timeouts.section['two']
        + rank_mon_cli.section_timeouts.section['three']
    )

    # no rank heartbeat timeouts, as there were no heartbeats sent
    assert rank_mon_cli.hb_timeouts.are_valid is False

    # Dump timeouts to a file, so that it can be checked in the 2nd run.
    if rank == 0:
        state = {
            'rmon_cli_state': rank_mon_cli.state_dict(),
            'timeouts1_section': rank_mon_cli.section_timeouts.section,
            'timeouts1_out_of_section': rank_mon_cli.section_timeouts.out_of_section,
        }

        dest_file = os.path.join(tmp_dir, TIMEOUTS_FILENAME)
        with open(dest_file, "w") as f:
            json.dump(state, f)

    # Ensure that all ranks have the same state
    gathered = [None] * TEST_WORLD_SIZE
    torch.distributed.all_gather_object(gathered, rank_mon_cli.state_dict())
    for i in range(1, TEST_WORLD_SIZE):
        assert gathered[i] == gathered[0]

    sys.exit(0)


def _rank_main_2nd_run(*args, tmp_dir, **kwargs):
    """
    Check if timeouts calculated during 1st run are available.
    """

    _install_signal_handler()
    _set_rmon_socket_env_var_for_this_rank()

    rank_mon_cli = fault_tolerance.RankMonitorClient()

    # timeouts info is not availabe until initialized
    assert rank_mon_cli.hb_timeouts is None
    assert rank_mon_cli.section_timeouts is None

    # load state from 1st run
    src_file = os.path.join(tmp_dir, TIMEOUTS_FILENAME)
    with open(src_file, "r") as f:
        state = json.load(f)

    rank_mon_cli.load_state_dict(state['rmon_cli_state'])
    rank_mon_cli.init_workload_monitoring()

    # check if the timeouts are restored correctly
    assert rank_mon_cli.section_timeouts.are_valid is True
    assert rank_mon_cli.section_timeouts.were_calculated is True
    # heartbeats not used in this test
    assert rank_mon_cli.hb_timeouts.are_valid is False
    assert rank_mon_cli.hb_timeouts.were_calculated is False

    assert rank_mon_cli.section_timeouts.out_of_section == pytest.approx(
        state['timeouts1_out_of_section'], rel=0.01
    )
    for section, timeout in state['timeouts1_section'].items():
        assert rank_mon_cli.section_timeouts.section[section] == pytest.approx(timeout, rel=0.01)

    # some more heartbeats
    rank_mon_cli.send_heartbeat()
    rank_mon_cli.send_heartbeat()
    rank_mon_cli.send_heartbeat()

    sys.exit(0)


def test_timeouts(tmp_dir):
    # 1st run calculates timeouts and exits.
    # 2nd run checks if the calculated timeouts are available.

    mp_ctx = torch.multiprocessing.get_context("spawn")

    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = None
    ft_cfg.rank_heartbeat_timeout = None
    ft_cfg.rank_section_timeouts = {"one": 1.0, "two": None, "three": 999.0}
    ft_cfg.rank_out_of_section_timeout = 123.0
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
    ft_cfg.initial_rank_heartbeat_timeout = None
    ft_cfg.rank_heartbeat_timeout = None
    ft_cfg.rank_section_timeouts = {"one": 999.0, "two": 999.0, "three": 999.0}
    ft_cfg.rank_out_of_section_timeout = 999.0
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
