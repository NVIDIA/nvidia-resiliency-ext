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

import argparse
import os
import signal
import sys
import time

import torch.distributed as dist

from nvidia_resiliency_ext import fault_tolerance

WORLD_SIZE = 4
DEFAULT_TIMEOUT = 30

# NOTE: logs in capital letters are checked by the test runner


def test_rank_not_send_initial_hb(args):
    rank = dist.get_rank()
    rank_mon_cli = fault_tolerance.RankMonitorClient()
    rank_mon_cli.init_workload_monitoring()
    if args.which_rank == rank:
        print("RANK IS SKIPPING INITIAL HB")
        time.sleep(3600)
    else:
        while True:
            rank_mon_cli.send_heartbeat()
            time.sleep(1)


def test_rank_failed(args):
    rank = dist.get_rank()
    rank_mon_cli = fault_tolerance.RankMonitorClient()
    rank_mon_cli.init_workload_monitoring()
    rank_mon_cli.send_heartbeat()
    if args.which_rank == rank:
        print("RANK FAILED")
        raise RuntimeError(f"This is dummy exception to simulate fault in rank {rank}")
    else:
        while True:
            time.sleep(1)
            rank_mon_cli.send_heartbeat()


def test_ranks_exit_gracefully(args):
    rank = dist.get_rank()
    rank_mon_cli = fault_tolerance.RankMonitorClient()
    rank_mon_cli.init_workload_monitoring()
    if rank == 0:
        print("RANK EXITS GRACEFULLY")
    sys.exit(0)


def test_with_sigterm(args):
    rank = dist.get_rank()
    rank_mon_cli = fault_tolerance.RankMonitorClient()
    rank_mon_cli.init_workload_monitoring()
    rank_mon_cli.send_heartbeat()
    if rank == 0:
        # Rank0 sends SIGTERM to the launcher
        print("SIGTERM SENT TO LAUNCHER")
        os.kill(os.getppid(), signal.SIGTERM)
    while True:
        time.sleep(1)
        rank_mon_cli.send_heartbeat()


def test_dump_ft_config(args):
    rank = dist.get_rank()
    rank_mon_cli = fault_tolerance.RankMonitorClient()
    rank_mon_cli.init_workload_monitoring()
    if rank == 0:
        # dump FT config that was obtained from the monitor during initialization
        cfg_dump_path = os.path.join(args.tmp_dir, "cfg_dump.yaml")
        rank_mon_cli.cfg.to_yaml_file(cfg_dump_path)
    sys.exit(0)


def _get_restart_cnt(tmp_dir, update=False):
    if tmp_dir is None:
        return 0
    restart_cnt_file = os.path.join(tmp_dir, "restart_cnt.txt")
    restart_cnt = 0
    if os.path.exists(restart_cnt_file):
        with open(restart_cnt_file, "r") as f:
            restart_cnt = int(f.read())
    dist.barrier()  # ensure all ranks read the value before updating
    if update:
        with open(restart_cnt_file, "w") as f:
            f.write(str(restart_cnt + 1))
    return restart_cnt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str)
    parser.add_argument("--which_rank", type=int, default=0)
    parser.add_argument("--term_handler", type=str, default='default')
    parser.add_argument("--tmp_dir", type=str, default=None)

    args = parser.parse_args()

    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    restart_cnt = _get_restart_cnt(args.tmp_dir, update=(rank == 0))

    dist.barrier()

    if rank == 0:
        print("ALL RANKS STARTED")
        print(f"RESTART #{restart_cnt}")

    def _sigterm_handler_return0(*args, **kwargs):
        if rank == 0:
            print("RANK GOT SIGTERM: RETURN0")
        sys.exit(0)

    def _sigterm_handler_ignore(*args, **kwargs):
        if rank == 0:
            print("RANK GOT SIGTERM: IGNORED")
        pass

    if args.term_handler == "default":
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    elif args.term_handler == "return0":
        signal.signal(signal.SIGTERM, _sigterm_handler_return0)
    elif args.term_handler == "ignore":
        signal.signal(signal.SIGTERM, _sigterm_handler_ignore)
    else:
        raise ValueError(f"Unknown term_handler: {args.term_handler}")

    if args.scenario == "test_rank_not_send_initial_hb":
        test_rank_not_send_initial_hb(args)
    elif args.scenario == "test_rank_failed":
        test_rank_failed(args)
    elif args.scenario == "test_ranks_exit_gracefully":
        test_ranks_exit_gracefully(args)
    elif args.scenario in [
        "test_launcher_sigterm_graceful_exit",
        "test_launcher_sigterm_ignored",
    ]:
        # the only difference is the signal handler installed at the beginning
        test_with_sigterm(args)
    elif args.scenario == "test_ranks_restart":
        # change scenarios for each restart
        if restart_cnt == 0:
            test_rank_not_send_initial_hb(args)
        elif restart_cnt == 1:
            test_rank_failed(args)
        elif restart_cnt == 2:
            test_ranks_exit_gracefully(args)
        else:
            raise ValueError(f"restart_cnt should be <= 2, it is {restart_cnt}")
    elif args.scenario == "dump_cfg":
        test_dump_ft_config(args)
    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")
