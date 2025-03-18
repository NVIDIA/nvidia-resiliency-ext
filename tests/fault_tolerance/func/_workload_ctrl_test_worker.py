import argparse
import os
import random
import signal
import sys
import time

import torch
import torch.distributed

import nvidia_resiliency_ext.fault_tolerance as ft


def _sigusr1_handler(*args, **kwargs):
    print(f"PID={os.getpid()} SIGUSR1 received!", file=sys.stderr)
    sys.exit(signal.SIGUSR1.value)


def _sigusr2_handler(*args, **kwargs):
    print(f"PID={os.getpid()} SIGUSR2 received!", file=sys.stderr)
    sys.exit(signal.SIGUSR2.value)


def _sigterm_handler(*args, **kwargs):
    print(f"PID={os.getpid()} SIGTERM received!", file=sys.stderr)
    sys.exit(signal.SIGTERM.value)


def get_cmd_file_content(ctl_file):
    # for synchronization, read on rank0 and broadcast the content
    obj_list = [None]
    if rank == 0:
        cmd_file_content = None
        if os.path.exists(ctl_file):
            with open(ctl_file, "r") as file:
                cmd_file_content = file.read().strip()
            os.unlink(ctl_file)
        obj_list = [cmd_file_content]
    torch.distributed.broadcast_object_list(obj_list, src=0)
    cmd_file_content = obj_list[0]
    return cmd_file_content


def get_random_rank(this_rank, is_agent_rdzv_host):
    # returns a randomly selected rank that is not run by the rdzv hosting agent
    world_size = torch.distributed.get_world_size()
    obj_list = [None] * world_size
    torch.distributed.all_gather_object(
        obj_list,
        (
            this_rank,
            is_agent_rdzv_host,
        ),
    )
    sel_rank = torch.tensor([-1], dtype=torch.int32)
    if this_rank == 0:
        candidates = [rank for rank, is_host in obj_list if not is_host]
        if not candidates:
            raise RuntimeError("Cannot find any ranks that are run by non-RDZV-hosting agents.")
        sel_rank[0] = random.choice(candidates)
    torch.distributed.broadcast(sel_rank, src=0)
    return sel_rank.item()


def maybe_handle_control_cmd(this_rank, ctl_file, is_agent_rdzv_host):

    cmd_file_content = get_cmd_file_content(ctl_file)
    if not cmd_file_content:
        return

    split_file_content = cmd_file_content.split()
    cmd = split_file_content[0]

    selected_rank = get_random_rank(this_rank, is_agent_rdzv_host)
    if this_rank != selected_rank:
        # this halts waits when a selected node is executing command
        time.sleep(60)
        return

    print(f"Rank:{this_rank} is executing command: {cmd}")

    if cmd == "exclude_rand_node_that_is_not_rdzv_host":
        req = ft.WorkloadControlRequest(ft.WorkloadAction.ExcludeThisNode, "test node excl req")
        ft_cli.send_workload_control_request(req)
        sys.exit(123)
    elif cmd == "shutdown_workload":
        req = ft.WorkloadControlRequest(ft.WorkloadAction.ShutdownWorkload, "test shutdown req")
        ft_cli.send_workload_control_request(req)
        sys.exit(123)
    elif cmd == "terminate_rand_rank":
        sys.exit(123)
    else:
        raise RuntimeError(f"Unexpected request type: {cmd}")


def update_run_cnt(run_cnt_file):
    count = 0
    try:
        with open(run_cnt_file, "r+") as f:
            count = int(f.read().strip())
    except FileNotFoundError:
        pass
    with open(run_cnt_file, "w") as f:
        f.write(f"{count+1}\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Workload Control Test Worker")
    parser.add_argument(
        "--max-time",
        type=int,
        required=False,
        default=600,
        help="Maximum execution time in seconds",
    )
    parser.add_argument(
        "--ctl-file",
        type=str,
        required=False,
        default="/tmp/_rank_ctl.txt",
        help="Path to the control file",
    )
    parser.add_argument(
        "--run-cnt-file",
        type=str,
        required=False,
        default="/tmp/_runs_cnt.txt",
        help="Path to the runs count file",
    )
    parser.add_argument(
        "--is-agent-rdzv-host",
        type=int,
        required=False,
        default=0,
        help="Is this rank spawned by the rdzv host agent",
    )
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGUSR1, _sigusr1_handler)
    signal.signal(signal.SIGUSR2, _sigusr2_handler)

    local_rank = int(os.environ['LOCAL_RANK'])

    print(f"Initializing... PID={os.getpid()}")

    pg = torch.distributed.init_process_group(backend='gloo')
    rank = torch.distributed.get_rank()

    if rank == 0:
        update_run_cnt(args.run_cnt_file)

    ft_cli = ft.RankMonitorClient()
    ft_cli.init_workload_monitoring()

    print(f"Rank={rank}: Sending heartbeats...")

    start_time = time.monotonic()

    while True:
        maybe_handle_control_cmd(
            this_rank=rank, ctl_file=args.ctl_file, is_agent_rdzv_host=args.is_agent_rdzv_host
        )
        elapsed_time = time.monotonic() - start_time
        if elapsed_time > args.max_time:
            print(f"Rank={rank}: Time limit reached")
            break
        ft_cli.send_heartbeat()
        time.sleep(0.1)

    ft_cli.shutdown_workload_monitoring()
    torch.distributed.destroy_process_group()

    print(f"Rank={rank}: Done! PID={os.getpid()}")
    sys.exit(0)
