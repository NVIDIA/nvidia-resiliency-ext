# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This example demonstrates how to integrate ``inprocess`` and ``fault_tolerance``
# into an existing PyTorch training codebase. For simplicity, ``inprocess`` does not
# filter out any ranks, and there are no idle or spare ranks. Otherwise, fault tolerance (FT)
# would need to be disabled on inactive ranks.
#
# To run this example, use the accompanying bash script:
# ./examples/fault_tolerance/run_inprocess_injob_example.sh

import argparse
import contextlib
import datetime
import logging
import os
import pathlib
import random
import signal
import time
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch

import nvidia_resiliency_ext.fault_tolerance as fault_tolerance
import nvidia_resiliency_ext.inprocess as inprocess

raise_timestamp = None


def _get_last_sim_fault_iter_path(rank, target_dir="/tmp/") -> str:
    # Returns the path of the file that stores the last simulated fault iteration for this rank
    return os.path.join(target_dir, f"_injob_inproc_example_rank{rank}_failed_iter.txt")


def _save_last_sim_fault_iter(rank, iteration, target_dir="/tmp/"):
    file_path = _get_last_sim_fault_iter_path(rank=rank, target_dir=target_dir)
    with open(file_path, mode='w') as f:
        f.write(f"{iteration}")


def _get_last_sim_fault_iter(rank, target_dir="/tmp/") -> int:
    file_path = _get_last_sim_fault_iter_path(rank=rank, target_dir=target_dir)
    if os.path.exists(file_path):
        with open(file_path, mode='r') as f:
            return int(f.read())
    return None


@dataclass
class _SimFaultDesc:
    """Represents a simulated fault description"""

    rank: int
    iteration: int
    fault_type: str

    @classmethod
    def from_str(cls, str_desc):
        try:
            split = str_desc.split(':')
            return cls(int(split[0]), int(split[1]), split[2].strip())
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid format for a simulated fault description: {str_desc}"
            )


def _parse_fault_desc_arg(value) -> Mapping[Tuple[int, int], _SimFaultDesc]:
    # Returns a mapping of (rank, iteration) to the simulated fault that should occur at that point.
    rank_iter_to_fault = dict()
    if value:
        for str_desc in value.split(','):
            f = _SimFaultDesc.from_str(str_desc)
            rank_iter_to_fault[(f.rank, f.iteration)] = f
    return rank_iter_to_fault


def _maybe_simulate_fault(rank, iteration, rank_iter_to_fault):

    # Checks whether a simulated fault should be triggered at the given rank and iteration.
    # Executes the simulated fault if the conditions are met.

    fault_desc = rank_iter_to_fault.get((rank, iteration), None)

    if fault_desc is None:
        return

    if _get_last_sim_fault_iter(rank) == iteration:
        # Prevents re-triggering the same fault after resuming from a checkpoint.
        logging.info(f'Skipped sim fault {fault_desc} as it was triggered before')
        return

    _save_last_sim_fault_iter(rank, iteration)

    logging.info(f'\n\n\n### Issuing simulated fault {fault_desc} ###\n\n\n')

    global raise_timestamp
    raise_timestamp = time.perf_counter()

    if fault_desc.fault_type == 'exc':
        raise RuntimeError(f'example fault at {iteration=} from {rank=}')
    elif fault_desc.fault_type == 'sigkill':
        os.kill(os.getpid(), signal.SIGKILL)
    elif fault_desc.fault_type == 'sleep':
        time.sleep(int(1e6))
    else:
        raise BaseException(f"Unexpected fault type {fault_desc.fault_type}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inprocess and Fault Tolerance Example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--size',
        default=64,
        type=int,
        help='model hidden size',
    )
    parser.add_argument(
        '--layers',
        default=4,
        type=int,
        help='number of layers',
    )
    parser.add_argument(
        '--log-interval',
        default=100,
        type=int,
        help='logging interval',
    )
    parser.add_argument(
        '--chkpt-interval',
        default=100,
        type=int,
        help='checkpointing interval',
    )
    parser.add_argument(
        '--total-iterations',
        default=1000000,
        type=int,
        help='total training iterations',
    )
    parser.add_argument(
        '--seed',
        default=None,
        type=int,
        help='random seed, time-based if None',
    )
    parser.add_argument(
        '--path',
        default='/tmp/',
        type=str,
        help='directory for the checkpoint file',
    )
    parser.add_argument(
        '--fault-iters',
        default='',
        type=_parse_fault_desc_arg,
        help='Comma-separated list of rank:iter:fault tuples for fault injection. '
        'fault can be exc|sleep|sigkill. Example: 0:1000:exc,1:2000,sleep',
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='device',
    )
    parser.add_argument(
        '--log-level',
        type=lambda s: logging._nameToLevel[s.upper()],
        default=logging.INFO,
        help='logging level',
    )

    return parser.parse_args()


# TCPStore created by the Wrapper uses ``(MASTER_PORT + 2)`` port for the
# internal Wrapper TCPStore to avoid conflicts with application's TCPStore
# listening on ``(MASTER_PORT + 1)``, and with a TCPStore created by
# ``torch.distributed.run`` listening on ``MASTER_PORT``.
#
# An instance of ``inprocess.CallWrapper` is automatically injected into
# wrapped function arguments when Wrapper is invoked.


@inprocess.Wrapper(
    store_kwargs={'port': int(os.getenv('MASTER_PORT', 29500)) + 2},
    health_check=inprocess.health_check.CudaHealthCheck(),
)
def train(
    ft_client,
    base_store,
    model,
    opt,
    backend,
    device,
    timeout,
    args,
    call_wrapper: Optional[inprocess.CallWrapper] = None,
):
    global raise_timestamp
    if raise_timestamp is not None:
        restart_latency = time.perf_counter() - raise_timestamp
        logging.info(f'restart latency: {restart_latency:.3f}s')
    raise_timestamp = None

    log_interval = args.log_interval
    chkpt_interval = args.chkpt_interval

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    logging.info(f"### STARTING RANK {rank} IN WORLD_SIZE {world_size} ###")

    # Reconnects FT so that rank monitors are aware of potential changes in rank-to-node mapping
    if ft_client.is_initialized:
        ft_client.shutdown_workload_monitoring()
    ft_client.init_workload_monitoring()

    # Create a new Store by adding a prefix based on the current inprocess
    # restart iteration. PrefixStore wraps the baseline TCPStore which is
    # reused for all restart iterations
    store = torch.distributed.PrefixStore(str(call_wrapper.iteration), base_store)

    torch.distributed.init_process_group(
        backend,
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=timeout,
    )

    model_ddp = torch.nn.parallel.DistributedDataParallel(model)

    iteration = 0
    loss = torch.tensor(float('nan'))
    checkpoint_path = pathlib.Path(args.path) / '_in_process_example_checkpoint.pt'

    # Application loads state from the latest checkpoint on every restart
    # iteration of the wrapped function.
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
        torch.set_rng_state(checkpoint['rng'])
        iteration = checkpoint['iteration']
        ft_client.load_state_dict(checkpoint['ft_state'])
    else:
        # if starting from scratch
        with contextlib.suppress(FileNotFoundError):
            os.unlink(_get_last_sim_fault_iter_path(rank))

    if args.seed is not None:
        random.seed(args.seed + iteration * world_size + rank)
    else:
        random.seed(time.perf_counter_ns())

    for iteration in range(iteration, args.total_iterations):

        # Application periodically saves a checkpoint. The checkpoint allows
        # the application to continue from previous state after a restart.
        if iteration % chkpt_interval == chkpt_interval - 1:
            torch.distributed.barrier()
            if rank == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'opt': opt.state_dict(),
                    'rng': torch.get_rng_state(),
                    'iteration': iteration,
                    'ft_state': ft_client.state_dict(),
                }
                # Saving the checkpoint is performed within atomic() context
                # manager to ensure that the main thread won't execute
                # torch.save while a restart procedure is in progress.
                with call_wrapper.atomic():
                    torch.save(checkpoint, checkpoint_path)

        _maybe_simulate_fault(rank, iteration, args.fault_iters)

        inp = torch.rand(args.size, args.size).to(device)
        model.zero_grad()
        out = model_ddp(inp)
        loss = out.square().mean()
        loss.backward()
        opt.step()
        loss.item()

        if rank == 0 and iteration % log_interval == log_interval - 1:
            logging.info(f'{rank=} {iteration=} {loss.item()=}')

        ft_client.send_heartbeat()  # notifies FT that the training process is still active.


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=args.log_level,
    )

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    if rank == 0:
        logging.info(f'\n##### NEW RUN {args} #####n')

    if args.device == 'cuda':
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda')
        backend = 'nccl'
        timeout = datetime.timedelta(seconds=150)
    elif args.device == 'cpu':
        device = torch.device('cpu')
        backend = 'gloo'
        timeout = datetime.timedelta(seconds=10)
    else:
        raise RuntimeError

    # All objects created in ``main()`` are constructed only once, and reused
    # for all restart iterations.
    if args.seed is not None:
        torch.manual_seed(args.seed)
    model = torch.nn.Sequential(
        *[torch.nn.Linear(args.size, args.size) for _ in range(args.layers)]
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)

    # TCPStore uses ``(MASTER_PORT + 1)`` to avoid conflicts with TCPStore
    # created by ``torch.distributed.run`` and listening on ``MASTER_PORT``.
    store = torch.distributed.TCPStore(
        host_name=os.environ['MASTER_ADDR'],
        port=int(os.environ['MASTER_PORT']) + 1,
        world_size=int(os.environ['WORLD_SIZE']),
        is_master=(int(os.environ['RANK']) == 0),
        multi_tenant=True,
        wait_for_workers=True,
        use_libuv=True,
    )

    # Prepares the FT client instance, it will be initialized in the ``train()``.
    ft_client = fault_tolerance.RankMonitorClient()

    try:
        # Call the wrapped function.
        # ``train()`` is automatically restarted to recover from faults.
        train(ft_client, store, model, opt, backend, device, timeout, args)
    finally:
        if ft_client.is_initialized:
            ft_client.shutdown_workload_monitoring()


if __name__ == '__main__':
    main()
