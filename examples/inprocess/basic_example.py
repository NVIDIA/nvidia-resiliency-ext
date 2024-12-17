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


# This example demonstrates how to integrate ``inprocess.Wrapper()`` into an
# existing PyTorch training codebase.
#
# In this case, the entire ``main()`` function is wrapped. While all features
# of ``inprocess.Wrapper()`` are available and active, the Wrapper is
# configured to restart the entire application upon any failure. Consequently,
# the application state is not preserved between restarts and the entire
# ``main()`` is relaunched, leading to less efficient recovery from failures.
#
# NOTE: inprocess.Wrapper is not fully compatible with modern
# ``torch.distributed.run``, because it automatically terminates all local
# workers upon any local worker process failure; in this case inprocess.Wrapper
# can only recover from transient faults that don't terminate any of the
# training processes

import argparse
import datetime
import logging
import os
import pathlib
import random
import time
from typing import Optional

import torch

import nvidia_resiliency_ext.inprocess as inprocess

raise_timestamp = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inprocess Restart Basic Example',
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
        '--fault-prob',
        default=0.001,
        type=float,
        help='fault injection probability',
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
# listening on ``(MASTER_PORT + 1)``, and with TCPStore created by
# ``torch.distributed.run`` listening on ``MASTER_PORT``.
@inprocess.Wrapper(
    store_kwargs={'port': int(os.getenv('MASTER_PORT', 29500)) + 2},
    health_check=inprocess.health_check.CudaHealthCheck(),
)
def main(call_wrapper: Optional[inprocess.CallWrapper] = None):
    global raise_timestamp
    if raise_timestamp is not None:
        restart_latency = time.perf_counter() - raise_timestamp
        logging.info(f'restart latency: {restart_latency:.3f}s')
    raise_timestamp = None

    args = parse_args()
    logging.info(f'{args}')

    log_interval = args.log_interval
    chkpt_interval = args.chkpt_interval

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

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

    if args.seed is not None:
        torch.manual_seed(args.seed)
    model = torch.nn.Sequential(
        *[torch.nn.Linear(args.size, args.size) for _ in range(args.layers)]
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)

    # TCPStore uses ``(MASTER_PORT + 1)`` to avoid conflicts with a TCPStore
    # created by ``torch.distributed.run`` and listening on ``MASTER_PORT``.
    store = torch.distributed.TCPStore(
        host_name=os.environ['MASTER_ADDR'],
        port=int(os.environ['MASTER_PORT']) + 1,
        world_size=int(os.environ['WORLD_SIZE']),
        is_master=int(os.environ['RANK']) == 0,
        multi_tenant=True,
        wait_for_workers=True,
        use_libuv=True,
    )

    torch.distributed.init_process_group(
        backend=backend,
        store=store,
        rank=int(os.environ['RANK']),
        world_size=int(os.environ['WORLD_SIZE']),
        timeout=timeout,
    )
    model_ddp = torch.nn.parallel.DistributedDataParallel(model)

    iteration = 0
    loss = torch.tensor(float('nan'))
    checkpoint_path = pathlib.Path(args.path) / 'checkpoint.pt'

    # Application loads state from the latest checkpoint on every restart
    # iteration of the wrapped function.
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
        torch.set_rng_state(checkpoint['rng'])
        iteration = checkpoint['iteration']

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
                }
                # Saving the checkpoint is performed within atomic() context
                # manager to ensure that the main thread won't execute
                # torch.save while a restart procedure is in progress.
                with call_wrapper.atomic():
                    torch.save(checkpoint, checkpoint_path)

        # Randomly trigger an example fault
        if random.random() < args.fault_prob:
            raise_timestamp = time.perf_counter()
            raise RuntimeError(f'example fault at {iteration=} from {rank=}')

        inp = torch.rand(args.size, args.size).to(device)
        model.zero_grad()
        out = model_ddp(inp)
        loss = out.square().mean()
        loss.backward()
        opt.step()
        loss.item()

        if rank == 0 and iteration % log_interval == log_interval - 1:
            logging.info(f'{rank=} {iteration=} {loss.item()=}')


if __name__ == '__main__':
    # ``inprocess.Wrapper`` uses logging library to output messages. In this
    # example the Wrapper is applied to ``main()``, therefore logging needs to
    # be initialized and configured before the Wrapper is launched.
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=args.log_level,
    )
    main()
