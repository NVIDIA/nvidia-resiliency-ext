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
# This example show the optimal usage:
# - only the training loop and objects depending on a torch distributed process
# group are being restarted upon a failure
# - process-group-independent objects (e.g. TCPStore, Model, Optimizer) are
# created once, and reused between all restart iterations to minimize restart
# latency
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

os.environ['TORCH_CPP_LOG_LEVEL'] = 'error'
import torch

import nvidia_resiliency_ext.inprocess as inprocess

raise_timestamp = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inprocess Restart Optimal Example',
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

    parser.add_argument(
        '--nested-restarter',
        action='store_true',
        help='extra logging for the nested restarter',
    )

    return parser.parse_args()


def train(
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


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=args.log_level,
    )
    logging.info(f'{args}')

    local_rank = int(os.environ['LOCAL_RANK'])

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

    # In this example, all objects created in ``main()`` are constructed only once and reused
    # for all restart iterations. *This* model and optimizer are process-group independent,
    # so they can safely be created outside of the wrapped function and reused.
    if args.seed is not None:
        torch.manual_seed(args.seed)
    model = torch.nn.Sequential(
        *[torch.nn.Linear(args.size, args.size) for _ in range(args.layers)]
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)

    # TCPStore uses ``(MASTER_PORT + 2)`` to avoid conflicts with TCPStore
    # created by ``torch.distributed.run`` and listening on ``MASTER_PORT``,
    # and Wrapper's TCPStore listening on ``(MASTER_PORT + 1)``
    store = torch.distributed.TCPStore(
        host_name=os.environ['MASTER_ADDR'],
        port=int(os.environ['MASTER_PORT']) + 2,
        world_size=int(os.environ['WORLD_SIZE']),
        is_master=(int(os.environ['RANK']) == 0),
        multi_tenant=True,
        wait_for_workers=True,
        use_libuv=True,
    )

    # TCPStore created by the Wrapper uses ``(MASTER_PORT + 1)`` port for the
    # internal Wrapper's TCPStore to avoid conflicts with application's TCPStore
    # listening on ``(MASTER_PORT + 2)``, and with a TCPStore created by
    # ``torch.distributed.run`` listening on ``MASTER_PORT``.
    #
    wrapper_kwargs = {
        'store_kwargs': {'port': int(os.getenv('MASTER_PORT', 29500)) + 1},
        'health_check': inprocess.health_check.CudaHealthCheck(),
    }

    if args.nested_restarter:
        wrapper_kwargs['initialize'] = inprocess.nested_restarter.NestedRestarterHandlingCompleted()
        wrapper_kwargs['abort'] = inprocess.Compose(
            inprocess.abort.AbortTorchDistributed(),
            inprocess.nested_restarter.NestedRestarterHandlingStarting(),
        )
        wrapper_kwargs['completion'] = inprocess.nested_restarter.NestedRestarterFinalized()
        wrapper_kwargs['terminate'] = inprocess.nested_restarter.NestedRestarterAborted()

    # An instance of ``inprocess.CallWrapper` is automatically injected into
    # wrapped function arguments when Wrapper is invoked.
    wrapped_train = inprocess.Wrapper(**wrapper_kwargs)(train)

    # Call the wrapped function.
    # ``wrapped_train()`` is automatically restarted to recover from faults.
    wrapped_train(store, model, opt, backend, device, timeout, args)


if __name__ == '__main__':
    main()
