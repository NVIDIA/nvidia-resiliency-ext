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

import argparse
import collections
import datetime
import enum
import faulthandler
import logging
import multiprocessing
import os
import random
import signal
import sys
import time
import warnings
from datetime import timedelta
from typing import Optional

from nvidia_resiliency_ext.device_utils import get_current_device, get_distributed_init_method
import torch
import torch.nn as nn
from packaging import version

import nvidia_resiliency_ext.inprocess as inprocess
import nvidia_resiliency_ext.inprocess.tools as tools

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'


def unixTimestamp():
    now = datetime.datetime.utcfromtimestamp(time.time())
    return f'{str(now)} {os.getpid()} |> '


class InjectedException(Exception):
    pass


def parse_args(namespace=None, allow_extras=True):
    parser = argparse.ArgumentParser()

    inproc = parser.add_argument_group()
    inproc.add_argument('--min-world-size', default=1, type=int)
    inproc.add_argument('--min-active-world-size', default=1, type=int)
    inproc.add_argument('--max-world-size', default=None, type=int)
    inproc.add_argument('--world-size-div', default=1, type=int)
    inproc.add_argument('--groupsize', default=1, type=int)
    inproc.add_argument('--max-iterations', default=None, type=int)
    inproc.add_argument('--max-rank-faults', default=None, type=int)
    inproc.add_argument('--signal', type=lambda s: signal.Signals[s.upper()])
    inproc.add_argument('--health-timeout', default=5, type=float)
    inproc.add_argument('--progress-watchdog-int', default=0.01, type=float)
    inproc.add_argument('--monitor-thread-int', default=0.01, type=float)
    inproc.add_argument('--monitor-process-int', default=0.01, type=float)
    inproc.add_argument('--soft-timeout', default=60, type=float)
    inproc.add_argument('--hard-timeout', default=90, type=float)
    inproc.add_argument('--heartbeat-timeout', default=30, type=float)
    inproc.add_argument('--last-call-wait', default=1, type=float)
    inproc.add_argument('--termination-grace-time', default=5, type=float)
    inproc.add_argument('--store-timeout', default=60, type=float)
    inproc.add_argument('--barrier-timeout', default=120, type=float)
    inproc.add_argument('--completion-timeout', default=120, type=float)
    inproc.add_argument('--disabled', default=False, action='store_true')
    inproc.add_argument('--logfile', default='/tmp/inproc_monitor_{rank}.log')

    train = parser.add_argument_group()
    train.add_argument('--seed', default=123, type=int)
    train.add_argument('--world-size', default=8, type=int)
    train.add_argument('--last-iteration', default=None, type=int)
    train.add_argument('--backend', default='gloo', choices=['gloo', 'nccl'])
    train.add_argument('--min-faults', default=1, type=int)
    train.add_argument('--max-faults', default=1, type=int)
    train.add_argument('--process-group-timeout', default=120, type=float)
    train.add_argument('--distributed', default=False, action='store_true')
    train.add_argument('--check-fault', default=False, action='store_true')
    train.add_argument('--faulthandler', default=False, action='store_true')
    train.add_argument('--traceback', default=False, action='store_true')
    train.add_argument('--ignore-sigterm', default=False, action='store_true')
    train.add_argument('--all-reduce', default=0, type=int)
    train.add_argument('--nproc-per-node', default=None, type=int)
    train.add_argument('--size', default=16, type=int)
    train.add_argument('--layers', default=16, type=int)
    train.add_argument('--batch', default=1024, type=int)
    train.add_argument('--hidden', default=1024, type=int)
    train.add_argument('--log-interval', default=10, type=int)
    train.add_argument('--sync-interval', default=1, type=int)
    train.add_argument('--keep-alive', default=1, type=int)
    train.add_argument('--train-stall', default=0.1, type=float)
    train.add_argument('--train-stall-chunks', default=1000, type=int)
    train.add_argument('--train-sleep', default=0.1, type=float)
    train.add_argument('--progress', default=False, action='store_true')
    train.add_argument('--ext-min-delay', default=1, type=float)
    train.add_argument('--ext-max-delay', default=2, type=float)
    train.add_argument(
        '--ext-fault',
        type=lambda s: tools.inject_fault.Fault[s.upper()],
        default=None,
        nargs='+',
    )

    train.add_argument(
        '--fault',
        type=lambda s: Fault[s.upper()],
        default=None,
        nargs='+',
    )
    train.add_argument(
        '--log',
        type=lambda s: logging._nameToLevel[s.upper()],
        default=logging.INFO,
    )
    train.add_argument(
        '--mode', type=lambda s: Mode[s.upper()], default=Mode.SPIN
    )

    args, extras = parser.parse_known_args(namespace=namespace)
    if not allow_extras:
        assert not extras, extras

    if args.nproc_per_node is None:
        args.nproc_per_node = args.world_size

    assert args.min_faults > 0
    assert args.keep_alive <= args.min_world_size

    if not args.fault and not args.ext_fault:
        warnings.warn('fault injection mode was not specified')

    return args


class Fault(enum.Enum):
    CPU = enum.auto()
    GPU = enum.auto()
    EXC = enum.auto()
    KILL = enum.auto()
    TERM = enum.auto()
    EXIT = enum.auto()
    ABORT = enum.auto()


class Mode(enum.Enum):
    SPIN = enum.auto()
    TRAIN = enum.auto()


def maybe_trigger_fault(rank, world_size, args):
    log = logging.getLogger()
    num_ranks_to_trigger = min(
        random.randint(args.min_faults, args.max_faults),
        world_size - args.keep_alive,
    )

    # range starts from keep_alive because first keep_alive ranks can't fail,
    # typically keep_alive = 1 because rank 0 hosts TCPStore
    ranks_to_trigger = random.sample(
        range(args.keep_alive, world_size), num_ranks_to_trigger
    )

    if (
        args.check_fault
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        all_ranks_to_trigger = [None] * world_size
        torch.distributed.all_gather_object(
            all_ranks_to_trigger, ranks_to_trigger
        )

        for other_ranks_to_trigger in all_ranks_to_trigger:
            if other_ranks_to_trigger != ranks_to_trigger:
                log.critical(f'inconsistent {ranks_to_trigger=}')
                os.kill(os.getpid(), signal.SIGKILL)

    if isinstance(args.fault, collections.abc.Sequence):
        generator = random.Random()
        generator.seed(random.randint(0, 1000000) + rank)
        fault = generator.sample(args.fault, 1)[0]
    else:
        fault = args.fault

    if rank in ranks_to_trigger:
        log.info(f'{rank=} triggers {fault=}')

        if fault == Fault.GPU:
            b = torch.ones(1, dtype=torch.int64).cuda()
            a = torch.ones(1, dtype=torch.int64).cuda()
            a[b] = 0

        elif fault == Fault.CPU:
            b = torch.ones(1, dtype=torch.int64)
            a = torch.ones(1, dtype=torch.int64)
            a[b] = 0

        elif fault == Fault.EXC:
            raise InjectedException

        elif fault == Fault.EXIT:
            sys.exit(0)

        elif fault == Fault.KILL:
            os.kill(os.getpid(), signal.SIGKILL)

        elif fault == Fault.TERM:
            os.kill(os.getpid(), signal.SIGTERM)

        elif fault == Fault.ABORT:
            os.abort()


class RankFilter(logging.Filter):
    def __init__(self, rank, keyword, show_traceback):
        super().__init__()
        self.rank = rank
        self.keyword = keyword
        self.show_traceback = show_traceback

    def filter(self, record):
        record.rank = self.rank
        if self.show_traceback and sys.exc_info()[0] is not None:
            record.exc_info = sys.exc_info()
        if self.keyword in record.msg and self.rank != 0:
            return False
        else:
            return True


class AdaptiveFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)
        self.max_widths = {}

    def format(self, record):
        # Dynamically adjust the widths based on the record content
        formatted = super().format(record)
        parts = formatted.split('|')

        # Update maximum widths for each part
        for index, part in enumerate(parts[:-1]):
            current_width = len(part)
            if index not in self.max_widths:
                self.max_widths[index] = current_width
            else:
                if current_width > self.max_widths[index]:
                    self.max_widths[index] = current_width

        # Apply padding for alignment
        padded_parts = [
            part.ljust(self.max_widths.get(index, 0))
            for index, part in enumerate(parts)
        ]

        return '|'.join(padded_parts)


def training_main():
    args = parse_args(allow_extras=False)

    rank = int(os.environ['RANK'])

    logging.basicConfig(
        level=args.log,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=f'/tmp/example_{rank}.log',
        filemode='w',
        force=True,
    )
    rank_filter = RankFilter(rank, '***', args.traceback)
    console = logging.StreamHandler(sys.stderr)
    format = (
        "%(asctime)s | %(levelname)s | %(name)s | %(rank)-3s | %(message)s"
    )
    formatter = AdaptiveFormatter(format)
    console.setFormatter(formatter)
    console.addFilter(rank_filter)
    console.setLevel(args.log)
    logging.getLogger().addHandler(console)

    base_store = torch.distributed.TCPStore(
        host_name=os.environ['MASTER_ADDR'],
        port=int(os.environ['MASTER_PORT']),
        world_size=int(os.environ['WORLD_SIZE']),
        is_master=int(os.environ['RANK']) == 0,
        multi_tenant=True,
        wait_for_workers=True,
        use_libuv=True,
    )

    wrapped = inprocess.Wrapper(
        store_kwargs={
            'timeout': timedelta(seconds=args.store_timeout),
            'port': int(os.environ['MASTER_PORT']) + 1,
        },
        initialize=inprocess.Compose(
            inprocess.initialize.RetryController(
                max_iterations=args.max_iterations,
                min_world_size=args.min_world_size,
                min_active_world_size=args.min_active_world_size,
            ),
        ),
        finalize=None,
        health_check=inprocess.Compose(
            inprocess.health_check.CudaHealthCheck(
                timedelta(seconds=args.health_timeout)
            ),
            inprocess.health_check.FaultCounter(args.max_rank_faults),
        ),
        rank_assignment=inprocess.Compose(
            inprocess.rank_assignment.ShiftRanks(),
            inprocess.rank_assignment.FilterGroupedByKey(
                key_or_fn=lambda rank, _: rank // args.groupsize,
                condition=lambda count: count == args.groupsize,
            ),
        ),
        rank_filter=inprocess.Compose(
            inprocess.rank_filter.MaxActiveWorldSize(args.max_world_size),
            inprocess.rank_filter.WorldSizeDivisibleBy(args.world_size_div),
        ),
        monitor_thread_interval=timedelta(seconds=args.monitor_thread_int),
        monitor_process_interval=timedelta(seconds=args.monitor_process_int),
        progress_watchdog_interval=timedelta(
            seconds=args.progress_watchdog_int
        ),
        soft_timeout=timedelta(seconds=args.soft_timeout),
        hard_timeout=timedelta(seconds=args.hard_timeout),
        heartbeat_timeout=timedelta(seconds=args.heartbeat_timeout),
        barrier_timeout=timedelta(seconds=args.barrier_timeout),
        completion_timeout=timedelta(seconds=args.completion_timeout),
        last_call_wait=timedelta(seconds=args.last_call_wait),
        termination_grace_time=timedelta(seconds=args.termination_grace_time),
        monitor_process_logfile=args.logfile,
        enabled=not args.disabled,
    )(train)

    wrapped(base_store)


def train(
    base_store: torch.distributed.TCPStore = None,
    call_wrapper: inprocess.CallWrapper = None,
    namespace: Optional[argparse.Namespace] = None,
):
    args = parse_args(namespace)

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    if call_wrapper is not None:
        iteration = call_wrapper.iteration
    else:
        iteration = None

    if args.last_iteration is not None and args.last_iteration == iteration:
        return

    logging.info(f'*** starting {iteration=} {world_size=} ***')

    if args.faulthandler:
        faulthandler.register(
            signal.SIGTERM, file=open(f'/tmp/fault_{rank}.log', 'w')
        )

    if args.ignore_sigterm:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    if iteration is not None:
        random.seed(args.seed + iteration)
    else:
        random.seed(args.seed)

    if args.backend == 'gloo':
        device = torch.device('cpu')
    elif args.backend == 'nccl' or args.backend == 'xla':
        device = get_current_device()
    else:
        raise RuntimeError(f"Unsupported backend: {args.backend}")

    if args.distributed:
        if base_store is not None:
            store = torch.distributed.PrefixStore(str(iteration), base_store)

            torch.distributed.init_process_group(
                backend=args.backend,
                store=store,
                rank=int(os.environ['RANK']),
                world_size=int(os.environ['WORLD_SIZE']),
                timeout=datetime.timedelta(seconds=args.process_group_timeout),
            )
        else:
            init_method = None
            if args.backend == 'nccl' or args.backend == 'xla':
                init_method=get_distributed_init_method(),
            torch.distributed.init_process_group(
                backend=args.backend,
                rank=int(os.environ['RANK']),
                world_size=int(os.environ['WORLD_SIZE']),
                init_method=init_method,
                timeout=datetime.timedelta(seconds=args.process_group_timeout),
            )

        tensor = torch.ones(args.size, dtype=torch.int64, device=device)
        torch.distributed.all_reduce(tensor)
        assert (tensor == world_size).all()
        if device.type == 'cuda':
            torch.cuda.synchronize()

        for _ in range(args.all_reduce):
            torch.distributed.all_reduce(tensor)

    if call_wrapper is not None:
        call_wrapper.ping()

    for i in range(args.train_stall_chunks):
        time.sleep(args.train_stall / args.train_stall_chunks)

    if args.fault is not None:
        maybe_trigger_fault(rank, world_size, args)

    if args.ext_fault is not None:
        tools.inject_fault.inject_fault(
            faults=args.ext_fault,
            num_faults=(args.min_faults, args.max_faults),
            keep_alive=args.keep_alive,
            delay=(args.ext_min_delay, args.ext_max_delay),
            seed=args.seed + iteration,
        )

    if args.distributed and device.type == 'cuda':
        torch.cuda.synchronize()

    if args.distributed:
        for _ in range(args.all_reduce):
            torch.distributed.all_reduce(tensor)

    if world_size == args.keep_alive:
        logging.critical(f'{rank=} world_size == keep_alive, returning')
        return

    if args.mode == Mode.SPIN:
        while True:
            if call_wrapper is not None:
                call_wrapper.ping()
            # arbitrary sleep to emulate actual training code
            if args.progress:
                print(f'{rank=} training')
            time.sleep(args.train_sleep)

    elif args.mode == Mode.TRAIN:
        layers = args.layers
        batch = args.batch
        hidden = args.hidden
        log_interval = args.log_interval
        sync_interval = args.sync_interval

        model = nn.Sequential(
            *[nn.Linear(hidden, hidden, bias=False) for _ in range(layers)]
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-6)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model)

        iteration = 0
        if device.type == 'cuda':
            torch.cuda.synchronize()
        timer = time.perf_counter()
        while True:
            if call_wrapper is not None:
                call_wrapper.ping()
            model.zero_grad()
            inp = torch.rand(batch, hidden, device=device)
            out = model(inp)
            out.backward(torch.ones_like(out))
            opt.step()
            iteration += 1

            if iteration % sync_interval == (sync_interval - 1):
                if device.type == 'cuda':
                    torch.cuda.synchronize()

            if iteration % log_interval == (log_interval - 1):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - timer) / log_interval * 1000
                print(f'{rank=} {iteration=} {elapsed=}')
                timer = time.perf_counter()


def main():
    args = parse_args(allow_extras=False)

    rank = 'M'
    logging.basicConfig(
        level=args.log,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=f'/tmp/example_{rank}.log',
        filemode='w',
        force=True,
    )
    rank_filter = RankFilter(rank, '***', args.traceback)
    console = logging.StreamHandler(sys.stderr)
    format = (
        "%(asctime)s | %(levelname)s | %(name)s | %(rank)-3s | %(message)s"
    )
    formatter = AdaptiveFormatter(format)
    console.setFormatter(formatter)
    console.addFilter(rank_filter)
    console.setLevel(args.log)
    logging.getLogger().addHandler(console)
    log = logging.getLogger()

    log.info(args)

    random.seed(args.seed)

    if args.distributed and args.backend == 'nccl':
        if version.parse(torch.__version__) < version.parse('2.3.0'):
            msg = 'PyT >= 2.3.0 is required to terminate hung NCCL collectives'
            warnings.warn(msg)

    # standard code to start multiple training processes
    os.environ['WORLD_SIZE'] = str(args.world_size)

    log.info('creating processes')
    ctx = multiprocessing.get_context('fork')
    procs = []
    for rank in range(args.world_size):
        local_rank = rank % args.nproc_per_node

        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)

        p = ctx.Process(
            target=training_main,
            name=f'Worker-{rank}',
        )
        p.start()
        procs.append(p)

    try:
        for rank, p in enumerate(procs):
            p.join()
        log.info('all processes joined')
    except:
        log.info('killing processes')
        for rank, p in enumerate(procs):
            p.kill()
    log.info('finished')


if __name__ == '__main__':
    main()
