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

import ctypes
import enum
import logging
import multiprocessing
import os
import random
import re
import signal
import sys
import threading
import time
from typing import Any, Callable, Optional

import torch


class Fault(enum.Enum):
    GPU_ERROR = enum.auto()
    GPU_SLEEP = enum.auto()
    ASYNC_EXC = enum.auto()
    SIGNAL_EXC = enum.auto()
    OS_ABORT = enum.auto()
    LOCK_GIL = enum.auto()
    SEGFAULT = enum.auto()
    SIGINT = enum.auto()
    SIGKILL = enum.auto()
    SIGTERM = enum.auto()
    SIGSTOP = enum.auto()


class InjectedException(Exception):
    pass


def async_raise(tid, exc_type):
    set_async_exc = ctypes.pythonapi.PyThreadState_SetAsyncExc
    set_async_exc.argtypes = (ctypes.c_ulong, ctypes.py_object)
    set_async_exc.restype = ctypes.c_int

    if not sys.is_finalizing():
        res = set_async_exc(tid, exc_type)
    else:
        res = 1

    if res == 0:
        raise RuntimeError
    elif res > 1:
        set_async_exc(tid, None)
        raise RuntimeError


def termination_signal_handler(signum, frame):
    if not sys.is_finalizing():
        raise InjectedException


def async_raise_exception(tid, delay, callback):
    time.sleep(delay)
    log = logging.getLogger(__name__)
    log.critical(f'asynchronously raising {InjectedException}')
    if callback is not None:
        callback()
    async_raise(tid, InjectedException)


def raise_gpu_error(delay, callback):
    time.sleep(delay)
    log = logging.getLogger(__name__)
    device = torch.device(torch.cuda.current_device())
    log.critical(f'raising GPU error on {device}')
    if callback is not None:
        callback()
    b = torch.ones(1, dtype=torch.int64).to(device)
    a = torch.ones(1, dtype=torch.int64).to(device)
    a[b] = 0


def gpu_sleep(delay, device, callback):
    time.sleep(delay)
    log = logging.getLogger(__name__)
    log.critical(f'GPU sleep on {device=}')
    if callback is not None:
        callback()
    torch.cuda.set_device(device)
    torch.cuda._sleep(1 << 62)


def lock_gil(delay, callback):
    time.sleep(delay)
    log = logging.getLogger(__name__)
    log.critical('locking GIL')
    if callback is not None:
        callback()
    re.match(r'(a?){40}a{40}', 'a' * 40)


def segfault(delay, callback):
    time.sleep(delay)
    log = logging.getLogger(__name__)
    log.critical('raising segmentation fault')
    if callback is not None:
        callback()
    ctypes.string_at(1)


def send_signal(pid, signal, delay, callback):
    time.sleep(delay)
    log = logging.getLogger(__name__)
    log.critical(f'sending {signal=} to {pid=}')
    if callback is not None:
        callback()
    os.kill(pid, signal)


def abort(delay, callback):
    time.sleep(delay)
    log = logging.getLogger(__name__)
    log.critical('aborting')
    if callback is not None:
        callback()
    os.abort()


def inject_fault(
    faults: tuple[Fault],
    num_faults: int | tuple[int, int],
    keep_alive: int,
    delay: float | tuple[float, float],
    seed: int,
    callback: Optional[Callable[[], Any]] = None,
):
    log = logging.getLogger(__name__)

    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    generator = random.Random()
    generator.seed(seed)

    ctx = multiprocessing.get_context('fork')

    if isinstance(num_faults, int):
        min_faults, max_faults = num_faults, num_faults
    else:
        min_faults, max_faults = num_faults

    if not isinstance(delay, float):
        delay = generator.uniform(delay[0], delay[1])

    num_ranks_to_inject = min(generator.randint(min_faults, max_faults), world_size - keep_alive)
    ranks_to_inject = generator.sample(range(keep_alive, world_size), num_ranks_to_inject)
    fault = generator.sample(faults, 1)[0]

    if rank in ranks_to_inject:
        log.info(f'{seed=} {num_ranks_to_inject=} {ranks_to_inject=} ' f'{fault=} {delay=:.3f}')

        if fault == Fault.ASYNC_EXC:
            thread = threading.Thread(
                target=async_raise_exception,
                args=(threading.main_thread().ident, delay, callback),
                daemon=True,
            )
            thread.start()
        elif fault == Fault.SIGNAL_EXC:
            signal.signal(signal.SIGUSR1, termination_signal_handler)
            p = ctx.Process(
                target=send_signal,
                args=(os.getpid(), signal.SIGUSR1, delay, callback),
                daemon=True,
            )
            p.start()
        elif fault == Fault.GPU_ERROR:
            thread = threading.Thread(
                target=raise_gpu_error,
                args=(delay, callback),
                daemon=True,
            )
            thread.start()
        elif fault == Fault.LOCK_GIL:
            thread = threading.Thread(target=lock_gil, args=(delay, callback), daemon=True)
            thread.start()
        elif fault == Fault.GPU_SLEEP:
            device = torch.cuda.current_device()
            thread = threading.Thread(target=gpu_sleep, args=(delay, device, callback), daemon=True)
            thread.start()
        elif fault == Fault.SEGFAULT:
            thread = threading.Thread(target=segfault, args=(delay, callback), daemon=True)
            thread.start()
        elif fault == Fault.OS_ABORT:
            thread = threading.Thread(target=abort, args=(delay, callback), daemon=True)
            thread.start()
        elif fault == Fault.SIGKILL:
            p = ctx.Process(
                target=send_signal,
                args=(os.getpid(), signal.SIGKILL, delay, callback),
                daemon=True,
            )
            p.start()
        elif fault == Fault.SIGTERM:
            p = ctx.Process(
                target=send_signal,
                args=(os.getpid(), signal.SIGTERM, delay, callback),
                daemon=True,
            )
            p.start()
        elif fault == Fault.SIGINT:
            p = ctx.Process(
                target=send_signal,
                args=(os.getpid(), signal.SIGINT, delay, callback),
                daemon=True,
            )
            p.start()
        elif fault == Fault.SIGSTOP:
            p = ctx.Process(
                target=send_signal,
                args=(os.getpid(), signal.SIGSTOP, delay, callback),
                daemon=True,
            )
            p.start()
        else:
            raise RuntimeError
