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
import datetime
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
from typing import Any, Callable, Optional, Union

import torch


class Fault(enum.Enum):
    GPU_ERROR = enum.auto()
    GPU_SLEEP = enum.auto()
    WORKLOAD_EXC = enum.auto()
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


# Define the multiprocessing context at module level
ctx = multiprocessing.get_context('fork')
_registered_faults = {}


def register_fault(fault_name_or_enum: Union[str, Fault], handler: Callable):
    """
    Register a fault type and its handler.

    Args:
        fault_name_or_enum: Either a string name for a new fault or an existing Fault enum
        handler: Lambda function that implements the fault injection

    Returns:
        The Fault enum (either existing or newly created)
    """
    if isinstance(fault_name_or_enum, Fault):
        # Using an existing enum
        fault_enum = fault_name_or_enum
    else:
        # Add the new fault to the Fault enum
        new_fault = enum.auto()
        Fault._value2member_map_[new_fault] = fault_enum = type(Fault)(
            fault_name_or_enum, new_fault
        )
        Fault._member_names_.append(fault_name_or_enum)
        Fault._member_map_[fault_name_or_enum] = fault_enum

    # Register the handler
    _registered_faults[fault_enum] = handler

    return fault_enum


def dispatch_fault_injection(fault, delay, callback):
    if fault in _registered_faults:
        _registered_faults[fault](delay, callback)
    else:
        raise RuntimeError(f"Unknown fault type: {fault}")


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


def workload_exception(delay, callback):
    time.sleep(delay)
    log = logging.getLogger(__name__)
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
    log.critical(f'raising workload exception at {timestamp}')
    if callback is not None:
        callback()
    workload_raise_event.set()


# Register the workload exception fault
workload_raise_event = threading.Event()
register_fault(
    Fault.WORKLOAD_EXC,
    lambda delay, callback: threading.Thread(
        target=workload_exception, args=(delay, callback), daemon=True
    ).start(),
)


def maybe_raise_workload_exception():
    """
    Called in a workload as partner to workload_exception.

    When the workload_exception is triggered, a sentinel is set
    and if a workload calls this function, it can raise a gentle
    exception.
    """
    if workload_raise_event.is_set():
        workload_raise_event.clear()
        raise InjectedException


def async_raise_exception(tid, delay, callback):
    time.sleep(delay)
    log = logging.getLogger(__name__)
    log.critical(f'asynchronously raising {InjectedException}')
    if callback is not None:
        callback()
    async_raise(tid, InjectedException)


# Register the async exception fault
register_fault(
    Fault.ASYNC_EXC,
    lambda delay, callback: threading.Thread(
        target=async_raise_exception,
        args=(threading.main_thread().ident, delay, callback),
        daemon=True,
    ).start(),
)


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


# Register GPU error fault
register_fault(
    Fault.GPU_ERROR,
    lambda delay, callback: threading.Thread(
        target=raise_gpu_error,
        args=(delay, callback),
        daemon=True,
    ).start(),
)


def gpu_sleep(delay, device, callback):
    time.sleep(delay)
    log = logging.getLogger(__name__)
    log.critical(f'GPU sleep on {device=}')
    if callback is not None:
        callback()
    torch.cuda.set_device(device)
    torch.cuda._sleep(1 << 62)


# Register GPU sleep fault
register_fault(
    Fault.GPU_SLEEP,
    lambda delay, callback: threading.Thread(
        target=gpu_sleep, args=(delay, torch.cuda.current_device(), callback), daemon=True
    ).start(),
)


def lock_gil(delay, callback):
    time.sleep(delay)
    log = logging.getLogger(__name__)
    log.critical('locking GIL')
    if callback is not None:
        callback()
    re.match(r'(a?){40}a{40}', 'a' * 40)


# Register lock GIL fault
register_fault(
    Fault.LOCK_GIL,
    lambda delay, callback: threading.Thread(
        target=lock_gil, args=(delay, callback), daemon=True
    ).start(),
)


def segfault(delay, callback):
    time.sleep(delay)
    log = logging.getLogger(__name__)
    log.critical('raising segmentation fault')
    if callback is not None:
        callback()
    ctypes.string_at(1)


# Register segfault fault
register_fault(
    Fault.SEGFAULT,
    lambda delay, callback: threading.Thread(
        target=segfault, args=(delay, callback), daemon=True
    ).start(),
)


def send_signal(pid, signal, delay, callback):
    time.sleep(delay)
    log = logging.getLogger(__name__)
    log.critical(f'sending {signal=} to {pid=}')
    if callback is not None:
        callback()
    os.kill(pid, signal)


# Register signal faults
register_fault(
    Fault.SIGNAL_EXC,
    lambda delay, callback: (
        signal.signal(signal.SIGUSR1, termination_signal_handler),
        ctx.Process(
            target=send_signal,
            args=(os.getpid(), signal.SIGUSR1, delay, callback),
            daemon=True,
        ).start(),
    )[1],
)  # Return the result of .start()

register_fault(
    Fault.SIGKILL,
    lambda delay, callback: ctx.Process(
        target=send_signal,
        args=(os.getpid(), signal.SIGKILL, delay, callback),
        daemon=True,
    ).start(),
)

register_fault(
    Fault.SIGTERM,
    lambda delay, callback: ctx.Process(
        target=send_signal,
        args=(os.getpid(), signal.SIGTERM, delay, callback),
        daemon=True,
    ).start(),
)

register_fault(
    Fault.SIGINT,
    lambda delay, callback: ctx.Process(
        target=send_signal,
        args=(os.getpid(), signal.SIGINT, delay, callback),
        daemon=True,
    ).start(),
)

register_fault(
    Fault.SIGSTOP,
    lambda delay, callback: ctx.Process(
        target=send_signal,
        args=(os.getpid(), signal.SIGSTOP, delay, callback),
        daemon=True,
    ).start(),
)


def abort(delay, callback):
    time.sleep(delay)
    log = logging.getLogger(__name__)
    log.critical('aborting')
    if callback is not None:
        callback()
    os.abort()


# Register abort fault
register_fault(
    Fault.OS_ABORT,
    lambda delay, callback: threading.Thread(
        target=abort, args=(delay, callback), daemon=True
    ).start(),
)


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

        dispatch_fault_injection(fault, delay, callback)
