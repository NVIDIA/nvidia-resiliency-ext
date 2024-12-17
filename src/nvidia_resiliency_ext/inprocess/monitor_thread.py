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

import contextlib
import ctypes
import inspect
import logging
import os
import signal
import sys
import threading
import time

import torch

from . import attribution
from . import exception
from .attribution import Interruption
from .attribution import InterruptionRecord
from .logging import log_exc
from .progress_watchdog import ProgressWatchdog
from .state import Status


class RankShouldRestart(BaseException):
    r'''
    :py:exc:`BaseException` asynchronously raised in the main thread to
    interrupt the execution of the function wrapped with
    :py:class:`inprocess.Wrapper`.
    '''

    def __del__(self):
        log = logging.getLogger(__name__)
        if log.isEnabledFor(logging.DEBUG):
            from . import wrap

            stack = inspect.stack(context=0)

            if len(stack) > 1 and stack[1].filename != wrap.__file__:
                locations = [
                    f'{info.frame.f_code.co_filename}:{info.frame.f_lineno}'
                    for info in stack[1:]
                ]
                traceback = ' <- '.join(locations)
                log.debug(f'{type(self).__name__} suppressed at {traceback}')
            del stack


def async_raise(tid, exc_type, event=None):
    if event is not None:
        event.wait()

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


def delayed_async_raise(tid, exc_type):
    event = threading.Event()
    thread = threading.Thread(
        target=async_raise,
        args=(tid, exc_type, event),
        daemon=True,
    )
    thread.start()
    event.set()


@contextlib.contextmanager
def reraise_if_unraisable(exc_type):
    def wrap(fn):
        def wrapped(*args, **kwargs):
            fn(*args, **kwargs)
            reraising_callback(*args, **kwargs)

        return wrapped

    def reraising_callback(unraisable_hook_args):
        if (
            issubclass(unraisable_hook_args.exc_type, exc_type)
            and not sys.is_finalizing()
        ):
            log = logging.getLogger(__name__)
            log.debug(f'sys.unraisablehook raises {exc_type}')
            delayed_async_raise(threading.main_thread().ident, exc_type)

    original_unraisablehook = sys.unraisablehook
    sys.unraisablehook = wrap(sys.unraisablehook)
    yield
    sys.unraisablehook = original_unraisablehook


def async_abort_main_thread(abort_signal=None, msg=None):
    if abort_signal is None:
        if msg is not None:
            DynamicRankShouldRestart = type(
                'RankShouldRestart',
                (RankShouldRestart,),
                {
                    '__init__': lambda self: super(
                        DynamicRankShouldRestart, self
                    ).__init__(msg)
                },
            )
            exc_type = DynamicRankShouldRestart
        else:
            exc_type = RankShouldRestart

        async_raise(threading.main_thread().ident, exc_type)
    else:
        os.kill(os.getpid(), abort_signal)


def abort_signal_handler(signum, frame):
    if not sys.is_finalizing():
        raise RankShouldRestart


class MonitorThread(threading.Thread):
    def __init__(
        self,
        state,
        abort,
        interval,
        abort_signal,
        progress_watchdog,
        soft_timeout,
        monitor_timeout,
        last_call_wait,
        atomic_lock,
        daemon,
    ):
        self.state = state
        self.abort = abort
        self.interval = interval
        self.abort_signal = abort_signal
        self.progress_watchdog = progress_watchdog
        self.soft_timeout = soft_timeout
        self.monitor_timeout = monitor_timeout
        self.last_call_wait = last_call_wait
        self.atomic_lock = atomic_lock

        self.should_stop = threading.Event()
        self.stop_raising = threading.Event()
        self.start_loop = threading.Event()
        self.loop_started = threading.Event()

        if abort_signal is not None:
            signal.signal(abort_signal, abort_signal_handler)

        super().__init__(
            name=f'{type(self).__name__}-{state.rank}',
            daemon=daemon,
        )

    def start_monitoring(self):
        self.start_loop.set()
        if not self.loop_started.wait(self.monitor_timeout.total_seconds()):
            raise exception.TimeoutError

    def run(self):
        log = logging.getLogger(__name__)
        rank = self.state.rank
        store = self.state.store
        state = self.state

        if not self.start_loop.wait(self.monitor_timeout.total_seconds()):
            raise exception.TimeoutError

        while not self.should_stop.is_set():
            self.loop_started.set()

            timed_out, _ = ProgressWatchdog.is_timed_out(
                self.progress_watchdog, self.soft_timeout
            )

            if timed_out:
                store.record_interrupted(
                    InterruptionRecord(rank, Interruption.SOFT_TIMEOUT)
                )

            if store.is_any_rank_iterrupted():
                self.state.status = Status.ABORTING
                time.sleep(self.last_call_wait.total_seconds())
                store.lock_interruption_records()
                interruption_records = store.get_interruption_records()
                msg = attribution.format_interruption_records(
                    interruption_records
                )

                with self.atomic_lock:
                    try:
                        if self.abort is not None:
                            state = self.abort(state)
                    except Exception as abort_ex:
                        log.critical(log_exc(state, abort_ex, 'abort_ex'))

                    self.progress_watchdog.pause_and_drain()

                    log.debug(f'{rank=} async_abort_main_thread')
                    async_abort_main_thread(self.abort_signal, msg)

                    if self.stop_raising.wait(
                        self.soft_timeout.total_seconds()
                    ):
                        break

                    log.debug(f'{rank=} spin async_abort_main_thread')
                    while not self.stop_raising.is_set():
                        async_abort_main_thread(self.abort_signal)
                    self.state.status = Status.ABORTED
                    break
            else:
                time.sleep(self.interval.total_seconds())

    def final_check_and_shutdown(self, timeout=None):
        self.stop_raising.set()
        self.join(timeout)
        if self.is_alive():
            raise RuntimeError

    def shutdown(self):
        try:
            self.stop_raising.set()
            self.should_stop.set()
            self.join(self.monitor_timeout.total_seconds())
            if self.is_alive():
                raise exception.TimeoutError
        except RankShouldRestart:
            pass
        finally:
            log = logging.getLogger(__name__)
            self.join(self.monitor_timeout.total_seconds())
            if self.is_alive():
                raise exception.TimeoutError
            log.debug('terminated')
