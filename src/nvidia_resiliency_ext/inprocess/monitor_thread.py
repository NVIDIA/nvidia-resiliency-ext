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
import sys
import threading
import time
from datetime import timedelta

import torch

from . import attribution, exception
from .utils import log_exc


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
            allowed_fnames = (__file__, wrap.__file__)

            if len(stack) > 1 and stack[1].filename not in allowed_fnames:
                locations = [
                    f'{info.frame.f_code.co_filename}:{info.frame.f_lineno}' for info in stack[1:]
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
        if issubclass(unraisable_hook_args.exc_type, exc_type) and not sys.is_finalizing():
            log = logging.getLogger(__name__)
            log.debug(f'sys.unraisablehook raises {exc_type}')
            delayed_async_raise(threading.main_thread().ident, exc_type)

    original_unraisablehook = sys.unraisablehook
    sys.unraisablehook = wrap(sys.unraisablehook)
    yield
    sys.unraisablehook = original_unraisablehook


def async_abort_main_thread(msg=None):
    if msg is not None:
        DynamicRankShouldRestart = type(
            'RankShouldRestart',
            (RankShouldRestart,),
            {'__init__': lambda self: super(DynamicRankShouldRestart, self).__init__(msg)},
        )
        exc_type = DynamicRankShouldRestart
    else:
        exc_type = RankShouldRestart

    async_raise(threading.main_thread().ident, exc_type)


class MonitorThread(threading.Thread):
    def __init__(
        self,
        state,
        store,
        abort,
        interval,
        progress_watchdog,
        soft_timeout,
        last_call_wait,
        atomic_lock,
        daemon,
    ):
        self.state = state
        self.store = store
        self.abort = abort
        self.interval = interval
        self.progress_watchdog = progress_watchdog
        self.soft_timeout = soft_timeout
        self.last_call_wait = last_call_wait
        self.atomic_lock = atomic_lock

        self.stop_raising = threading.Event()
        self.stop_loop = threading.Event()

        self.monitor_timeout = 5 * interval + last_call_wait + timedelta(seconds=5)
        self.log = logging.getLogger(__name__)

        super().__init__(
            name=f'{type(self).__name__}-{state.rank}',
            daemon=daemon,
        )

    def run(self):
        log = logging.getLogger(__name__)
        state = self.state
        store = self.store

        while not self.stop_loop.is_set():
            try:
                store.wait_for_interrupted(self.interval)
            except torch.distributed.DistStoreError:
                time.sleep(sys.getswitchinterval())
            else:
                time.sleep(self.last_call_wait.total_seconds())
                store.lock_interruption_records()
                interruption_records = store.get_interruption_records()
                msg = attribution.format_interruption_records(interruption_records)

                with self.atomic_lock:
                    try:
                        if self.abort is not None:
                            self.abort(state)
                    except Exception as abort_ex:
                        log.critical(log_exc(state, abort_ex, 'abort_ex'))

                    self.progress_watchdog.pause_and_synchronize()
                    async_abort_main_thread(msg)
                    self.stop_raising.wait(self.soft_timeout.total_seconds())
                    while not self.stop_raising.is_set():
                        async_abort_main_thread()

                    break

    def maybe_join(self, timeout: float = None):
        self.stop_raising.set()
        self.join(timeout)
        if self.is_alive():
            raise exception.TimeoutError

    def shutdown(self):
        try:
            self.stop_raising.set()
            self.stop_loop.set()
            start = time.monotonic()
            while timedelta(seconds=(time.monotonic() - start)) < self.monitor_timeout:
                if not self.is_alive():
                    break
                time.sleep(sys.getswitchinterval())
            else:
                raise exception.TimeoutError
        except RankShouldRestart:
            pass
        finally:
            log = logging.getLogger(__name__)
            self.join(self.monitor_timeout.total_seconds())
            self.progress_watchdog.resume()
            if self.is_alive():
                raise exception.TimeoutError
            log.debug('terminated')
