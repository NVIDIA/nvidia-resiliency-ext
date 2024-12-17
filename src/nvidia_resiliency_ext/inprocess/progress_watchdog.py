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
import multiprocessing
import threading
import time
import typing

from . import exception


class Timestamp(typing.NamedTuple):
    auto: float = None
    manual: float = None


class ProgressWatchdog(threading.Thread):
    def __init__(self, rank, connection, interval, daemon):
        self.rank = rank
        self.connection = connection
        self.interval = interval
        self.spin_interval = interval / 64
        self.send_lock = threading.RLock()

        self.timestamp = Timestamp(auto=time.monotonic())
        self.num_scheduled = 0

        self.loop_started = threading.Event()
        self.should_wait = threading.Event()
        self.is_synchronized = threading.Event()
        self.done_waiting = threading.Event()
        self.should_stop = threading.Event()

        super().__init__(name=f'{type(self).__name__}-{rank}', daemon=daemon)

    def reset(self):
        self.timestamp = Timestamp(auto=time.monotonic())
        self.send()

    def ping(self):
        self.timestamp = Timestamp(
            auto=self.timestamp.auto, manual=time.monotonic()
        )
        self.send()

    def send(self):
        with self.send_lock:
            self.connection.send(self.timestamp)

    @staticmethod
    def is_timed_out(source, timeout, timestamp=None):
        if timestamp is None:
            timestamp = Timestamp()

        if isinstance(source, ProgressWatchdog):
            timestamp = source.timestamp
        elif isinstance(source, multiprocessing.connection.Connection):
            while source.poll():
                try:
                    timestamp = source.recv()
                except EOFError:
                    break
        else:
            raise exception.InternalError

        current_time = time.monotonic()

        timed_out = any(
            item is not None and current_time - item > timeout.total_seconds()
            for item in timestamp
        )

        return timed_out, timestamp

    @staticmethod
    @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
    def get_automatic_timestamp(data_ptr):
        timestamp_ptr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_double))
        num_completed_ptr = ctypes.cast(
            ctypes.c_void_p(data_ptr).value + ctypes.sizeof(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
        )

        timestamp_ptr.contents.value = time.monotonic()
        num_completed_ptr.contents.value += 1

        return 0

    def run(self):
        add_pending_call = ctypes.pythonapi.Py_AddPendingCall
        add_pending_call.restype = ctypes.c_int
        add_pending_call.argtypes = [
            ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p),
            ctypes.c_void_p,
        ]

        buffer = ctypes.create_string_buffer(
            0, ctypes.sizeof(ctypes.c_double) + ctypes.sizeof(ctypes.c_int64)
        )
        timestamp_ptr = ctypes.cast(
            ctypes.addressof(buffer), ctypes.POINTER(ctypes.c_double)
        )
        num_completed_ptr = ctypes.cast(
            ctypes.addressof(buffer) + ctypes.sizeof(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
        )

        timestamp_ptr.contents.value = self.timestamp.auto
        num_completed_ptr.contents.value = 0

        while not self.should_stop.is_set():
            self.loop_started.set()

            while self.should_wait.is_set():
                self.done_waiting.clear()
                if self.num_scheduled == num_completed_ptr.contents.value:
                    self.is_synchronized.set()
                time.sleep(self.spin_interval.total_seconds())
            self.done_waiting.set()

            adding_status = add_pending_call(
                self.get_automatic_timestamp, ctypes.addressof(buffer)
            )
            if adding_status == 0:
                self.num_scheduled += 1
                self.timestamp = Timestamp(
                    auto=timestamp_ptr.contents.value,
                    manual=self.timestamp.manual,
                )
                self.send()
            time.sleep(self.interval.total_seconds())

        self.loop_started.clear()

    def resume(self):
        self.should_wait.clear()
        self.done_waiting.wait()
        self.is_synchronized.clear()

    def pause_and_drain(self):
        self.should_wait.set()
        self.is_synchronized.wait()
        self.reset()

    def spin_drain(self):
        while not self.is_synchronized.wait(
            self.spin_interval.total_seconds()
        ):
            pass

    def shutdown(self, timeout=None):
        self.should_stop.set()
        self.join(timeout)
        if self.is_alive():
            raise exception.InternalError
