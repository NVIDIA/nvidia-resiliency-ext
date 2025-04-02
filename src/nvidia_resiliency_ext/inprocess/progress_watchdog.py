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
import dataclasses
import logging
import sys
import threading
import time
from datetime import timedelta
from typing import Optional

from . import exception

MAX_PENDING = 1024


@dataclasses.dataclass
class Timestamp:
    auto: float = None
    manual: float = None

    def is_timed_out(self, timeout: timedelta) -> bool:
        current_time = time.monotonic()
        timeout_s = timeout.total_seconds()

        for tstamp in (self.auto, self.manual):
            if tstamp is not None and (current_time - tstamp > timeout_s):
                return True

        return False


class ProgressWatchdog(threading.Thread):
    @staticmethod
    @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
    def get_automatic_timestamp(data_ptr: ctypes.c_void_p) -> int:
        timestamp_ptr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_double))
        num_completed_ptr = ctypes.cast(
            ctypes.c_void_p(data_ptr).value + ctypes.sizeof(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
        )

        timestamp_ptr.contents.value = time.monotonic()
        num_completed_ptr.contents.value = (num_completed_ptr.contents.value + 1) % MAX_PENDING
        return 0

    def __init__(
        self,
        rank: int,
        monitor_process,
        interval: timedelta,
    ):
        super().__init__(name=f'{type(self).__name__}-{rank}')
        self.rank = rank
        self.monitor_process = monitor_process
        self.interval = interval

        self.timestamp = Timestamp(auto=time.monotonic())
        self.num_scheduled = 0
        self.switch_interval = timedelta(seconds=sys.getswitchinterval())

        # Events to control thread state
        self.interval_event = threading.Event()
        self.pause_requested = threading.Event()
        self.paused = threading.Event()
        self.should_stop = threading.Event()

        self.log = logging.getLogger(__name__)

    def reset(self):
        self.timestamp = Timestamp(auto=time.monotonic())
        self.send()

    def ping(self):
        self.timestamp.manual = time.monotonic()
        self.send()

    def send(self):
        self.monitor_process.send_timestamp(self.timestamp)

    def run(self):
        add_pending_call = ctypes.pythonapi.Py_AddPendingCall
        add_pending_call.restype = ctypes.c_int
        add_pending_call.argtypes = [
            ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p),
            ctypes.c_void_p,
        ]

        # Create buffer for timestamp and completed count
        buffer_size = ctypes.sizeof(ctypes.c_double) + ctypes.sizeof(ctypes.c_int64)
        buffer = ctypes.create_string_buffer(buffer_size)
        timestamp_ptr = ctypes.cast(ctypes.addressof(buffer), ctypes.POINTER(ctypes.c_double))
        num_completed_ptr = ctypes.cast(
            ctypes.addressof(buffer) + ctypes.sizeof(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
        )

        timestamp_ptr.contents.value = self.timestamp.auto
        num_completed_ptr.contents.value = 0

        while not self.should_stop.is_set():
            # Wait for either interval timeout or an event signal
            self.interval_event.wait(self.interval.total_seconds())
            self.interval_event.clear()

            # If pause requested, synchronize and set paused
            if self.pause_requested.is_set():
                # Wait until all scheduled are completed
                while self.num_scheduled != num_completed_ptr.contents.value:
                    time.sleep(self.switch_interval.total_seconds())

                # Now we are synchronized
                self.paused.set()

                # Wait here until pause is lifted or we should stop
                while self.pause_requested.is_set() and not self.should_stop.is_set():
                    time.sleep(self.switch_interval.total_seconds())

                # If we are resuming normal operation
                self.paused.clear()

            # If we should stop, break after sync is done
            if self.should_stop.is_set():
                break

            # Normal operation: schedule pending call
            adding_status = add_pending_call(self.get_automatic_timestamp, ctypes.addressof(buffer))
            if adding_status == 0:
                self.num_scheduled = (self.num_scheduled + 1) % MAX_PENDING
                self.timestamp.auto = timestamp_ptr.contents.value
                self.send()

    def pause_and_synchronize(self):
        r'''
        Request the thread to pause and synchronize pending calls. This method
        blocks until the thread is synchronized and paused.
        '''
        self.pause_requested.set()
        # Wake the thread if waiting
        self.interval_event.set()
        # Wait until paused is set
        self.paused.wait()
        # Reset timestamp
        self.reset()

    def spin_till_paused(self):
        while not self.paused.wait(self.switch_interval.total_seconds()):
            pass

    def resume(self):
        r'''
        Resume the thread from a paused state. This clears the pause request
        and waits until the thread acknowledges resumption.
        '''
        if not self.is_alive():
            raise RuntimeError

        self.pause_requested.clear()
        # Wake the thread if it's waiting in paused state
        self.interval_event.set()
        while self.paused.is_set():
            time.sleep(self.switch_interval.total_seconds())

    def shutdown(self, timeout: Optional[float] = None):
        r'''
        Gracefully shut down the thread:
        1. Pause and drain to ensure no pending calls remain.
        2. Set should_stop and wake the thread.
        3. Join the thread.
        '''
        if not self.is_alive():
            return

        self.pause_and_synchronize()
        self.should_stop.set()
        # Wake the thread to exit
        self.interval_event.set()
        self.join(timeout=timeout)
        if self.is_alive():
            raise exception.InternalError
