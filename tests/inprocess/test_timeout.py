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

import datetime
import multiprocessing
import os
import time
import unittest


import nvidia_resiliency_ext.inprocess as inprocess

from . import common


def kwargs():
    return {
        "store_kwargs": {"port": common.find_free_port()},
        "progress_watchdog_interval": datetime.timedelta(seconds=1e-3),
        "monitor_process_interval": datetime.timedelta(seconds=1e-3),
        "monitor_thread_interval": datetime.timedelta(seconds=1e-3),
        "last_call_wait": datetime.timedelta(seconds=1e-3),
    }


@unittest.mock.patch.dict(
    os.environ,
    {
        "RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "localhost",
    },
)
class TestSoft(unittest.TestCase):
    def test_chunked_sleep(self):
        timelimit = 10

        @inprocess.Wrapper(
            initialize=inprocess.initialize.RetryController(max_iterations=1),
            soft_timeout=datetime.timedelta(seconds=1),
            **kwargs(),
        )
        def fn(call_wrapper: inprocess.CallWrapper = None):
            call_wrapper.ping()
            num_chunks = 1000
            for _ in range(num_chunks):
                time.sleep(timelimit / num_chunks)

        start = time.perf_counter()
        with self.assertRaises(inprocess.exception.RestartAbort):
            fn()
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, timelimit / 2)

    def test_chunked_sleep_restart(self):
        timelimit = 10
        counter = 0
        max_iterations = 3
        ret_val = 123

        @inprocess.Wrapper(
            initialize=inprocess.initialize.RetryController(max_iterations=max_iterations),
            soft_timeout=datetime.timedelta(seconds=0.5),
            **kwargs(),
        )
        def fn(call_wrapper: inprocess.CallWrapper = None):
            call_wrapper.ping()
            nonlocal counter
            counter += 1
            if counter == max_iterations:
                return ret_val
            else:
                num_chunks = 10000
                for _ in range(num_chunks):
                    time.sleep(timelimit / num_chunks)

        start = time.perf_counter()
        ret = fn()
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, timelimit / 2)
        self.assertEqual(ret, ret_val)


@unittest.mock.patch.dict(
    os.environ,
    {
        "RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "localhost",
    },
)
class TestHard(unittest.TestCase):
    @staticmethod
    def launch(fn, timeout=datetime.timedelta(seconds=10)):
        ctx = multiprocessing.get_context("fork")
        proc = ctx.Process(target=fn)
        start = time.perf_counter()
        proc.start()
        proc.join()
        elapsed = time.perf_counter() - start
        proc.kill()
        return elapsed, proc.exitcode

    def test_sleep(self):
        timelimit = 10

        def run():
            @inprocess.Wrapper(
                soft_timeout=datetime.timedelta(seconds=0.5),
                hard_timeout=datetime.timedelta(seconds=1),
                termination_grace_time=datetime.timedelta(seconds=1e-3),
                **kwargs(),
            )
            def fn():
                time.sleep(timelimit)

            fn()

        elapsed, exitcode = self.launch(run)
        self.assertLess(elapsed, timelimit / 2)
        self.assertEqual(exitcode, -15)
