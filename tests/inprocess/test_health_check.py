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
import sys
import threading
import time
import unittest

import torch

import nvidia_resiliency_ext.inprocess as inprocess

from . import common  # noqa: F401


@unittest.skipIf(not torch.cuda.is_available(), 'cuda not available')
class TestCudaHealthCheck(unittest.TestCase):
    @staticmethod
    def launch(fn, timeout=datetime.timedelta(seconds=10)):
        ctx = multiprocessing.get_context('fork')
        proc = ctx.Process(target=fn)
        start_time = time.perf_counter()
        proc.start()

        proc.join(timeout.total_seconds())
        if proc.exitcode is None:
            proc.kill()
            proc.join()
        stop_time = time.perf_counter()
        elapsed = stop_time - start_time
        return proc.exitcode, elapsed

    def test_basic(self):
        def run():
            check = inprocess.health_check.CudaHealthCheck()
            check(None)
            torch.cuda.synchronize()

        exitcode, _ = self.launch(run)
        self.assertEqual(exitcode, 0)

    def test_timeout(self):
        def run():
            torch.ones(1).cuda()
            check = inprocess.health_check.CudaHealthCheck(datetime.timedelta(seconds=1))
            torch.cuda._sleep(1 << 40)
            try:
                check(None)
                sys.exit(1)
            except inprocess.exception.TimeoutError:
                sys.exit(0)

        exitcode, elapsed = self.launch(run)
        self.assertEqual(exitcode, 0)
        self.assertLess(elapsed, 10)

    @unittest.mock.patch.object(threading, 'excepthook', new=lambda _: None)
    def test_raises(self):
        def run():
            check = inprocess.health_check.CudaHealthCheck(datetime.timedelta(seconds=5))
            b = torch.ones(1, dtype=torch.int64).cuda()
            a = torch.ones(1, dtype=torch.int64).cuda()
            a[b] = 0
            try:
                check(None)
                sys.exit(1)
            except RuntimeError as ex:
                if 'CUDA' in str(ex):
                    sys.exit(0)
                sys.exit(1)

        exitcode, elapsed = self.launch(run)
        self.assertEqual(exitcode, 0)
        self.assertLess(elapsed, 10)
