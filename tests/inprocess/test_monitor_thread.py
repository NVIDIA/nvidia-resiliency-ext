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

import gc
import sys
import threading
import time
import unittest

import nvidia_resiliency_ext.inprocess as inprocess


class TestException(Exception):
    pass


class TestAsyncRaise(unittest.TestCase):
    def test(self):
        def run():
            inprocess.monitor_thread.async_raise(threading.main_thread().ident, TestException)

        thread = threading.Thread(target=run)
        with self.assertRaises(TestException):
            thread.start()
            thread.join()

    def test_nonstop_one_try(self, iterations=10):
        def run(start_event, event):
            start_event.wait()
            while not event.is_set():
                inprocess.monitor_thread.async_raise(threading.main_thread().ident, TestException)

        for _ in range(iterations):
            stop_event = threading.Event()
            start_event = threading.Event()
            thread = threading.Thread(target=run, args=(start_event, stop_event))
            thread.start()

            try:
                start_event.set()
                while True:
                    time.sleep(1e-6)
            except TestException:
                try:
                    stop_event.set()
                    thread.join()
                except TestException:
                    pass
                finally:
                    thread.join()

            self.assertFalse(thread.is_alive())

            for i in range(100):
                time.sleep(1e-6)

    def test_nonstop_while_true(self, iterations=10):
        def run(start_event, event):
            start_event.wait()
            while not event.is_set():
                inprocess.monitor_thread.async_raise(threading.main_thread().ident, TestException)

        for _ in range(iterations):
            stop_event = threading.Event()
            start_event = threading.Event()
            thread = threading.Thread(target=run, args=(start_event, stop_event))
            thread.start()

            try:
                start_event.set()
                while True:
                    time.sleep(1e-6)
            except TestException:
                while True:
                    try:
                        stop_event.set()
                        thread.join()
                        break
                    except TestException:
                        pass
                    finally:
                        thread.join()

            self.assertFalse(thread.is_alive())

            for i in range(100):
                time.sleep(1e-6)


class TestAsyncReRaise(unittest.TestCase):
    @unittest.mock.patch.object(sys, 'unraisablehook', new=lambda *args, **kwargs: None)
    def test_context_manager(self):
        class Foo:
            def __del__(self):
                raise TestException

        original_hook = sys.unraisablehook

        with inprocess.monitor_thread.reraise_if_unraisable(TestException):
            foo = Foo()
            with self.assertRaises(TestException):
                del foo
                gc.collect()
                while True:
                    time.sleep(1e-4)

        self.assertEqual(original_hook, sys.unraisablehook)

    @unittest.mock.patch.object(sys, 'unraisablehook', new=lambda *args, **kwargs: None)
    def test_decorator(self):
        class Foo:
            def __del__(self):
                raise TestException

        @inprocess.monitor_thread.reraise_if_unraisable(TestException)
        def run():
            foo = Foo()
            with self.assertRaises(TestException):
                del foo
                gc.collect()
                while True:
                    time.sleep(1e-4)

        original_hook = sys.unraisablehook
        run()
        self.assertEqual(original_hook, sys.unraisablehook)
