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

import dataclasses
import datetime
import gc
import io
import logging
import os
import tempfile
import threading
import unittest
import warnings
import weakref
from typing import Optional

import torch

import nvidia_resiliency_ext.inprocess as inprocess

from . import common


@common.apply_all_tests(common.retry())
class TestCase(unittest.TestCase):
    def setUp(self):
        self.patcher = unittest.mock.patch.dict(
            os.environ,
            {
                'RANK': '0',
                'WORLD_SIZE': '1',
                'MASTER_ADDR': 'localhost',
                'MASTER_PORT': str(common.find_free_port()),
            },
        )
        self.patcher.start()

        self.initial_threads = threading.enumerate()

        root_log = logging.getLogger()
        self.initial_num_root_log_handlers = len(root_log.handlers)

    def tearDown(self):
        self.patcher.stop()

        self.assertEqual(
            len(threading.enumerate()),
            len(self.initial_threads),
            threading.enumerate(),
        )
        self.assertEqual(threading.enumerate(), self.initial_threads)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stores = [obj for obj in gc.get_objects() if isinstance(obj, inprocess.store.TCPStore)]
        for store in stores:
            referrers = [ref for ref in gc.get_referrers(store)]
            print(f'{store=}')
            for idx, ref in enumerate(referrers):
                print(f'{idx=}, {ref=}')
        self.assertEqual(len(stores), 0, stores)

        root_log = logging.getLogger()
        self.assertEqual(
            len(root_log.handlers),
            self.initial_num_root_log_handlers,
            root_log.handlers,
        )

        inproc_log = logging.getLogger(inprocess.__name__)
        self.assertEqual(len(inproc_log.handlers), 0, inproc_log.handlers)

    def kwargs(self):
        return {
            'store_kwargs': {
                'port': common.find_free_port(),
                'timeout': datetime.timedelta(seconds=10),
            },
            'monitor_thread_interval': datetime.timedelta(seconds=1e-3),
            'monitor_process_interval': datetime.timedelta(seconds=1e-3),
            'progress_watchdog_interval': datetime.timedelta(seconds=1e-3),
            'last_call_wait': datetime.timedelta(seconds=1e-3),
            'termination_grace_time': datetime.timedelta(seconds=1e-3),
        }


class TestReturn(TestCase):
    def test_const(self):
        @inprocess.Wrapper(**self.kwargs())
        def fn():
            return 123

        self.assertEqual(fn(), 123)

    def test_passthrough(self):
        @inprocess.Wrapper(**self.kwargs())
        def fn(a, b, c, d=4):
            return a, b, c, d

        self.assertEqual(fn(1, b=2, **{'c': 3}), (1, 2, 3, 4))


class TestApply(TestCase):
    @staticmethod
    def fn_impl(call_wrapper):
        assert call_wrapper.iteration == 0
        call_wrapper.ping()
        with call_wrapper.atomic():
            call_wrapper.ping()
        return 0

    def test_decorator(self):
        @inprocess.Wrapper(**self.kwargs())
        def fn(call_wrapper: Optional[inprocess.CallWrapper] = None):
            return self.fn_impl(call_wrapper)

        self.assertEqual(fn(), 0)

    def test_overwrite(self):
        def fn(call_wrapper: Optional[inprocess.CallWrapper] = None):
            return self.fn_impl(call_wrapper)

        fn = inprocess.Wrapper(**self.kwargs())(fn)
        self.assertEqual(fn(), 0)

    def test_assign(self):
        def fn(call_wrapper: Optional[inprocess.CallWrapper] = None):
            return self.fn_impl(call_wrapper)

        wrapped = inprocess.Wrapper(**self.kwargs())(fn)
        self.assertEqual(wrapped(), 0)


class TestInit(TestCase):
    @unittest.mock.patch.dict(
        os.environ,
        {
            'RANK': '0',
            'WORLD_SIZE': '1',
            'MASTER_ADDR': 'localhost',
            'MASTER_PORT': str(common.find_free_port()),
        },
    )
    def test_distributed(self):
        max_iterations = 2

        @inprocess.Wrapper(
            initialize=inprocess.initialize.RetryController(max_iterations=max_iterations),
            **self.kwargs(),
        )
        def fn():
            pass

        torch.distributed.init_process_group('gloo')
        with self.assertRaisesRegex(
            ValueError,
            'not torch.distributed.is_initialized()',
        ):
            fn()
        torch.distributed.destroy_process_group()


class TestException(TestCase):
    def test_max_iterations(self):
        counter = 0
        max_iterations = 4

        @inprocess.Wrapper(
            initialize=inprocess.initialize.RetryController(max_iterations=max_iterations),
            **self.kwargs(),
        )
        def fn():
            nonlocal counter
            counter += 1
            raise RuntimeError

        with self.assertRaises(inprocess.exception.RestartAbort):
            fn()
        self.assertEqual(counter, max_iterations)

    def test_success(self):
        counter = 0
        max_iterations = 4
        ret_val = 123

        @inprocess.Wrapper(
            initialize=inprocess.initialize.RetryController(max_iterations=max_iterations),
            **self.kwargs(),
        )
        def fn():
            nonlocal counter
            counter += 1
            if counter == max_iterations:
                return ret_val
            else:
                raise RuntimeError

        ret = fn()
        self.assertEqual(counter, max_iterations)
        self.assertEqual(ret, ret_val)

    def test_faulty_store_factory(self):
        class Store(inprocess.store.StoreMixin):
            def __init__(self):
                raise ZeroDivisionError

        @inprocess.Wrapper(
            store_factory=Store,
            initialize=inprocess.Compose(
                inprocess.initialize.RetryController(max_iterations=2),
            ),
        )
        def fn():
            pass

        with self.assertRaises(ZeroDivisionError):
            fn()

    def test_faulty_initialize(self):
        class Initialize(inprocess.initialize.Initialize):
            def __call__(self, state):
                raise ZeroDivisionError

        @inprocess.Wrapper(
            initialize=inprocess.Compose(
                Initialize(),
                inprocess.initialize.RetryController(max_iterations=2),
            ),
            **self.kwargs(),
        )
        def fn():
            pass

        with self.assertRaises(inprocess.exception.RestartAbort):
            fn()

    def test_faulty_abort(self):
        class Abort(inprocess.abort.Abort):
            def __call__(self, state):
                raise ZeroDivisionError

        @inprocess.Wrapper(
            initialize=inprocess.Compose(
                inprocess.initialize.RetryController(max_iterations=2),
            ),
            **self.kwargs(),
        )
        def fn():
            raise RuntimeError

        with self.assertRaises(inprocess.exception.RestartAbort):
            fn()

    def test_faulty_finalize(self):
        class Finalize(inprocess.finalize.Finalize):
            def __call__(self, state):
                raise ZeroDivisionError

        @inprocess.Wrapper(
            initialize=inprocess.initialize.RetryController(max_iterations=2),
            finalize=Finalize(),
            **self.kwargs(),
        )
        def fn():
            raise RuntimeError

        with self.assertRaises(ZeroDivisionError):
            fn()

    def test_unhealthy(self):
        class HealthCheck(inprocess.health_check.HealthCheck):
            def __call__(self, state):
                raise ZeroDivisionError

        @inprocess.Wrapper(**self.kwargs(), health_check=HealthCheck())
        def fn():
            raise RuntimeError

        with self.assertRaises(inprocess.exception.HealthCheckError):
            fn()

    def test_transient_unhealthy(self):
        class HealthCheck(inprocess.health_check.HealthCheck):
            def __init__(self):
                self.fail = True

            def __call__(self, state):
                if self.fail:
                    self.fail = False
                    raise ZeroDivisionError
                return state

        @inprocess.Wrapper(**self.kwargs(), health_check=HealthCheck())
        def fn():
            return

        fn()

    def test_faulty_rank_assignment(self):
        class RankAssignment(inprocess.rank_assignment.RankAssignment):
            def __call__(self, ctx):
                raise ZeroDivisionError

        @inprocess.Wrapper(
            initialize=inprocess.initialize.RetryController(max_iterations=2),
            rank_assignment=RankAssignment(),
            **self.kwargs(),
        )
        def fn():
            raise RuntimeError

        with self.assertRaises(ZeroDivisionError):
            fn()

    def test_faulty_second_rank_assignment(self):
        class RankAssignment(inprocess.rank_assignment.RankAssignment):
            def __init__(self):
                self.should_raise = False

            def __call__(self, ctx):
                if self.should_raise:
                    raise ZeroDivisionError
                else:
                    self.should_raise = True
                    return ctx

        @inprocess.Wrapper(
            initialize=inprocess.initialize.RetryController(max_iterations=2),
            rank_assignment=RankAssignment(),
            **self.kwargs(),
        )
        def fn():
            raise RuntimeError

        with self.assertRaises(ZeroDivisionError):
            fn()

    @common.silence_deprecation_warnings()
    def test_faulty_rank_filter(self):
        class RankFilter(inprocess.rank_filter.RankFilter):
            def __call__(self, state):
                raise ZeroDivisionError

        @inprocess.Wrapper(
            initialize=inprocess.initialize.RetryController(max_iterations=2),
            rank_filter=RankFilter(),
            **self.kwargs(),
        )
        def fn():
            raise RuntimeError

        with self.assertRaises(ZeroDivisionError):
            fn()

    @common.silence_deprecation_warnings()
    def test_faulty_second_rank_filter(self):
        class RankFilter(inprocess.rank_filter.RankFilter):
            def __init__(self):
                self.should_raise = False

            def __call__(self, state):
                if self.should_raise:
                    raise ZeroDivisionError
                else:
                    self.should_raise = True
                    return state

        @inprocess.Wrapper(
            initialize=inprocess.initialize.RetryController(max_iterations=2),
            rank_filter=RankFilter(),
            **self.kwargs(),
        )
        def fn():
            raise RuntimeError

        with self.assertRaises(ZeroDivisionError):
            fn()

    def test_explicit_restart(self):
        @inprocess.Wrapper(
            initialize=inprocess.initialize.RetryController(max_iterations=2),
            **self.kwargs(),
        )
        def fn():
            raise inprocess.monitor_thread.RankShouldRestart

        with self.assertRaises(inprocess.exception.RestartAbort):
            fn()

    def test_system_exit(self):
        @inprocess.Wrapper(**self.kwargs())
        def fn():
            raise SystemExit

        with self.assertRaises(SystemExit):
            fn()

    def test_keyboard_interrupt(self):
        @inprocess.Wrapper(**self.kwargs())
        def fn():
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            fn()


class TestState(TestCase):
    def test_frozen_initialize(self):
        class Initialize(inprocess.initialize.Initialize):
            def __call__(self, state):
                state.rank = 0
                return state

        @inprocess.Wrapper(
            initialize=inprocess.Compose(
                Initialize(),
                inprocess.initialize.RetryController(max_iterations=1),
            ),
            **self.kwargs(),
        )
        def fn():
            pass

        with self.assertRaises(inprocess.exception.RestartAbort):
            fn()

    def test_frozen_finalize(self):
        class Finalize(inprocess.finalize.Finalize):
            def __call__(self, state):
                state.rank = 0
                return state

        @inprocess.Wrapper(
            initialize=inprocess.initialize.RetryController(max_iterations=1),
            finalize=Finalize(),
            **self.kwargs(),
        )
        def fn():
            raise RuntimeError

        with self.assertRaises(dataclasses.FrozenInstanceError):
            fn()

    def test_frozen_health_check(self):
        class HealthCheck(inprocess.health_check.HealthCheck):
            def __call__(self, state):
                state.rank = 0
                return state

        @inprocess.Wrapper(
            **self.kwargs(),
            initialize=inprocess.initialize.RetryController(max_iterations=1),
            health_check=HealthCheck(),
        )
        def fn():
            raise RuntimeError

        with self.assertRaises(inprocess.exception.HealthCheckError):
            fn()


class TestMultiCall(TestCase):
    def test_same_fn(self):
        @inprocess.Wrapper(**self.kwargs())
        def fn():
            return 123

        for _ in range(2):
            self.assertEqual(fn(), 123)

    def test_same_fn_error(self):
        max_iterations = 3
        iteration = None

        @inprocess.Wrapper(
            **self.kwargs(),
            initialize=inprocess.initialize.RetryController(max_iterations=max_iterations),
        )
        def fn(wrapped: inprocess.CallWrapper = None):
            nonlocal iteration
            iteration = wrapped.iteration
            raise ZeroDivisionError

        def run():
            with self.assertRaises(inprocess.exception.RestartAbort):
                fn()
            self.assertEqual(iteration, max_iterations - 1)

        for _ in range(2):
            run()

    def test_different(self):
        @inprocess.Wrapper(**self.kwargs())
        def fn():
            return 123

        @inprocess.Wrapper(**self.kwargs())
        def gn():
            return 456

        self.assertEqual(fn(), 123)
        self.assertEqual(gn(), 456)


class TestLogging(TestCase):
    def setUp(self):
        super().setUp()

        root = logging.getLogger()
        self.original_level = root.level
        self.original_handlers = root.handlers[:]
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    def tearDown(self):
        root = logging.getLogger()
        root.level = self.original_level
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        for handler in self.original_handlers:
            root.addHandler(handler)

        super().tearDown()

    def test_not_initialized(self):
        @inprocess.Wrapper(**self.kwargs())
        def fn():
            return

        with self.assertWarns(UserWarning):
            fn()

        logger = logging.getLogger(inprocess.wrap.__name__)
        self.assertEqual(logger.handlers, [])

    def test_logging(self):
        @inprocess.Wrapper(**self.kwargs())
        def fn():
            return

        stream = io.StringIO()
        logging.basicConfig(level=logging.DEBUG, stream=stream, force=True)

        fn()
        self.assertGreater(len(stream.getvalue()), 0)

    def test_logging_twice(self):
        @inprocess.Wrapper(**self.kwargs())
        def fn():
            return

        first_stream = io.StringIO()
        logging.basicConfig(level=logging.INFO, stream=first_stream)
        fn()
        first = first_stream.getvalue()

        second_stream = io.StringIO()
        logging.basicConfig(level=logging.INFO, stream=second_stream, force=True)
        fn()
        first_again = first_stream.getvalue()
        second = second_stream.getvalue()

        self.assertEqual(len(first), len(first_again))
        self.assertEqual(len(first), len(second))

    def test_monitor_process(self):
        root = logging.getLogger()
        root.level = logging.INFO

        with tempfile.NamedTemporaryFile('r') as tmp_file:

            @inprocess.Wrapper(
                **self.kwargs(),
                monitor_process_logfile=tmp_file.name,
            )
            def fn():
                return

            with self.assertWarns(UserWarning):
                fn()

            with open(tmp_file.name, mode='r') as fp:
                data = fp.read()

            self.assertTrue(data, data)
            self.assertIn('target_pid', data, data)
            self.assertIn('monitor_process', data, data)

    def test_monitor_process_pidfile(self):
        """Test that monitor_process_pidfile parameter works correctly."""
        with tempfile.NamedTemporaryFile('r') as tmp_file:
            pid_file_path = tmp_file.name

            @inprocess.Wrapper(
                **self.kwargs(),
                monitor_process_pidfile=pid_file_path,
            )
            def fn():
                return

            with self.assertWarns(UserWarning):
                fn()

            # Check that the PID file was created and contains a valid PID
            self.assertTrue(os.path.exists(pid_file_path))
            with open(pid_file_path, 'r') as fp:
                pid_content = fp.read().strip()
                self.assertTrue(pid_content.isdigit())
                pid = int(pid_content)
                self.assertGreater(pid, 0)

    def test_monitor_process_pidfile_with_rank_placeholder(self):
        """Test that monitor_process_pidfile works with {rank} placeholder."""
        with tempfile.NamedTemporaryFile('r') as tmp_file:
            pid_file_path = tmp_file.name + "_{rank}.pid"

            @inprocess.Wrapper(
                **self.kwargs(),
                monitor_process_pidfile=pid_file_path,
            )
            def fn():
                return

            with self.assertWarns(UserWarning):
                fn()

            # Check that the PID file was created with rank substitution
            expected_pid_file = pid_file_path.format(rank=0)  # Assuming rank 0
            self.assertTrue(os.path.exists(expected_pid_file))
            with open(expected_pid_file, 'r') as fp:
                pid_content = fp.read().strip()
                self.assertTrue(pid_content.isdigit())
                pid = int(pid_content)
                self.assertGreater(pid, 0)

            # Clean up
            if os.path.exists(expected_pid_file):
                os.remove(expected_pid_file)


class TestTCPStore(TestCase):
    def test_reinit_clears_keys(self):
        @inprocess.Wrapper(
            initialize=inprocess.initialize.RetryController(max_iterations=3),
            **self.kwargs(),
        )
        def fn():
            hidden = 8
            model = torch.nn.Linear(hidden, hidden)

            torch.distributed.init_process_group('gloo')
            model = torch.nn.parallel.DistributedDataParallel(model)
            store_ref = weakref.ref(torch.distributed.distributed_c10d._get_default_store())
            once = store_ref().add('key', 1)
            self.assertEqual(once, 1)

            for i in range(5):
                inp = torch.rand(hidden, hidden)
                model.zero_grad()
                out = model(inp)
                loss = out.square().mean()
                loss.backward()

            raise ZeroDivisionError

        with self.assertRaisesRegex(inprocess.exception.RestartAbort, 'iteration=3'):
            fn()
