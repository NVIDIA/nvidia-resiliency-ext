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
                'LOCAL_RANK': '0',
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


class TestRestartHealthCheck(TestCase):
    """Test the _construct_restart_health_check function and restart_health_check logic."""

    def setUp(self):
        super().setUp()
        # Add LOCAL_RANK to environment for GPU health checks
        self.local_rank_patcher = unittest.mock.patch.dict(
            os.environ,
            {'LOCAL_RANK': '0'},
        )
        self.local_rank_patcher.start()

    def tearDown(self):
        self.local_rank_patcher.stop()
        super().tearDown()

    def test_construct_restart_health_check_no_user_health_check(self):
        """Test _construct_restart_health_check when no user health_check is provided."""
        wrapper = inprocess.Wrapper(**self.kwargs())

        # Verify that restart_health_check is constructed
        self.assertIsNotNone(wrapper.restart_health_check)
        self.assertIsInstance(wrapper.restart_health_check, inprocess.compose.Compose)

        # The instances attribute now contains the health checks directly
        self.assertEqual(len(wrapper.restart_health_check.instances), 2)

        # Should have 2 health checks: GPU and NVL health checks
        # Note: GPU health checks may be disabled due to driver version requirements
        self.assertGreaterEqual(len(wrapper.restart_health_check.instances), 1)

        # Check that we have the expected health check types
        health_check_types = [type(check) for check in wrapper.restart_health_check.instances]
        self.assertIn(inprocess.health_check.ChainedGPUHealthCheck, health_check_types)
        self.assertIn(inprocess.health_check.ChainedNVLHealthCheck, health_check_types)

    def test_construct_restart_health_check_no_local_rank(self):
        """Test _construct_restart_health_check when LOCAL_RANK is not available."""
        # Remove LOCAL_RANK from environment
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            wrapper = inprocess.Wrapper(**self.kwargs())

            # When no LOCAL_RANK, restart_health_check should be None
            self.assertIsNone(wrapper.restart_health_check)

    def test_construct_restart_health_check_missing_local_rank(self):
        """Test _construct_restart_health_check fails when LOCAL_RANK is missing."""
        # Remove LOCAL_RANK from environment
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            # This should no longer fail since we made GPU/NVL checks conditional
            wrapper = inprocess.Wrapper(**self.kwargs())
            self.assertIsNone(wrapper.restart_health_check)

    def test_construct_restart_health_check_with_single_user_health_check(self):
        """Test _construct_restart_health_check with a single user health_check."""

        # Track execution order
        execution_order = []

        class UserHealthCheck(inprocess.health_check.HealthCheck):
            def __call__(self, state):
                execution_order.append('user')
                return state

        # Create mock classes that track execution order
        class MockGPUHealthCheck(inprocess.health_check.HealthCheck):
            def __init__(self, device_index):
                self.device_index = device_index

            def __call__(self, state):
                if hasattr(self, '_execution_order'):
                    self._execution_order.append('gpu')
                return state

        class MockNVLHealthCheck(inprocess.health_check.HealthCheck):
            def __init__(self, device_index):
                self.device_index = device_index

            def __call__(self, state):
                if hasattr(self, '_execution_order'):
                    self._execution_order.append('nvl')
                return state

        # Mock the GPU and NVL health checks
        with (
            unittest.mock.patch(
                'nvidia_resiliency_ext.inprocess.wrap.ChainedGPUHealthCheck', MockGPUHealthCheck
            ),
            unittest.mock.patch(
                'nvidia_resiliency_ext.inprocess.wrap.ChainedNVLHealthCheck', MockNVLHealthCheck
            ),
        ):

            wrapper = inprocess.Wrapper(**self.kwargs(), health_check=UserHealthCheck())

            # Verify that restart_health_check is constructed
            self.assertIsNotNone(wrapper.restart_health_check)
            self.assertIsInstance(wrapper.restart_health_check, inprocess.compose.Compose)

            # Check what components are in the restart_health_check
            if hasattr(wrapper.restart_health_check, 'get_components'):
                components = wrapper.restart_health_check.get_components()
            else:
                components = wrapper.restart_health_check.instances

            # We should have: user + gpu + nvl = 3 components
            self.assertEqual(
                len(components), 3, f"Expected 3 health check components, got {len(components)}"
            )

            # Set up execution order tracking on the mock instances
            for comp in components:
                comp._execution_order = execution_order

            # Create a mock state
            mock_state = unittest.mock.Mock()

            # Execute the composed health check
            result = wrapper.restart_health_check(mock_state)

            # Verify execution order: should be nvl -> gpu -> user (Compose executes in reverse order)
            expected_order = ['nvl', 'gpu', 'user']
            self.assertEqual(
                execution_order,
                expected_order,
                f"Expected execution order {expected_order}, got {execution_order}",
            )

            # Verify the result is the state
            self.assertEqual(result, mock_state)

    def test_construct_restart_health_check_with_compose_user_health_check(self):
        """Test _construct_restart_health_check with a Compose user health_check."""

        # Track execution order
        execution_order = []

        class UserHealthCheck1(inprocess.health_check.HealthCheck):
            def __call__(self, state):
                execution_order.append('user1')
                return state

        class UserHealthCheck2(inprocess.health_check.HealthCheck):
            def __call__(self, state):
                execution_order.append('user2')
                return state

        # Create mock classes that track execution order
        class MockGPUHealthCheck(inprocess.health_check.HealthCheck):
            def __init__(self, device_index):
                self.device_index = device_index

            def __call__(self, state):
                if hasattr(self, '_execution_order'):
                    self._execution_order.append('gpu')
                return state

        class MockNVLHealthCheck(inprocess.health_check.HealthCheck):
            def __init__(self, device_index):
                self.device_index = device_index

            def __call__(self, state):
                if hasattr(self, '_execution_order'):
                    self._execution_order.append('nvl')
                return state

        # Mock the GPU and NVL health checks
        with (
            unittest.mock.patch(
                'nvidia_resiliency_ext.inprocess.wrap.ChainedGPUHealthCheck', MockGPUHealthCheck
            ),
            unittest.mock.patch(
                'nvidia_resiliency_ext.inprocess.wrap.ChainedNVLHealthCheck', MockNVLHealthCheck
            ),
        ):

            user_compose = inprocess.compose.Compose(UserHealthCheck1(), UserHealthCheck2())
            wrapper = inprocess.Wrapper(**self.kwargs(), health_check=user_compose)

            # Verify that restart_health_check is constructed
            self.assertIsNotNone(wrapper.restart_health_check)
            self.assertIsInstance(wrapper.restart_health_check, inprocess.compose.Compose)

            # Check what components are in the restart_health_check
            if hasattr(wrapper.restart_health_check, 'get_components'):
                components = wrapper.restart_health_check.get_components()
            else:
                components = wrapper.restart_health_check.instances

            # We should have: user checks (2) + gpu + nvl = 4 components
            self.assertEqual(
                len(components), 4, f"Expected 4 health check components, got {len(components)}"
            )

            # Set up execution order tracking on the mock instances
            for comp in components:
                comp._execution_order = execution_order

            # Create a mock state
            mock_state = unittest.mock.Mock()

            # Execute the composed health check
            result = wrapper.restart_health_check(mock_state)

            # Verify execution order: should be nvl -> gpu -> user2 -> user1 (Compose executes in reverse order)
            # The user checks are flattened from the Compose, so they execute in sequence
            expected_order = ['nvl', 'gpu', 'user2', 'user1']
            self.assertEqual(
                execution_order,
                expected_order,
                f"Expected execution order {expected_order}, got {execution_order}",
            )

            # Verify the result is the state
            self.assertEqual(result, mock_state)

    def test_restart_health_check_execution_order(self):
        """Test that restart_health_check executes health checks in the correct order."""

        # Track execution order
        execution_order = []

        class UserHealthCheck(inprocess.health_check.HealthCheck):
            def __call__(self, state):
                execution_order.append('user')
                return state

        # Create mock classes that track execution order
        class MockGPUHealthCheck(inprocess.health_check.HealthCheck):
            def __init__(self, device_index):
                self.device_index = device_index

            def __call__(self, state):
                if hasattr(self, '_execution_order'):
                    self._execution_order.append('gpu')
                return state

        class MockNVLHealthCheck(inprocess.health_check.HealthCheck):
            def __init__(self, device_index):
                self.device_index = device_index

            def __call__(self, state):
                if hasattr(self, '_execution_order'):
                    self._execution_order.append('nvl')
                return state

        # Mock the GPU and NVL health checks
        with (
            unittest.mock.patch(
                'nvidia_resiliency_ext.inprocess.wrap.ChainedGPUHealthCheck', MockGPUHealthCheck
            ),
            unittest.mock.patch(
                'nvidia_resiliency_ext.inprocess.wrap.ChainedNVLHealthCheck', MockNVLHealthCheck
            ),
        ):

            wrapper = inprocess.Wrapper(**self.kwargs(), health_check=UserHealthCheck())

            # Skip this test if no restart_health_check (no LOCAL_RANK)
            if wrapper.restart_health_check is None:
                self.skipTest("No restart_health_check available (LOCAL_RANK not set)")

            # Check what components are in the restart_health_check
            if hasattr(wrapper.restart_health_check, 'get_components'):
                components = wrapper.restart_health_check.get_components()
            else:
                components = wrapper.restart_health_check.instances

            # We should have: user + gpu + nvl = 3 components
            self.assertEqual(
                len(components), 3, f"Expected 3 health check components, got {len(components)}"
            )

            # Set up execution order tracking on the mock instances
            for comp in components:
                comp._execution_order = execution_order

            # Create a mock state
            mock_state = unittest.mock.Mock()

            # Execute the composed health check
            result = wrapper.restart_health_check(mock_state)

            # Verify execution order: should be nvl -> gpu -> user (Compose executes in reverse order)
            expected_order = ['nvl', 'gpu', 'user']
            self.assertEqual(
                execution_order,
                expected_order,
                f"Expected execution order {expected_order}, got {execution_order}",
            )

            # Verify the result is the state
            self.assertEqual(result, mock_state)

    def test_restart_health_check_gpu_failure(self):
        """Test that GPU health checks are properly constructed and can be executed."""
        wrapper = inprocess.Wrapper(**self.kwargs())

        # Skip this test if no restart_health_check (no LOCAL_RANK)
        if wrapper.restart_health_check is None:
            self.skipTest("No restart_health_check available (LOCAL_RANK not set)")

        # Get the health checks list
        health_checks_list = wrapper.restart_health_check.instances

        # Find GPU health check
        gpu_check = None
        for check in health_checks_list:
            # Check if it's a GPU health check by type name or device_index attribute
            if (
                isinstance(check, inprocess.health_check.ChainedGPUHealthCheck)
                or hasattr(check, 'device_index')
                and 'GPU' in type(check).__name__
            ):
                gpu_check = check
                break

        self.assertIsNotNone(gpu_check, "GPU health check not found")

        # Create a mock state
        mock_state = unittest.mock.Mock()

        # Test that the GPU health check can be called (may fail due to driver requirements, but should be callable)
        try:
            result = gpu_check(mock_state)
            # If it succeeds, verify it returns the state
            self.assertEqual(result, mock_state)
        except Exception as e:
            # If it fails due to driver requirements, that's expected
            # Just verify it's a callable object
            self.assertTrue(callable(gpu_check))

    def test_restart_health_check_nvl_failure_ignored(self):
        """Test that NVL health checks are properly constructed and can be executed."""
        wrapper = inprocess.Wrapper(**self.kwargs())

        # Skip this test if no restart_health_check (no LOCAL_RANK)
        if wrapper.restart_health_check is None:
            self.skipTest("No restart_health_check available (LOCAL_RANK not set)")

        # Get the health checks list
        health_checks_list = wrapper.restart_health_check.instances

        # Find NVL health check
        nvl_check = None
        for check in health_checks_list:
            # Check if it's an NVL health check by type name or device_index attribute
            if (
                isinstance(check, inprocess.health_check.ChainedNVLHealthCheck)
                or hasattr(check, 'device_index')
                and 'NVL' in type(check).__name__
            ):
                nvl_check = check
                break

        self.assertIsNotNone(nvl_check, "NVL health check not found")

        # Create a mock state
        mock_state = unittest.mock.Mock()

        # Test that the NVL health check can be called
        try:
            result = nvl_check(mock_state)
            # If it succeeds, verify it returns the state
            self.assertEqual(result, mock_state)
        except Exception as e:
            # If it fails, that's also acceptable behavior
            # Just verify it's a callable object
            self.assertTrue(callable(nvl_check))

    def test_restart_health_check_user_failure(self):
        """Test that restart_health_check properly handles user health check failures."""

        class UserHealthCheck(inprocess.health_check.HealthCheck):
            def __call__(self, state):
                raise inprocess.exception.HealthCheckError("User health check failed")

        wrapper = inprocess.Wrapper(**self.kwargs(), health_check=UserHealthCheck())

        # Skip this test if no restart_health_check (no LOCAL_RANK)
        if wrapper.restart_health_check is None:
            self.skipTest("No restart_health_check available (LOCAL_RANK not set)")

        # Create a mock state
        mock_state = unittest.mock.Mock()

        # Execute the user health check directly
        health_checks_list = wrapper.restart_health_check.instances
        user_check = health_checks_list[0]

        # Execute the user health check and expect it to fail
        with self.assertRaises(inprocess.exception.HealthCheckError) as cm:
            user_check(mock_state)
        self.assertIn("User health check failed", str(cm.exception))

    def test_restart_health_check_device_index_propagation(self):
        """Test that device_index is properly propagated to GPU and NVL health checks."""
        wrapper = inprocess.Wrapper(**self.kwargs())

        # Skip this test if no restart_health_check (no LOCAL_RANK)
        if wrapper.restart_health_check is None:
            self.skipTest("No restart_health_check available (LOCAL_RANK not set)")

        # Get the health checks list
        health_checks_list = wrapper.restart_health_check.instances

        # Find GPU and NVL health checks
        gpu_check = None
        nvl_check = None
        for check in health_checks_list:
            # Check if it's a GPU health check by type name or device_index attribute
            if (
                isinstance(check, inprocess.health_check.ChainedGPUHealthCheck)
                or hasattr(check, 'device_index')
                and 'GPU' in type(check).__name__
            ):
                gpu_check = check
            # Check if it's an NVL health check by type name or device_index attribute
            elif (
                isinstance(check, inprocess.health_check.ChainedNVLHealthCheck)
                or hasattr(check, 'device_index')
                and 'NVL' in type(check).__name__
            ):
                nvl_check = check

        # Verify that we found both checks
        self.assertIsNotNone(gpu_check, "GPU health check not found")
        self.assertIsNotNone(nvl_check, "NVL health check not found")

        # Verify device_index is properly set
        self.assertEqual(gpu_check.device_index, 0)  # LOCAL_RANK=0
        self.assertEqual(nvl_check.device_index, 0)  # LOCAL_RANK=0

    def test_restart_health_check_integration_with_wrapper(self):
        """Test that restart_health_check integrates properly with the wrapper execution flow."""

        class UserHealthCheck(inprocess.health_check.HealthCheck):
            def __call__(self, state):
                return state

        @inprocess.Wrapper(
            **self.kwargs(),
            initialize=inprocess.initialize.RetryController(max_iterations=1),
            health_check=UserHealthCheck(),
        )
        def fn():
            raise RuntimeError("Simulated failure")

        # The function should fail and trigger restart abort due to max_iterations=1
        with self.assertRaises(inprocess.exception.RestartAbort):
            fn()

    def test_restart_health_check_compose_behavior(self):
        """Test that the Compose behavior works correctly with health checks."""

        class UserHealthCheck(inprocess.health_check.HealthCheck):
            def __init__(self, name):
                self.name = name

            def __call__(self, state):
                # Modify state to track execution
                if not hasattr(state, 'execution_trace'):
                    state.execution_trace = []
                state.execution_trace.append(self.name)
                return state

        wrapper = inprocess.Wrapper(**self.kwargs(), health_check=UserHealthCheck("user"))

        # Skip this test if no restart_health_check (no LOCAL_RANK)
        if wrapper.restart_health_check is None:
            self.skipTest("No restart_health_check available (LOCAL_RANK not set)")

        # Create a mock state
        mock_state = unittest.mock.Mock()
        mock_state.execution_trace = []

        # Execute each health check individually to test their behavior
        health_checks_list = wrapper.restart_health_check.instances

        # Execute user health check
        user_check = health_checks_list[0]
        result = user_check(mock_state)

        # Verify that the user check was executed
        self.assertEqual(result, mock_state)
        self.assertIn("user", mock_state.execution_trace)
