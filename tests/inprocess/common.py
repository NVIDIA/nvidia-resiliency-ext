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

import collections
import contextlib
import faulthandler
import functools
import inspect
import itertools
import logging
import multiprocessing
import os
import random
import socket
import sys
import tempfile
import threading
import time
import traceback
import types
import unittest
from enum import Enum
from functools import wraps
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import torch

if level := os.getenv('LOG', None):
    loglevel = getattr(logging, level.upper())
else:
    loglevel = logging.CRITICAL + 1

logging.basicConfig(
    level=loglevel,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
)
logger = logging.getLogger(__name__)

os.environ['NCCL_NVLS_ENABLE'] = '0'
os.environ['NCCL_NET_PLUGIN'] = '"none"'


def find_free_port(host='localhost'):
    with contextlib.closing(
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        _, port = sock.getsockname()
        return port


def is_port_available(port, host='localhost'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def wait_for_port(port, host='localhost', timeout=5, check_interval=0.1):
    start_time = time.time()

    while True:
        if is_port_available(port, host):
            return True

        if timeout and (time.time() - start_time) > timeout:
            return False

        time.sleep(check_interval)


def wrap_store(store, *args):
    prefix = '_'.join(str(arg) for arg in args)
    return torch.distributed.PrefixStore(prefix, store)


def unroll_generators(params):
    unrolled = []
    for param in params:
        generators = {k: v for k, v in param.items() if inspect.isgenerator(v)}
        if generators:
            generator_keys = generators.keys()
            for gen_vals in itertools.product(*generators.values()):
                update = zip(generator_keys, gen_vals)
                current_param = param.copy()
                current_param.update(update)
                unrolled.append(current_param)
        else:
            unrolled.append(param)
    return unrolled


def subtests(params):
    r'''
    A decorator to specify hyperparameters for test cases.

    It accepts a list of dictionaries containing keyword args for a decorated
    unit test method.

    Args:
        params: list of dictionaries with test hyperparameters

    Examples::

        @subtests(
            [
                {
                    'm': 1024,
                    'n': 1024,
                    'k': k,
                    'transpose_a': ta,
                    'transpose_b': tb,
                    'dtype': torch.float16,
                }
                for ta, tb in itertools.product([False, True], repeat=2)
                for k in [512, 1024]
            ]
        )
        def test_op(self, m, n, k, transpose_a, transpose_b, dtype):
            ...
    '''

    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            self, *rest = args
            signature = inspect.signature(func)

            for param in params:
                with self.subTest(**param):
                    current_kwargs = kwargs.copy()
                    current_kwargs.update(param)

                    if os.getenv('VERBOSE', False):
                        test = (
                            self._subtest
                            if self._subtest is not None
                            else self
                        )
                        print(test)

                    func(*args, **current_kwargs)

        return inner_wrapper

    return outer_wrapper


def instantiate_parametrized_tests(generic_cls):
    '''
    Instantiates tests that have been decorated with a parametrize_fn. This is
    generally performed by a decorator subclass of _TestParametrizer. The
    generic test will be replaced on the test class by parametrized tests with
    specialized names. This should be used instead of
    instantiate_device_type_tests() if the test class contains device-agnostic
    tests.

    You can also use it as a class decorator. E.g.

    ```
    @instantiate_parametrized_tests
    class TestFoo(TestCase):
        ...
    ```

    Args:
        generic_cls (class): Generic test class object containing tests (e.g.
            TestFoo)
    '''
    for attr_name in tuple(dir(generic_cls)):
        class_attr = getattr(generic_cls, attr_name)
        if not hasattr(class_attr, 'parametrize_fn'):
            continue

        # Remove the generic test from the test class.
        delattr(generic_cls, attr_name)

        # Add parametrized tests to the test class.
        def instantiate_test_helper(cls, name, test, param_kwargs):
            @wraps(test)
            def instantiated_test(self, param_kwargs=param_kwargs):
                test(self, **param_kwargs)

            assert not hasattr(
                generic_cls, name
            ), f'Redefinition of test {name}'
            setattr(generic_cls, name, instantiated_test)

        for (
            test,
            test_suffix,
            param_kwargs,
            decorator_fn,
        ) in class_attr.parametrize_fn(
            class_attr, generic_cls=generic_cls, device_cls=None
        ):
            full_name = f'{test.__name__}_{test_suffix}'

            # Apply decorators based on full param kwargs.
            for decorator in decorator_fn(param_kwargs):
                test = decorator(test)

            instantiate_test_helper(
                cls=generic_cls,
                name=full_name,
                test=test,
                param_kwargs=param_kwargs,
            )
    return generic_cls


class _TestParametrizer:
    '''
    Decorator class for parametrizing a test function, yielding a set of new
    tests spawned from the original generic test, each specialized for a
    specific set of test inputs. For example, parametrizing a test across the
    set of ops will result in a test function per op.

    The decision of how to parametrize / what to parametrize over is intended
    to be implemented by each derived class.

    In the details, the decorator adds a 'parametrize_fn' property to the test
    function. This function is intended to be called later by one of:
      * Device-specific test instantiation via instantiate_device_type_tests().
      Note that for this case there is no need to explicitly parametrize over
      device type, as that is handled separately.
      * Device-agnostic parametrized test instantiation via
      instantiate_parametrized_tests().

    If the decorator is applied to a test function that already has a
    'parametrize_fn' property, a new composite 'parametrize_fn' will be created
    that generates tests with the product of the parameters generated by the
    old and new parametrize_fns. This allows for convenient composability of
    decorators.
    '''

    def _parametrize_test(self, test, generic_cls, device_cls):
        '''
        Parametrizes the given test function across whatever dimension is
        specified by the derived class. Tests can be parametrized over any
        arbitrary dimension or combination of dimensions, such as all ops, all
        modules, or all ops + their associated dtypes.

        Args:
            test (fn): Test function to parametrize over
            generic_cls (class): Generic test class object containing tests
                (e.g. TestFoo)
            device_cls (class): Device-specialized test class object (e.g.
                TestFooCPU); set to None if the tests are not part of a
                device-specific set

        Returns:
            Generator object returning 4-tuples of:
                test (fn): Parametrized test function; must support a device
                    arg and args for any params
                test_name (str): Parametrized suffix for the test (e.g.
                    opname_int64); will be appended to the base name of the
                    test
                param_kwargs (dict): Param kwargs to pass to the test (e.g.
                    {'op': 'add', 'dtype': torch.int64})
                decorator_fn (callable): Callable[[Dict], List] for list of
                    decorators to apply given param_kwargs
        '''
        raise NotImplementedError

    def __call__(self, fn):
        if hasattr(fn, 'parametrize_fn'):
            # Do composition with the product of args.
            old_parametrize_fn = fn.parametrize_fn
            new_parametrize_fn = self._parametrize_test
            fn.parametrize_fn = compose_parametrize_fns(
                old_parametrize_fn, new_parametrize_fn
            )
        else:
            fn.parametrize_fn = self._parametrize_test
        return fn


def compose_parametrize_fns(old_parametrize_fn, new_parametrize_fn):
    '''
    Returns a parametrize_fn that parametrizes over the product of the
    parameters handled by the given parametrize_fns. Each given parametrize_fn
    should each have the signature f(test, generic_cls, device_cls).

    The test names will be a combination of the names produced by the
    parametrize_fns in '<new_name>_<old_name>' order. This order is done to
    match intuition for constructed names when composing multiple decorators;
    the names will be built in top to bottom order when stacking
    parametrization decorators.

    Args:
        old_parametrize_fn (callable) - First parametrize_fn to compose.
        new_parametrize_fn (callable) - Second parametrize_fn to compose.
    '''

    def composite_fn(
        test,
        generic_cls,
        device_cls,
        old_parametrize_fn=old_parametrize_fn,
        new_parametrize_fn=new_parametrize_fn,
    ):
        old_tests = list(old_parametrize_fn(test, generic_cls, device_cls))
        for old_test, old_test_name, old_param_kwargs, old_dec_fn in old_tests:
            for (
                new_test,
                new_test_name,
                new_param_kwargs,
                new_dec_fn,
            ) in new_parametrize_fn(old_test, generic_cls, device_cls):
                redundant_params = set(old_param_kwargs.keys()).intersection(
                    new_param_kwargs.keys()
                )
                if redundant_params:
                    raise RuntimeError(
                        'Parametrization over the same parameter by multiple '
                        'parametrization decorators is not supported. For '
                        'test "{}", the following parameters are handled '
                        'multiple times: {}'.format(
                            test.__name__, redundant_params
                        )
                    )
                full_param_kwargs = {**old_param_kwargs, **new_param_kwargs}
                merged_test_name = '{}{}{}'.format(
                    new_test_name,
                    '_' if old_test_name != '' and new_test_name != '' else '',
                    old_test_name,
                )

                def merged_decorator_fn(
                    param_kwargs, old_dec_fn=old_dec_fn, new_dec_fn=new_dec_fn
                ):
                    return list(old_dec_fn(param_kwargs)) + list(
                        new_dec_fn(param_kwargs)
                    )

                yield (
                    new_test,
                    merged_test_name,
                    full_param_kwargs,
                    merged_decorator_fn,
                )

    return composite_fn


class subtest:
    '''
    Explicit subtest case for use with test parametrization. Allows for
    explicit naming of individual subtest cases as well as applying decorators
    to the parametrized test.

    Args:
        arg_values (iterable): Iterable of arg values (e.g. range(10)) or
            tuples of arg values (e.g. [(1, 2), (3, 4)]).
        name (str): Optional name to use for the test.
        decorators (iterable): Iterable of decorators to apply to the generated
            test.
    '''

    __slots__ = ['arg_values', 'name', 'decorators']

    def __init__(self, arg_values, name=None, decorators=None):
        self.arg_values = arg_values
        self.name = name
        self.decorators = decorators if decorators else []


def dtype_name(dtype):
    """Returns the pretty name of the dtype (e.g. torch.int64 -> int64)."""
    return str(dtype).split('.')[1]


class parametrize(_TestParametrizer):
    '''
    Decorator for applying generic test parametrizations.

    The interface for this decorator is modeled after
    `@pytest.mark.parametrize`. Basic usage between this decorator and pytest's
    is identical. The first argument should be a string containing
    comma-separated names of parameters for the test, and the second argument
    should be an iterable returning values or tuples of values for the case of
    multiple parameters.

    Beyond this basic usage, the decorator provides some additional
    functionality that pytest does not.

    1. Parametrized tests end up as generated test functions on unittest test
    classes. Since this differs from how pytest works, this decorator takes on
    the additional responsibility of naming these test functions. The default
    test names consists of the test's base name followed by each parameter name
    + value (e.g. "test_bar_x_1_y_foo"), but custom names can be defined using
    `name_fn` or the `subtest` structure (see below).

    2. The decorator specially handles parameter values of type `subtest`,
    which allows for more fine-grained control over both test naming and test
    execution. In particular, it can be used to tag subtests with explicit test
    names or apply arbitrary decorators (see examples below).

    Examples::

        @parametrize('x', range(5))
        def test_foo(self, x):
            ...

        @parametrize('x,y', [(1, 'foo'), (2, 'bar'), (3, 'baz')])
        def test_bar(self, x, y):
            ...

        @parametrize('x,y', [(1, 'foo'), (2, 'bar'), (3, 'baz')],
                     name_fn=lambda x, y: '{}_{}'.format(x, y))
        def test_bar_custom_names(self, x, y):
            ...

        @parametrize(
            'x, y',
            [
                subtest((1, 2), name='double'),
                subtest((1, 3), name='triple',
                        decorators=[unittest.expectedFailure]
                        ),
                subtest((1, 4), name='quadruple'),
            ],
        )
        def test_baz(self, x, y):
            ...

    To actually instantiate the parametrized tests, one of
    instantiate_parametrized_tests() or instantiate_device_type_tests() should
    be called. The former is intended for test classes that contain
    device-agnostic tests, while the latter should be used for test classes
    that contain device-specific tests. Both support arbitrary parametrizations
    using the decorator.

    Args:
        arg_str (str): String of arg names separate by commas (e.g. 'x,y').
        arg_values (iterable): Iterable of arg values (e.g. range(10)) or
            tuples of arg values (e.g. [(1, 2), (3, 4)]).
        name_fn (Callable): Optional function that takes in parameters and
            returns subtest name.
    '''

    def __init__(self, arg_str, arg_values, name_fn=None):
        self.arg_names: List[str] = [
            s.strip() for s in arg_str.split(',') if s != ''
        ]
        self.arg_values = arg_values
        self.name_fn = name_fn

    def _formatted_str_repr(self, idx, name, value):
        '''
        Returns a string representation for the given arg that is suitable
        for use in test function names.
        '''
        if isinstance(value, torch.dtype):
            return dtype_name(value)
        elif isinstance(value, torch.device):
            return str(value)
        # Can't use isinstance as it would cause a circular import
        elif type(value).__name__ in {'OpInfo', 'ModuleInfo'}:
            return value.formatted_name
        elif isinstance(value, (int, float, str)):
            return f'{name}_{str(value).replace(".", "_")}'
        else:
            return f'{name}{idx}'

    def _default_subtest_name(self, idx, values):
        return '_'.join(
            [
                self._formatted_str_repr(idx, a, v)
                for a, v in zip(self.arg_names, values)
            ]
        )

    def _get_subtest_name(self, idx, values, explicit_name=None):
        if explicit_name:
            subtest_name = explicit_name
        elif self.name_fn:
            subtest_name = self.name_fn(*values)
        else:
            subtest_name = self._default_subtest_name(idx, values)
        return subtest_name

    def _parametrize_test(self, test, generic_cls, device_cls):
        if len(self.arg_names) == 0:
            # No additional parameters needed for the test.
            test_name = ''
            yield (test, test_name, {}, lambda _: [])
        else:
            # Each 'values' item is expected to be either:
            # * A tuple of values with one for each arg. For a single arg, a
            #   single item is expected.
            # * A subtest instance with arg_values matching the previous.
            values = check_exhausted_iterator = object()
            for idx, values in enumerate(self.arg_values):
                maybe_name = None

                decorators = []
                if isinstance(values, subtest):
                    sub = values
                    values = sub.arg_values
                    maybe_name = sub.name

                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        return test(*args, **kwargs)

                    decorators = sub.decorators
                    gen_test = test_wrapper
                else:
                    gen_test = test

                values = list(values) if len(self.arg_names) > 1 else [values]
                if len(values) != len(self.arg_names):
                    raise RuntimeError(
                        f'Expected # values == # arg names, but got: '
                        f'{len(values)} values and {len(self.arg_names)} '
                        f'names for test "{test.__name__}"'
                    )

                param_kwargs = dict(zip(self.arg_names, values))

                test_name = self._get_subtest_name(
                    idx, values, explicit_name=maybe_name
                )

                def decorator_fn(_, decorators=decorators):
                    return decorators

                yield (gen_test, test_name, param_kwargs, decorator_fn)

            if values is check_exhausted_iterator:
                raise ValueError(
                    f'{test}: An empty arg_values was passed to @parametrize. '
                    'Note that this may result from reuse of a generator.'
                )


class MultiProcessTestCase(unittest.TestCase):
    DEFAULT_TIMEOUT = 300
    MAIN_PROCESS_RANK = -1
    # This exit code is used to indicate that the test code had an error and
    # exited abnormally. There are certain tests that might use sys.exit() to
    # simulate failures and in those cases, we can't have an exit code of 0,
    # but we still want to ensure we didn't run into any other errors.
    TEST_ERROR_EXIT_CODE = 10

    @property
    def world_size(self) -> int:
        return 2

    def join_or_run(self, fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                self._join_processes(fn)
            else:
                fn()

        return types.MethodType(wrapper, self)

    # The main process spawns N subprocesses that run the test.
    # Constructor patches current instance test method to
    # assume the role of the main process and join its subprocesses,
    # or run the underlying test function.
    def __init__(self, method_name: str = 'runTest') -> None:
        super().__init__(method_name)
        fn = getattr(self, method_name)
        setattr(self, method_name, self.join_or_run(fn))

    def setUp(self) -> None:
        super().setUp()
        self.skip_return_code_checks = []
        self.skip_all_return_code_checks = False
        self.processes = []
        self.rank = self.MAIN_PROCESS_RANK
        self.file_name = tempfile.NamedTemporaryFile(delete=False).name
        # pid to pipe consisting of error message from process.
        self.pid_to_pipe = {}

    def tearDown(self) -> None:
        super().tearDown()
        for p in self.processes:
            p.terminate()
        # Each Process instance holds a few open file descriptors. The unittest
        # runner creates a new TestCase instance for each test method and keeps
        # it alive until the end of the entire suite. We must thus reset the
        # processes to prevent an effective file descriptor leak.
        self.processes = []
        self.assertTrue(wait_for_port(29500))

    def _current_test_name(self) -> str:
        # self.id() == e.g. '__main__.TestDistributed.TestAdd.test_get_rank'
        return self.id().split('.')[-1]

    def _start_processes(self, proc) -> None:
        self.processes = []

        for rank in range(int(self.world_size)):
            parent_conn, child_conn = torch.multiprocessing.Pipe()
            process = proc(
                target=self.__class__._run,
                name='process ' + str(rank),
                args=(
                    rank,
                    self._current_test_name(),
                    self.file_name,
                    child_conn,
                ),
            )
            process.start()
            logger.info('Started process %s with pid %s', rank, process.pid)
            self.pid_to_pipe[process.pid] = parent_conn
            self.processes.append(process)

    def _spawn_processes(self) -> None:
        proc = torch.multiprocessing.get_context('fork').Process
        self._start_processes(proc)

    class Event(Enum):
        GET_TRACEBACK = 1

    @staticmethod
    def _event_listener(parent_pipe, signal_pipe, rank: int):
        logger.info('Starting event listener thread for rank %s', rank)
        while True:
            ready_pipes = multiprocessing.connection.wait(
                [parent_pipe, signal_pipe]
            )

            if parent_pipe in ready_pipes:

                if parent_pipe.closed:
                    logger.info(
                        'Pipe closed for process %s, stopping event listener '
                        'thread',
                        rank,
                    )
                    return

                event = parent_pipe.recv()
                logger.info('Received event %s on process %s', event, rank)

                if event == MultiProcessTestCase.Event.GET_TRACEBACK:
                    # Return traceback to the parent process.
                    with tempfile.NamedTemporaryFile(mode='r+') as tmp_file:
                        faulthandler.dump_traceback(tmp_file)
                        # Flush buffers and seek to read from the beginning
                        tmp_file.flush()
                        tmp_file.seek(0)
                        parent_pipe.send(tmp_file.read())

                        logger.info('Process %s sent traceback', rank)

            if signal_pipe in ready_pipes:
                return

    @classmethod
    def _run(
        cls, rank: int, test_name: str, file_name: str, parent_pipe
    ) -> None:
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(self.world_size)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        self.run_test(test_name, parent_pipe)

    def run_test(self, test_name: str, parent_pipe) -> None:
        # Start event listener thread.
        signal_recv_pipe, signal_send_pipe = torch.multiprocessing.Pipe(
            duplex=False
        )
        event_listener_thread = threading.Thread(
            target=MultiProcessTestCase._event_listener,
            args=(parent_pipe, signal_recv_pipe, self.rank),
            daemon=True,
        )
        event_listener_thread.start()
        if sys.platform != 'win32' and sys.platform != 'darwin':
            # Register signal handler to dump stack traces on FATALs.
            # Windows and MacOS do not support the signal handlers.
            torch._C._set_print_stack_traces_on_fatal_signal(True)
        # Show full C++ stacktraces when a Python error originating from C++ is
        # raised.
        os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'

        # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
        # We're retrieving a corresponding test and executing it.
        try:
            getattr(self, test_name)()
        except unittest.SkipTest as se:
            logger.info(
                'Process %s skipping test %s for following reason: %s',
                self.rank,
                test_name,
                str(se),
            )
            sys.exit(-1)
        except Exception as e:
            logger.error(
                'Caught exception: \n%s exiting '
                'process %s with exit code: %s',
                traceback.format_exc(),
                self.rank,
                MultiProcessTestCase.TEST_ERROR_EXIT_CODE,
            )
            # Send error to parent process.
            parent_pipe.send(traceback.format_exc())
            sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
        finally:
            if signal_send_pipe is not None:
                signal_send_pipe.send(None)

            assert event_listener_thread is not None
            event_listener_thread.join()
            # Close pipe after done with test.
            parent_pipe.close()

    def _get_timedout_process_traceback(self) -> None:
        pipes = []
        for i, process in enumerate(self.processes):
            if process.exitcode is None:
                pipe = self.pid_to_pipe[process.pid]
                try:
                    pipe.send(MultiProcessTestCase.Event.GET_TRACEBACK)
                    pipes.append((i, pipe))
                except ConnectionError as e:
                    logger.error(
                        f'Encountered error while trying to get traceback for '
                        f'process {i}: {e}'
                    )

        # Wait for results.
        for rank, pipe in pipes:
            try:
                # Wait for traceback
                if pipe.poll(5):
                    if pipe.closed:
                        logger.info(
                            f'Pipe closed for process {rank}, cannot retrieve '
                            f'traceback'
                        )
                        continue

                    traceback = pipe.recv()
                    logger.error(
                        f'Process {rank} timed out with traceback: '
                        f'\n\n{traceback}',
                    )
                else:
                    logger.error(
                        f'Could not retrieve traceback for timed out process:'
                        f'{rank}'
                    )
            except ConnectionError as e:
                logger.error(
                    f'Encountered error while trying to get traceback for '
                    f'process {rank}: {e}'
                )

    def _join_processes(self, fn) -> None:
        timeout = self.DEFAULT_TIMEOUT
        start_time = time.time()
        subprocess_error = False
        try:
            while True:
                # check to see if any subprocess exited with an error early.
                for i, p in enumerate(self.processes):
                    # This is the exit code processes exit with if they
                    # encountered an exception.
                    if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE:
                        print(
                            f'Process {i} terminated with exit code '
                            f'{p.exitcode}, terminating remaining processes.'
                        )
                        active_children = (
                            torch.multiprocessing.active_children()
                        )
                        for ac in active_children:
                            ac.terminate()
                        subprocess_error = True
                        break
                if subprocess_error:
                    break
                # All processes have joined cleanly if they all a valid
                # exitcode
                if all(p.exitcode is not None for p in self.processes):
                    break
                # Check if we should time out the test. If so, we terminate
                # each process.
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    self._get_timedout_process_traceback()
                    print(
                        f'Timing out after {timeout} seconds and killing'
                        f'subprocesses.'
                    )
                    for p in self.processes:
                        p.terminate()
                    break
                # Sleep to avoid excessive busy polling.
                time.sleep(0.1)

            elapsed_time = time.time() - start_time

            if (
                self.skip_all_return_code_checks
                or fn in self.skip_return_code_checks
            ):
                self._check_no_test_errors(elapsed_time)
            else:
                self._check_return_codes(elapsed_time)
        finally:
            # Close all pipes
            for pipe in self.pid_to_pipe.values():
                pipe.close()

    def _check_no_test_errors(self, elapsed_time) -> None:
        '''
        Checks that we didn't have any errors thrown in the child processes.
        '''
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(
                    f'Process {i} timed out after {elapsed_time} seconds'
                )
            self.assertNotEqual(self.TEST_ERROR_EXIT_CODE, p.exitcode)

    def _check_return_codes(self, elapsed_time) -> None:
        '''
        Checks that the return codes of all spawned processes match, and skips
        tests if they returned a return code indicating a skipping condition.
        '''
        # If no processes are spawned, there is nothing to check.
        if not self.processes:
            logger.warning(
                'Note: no subprocesses were spawned, test was likely skipped.'
            )
            return

        first_process = self.processes[0]
        # first, we check if there are errors in actual processes
        # (via TEST_ERROR_EXIT CODE), and raise an exception for those.
        # the reason we do this is to attempt to raise a more helpful error
        # message than 'Process x terminated/timed out'
        # TODO: we should pipe the exception of the failed subprocess here.
        # Currently, the actual exception is displayed as a logging output.
        errored_processes = [
            (i, p)
            for i, p in enumerate(self.processes)
            if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE
        ]
        if errored_processes:
            error = ''
            for i, process in errored_processes:
                # Get error from pipe.
                error_message = self.pid_to_pipe[process.pid].recv()
                error += (
                    'Process {} exited with error code {} and '
                    'exception:\n{}\n'.format(
                        i,
                        MultiProcessTestCase.TEST_ERROR_EXIT_CODE,
                        error_message,
                    )
                )

            raise RuntimeError(error)
        # If no process exited uncleanly, we check for timeouts, and then
        # ensure each process exited cleanly.
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(
                    f'Process {i} terminated or timed out after '
                    f'{elapsed_time} seconds'
                )
            self.assertEqual(
                p.exitcode,
                first_process.exitcode,
                msg='Expect process {} exit code to match Process 0 exit code '
                'of {}, but got {}'.format(
                    i, first_process.exitcode, p.exitcode
                ),
            )
        self.assertEqual(
            first_process.exitcode,
            0,
            msg=(
                f'Expected zero exit code but got {first_process.exitcode} '
                f'for pid: {first_process.pid}',
            ),
        )

    @property
    def is_master(self) -> bool:
        return self.rank == 0
