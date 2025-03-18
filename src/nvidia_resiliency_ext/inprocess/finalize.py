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

import abc
import datetime
import threading
from typing import Any, Callable, Optional

from . import exception
from .state import FrozenState


class Finalize(abc.ABC):
    r'''
    Abstract base class for ``finalize`` argument for
    :py:class:`inprocess.Wrapper`.

    :py:class:`Finalize` brings the process into a state where a restart of the
    wrapped function may be attempted, e.g.: deinitialize any global variables
    or synchronize with any asynchronous tasks issued by the wrapped function
    that was not already performed by exception handlers in the wrapped
    function.

    Any failure during execution of :py:class:`Finalize` should raise an
    exception. In this case the health check is skipped, exception is reraised
    by the wrapper, and it should cause termination of the main Python
    interpreter process.

    :py:class:`Finalize` class is executed after a fault was detected,
    distributed group was destroyed, but before the
    :py:class:`inprocess.health_check.HealthCheck` is performed.

    Multiple instances of :py:class:`Finalize` could be composed with
    :py:class:`inprocess.Compose` to achieve the desired behavior.
    '''

    @abc.abstractmethod
    def __call__(self, state: FrozenState) -> FrozenState:
        r'''
        Implementation of a :py:class:`Finalize`.

        Args:
            state: read-only :py:class:`Wrapper` state

        Returns:
            Forwarded read-only input ``state``.
        '''
        raise NotImplementedError


class ThreadedFinalize(Finalize):
    r'''
    Executes the provided finalize ``fn`` function with specified positional
    and keyword arguments in a separate :py:class:`threading.Thread`.

    Raises an exception if execution takes longer than the specified
    ``timeout``.

    Args:
        timeout: timeout for a thread executing ``fn``
        fn: function to be executed
        args: tuple of positional arguments
        kwargs: dictionary of keyword arguments
    '''

    def __init__(
        self,
        timeout: datetime.timedelta,
        fn: Callable[..., Any],
        args: Optional[tuple[Any, ...]] = (),
        kwargs: Optional[dict[str, Any]] = None,
    ):
        if kwargs is None:
            kwargs = {}

        self.timeout = timeout
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self, state: FrozenState) -> FrozenState:
        rank = state.rank
        thread = threading.Thread(
            target=self.fn,
            name=f'{type(self).__name__}-{rank}',
            args=self.args,
            kwargs=self.kwargs,
            daemon=True,
        )
        thread.start()
        thread.join(self.timeout.total_seconds())
        if thread.is_alive():
            raise exception.TimeoutError

        return state
