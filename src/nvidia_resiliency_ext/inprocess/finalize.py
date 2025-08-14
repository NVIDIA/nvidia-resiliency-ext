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
import logging
import os
import threading
from typing import Any, Callable, Optional

import torch

from . import exception
from .state import FrozenState
from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig


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

        # Ensure CUDA is available and initialized, raise exception if not
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if not torch.cuda.is_initialized():
            raise RuntimeError("CUDA is not initialized")

        # Determine device in main thread before creating the thread
        if (local_rank := os.getenv('LOCAL_RANK', None)) is not None:
            device_id = torch.device(int(local_rank))
        else:
            device_id = torch.device(torch.cuda.current_device())

        def wrapped_fn():
            # Set CUDA device in the thread
            if device_id is not None:
                log = logging.getLogger(LogConfig.name)
                log.debug(f'Setting CUDA device to {device_id} in ThreadedFinalize')
                torch.cuda.set_device(device_id)

            # Call the original function
            return self.fn(*self.args, **self.kwargs)

        thread = threading.Thread(
            target=wrapped_fn,
            name=f'{type(self).__name__}-{rank}',
            daemon=True,
        )
        thread.start()
        thread.join(self.timeout.total_seconds())
        if thread.is_alive():
            raise exception.TimeoutError

        return state
