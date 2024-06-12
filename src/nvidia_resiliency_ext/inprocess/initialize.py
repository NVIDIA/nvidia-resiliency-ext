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
from typing import Optional

from . import exception
from .state import FrozenState


class Initialize(abc.ABC):
    r'''
    Abstract base class for ``initialize`` argument for
    :py:class:`inprocess.Wrapper`.

    :py:class:`Initialize` is executed at the start of every restart iteration,
    including the first one. :py:class:`Initialize` can raise exceptions (e.g.,
    if specific preconditions are not met). Raising a standard Python
    :py:exc:`Exception` triggers another restart, while raising a
    :py:exc:`BaseException` terminates the wrapper.

    Multiple instances of :py:class:`Initialize` could be composed with
    :py:class:`inprocess.Compose` to achieve the desired behavior.
    '''

    @abc.abstractmethod
    def __call__(self, state: FrozenState) -> FrozenState:
        r'''
        Implementation of a :py:class:`Initialize`.

        Args:
            state: read-only :py:class:`Wrapper` state

        Returns:
            Forwarded read-only input ``state``.
        '''
        raise NotImplementedError


class RetryController(Initialize):
    r'''
    Controls retry logic for distributed training based on specified iteration
    and world size limits.

    This class manages the conditions under which distributed training retries
    are allowed, raising a :py:exc:`inprocess.exception.RestartAbort` exception
    when the conditions are not met.

    Args:
        max_iterations: the maximum number of iterations allowed before
            aborting retries. If :py:obj:`None`, there is no iteration limit
        min_world_size: The minimum required world size to proceed with
            execution
        min_active_world_size: The minimum required active world size to
            proceed with execution
    '''

    def __init__(
        self,
        max_iterations: Optional[int] = None,
        min_world_size: int = 1,
        min_active_world_size: int = 1,
    ):
        self.max_iterations = max_iterations
        self.min_world_size = min_world_size
        self.min_active_world_size = min_active_world_size

    def __call__(self, state: FrozenState) -> FrozenState:
        if (
            state.world_size < self.min_world_size
            or state.active_world_size < self.min_active_world_size
            or (self.max_iterations is not None and state.iteration >= self.max_iterations)
        ):
            msg = (
                f'{state.iteration=} {self.max_iterations=} '
                f'{state.world_size=} {self.min_world_size=} '
                f'{state.active_world_size=} {self.min_active_world_size=} '
            )
            raise exception.RestartAbort(msg)
        return state
