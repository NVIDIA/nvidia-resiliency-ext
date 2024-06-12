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
import warnings
from typing import Optional

from . import rank_assignment
from .state import State


class RankFilter(abc.ABC):
    r'''
    .. note::
       :py:class:`RankFilter` is deprecated and will be removed in the next
       release. The functionality is merged into
       :py:class:`inprocess.rank_assignment.RankAssignment`.

    :py:class:`RankFilter` selects which ranks are active in the current
    restart iteration of :py:class:`inprocess.Wrapper`.

    Active ranks call the provided wrapped function. Inactive ranks are waiting
    idle, and could serve as a pool of static, preallocated and preinitialized
    reserve ranks. Reserve ranks would be activated in a subsequent restart
    iteration if previously active ranks were terminated or became unhealthy.

    Multiple instances of :py:class:`RankFilter` could be composed with
    :py:class:`inprocess.Compose` to achieve the desired behavior.
    '''

    @abc.abstractmethod
    def __call__(self, state: State) -> State:
        raise NotImplementedError


class MaxActiveWorldSize(RankFilter):
    r'''
    .. note::
       :py:class:`MaxActiveWorldSize` is deprecated and will be removed in the
       next release. The functionality is moved to
       :py:class:`inprocess.rank_assignment.MaxActiveWorldSize`.

    :py:class:`MaxActiveWorldSize` ensures that the active world size is no
    greater than the specified ``max_active_world_size``. Ranks with indices
    less than the active world size are active and calling the wrapped
    function, while ranks outside this range are inactive.

    Args:
        max_active_world_size: maximum active world size, no limit if
            :py:obj:`None`
    '''

    def __init__(self, max_active_world_size: Optional[int] = None):
        warnings.warn(
            'rank_filter.MaxActiveWorldSize is deprecated and will be removed '
            'in the next release. The functionality is moved to '
            'rank_assignment.MaxActiveWorldSize',
            DeprecationWarning,
            stacklevel=2,
        )
        self.impl = rank_assignment.MaxActiveWorldSize(max_active_world_size)

    def __call__(self, state: State) -> State:
        ctx = rank_assignment.RankAssignmentCtx(state=state, store=None, terminated_ranks=set())
        return self.impl(ctx).state


class ActiveWorldSizeDivisibleBy(RankFilter):
    r'''
    .. note::
       :py:class:`ActiveWorldSizeDivisibleBy` is deprecated and will be removed
       in the next release. The functionality is moved to
       :py:class:`inprocess.rank_assignment.ActiveWorldSizeDivisibleBy`.

    :py:class:`ActiveWorldSizeDivisibleBy` ensures that the active world size
    is divisible by a given number. Ranks within the adjusted world size are
    marked as active and are calling the wrapped function, while ranks outside
    this range are marked as inactive.

    Args:
        divisor: the divisor to adjust the active world size by
    '''

    def __init__(self, divisor: int = 1):
        warnings.warn(
            'rank_filter.ActiveWorldSizeDivisibleBy is deprecated and will be '
            'removed in the next release. The functionality is moved to '
            'rank_assignment.ActiveWorldSizeDivisibleBy',
            DeprecationWarning,
            stacklevel=2,
        )
        self.impl = rank_assignment.ActiveWorldSizeDivisibleBy(divisor)

    def __call__(self, state: State) -> State:
        ctx = rank_assignment.RankAssignmentCtx(state=state, store=None, terminated_ranks=set())
        return self.impl(ctx).state


class WorldSizeDivisibleBy(ActiveWorldSizeDivisibleBy):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            'rank_filter.WorldSizeDivisibleBy is deprecated and will be '
            'removed in the next release. The functionality is moved to '
            'rank_assignment.ActiveWorldSizeDivisibleBy',
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__init__(*args, **kwargs)
