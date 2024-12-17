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

from .state import State
from .state import Status


class RankFilter(abc.ABC):
    r'''
    :py:class:`RankFilter` selects which ranks are active in the current
    restart iteration of :py:class:`inprocess.Wrapper`.

    Active ranks call the provided wrapped function. Inactive ranks are waiting
    idle, and could serve as a pool of static, preallocated and preinitialized
    spare ranks. Spare ranks would be activated in a subsequent restart
    iteration if previously active ranks were terminated or became unhealthy.

    Multiple instances of :py:class:`RankFilter` could be composed with
    :py:class:`inprocess.Compose` to achieve the desired behavior.
    '''

    @abc.abstractmethod
    def __call__(self, state: State) -> State:
        raise NotImplementedError


class MaxActiveWorldSize(RankFilter):
    r'''
    :py:class:`MaxActiveWorldSize` ensures that the active world size is no
    greater than the specified ``max_active_world_size``. Ranks with indices
    less than the active world size are active and calling the wrapped
    function, while ranks outside this range are inactive (sleeping).

    Args:
        max_active_world_size: maximum active world size, no limit if
            :py:obj:`None`
    '''

    def __init__(self, max_active_world_size: Optional[int]):
        self.max_active_world_size = max_active_world_size

    def __call__(self, state: State) -> State:
        active_world_size = state.active_world_size
        if self.max_active_world_size is not None:
            active_world_size = min(
                active_world_size, self.max_active_world_size
            )
        if state.rank < active_world_size:
            status = Status.ACTIVE
        else:
            status = Status.INACTIVE

        state.active_world_size = active_world_size
        state.status = status
        return state


class WorldSizeDivisibleBy(RankFilter):
    r'''
    :py:class:`WorldSizeDivisibleBy` ensures that the active world size is
    divisible by a given number. Ranks within the adjusted world size are
    marked as active and are calling the wrapped function, while ranks outside
    this range are marked as inactive (sleeping).

    Args:
        divisor: the divisor to adjust the active world size by
    '''

    def __init__(self, divisor: int = 1) -> None:
        self.divisor = divisor

    def __call__(self, state: State) -> State:
        divisor = self.divisor
        active_world_size = state.active_world_size // divisor * divisor
        if state.rank < active_world_size:
            status = Status.ACTIVE
        else:
            status = Status.INACTIVE

        state.active_world_size = active_world_size
        state.status = status
        return state
