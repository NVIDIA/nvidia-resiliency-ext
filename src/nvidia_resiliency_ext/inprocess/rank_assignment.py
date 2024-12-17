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
from typing import Callable, Union

from . import exception
from .state import State


class RankDiscarded(exception.RestartError):
    r'''
    Exception raised when unhealthy distributed rank is discarded by
    :py:class:`inprocess.rank_assignment.RankAssignment`.
    '''

    pass


class RankAssignment(abc.ABC):
    r'''
    Abstract base class for ``rank_assignment`` argument for
    :py:class:`inprocess.Wrapper`.


    :py:class:`RankAssignment` is responsible for reassigning distributed ranks
    and computing the new world size for the next iteration of the wrapped
    function.

    Multiple instances of :py:class:`RankAssignment` could be composed with
    :py:class:`inprocess.Compose` to achieve the desired behavior.
    '''

    @abc.abstractmethod
    def __call__(
        self, state: State, terminated_ranks: set[int]
    ) -> (State, set[int]):
        raise NotImplementedError


class FillGaps(RankAssignment):
    r'''
    A class for reassigning distributed ranks, filling in gaps caused by
    terminated or unhealthy ranks.

    The :py:class:`FillGaps` class is a specialized rank assignment strategy
    that reorders ranks to fill gaps created by terminated or unhealthy ranks.
    It preserves the previous rank assignment for the first ``world_size -
    len(terminated_ranks)`` healthy ranks; the remaining healthy ranks are
    reassigned to fill in gaps left by unhealthy ranks.

    Example:

    .. code-block:: python

        |<--- preserved --->|<- moved ->|     |<--new world size->|

        +---+---+---+---+---+---+---+---+     +---+---+---+---+---+
        | 0 | X | 2 | 3 | X | X | 6 | 7 | --> | 0 | 6 | 2 | 3 | 7 |
        +---+---+---+---+---+---+---+---+     +---+---+---+---+---+
              ^           ^        |  |
              |           |        |  |
              ---------------------   |
                          |           |
                          -------------
    '''

    def __call__(
        self, state: State, terminated_ranks: set[int]
    ) -> (State, set[int]):
        rank = state.rank
        world_size = state.world_size

        ordered_terminated_ranks = sorted(list(terminated_ranks))
        world_size = world_size - len(terminated_ranks)

        if rank in terminated_ranks:
            raise RankDiscarded(f'{rank=} {terminated_ranks=}')
        elif rank >= world_size:
            rank = ordered_terminated_ranks[rank - world_size]

        state.rank = rank
        state.world_size = world_size
        terminated_ranks = set()
        return state, terminated_ranks


class ShiftRanks(RankAssignment):
    r'''
    A class for reassigning distributed ranks, filling in gaps caused by
    terminated or unhealthy ranks.

    The :py:class:`ShiftRanks` class is a specialized rank assignment strategy
    that shifts all healthy ranks to the left to fill gaps created by
    terminated or unhealthy ranks. :py:class:`ShiftRanks` preserves the
    relative order of all healthy ranks, but all ranks past the first unhealthy
    rank are reassigned (shifted).

    Example:

    .. code-block:: python

         <-   ->|<------- moved ------->|     |<--new world size->|

                  ----
                  v   |
        +---+---+---+---+---+---+---+---+     +---+---+---+---+---+
        | 0 | X | 2 | 3 | X | X | 6 | 7 | --> | 0 | 2 | 3 | 6 | 7 |
        +---+---+---+---+---+---+---+---+     +---+---+---+---+---+
              ^   |   ^   ^       |   |
              |   |   |   |       |   |
              ----     ------------   |
                          |           |
                          ------------

    '''

    def __call__(
        self, state: State, terminated_ranks: set[int]
    ) -> (State, set[int]):
        rank = state.rank
        world_size = state.world_size

        world_size = world_size - len(terminated_ranks)
        if rank in terminated_ranks:
            raise RankDiscarded(f'{rank=} {terminated_ranks=}')
        else:
            rank = rank - sum(
                rank > terminated_rank for terminated_rank in terminated_ranks
            )

        state.rank = rank
        state.world_size = world_size
        terminated_ranks = set()
        return state, terminated_ranks


class FilterGroupedByKey(RankAssignment):
    r'''
    A class for filtering distributed ranks by grouping by a key.

    :py:class:`FilterGroupedByKey` organizes ranks into groups based on a
    specified string key. For each group, it increments a group counter by 1
    for every healthy rank. A given boolean ``condition`` is then evaluated for
    each rank, with the corresponding group counter passed as input.

    - If ``condition(group_counter)`` evaluates to ``True``, the rank is
      preserved.
    - If it evaluates to ``False``, the rank is considered unhealthy and marked
      for termination.

    :py:class:`FilterGroupedByKey` needs to be followed by another
    :py:class:`RankAssignment` that performs the actual rank termination by
    raising :py:exc:`RankDiscarded` exception.

    .. code-block:: python

        condition = lambda count: count == 2

        +---+---+---+---+---+---+---+---+     +---+---+---+---+---+---+---+---+
        | 0 | X | 2 | 3 | X | X | 6 | 7 | --> | X | X | 2 | 3 | X | X | 6 | 7 |
        +---+---+---+---+---+---+---+---+     +---+---+---+---+---+---+---+---+
        | key=0 | key=1 | key=2 | key=3 |     | key=0 | key=1 | key=2 | key=3 |
        |       |       |       |       |     |       |       |       |       |
        |count=1|count=2|count=0|count=2|     | False | True  | False | True  |

    Example:

    .. code-block:: python

        # hostname is the group key, and condition checks if exactly 8 ranks
        # corresponding to a given hostname are in a healthy state, if the
        # count is different than 8, all ranks from corresponding hostname are
        # considered unhealthy, and terminated; remaining healthy ranks are
        # shifted to the left to fill all gaps created by unhealthy ranks.

        rank_assignment = (
            inprocess.Compose(
                inprocess.rank_assignment.ShiftRanks(),
                inprocess.rank_assignment.FilterGroupedByKey(
                    key_or_fn=lambda _, _: socket.gethostname(),
                    condition=lambda count: count == 8,
                ),
            ),
        ),


    Args:
        key_or_fn: a string key, or a ``Callable`` evaluated with ``(rank,
            world_size)`` as the input to produce a string key
        condition: condition to be evaluated with group counter as the input,
            if ``False`` the rank is terminated
        timeout: timeout for distributed barrier

    '''

    instance_count = 0

    def __init__(
        self,
        key_or_fn: Union[str, Callable[[int, int], str]],
        condition: Callable[int, bool],
        timeout: datetime.timedelta = datetime.timedelta(seconds=60),
    ):
        self.key_or_fn = key_or_fn
        self.condition = condition
        self.timeout = timeout

        self.name = f'{type(self).__name__}_{type(self).instance_count}'
        type(self).instance_count += 1

    def __call__(
        self, state: State, terminated_ranks: set[int]
    ) -> (State, set[int]):
        COUNT_ALIVE_BARRIER = f'count_alive_barrier_{self.name}'
        SUBMIT_MISMATCHING_BARRIER = f'submit_mismatching_barrier_{self.name}'
        RANKS_TO_TERMINATE = f'ranks_to_terminate_{self.name}'

        rank = state.rank
        world_size = state.world_size
        store = state.store

        alive_world_size = world_size - len(terminated_ranks)

        if rank not in terminated_ranks:
            key = (
                self.key_or_fn(rank, world_size)
                if callable(self.key_or_fn)
                else self.key_or_fn
            )
            key = f'filter_grouped_by_key_{self.name}_{key}'
            store.add(key, 1)
            store.barrier(
                rank=rank,
                group_name=COUNT_ALIVE_BARRIER,
                rendezvous_count=alive_world_size,
                timeout=self.timeout,
            )

            if not self.condition(int(store.get(key))):
                store.append(RANKS_TO_TERMINATE, f'{rank},')

            store.barrier(
                rank=rank,
                group_name=SUBMIT_MISMATCHING_BARRIER,
                rendezvous_count=alive_world_size,
                timeout=self.timeout,
            )
            store.delete_key(key)

            if store.check([RANKS_TO_TERMINATE]):
                ranks_to_terminate = set(
                    int(r)
                    for r in store.get(RANKS_TO_TERMINATE)
                    .decode()
                    .rstrip(',')
                    .split(',')
                )
            else:
                ranks_to_terminate = set()

            terminated_ranks = terminated_ranks.union(ranks_to_terminate)

        return state, terminated_ranks
