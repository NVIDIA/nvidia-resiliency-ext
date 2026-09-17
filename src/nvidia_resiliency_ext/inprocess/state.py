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
import enum
import os
from typing import Optional


class Mode(enum.Enum):
    r'''
    Indicates operational mode of the current distributed rank.

    Attributes:
        INITIALIZED: the :py:class:`State` was initialized,
            :py:class:`RankAssignment` was not yet performed
        ACTIVE: the rank calls the wrapped function
        INACTIVE: the rank is waiting idle
        TERMINATED: the rank was terminated
    '''

    INITIALIZED = enum.auto()
    ACTIVE = enum.auto()
    INACTIVE = enum.auto()
    TERMINATED = enum.auto()


@dataclasses.dataclass
class State:
    r'''
    Represents the current state of the :py:class:`inprocess.Wrapper`.

    Args:
        rank: a distributed rank index as seen by the
            :py:class:`inprocess.Wrapper`, :py:obj:`None` for terminated ranks
        world_size: a total number of distributed ranks controlled by the
            :py:class:`inprocess.Wrapper`
        active_rank: a distributed rank index passed to the wrapped function
        active_world_size: a total number of distributed ranks passed to the
            wrapped function
        initial_rank: an distributed rank index, captured when the
            :py:class:`Wrapper` was invoked
        initial_world_size: a total number of initial distributed ranks
        iteration: index of the current restart iteration (from global_iteration_counter)
        job_restart_count: actual count of job restarts (from job_restart_counter)
        mode: operational mode
        fn_exception: an instance of :py:exc:`Exception` raised by the wrapped
            function in the current restart iteration, :py:obj:`None` if no
            exception was raised
    '''

    rank: Optional[int]
    world_size: int

    active_rank: Optional[int] = None
    active_world_size: Optional[int] = None

    initial_rank: Optional[int] = None
    initial_world_size: Optional[int] = None

    iteration: int = 0
    job_restart_count: int = 0
    mode: Mode = Mode.INITIALIZED
    fn_exception: Optional[Exception] = None

    def __post_init__(self):
        if self.initial_rank is None:
            self.initial_rank = self.rank
        if self.initial_world_size is None:
            self.initial_world_size = self.world_size

    @classmethod
    def from_env(cls, store=None):
        rank = int(os.getenv('RANK', 0))
        world_size = int(os.getenv('WORLD_SIZE', 1))

        # Get iteration from global counter if store is provided
        iteration = 0
        job_restart_count = 0
        if store is not None:
            iteration = store.get_global_iteration_counter()
            job_restart_count = store.get_job_restart_counter()

        return cls(
            rank=rank,
            world_size=world_size,
            initial_rank=rank,
            initial_world_size=world_size,
            iteration=iteration,
            job_restart_count=job_restart_count,
        )

    def set_distributed_vars(self):
        os.environ['RANK'] = str(self.active_rank)
        os.environ['WORLD_SIZE'] = str(self.active_world_size)

    def advance(self, base_store=None, prefix_store=None):
        """
        Advance the state to the next iteration and update restart counters.

        Args:
            base_store: Optional base store instance to update global counters.
                       If provided, will increment job_restart_counter when needed.
            prefix_store: Optional prefix store instance to update iteration-specific counters.
                         If provided, will increment ranks_restart_counter for current iteration.
        """
        self.iteration += 1
        self.fn_exception = None

        # Update restart counters if stores are provided
        if prefix_store is None:
            return

        # Increment the ranks restart counter for this iteration
        # Use the prefix store directly - no need to construct the key manually
        new_ranks_count = prefix_store.increment_ranks_restart_counter()

        # Check if we should increment the job restart counter
        # This happens when all active ranks have completed at least one iteration
        if base_store is not None and prefix_store.should_increment_job_restart_counter(
            self.world_size or self.active_world_size, new_ranks_count
        ):
            base_store.increment_job_restart_counter()
            # Update the local state to reflect the new job restart count
            self.job_restart_count = base_store.get_job_restart_counter()

    def freeze(self):
        frozen = FrozenState(**dataclasses.asdict(self))
        return frozen

    def copy_from(self, other, fields=None):
        for field in dataclasses.fields(self):
            if fields is None or field.name in fields:
                setattr(self, field.name, getattr(other, field.name))


def freeze_dataclass(cls):
    fields = [(f.name, f.type, f) for f in dataclasses.fields(cls)]
    FrozenClass = dataclasses.make_dataclass(
        cls_name=f'Frozen{cls.__name__}', fields=fields, frozen=True
    )
    return FrozenClass


FrozenState = freeze_dataclass(State)
FrozenState.__doc__ = r'''
    :py:class:`inprocess.FrozenState` is identical to
    :py:class:`inprocess.State`, except all fields are read-only.
'''
