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

from .store import StoreMixin


class Status(enum.Enum):
    INITIALIZED = enum.auto()
    ACTIVE = enum.auto()
    INACTIVE = enum.auto()
    ABORTING = enum.auto()
    ABORTED = enum.auto()


@dataclasses.dataclass
class State:
    rank: Optional[int]
    initial_rank: int
    world_size: int
    iteration: int
    active_world_size: int
    initial_world_size: int
    status: Status
    store: StoreMixin

    def __init__(self):
        self.get_distributed_vars()
        self.iteration = 0
        self.initial_rank = self.rank
        self.initial_world_size = self.world_size
        self.activate_all_ranks()
        self.status = Status.INITIALIZED

    def get_distributed_vars(self):
        self.rank = int(os.getenv('RANK', 0))
        self.world_size = int(os.getenv('WORLD_SIZE', 1))

    def set_distributed_vars(self):
        os.environ['RANK'] = str(self.rank)
        os.environ['WORLD_SIZE'] = str(self.active_world_size)

    def activate_all_ranks(self):
        self.status = Status.ACTIVE
        self.active_world_size = self.world_size

    def advance(self):
        self.iteration += 1
