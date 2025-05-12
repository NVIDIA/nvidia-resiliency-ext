# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional

from ..fault_tolerance.rank_monitor_server import RankMonitorLogger
from .abort import Abort
from .completion import Completion
from .initialize import Initialize
from .state import FrozenState
from .terminate import Terminate


class NestedRestarterLogger(RankMonitorLogger):
    """Logger used in the nested restarter process"""

    def __init__(self):
        super().__init__(name="InprocessRestarter", is_restarter_logger=True)


@dataclasses.dataclass
class NestedRestarterCallback:
    r'''
    Callback for logging the NVRx nested restarter integration.
    '''

    _shared_logger = NestedRestarterLogger()

    restarter_state: str
    restarter_stage: Optional[str] = None
    logger: NestedRestarterLogger = dataclasses.field(default=_shared_logger)
    special_rank: int = 0

    def __call__(self, state: FrozenState) -> FrozenState:

        if state.initial_rank == self.special_rank:
            self.logger.set_connected_rank(state.initial_rank)
            msg = f'[NestedRestarter] name=[InProcess] state={self.restarter_state}'
            if self.restarter_stage is not None:
                msg += f" stage={self.restarter_stage}"

            self.logger.log_for_restarter(msg)

        return state


@dataclasses.dataclass
class NestedRestarterHandlingCompleted(Initialize, NestedRestarterCallback):

    restarter_state: str = 'initialize'
    restarter_stage: str = None

    def __init__(self, special_rank: int = 0):
        self._called_once = False
        self.special_rank = special_rank
        self.logger = NestedRestarterCallback._shared_logger

    def __call__(self, state: FrozenState) -> FrozenState:

        # Apply the callback functionality
        state = NestedRestarterCallback.__call__(self, state)

        if not self._called_once:
            self._called_once = True
            self.restarter_state = 'handling'
            self.restarter_stage = 'completed'

        return state


@dataclasses.dataclass
class NestedRestarterHandlingStarting(Abort, NestedRestarterCallback):
    restarter_state: str = 'handling'
    restarter_stage: str = 'starting'

    def __call__(self, state: FrozenState) -> FrozenState:
        return NestedRestarterCallback.__call__(self, state)


@dataclasses.dataclass
class NestedRestarterFinalized(Completion, NestedRestarterCallback):
    restarter_state: str = 'finalized'

    def __call__(self, state: FrozenState) -> FrozenState:
        return NestedRestarterCallback.__call__(self, state)


@dataclasses.dataclass
class NestedRestarterAborted(Terminate, NestedRestarterCallback):
    restarter_state: str = 'aborted'

    def __call__(self, state: FrozenState) -> FrozenState:
        return NestedRestarterCallback.__call__(self, state)
