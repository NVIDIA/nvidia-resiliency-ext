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

import logging
import dataclasses
from typing import Optional
from ..fault_tolerance.rank_monitor_server import RankMonitorLogger

from .callback import Callback
from .state import FrozenState

class NestedRestarterLogger(RankMonitorLogger):
    """Logger used in the nested restarter process"""

    def __init__(self):
        super().__init__(name="InprocessRestarter", is_restarter_logger=True)

    def _setup_logger(self):
        self.setLevel(self.level)
        ch = logging.StreamHandler()
        ch.setLevel(self.level)
        formatter = logging.Formatter(f"[%(asctime)s] [%(levelname)s] [{self.name}@{self.hostname}] %(message)s")
        ch.setFormatter(formatter)
        self.addHandler(ch)
        self.propagate = False

@dataclasses.dataclass
class NestedRestarter(Callback):
    r'''
    Callback for logging the NVRx nested restarter integration.
    '''

    _shared_logger = NestedRestarterLogger()

    restarter_state: str
    restarter_stage: Optional[str] = None
    logger: NestedRestarterLogger = dataclasses.field(default=_shared_logger)
    rank_set: bool = False

    def __call__(self, state: FrozenState) -> FrozenState:
        if not self.rank_set:
            self.logger.set_connected_rank(state.rank)
            self.rank_set = True
        msg = f'active_rank={state.active_rank} [NestedRestarter] name=[InProcess] state={self.restarter_state}'
        if self.restarter_stage is not None:
            msg += f" stage={self.restarter_stage}"
        self.logger.log_for_restarter(msg)

        return state

@dataclasses.dataclass
class NestedRestarterInitialize(NestedRestarter):
    restarter_state: str = 'initialize'

@dataclasses.dataclass
class NestedRestarterFinalized(NestedRestarter):
    restarter_state: str = 'finalized'

@dataclasses.dataclass
class NestedRestarterAborted(NestedRestarter):
    restarter_state: str = 'aborted'

@dataclasses.dataclass
class NestedRestarterHandlingStarting(NestedRestarter):
    restarter_state: str = 'handling'
    restarter_stage: str = 'starting'

@dataclasses.dataclass
class NestedRestarterHandlingProcessing(NestedRestarter):
    restarter_state: str = 'handling'
    restarter_stage: str = 'processing'

@dataclasses.dataclass
class NestedRestarterHandlingCompleted(NestedRestarter):
    restarter_state: str = 'handling'
    restarter_stage: str = 'completed'