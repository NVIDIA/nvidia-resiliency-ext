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

from .callback import Callback
from .state import FrozenState


@dataclasses.dataclass
class NestedRestarter(Callback):
    r'''
    Callback for logging the NVRx nested restarter integration.
    '''

    restarter_state: str
    restarter_stage: Optional[str] = None

    def __call__(self, state: FrozenState) -> FrozenState:
        log = logging.getLogger(__name__)
        msg = f'[NestedRestarter] name=[InProcess] state={self.restarter_state}'
        if self.restarter_stage is not None:
            msg += f" stage={self.restarter_stage}"
        log.info(msg)

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