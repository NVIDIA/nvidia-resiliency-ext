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

import abc
from typing import Optional

from . import exception
from .state import FrozenState

class CompletionCallback(abc.ABC):
    r'''
    Abstract base class for ``global_finalize_success`` argument for
    :py:class:`inprocess.Wrapper`.

    :py:class:`CompletionCallback` is executed by any unterminated rank when
    it has completed the workload wrapped by inprocess.

    Multiple instances of :py:class:`CompletionCallback` could be composed with
    :py:class:`inprocess.Compose` to achieve the desired behavior.
    '''

    @abc.abstractmethod
    def __call__(self, state: FrozenState) -> FrozenState:
        r'''
        Implementation of a :py:class:`CompletionCallback`.

        Args:
            state: read-only :py:class:`Wrapper` state

        Returns:
            Forwarded read-only input ``state``.
        '''
        raise NotImplementedError
