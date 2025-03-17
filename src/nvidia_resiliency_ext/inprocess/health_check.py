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
import logging
import os
import threading

import torch

from . import exception
from .state import FrozenState


class HealthCheck(abc.ABC):
    r'''
    Abstract base class for ``health_check`` argument for
    :py:class:`inprocess.Wrapper`.

    :py:class:`HealthCheck` ensures the worker is in a healthy state and can
    execute the workload.

    Health checks are executed after the target function failure was discovered
    (on local, or other distributed ranks), local distributed group was
    destroyed, and after the user-provided
    :py:class:`inprocess.finalize.Finalize` finished.

    :py:class:`HealthCheck` is executed to filter out unhealthy ranks (e.g. due
    to corrupted CUDA context). The execution should be local to a given rank,
    other ranks may have already been terminated, lost or still executing the
    wrapped function.

    Unhealthy state is reported by raising an :py:exc:`Exception`. The
    exception is reraised by the :py:class:`inprocess.Wrapper`, and should lead
    to termination of the main Python interpreter process.

    Multiple instances of :py:class:`HealthCheck` could be composed with
    :py:class:`inprocess.Compose` to achieve the desired behavior.
    '''

    @abc.abstractmethod
    def __call__(self, state: FrozenState) -> FrozenState:
        r'''
        Implementation of a :py:class:`HealthCheck`.

        Args:
            state: read-only :py:class:`Wrapper` state

        Returns:
            Forwarded read-only input ``state``.
        '''
        raise NotImplementedError


class CudaHealthCheck(HealthCheck):
    r'''
    Ensures that CUDA context for the current process is in a healthy state.

    Synchronizes with the GPU. Uses the device corresponding to ``LOCAL_RANK``
    environment variable, or the main thread's default CUDA device if
    ``LOCAL_RANK`` was not specified in the environment.

    Args:
        timeout: timeout for synchronization with the GPU
    '''

    def __init__(self, timeout=datetime.timedelta(seconds=30)):
        self.timeout = timeout

    def __call__(self, state: FrozenState) -> FrozenState:
        log = logging.getLogger(__name__)
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            if (local_rank := os.getenv('LOCAL_RANK', None)) is not None:
                device = torch.device(int(local_rank))
            else:
                device = torch.device(torch.cuda.current_device())

            # sync waits for completion of all issued CUDA kernels, this could
            # take very long if CPU app code ran far ahead of CUDA code, but
            # there is no other way around, there is no way to cancel pending
            # CUDA kernels, and any pending kernel may corrupt CUDA context
            thread = threading.Thread(
                target=torch.cuda.synchronize,
                args=(device,),
                name=f'{type(self).__name__}Sync',
                daemon=True,
            )
            log.debug(f'1st torch.cuda.synchronize({device=})')
            thread.start()
            thread.join(self.timeout.total_seconds())
            if thread.is_alive():
                log.debug('torch.cuda.synchronize() timed out')
                raise exception.TimeoutError

            # 2nd sync to check if CUDA context is healthy
            log.debug(f'2nd torch.cuda.synchronize({device=})')
            torch.cuda.synchronize(device)
        return state


class FaultCounterExceeded(exception.RestartError):
    r'''
    Exception raised by :py:class:`FaultCounter` when number of faults on the
    current rank exceeds the threshold.
    '''

    pass


class FaultCounter(HealthCheck):
    r'''
    :py:class:`FaultCounter` counts faults caused by the current process. The
    process is terminated if total number of faults exceeds the
    ``max_rank_faults`` threshold.

    Args:
        max_rank_faults: maximum number of faults cause by the process
    '''

    def __init__(self, max_rank_faults=None):
        self.max_rank_faults = max_rank_faults
        self.faults_count = 0

    def __call__(self, state: FrozenState) -> FrozenState:
        if state.fn_exception is None:
            return state

        self.faults_count += 1
        max_rank_faults = self.max_rank_faults
        faults_count = self.faults_count

        if max_rank_faults is not None and faults_count > max_rank_faults:
            raise FaultCounterExceeded(f'{faults_count=} / {max_rank_faults=}')
        return state
