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
import concurrent.futures

import torch

from . import utils
from .state import FrozenState


class Abort(abc.ABC):
    r'''
    Abstract base class for ``abort`` argument for
    :py:class:`inprocess.Wrapper`.

    An instance of :py:class:`Abort` is triggered by a separate monitoring
    thread within :py:class:`inprocess.Wrapper` as part of the termination
    mechanism when a fault is detected. Its primary purpose is to unblock the
    main thread, which might be waiting for results from other distributed
    ranks that are either already terminated or unresponsive. For example, this
    could occur during a distributed collective operation attempting to
    communicate with a terminated rank.

    Multiple instances of :py:class:`Abort` could be composed with
    :py:class:`inprocess.Compose` to achieve the desired behavior.
    '''

    @abc.abstractmethod
    def __call__(self, state: FrozenState) -> FrozenState:
        r'''
        Implementation of a :py:class:`Abort`.

        Args:
            state: read-only :py:class:`Wrapper` state

        Returns:
            Forwarded read-only input ``state``.
        '''
        raise NotImplementedError


class AbortTorchDistributed(Abort):
    r'''
    Aborts PyTorch distributed collectives, and destroys all PyTorch
    distributed process groups.

    This functionality is implemented by invoking
    :py:func:`torch.distributed.destroy_process_group` in a separate Python
    thread for each distributed group that has been created.
    '''

    @staticmethod
    def shutdown_all_process_group_backends():
        device = torch.device('cuda')
        process_groups = list(torch.distributed.distributed_c10d._world.pg_names)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(process_groups)) as executor:
            futures = [
                executor.submit(
                    AbortTorchDistributed.shutdown_process_group_backend,
                    group,
                    device,
                )
                for group in process_groups
            ]
            done, not_done = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.ALL_COMPLETED,
            )

    @staticmethod
    def shutdown_process_group_backend(group, device):
        if isinstance(group, torch.distributed.ProcessGroup):
            backend = group._get_backend(device)

            if isinstance(
                backend,
                torch.distributed.distributed_c10d.ProcessGroupNCCL,
            ):
                if utils.torch_older_than('2.6.0'):
                    backend._shutdown()
                else:
                    backend.abort()

    def __call__(self, state: FrozenState) -> FrozenState:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            AbortTorchDistributed.shutdown_all_process_group_backends()
            torch.distributed.destroy_process_group()
        return state
