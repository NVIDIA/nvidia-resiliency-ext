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
import logging
import os

import torch

from nvidia_resiliency_ext.attribution.trace_analyzer.trace_collector import TorchFRTraceCollector

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

    _logging_printed: bool = False

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

    @classmethod
    def collect_fr_trace(cls):
        def _check_fr_env():
            check_env_variables = ['TORCH_NCCL_DUMP_ON_TIMEOUT', 'TORCH_NCCL_ENABLE_MONITORING']
            rank = torch.distributed.get_rank()
            for env_var in check_env_variables:
                env_value = os.environ.get(env_var, '0')
                # Convert string boolean values to integers
                if env_value.lower() in ('true', '1', 'yes', 'on'):
                    env_value_int = 1
                elif env_value.lower() in ('false', '0', 'no', 'off'):
                    env_value_int = 0
                else:
                    try:
                        env_value_int = int(env_value)
                    except ValueError:
                        env_value_int = 0  # Default to False for invalid values

                if bool(env_value_int) is False and rank == 0 and not cls._logging_printed:
                    log = logging.getLogger(__name__)
                    log.info(
                        f"Environment variable {env_var} is set to {env_value}"
                        f", FR trace collection is disabled"
                    )
                    cls._logging_printed = True
                    return False
            return True

        if _check_fr_env() is True:
            trace_path = os.environ.get('NVRX_FR_TRACE_PATH', None)
            if trace_path is None:
                return
            if not os.path.exists(trace_path):
                os.makedirs(trace_path)
            trace_analyzer = TorchFRTraceCollector(trace_path)
            trace_analyzer.collect()

    def __call__(self, state: FrozenState) -> FrozenState:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.collect_fr_trace()
            AbortTorchDistributed.shutdown_all_process_group_backends()
            torch.distributed.destroy_process_group()
        return state


class AbortTransformerEngine(Abort):
    r'''
    Aborts TransformerEngine Userbuffer.

    '''

    def __call__(self, state: FrozenState) -> FrozenState:
        try:
            import transformer_engine.pytorch as te
        except Exception:
            pass
        else:
            te.module.base.destroy_ub()

        try:
            import transformer_engine.pytorch.fp8 as te_fp8
        except Exception:
            pass
        else:
            # Clear a class-member containing a process group
            te_fp8.FP8GlobalStateManager.reset()

        return state
