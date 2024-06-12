# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
TorchAsyncCheckpoint defines a wrapper for the async version of `torch.save` with
an additional method to synchronize async saving requests
"""


import logging

import torch

from ..utils import preload_tensors, wrap_for_async
from .core import AsyncCallsQueue, AsyncRequest

logger = logging.getLogger(__name__)


class TorchAsyncCheckpoint(object):
    async_fn = None

    def __init__(self, persistent_queue=False):
        self.save = torch.save
        self._async_calls_queue = AsyncCallsQueue(persistent=persistent_queue)
        # Use direct torch.save for persistent queue, avoid unnecessary wrapping
        TorchAsyncCheckpoint.async_fn = (
            torch.save if persistent_queue else wrap_for_async(torch.save)
        )

    def async_save(self, state_dict, *args, **kwargs):
        """
        Keeps the original interface of `torch.save`
        Schedules a `AsyncReuqest` with preloading tensors to CPU with pinned memcpy
        """

        preloaded_sd = preload_tensors(state_dict)
        torch.cuda.synchronize()
        async_request = AsyncRequest(
            TorchAsyncCheckpoint.async_fn, (preloaded_sd, *args), [], kwargs
        )
        self._async_calls_queue.schedule_async_request(async_request)

    def finalize_async_save(self, blocking: bool = False, no_dist=True, terminate=False):
        """Finalizes active async save calls.

        Args:
            blocking (bool, optional): if True, will wait until all active requests
                are done. Otherwise, finalizes only the async request that already
                finished. Defaults to False.
            no_dist (bool, Optional): if True, training ranks simply check its
                asynchronous checkpoint writer without synchronization.
            terminate (bool, optional): if True, the asynchronous queue will
                be closed as the last action of this function.
        """
        if blocking and self._async_calls_queue.get_num_unfinalized_calls() > 0:
            if torch.distributed.get_rank() == 0:
                logger.info(
                    'Unfinalized async checkpoint saves. Finalizing them synchronously now.'
                )

        self._async_calls_queue.maybe_finalize_async_calls(blocking, no_dist=no_dist)
        if terminate:
            self._async_calls_queue.close()
