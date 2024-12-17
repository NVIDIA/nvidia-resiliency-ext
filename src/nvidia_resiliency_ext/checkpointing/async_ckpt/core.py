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
This module provides an async utilities which allow to start
a checkpoint save process in the background.
"""

import torch
import logging
from torch import multiprocessing as mp
from collections import deque
from time import time
from typing import Callable, List, NamedTuple, Optional, Tuple, Dict

logger = logging.getLogger(__name__)


class AsyncRequest(NamedTuple):
    """Represents an async request that needs to be scheduled for execution.

    Args:
        async_fn (Callable, optional): async function to call. None represents noop.
        async_fn_args (Tuple): args to pass to `async_fn`.
        finalize_fns (List[Callable]): list of functions to call to finalize the request.
            These functions will be called synchronously after `async_fn` is done
            *on all ranks*.
    """

    async_fn: Optional[Callable]
    async_fn_args: Tuple
    finalize_fns: List[Callable]
    async_fn_kwargs: Dict = {}
    is_frozen: bool = False

    def add_finalize_fn(self, fn: Callable) -> None:
        """Adds a new finalize function to the request.

        Args:
            fn (Callable): function to add to the async request. This function
                will be called *after* existing finalization functions.

        Returns:
            None
        """
        if self.is_frozen:
            raise RuntimeError('Cannot add finalization functions to a frozen AsyncRequest')
        self.finalize_fns.append(fn)

    def execute_sync(self) -> None:
        """Helper to synchronously execute the request.

        This logic is equivalent to what should happen in case of the async call.
        """
        if self.async_fn is not None:
            self.async_fn(*self.async_fn_args)
        torch.distributed.barrier()
        for finalize_fn in self.finalize_fns:
            finalize_fn()

    def freeze(self) -> 'AsyncRequest':
        """Freezes the async request, disallowing adding new finalization functions.

        Returns:
            AsyncRequest: new async request with all same fields except for the
                `is_frozen` flag.
        """
        return self._replace(is_frozen=True)


class DistributedAsyncCaller:
    """Wrapper around mp.Process that ensures correct semantic of distributed finalization.

    Starts process asynchronously and allows checking if all processes on all ranks are done.
    """

    def __init__(self):
        self.process: Optional[mp.Process] = None
        self.start_time: Optional[float] = None

    def schedule_async_call(
        self, async_fn: Optional[Callable], save_args: Tuple, save_kwargs: Dict = {}
    ) -> None:
        """Spawn a process with `async_fn` as the target.

        This method must be called on all ranks.

        Args:
            async_fn (Callable, optional): async function to call. If None,
                no process will be started.
            save_args (Tuple): async function args.
            save_kwargs (Dict): async function kwargs. Default is None
        """
        if async_fn is None:
            return  # nothing to do
        start_sync = time()
        torch.cuda.synchronize()
        end_sync = time()
        logger.debug(
            f"rank: {torch.distributed.get_rank()}, takes {end_sync - start_sync} to finish D2H "
        )

        ctx = mp.get_context('fork')
        self.start_time = time()
        self.process = ctx.Process(target=async_fn, args=save_args, kwargs=save_kwargs)
        self.process.start()
        init_time = time()
        logger.debug(
            f"rank: {torch.distributed.get_rank()}, takes {init_time - self.start_time} to schedule async ckpt "
        )

    def is_current_async_call_done(self, blocking=False, no_dist=True) -> bool:
        """Check if async save is finished on all ranks.

        For semantic correctness, requires rank synchronization in each check.
        This method must be called on all ranks.

        Args:
            blocking (bool, optional): if True, will wait until the call is done
                on all ranks. Otherwise, returns immediately if at least one rank
                is still active. Defaults to False.
            no_dist (bool, Optional): if True, training ranks simply check its
                asynchronous checkpoint writer without synchronization.

        Returns:
            bool: True if all ranks are done (immediately of after active wait
                if `blocking` is True), False if at least one rank is still active.
        """
        # The following takes the same overhead as torch.distributed.barrier (single integer all-reduce)
        is_alive = int(self.process.is_alive()) if self.process is not None else 0

        is_done = is_alive
        if not no_dist:
            ten = torch.tensor([is_alive], dtype=torch.int, device=torch.cuda.current_device())
            logger.debug(
                f"rank: {torch.distributed.get_rank()}, DistributedAsyncCaller is_alive: {is_alive}"
            )
            torch.distributed.all_reduce(ten)
            is_done = ten[0]

        if is_alive > 0 and not blocking:
            return False
        else:
            if self.process is not None:
                logger.debug(f"rank: {torch.distributed.get_rank()}, joining self.process")
                self.process.join()
                self.process = None

                logger.debug(
                    f"DistributedAsyncCaller: {torch.distributed.get_rank()}, Async process join finished after {time() - self.start_time:.2f}s from forking"
                )
                self.start_time = None
            return True


class _ActiveAsyncRequest(NamedTuple):
    """Helper to represent an active async call.

    Args:
        idx (int): index of the call (starting from 0)
        async_caller (DistributedAsyncCaller): async caller instance that represents
            the async process handling the async request
        async_request (AsyncRequest):  async request that is being called
    """

    idx: int
    async_caller: DistributedAsyncCaller
    async_request: AsyncRequest


class AsyncCallsQueue:
    """Manages a queue of async calls.

    Allows adding a new async call with `schedule_async_request` and finalizing
    active calls with `maybe_finalize_async_calls`.
    """

    def __init__(self):
        self.async_calls: deque[_ActiveAsyncRequest] = deque([])
        self.call_idx: int = -1

    def schedule_async_request(self, async_request: AsyncRequest) -> int:
        """Start a new async call and add it to a queue of active async calls.

        This method must be called on all ranks.

        Args:
            async_request (AsyncRequest): async request to start.

        Returns:
            int: index of the async call that was started.
                This can help the user keep track of the async calls.
        """
        self.call_idx += 1
        async_caller = DistributedAsyncCaller()
        async_request = async_request.freeze()
        async_caller.schedule_async_call(
            async_request.async_fn, async_request.async_fn_args, async_request.async_fn_kwargs
        )
        self.async_calls.append(_ActiveAsyncRequest(self.call_idx, async_caller, async_request))
        return self.call_idx

    def maybe_finalize_async_calls(self, blocking=False, no_dist=True) -> List[int]:
        """Finalizes all available calls.

        This method must be called on all ranks.

        Args:
            blocking (bool, optional): if True, will wait until all active requests
                are done. Otherwise, finalizes only the async request that already
                finished. Defaults to False.
            no_dist (bool, Optional): if True, training ranks simply check its
                asynchronous checkpoint writer without synchronization.

        Returns:
            List[int]: list of indices (as returned by `schedule_async_request`)
                of async calls that have been successfully finalized.
        """
        call_idx_finalized = []
        while self.async_calls:
            next_async_done = self.async_calls[0].async_caller.is_current_async_call_done(
                blocking, no_dist
            )
            if not next_async_done:
                break
            call_idx, _, async_request = self.async_calls.popleft()
            for finalize_fn in async_request.finalize_fns:
                finalize_fn()
            call_idx_finalized.append(call_idx)
        return call_idx_finalized

    def get_num_unfinalized_calls(self):
        """Get the number of active async calls."""
        return len(self.async_calls)

    def close(self):
        """Finalize all calls upon closing."""
        self.maybe_finalize_async_calls(blocking=True)
