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
import gc
import logging
import os
import signal
import subprocess
import weakref
from abc import ABC, abstractmethod
from collections import deque
from queue import Empty
from time import sleep, time
from typing import Callable, ClassVar, Dict, List, NamedTuple, Optional, Tuple

import torch
from torch import multiprocessing as mp

from ..utils import _disable_gc, debug_time

logger = logging.getLogger(__name__)


def _set_process_qos(cpu_priority: int, io_priority: Optional[int]) -> None:
    """
    Set QoS (Quality of Service) for the current checkpoint writer process.
    This ensures checkpoint writing doesn't interfere with training.

    Args:
        cpu_priority: Nice value for CPU scheduling (0-19, higher = lower priority).
                     Default 10 is moderately deprioritized.
        io_priority: ionice scheduling class. If None, I/O priority is unchanged.
                    Valid values (0-3):
                      0 = none/unspecified (kernel default)
                      1 = realtime — **highest** I/O priority; pre-empts all other I/O.
                          NOT recommended for checkpoint workers; will starve training I/O.
                      2 = best-effort (OS default)
                      3 = idle — lowest priority; runs only when no other process needs I/O.
                          Recommended for checkpoint deprioritization.

    Note: Requires appropriate permissions. Failures are logged but not fatal.
    """
    pid = os.getpid()

    # Set CPU priority (nice value). os.nice(increment) adds to current;
    # get current with os.nice(0). Only increase nice (deprioritize);
    # decreasing requires superuser.
    if cpu_priority is not None and cpu_priority >= 0 and cpu_priority <= 19:
        try:
            current_nice = os.nice(0)  # 0 = no change, returns current nice value
            increment = cpu_priority - current_nice
            if increment <= 0:
                logger.warning(
                    "PID %s: Skipping CPU nice (current %s already >= target %s; "
                    "lowering requires superuser",
                    pid,
                    current_nice,
                    cpu_priority,
                )
            else:
                new_nice = os.nice(increment)
                logger.debug(
                    "PID %s: Set CPU nice from %s to %s (target %s)",
                    pid,
                    current_nice,
                    new_nice,
                    cpu_priority,
                )
        except (OSError, PermissionError) as e:
            logger.warning(f"PID {pid}: Failed to set CPU priority: {e}")

    # Set I/O priority (ionice) - Linux only
    if io_priority is not None:
        if io_priority not in range(4):
            logger.warning(
                f"PID {pid}: Invalid io_priority {io_priority!r}; must be 0-3. Skipping ionice."
            )
        else:
            if io_priority <= 2:
                logger.warning(
                    f"PID {pid}: io_priority={io_priority} will NOT deprioritize I/O "
                    f"(class 1=realtime escalates priority, class 2=best-effort is OS default). "
                    f"Use io_priority=3 (idle) to deprioritize checkpoint I/O. Proceeding anyway."
                )
            try:
                # ionice -c <class> -p <pid>
                # class 3 = idle (only when no other process needs I/O)
                # class 2 = best-effort (default, can set priority 0-7)
                subprocess.run(
                    ["ionice", "-c", str(io_priority), "-p", str(pid)],
                    check=True,
                    capture_output=True,
                )
                logger.debug(f"PID {pid}: Set I/O priority class to {io_priority}")
            except (subprocess.CalledProcessError, FileNotFoundError, PermissionError) as e:
                logger.warning(f"PID {pid}: Failed to set I/O priority: {e}")


class AsyncRequest(NamedTuple):
    """Represents an async request that needs to be scheduled for execution.

    Args:
        async_fn (Callable, optional): async function to call. None represents noop.
        async_fn_args (Tuple): args to pass to `async_fn`.
        finalize_fns (List[Callable]): list of functions to call to finalize the request.
            These functions will be called synchronously after `async_fn` is done
            *on all ranks*.
        async_fn_kwargs (Tuple): kwargs to pass to `async_fn`.
        preload_fn (Callable): preload function to stage tensors from GPU to Host.
            This should be self-contained with a proper list of arguments with  `partial`.
        is_frozen (Bool): a flag to indicate this async request can be modified or not.
        call_idx (int): index variable used to order async requests for synchronization
                        in preloading and writing tensors on the async caller

    """

    async_fn: Optional[Callable]
    async_fn_args: Tuple
    finalize_fns: List[Callable]
    async_fn_kwargs: Optional[Dict] = None
    preload_fn: Callable = None
    is_frozen: bool = False
    call_idx: int = 0

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
        # preload tensors.
        async_fn_args = list(self.async_fn_args)
        if self.preload_fn:
            # the 2nd arg is state dict
            async_fn_args[1] = self.preload_fn()
        # persist the state
        if self.async_fn is not None:
            async_fn_kwargs = dict(self.async_fn_kwargs or {})
            self.async_fn(*async_fn_args, **async_fn_kwargs)
        # This utility implements a sync cp save. Hence the barrier.
        torch.distributed.barrier()
        # Finalize the CP state
        self.execute_finalize_fns(validate_matching_call_idx=False)

    def freeze(self) -> 'AsyncRequest':
        """Freezes the async request, disallowing adding new finalization functions.

        Returns:
            AsyncRequest: new async request with all same fields except for the
                `is_frozen` flag.
        """
        return self._replace(is_frozen=True)

    def execute_finalize_fns(self, validate_matching_call_idx: bool = True) -> int:
        """Execute all the finalize functions associated with this async request.

        Args:
            validate_matching_call_idx (bool, optional): Validate that all ranks
                invoke CP finalize on the same call_idx. This is typically useful in
                async CP stages where multiple CP requests can be pending.
                This validation is unnecessary during synchronous CP step.
                When this param is True, an AllReduce Sync across all participating
                ranks is invoked. Default set to True for conservative validation.

        Returns:
            call_idx: The call_idx of async request that has been finalized
        """
        with debug_time("finalize", logger):
            for finalize_fn in self.finalize_fns:
                finalize_fn()

            # Validate that matching call_idx are invoked from all ranks.
            # This ensures all ranks are correctly participating in CP save invocations
            if validate_matching_call_idx:
                ten = torch.tensor(
                    [self.call_idx], dtype=torch.int, device=torch.cuda.current_device()
                )
                torch.distributed.all_reduce(ten, op=torch.distributed.ReduceOp.MAX)
                assert ten.item() == self.call_idx, "Unmatched async calls. "
                "That probably means not all ranks are participating in async finalization"
        return self.call_idx


class ObjectTracker(type):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls._instances = weakref.WeakSet()

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        cls._instances.add(instance)
        return instance

    def get_instances(cls):
        return list(cls._instances)


class AsyncCaller(ABC):
    """Wrapper around mp.Process that ensures correct semantic of distributed finalization.

    Starts process asynchronously and allows checking if all processes on all ranks are done.
    """

    def __init__(self):
        self.process: Optional[mp.Process] = None
        self.start_time: Optional[float] = None
        # Store the rank for logging, in case torch.distributed is destroyed
        # before AsyncCaller shuts down.
        self.rank: int = None

    @abstractmethod
    def schedule_async_call(self, async_req: AsyncRequest) -> None:
        """Schedule `async_req` with some process forking or reusing
           persistent worker

        This method must be called on all ranks.

        Args:
            async_req (AsyncRequest): `AsyncRequest` object containing to
                                       start async process
        """
        raise NotImplementedError("This should be implemented")

    @abstractmethod
    def is_current_async_call_done(self, blocking: bool, no_dist: bool) -> bool:
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
        raise NotImplementedError("This should be implemented")

    def sync_all_async_calls(self, is_alive: int) -> bool:
        """Check if all ranks have completed async checkpoint writing

        Args:
            is_alive (bool): if True, the current async request is not completed

        Returns:
            bool: True if all ranks are done, False if at least one rank is still active.

        """
        ten = torch.tensor([is_alive], dtype=torch.int, device=torch.cuda.current_device())
        torch.distributed.all_reduce(ten)
        return ten[0] == 0

    @abstractmethod
    def close(self, abort=False):
        """Terminate the async caller at exit of an application or some termination conditions

        Args:
            abort (bool, optional): Default to False. Needs to be manually set to true when
                the checkpoint async process needs to be aborted.
        """
        logger.info(f"AsyncCaller: {torch.distributed.get_rank()}, Destroying Async Caller")

    @abstractmethod
    def __del__(self):
        raise NotImplementedError("This should be implemented")


class TemporalAsyncCaller(AsyncCaller):
    """Wrapper around mp.Process that ensures correct semantic of distributed finalization.

    Starts process asynchronously and allows checking if all processes on all ranks are done.
    """

    def __init__(self):
        super().__init__()
        self.preloaded_holder = None

    @_disable_gc()
    def schedule_async_call(self, async_req: AsyncRequest) -> None:
        """Spawn a process with `async_fn` as the target.

        This method must be called on all ranks.

        Args:
            async_fn (Callable, optional): async function to call. If None,
                no process will be started.
            async_req (AsyncRequest): `AsyncRequest` object containing to
                                       start async process
        """
        if async_req.async_fn is None:
            return  # nothing to do

        async_fn_args = list(async_req.async_fn_args)
        if async_req.preload_fn:
            # If there's a preload_fn in `async_req`, we call this func
            # to do the defined action in `async_req.preload_fn` to
            # stage GPU tensors to its defined destination
            async_fn_args[1] = async_req.preload_fn()
            self.preloaded_holder = async_fn_args[1]

        if self.rank is None:
            self.rank = torch.distributed.get_rank()

        start_sync = time()
        torch.cuda.synchronize()
        end_sync = time()
        logger.debug(f"rank: {self.rank}, takes {end_sync - start_sync} to finish D2H ")

        ctx = mp.get_context('fork')
        self.start_time = time()
        async_fn_kwargs = dict(async_req.async_fn_kwargs or {})
        self.process = ctx.Process(
            target=async_req.async_fn, args=async_fn_args, kwargs=async_fn_kwargs
        )
        self.process.start()
        init_time = time()
        logger.debug(
            f"rank: {self.rank}, takes {init_time - self.start_time} to schedule async ckpt "
        )

    def is_current_async_call_done(self, blocking: bool = False, no_dist: bool = False) -> bool:
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
        # The following takes the same overhead
        # as torch.distributed.barrier (single integer all-reduce)
        is_alive = int(self.process.is_alive()) if self.process is not None else 0
        is_done = not is_alive if no_dist else self.sync_all_async_calls(is_alive)

        if is_done or blocking:
            # Process join is called in the following cases
            # 1. blocking == True -> regardless of is_done
            # 2. blocking == False (non-blocking)
            #    -> is_done == True: async requests on all ranks are identified to be finished
            #    `self.close()` makes sure the async callers terminated
            self.close()
            is_done = True
        return is_done

    def close(self, abort=False):
        """For TemporalAsyncCaller, this method is called explictly in `is_current_async_call_done`

        This method make sure the TemporalAsyncCaller terminated
        with all its assigned async request completed

        Args:
            abort (bool, optional): Default to False. Needs to be manually set to true when
                the checkpoint async process needs to be aborted.
        """
        if self.process:
            logger.debug(f"rank: {self.rank}, joining self.process")
            if abort:
                logger.warning(f"Temporal worker aborted in rank {self.rank}")
                self.process.kill()
            else:
                self.process.join()
            self.process = None
            logger.debug(
                "TemporalAsyncCaller: Async process join finished "
                f"after {time() - self.start_time:.2f}s from forking"
            )
            self.start_time = None
            self.preloaded_holder = None

    def __del__(self):
        pass

    def _debug_is_async_process_running(self):
        """Utility for unit test purpose to validate expected state of the async process that performs async CP."""
        if self.process is None:
            return False
        return self.process.is_alive()


class PersistentAsyncCaller(AsyncCaller):
    """Wrapper around mp.Process that ensures correct semantic of distributed finalization.

    Starts process asynchronously and allows checking if all processes on all ranks are done.
    """

    # Worker-side cache for consistent data structures.
    # Key: identifier key
    # Value: (separation_hint, items, resolved_data, thread_count, storage_plan)
    # This cache contains IPC handles and must be cleaned up properly.
    _worker_data_cache: Dict = {}

    # Callbacks invoked in the training process whenever a fresh worker is spawned.
    # Used by FileSystemWriterAsync to invalidate training-side shm caches so the
    # next checkpoint re-sends actual tensor data to the new worker.
    _worker_restart_callbacks: ClassVar[List[Callable]] = []

    @classmethod
    def register_worker_restart_callback(cls, fn: Callable) -> None:
        """Register a callable to be invoked when a new worker process is started."""
        cls._worker_restart_callbacks.append(fn)

    def __init__(
        self,
        is_daemon: bool = True,
        cpu_priority: int = 10,
        io_priority: Optional[int] = None,
        sigterm_timeout: float = 30.0,
        cpu_shm_mode: bool = False,
    ):
        self.process: mp.Process = None
        self.start_time: Optional[float] = None
        self.sigterm_timeout = sigterm_timeout
        ctx = mp.get_context('spawn')
        # main queue to deliver `AsyncRequest` from host to the ckpt worker
        self.queue: mp.JoinableQueue = ctx.JoinableQueue()
        # Queue used to synchronize for the completion of preloading tensors to host
        # between a trainer and ckpt worker
        self.preload_q: mp.JoinableQueue = ctx.JoinableQueue()
        # Queue used to inform trainer when the saving is completed
        self.comp_q: mp.Queue = ctx.Queue()
        self.cur_item: int = None
        self.cur_idx: int = -1
        self.rank: int = None
        # When background_worker_is_daemon flag is True, the async background
        # worker is spawned as a daemon making async worker shutdown cleaner.
        # The restriction of spawning the async worker as a daemon is that
        # the FileWriter performing the FileIO in the background process cannot
        # be parallelized with multi-processing.
        self.background_worker_is_daemon = is_daemon
        self.cpu_priority = cpu_priority
        self.io_priority = io_priority
        self.cpu_shm_mode = cpu_shm_mode

    def _start_worker(self, rank: int) -> None:
        """Start the background worker process.

        Args:
            rank (int): the rank of the current trainer process.
        """
        ctx = mp.get_context('spawn')
        logger.info(f"PersistentAsyncCaller: {rank}, Starting Async Caller")
        if self.background_worker_is_daemon:
            async_loop_target = PersistentAsyncCaller.async_loop_for_daemon_worker
        else:
            async_loop_target = PersistentAsyncCaller.async_loop

        self.process = ctx.Process(
            target=async_loop_target,
            args=(
                rank,
                self.queue,
                self.preload_q,
                self.comp_q,
                logger.getEffectiveLevel(),
                self.cpu_priority,
                self.io_priority,
                self.cpu_shm_mode,
            ),
            daemon=self.background_worker_is_daemon,
        )
        self.process.start()
        logger.debug(f"PersistentAsyncCaller: {rank}, Started Async Caller {self.process}")
        for cb in PersistentAsyncCaller._worker_restart_callbacks:
            cb()

    def schedule_async_call(self, async_req: AsyncRequest) -> None:
        """Put `AsyncRequest` to the Persistent Async Caller

        This method must be called on all ranks. The async_req object is pickled and
        sent to the persistent async worker via a JoinableQueue.
        Therefore, all arguments within async_req must be picklable.

        Args:
            async_fn (Callable, optional): async function to call. If None,
                no process will be started.
            async_req (AsyncRequest): `AsyncRequest` object containing to
                                       schedule a checkpointing request
        """
        if async_req.async_fn is None:
            return  # nothing to do

        if self.rank is None:
            self.rank = torch.distributed.get_rank()

        start_sync = end_sync = None

        self.start_time = time()
        if self.process is None:
            self._start_worker(self.rank)

        if async_req.preload_fn:
            self.preload_q.put(async_req.call_idx)
        self.queue.put(async_req)
        logger.debug(f"rank: {self.rank}, put {async_req.call_idx}")

        if async_req.preload_fn:
            start_sync = time()
            # Synchronize for pre-staging tensors
            self.preload_q.join()
            end_sync = time()

            logger.debug(f"rank: {self.rank}, takes {end_sync - start_sync} to finish D2H ")

        init_time = time()
        logger.debug(
            f"rank: {self.rank}, takes {init_time - self.start_time} " "to schedule async ckpt "
        )

    def is_current_async_call_done(self, blocking: bool = False, no_dist: bool = False) -> bool:
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

        is_alive: bool = False

        if self.process:
            while self.cur_item is None:
                try:
                    # Retrieve comp call_idx without waiting
                    self.cur_item = self.comp_q.get_nowait()
                except Empty:
                    # This method is called after any `AsyncRequest` is pushed to the main loop
                    # So, the background writing is still active
                    # before the worker put call_idx to `comp_q`
                    if not blocking:
                        is_alive = True
                        break
                    sleep(0.1)

        if self.cur_item is not None:
            logger.debug(f"rank: {self.rank}, item: {self.cur_item}" f" is completed, {is_alive}")

        is_done = not is_alive if no_dist else self.sync_all_async_calls(is_alive)
        # This is set to False when blocking == False so this routine is called again
        # to simply call `sync_all_async_calls` to check if other ranks complete the writing
        if is_done:
            # The current request is completed globally. Reset the current item for polling.
            logger.debug(
                f"rank: {self.rank}, item: {self.cur_item}" f" is completed globally, {is_done}"
            )
            self.cur_item = None

        return is_done

    def close(self, abort=False):
        """Wait on the left async requests and terminate the PersistentAsyncCaller

        Signals the PersistentAsyncCaller by sending a 'DONE' message to make it terminated

        Args:
            abort (bool, optional): Default to False. Needs to be manually set to true when
                the checkpoint async process needs to be aborted.
        """
        logger.info(f"PersistentAsyncCaller: {self.rank}, Destroying Async Caller")
        if self.process:
            if abort:
                logger.error(f"Persistent worker aborted in rank {self.rank}")
                # Use SIGTERM first so the worker's signal handler can run
                # cleanup_worker_data_cache() via the try/finally block, releasing
                # CUDA IPC handles before the process exits.
                self.process.terminate()
                self.process.join(timeout=self.sigterm_timeout)
                if self.process.is_alive():
                    logger.warning(
                        f"Persistent worker (rank {self.rank}) did not exit within "
                        f"{self.sigterm_timeout}s after SIGTERM; sending SIGKILL"
                    )
                    # Before SIGKILL, close the queues from the parent side to
                    # release any buffered CUDA IPC handles. Without this, SIGKILL
                    # leaves dangling IPC state that causes SIGSEGV in the parent
                    # during CUDA cleanup at exit.
                    for q in (self.queue, self.preload_q):
                        try:
                            q.cancel_join_thread()
                            q.close()
                        except Exception:
                            pass
                    gc.collect()
                    self.process.kill()
                    self.process.join()
            else:
                self.queue.put('DONE')
                self.queue.join()
                self.process.join()

            self.process = None

    def __del__(self):
        self.close()

    def _debug_is_async_process_running(self):
        """Utility for unit test purpose to validate expected state of the async process that performs async CP."""
        if self.process is None:
            return False
        return self.process.is_alive()

    @classmethod
    def cleanup_worker_data_cache(cls):
        """Clean up the worker data cache and release IPC handles.

        This function should be called when the async worker is being torn down
        to properly release any IPC handles stored in the cache.
        """
        if cls._worker_data_cache:
            logger.info(f"Cleaning up worker data cache with {len(cls._worker_data_cache)} entries")
            # Clear all cached data structures which may contain IPC handles
            cls._worker_data_cache.clear()
            gc.collect()
            torch.cuda.empty_cache()

    @staticmethod
    def async_process_target(
        rank: int,
        queue: mp.JoinableQueue,
        preload_q: mp.JoinableQueue,
        comp_q: mp.Queue,
        log_level: int = logging.INFO,
        cpu_priority: int = 10,
        io_priority: Optional[int] = None,
        cpu_shm_mode: bool = False,
    ):
        """Main function for the persistent checkpoint worker

        The persisent worker is created once and terminated at exit or
        when application calls `close()` explictily

        This routine receives `AsyncRequest` and does `preload_fn` first and
        put the integer value in `preload_q` to inform the trainer to proceed.
        When the `async_fn` from the request` is completed (background saving is done),
        it puts a integer value to `comp_q` to notify the trainer the completion.

        Args:
            rank (int): the rank of the trainer where the persistent worker is created.
            queue (mp.JoinableQueue): the main queue used to receive `AsyncRequest`
                                      from the training rank
            preload_q (mp.JoinableQueue): a queue to inform trainer that preloading of tensors
                                          from GPU to Host or dedicated location is completed
            comp_q (mp.Queue): a queue to inform the training rank the completion of scheduled
                               async checkpoint request
            log_level (int, Optional): an integer to set log-level in this spawned process
                                       to get aligned with the training rank's logging level
            cpu_priority (int): Nice value for CPU scheduling (0-19, higher = lower priority).
                               Default 10 deprioritizes checkpoint writing vs training.
            io_priority (int, Optional): ionice scheduling class (0-3). Default None leaves
                                        I/O priority unchanged. Use 3 (idle) to deprioritize
                                        checkpoint I/O. NOTE: class 1 = realtime (highest
                                        priority — NOT recommended for checkpoint workers).

        """
        # Align library loggers in this process without mutating the root logger
        logging.getLogger("nvidia_resiliency_ext").setLevel(log_level)
        logger = logging.getLogger(__name__)
        if rank == 0:
            logger.info(f"PersistentAsyncCaller: persistent ckpt worker for {rank} has started")
        else:
            logger.debug(f"PersistentAsyncCaller: persistent ckpt worker for {rank} has started")
        if not cpu_shm_mode:
            # Set CUDA device to appropriate local_rank to ensure allocations / CUDA contexts
            # in this new process are on the right device, and device 0 on the node does not
            # take on undue memory burden from other devices on node (default behavior without
            # this line).
            device_id = rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            # Allocate a small dummy tensor to force CUDA context initialization before any
            # IPC handle is received via the queue. This prevents a handle type mismatch
            # between the producer and consumer that otherwise manifests as:
            #   RuntimeError: pidfd_getfd: Bad file descriptor
            # See https://github.com/pytorch/pytorch/issues/179220 for details.
            torch.empty(1, device=f'cuda:{device_id}')

        # Set QoS to deprioritize checkpoint writing vs training.
        # This prevents checkpoint I/O from interfering with data loader.
        _set_process_qos(cpu_priority=cpu_priority, io_priority=io_priority)

        # Register a SIGTERM handler that raises SystemExit so the finally block
        # below runs cleanup_worker_data_cache() and releases CUDA IPC handles.
        # Without this, SIGTERM (sent by close(abort=True)) bypasses Python cleanup,
        # leaving dangling CUDA IPC handles that cause a SIGSEGV in the parent at exit.
        def _handle_sigterm(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, _handle_sigterm)

        # Start busy loop waiting for and executing checkpoint saves.
        try:
            while True:
                item = queue.get()
                if isinstance(item, str) and item == 'DONE':
                    queue.task_done()
                    break
                elif isinstance(item, AsyncRequest):
                    async_fn_args = list(item.async_fn_args)
                    if item.preload_fn:
                        call_idx = preload_q.get()
                        # the 2nd arg is state dict
                        async_fn_args[1] = item.preload_fn()
                        logger.debug(f"{rank} has completed D2H of {call_idx}")
                        preload_q.task_done()
                    if item.async_fn is not None:
                        async_fn_kwargs = dict(item.async_fn_kwargs or {})
                        item.async_fn(*async_fn_args, **async_fn_kwargs)
                    logger.debug(f"{rank} has completed saving {item.call_idx}")
                    comp_q.put(item.call_idx)
                    queue.task_done()
                    del async_fn_args
                del item
                gc.collect()
        except RuntimeError as e:
            if "pidfd_getfd" in str(e) and "Operation not permitted" in str(e):
                raise RuntimeError(
                    "Failed to receive CUDA IPC handle from the training process "
                    "(pidfd_getfd: Operation not permitted). This is a kernel security "
                    "restriction. To allow cross-process file-descriptor passing, run: "
                    "  sudo sysctl kernel.yama.ptrace_scope=0"
                ) from e
            raise
        finally:
            # Cleanup worker data cache before exiting, regardless of how the loop exits
            # (normal termination via 'DONE' sentinel or unhandled exception).
            PersistentAsyncCaller.cleanup_worker_data_cache()
        if rank == 0:
            logger.info(f"PersistentAsyncCaller: persistent ckpt worker for {rank} has terminated")
        else:
            logger.debug(f"PersistentAsyncCaller: persistent ckpt worker for {rank} has terminated")

    @staticmethod
    @_disable_gc()
    def async_loop(
        rank: int,
        queue: mp.JoinableQueue,
        preload_q: mp.JoinableQueue,
        comp_q: mp.Queue,
        log_level: int = logging.INFO,
        cpu_priority: int = 10,
        io_priority: Optional[int] = None,
        cpu_shm_mode: bool = False,
    ):
        """
        Main function for the persistent checkpoint worker called by a non daemon async process.
        In this loop, child processes may be created (For example: to parallelize File IO)
        """
        PersistentAsyncCaller.async_process_target(
            rank, queue, preload_q, comp_q, log_level, cpu_priority, io_priority, cpu_shm_mode
        )

    @staticmethod
    def async_loop_for_daemon_worker(
        rank: int,
        queue: mp.JoinableQueue,
        preload_q: mp.JoinableQueue,
        comp_q: mp.Queue,
        log_level: int = logging.INFO,
        cpu_priority: int = 10,
        io_priority: Optional[int] = None,
        cpu_shm_mode: bool = False,
    ):
        """
        Main function for the persistent checkpoint worker called by a daemon async process
        """
        PersistentAsyncCaller.async_process_target(
            rank, queue, preload_q, comp_q, log_level, cpu_priority, io_priority, cpu_shm_mode
        )


class _ActiveAsyncRequest(NamedTuple):
    """Helper to represent an active async call.

    Args:
        idx (int): index of the call (starting from 0)
        async_caller (DistributedAsyncCaller): async caller instance that represents
            the async process handling the async request
        async_request (AsyncRequest):  async request that is being called
    """

    idx: int
    async_caller: AsyncCaller
    async_request: AsyncRequest


class AsyncCallsQueue(metaclass=ObjectTracker):
    """Manages a queue of async calls.

    Allows adding a new async call with `schedule_async_request` and finalizing
    active calls with `maybe_finalize_async_calls`.
    """

    # Class-level slot for a pre-warmed persistent caller (set by warmup_persistent_caller).
    # Consumed on first use by _get_async_caller.
    _warmup_persistent_caller: Optional[PersistentAsyncCaller] = None

    def __init__(
        self,
        persistent: bool = True,
        is_daemon: bool = True,
        cpu_priority: int = 10,
        io_priority: Optional[int] = None,
        sigterm_timeout: float = 30.0,
        cpu_shm_mode: bool = False,
    ):
        self.async_calls: deque[_ActiveAsyncRequest] = deque([])
        self.call_idx: int = -1
        self.persistent: bool = persistent
        self.is_daemon: bool = is_daemon
        self.cpu_priority = cpu_priority
        self.io_priority = io_priority
        self.sigterm_timeout = sigterm_timeout
        self.cpu_shm_mode = cpu_shm_mode
        self.persistent_caller: AsyncCaller = None

    def _get_async_caller(self):
        if not self.persistent:
            logger.warning("The TemporalAsyncCaller will be deprecated soon. ")
            return TemporalAsyncCaller()
        if self.persistent_caller is None:
            # Consume the pre-warmed caller if available.
            # No locking needed: _warmup_persistent_caller is a class-level variable that is not
            # shared across processes (each process has its own copy), and we expect only the
            # main trainer thread to call this routine, so there is no concurrent access.
            if AsyncCallsQueue._warmup_persistent_caller is not None:
                warmed = AsyncCallsQueue._warmup_persistent_caller
                AsyncCallsQueue._warmup_persistent_caller = None
                if warmed.process is not None and not warmed.process.is_alive():
                    logger.warning(
                        "Pre-warmed async caller process (PID %s) is no longer alive; "
                        "starting a fresh worker.",
                        warmed.process.pid,
                    )
                    warmed.process.join()  # reap the zombie before discarding
                    warmed.process = None
                self.persistent_caller = warmed
            else:
                self.persistent_caller = PersistentAsyncCaller(
                    is_daemon=self.is_daemon,
                    cpu_priority=self.cpu_priority,
                    io_priority=self.io_priority,
                    sigterm_timeout=self.sigterm_timeout,
                    cpu_shm_mode=self.cpu_shm_mode,
                )
        return self.persistent_caller

    @classmethod
    def warmup_persistent_caller(
        cls,
        rank: int,
        is_daemon: bool = True,
        cpu_priority: int = 10,
        io_priority: Optional[int] = None,
        sigterm_timeout: float = 30.0,
    ):
        """Pre-start the persistent async worker to avoid startup latency on the first checkpoint.

        Args:
            rank (int): the current distributed rank.
            is_daemon (bool): whether to spawn the worker as a daemon process.
            cpu_priority (int): Nice value for CPU scheduling (0-19, higher = lower priority).
            io_priority (int, Optional): ionice scheduling class (0-3). Use 3 (idle) to
                deprioritize checkpoint I/O. NOTE: class 1 = realtime (highest priority —
                NOT recommended for checkpoint workers).
            sigterm_timeout (float): seconds to wait after SIGTERM before escalating to SIGKILL.
        """
        if cls._warmup_persistent_caller is None:
            caller = PersistentAsyncCaller(
                is_daemon=is_daemon,
                cpu_priority=cpu_priority,
                io_priority=io_priority,
                sigterm_timeout=sigterm_timeout,
            )
            caller._start_worker(rank)
            caller.rank = rank
            cls._warmup_persistent_caller = caller

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
        # For CPU shm path: shm tensors are shared between training and worker, so the
        # previous write must complete before training overwrites them with new values.
        # Drain any pending writes now, before dispatching the next checkpoint.
        # (For the GPU IPC path this is a no-op since write_fence is not set.)
        if getattr(async_request.preload_fn, 'write_fence', False) and self.async_calls:
            self.maybe_finalize_async_calls(blocking=True, no_dist=True)
        async_caller = self._get_async_caller()
        # Backward compatibility for local checkpointing built with the old AsyncRequest
        if len(async_request._fields) != len(AsyncRequest._fields):
            async_request = AsyncRequest(**async_request._asdict())
        async_request = async_request.freeze()
        async_caller.schedule_async_call(
            async_request._replace(call_idx=self.call_idx, finalize_fns=[])
        )
        self.async_calls.append(_ActiveAsyncRequest(self.call_idx, async_caller, async_request))
        return self.call_idx

    def maybe_finalize_async_calls(self, blocking=False, no_dist=False) -> List[int]:
        """Finalizes all available calls.

        This method must be called on all ranks.

        Args:
            blocking (bool, optional): if True, will wait until all active requests
                are done. Otherwise, finalizes only the async request that already
                finished. Defaults to False.
            no_dist (bool, optional): if True, training ranks simply check its
                asynchronous checkpoint writer without synchronization.
        Returns:
            List[int]: list of indices (as returned by `schedule_async_request`)
                of async calls that have been successfully finalized.
        Raises:
            CheckpointException: if any rank(s) raised an exception during checkpoint
                writing, the exceptions are wrapped and raised on all ranks.
        """
        call_idx_finalized = []
        while self.async_calls:
            next_async_done = self.async_calls[0].async_caller.is_current_async_call_done(
                blocking, no_dist
            )
            if not next_async_done:
                break
            with debug_time("finalize", logger):
                _, _, async_request = self.async_calls.popleft()
                call_idx = async_request.execute_finalize_fns(
                    validate_matching_call_idx=(not no_dist)
                )
                call_idx_finalized.append(call_idx)
        return call_idx_finalized

    def get_num_unfinalized_calls(self):
        """Get the number of active async calls."""
        return len(self.async_calls)

    def close(self, abort=False):
        """Finalize all calls upon closing.

        Args:
            abort (bool, optional): Default to False. Needs to be manually set to true when
                the checkpoint async process needs to be aborted.
        """
        # For a clean shut down scenario with valid async processes running,
        # finalize all pending async calls
        if not abort and (self.persistent is False or self.persistent_caller is not None):
            self.maybe_finalize_async_calls(blocking=True)
        if self.persistent and self.persistent_caller:
            self.persistent_caller.close(abort=abort)

        # Clean up any pre-warmed worker that was never consumed by a checkpoint schedule.
        if AsyncCallsQueue._warmup_persistent_caller is not None:
            AsyncCallsQueue._warmup_persistent_caller.close(abort=abort)
            AsyncCallsQueue._warmup_persistent_caller = None

        # Reset all class params
        self.call_idx = -1
        self.persistent_caller = None

    def __del__(self):
        """Ensure clean closure of AsyncCallsQueue at object deletion"""
        self.close()


def abort_nvrx_checkpoint():
    """Abort NVRx Checkpoint Utility. This will close the AsyncCallsQueue that manages async checkpoints"""
    # close the async calls queue which will ensure a clean restart
    # of the CP async process in subsequent async save requests.
    for async_queue in AsyncCallsQueue.get_instances():
        async_queue.close(abort=True)
