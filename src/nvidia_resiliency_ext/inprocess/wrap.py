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

import contextlib
import datetime
import functools
import gc
import logging
import pathlib
import sys
import threading
import time
import warnings
from datetime import timedelta
from typing import Any, Optional

import torch

from . import param_utils, utils
from .abort import Abort, AbortTorchDistributed
from .attribution import Interruption, InterruptionRecord
from .compose import Compose
from .exception import HealthCheckError, InternalError
from .finalize import Finalize
from .health_check import HealthCheck
from .initialize import Initialize
from .monitor_process import MonitorProcess
from .monitor_thread import MonitorThread, RankShouldRestart, reraise_if_unraisable
from .param_utils import enforce_subclass, enforce_type, enforce_value
from .progress_watchdog import ProgressWatchdog
from .rank_assignment import ActivateAllRanks, RankAssignment, RankAssignmentCtx, ShiftRanks
from .rank_filter import RankFilter
from .state import Mode, State
from .store import PrefixStore, StoreMixin, TCPStore
from .utils import log_exc


class HealthCheckPassed(Exception):
    pass


def reserve_fn(state, store, progress_watchdog, progress_watchdog_interval):
    log = logging.getLogger(__name__)
    rank = state.rank

    log.debug(f'{rank=} starting reserve_fn')
    progress_watchdog.ping()

    while True:
        try:
            store.wait_for_completed(progress_watchdog_interval)
            break
        except torch.distributed.DistStoreError:
            progress_watchdog.ping()
            time.sleep(sys.getswitchinterval())

    log.debug(f'{rank=} finished reserve_fn')


class Wrapper:
    r'''
    Python function wrapper that adds restart capabilities to an existing
    Python function implementing distributed PyTorch workload.

    Upon a fault, the wrapped function is restarted across all distributed
    ranks, within the same operating system process. Wrapped function restart
    invocation excludes distributed ranks that are terminated, missing, or
    deemed unhealthy. When a failure occurs on any worker, the wrapper ensures
    the function restarts simultaneously on all healthy ranks. This process
    continues until all ranks complete execution successfully or a termination
    condition is met.

    See the :doc:`Usage Guide <../usage_guide>` for detailed documentation.

    Args:
        store_factory: Factory to construct the internal distributed store for
            communication between ranks.
        store_kwargs: Dictionary of keyword arguments to construct the internal
            store with ``store_factory(**store_kwargs)``.
        initialize: Rank-local initialize.
        abort: Asynchronously aborts execution.
        finalize: Rank-local finalize.
        health_check: Rank-local health check.
        rank_assignment: Reassigns ranks, computes the new world size and
            specifies which ranks are calling the wrapped function.
        rank_filter: (DEPRECATED) Specifies ranks actively calling the wrapped
            function.
        monitor_thread_interval: Monitoring interval for the monitor thread.
        monitor_process_interval: Monitoring interval for the monitor process.
        heartbeat_interval: Monitoring interval for detecting unresponsive
            ranks.
        progress_watchdog_interval: Interval for automatic progress watchdog
            timestamp updates.
        soft_timeout: Soft progress timeout. Timed-out rank executes
            asynchronous abort, and participates in the restart if healthy.
        hard_timeout: Hard progress timeout. Timed-out rank is terminated.
        heartbeat_timeout: Timeout for a missing rank detection heartbeat.
        barrier_timeout: Barrier timeout.
        completion_timeout: Completion barrier timeout.
        last_call_wait: Time interval for other ranks to report concurrent
            terminal failures.
        termination_grace_time: Interval between ``SIGTERM`` and ``SIGKILL``
            signals issued by the hard timeout mechanism.
        monitor_process_logfile: Absolute filepath for the monitor process
            logfile. It may contain "{rank}" placeholder, to be filled with
            initial integer rank id.
        enabled: Enables the wrapper.

    Returns:
        Returns the value of the wrapped function if all ranks successfully
        completed execution.
        Inactive ranks return :py:obj:`None`.
    '''

    def __init__(
        self,
        *,
        store_factory: type[StoreMixin] = TCPStore,
        store_kwargs: Optional[dict[str, Any]] = None,
        initialize: Optional[Initialize] = None,
        abort: Optional[Abort] = AbortTorchDistributed(),
        finalize: Optional[Finalize] = None,
        health_check: Optional[HealthCheck] = None,
        rank_assignment: RankAssignment = Compose(
            ActivateAllRanks(),
            ShiftRanks(),
        ),
        rank_filter: Optional[RankFilter] = None,
        monitor_thread_interval: datetime.timedelta = timedelta(seconds=1),
        monitor_process_interval: datetime.timedelta = timedelta(seconds=1),
        heartbeat_interval: datetime.timedelta = timedelta(seconds=1),
        progress_watchdog_interval: datetime.timedelta = timedelta(seconds=1),
        soft_timeout: datetime.timedelta = timedelta(seconds=60),
        hard_timeout: datetime.timedelta = timedelta(seconds=90),
        heartbeat_timeout: datetime.timedelta = timedelta(seconds=30),
        barrier_timeout: datetime.timedelta = timedelta(seconds=120),
        completion_timeout: datetime.timedelta = timedelta(seconds=120),
        last_call_wait: datetime.timedelta = timedelta(seconds=1),
        termination_grace_time: datetime.timedelta = timedelta(seconds=5),
        monitor_process_logfile: Optional[str] = None,
        enabled: bool = True,
    ):
        enforce_subclass('store_factory', StoreMixin)
        enforce_type('store_kwargs', (dict, type(None)))
        enforce_type('initialize', (Initialize, type(None)))
        enforce_type('abort', (Abort, type(None)))
        enforce_type('finalize', (Finalize, type(None)))
        enforce_type('health_check', (HealthCheck, type(None)))
        enforce_type('rank_assignment', RankAssignment)
        enforce_type('rank_filter', (RankFilter, type(None)))
        enforce_type('monitor_thread_interval', datetime.timedelta)
        enforce_type('monitor_process_interval', datetime.timedelta)
        enforce_type('heartbeat_interval', datetime.timedelta)
        enforce_type('progress_watchdog_interval', datetime.timedelta)
        enforce_type('soft_timeout', datetime.timedelta)
        enforce_type('hard_timeout', datetime.timedelta)
        enforce_type('heartbeat_timeout', datetime.timedelta)
        enforce_type('barrier_timeout', datetime.timedelta)
        enforce_type('completion_timeout', datetime.timedelta)
        enforce_type('last_call_wait', datetime.timedelta)
        enforce_type('termination_grace_time', datetime.timedelta)
        enforce_type('monitor_process_logfile', (str, type(None)))
        enforce_type('enabled', bool)

        enforce_value(soft_timeout < hard_timeout < barrier_timeout)
        enforce_value(monitor_process_interval < barrier_timeout)
        enforce_value(heartbeat_timeout < barrier_timeout)
        enforce_value(heartbeat_interval < heartbeat_timeout)
        enforce_value(monitor_process_interval < heartbeat_timeout)
        enforce_value(monitor_process_interval < soft_timeout)
        enforce_value(monitor_thread_interval < soft_timeout)
        enforce_value(progress_watchdog_interval < soft_timeout)

        if monitor_process_logfile is not None:
            enforce_value(pathlib.Path(monitor_process_logfile).is_absolute())

        enforce_value(torch.distributed.is_available())

        if rank_filter is not None:
            warnings.warn(
                'The "rank_filter" argument is deprecated and will be removed '
                'in the next release. The functionality is merged into '
                '"rank_assignment".',
                DeprecationWarning,
                stacklevel=2,
            )

        if store_kwargs is None:
            store_kwargs = {}

        self.store_factory = store_factory
        self.store_kwargs = store_kwargs
        self.initialize = initialize
        self.abort = abort
        self.finalize = finalize
        self.health_check = health_check
        self.rank_assignment = rank_assignment
        self.rank_filter = rank_filter
        self.monitor_thread_interval = monitor_thread_interval
        self.monitor_process_interval = monitor_process_interval
        self.heartbeat_interval = heartbeat_interval
        self.progress_watchdog_interval = progress_watchdog_interval
        self.soft_timeout = soft_timeout
        self.hard_timeout = hard_timeout
        self.heartbeat_timeout = heartbeat_timeout
        self.barrier_timeout = barrier_timeout
        self.completion_timeout = completion_timeout
        self.last_call_wait = last_call_wait
        self.termination_grace_time = termination_grace_time
        self.monitor_process_logfile = monitor_process_logfile
        self.enabled = enabled

    def __call__(self, fn):
        if not self.enabled:
            return fn

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with CallWrapper(self) as call_wrapper:
                return call_wrapper(fn, args, kwargs)

        return wrapped


class CallWrapper:
    r'''
    The :py:class:`CallWrapper` encapsulates the state and execution flow of
    the restart capabilities of the :py:class:`Wrapper` for a single invocation
    of the wrapped function. This design ensures that each call operates
    independently, with the restart state tied to the specific invocation
    rather than the function's definition.


    The :py:class:`CallWrapper` is automatically created by the
    :py:class:`Wrapper` with every invocation of the wrapped function. The
    active :py:class:`CallWrapper` instance is then passed as the value for any
    function argument annotated with ``CallWrapper`` or
    ``typing.Optional[CallWrapper]``. This allows the wrapped function to
    access and interact with the state of the :py:class:`Wrapper` during its
    execution.
    '''

    def __init__(self, wrapper: Wrapper):
        self.monitor_process = None
        self.progress_watchdog = None
        self.base_store = None
        self.state = None

        try:
            utils.Logging.initialize()
            log = logging.getLogger(__name__)

            enforce_value(not torch.distributed.is_initialized())

            state = State.from_env()

            self.monitor_process = MonitorProcess(
                rank=state.rank,
                world_size=state.world_size,
                soft_timeout=wrapper.soft_timeout,
                hard_timeout=wrapper.hard_timeout,
                termination_grace_time=wrapper.termination_grace_time,
                barrier_timeout=wrapper.barrier_timeout,
                interval=wrapper.monitor_process_interval,
                heartbeat_interval=wrapper.heartbeat_interval,
                heartbeat_timeout=wrapper.heartbeat_timeout,
                log_filename=wrapper.monitor_process_logfile,
                store_factory=wrapper.store_factory,
                store_kwargs=wrapper.store_kwargs,
            )

            store_kwargs = wrapper.store_kwargs
            base_store = wrapper.store_factory(**store_kwargs)
            log.debug(f'{base_store=} {store_kwargs=}')

            base_store.initial_barrier(
                ranks=[state.rank],
                rendezvous_count=state.world_size,
                timeout=wrapper.barrier_timeout,
            )
            base_store.set_initial_rank(state.rank, state.initial_rank)
            self.monitor_process.can_create_store()

            self.progress_watchdog = ProgressWatchdog(
                rank=state.rank,
                monitor_process=self.monitor_process,
                interval=wrapper.progress_watchdog_interval,
            )
            self.progress_watchdog.start()

            self.atomic_lock = threading.RLock()

            self.store = base_store
            self.base_store = base_store
            self.state = state
            self.wrapper = wrapper

        except Exception:
            self.shutdown()
            raise

    def shutdown(self):
        if self.state is not None and self.state.initial_rank in self.base_store.critical_ranks:
            timeout = timedelta.max
        else:
            timeout = timedelta(0)

        if self.base_store is not None:
            self.base_store.termination_barrier(
                ranks=[self.state.initial_rank],
                rendezvous_count=self.state.initial_world_size,
                timeout=timeout,
            )

        if self.progress_watchdog is not None:
            self.progress_watchdog.shutdown()
        if self.monitor_process is not None:
            self.monitor_process.shutdown()

        utils.Logging.deinitialize()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()

    @property
    def iteration(self) -> int:
        r'''
        Returns integer index of the current restart iteration.
        '''
        return self.state.iteration

    def ping(self) -> None:
        r'''
        Sends a heartbeat to indicate that the workload is making meaningful
        forward progress.

        The optional manual progress timeout is initiated with the first call
        to :py:meth:`CallWrapper.ping` on each rank in a restart iteration.
        Once the timeout is activated, every distributed rank must periodically
        invoke :py:meth:`CallWrapper.ping` to confirm ongoing progress. If any
        rank fails to report progress within the specified ``soft_timeout`` or
        ``hard_timeout`` intervals for the :py:class:`Wrapper`, the rank will
        be considered unresponsive, and a restart of the wrapped function will
        be attempted.
        '''
        self.progress_watchdog.ping()

    @contextlib.contextmanager
    def atomic(self):
        r'''
        A context manager to wrap a section of the workload that must not be
        executed while the termination procedure is in progress.

        :py:meth:`atomic` is implemented with a reentrant lock, shared between
        the termination procedure and atomic section in the wrapped function.
        The termination procedure won't be launched if the main thread is
        executing :py:meth:`inprocess.CallWrapper.atomic` code block, and the
        main thread won't enter into :py:meth:`inprocess.CallWrapper.atomic`
        code block if termination procedure is already in progress.
        '''
        while not self.atomic_lock.acquire(blocking=False):
            pass

        try:
            yield
        finally:
            self.atomic_lock.release()

    @reraise_if_unraisable(RankShouldRestart)
    def __call__(self, fn, args, kwargs):
        log = logging.getLogger(__name__)

        store = self.base_store
        base_store = self.base_store
        state = self.state
        monitor_process = self.monitor_process
        wrapper = self.wrapper
        progress_watchdog = self.progress_watchdog

        rank_assignment_ctx = RankAssignmentCtx(state, store, set())
        reassigned_ctx = wrapper.rank_assignment(rank_assignment_ctx)
        self.state = state = reassigned_ctx.state

        if wrapper.rank_filter is not None:
            state = wrapper.rank_filter(state)
        state.set_distributed_vars()

        monitor_process.start()
        ret = None

        try:
            while True:
                store = PrefixStore(state.iteration, base_store)
                store.set_initial_rank(state.rank, state.initial_rank)
                monitor_process.start_iteration(
                    state.rank,
                    state.world_size,
                    state.iteration,
                )
                progress_watchdog.reset()

                args, kwargs = param_utils.substitute_param_value(
                    fn,
                    args,
                    kwargs,
                    {
                        CallWrapper: self,
                    },
                )

                try:
                    try:
                        monitor_thread = MonitorThread(
                            state=state.freeze(),
                            store=store,
                            abort=wrapper.abort,
                            interval=wrapper.monitor_thread_interval,
                            progress_watchdog=progress_watchdog,
                            soft_timeout=wrapper.soft_timeout,
                            last_call_wait=wrapper.last_call_wait,
                            atomic_lock=self.atomic_lock,
                            daemon=True,
                        )
                        monitor_thread.start()

                        try:
                            try:
                                if wrapper.initialize is not None:
                                    wrapper.initialize(state.freeze())
                            except Exception as init_ex:
                                log.error(log_exc(state, init_ex, 'init_ex'))
                                raise

                            try:
                                if wrapper.health_check is not None:
                                    wrapper.health_check(state.freeze())
                            except Exception as health_ex:
                                log.error(log_exc(state, health_ex, 'health_ex'))
                                raise HealthCheckError from health_ex

                            if state.mode == Mode.ACTIVE:
                                ret = fn(*args, **kwargs)
                                store.record_completed()
                            elif state.mode == Mode.INACTIVE:
                                ret = reserve_fn(
                                    state,
                                    store,
                                    progress_watchdog,
                                    wrapper.progress_watchdog_interval,
                                )
                            else:
                                raise InternalError(f'{state}')

                            progress_watchdog.reset()
                            store.completion_barrier(
                                ranks=[state.rank],
                                rendezvous_count=state.world_size,
                                timeout=wrapper.completion_timeout,
                                timeout_chunk=wrapper.progress_watchdog_interval,
                            )
                        except Exception as fn_ex:
                            try:
                                log.error(log_exc(state, fn_ex, 'fn_ex'))
                                monitor_process.record_interrupted(
                                    [InterruptionRecord(state.rank, Interruption.EXCEPTION)]
                                )
                                progress_watchdog.spin_till_paused()
                                monitor_thread.maybe_join()
                            except RankShouldRestart as async_ex:
                                log.debug(log_exc(state, async_ex, 'async_ex'))
                                monitor_thread.shutdown()
                                raise async_ex from fn_ex
                            except Exception as other_ex:
                                log.critical(log_exc(state, other_ex, 'other_ex'))
                                raise InternalError(f'{state}') from other_ex
                            else:
                                raise InternalError(f'{state}') from fn_ex
                    except RankShouldRestart as term_ex:
                        log.warning(log_exc(state, term_ex, 'term_ex'))
                        monitor_thread.shutdown()

                        state.fn_exception = term_ex.__cause__

                        try:
                            if wrapper.finalize is not None:
                                wrapper.finalize(state.freeze())
                        except Exception as finalize_ex:
                            log.error(log_exc(state, finalize_ex, 'finalize_ex'))
                            raise finalize_ex from term_ex

                        try:
                            if wrapper.health_check is not None:
                                wrapper.health_check(state.freeze())
                        except Exception as health_ex:
                            log.error(log_exc(state, health_ex, 'health_ex'))
                            try:
                                raise health_ex from term_ex
                            except Exception:
                                raise HealthCheckError from health_ex
                        else:
                            raise HealthCheckPassed from term_ex

                    except Exception as term_ex:
                        log.critical(log_exc(state, term_ex, 'term_ex'))
                        raise InternalError(f'{state}') from term_ex
                    finally:
                        monitor_thread.shutdown()
                except HealthCheckPassed as restart_ex:
                    log.info(log_exc(state, restart_ex, 'restart_ex'))
                    store.iteration_barrier(
                        ranks=[state.rank],
                        rendezvous_count=state.world_size,
                        timeout=wrapper.barrier_timeout,
                    )
                    monitor_process.disable_sibling_monitor()

                    terminated_ranks = store.get_terminated_ranks()

                    rank_assignment_ctx = RankAssignmentCtx(state, store, terminated_ranks)
                    reassigned_ctx = wrapper.rank_assignment(rank_assignment_ctx)
                    self.state = state = reassigned_ctx.state

                    if wrapper.rank_filter is not None:
                        state = wrapper.rank_filter(state)
                    state.set_distributed_vars()
                else:
                    break
                finally:
                    while gc.collect():
                        pass

                state.advance()

        except BaseException as exit_ex:
            log.critical(log_exc(state, exit_ex, 'exit_ex'))
            store.record_interrupted([InterruptionRecord(state.rank, Interruption.BASE_EXCEPTION)])
            store.record_terminated_ranks([state.rank])

            store.iteration_barrier(
                ranks=[state.rank],
                rendezvous_count=state.world_size,
                timeout=wrapper.barrier_timeout,
            )
            monitor_process.disable_sibling_monitor()
            raise exit_ex

        rank = state.rank
        log.debug(f'{rank=} returning')
        return ret
