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

import datetime
import enum
import logging
import multiprocessing
import os
import queue
import sys
import time
import traceback
from typing import Any, Optional

import psutil

from .attribution import Interruption, InterruptionRecord
from .progress_watchdog import Timestamp
from .sibling_monitor import SiblingMonitor
from .store import PrefixStore, StoreMixin
from .utils import find_nearest_handler


class Message(enum.Enum):
    TIMESTAMP = enum.auto()
    DISABLE_SIBLING_MONITOR = enum.auto()
    RECORD_INTERRUPTED = enum.auto()
    DAEMON_PID = enum.auto()
    ITERATION_START = enum.auto()
    TERMINATE = enum.auto()


def is_process_active(process):
    is_active = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    return is_active


def terminate_process(process: psutil.Process, termination_grace_time: datetime.timedelta):
    log = logging.getLogger(__name__)
    try:
        log.info(f'SIGCONT {process}')
        process.resume()
        log.info(f'SIGTERM {process}')
        process.terminate()
    except psutil.NoSuchProcess:
        pass

    try:
        process.wait(termination_grace_time.total_seconds())
    except psutil.TimeoutExpired:
        while is_process_active(process):
            try:
                log.info(f'SIGCONT {process}')
                process.resume()
                log.info(f'SIGTERM {process}')
                process.terminate()
                log.info(f'SIGKILL {process}')
                process.kill()
            except psutil.NoSuchProcess:
                break
            else:
                time.sleep(sys.getswitchinterval())


def daemonize_fn(fn, fn_args=(), fn_kwargs=None):
    if fn_kwargs is None:
        fn_kwargs = {}

    # Fork the first time
    try:
        pid = os.fork()
        if pid > 0:
            # Exit first parent
            sys.exit(0)
    except OSError as e:
        sys.stderr.write(f'fork #1 failed: {e}\n')
        sys.exit(1)

    # Decouple from parent environment
    os.chdir('/')
    os.setsid()
    os.umask(0)

    # Fork the second time
    try:
        pid = os.fork()
        if pid > 0:
            # Exit second parent
            return pid
    except OSError as e:
        sys.stderr.write(f'fork #2 failed: {e}\n')
        sys.exit(1)

    # Now we are in the grandchild process
    # Close standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()
    si = open('/dev/null', 'r')
    so = open('/dev/null', 'a+')
    se = open('/dev/null', 'a+')
    os.dup2(si.fileno(), sys.stdin.fileno())
    os.dup2(so.fileno(), sys.stdout.fileno())
    os.dup2(se.fileno(), sys.stderr.fileno())

    fn(*fn_args, **fn_kwargs)


class MonitorProcess:
    @staticmethod
    def run(
        rank: int,
        world_size: int,
        target_pid: int,
        msg_queue: multiprocessing.Queue,
        main_store_created: multiprocessing.Event,
        should_start: multiprocessing.Event,
        should_stop: multiprocessing.Event,
        soft_timeout: datetime.timedelta,
        hard_timeout: datetime.timedelta,
        termination_grace_time: datetime.timedelta,
        barrier_timeout: datetime.timedelta,
        interval: datetime.timedelta,
        heartbeat_interval: datetime.timedelta,
        heartbeat_timeout: datetime.timedelta,
        log_filename: Optional[str],
        log_level: int,
        store_factory: type[StoreMixin],
        store_kwargs: dict[str, Any],
    ):
        if log_filename is not None:
            log_filename = log_filename.format(rank=rank)
        else:
            log_filename = os.devnull

        parent_module_name = MonitorProcess.__module__.split('.')[-2]
        parent_logger = logging.getLogger(parent_module_name)
        parent_logger.propagate = True

        try:
            nearest_file_handler = find_nearest_handler(
                logging.getLogger(__name__),
                logging.FileHandler,
            )
            log_format = nearest_file_handler.formatter._fmt
        except Exception:
            log_format = '%(asctime)s | %(levelname)-5s | %(name)s | %(message)s'

        logging.basicConfig(
            filename=log_filename,
            filemode='w',
            level=log_level,
            format=log_format,
            force=True,
        )
        log = logging.getLogger(__name__)

        daemon_pid = os.getpid()
        log.info(f'{target_pid=} {daemon_pid=} {log_level=} {rank=} {world_size=}')

        target_process = psutil.Process(target_pid)
        msg_queue.put((Message.DAEMON_PID, daemon_pid))

        while not main_store_created.wait(interval.total_seconds()):
            log.info('waiting for the store to be created')
            if not is_process_active(target_process) or should_stop.is_set():
                log.critical('target process is terminated, shutting down')
                sys.exit(-1)

        base_store = store_factory(**store_kwargs)
        log.info(f'{base_store=} {store_kwargs=}')

        while not should_start.wait(interval.total_seconds()):
            log.info('waiting for the main process')
            if not is_process_active(target_process) or should_stop.is_set():
                log.critical('target process is terminated, shutting down')
                sys.exit(-1)

        initial_rank = rank
        initial_world_size = world_size
        initial_hard_timeout = hard_timeout
        soft_timeout_expired = False
        timestamp = Timestamp(auto=time.monotonic())
        sibling_monitor = None

        try:
            while not should_stop.is_set():
                try:
                    msg, content = msg_queue.get(timeout=interval.total_seconds())

                    match msg:
                        case Message.ITERATION_START:
                            rank, world_size, iteration = content
                            log.info(f'starting {iteration=} {rank=} {world_size=}')
                            store = PrefixStore(iteration, base_store)
                            sibling_monitor = SiblingMonitor(
                                store=store,
                                rank=rank,
                                world_size=world_size,
                                initial_world_size=initial_world_size,
                                timeout=heartbeat_timeout,
                                interval=heartbeat_interval,
                            )
                            sibling_monitor.start()
                            hard_timeout = initial_hard_timeout
                            soft_timeout_expired = False
                            continue
                        case Message.TIMESTAMP:
                            timestamp = content
                            continue
                        case Message.DISABLE_SIBLING_MONITOR:
                            if sibling_monitor is not None:
                                sibling_monitor.shutdown()
                        case Message.TERMINATE:
                            break
                        case Message.RECORD_INTERRUPTED:
                            record = content
                            store.record_interrupted([record])
                        case _:
                            raise RuntimeError
                except queue.Empty:
                    pass

                try:
                    status = target_process.status()
                except Exception as ex:
                    status = ex
                log.debug(f'{status=} {timestamp=}')

                if not soft_timeout_expired and timestamp.is_timed_out(soft_timeout):
                    store.record_interrupted([InterruptionRecord(rank, Interruption.SOFT_TIMEOUT)])
                    soft_timeout_expired = True

                if timestamp.is_timed_out(hard_timeout):
                    already_arrived = store.is_rank_at_reentrant_barrier(
                        rank=rank,
                        group_name=StoreMixin.ITERATION_BARRIER,
                    )
                    if already_arrived and hard_timeout != barrier_timeout:
                        hard_timeout = barrier_timeout
                        continue

                    interruption = Interruption.HARD_TIMEOUT
                    store.record_interrupted([InterruptionRecord(rank, interruption)])
                    store.record_terminated_ranks([rank])
                    terminate_process(target_process, termination_grace_time)

                if not is_process_active(target_process):
                    log.info('target process is terminated')

                    interruption = Interruption.TERMINATED
                    store.record_interrupted([InterruptionRecord(rank, interruption)])
                    store.record_terminated_ranks([rank])

                    store.iteration_barrier(
                        ranks=[rank],
                        rendezvous_count=world_size,
                        timeout=barrier_timeout,
                        timeout_chunk=interval,
                    )

                    sibling_monitor.shutdown()

                    base_store.termination_barrier(
                        ranks=[initial_rank],
                        rendezvous_count=initial_world_size,
                        timeout=datetime.timedelta(0),
                    )

                    break
        except Exception as ex:
            log.critical(f'MonitorProcess raised {ex}')
            traceback_str = traceback.format_exc()
            log.critical(traceback_str)

            interruption = Interruption.MONITOR_PROCESS_EXCEPTION
            store.record_interrupted([InterruptionRecord(rank, interruption)])
            store.record_terminated_ranks([rank])

            store.iteration_barrier(
                ranks=[rank],
                rendezvous_count=world_size,
                timeout=barrier_timeout,
                timeout_chunk=interval,
            )

            sibling_monitor.shutdown()

            if initial_rank in base_store.critical_ranks:
                timeout = datetime.timedelta.max
            else:
                timeout = datetime.timedelta(0)

            base_store.termination_barrier(
                ranks=[initial_rank],
                rendezvous_count=initial_world_size,
                timeout=timeout,
            )
        else:
            log.critical('MonitorProcess clean shutdown')
        finally:
            if sibling_monitor is not None:
                sibling_monitor.shutdown()

    def can_create_store(self):
        self.main_store_created.set()

    def start(self):
        self.should_start.set()

    def disable_sibling_monitor(self):
        self.msg_queue.put((Message.DISABLE_SIBLING_MONITOR, None))

    def send_timestamp(self, timestamp):
        self.msg_queue.put((Message.TIMESTAMP, timestamp))

    def record_interrupted(self, record=None):
        self.msg_queue.put((Message.RECORD_INTERRUPTED, record))

    def __init__(
        self,
        rank,
        world_size,
        soft_timeout,
        hard_timeout,
        termination_grace_time,
        barrier_timeout,
        interval,
        heartbeat_interval,
        heartbeat_timeout,
        log_filename,
        store_factory,
        store_kwargs,
    ):
        log = logging.getLogger(__name__)

        self.termination_grace_time = termination_grace_time
        self.barrier_timeout = barrier_timeout
        self.interval = interval

        ctx = multiprocessing.get_context('fork')
        self.msg_queue = ctx.Queue()
        self.main_store_created = ctx.Event()
        self.should_start = ctx.Event()
        self.should_stop = ctx.Event()

        process_monitor_creator = ctx.Process(
            target=daemonize_fn,
            kwargs={
                'fn': self.run,
                'fn_kwargs': {
                    'rank': rank,
                    'world_size': world_size,
                    'target_pid': os.getpid(),
                    'msg_queue': self.msg_queue,
                    'main_store_created': self.main_store_created,
                    'should_start': self.should_start,
                    'should_stop': self.should_stop,
                    'soft_timeout': soft_timeout,
                    'hard_timeout': hard_timeout,
                    'termination_grace_time': termination_grace_time,
                    'barrier_timeout': barrier_timeout,
                    'interval': interval,
                    'heartbeat_interval': heartbeat_interval,
                    'heartbeat_timeout': heartbeat_timeout,
                    'log_filename': log_filename,
                    'log_level': log.getEffectiveLevel(),
                    'store_factory': store_factory,
                    'store_kwargs': store_kwargs,
                },
            },
            daemon=True,
        )
        process_monitor_creator.start()
        process_monitor_creator.join()

        msg, process_monitor_pid = self.msg_queue.get()
        assert msg == Message.DAEMON_PID
        self.process_monitor_proc = psutil.Process(process_monitor_pid)

    def start_iteration(self, rank, world_size, iteration):
        self.msg_queue.put((Message.ITERATION_START, (rank, world_size, iteration)))

    def shutdown(self):
        self.should_stop.set()
        self.msg_queue.close()
        self.msg_queue.join_thread()
