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
import logging
import multiprocessing
import os
import sys
import threading
import time
import traceback

import psutil

from . import exception
from .attribution import Interruption
from .attribution import InterruptionRecord
from .logging import find_nearest_handler
from .progress_watchdog import ProgressWatchdog
from .progress_watchdog import Timestamp
from .sibling_monitor import Heartbeat
from .sibling_monitor import SiblingMonitor
from .store import PrefixStore
from .store import StoreMixin


def is_process_active(process):
    is_active = (
        process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    )
    return is_active


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
        rank,
        world_size,
        training_pid,
        conns_time,
        conns_ctrl,
        main_store_created,
        should_start,
        iteration_loop_started,
        stop_iteration_loop,
        should_stop,
        progress_timeout,
        termination_grace_time,
        barrier_timeout,
        interval,
        heartbeat_timeout,
        log_filename,
        log_level,
        store_factory,
        store_kwargs,
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
        except:
            log_format = (
                '%(asctime)s | %(levelname)-5s | %(name)s | %(message)s'
            )

        logging.basicConfig(
            filename=log_filename,
            filemode='w',
            level=log_level,
            format=log_format,
            force=True,
        )
        log = logging.getLogger(__name__)

        daemon_pid = os.getpid()
        log.info(
            f'{training_pid=} {daemon_pid=} {log_level=} {rank=} {world_size=}'
        )

        initial_rank = rank

        parent_conn_time, child_conn_time = conns_time
        parent_conn_ctrl, child_conn_ctrl = conns_ctrl
        parent_conn_time.close()
        parent_conn_ctrl.close()

        training_process = psutil.Process(training_pid)

        child_conn_ctrl.send(daemon_pid)
        child_conn_ctrl.recv()

        while not main_store_created.wait(interval.total_seconds()):
            log.info('waiting for the store to be created')
            if not is_process_active(training_process) or should_stop.is_set():
                log.critical('monitored process is terminated, shutting down')
                sys.exit(-1)

        base_store = store_factory(**store_kwargs)
        log.info(f'{base_store=} {store_kwargs=}')

        heartbeat = None
        sibling_monitor = None

        while not should_start.wait(interval.total_seconds()):
            log.info('waiting for the main process')
            if not is_process_active(training_process) or should_stop.is_set():
                log.critical('monitored process is terminated, shutting down')
                sys.exit(-1)

        try:
            while not should_stop.is_set():
                try:
                    rank, world_size, iteration = child_conn_ctrl.recv()
                    child_conn_ctrl.send((rank, world_size, iteration))
                    child_conn_ctrl.recv()
                except EOFError:
                    traceback_str = traceback.format_exc()
                    log.info(traceback_str)
                    break

                log.info(f'starting {iteration=} {rank=} {world_size=}')
                store = PrefixStore(iteration, base_store)

                if sibling_monitor is not None:
                    sibling_monitor.join()
                if heartbeat is not None:
                    heartbeat.shutdown()

                heartbeat = Heartbeat(rank, store, interval, heartbeat_timeout)
                heartbeat.start()

                sibling_monitor = SiblingMonitor(
                    store,
                    rank,
                    world_size,
                    heartbeat_timeout,
                    barrier_timeout,
                    interval,
                )

                timestamp = Timestamp()
                active_progress_timeout = progress_timeout
                extended_timeout = False

                while not stop_iteration_loop.is_set():
                    iteration_loop_started.set()

                    timed_out, timestamp = ProgressWatchdog.is_timed_out(
                        child_conn_time, active_progress_timeout, timestamp
                    )
                    try:
                        status = training_process.status()
                    except Exception as ex:
                        status = ex
                    log.debug(f'{status=} {timed_out=} {timestamp=}')

                    sibling_monitor.check_heartbeats()

                    interruption = None
                    if timed_out:
                        already_arrived = store.is_rank_at_reentrant_barrier(
                            rank=rank,
                            group_name=StoreMixin.TERMINATION_BARRIER,
                        )
                        if already_arrived and not extended_timeout:
                            log.info(
                                f'extending {active_progress_timeout=} '
                                f'to {barrier_timeout=}'
                            )
                            active_progress_timeout = barrier_timeout
                            extended_timeout = True
                            continue

                        interruption = Interruption.HARD_TIMEOUT
                        store.record_terminated_rank(rank)
                        try:
                            log.info(f'SIGCONT {training_process}')
                            training_process.resume()
                            log.info(f'SIGTERM {training_process}')
                            training_process.terminate()
                        except psutil.NoSuchProcess:
                            pass

                        try:
                            training_process.wait(
                                termination_grace_time.total_seconds()
                            )
                        except psutil.TimeoutExpired:
                            while is_process_active(training_process):
                                try:
                                    log.info(f'SIGCONT {training_process}')
                                    training_process.resume()
                                    log.info(f'SIGTERM {training_process}')
                                    training_process.terminate()
                                    log.info(f'SIGKILL {training_process}')
                                    training_process.kill()
                                except psutil.NoSuchProcess:
                                    break
                                else:
                                    time.sleep(
                                        termination_grace_time.total_seconds()
                                        / 100
                                    )

                    if not is_process_active(training_process):
                        log.info(f'monitored process is terminated')
                        should_stop.set()

                        if interruption is None:
                            interruption = Interruption.TERMINATED

                        store.record_interrupted(
                            InterruptionRecord(rank, interruption)
                        )
                        store.record_terminated_rank(rank)

                        store.termination_barrier(
                            rank=rank,
                            rendezvous_count=world_size,
                            timeout=barrier_timeout,
                        )
                        store.base_store.record_base_terminated_rank(
                            initial_rank
                        )

                        sibling_monitor.join()
                        heartbeat.shutdown()
                        break
                    else:
                        time.sleep(interval.total_seconds())
                iteration_loop_started.clear()
                log.info(f'finished {iteration=} {rank=} {world_size=}')
            if sibling_monitor is not None:
                sibling_monitor.join()
            if heartbeat is not None:
                heartbeat.shutdown()
            log.critical('loop terminated')
        except Exception:
            traceback_str = traceback.format_exc()
            log.critical(traceback_str)
        finally:
            if sibling_monitor is not None:
                sibling_monitor.join()
            if heartbeat is not None:
                heartbeat.shutdown()
        log.critical('process shutdown')

    def can_create_store(self):
        self.main_store_created.set()

    def start(self):
        self.should_start.set()

    def __init__(
        self,
        rank,
        world_size,
        progress_timeout,
        termination_grace_time,
        barrier_timeout,
        interval,
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
        self.main_store_created = ctx.Event()
        self.should_start = ctx.Event()
        self.iteration_loop_started = ctx.Event()
        self.stop_iteration_loop = ctx.Event()
        self.should_stop = ctx.Event()

        child_conn_time, parent_conn_time = ctx.Pipe(duplex=False)
        parent_conn_ctrl, child_conn_ctrl = ctx.Pipe()
        self.parent_conn_time = parent_conn_time
        self.parent_conn_ctrl = parent_conn_ctrl

        process_monitor_creator = ctx.Process(
            target=daemonize_fn,
            kwargs={
                'fn': self.run,
                'fn_kwargs': {
                    'rank': rank,
                    'world_size': world_size,
                    'training_pid': os.getpid(),
                    'conns_time': (parent_conn_time, child_conn_time),
                    'conns_ctrl': (parent_conn_ctrl, child_conn_ctrl),
                    'main_store_created': self.main_store_created,
                    'should_start': self.should_start,
                    'iteration_loop_started': self.iteration_loop_started,
                    'stop_iteration_loop': self.stop_iteration_loop,
                    'should_stop': self.should_stop,
                    'progress_timeout': progress_timeout,
                    'termination_grace_time': termination_grace_time,
                    'barrier_timeout': barrier_timeout,
                    'interval': interval,
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

        child_conn_time.close()
        child_conn_ctrl.close()

        process_monitor_pid = parent_conn_ctrl.recv()
        parent_conn_ctrl.send(process_monitor_pid)

        self.process_monitor_proc = psutil.Process(process_monitor_pid)

    def start_iteration(self, rank, world_size, iteration):
        self.stop_iteration_loop.set()
        self.parent_conn_ctrl.send((rank, world_size, iteration))
        self.parent_conn_ctrl.recv()
        self.stop_iteration_loop.clear()
        self.parent_conn_ctrl.send(None)
        self.iteration_loop_started.wait()

    def shutdown(self):
        self.should_stop.set()
        self.stop_iteration_loop.set()

        max_iteration_time = (
            self.termination_grace_time.total_seconds()
            + self.barrier_timeout.total_seconds()
            + self.interval.total_seconds()
        )
        timeout = 2 * max_iteration_time + 5
        try:
            self.process_monitor_proc.wait(timeout)
        except psutil.TimeoutExpired:
            try:
                self.process_monitor_proc.terminate()
            except psutil.NoSuchProcess:
                pass
