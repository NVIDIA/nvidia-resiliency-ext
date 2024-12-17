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
import threading
import time

from . import exception
from .attribution import Interruption
from .attribution import InterruptionRecord


def terminate_unresponsive_ranks(
    store, ranks, world_size, barrier_timeout, interval
):
    log = logging.getLogger(__name__)

    store.record_interrupted(
        [InterruptionRecord(rank, Interruption.UNRESPONSIVE) for rank in ranks]
    )

    for rank in ranks:
        store.record_terminated_rank(rank)

    store.termination_barrier(
        rank=ranks,
        rendezvous_count=world_size,
        timeout=barrier_timeout,
        timeout_chunk=interval / 2,
    )
    for rank in ranks:
        initial_rank = store.get_initial_rank(rank)
        store.base_store.record_base_terminated_rank(initial_rank)


class Heartbeat(threading.Thread):
    def __init__(self, rank, store, interval, timeout):
        self.rank = rank
        self.store = store
        self.interval = interval
        self.timeout = timeout

        self.should_stop = threading.Event()

        super().__init__(name=f'{type(self).__name__}-{rank}', daemon=True)

    def run(self):
        log = logging.getLogger(__name__)
        rank = self.rank

        while not self.should_stop.is_set():
            log.debug(f'Sending heartbeat from {rank=}')
            self.store.send_heartbeat(rank)
            time.sleep(self.interval.total_seconds())

    def shutdown(self, timeout=None):
        log = logging.getLogger(__name__)
        log.debug(f'Shutting down heartbeat {timeout=}')

        if timeout is None:
            timeout = self.timeout.total_seconds()

        self.should_stop.set()
        self.join(timeout)
        if self.is_alive():
            raise exception.InternalError


class SiblingMonitor:
    def __init__(
        self,
        store,
        rank,
        world_size,
        heartbeat_timeout,
        barrier_timeout,
        interval,
    ):
        self.store = store
        self.rank = rank
        self.world_size = world_size
        self.heartbeat_timeout = heartbeat_timeout
        self.barrier_timeout = barrier_timeout
        self.interval = interval

        self.termination_threads = []
        self.seen_unresponsive_ranks = set()

        self.sibling_rank = (rank + 1) % world_size

    def check_heartbeats(self):
        log = logging.getLogger(__name__)

        sibling_heartbeat = self.store.get_heartbeat(self.sibling_rank)
        sibling_delta = datetime.timedelta(
            microseconds=(time.time_ns() - sibling_heartbeat) / 1e3
        )

        if sibling_delta > self.heartbeat_timeout:
            heartbeats = self.store.get_all_heartbeats(self.world_size)
            now_ns = time.time_ns()

            current_unresponsive_ranks = set(
                rank
                for rank, heartbeat in enumerate(heartbeats)
                if datetime.timedelta(microseconds=(now_ns - heartbeat) / 1e3)
                > self.heartbeat_timeout
            )

            new_unresponsive_ranks = (
                current_unresponsive_ranks - self.seen_unresponsive_ranks
            )

            if new_unresponsive_ranks:
                log.debug(f'{new_unresponsive_ranks=}')
                termination_thread = threading.Thread(
                    target=terminate_unresponsive_ranks,
                    args=(
                        self.store,
                        new_unresponsive_ranks,
                        self.world_size,
                        self.barrier_timeout,
                        self.interval,
                    ),
                    daemon=True,
                )
                termination_thread.start()
                self.seen_unresponsive_ranks.update(new_unresponsive_ranks)
                self.termination_threads.append(termination_thread)

    def join(self):
        timeout = 2 * self.barrier_timeout + datetime.timedelta(seconds=1)
        for termination_thread in self.termination_threads:
            termination_thread.join(timeout.total_seconds())

        for termination_thread in self.termination_threads:
            if termination_thread.is_alive():
                raise exception.TimeoutError
