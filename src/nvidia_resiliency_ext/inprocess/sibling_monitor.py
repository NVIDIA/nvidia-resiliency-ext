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

import logging
import threading
import time
from datetime import timedelta

from . import exception
from .attribution import Interruption, InterruptionRecord


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

        while not self.should_stop.wait(self.interval.total_seconds()):
            log.debug(f'Sending heartbeat from {rank=}')
            self.store.send_heartbeat(rank)

    def shutdown(self, timeout=None):
        log = logging.getLogger(__name__)

        if timeout is None:
            timeout = self.timeout.total_seconds()

        is_alive = self.is_alive()
        log.debug(f'Shutting down heartbeat {is_alive=} {timeout=}')

        self.should_stop.set()
        self.join(timeout)
        if self.is_alive():
            raise exception.InternalError


class SiblingMonitor(threading.Thread):
    def __init__(
        self,
        store,
        rank,
        world_size,
        initial_world_size,
        timeout,
        interval,
    ):
        self.store = store
        self.rank = rank
        self.world_size = world_size
        self.initial_world_size = initial_world_size
        self.timeout = timeout
        self.interval = interval
        self.should_stop = threading.Event()

        self.seen_unresponsive_ranks = set()

        self.sibling_rank = (rank + 1) % world_size

        super().__init__(name=f'{type(self).__name__}-{rank}')

    def run(self):
        heartbeat = Heartbeat(self.rank, self.store, self.interval, self.timeout)
        heartbeat.start()

        self.store.send_heartbeat(self.sibling_rank)

        while not self.should_stop.wait(self.interval.total_seconds()):
            self.check_heartbeats()

        heartbeat.shutdown()
        heartbeat.join()

    def shutdown(self, timeout=None):
        log = logging.getLogger(__name__)

        if timeout is None:
            timeout = self.timeout.total_seconds()

        is_alive = self.is_alive()
        log.debug(f'Shutting down SiblingMonitor {is_alive=} {timeout=}')

        self.should_stop.set()
        self.join(timeout)
        if self.is_alive():
            raise exception.InternalError

    def check_heartbeats(self):
        log = logging.getLogger(__name__)

        sibling_heartbeat = self.store.get_heartbeat(self.sibling_rank)
        sibling_delta = timedelta(seconds=(time.time() - sibling_heartbeat))

        if sibling_delta > self.timeout:
            heartbeats = self.store.get_all_heartbeats(self.world_size)
            now = time.time()

            current_unresponsive_ranks = set(
                rank
                for rank, heartbeat in enumerate(heartbeats)
                if timedelta(seconds=(now - heartbeat)) > self.timeout
            )

            new_unresponsive_ranks = current_unresponsive_ranks - self.seen_unresponsive_ranks

            if new_unresponsive_ranks:
                log.debug(f'{new_unresponsive_ranks=}')
                self.terminate_unresponsive_ranks(new_unresponsive_ranks)
                self.seen_unresponsive_ranks.update(new_unresponsive_ranks)

    def terminate_unresponsive_ranks(self, unresponsive_ranks):
        self.store.record_interrupted(
            [InterruptionRecord(rank, Interruption.UNRESPONSIVE) for rank in unresponsive_ranks]
        )

        self.store.record_terminated_ranks(unresponsive_ranks)

        self.store.iteration_barrier(
            ranks=unresponsive_ranks,
            rendezvous_count=self.world_size,
            timeout=timedelta(0),
        )

        unresponsive_initial_ranks = self.store.get_initial_ranks(unresponsive_ranks)
        self.store.base_store.termination_barrier(
            ranks=unresponsive_initial_ranks,
            rendezvous_count=self.initial_world_size,
            timeout=timedelta(0),
        )
