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

import collections
import datetime
import functools
import inspect
import logging
import os
import time
from typing import Optional

import torch

from . import exception
from .attribution import InterruptionRecord
from .logging import log_exc


class BarrierError(exception.RestartError):
    pass


class BarrierTimeout(BarrierError):
    pass


class BarrierOverflow(BarrierError):
    pass


class StoreMixin:
    NUM_COMPLETED_RANKS = 'num_completed_ranks'
    NUM_INTERRUPTED_RANKS = 'num_interrupted_ranks'
    NUM_TERMINATED_RANKS = 'num_terminated_ranks'

    INTERRUPTION_RECORDS = 'interruption_records'
    INTERRUPTION_RECORDS_LOCK = 'interruption_records_lock'

    TERMINATED_RANK = 'terminated_rank_{idx}'
    INITIAL_RANK = 'initial_rank_{rank}'
    BASE_TERMINATED_LIST = 'base_terminated_list'
    HEARTBEAT = 'heartbeat_{rank}'

    BARRIER_PREFIX = 'inprocess_barrier_prefix'
    STORE_PREFIX = '_inprocess_{iteration}'

    INITIAL_BARRIER = 'initial_barrier'
    ITERATION_BARRIER = 'iteration_barrier'
    COMPLETION_BARRIER = 'completion_barrier'
    TERMINATION_BARRIER = 'termination_barrier'

    @property
    def critical_ranks(self):
        return ()

    def set_initial_rank(self, rank, initial_rank):
        self.set(self.INITIAL_RANK.format(rank=rank), str(initial_rank))

    def get_initial_rank(self, rank):
        return int(self.get(self.INITIAL_RANK.format(rank=rank)))

    def send_heartbeat(self, rank):
        self.set(self.HEARTBEAT.format(rank=rank), str(time.time_ns()))

    def get_heartbeat(self, rank):
        return int(self.get(self.HEARTBEAT.format(rank=rank)))

    def get_all_heartbeats(self, world_size):
        return [
            int(heartbeat)
            for heartbeat in self.multi_get(
                [
                    self.HEARTBEAT.format(rank=rank)
                    for rank in range(world_size)
                ]
            )
        ]

    def record_base_terminated_rank(self, rank):
        self.append(self.BASE_TERMINATED_LIST, f'{rank},')

    def get_base_terminated_count(self):
        terminated_ranks = set(
            [
                int(r)
                for r in self.get(self.BASE_TERMINATED_LIST)
                .decode()
                .rstrip(',')
                .split(',')
                if r.strip()
            ]
        )
        terminated_count = len(terminated_ranks)
        return terminated_count

    def record_interrupted(self, record=None):
        if record is not None:
            if not isinstance(record, list):
                record = [record]

            self.append(self.INTERRUPTION_RECORDS, '')

            records_are_locked = bool(
                self.add(self.INTERRUPTION_RECORDS_LOCK, 0)
            )
            if not records_are_locked:
                msg = ';'.join(str(r) for r in record)
                self.append(self.INTERRUPTION_RECORDS, f'{msg};')

        self.add(self.NUM_INTERRUPTED_RANKS, 1)

    def lock_interruption_records(self):
        self.add(self.INTERRUPTION_RECORDS_LOCK, 1)

    def get_interruption_records(self):
        self.append(self.INTERRUPTION_RECORDS, '')
        records = [
            InterruptionRecord.from_str(record)
            for record in self.get(self.INTERRUPTION_RECORDS)
            .decode()
            .rstrip(';')
            .split(';')
            if record.strip()
        ]
        return records

    def is_any_rank_iterrupted(self):
        return self.add(self.NUM_INTERRUPTED_RANKS, 0) > 0

    def record_completed(self):
        self.add(self.NUM_COMPLETED_RANKS, 1)

    def is_any_rank_completed(self):
        return self.add(self.NUM_COMPLETED_RANKS, 0) > 0

    def record_terminated_rank(self, rank):
        next_terminated_rank_idx = self.add(self.NUM_TERMINATED_RANKS, 1) - 1
        self.set(
            self.TERMINATED_RANK.format(idx=next_terminated_rank_idx),
            str(rank),
        )

    def get_terminated_ranks(self):
        num_terminated_ranks = int(self.add(self.NUM_TERMINATED_RANKS, 0))
        terminated_ranks = set(
            int(terminated_rank)
            for terminated_rank in self.multi_get(
                [
                    self.TERMINATED_RANK.format(idx=i)
                    for i in range(num_terminated_ranks)
                ]
            )
        )
        return terminated_ranks

    def barrier(
        self,
        rank,
        group_name,
        rendezvous_count,
        timeout,
        timeout_chunk=None,
    ):
        log = logging.getLogger(__name__)
        cn = inspect.currentframe().f_code.co_name
        log.debug(f'{rank=} enters {group_name=} {cn} {rendezvous_count=}')

        store_key = f'{self.BARRIER_PREFIX}:{cn}:{group_name}'
        last_worker_arrived_key = f'{store_key}:last_worker_arrived'
        arrived_key = f'{store_key}:arrived'

        arrived_count = self.add(store_key, 1)
        self.append(arrived_key, f'{rank},')

        if arrived_count > rendezvous_count:
            arrived_ranks = sorted(
                [
                    int(r)
                    for r in self.get(arrived_key)
                    .decode()
                    .rstrip(',')
                    .split(',')
                    if r.strip()
                ]
            )
            raise BarrierOverflow(
                f'{rank=} {rendezvous_count=} {group_name=} {arrived_ranks=}'
            )

        if arrived_count == rendezvous_count:
            self.set(last_worker_arrived_key, '1')

        if timeout_chunk is None:
            timeout_chunk = timeout
        else:
            timeout_chunk = min(timeout_chunk, timeout)

        start = time.monotonic()
        while True:
            try:
                self.wait([last_worker_arrived_key], timeout_chunk)
                break
            except Exception as ex:
                if (
                    datetime.timedelta(seconds=(time.monotonic() - start))
                    > timeout
                ):
                    raise BarrierTimeout(
                        f'{rank=} {rendezvous_count=} {group_name=} {timeout=}'
                    ) from ex

        log.debug(f'{rank=} exits {group_name=} {cn} {rendezvous_count=}')

    def is_rank_at_reentrant_barrier(
        self,
        rank,
        group_name,
    ):
        log = logging.getLogger(__name__)
        barrier_name = self.reentrant_barrier.__name__
        store_key = f'{self.BARRIER_PREFIX}:{barrier_name}:{group_name}'
        arrived_key = f'{store_key}:arrived'
        self.append(arrived_key, '')
        arrived_ranks = set(
            [
                int(r)
                for r in self.get(arrived_key).decode().rstrip(',').split(',')
                if r.strip()
            ]
        )
        log.debug(f'{rank=} {arrived_ranks=}')
        arrived = rank in arrived_ranks
        if arrived:
            log.debug(f'{rank=} already arrived {group_name=}')
        return arrived

    def reentrant_barrier(
        self,
        rank,
        group_name,
        rendezvous_count,
        timeout,
        timeout_chunk=None,
    ):
        log = logging.getLogger(__name__)
        cn = inspect.currentframe().f_code.co_name
        log.debug(f'{rank=} enters {group_name=} {cn} {rendezvous_count=}')

        store_key = f'{self.BARRIER_PREFIX}:{cn}:{group_name}'
        last_worker_arrived_key = f'{store_key}:last_worker_arrived'
        arrived_key = f'{store_key}:arrived'

        if isinstance(rank, int):
            self.append(arrived_key, f'{rank},')
        elif isinstance(rank, collections.abc.Iterable):
            ranks = ','.join(str(r) for r in rank)
            self.append(arrived_key, f'{ranks},')
        else:
            raise RuntimeError

        arrived_ranks = set(
            [
                int(r)
                for r in self.get(arrived_key).decode().rstrip(',').split(',')
                if r.strip()
            ]
        )
        arrived_count = len(arrived_ranks)

        if arrived_count > rendezvous_count:
            arrived_list = sorted(list(arrived_ranks))
            raise BarrierOverflow(
                f'{rank=} {rendezvous_count=} {group_name=} {arrived_list=}'
            )

        if arrived_count == rendezvous_count:
            self.set(last_worker_arrived_key, '1')

        if timeout_chunk is None:
            timeout_chunk = timeout
        else:
            timeout_chunk = min(timeout_chunk, timeout)

        start = time.monotonic()
        while True:
            try:
                self.wait([last_worker_arrived_key], timeout_chunk)
                break
            except Exception as ex:
                if (
                    datetime.timedelta(seconds=(time.monotonic() - start))
                    > timeout
                ):
                    raise BarrierTimeout(
                        f'{rank=} {rendezvous_count=} {group_name=} {timeout=}'
                    ) from ex

        log.debug(f'{rank=} exits {group_name=} {cn} {rendezvous_count=}')

    initial_barrier = functools.partialmethod(
        barrier,
        group_name=INITIAL_BARRIER,
    )
    iteration_barrier = functools.partialmethod(
        barrier,
        group_name=ITERATION_BARRIER,
    )
    completion_barrier = functools.partialmethod(
        barrier,
        group_name=COMPLETION_BARRIER,
    )
    termination_barrier = functools.partialmethod(
        reentrant_barrier,
        group_name=TERMINATION_BARRIER,
    )


class TCPStore(torch.distributed.TCPStore, StoreMixin):
    TCP_STORE_HOST_RANK = 0

    def __init__(
        self,
        host_name: Optional[str] = None,
        port: Optional[int] = None,
        world_size: Optional[int] = None,
        timeout: datetime.timedelta = datetime.timedelta(seconds=300),
        wait_for_workers: bool = True,
        multi_tenant: bool = False,
        use_libuv: bool = True,
    ):
        log = logging.getLogger(__name__)

        if host_name is None:
            host_name = os.environ['MASTER_ADDR']
        if port is None:
            port = int(os.environ['MASTER_PORT'])
        if world_size is None:
            world_size = int(os.environ['WORLD_SIZE'])

        rank = int(os.environ['RANK'])

        kwargs = {
            'host_name': host_name,
            'port': port,
            'world_size': world_size,
            'timeout': timeout,
            'wait_for_workers': wait_for_workers,
            'multi_tenant': multi_tenant,
            'use_libuv': use_libuv,
        }

        if rank == self.TCP_STORE_HOST_RANK:
            try:
                super().__init__(is_master=True, **kwargs)
                log.debug(f'{rank=} hosting {type(self).__name__}({kwargs})')
            except Exception as store_ex:
                log.debug(log_exc(rank, store_ex, 'store_ex'))
                super().__init__(is_master=False, **kwargs)
        else:
            super().__init__(is_master=False, **kwargs)

    @property
    def critical_ranks(self):
        return (self.TCP_STORE_HOST_RANK,)


class PrefixStore(torch.distributed.PrefixStore, StoreMixin):
    def __init__(self, iteration, store):
        prefix = self.STORE_PREFIX.format(iteration=iteration)
        self.base_store = store
        super().__init__(prefix, store)


class FileStore(torch.distributed.FileStore, StoreMixin):
    pass
