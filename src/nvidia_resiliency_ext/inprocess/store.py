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
import functools
import inspect
import logging
import os

# Issue: [B403:blacklist] Consider possible security implications associated with pickle module.
# Severity: Low   Confidence: High
# CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
# More Info: https://bandit.readthedocs.io/en/1.8.3/blacklists/blacklist_imports.html#b403-import-pickle
import pickle  # nosec
import sys
import time
from collections.abc import Iterable
from typing import Optional

import torch

from . import exception
from .attribution import InterruptionRecord
from .state import Mode


class BarrierError(exception.RestartError):
    pass


class BarrierTimeout(BarrierError):
    pass


class BarrierOverflow(BarrierError):
    pass


class StoreMixin:
    ANY_RANK_INTERRUPTED = 'any_rank_interrupted'
    ANY_RANK_COMPLETED = 'any_rank_completed'

    INTERRUPTION_RECORDS = 'interruption_records'
    INTERRUPTION_RECORDS_LOCK = 'interruption_records_lock'

    TERMINATED_RANKS = 'terminated_ranks'
    INITIAL_RANK = 'initial_rank_{rank}'
    ACTIVE_RANK = 'active_rank_{rank}'
    HEARTBEAT = 'heartbeat_{rank}'

    STATE = 'state_{rank}'
    KEY = 'key_{rank}'

    BARRIER_PREFIX = 'inprocess_barrier_prefix'
    STORE_PREFIX = '_inprocess_{iteration}'

    INITIAL_BARRIER = 'initial_barrier'
    COMPLETION_BARRIER = 'completion_barrier'
    ITERATION_BARRIER = 'iteration_barrier'
    TERMINATION_BARRIER = 'termination_barrier'

    # Two-step acknowledge phase keys
    INITIAL_BARRIER_ACK = 'initial_barrier_ack'

    # Global iteration counter
    GLOBAL_ITERATION_COUNTER = 'global_iteration_counter'

    @property
    def critical_ranks(self):
        return ()

    def clear_initial_barrier_keys(self):
        """Clear all initial barrier related keys from the store."""
        log = logging.getLogger(__name__)

        # Keys to clear
        keys_to_clear = [
            f'{self.BARRIER_PREFIX}:barrier:{self.INITIAL_BARRIER}',
            f'{self.BARRIER_PREFIX}:barrier:{self.INITIAL_BARRIER}:last_worker_arrived',
            f'{self.BARRIER_PREFIX}:barrier:{self.INITIAL_BARRIER}:arrived',
            f'{self.BARRIER_PREFIX}:barrier:{self.INITIAL_BARRIER_ACK}',
        ]

        try:
            for key in keys_to_clear:
                self.delete_key(key)
        except Exception as e:
            log.warning(f'Failed to clear some initial barrier keys: {e}')

    def get_global_iteration_counter(self) -> int:
        """Get the current global iteration counter value."""
        # First check if the key exists (non-blocking)
        if not self.check([self.GLOBAL_ITERATION_COUNTER]):
            # Key doesn't exist, return 0 as the initial value
            return 0

        try:
            return int(self.get(self.GLOBAL_ITERATION_COUNTER))
        except (torch.distributed.DistStoreError, ValueError):
            # If the key doesn't exist, return 0 as the initial value
            return 0

    def increment_global_iteration_counter(self, delta: int = 100) -> int:
        """Increment the global iteration counter by delta and return the new value."""
        return self.add(self.GLOBAL_ITERATION_COUNTER, delta)

    def initial_barrier_acknowledge(self, rank: int, world_size: int, timeout: datetime.timedelta):
        """Two-step acknowledge phase for initial barrier completion.

        Step 1: All ranks acknowledge that initial barrier is completed
        Step 2: Rank 0 waits for all acknowledgments and then clears the keys
        """
        log = logging.getLogger(__name__)

        # Step 1: All ranks acknowledge completion
        ack_key = f'{self.BARRIER_PREFIX}:barrier:{self.INITIAL_BARRIER_ACK}'

        # Add this rank's acknowledgment
        arrived_count = self.add(ack_key, 1)
        log.debug(f'{rank=} acknowledged initial barrier completion, count={arrived_count}')

        # Step 2: Rank 0 waits for all acknowledgments and clears keys
        if rank == 0:
            # Wait for all ranks to acknowledge
            start = time.monotonic()
            while True:
                try:
                    current_count = int(self.get(ack_key))
                    if current_count >= world_size:
                        log.debug(f'{rank=} all ranks acknowledged, clearing initial barrier keys')
                        self.clear_initial_barrier_keys()
                        # Increment global iteration counter by 100
                        new_iteration = self.increment_global_iteration_counter(100)
                        break
                except (torch.distributed.DistStoreError, ValueError) as ex:
                    if datetime.timedelta(seconds=(time.monotonic() - start)) > timeout:
                        log.warning(f'{rank=} timeout waiting for all acknowledgments: {ex}')
                        break
                    time.sleep(sys.getswitchinterval())

    def get_packed(self, key: str, sep: str):
        return self.get(key).decode().rstrip(sep).split(sep)

    def set_active_rank(self, rank, active):
        match active:
            case Mode.ACTIVE:
                active_str = '1'
            case Mode.INACTIVE:
                active_str = ''
            case _:
                raise RuntimeError
        self.set(self.ACTIVE_RANK.format(rank=rank), active_str)

    def get_all_active_ranks(self, world_size):
        return [
            bool(active)
            for active in self.multi_get(
                [self.ACTIVE_RANK.format(rank=rank) for rank in range(world_size)]
            )
        ]

    def set_initial_rank(self, rank, initial_rank):
        self.set(self.INITIAL_RANK.format(rank=rank), str(initial_rank))

    def get_initial_ranks(self, ranks: Iterable[int]) -> list[int]:
        return self.multi_get([self.INITIAL_RANK.format(rank=rank) for rank in ranks])

    def send_heartbeat(self, rank: int):
        self.set(self.HEARTBEAT.format(rank=rank), str(time.time()))

    def send_state(self, state, rank: int):
        self.set(self.STATE.format(rank=rank), pickle.dumps(state))

    def send_key(self, key, rank: int):
        self.set(self.KEY.format(rank=rank), pickle.dumps(key))

    def get_states(self, ranks):
        states = [
            # Issue: [B301:blacklist] Pickle and modules that wrap it can be unsafe when used to deserialize untrusted data, possible security issue.
            # Severity: Medium   Confidence: High
            # CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
            # More Info: https://bandit.readthedocs.io/en/1.8.3/blacklists/blacklist_calls.html#b301-pickle
            pickle.loads(state)  # nosec
            for state in self.multi_get([self.STATE.format(rank=rank) for rank in ranks])
        ]
        return states

    def get_keys(self, ranks):
        keys = [
            # Issue: [B301:blacklist] Pickle and modules that wrap it can be unsafe when used to deserialize untrusted data, possible security issue.
            # Severity: Medium   Confidence: High
            # CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
            # More Info: https://bandit.readthedocs.io/en/1.8.3/blacklists/blacklist_calls.html#b301-pickle
            pickle.loads(key)  # nosec
            for key in self.multi_get([self.KEY.format(rank=rank) for rank in ranks])
        ]
        return keys

    def get_heartbeat(self, rank: int) -> float:
        return float(self.get(self.HEARTBEAT.format(rank=rank)))

    def get_all_heartbeats(self, world_size: int) -> list[float]:
        return [
            float(heartbeat)
            for heartbeat in self.multi_get(
                [self.HEARTBEAT.format(rank=rank) for rank in range(world_size)]
            )
        ]

    def record_interrupted(self, records: Optional[Iterable[InterruptionRecord]] = None):
        if records is not None:
            self.append(self.INTERRUPTION_RECORDS, '')

            records_are_locked = bool(self.add(self.INTERRUPTION_RECORDS_LOCK, 0))
            if not records_are_locked:
                msg = ';'.join(str(r) for r in records)
                self.append(self.INTERRUPTION_RECORDS, f'{msg};')

        self.set(self.ANY_RANK_INTERRUPTED, '')

    def lock_interruption_records(self):
        self.add(self.INTERRUPTION_RECORDS_LOCK, 1)

    def get_interruption_records(self) -> list[InterruptionRecord]:
        self.append(self.INTERRUPTION_RECORDS, '')
        records = [
            InterruptionRecord.from_str(record)
            for record in self.get_packed(self.INTERRUPTION_RECORDS, ';')
            if record.strip()
        ]
        return records

    def wait_for_interrupted(self, timeout: datetime.timedelta):
        self.wait([self.ANY_RANK_INTERRUPTED], timeout)

    def wait_for_completed(self, timeout: datetime.timedelta):
        self.wait([self.ANY_RANK_COMPLETED], timeout)

    def record_completed(self):
        self.set(self.ANY_RANK_COMPLETED, '')

    def record_terminated_ranks(self, ranks: Iterable[int]):
        ranks_str = ','.join(str(r) for r in ranks)
        if ranks_str:
            self.append(self.TERMINATED_RANKS, f'{ranks_str},')

    def get_terminated_ranks(self) -> set[int]:
        self.append(self.TERMINATED_RANKS, '')
        terminated_ranks = set(
            [int(r) for r in self.get_packed(self.TERMINATED_RANKS, ',') if r.strip()]
        )
        return terminated_ranks

    def barrier(
        self,
        ranks: Iterable[int],
        group_name: str,
        rendezvous_count: int,
        timeout: datetime.timedelta,
        timeout_chunk: Optional[datetime.timedelta] = None,
    ):
        log = logging.getLogger(__name__)
        cn = inspect.currentframe().f_code.co_name
        log.debug(f'{ranks=} enter {group_name=} {cn} {rendezvous_count=}')

        store_key = f'{self.BARRIER_PREFIX}:{cn}:{group_name}'
        last_worker_arrived_key = f'{store_key}:last_worker_arrived'
        arrived_key = f'{store_key}:arrived'

        arrived_count = self.add(store_key, len(set(ranks)))
        ranks_str = ','.join(str(r) for r in ranks)
        self.append(arrived_key, f'{ranks_str},')

        if arrived_count > rendezvous_count:
            arrived_ranks = sorted([int(r) for r in self.get_packed(arrived_key, ',') if r.strip()])
            raise BarrierOverflow(f'{ranks=} {rendezvous_count=} {group_name=} {arrived_ranks=}')

        if arrived_count == rendezvous_count:
            self.set(last_worker_arrived_key, '1')

        if timeout_chunk is None:
            timeout_chunk = timeout
        else:
            timeout_chunk = min(timeout_chunk, timeout)

        if timeout and timeout_chunk:
            start = time.monotonic()
            while True:
                try:
                    self.wait([last_worker_arrived_key], timeout_chunk)
                    break
                except torch.distributed.DistStoreError as ex:
                    if datetime.timedelta(seconds=(time.monotonic() - start)) > timeout:
                        raise BarrierTimeout(
                            f'{ranks=} {rendezvous_count=} {group_name=} ' f'{timeout=}'
                        ) from ex
                    time.sleep(sys.getswitchinterval())

        log.debug(f'{ranks=} exits {group_name=} {cn} {rendezvous_count=}')

    def is_rank_at_reentrant_barrier(
        self,
        rank: int,
        group_name: str,
    ):
        log = logging.getLogger(__name__)
        barrier_name = self.reentrant_barrier.__name__
        store_key = f'{self.BARRIER_PREFIX}:{barrier_name}:{group_name}'
        arrived_key = f'{store_key}:arrived'
        self.append(arrived_key, '')
        arrived_ranks = set([int(r) for r in self.get_packed(arrived_key, ',') if r.strip()])
        log.debug(f'{rank=} {arrived_ranks=}')
        arrived = rank in arrived_ranks
        if arrived:
            log.debug(f'{rank=} already arrived {group_name=}')
        return arrived

    def reentrant_barrier(
        self,
        ranks: Iterable[int],
        group_name: str,
        rendezvous_count: int,
        timeout: datetime.timedelta,
        timeout_chunk: Optional[datetime.timedelta] = None,
    ):
        log = logging.getLogger(__name__)
        cn = inspect.currentframe().f_code.co_name
        log.debug(f'{ranks=} enter {group_name=} {cn} {rendezvous_count=}')

        store_key = f'{self.BARRIER_PREFIX}:{cn}:{group_name}'
        last_worker_arrived_key = f'{store_key}:last_worker_arrived'
        arrived_key = f'{store_key}:arrived'

        ranks_str = ','.join(str(r) for r in ranks)
        self.append(arrived_key, f'{ranks_str},')

        arrived_ranks = set([int(r) for r in self.get_packed(arrived_key, ',') if r.strip()])
        arrived_count = len(arrived_ranks)

        if arrived_count > rendezvous_count:
            arrived_ranks = sorted(list(arrived_ranks))
            raise BarrierOverflow(f'{ranks=} {rendezvous_count=} {group_name=} {arrived_ranks=}')

        if arrived_count == rendezvous_count:
            self.set(last_worker_arrived_key, '1')

        if timeout_chunk is None:
            timeout_chunk = timeout
        else:
            timeout_chunk = min(timeout_chunk, timeout)

        if timeout and timeout_chunk:
            start = time.monotonic()
            while True:
                try:
                    self.wait([last_worker_arrived_key], timeout_chunk)
                    break
                except torch.distributed.DistStoreError as ex:
                    if datetime.timedelta(seconds=(time.monotonic() - start)) > timeout:
                        raise BarrierTimeout(
                            f'{ranks=} {rendezvous_count=} {group_name=} ' f'{timeout=}'
                        ) from ex
                    time.sleep(sys.getswitchinterval())

        log.debug(f'{ranks=} exits {group_name=} {cn} {rendezvous_count=}')

    initial_barrier = functools.partialmethod(
        barrier,
        group_name=INITIAL_BARRIER,
    )
    completion_barrier = functools.partialmethod(
        barrier,
        group_name=COMPLETION_BARRIER,
    )
    iteration_barrier = functools.partialmethod(
        reentrant_barrier,
        group_name=ITERATION_BARRIER,
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
        tcp_store_host_rank: Optional[int] = None,
    ):
        log = logging.getLogger(__name__)

        if host_name is None:
            host_name = os.environ['MASTER_ADDR']
        if port is None:
            port = int(os.environ['MASTER_PORT'])
        if world_size is None:
            world_size = int(os.environ['WORLD_SIZE'])

        rank = int(os.environ['RANK'])

        # Save the host rank for later use
        self.tcp_store_host_rank = tcp_store_host_rank or self.TCP_STORE_HOST_RANK

        kwargs = {
            'host_name': host_name,
            'port': port,
            'world_size': world_size,
            'timeout': timeout,
            'wait_for_workers': wait_for_workers,
            'multi_tenant': multi_tenant,
            'use_libuv': use_libuv,
        }

        if rank == self.tcp_store_host_rank:
            super().__init__(is_master=True, **kwargs)
            log.debug(f'{rank=} hosting {type(self).__name__}({kwargs})')
        else:
            super().__init__(is_master=False, **kwargs)

        # Log successful connection
        if rank == 0:
            log.info(f'Rank {rank}: Successfully connected to TCPStore at {host_name}:{port}')

    @property
    def critical_ranks(self):
        if self.tcp_store_host_rank == -1:
            return ()
        return (self.tcp_store_host_rank,)


class PrefixStore(torch.distributed.PrefixStore, StoreMixin):
    def __init__(self, iteration, store):
        prefix = self.STORE_PREFIX.format(iteration=iteration)
        self.base_store = store
        super().__init__(prefix, store)


class FileStore(torch.distributed.FileStore, StoreMixin):
    pass
