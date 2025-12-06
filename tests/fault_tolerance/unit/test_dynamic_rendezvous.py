# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import pickle
import threading
import time
from datetime import datetime, timedelta
from types import MethodType
from typing import Callable, Optional, Tuple, cast
from unittest import TestCase
from unittest.mock import MagicMock, Mock, call, patch

import torch
from torch.distributed import Store
from torch.distributed.elastic.agent.server.api import WorkerState
from torch.distributed.elastic.rendezvous.c10d_rendezvous_backend import C10dRendezvousBackend

from nvidia_resiliency_ext.fault_tolerance._ft_rendezvous import (
    FtRendezvousHandler,
    RendezvousBackend,
    RendezvousClosedError,
    RendezvousParameters,
    RendezvousSettings,
    RendezvousStateError,
    RendezvousTimeout,
    RendezvousTimeoutError,
    Token,
    UnhealthyNodeException,
    _Action,
    _BackendRendezvousStateHolder,
    _DistributedRendezvousOpExecutor,
    _NodeDesc,
    _RendezvousState,
    _RendezvousStateHolder,
    create_handler,
)


def _extract_rendezvous_info(rendezvous_result) -> Tuple[int, int]:
    """Extract rank and world_size from either RendezvousInfo or tuple.

    Args:
        rendezvous_result: Either a RendezvousInfo object or a tuple (store, rank, world_size)

    Returns:
        Tuple of (rank, world_size)
    """
    try:
        # Try to access as RendezvousInfo object
        return rendezvous_result.rank, rendezvous_result.world_size
    except AttributeError:
        # Fall back to tuple format (store, rank, world_size)
        return rendezvous_result[1], rendezvous_result[2]


def _extract_store(rendezvous_result) -> Store:
    """Extract store from either RendezvousInfo or tuple.

    Args:
        rendezvous_result: Either a RendezvousInfo object or a tuple (store, rank, world_size)

    Returns:
        Store object
    """
    try:
        # Try to access as RendezvousInfo object
        return rendezvous_result.store
    except AttributeError:
        # Fall back to tuple format (store, rank, world_size)
        return rendezvous_result[0]


class CustomAssertMixin:
    assertDictEqual: Callable

    def assert_state_equal(self, actual: _RendezvousState, expected: _RendezvousState) -> None:
        self.assertDictEqual(vars(actual), vars(expected))

    def assert_state_empty(self, actual: _RendezvousState) -> None:
        self.assertDictEqual(vars(actual), vars(_RendezvousState()))


# Basic classes like RendezvousTimeout, NodeDesc, NodeDescGenerator are unchanged from upstream PyTorch
# and are well-tested there. We focus on FT-specific functionality.


class AssignRanksTest(TestCase):
    """Test the _assign_ranks static method which handles rank assignment logic."""

    def test_assign_ranks_with_infra_rank_always_uses_infra(self) -> None:
        from nvidia_resiliency_ext.fault_tolerance._ft_rendezvous import (
            _DistributedRendezvousOpExecutor,
        )

        # Simulate 3 participants with infrastructure ranks 0, 1, 2
        participants = {
            _NodeDesc("node0", 1, 1): 0,
            _NodeDesc("node1", 1, 1): 1,
            _NodeDesc("node2", 1, 1): 2,
        }

        result = _DistributedRendezvousOpExecutor._assign_ranks(participants)

        # Should sort by infrastructure ranks and assign contiguous group ranks
        self.assertEqual(result[_NodeDesc("node0", 1, 1)], 0)
        self.assertEqual(result[_NodeDesc("node1", 1, 1)], 1)
        self.assertEqual(result[_NodeDesc("node2", 1, 1)], 2)

    def test_assign_ranks_uses_infra_ranks(self) -> None:
        from nvidia_resiliency_ext.fault_tolerance._ft_rendezvous import (
            _DistributedRendezvousOpExecutor,
        )

        # Simulate 3 participants with infrastructure ranks
        participants = {
            _NodeDesc("node0", 1, 1): 0,  # Infrastructure rank 0
            _NodeDesc("node1", 1, 1): 1,  # Infrastructure rank 1
            _NodeDesc("node2", 1, 1): 2,  # Infrastructure rank 2
        }

        result = _DistributedRendezvousOpExecutor._assign_ranks(participants)

        # Should sort by infrastructure ranks and assign contiguous group ranks
        self.assertEqual(result[_NodeDesc("node0", 1, 1)], 0)
        self.assertEqual(result[_NodeDesc("node1", 1, 1)], 1)
        self.assertEqual(result[_NodeDesc("node2", 1, 1)], 2)

    def test_assign_ranks_with_gaps_in_infra_ranks(self) -> None:
        from nvidia_resiliency_ext.fault_tolerance._ft_rendezvous import (
            _DistributedRendezvousOpExecutor,
        )

        # Infrastructure ranks with gaps (e.g., from SLURM with some nodes missing)
        # Infra ranks: [0, 5, 10] should map to group ranks [0, 1, 2]
        participants = {
            _NodeDesc("node0", 1, 1): 0,  # Infra rank 0
            _NodeDesc("node1", 1, 1): 5,  # Infra rank 5
            _NodeDesc("node2", 1, 1): 10,  # Infra rank 10
        }

        result = _DistributedRendezvousOpExecutor._assign_ranks(participants)

        # Should sort by infra rank and assign contiguous group ranks
        self.assertEqual(result[_NodeDesc("node0", 1, 1)], 0)  # Smallest infra rank -> group rank 0
        self.assertEqual(result[_NodeDesc("node1", 1, 1)], 1)  # Middle infra rank -> group rank 1
        self.assertEqual(result[_NodeDesc("node2", 1, 1)], 2)  # Largest infra rank -> group rank 2

    def test_assign_ranks_preserves_slurm_topology_order(self) -> None:
        """
        Test that participants are assigned group ranks in SLURM topology order
        (sorted by infrastructure rank), regardless of node descriptor sort order.
        """
        from nvidia_resiliency_ext.fault_tolerance._ft_rendezvous import (
            _DistributedRendezvousOpExecutor,
        )

        # Create nodes that sort alphabetically as: aaa < bbb < zzz
        # But assign them infrastructure ranks in reverse order
        node_aaa = _NodeDesc("aaa_node", 1, 1)
        node_bbb = _NodeDesc("bbb_node", 1, 1)
        node_zzz = _NodeDesc("zzz_node", 1, 1)

        # Verify sort order is as expected
        sorted_nodes = sorted([node_zzz, node_aaa, node_bbb])
        self.assertEqual(sorted_nodes, [node_aaa, node_bbb, node_zzz])

        # Assign infrastructure ranks in reverse of alphabetical order
        participants = {
            node_aaa: 102,  # Largest infra rank
            node_bbb: 101,  # Middle infra rank
            node_zzz: 100,  # Smallest infra rank
        }

        result = _DistributedRendezvousOpExecutor._assign_ranks(participants)

        # Group ranks should follow SLURM topology order (infra rank order), not alphabetical
        self.assertEqual(result[node_zzz], 0)  # Smallest infra rank -> group rank 0
        self.assertEqual(result[node_bbb], 1)  # Middle infra rank -> group rank 1
        self.assertEqual(result[node_aaa], 2)  # Largest infra rank -> group rank 2

    def test_assign_ranks_with_hot_spare_segment_none(self) -> None:
        """Test rank assignment with hot spare nodes (segment=None).

        When there are more participants than min_nodes, the legacy rendezvous
        assigns contiguous group ranks [0, 1, 2, ...] to all participants in
        SLURM topology order. The first min_nodes get active ranks [0..min_nodes-1],
        and extras become hot spares with ranks [min_nodes..N-1].
        """
        from nvidia_resiliency_ext.fault_tolerance._ft_rendezvous import (
            _DistributedRendezvousOpExecutor,
        )

        # Simulate 5 participants (infra ranks 0-4), where min_nodes=3
        # First 3 should be active [0,1,2], remaining 2 are hot spares [3,4]
        participants = {
            _NodeDesc("node0", 1, 1): 0,  # Active: group rank 0
            _NodeDesc("node1", 1, 1): 1,  # Active: group rank 1
            _NodeDesc("node2", 1, 1): 2,  # Active: group rank 2
            _NodeDesc("node3", 1, 1): 3,  # Hot spare: group rank 3
            _NodeDesc("node4", 1, 1): 4,  # Hot spare: group rank 4
        }

        result = _DistributedRendezvousOpExecutor._assign_ranks(participants)

        # All nodes get contiguous group ranks in SLURM topology order
        self.assertEqual(result[_NodeDesc("node0", 1, 1)], 0)
        self.assertEqual(result[_NodeDesc("node1", 1, 1)], 1)
        self.assertEqual(result[_NodeDesc("node2", 1, 1)], 2)
        self.assertEqual(result[_NodeDesc("node3", 1, 1)], 3)  # Hot spare
        self.assertEqual(result[_NodeDesc("node4", 1, 1)], 4)  # Hot spare

        # Verify all ranks are contiguous
        self.assertEqual(set(result.values()), {0, 1, 2, 3, 4})

    def test_assign_ranks_out_of_order_with_hot_spare(self) -> None:
        """Test that out-of-order infra rank arrivals work correctly with hot spares.

        Participants arrive in arbitrary order, but should be sorted by infra_rank
        and assigned group ranks accordingly, with extras becoming hot spares.
        """
        from nvidia_resiliency_ext.fault_tolerance._ft_rendezvous import (
            _DistributedRendezvousOpExecutor,
        )

        # Participants arrive out-of-order (infra ranks: 4, 1, 3, 0, 2)
        # After sorting by infra_rank: 0, 1, 2, 3, 4
        # First 3 get active ranks [0,1,2], remaining 2 are hot spares [3,4]
        participants = {
            _NodeDesc("node_d", 1, 1): 4,  # Out-of-order
            _NodeDesc("node_b", 1, 1): 1,
            _NodeDesc("node_c", 1, 1): 3,
            _NodeDesc("node_a", 1, 1): 0,
            _NodeDesc("node_e", 1, 1): 2,
        }

        result = _DistributedRendezvousOpExecutor._assign_ranks(participants)

        # Should be sorted by infra_rank and assigned accordingly
        self.assertEqual(result[_NodeDesc("node_a", 1, 1)], 0)  # infra_rank 0 -> group rank 0
        self.assertEqual(result[_NodeDesc("node_b", 1, 1)], 1)  # infra_rank 1 -> group rank 1
        self.assertEqual(result[_NodeDesc("node_e", 1, 1)], 2)  # infra_rank 2 -> group rank 2
        self.assertEqual(
            result[_NodeDesc("node_c", 1, 1)], 3
        )  # infra_rank 3 -> group rank 3 (hot spare)
        self.assertEqual(
            result[_NodeDesc("node_d", 1, 1)], 4
        )  # infra_rank 4 -> group rank 4 (hot spare)

        # Verify all ranks are contiguous
        self.assertEqual(set(result.values()), {0, 1, 2, 3, 4})

    def test_assign_ranks_with_gaps_and_hot_spare(self) -> None:
        """Test rank assignment with gaps in infra_ranks (simulating node failures) and hot spares.

        When some nodes fail and don't join, there are gaps in infra_ranks.
        E.g., if nodes 2, 4, 5 failed, we have infra_ranks [0, 1, 3, 6, 7] instead of [0-7].
        These should still get contiguous group ranks [0, 1, 2, 3, 4].
        With hot spares, the first min_nodes are active, rest are hot spares.
        """
        from nvidia_resiliency_ext.fault_tolerance._ft_rendezvous import (
            _DistributedRendezvousOpExecutor,
        )

        # Simulate 5 nodes joining with gaps: infra_ranks [0, 1, 3, 6, 7]
        # Missing nodes: 2, 4, 5 (failed to join)
        # Should get contiguous group ranks [0, 1, 2, 3, 4]
        # If min_nodes=3, first 3 are active, last 2 are hot spares
        participants = {
            _NodeDesc("node0", 1, 1): 0,  # infra_rank 0 -> group rank 0 (active)
            _NodeDesc("node1", 1, 1): 1,  # infra_rank 1 -> group rank 1 (active)
            _NodeDesc("node3", 1, 1): 3,  # infra_rank 3 -> group rank 2 (active)
            _NodeDesc("node6", 1, 1): 6,  # infra_rank 6 -> group rank 3 (hot spare)
            _NodeDesc("node7", 1, 1): 7,  # infra_rank 7 -> group rank 4 (hot spare)
        }

        result = _DistributedRendezvousOpExecutor._assign_ranks(participants)

        # Should get contiguous group ranks [0-4] based on sorted infra_rank
        self.assertEqual(result[_NodeDesc("node0", 1, 1)], 0)
        self.assertEqual(result[_NodeDesc("node1", 1, 1)], 1)
        self.assertEqual(result[_NodeDesc("node3", 1, 1)], 2)
        self.assertEqual(result[_NodeDesc("node6", 1, 1)], 3)
        self.assertEqual(result[_NodeDesc("node7", 1, 1)], 4)

        # Verify all ranks are contiguous
        self.assertEqual(set(result.values()), {0, 1, 2, 3, 4})

    def test_assign_ranks_with_large_gaps_and_hot_spare(self) -> None:
        """Test rank assignment with large gaps in infra_ranks and hot spares.

        More aggressive gap scenario: infra_ranks [0, 5, 10, 15, 20, 22, 24]
        Simulates many node failures (1-4, 6-9, 11-14, 16-19, 21, 23 all failed).
        """
        from nvidia_resiliency_ext.fault_tolerance._ft_rendezvous import (
            _DistributedRendezvousOpExecutor,
        )

        # Large gaps in infra_ranks
        participants = {
            _NodeDesc("node0", 1, 1): 0,
            _NodeDesc("node5", 1, 1): 5,
            _NodeDesc("node10", 1, 1): 10,
            _NodeDesc("node15", 1, 1): 15,
            _NodeDesc("node20", 1, 1): 20,
            _NodeDesc("node22", 1, 1): 22,
            _NodeDesc("node24", 1, 1): 24,
        }

        result = _DistributedRendezvousOpExecutor._assign_ranks(participants)

        # Should get contiguous group ranks [0-6], sorted by infra_rank
        self.assertEqual(result[_NodeDesc("node0", 1, 1)], 0)
        self.assertEqual(result[_NodeDesc("node5", 1, 1)], 1)
        self.assertEqual(result[_NodeDesc("node10", 1, 1)], 2)
        self.assertEqual(result[_NodeDesc("node15", 1, 1)], 3)
        self.assertEqual(result[_NodeDesc("node20", 1, 1)], 4)
        self.assertEqual(result[_NodeDesc("node22", 1, 1)], 5)
        self.assertEqual(result[_NodeDesc("node24", 1, 1)], 6)

        # Verify all ranks are contiguous
        self.assertEqual(set(result.values()), {0, 1, 2, 3, 4, 5, 6})


# RendezvousState is largely unchanged from upstream, but we test serialization
# since it includes FT-specific worker_states field
class RendezvousStateTest(TestCase):
    def test_state_includes_worker_states(self) -> None:
        """Test that RendezvousState includes FT-specific worker_states field."""
        state = _RendezvousState()
        self.assertIsInstance(state.worker_states, dict)
        self.assertEqual(len(state.worker_states), 0)


class FakeRendezvousBackend(RendezvousBackend):
    _state: Optional[bytes]
    _token: int

    def __init__(self) -> None:
        self._state = None
        self._token = 0

    @property
    def name(self) -> str:
        return "fake_backend"

    def get_state(self) -> Optional[Tuple[bytes, Token]]:
        if self._token == 0:
            return None

        return self._state, self._token  # type: ignore[return-value]

    def set_state(
        self, state: bytes, token: Optional[Token] = None
    ) -> Optional[Tuple[bytes, Token, bool]]:
        if token is None:
            token = 0

        if token == self._token:
            self._state = state
            self._token += 1

            has_set = True
        else:
            has_set = False

        return self._state, self._token, has_set  # type: ignore[return-value]

    def get_state_internal(self) -> _RendezvousState:
        return pickle.loads(cast(bytes, self._state))

    def set_state_internal(self, state: _RendezvousState) -> None:
        self._state = pickle.dumps(state)
        self._token += 1

    def corrupt_state(self) -> None:
        self._state = b"corrupt_state"
        self._token += 1


class BackendRendezvousStateHolderTest(TestCase, CustomAssertMixin):
    def setUp(self) -> None:
        self._backend = FakeRendezvousBackend()

        mock_get_state = MagicMock(wraps=self._backend.get_state)
        mock_set_state = MagicMock(wraps=self._backend.set_state)

        self._mock_backend = Mock()
        self._mock_backend.get_state = mock_get_state
        self._mock_backend.set_state = mock_set_state

        setattr(self._backend, "get_state", mock_get_state)  # noqa: B010
        setattr(self._backend, "set_state", mock_set_state)  # noqa: B010

        self._settings = RendezvousSettings(
            run_id="dummy_run_id",
            min_nodes=1,
            max_nodes=1,
            timeout=RendezvousTimeout(),
            keep_alive_interval=timedelta(seconds=30),
            keep_alive_max_attempt=3,
        )

        self._cache_duration = 0

        self._now = datetime(2000, 1, 1, hour=0, minute=0)

        self._datetime_patch = patch(
            "nvidia_resiliency_ext.fault_tolerance._ft_rendezvous.datetime"
        )

        mock_datetime = self._datetime_patch.start()
        mock_datetime.utcnow.return_value = self._now

    def tearDown(self) -> None:
        self._datetime_patch.stop()

    def _create_state(self) -> _RendezvousState:
        state = _RendezvousState()
        state.round = 999
        state.complete = True
        state.deadline = self._now
        state.closed = True
        state.participants = {
            _NodeDesc("dummy1", 1, 1): 0,
            _NodeDesc("dummy2", 1, 1): 1,
            _NodeDesc("dummy3", 1, 1): 2,
        }
        state.wait_list = {
            _NodeDesc("dummy4", 1, 1),
            _NodeDesc("dummy5", 1, 1),
        }
        state.last_heartbeats = {
            _NodeDesc("dummy1", 1, 1): self._now,
            _NodeDesc("dummy2", 1, 1): self._now - timedelta(seconds=15),
            _NodeDesc("dummy3", 1, 1): self._now - timedelta(seconds=30),
            _NodeDesc("dummy4", 1, 1): self._now - timedelta(seconds=60),
            _NodeDesc("dummy5", 1, 1): self._now - timedelta(seconds=90),
        }

        return state

    def _create_state_holder(self) -> _BackendRendezvousStateHolder:
        return _BackendRendezvousStateHolder(self._backend, self._settings, self._cache_duration)

    def test_init_initializes_state_holder(self) -> None:
        state_holder = self._create_state_holder()

        self.assert_state_empty(state_holder.state)

        self._mock_backend.assert_not_called()

    def test_sync_gets_empty_state_if_backend_state_does_not_exist(self) -> None:
        state_holder = self._create_state_holder()

        has_set = state_holder.sync()

        self.assertIsNone(has_set)

        self.assert_state_empty(state_holder.state)

        self.assertEqual(self._mock_backend.get_state.call_count, 1)
        self.assertEqual(self._mock_backend.set_state.call_count, 0)

    def test_sync_gets_backend_state_if_local_state_is_clean(self) -> None:
        state_holder = self._create_state_holder()
        expected_state = self._create_state()
        self._backend.set_state_internal(expected_state)

        has_set = state_holder.sync()

        self.assertIsNone(has_set)
        self.assert_state_equal(state_holder.state, expected_state)
        self.assertEqual(self._mock_backend.get_state.call_count, 1)
        self.assertEqual(self._mock_backend.set_state.call_count, 0)

    def test_sync_sets_backend_state_if_local_state_is_new_and_dirty(self) -> None:
        state_holder = self._create_state_holder()

        state_holder.state.round = 1000
        state_holder.mark_dirty()

        has_set = state_holder.sync()

        self.assertTrue(has_set)
        expected_state = self._backend.get_state_internal()
        self.assert_state_equal(state_holder.state, expected_state)
        self.assertEqual(self._mock_backend.get_state.call_count, 0)
        self.assertEqual(self._mock_backend.set_state.call_count, 1)

    def test_sync_cache_behavior(self) -> None:
        """Test cache behavior including expiration and reuse."""
        state = self._create_state()
        self._backend.set_state_internal(state)

        with patch("nvidia_resiliency_ext.fault_tolerance._ft_rendezvous.time") as mock_time:
            self._cache_duration = 1
            state_holder = self._create_state_holder()

            # First sync - should fetch from backend
            mock_time.monotonic.return_value = 5
            state_holder.sync()
            self.assertEqual(self._mock_backend.get_state.call_count, 1)

            # Second sync within cache duration - should use cache
            has_set = state_holder.sync()
            self.assertIsNone(has_set)
            self.assertEqual(self._mock_backend.get_state.call_count, 1)  # No additional calls

            # Sync after cache expiration - should fetch from backend again
            mock_time.monotonic.return_value = 5 + self._cache_duration + 0.01
            has_set = state_holder.sync()
            self.assertIsNone(has_set)
            self.assertEqual(self._mock_backend.get_state.call_count, 2)

    def test_sync_sanitizes_state(self) -> None:
        state = self._create_state()

        expected_state = copy.deepcopy(state)

        dead_node1 = _NodeDesc("dead1", 1, 1)
        dead_node2 = _NodeDesc("dead2", 1, 1)
        dead_node3 = _NodeDesc("dead3", 1, 1)
        dead_node4 = _NodeDesc("dead4", 1, 1)
        dead_node5 = _NodeDesc("dead5", 1, 1)

        state.last_heartbeats[dead_node1] = self._now - timedelta(seconds=91)
        state.last_heartbeats[dead_node2] = self._now - timedelta(seconds=100)
        state.last_heartbeats[dead_node3] = self._now - timedelta(seconds=110)
        state.last_heartbeats[dead_node4] = self._now - timedelta(seconds=120)
        state.last_heartbeats[dead_node5] = self._now - timedelta(seconds=130)

        state.participants[dead_node1] = 0
        state.participants[dead_node2] = 0
        state.participants[dead_node3] = 0

        state.wait_list.add(dead_node4)
        state.wait_list.add(dead_node5)

        self._backend.set_state_internal(state)

        state_holder = self._create_state_holder()

        state_holder.sync()

        self.assert_state_equal(state_holder.state, expected_state)

    def test_sync_sanitizes_state_if_no_participants_is_left(self) -> None:
        state = self._create_state()

        expected_state = copy.deepcopy(state)

        for node in state.last_heartbeats:
            state.last_heartbeats[node] = self._now - timedelta(seconds=100)

        expected_state.complete = False
        expected_state.round = 1000
        expected_state.participants = {}
        expected_state.wait_list = set()
        expected_state.last_heartbeats = {}

        self._backend.set_state_internal(state)

        state_holder = self._create_state_holder()

        state_holder.sync()

        self.assert_state_equal(state_holder.state, expected_state)

    def test_sync_raises_error_if_backend_state_is_corrupt(self) -> None:
        self._backend.corrupt_state()

        state_holder = self._create_state_holder()

        with self.assertRaisesRegex(
            RendezvousStateError,
            r"^The rendezvous state is corrupt. See inner exception for details.$",
        ):
            state_holder.sync()


class FakeRendezvousStateHolder(_RendezvousStateHolder):
    _state: _RendezvousState
    _dirty: Optional[bool]

    def __init__(self) -> None:
        self._state = _RendezvousState()
        self._dirty = None

    @property
    def state(self) -> _RendezvousState:
        return self._state

    @state.setter
    def state(self, value) -> None:
        self._state = value

    def sync(self) -> Optional[bool]:
        self._dirty, dirty = None, self._dirty

        return dirty

    def mark_dirty(self) -> None:
        self._dirty = True


class DistributedRendezvousOpExecutorTest(TestCase, CustomAssertMixin):
    def setUp(self) -> None:
        self._node = _NodeDesc("this_node", 1, 1)

        self._state_holder = FakeRendezvousStateHolder()

        mock_sync = MagicMock(wraps=self._state_holder.sync)
        mock_mark = MagicMock(wraps=self._state_holder.mark_dirty)

        self._mock_state_holder = Mock()
        self._mock_state_holder.sync = mock_sync
        self._mock_state_holder.mark = mock_mark

        setattr(self._state_holder, "sync", mock_sync)  # noqa: B010
        setattr(self._state_holder, "mark_dirty", mock_mark)  # noqa: B010

        self._state = self._state_holder.state

        self._min_nodes = 1
        self._max_nodes = 1

        self._timeout = RendezvousTimeout()

        self._now = datetime(2000, 1, 1, hour=0, minute=0)

        self._datetime_patch = patch(
            "nvidia_resiliency_ext.fault_tolerance._ft_rendezvous.datetime"
        )

        mock_datetime = self._datetime_patch.start()
        mock_datetime.utcnow.return_value = self._now

    def tearDown(self) -> None:
        self._datetime_patch.stop()

    def _create_settings(self) -> RendezvousSettings:
        return RendezvousSettings(
            run_id="dummy_run_id",
            min_nodes=self._min_nodes,
            max_nodes=self._max_nodes,
            timeout=self._timeout,
            keep_alive_interval=timedelta(seconds=30),
            keep_alive_max_attempt=3,
        )

    def _create_op_executor(
        self, settings: Optional[RendezvousSettings] = None
    ) -> _DistributedRendezvousOpExecutor:
        self._state_holder.state = self._state

        if settings is None:
            settings = self._create_settings()

        return _DistributedRendezvousOpExecutor(self._node, self._state_holder, settings)

    def _run_action(self, action: _Action) -> None:
        op_executor = self._create_op_executor()

        op = MagicMock(side_effect=[action, _Action.FINISH])

        op_executor.run(op, deadline=1)

    def _assert_action(self, action: _Action, expected_state: _RendezvousState) -> None:
        self._run_action(action)

        self.assert_state_equal(self._state, expected_state)

        self.assertListEqual(
            self._mock_state_holder.mock_calls, [call.sync(), call.mark(), call.sync()]
        )

    def test_run_passes_expected_context_and_deadline_to_state_handler(self) -> None:
        settings = self._create_settings()

        op_executor = self._create_op_executor(settings)

        op = MagicMock(return_value=_Action.FINISH)

        op_executor.run(op, deadline=3)

        ctx, deadline = op.call_args[0]  # args

        self.assertIs(ctx.node, self._node)
        self.assertIs(ctx.state, self._state)
        self.assertIs(ctx.settings, settings)

        self.assertEqual(deadline, 3)

    def test_run_keeps_alive(self) -> None:
        expected_state = _RendezvousState()

        expected_state.last_heartbeats[self._node] = self._now

        self._assert_action(_Action.KEEP_ALIVE, expected_state)

    def test_run_adds_to_participants(self) -> None:
        """Test adding node to participants from both clean state and waitlist."""
        # Test 1: Clean state
        expected_state = _RendezvousState()
        expected_state.participants[self._node] = 0
        expected_state.last_heartbeats[self._node] = self._now
        self._min_nodes = 2
        self._max_nodes = 2
        self._assert_action(_Action.ADD_TO_PARTICIPANTS, expected_state)

        # Test 2: Node was in waitlist
        self._mock_state_holder.reset_mock()
        self._state.wait_list.add(self._node)
        expected_state = _RendezvousState()
        expected_state.participants[self._node] = 0
        expected_state.last_heartbeats[self._node] = self._now
        self._assert_action(_Action.ADD_TO_PARTICIPANTS, expected_state)

    def _add_participants(
        self, num_participants: int, state: _RendezvousState, ranked: bool = False
    ) -> None:
        for i in range(num_participants):
            if ranked:
                node = _NodeDesc(f"dummy{i}", 1, 1)
                rank = i
            else:
                node = _NodeDesc(f"dummy{num_participants - i - 1}", 1, 1)  # Add in reverse.
                rank = 0

            state.participants[node] = rank

            state.worker_states[node] = WorkerState.HEALTHY

            state.last_heartbeats[node] = self._now

    def test_run_adds_to_participants_and_starts_last_call_if_min_nodes_is_reached(self) -> None:
        """Test that last call starts when min nodes is reached."""
        # Test with representative case instead of exhaustive loop
        num_participants = 2
        self._state = _RendezvousState()
        self._add_participants(num_participants, self._state)
        self._state.wait_list.add(self._node)

        expected_state = _RendezvousState()
        self._add_participants(num_participants, expected_state)
        expected_state.participants[self._node] = 0
        expected_state.last_heartbeats[self._node] = self._now
        expected_state.deadline = self._now + self._timeout.last_call

        self._min_nodes = num_participants + 1
        self._max_nodes = num_participants + 2

        self._assert_action(_Action.ADD_TO_PARTICIPANTS, expected_state)

    def test_run_adds_to_participants_and_completes_rendezvous_if_max_nodes_is_reached(
        self,
    ) -> None:
        """Test rendezvous completion when max nodes is reached."""
        # Test with representative cases instead of exhaustive loops
        test_cases = [
            (1, True),  # min=max=1, equal case
            (2, False),  # min=0, max=2, unequal case
        ]

        for num_participants, min_max_nodes_equal in test_cases:
            with self.subTest(
                num_participants=num_participants, min_max_nodes_equal=min_max_nodes_equal
            ):
                self._state = _RendezvousState()
                self._add_participants(num_participants, self._state)
                self._state.wait_list.add(self._node)
                self._state.deadline = self._now + self._timeout.last_call

                expected_state = _RendezvousState()
                self._add_participants(num_participants, expected_state, ranked=True)
                expected_state.participants[self._node] = num_participants
                expected_state.last_heartbeats[self._node] = self._now
                expected_state.worker_states[self._node] = WorkerState.HEALTHY
                expected_state.complete = True
                expected_state.deadline = None

                self._min_nodes = num_participants + 1 if min_max_nodes_equal else 0
                self._max_nodes = num_participants + 1

                self._assert_action(_Action.ADD_TO_PARTICIPANTS, expected_state)
                self._mock_state_holder.reset_mock()

    def test_run_adds_to_waitlist(self) -> None:
        expected_state = _RendezvousState()

        expected_state.wait_list.add(self._node)

        expected_state.last_heartbeats[self._node] = self._now

        self._assert_action(_Action.ADD_TO_WAIT_LIST, expected_state)

    def test_run_removes_from_participants(self) -> None:
        """Test removing node from participants in different states."""
        # Test with representative case instead of exhaustive loop
        complete, last_call_deadline = False, self._now

        self._state = _RendezvousState()
        self._add_participants(2, self._state)
        self._state.participants[self._node] = 0
        self._state.last_heartbeats[self._node] = self._now
        self._state.complete = complete
        self._state.deadline = last_call_deadline
        self._state.round = 1

        expected_state = _RendezvousState()
        self._add_participants(2, expected_state)
        expected_state.complete = complete
        expected_state.deadline = last_call_deadline
        expected_state.round = 1

        self._assert_action(_Action.REMOVE_FROM_PARTICIPANTS, expected_state)

    def test_run_removes_from_participants_and_moves_to_next_round_if_node_is_last_participant(
        self,
    ) -> None:
        self._state.participants[self._node] = 0

        self._state.last_heartbeats[self._node] = self._now

        self._state.complete = True

        self._state.round = 1

        expected_state = _RendezvousState()

        expected_state.complete = False

        expected_state.round = 2

        self._assert_action(_Action.REMOVE_FROM_PARTICIPANTS, expected_state)

    def test_run_removes_from_participants_and_clears_last_call_if_rendezvous_has_less_than_min_nodes(
        self,
    ) -> None:
        self._add_participants(2, self._state)

        self._state.participants[self._node] = 0

        self._state.last_heartbeats[self._node] = self._now

        self._state.deadline = self._now

        expected_state = _RendezvousState()

        self._add_participants(2, expected_state)

        self._min_nodes = 3
        self._max_nodes = 4

        self._assert_action(_Action.REMOVE_FROM_PARTICIPANTS, expected_state)

    def test_run_removes_from_waitlist(self) -> None:
        self._state.wait_list.add(self._node)

        self._state.last_heartbeats[self._node] = self._now

        expected_state = _RendezvousState()

        self._assert_action(_Action.REMOVE_FROM_WAIT_LIST, expected_state)

    def test_run_marks_rendezvous_closed(self) -> None:
        expected_state = _RendezvousState()

        expected_state.closed = True

        self._assert_action(_Action.MARK_RENDEZVOUS_CLOSED, expected_state)

    def test_run_raises_error_if_rendezvous_is_closed(self) -> None:
        with self.assertRaises(RendezvousClosedError):
            self._run_action(_Action.ERROR_CLOSED)

        self.assertListEqual(self._mock_state_holder.mock_calls, [call.sync()])

    def test_run_raises_error_if_operation_timed_out(self) -> None:
        with self.assertRaises(RendezvousTimeoutError):
            self._run_action(_Action.ERROR_TIMEOUT)

        self.assertListEqual(self._mock_state_holder.mock_calls, [call.sync()])

    def test_run_delays_execution_if_sync_requested(self) -> None:
        with patch("nvidia_resiliency_ext.fault_tolerance._ft_rendezvous._delay") as mock_delay:
            self._run_action(_Action.SYNC)

            mock_delay.assert_called_once_with(seconds=1)

        self.assertListEqual(self._mock_state_holder.mock_calls, [call.sync(), call.sync()])


# Rendezvous operations (Exit, Join, Close, KeepAlive) are largely unchanged from upstream PyTorch.
# The core operation logic is well-tested in upstream. We focus on FT-specific features like health checks.


class DummyStore(Store):
    def __init__(self):
        super().__init__()
        self._store = {}
        self._timeout = timedelta(seconds=30)

        # Set required keys for RendezvousStoreInfo.build()
        self._store["MASTER_ADDR"] = b"localhost"
        self._store["MASTER_PORT"] = b"29500"

    def get(self, key: str) -> bytes:
        # Handle prefixed keys by stripping the prefix
        if key.startswith("torch.rendezvous.dummy_run_id.0/"):
            key = key[len("torch.rendezvous.dummy_run_id.0/") :]

        if key not in self._store:
            raise RuntimeError(f"Key {key} not found in store")
        return self._store[key]

    def set(self, key: str, value: bytes) -> None:
        # Handle prefixed keys by stripping the prefix
        if key.startswith("torch.rendezvous.dummy_run_id.0/"):
            key = key[len("torch.rendezvous.dummy_run_id.0/") :]

        self._store[key] = value

    def add(self, key: str, num: int) -> int:
        if key not in self._store:
            self._store[key] = b"0"
        current = int(self._store[key].decode())
        new_value = current + num
        self._store[key] = str(new_value).encode()
        return new_value

    def check(self, keys: list) -> bool:
        return all(key in self._store for key in keys)

    def wait(self, keys: list, timeout: timedelta) -> None:
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout.total_seconds():
            if self.check(keys):
                return
            time.sleep(0.001)  # Small sleep to avoid busy waiting
        raise RuntimeError(f"Timeout waiting for keys: {keys}")

    def multi_get(self, keys: list) -> list:
        return [self.get(key) for key in keys]

    def multi_set(self, keys: list, values: list) -> None:
        for key, value in zip(keys, values):
            self.set(key, value)

    def append(self, key: str, value: str) -> None:
        if key not in self._store:
            self._store[key] = b""
        current = self._store[key].decode()
        self._store[key] = (current + value).encode()

    def set_timeout(self, timeout: timedelta) -> None:
        self._timeout = timeout


class DynamicRendezvousHandlerTest(TestCase):
    def setUp(self) -> None:
        self._node = _NodeDesc("this_node", 1, 1)

        self._min_nodes = 1
        self._max_nodes = 1

        self._join_timeout: Optional[timedelta] = None
        self._close_timeout: Optional[timedelta] = None
        self._heartbeat_timeout: Optional[timedelta] = None

        self._keep_alive_interval = timedelta(seconds=30)

        self._store = DummyStore()

        # Don't mock the get method as it interferes with PrefixStore functionality
        # The DummyStore's get method will handle the prefixed keys correctly

        self._state_holder = FakeRendezvousStateHolder()

        self._mock_sync = MagicMock(wraps=self._state_holder.sync)

        setattr(self._state_holder, "sync", self._mock_sync)  # noqa: B010

        self._state = self._state_holder.state

    def _create_handler(self) -> FtRendezvousHandler:
        settings = RendezvousSettings(
            run_id="dummy_run_id",
            min_nodes=self._min_nodes,
            max_nodes=self._max_nodes,
            timeout=RendezvousTimeout(
                join=self._join_timeout,
                close=self._close_timeout,
                heartbeat=self._heartbeat_timeout,
            ),
            keep_alive_interval=self._keep_alive_interval,
            keep_alive_max_attempt=3,
        )

        self._state_holder.state = self._state

        return FtRendezvousHandler(
            self._node, settings, "dummy_backend", self._store, self._state_holder
        )

    # Most handler functionality is inherited from upstream PyTorch.
    # We focus on testing FT-specific health check integration.

    @patch("nvidia_resiliency_ext.fault_tolerance._ft_rendezvous.GPUHealthCheck")
    def test_next_rendezvous_calls_health_check(self, MockGPUHealthCheck) -> None:
        """Test that next_rendezvous calls the health check before proceeding."""
        mock_health_checker = MockGPUHealthCheck.return_value
        mock_health_checker.return_value = True

        handler = self._create_handler()
        handler.next_rendezvous()

        MockGPUHealthCheck.assert_called_once()
        mock_health_checker.assert_called_once()

    @patch("nvidia_resiliency_ext.fault_tolerance._ft_rendezvous.GPUHealthCheck")
    def test_health_check_failure_raises_exception(self, MockGPUHealthCheck) -> None:
        """Test that health check failure raises UnhealthyNodeException."""
        mock_health_checker = MockGPUHealthCheck.return_value
        mock_health_checker.return_value = False

        handler = self._create_handler()

        with self.assertRaises(UnhealthyNodeException):
            handler.next_rendezvous()


class DummyRendezvousBackend(RendezvousBackend):
    def __init__(self):
        self._state = None

    @property
    def name(self):
        return "dummy_backend"

    def get_state(self):
        return (self._state, None) if self._state is not None else None

    def set_state(self, state, token):
        self._state = state  # Just store the state without token validation


class DynamicRendezvousHandlerFromBackendTest(TestCase):
    def setUp(self) -> None:
        self._run_id = "dummy_run_id"
        self._store = DummyStore()
        self._backend = DummyRendezvousBackend()
        self._min_nodes = 3
        self._max_nodes = 6
        self._timeout: Optional[RendezvousTimeout] = RendezvousTimeout()

    def _create_handler(self) -> FtRendezvousHandler:
        return FtRendezvousHandler.from_backend(
            run_id=self._run_id,
            store=self._store,
            backend=self._backend,
            min_nodes=self._min_nodes,
            max_nodes=self._max_nodes,
            timeout=self._timeout,
        )

    def test_init_initializes_handler(self) -> None:
        handler = self._create_handler()

        self.assertEqual(handler.get_backend(), self._backend.name)
        self.assertEqual(handler.get_run_id(), self._run_id)
        self.assertEqual(handler.settings.run_id, self._run_id)
        self.assertEqual(handler.settings.min_nodes, self._min_nodes)
        self.assertEqual(handler.settings.max_nodes, self._max_nodes)

        if self._timeout is None:
            self.assertIsNotNone(handler.settings.timeout)
        else:
            self.assertIs(handler.settings.timeout, self._timeout)

    def test_init_initializes_handler_variations(self) -> None:
        """Test handler initialization with different parameter combinations."""
        # Test 1: No timeout specified
        self._timeout = None
        self.test_init_initializes_handler()

        # Test 2: Min and max nodes equal
        self._timeout = RendezvousTimeout()  # Reset timeout
        self._min_nodes = 3
        self._max_nodes = 3
        self.test_init_initializes_handler()

    def test_init_raises_error_if_min_nodes_is_not_positive(self) -> None:
        for num in [0, -10]:
            with self.subTest(min_nodes=num):
                self._min_nodes = num

                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The minimum number of nodes \({num}\) must be greater than zero.$",
                ):
                    self._create_handler()

    def test_init_raises_error_if_max_nodes_is_less_than_min(self) -> None:
        self._min_nodes = 3
        self._max_nodes = 2

        with self.assertRaisesRegex(
            ValueError,
            rf"^The maximum number of nodes \({self._max_nodes}\) must be greater than or equal to "
            "the minimum number of nodes "
            rf"\({self._min_nodes}\).$",
        ):
            self._create_handler()


# Handler creation logic is standard factory pattern, well-tested in upstream.
# We focus on FT-specific integration tests instead.


def _ignore_exception(exception_type: Exception, fn: Callable):
    try:
        fn()
    except exception_type as e:
        pass


def _wait_for(condition, timeout=30, interval=1, name=None):
    term_flag = threading.Event()

    def _wait_while():
        while not term_flag.is_set():
            if condition():
                break
            else:
                time.sleep(interval)

    wait_thread = threading.Thread(target=_wait_while, name=name)
    wait_thread.start()
    wait_thread.join(timeout=timeout)
    if wait_thread.is_alive():
        term_flag.set()
        raise RuntimeError(f"_wait_for timeout. waited {timeout} seconds")


class _CapturingThread(threading.Thread):
    def __init__(self, target, ready_event=None, name=None, args=None, kwargs=None):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        super().__init__(target=target, name=name)
        self._target = target
        self._args = args
        self._kwargs = kwargs
        self.exception = None
        self.result = None
        self.ready_event = ready_event  # Event to signal readiness

    def run(self):
        try:
            if self.ready_event:
                self.ready_event.set()  # Signal that this thread has truly started
            # Ensure we only signal readiness after reaching an important execution point
            if self._target is not None:
                self.result = self._target(*self._args, **self._kwargs)

        except UnhealthyNodeException as e:
            self.exception = e
            return  # Exit immediately
        except Exception as e:
            self.exception = e
        finally:
            self._target = None  # Free references safely

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        if self.exception:
            raise self.exception
        return self.result

    def stop(self):
        if self.is_alive():
            self.join(timeout=1)  # Ensure the thread is properly terminated


class IntegrationTest(TestCase):
    def setUp(self) -> None:
        self._store = torch.distributed.HashStore()
        self._handlers = []
        self._backend = C10dRendezvousBackend(store=self._store, run_id="dummy_run_id00")

    def tearDown(self) -> None:
        for handler in self._handlers:
            handler.shutdown()
            # handler._stop_heartbeats()

    def _create_handler(self, **kwargs) -> FtRendezvousHandler:
        params = {
            "backend": self._backend.name,
            "endpoint": "dummy_endpoint",
            "run_id": "dummy_run_id",
            "min_nodes": 2,
            "max_nodes": 2,
            "join_timeout": "5",
            "last_call_timeout": "2",
            "local_addr": f"address_{len(self._handlers)}",
            "upscaling_enabled": True,
        }
        params.update(**kwargs)

        rzdv_params = RendezvousParameters(**params)

        handler = create_handler(self._store, self._backend, rzdv_params)

        self._handlers.append(handler)
        return handler

    # Basic rendezvous functionality is tested in upstream PyTorch.
    # We focus on FT-specific features: health checks and worker state management.

    @patch("nvidia_resiliency_ext.fault_tolerance._ft_rendezvous.GPUHealthCheck")
    def test_all_nodes_failed_rendezvous_with_health_check_(self, MockGPUHealthCheck) -> None:
        """Test that both nodes join the rendezvous when health check passes."""
        # Simulate first node as healthy and second node as unhealthy
        mock_health_checker = MockGPUHealthCheck.return_value
        mock_health_checker.side_effect = [False, False]  # Both nodes fail

        handler1 = self._create_handler(min_nodes=2, max_nodes=2)
        handler2 = self._create_handler(min_nodes=2, max_nodes=2)

        handler1_thread = _CapturingThread(target=handler1.next_rendezvous)
        handler2_thread = _CapturingThread(target=handler2.next_rendezvous)

        handler1_thread.start()
        handler2_thread.start()

        # Assert exceptions were raised during join
        with self.assertRaises(UnhealthyNodeException) as cm1:
            handler1_thread.join(timeout=30)

        with self.assertRaises(UnhealthyNodeException) as cm2:
            handler2_thread.join(timeout=30)

        # Optionally validate exception messages
        self.assertIn("Node", str(cm1.exception))
        self.assertIn("has an unhealthy GPU", str(cm1.exception))
        self.assertIn("Node", str(cm2.exception))
        self.assertIn("has an unhealthy GPU", str(cm2.exception))

    def test_one_node_failed_and_one_pass_rendezvous_with_health_check_(self) -> None:
        """Test that one node joins the rendezvous if another fails due to the health check."""
        max_nodes = 2
        for min_nodes in [1, 2]:
            self.setUp()

            handler1 = self._create_handler(min_nodes=min_nodes, max_nodes=max_nodes)
            handler2 = self._create_handler(min_nodes=min_nodes, max_nodes=max_nodes)

            # Simulate first node as healthy and second node as unhealthy
            def __mock_ensure_node_is_healthy(self, *args, **kwargs):
                if self is handler2:
                    raise UnhealthyNodeException("Node has an unhealthy GPU")
                else:
                    pass

            handler1.ensure_node_is_healthy = MethodType(__mock_ensure_node_is_healthy, handler1)
            handler2.ensure_node_is_healthy = MethodType(__mock_ensure_node_is_healthy, handler2)

            handler1_thread = _CapturingThread(target=handler1.next_rendezvous)
            handler2_thread = _CapturingThread(target=handler2.next_rendezvous)

            handler1_thread.start()
            handler2_thread.start()

            # Node 2 fails the health check
            with self.assertRaises(UnhealthyNodeException):
                handler2_thread.join(timeout=30)
            # Ensure unhealthy node's thread exits immediately
            self.assertIsInstance(handler2_thread.exception, UnhealthyNodeException)
            self.assertFalse(handler2_thread.is_alive())
            # Validate exception messages
            self.assertIn("Node", str(handler2_thread.exception))
            self.assertIn("has an unhealthy GPU", str(handler2_thread.exception))

            # Success of the other node depends on the min_nodes required
            if min_nodes == max_nodes:
                # Can't collect all nodes as the other one failed health check
                with self.assertRaises(RendezvousTimeoutError):
                    handler1_thread.join(timeout=30)
            else:
                pass  # All right, rendezvous completed without a faulty node

            self.tearDown()  # reset the instance before next iteration

    @patch("nvidia_resiliency_ext.fault_tolerance._ft_rendezvous.GPUHealthCheck")
    def test_rendezvous_with_one_failed_health_check_no_threads(self, MockGPUHealthCheck):
        """Test that only one node joins the rendezvous when the other fails health check."""
        # Each handler is processed in order: handler1 is tested first, followed by handler2.
        # No need for threads, locks, or synchronization.
        # Simulate first node as healthy and second node as unhealthy
        mock_health_checker = MockGPUHealthCheck.return_value
        mock_health_checker.side_effect = [True, False]  # First node passes, second fails

        # Create handlers with min and max nodes of 2
        handler1 = self._create_handler(min_nodes=1, max_nodes=2)
        handler2 = self._create_handler(min_nodes=1, max_nodes=2)

        # Simulate handler1 joining rendezvous
        rendezvous_info1 = handler1.next_rendezvous()

        # Simulate that handler2 fails health check
        with self.assertRaises(UnhealthyNodeException) as cm2:
            handler2.next_rendezvous()

        # Simulate that handler2 fails health check and doesn't join
        _wait_for(lambda: len(pickle.loads(self._backend.get_state()[0]).participants) == 1)

        state_and_token = self._backend.get_state()
        state = pickle.loads(state_and_token[0])
        # Verify that only handler1 is in participants and handler2 is not due to health check failure
        self.assertEqual(len(state.participants), 1)
        self.assertEqual(len(state.wait_list), 0)

    # Complex redundancy and upscaling logic is largely unchanged from upstream PyTorch.
    # We keep the core health check integration test as it's FT-specific.

    def test_worker_states_valid_transitions(self) -> None:
        valid_transitions = [
            (WorkerState.HEALTHY, WorkerState.SUCCEEDED),
            (WorkerState.HEALTHY, WorkerState.FAILED),
            (WorkerState.HEALTHY, WorkerState.UNHEALTHY),
            (WorkerState.HEALTHY, WorkerState.UNKNOWN),
            (WorkerState.UNHEALTHY, WorkerState.FAILED),
        ]
        for state1, state2 in valid_transitions:
            try:
                self.setUp()
                handler1 = self._create_handler(
                    min_nodes=1,
                    max_nodes=1,
                )
                handler1.next_rendezvous()
                node = handler1._this_node
                self.assertSetEqual(set(handler1.get_worker_states().keys()), {node})
                self.assertEqual(handler1.get_worker_states()[node], WorkerState.HEALTHY)
                self.assertEqual(handler1.try_set_worker_state(state1), state1)
                self.assertEqual(handler1.try_set_worker_state(state1), state1)
                self.assertEqual(handler1.get_worker_states()[node], state1)
                self.assertEqual(handler1.try_set_worker_state(state2), state2)
                self.assertEqual(handler1.get_worker_states()[node], state2)
                self.assertSetEqual(set(handler1.get_worker_states().keys()), {node})
            finally:
                self.tearDown()

    @patch.dict(os.environ, {"GROUP_RANK": "0"}, clear=False)
    def test_infra_rank_from_env_var(self) -> None:
        """Test that infrastructure rank is used when GROUP_RANK is set."""
        handler = self._create_handler(
            min_nodes=1,
            max_nodes=1,
        )

        # Perform rendezvous
        rdzv_info = handler.next_rendezvous()
        rank, world_size = _extract_rendezvous_info(rdzv_info)

        # Verify rank matches GROUP_RANK environment variable
        self.assertEqual(rank, 0)
        self.assertEqual(world_size, 1)

    @patch.dict(os.environ, {}, clear=False)
    def test_use_deterministic_rank_without_env_var(self) -> None:
        """Test that ranks are assigned deterministically when env vars are not set."""
        # Remove GROUP_RANK if it exists
        os.environ.pop("GROUP_RANK", None)
        os.environ.pop("SLURM_PROCID", None)

        handler = self._create_handler(
            min_nodes=1,
            max_nodes=1,
        )

        # Should succeed with deterministic rank assignment
        rdzv_info = handler.next_rendezvous()
        rank, world_size = _extract_rendezvous_info(rdzv_info)

        # Verify rank is assigned (should be 0 for single node)
        self.assertEqual(rank, 0)
        self.assertEqual(world_size, 1)

    def test_worker_states_invalid_transitions(self) -> None:
        # one final state should not be changed into another final state
        invalid_transitions = [
            (WorkerState.SUCCEEDED, WorkerState.FAILED),
            (WorkerState.FAILED, WorkerState.SUCCEEDED),
            (WorkerState.UNKNOWN, WorkerState.FAILED),
            (WorkerState.UNKNOWN, WorkerState.SUCCEEDED),
        ]
        for state1, state2 in invalid_transitions:
            try:
                self.setUp()
                handler1 = self._create_handler(
                    min_nodes=1,
                    max_nodes=1,
                )
                handler1.next_rendezvous()
                node = handler1._this_node
                self.assertSetEqual(set(handler1.get_worker_states().keys()), {node})
                self.assertEqual(handler1.get_worker_states()[node], WorkerState.HEALTHY)
                self.assertEqual(handler1.try_set_worker_state(state1), state1)
                self.assertEqual(handler1.try_set_worker_state(state1), state1)
                self.assertEqual(handler1.get_worker_states()[node], state1)
                self.assertEqual(handler1.try_set_worker_state(state2), state1)
            finally:
                self.tearDown()
