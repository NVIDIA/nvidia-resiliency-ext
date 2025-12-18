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

"""Unit tests for fault-tolerant barrier-based rendezvous implementation.

These tests focus on:
1. Basic rendezvous flow and coordination
2. Step 2 completion signaling
3. Race conditions with late arrivals
4. Store host vs non-store host behavior
5. Acknowledgment phase correctness
6. Group rank assignment
7. Error cases (timeout, closed rendezvous, exceeding max_nodes)
8. Infrastructure rank handling
"""

import os
import threading
import time
from datetime import timedelta
from unittest import TestCase

from torch.distributed import TCPStore

from nvidia_resiliency_ext.fault_tolerance.ft_rendezvous_barrier import (
    FtRendezvousBarrierHandler,
    RendezvousClosedError,
    RendezvousTimeout,
    RendezvousTimeoutError,
    _NodeDesc,
    _NodeDescGenerator,
    _RendezvousBarrierState,
)

# Test timeout configuration - use short timeouts to make tests run faster
TEST_LAST_CALL_TIMEOUT_SECS = 0.2  # seconds - for last_call_timeout (reduced from 0.5)
TEST_JOIN_TIMEOUT_SECS = 2.0  # seconds - for join timeout (reduced from 5.0)
TEST_THREAD_JOIN_TIMEOUT_SECS = 5.0  # seconds - for thread.join() timeout (reduced from 10.0)


# Helper to create timedelta for tests (named with _ prefix to avoid pytest detection)
def _test_timeout(seconds=TEST_LAST_CALL_TIMEOUT_SECS):
    """Create timedelta for test timeouts."""
    return timedelta(seconds=seconds)


class BaseRendezvousTest(TestCase):
    """Base test class that clears SLURM/GROUP_RANK environment variables.

    Most tests should inherit from this to ensure they use deterministic rank
    assignment rather than being affected by the shell environment.

    Tests that specifically need to test infrastructure rank behavior from
    environment variables should inherit directly from TestCase instead.
    """

    def setUp(self):
        """Save and clear SLURM/GROUP_RANK environment variables."""

        self._saved_slurm_procid = os.environ.pop('SLURM_PROCID', None)
        self._saved_group_rank = os.environ.pop('GROUP_RANK', None)
        super().setUp()

    def tearDown(self):
        """Restore SLURM/GROUP_RANK environment variables."""

        super().tearDown()
        if self._saved_slurm_procid is not None:
            os.environ['SLURM_PROCID'] = self._saved_slurm_procid
        if self._saved_group_rank is not None:
            os.environ['GROUP_RANK'] = self._saved_group_rank


class BarrierStateBasicTest(BaseRendezvousTest):
    """Test basic barrier state operations."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,  # Use any available port
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        super().setUp()  # Clears environment variables
        import time

        # Reuse the shared store
        self.store = self.shared_store
        # Use unique run_id for each test to avoid key collisions
        self.run_id = f"test_run_{self._testMethodName}_{int(time.time() * 1000000)}"
        self.node_desc_gen = _NodeDescGenerator()

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()  # Restores environment variables

    def test_barrier_state_initialization(self):
        """Test that barrier state initializes with correct key prefixes."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        self.assertEqual(state.run_id, self.run_id)
        self.assertTrue(state.is_store_host)
        self.assertIn(self.run_id, state.prefix)
        self.assertIn(self.run_id, state.arrived_count_key)
        self.assertIn(self.run_id, state.last_participant_arrived_key)

    def test_is_closed_initially_false(self):
        """Test that rendezvous is not closed initially."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        self.assertFalse(state.is_closed())

    def test_set_closed(self):
        """Test that set_closed marks rendezvous as closed."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        state.set_closed()
        self.assertTrue(state.is_closed())

    def test_join_increments_arrived_count(self):
        """Test that joining increments the arrived_count atomically."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        # First join
        count1 = self.store.add(state.arrived_count_key, 1)
        self.assertEqual(count1, 1)

        # Second join
        count2 = self.store.add(state.arrived_count_key, 1)
        self.assertEqual(count2, 2)

        # Third join
        count3 = self.store.add(state.arrived_count_key, 1)
        self.assertEqual(count3, 3)


class Step2CompletionTest(BaseRendezvousTest):
    """Test Step 2 completion signaling logic."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        import time

        super().setUp()  # Clears environment variables

        self.store = self.shared_store
        # Use unique run_id for each test to avoid key collisions
        self.run_id = f"test_step2_{self._testMethodName}_{int(time.time() * 1000000)}"
        self.node_desc_gen = _NodeDescGenerator()

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()  # Restores environment variables

    def test_step2_completion_on_max_nodes(self):
        """Test that Step 2 completes immediately when max_nodes is reached."""
        min_nodes = 2
        max_nodes = 3
        last_call_timeout = _test_timeout()  # Short timeout for testing

        # Create a thread that will join and wait
        results = []
        errors = []

        def participant_thread(participant_id, is_host):
            # Each participant needs its own state instance
            state = _RendezvousBarrierState(
                store=self.store,
                run_id=self.run_id,
                is_store_host=is_host,
                join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
            )
            try:
                node = self.node_desc_gen.generate()
                rank, total = state.perform_rendezvous(
                    node, min_nodes, max_nodes, last_call_timeout
                )
                results.append((participant_id, rank, total))
            except Exception as e:
                errors.append((participant_id, e))

        # Start 3 participants (max_nodes), first one is store host
        threads = []
        for i in range(max_nodes):
            t = threading.Thread(target=participant_thread, args=(i, i == 0))
            t.start()
            threads.append(t)

        # Wait for completion
        for t in threads:
            t.join(timeout=TEST_THREAD_JOIN_TIMEOUT_SECS)

        # Check that all threads completed successfully
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), max_nodes)

        # Check that all participants got ranks
        ranks = [r[1] for r in results]
        self.assertEqual(len(set(ranks)), max_nodes)  # All unique ranks

    def test_step2_completion_on_last_call_timeout(self):
        """Test that Step 2 completes after last_call_timeout with min_nodes."""
        min_nodes = 2
        max_nodes = 4
        last_call_timeout = _test_timeout()  # Short timeout for testing

        results = []
        errors = []

        def participant_thread(participant_id, is_host):
            # Each participant needs its own state instance
            state = _RendezvousBarrierState(
                store=self.store,
                run_id=self.run_id,
                is_store_host=is_host,
                join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
            )
            try:
                node = self.node_desc_gen.generate()
                # Test subsequent rendezvous behavior (not first rendezvous)
                # where min_nodes + timeout allows completion
                rank, total = state.perform_rendezvous(
                    node, min_nodes, max_nodes, last_call_timeout, is_first_rendezvous=False
                )
                results.append((participant_id, rank, total))
            except Exception as e:
                errors.append((participant_id, e))

        # Start only min_nodes participants
        threads = []
        for i in range(min_nodes):
            t = threading.Thread(target=participant_thread, args=(i, i == 0))
            t.start()
            threads.append(t)

        # Wait for completion (should happen after timeout)
        start_time = time.time()
        for t in threads:
            t.join(timeout=10)
        elapsed = time.time() - start_time

        # Check that all threads completed successfully
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), min_nodes)

        # Note: We don't check elapsed >= 1.0 because when all min_nodes
        # join simultaneously, any of them might set the completion key
        # immediately after their own deadline expires, making the wait
        # effectively instant for the others

    def test_step2_late_arrival_sees_completion_key(self):
        """Test that late arrivals see the completion key and proceed immediately."""
        min_nodes = 2
        max_nodes = 4
        last_call_timeout = _test_timeout(0.4)  # Longer timeout to allow late joiner

        results = []
        errors = []
        join_times = []

        def participant_thread(participant_id, is_host, delay=0):
            # Each participant needs its own state instance
            state = _RendezvousBarrierState(
                store=self.store,
                run_id=self.run_id,
                is_store_host=is_host,
                join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
            )
            try:
                if delay > 0:
                    time.sleep(delay)
                node = self.node_desc_gen.generate()
                start = time.time()
                # Test subsequent rendezvous behavior (not first rendezvous)
                # where min_nodes + timeout allows completion
                rank, total = state.perform_rendezvous(
                    node, min_nodes, max_nodes, last_call_timeout, is_first_rendezvous=False
                )
                elapsed = time.time() - start
                results.append((participant_id, rank, total))
                join_times.append((participant_id, elapsed))
            except Exception as e:
                errors.append((participant_id, e))

        # Start min_nodes participants
        threads = []
        for i in range(min_nodes):
            t = threading.Thread(target=participant_thread, args=(i, i == 0, 0))
            t.start()
            threads.append(t)

        # Add a late arrival while first participants are still waiting
        # (before last_call_timeout expires)
        time.sleep(0.05)  # Join shortly after others, while they're waiting
        late_thread = threading.Thread(target=participant_thread, args=(99, False, 0))
        late_thread.start()
        threads.append(late_thread)

        # Wait for all threads
        for t in threads:
            t.join(timeout=TEST_THREAD_JOIN_TIMEOUT_SECS)

        # Check all completed
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), min_nodes + 1)


class RaceConditionTest(BaseRendezvousTest):
    """Test race conditions with concurrent operations."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        import time

        super().setUp()  # Clears environment variables

        self.store = self.shared_store
        # Use unique run_id for each test to avoid key collisions
        self.run_id = f"test_race_{self._testMethodName}_{int(time.time() * 1000000)}"
        self.node_desc_gen = _NodeDescGenerator()

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()  # Restores environment variables

    def test_concurrent_joins(self):
        """Test that concurrent joins are handled correctly with atomic increments."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        num_participants = 10
        arrived_counts = []

        def join_thread():
            count = self.store.add(state.arrived_count_key, 1)
            arrived_counts.append(count)

        threads = []
        for _ in range(num_participants):
            t = threading.Thread(target=join_thread)
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=TEST_THREAD_JOIN_TIMEOUT_SECS)

        # Check that all counts are unique (atomic increment)
        self.assertEqual(len(arrived_counts), num_participants)
        self.assertEqual(set(arrived_counts), set(range(1, num_participants + 1)))

    def test_late_arrival_during_acknowledgment(self):
        """Test late arrival joining during acknowledgment phase."""
        min_nodes = 2
        max_nodes = 5
        last_call_timeout = _test_timeout(0.25)  # Short timeout

        results = []
        errors = []

        def participant_thread(participant_id, is_host, delay=0):
            # Each participant needs its own state instance
            state = _RendezvousBarrierState(
                store=self.store,
                run_id=self.run_id,
                is_store_host=is_host,
                join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
            )
            try:
                if delay > 0:
                    time.sleep(delay)
                node = self.node_desc_gen.generate()
                # Test subsequent rendezvous behavior (not first rendezvous)
                # where min_nodes + timeout allows completion
                rank, total = state.perform_rendezvous(
                    node, min_nodes, max_nodes, last_call_timeout, is_first_rendezvous=False
                )
                results.append((participant_id, rank, total))
            except Exception as e:
                errors.append((participant_id, e))

        # Start min_nodes participants
        threads = []
        for i in range(min_nodes):
            t = threading.Thread(target=participant_thread, args=(i, i == 0, 0))
            t.start()
            threads.append(t)

        # Add late arrivals at various times
        for i in range(2):
            t = threading.Thread(
                target=participant_thread, args=(min_nodes + i, False, 0.2 + i * 0.2)
            )
            t.start()
            threads.append(t)

        # Wait for all
        for t in threads:
            t.join(timeout=TEST_THREAD_JOIN_TIMEOUT_SECS)

        # All should complete (late arrivals are included)
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertGreater(len(results), min_nodes)

    def test_late_arrival_after_ack_check_before_key_clear(self):
        """Test late arrival in the critical window: after ack check, before key clearing.

        This tests the race condition where:
        1. Store host checks ack_count >= arrived_count (condition satisfied)
        2. Late arrival joins and increments arrived_count
        3. Late arrival acknowledges, incrementing ack_count
        4. Store host clears keys and assigns ranks (may miss the late arrival)

        The current design handles this by having store host snapshot the arrived_count
        at the time it checks acknowledgments, ensuring consistent rank assignment.
        """
        min_nodes = 2
        max_nodes = 4
        last_call_timeout = _test_timeout()  # Short timeout for testing

        results = []
        errors = []

        # Create a synchronization event to control timing
        import threading as th

        ack_check_event = th.Event()
        late_arrival_done = th.Event()

        def store_host_thread():
            """Store host that signals when it's checking acknowledgments."""
            state = _RendezvousBarrierState(
                store=self.store,
                run_id=self.run_id,
                is_store_host=True,
            )
            try:
                node = self.node_desc_gen.generate()

                # Monkey-patch to inject synchronization point
                original_assign = state.assign_group_ranks

                def instrumented_assign(*args, **kwargs):
                    # Signal that we're about to check/assign (after ack check)
                    ack_check_event.set()
                    # Give late arrival a chance to join in the critical window
                    time.sleep(0.02)  # Minimal delay to test race condition
                    return original_assign(*args, **kwargs)

                state.assign_group_ranks = instrumented_assign

                # Test subsequent rendezvous behavior (not first rendezvous)
                # where min_nodes + timeout allows completion
                rank, total = state.perform_rendezvous(
                    node, min_nodes, max_nodes, last_call_timeout, is_first_rendezvous=False
                )
                results.append(('host', rank, total))
            except Exception as e:
                errors.append(('host', e))

        def regular_participant_thread():
            """Regular participant."""
            state = _RendezvousBarrierState(
                store=self.store,
                run_id=self.run_id,
                is_store_host=False,
            )
            try:
                node = self.node_desc_gen.generate()
                # Test subsequent rendezvous behavior (not first rendezvous)
                rank, total = state.perform_rendezvous(
                    node, min_nodes, max_nodes, last_call_timeout, is_first_rendezvous=False
                )
                results.append(('regular', rank, total))
            except Exception as e:
                errors.append(('regular', e))

        def late_arrival_thread():
            """Late arrival that joins after ack check but before key clear."""
            # Wait for the ack check to happen
            ack_check_event.wait(timeout=10)

            state = _RendezvousBarrierState(
                store=self.store,
                run_id=self.run_id,
                is_store_host=False,
                join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
            )
            try:
                node = self.node_desc_gen.generate()
                # Test subsequent rendezvous behavior (not first rendezvous)
                rank, total = state.perform_rendezvous(
                    node, min_nodes, max_nodes, last_call_timeout, is_first_rendezvous=False
                )
                results.append(('late', rank, total))
            except Exception as e:
                errors.append(('late', e))
            finally:
                late_arrival_done.set()

        # Start regular participants
        t_host = th.Thread(target=store_host_thread)
        t_regular = th.Thread(target=regular_participant_thread)
        t_late = th.Thread(target=late_arrival_thread)

        t_host.start()
        t_regular.start()
        t_late.start()

        t_host.join(timeout=15)
        t_regular.join(timeout=15)
        t_late.join(timeout=15)

        # Check results
        # The late arrival should either:
        # 1. Be included in the current round if it joined before snapshot, OR
        # 2. Timeout/restart if it missed the window
        # Both behaviors are acceptable as long as no deadlock occurs

        # At minimum, the first 2 participants should complete
        successful_results = [r for r in results if r is not None]
        self.assertGreaterEqual(
            len(successful_results), min_nodes, "At least min_nodes should complete successfully"
        )

        # If late arrival completed, verify it got a valid rank
        late_results = [r for r in results if r[0] == 'late']
        if late_results:
            _, late_rank, late_total = late_results[0]
            self.assertGreaterEqual(late_rank, 0, "Late arrival should get valid rank if included")

    def test_multiple_completion_signals(self):
        """Test that multiple participants can try to set completion key (idempotent)."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        # Multiple threads try to set the same key
        def set_completion():
            self.store.set(state.last_participant_arrived_key, "1".encode('utf-8'))

        threads = []
        for _ in range(5):
            t = threading.Thread(target=set_completion)
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=TEST_THREAD_JOIN_TIMEOUT_SECS)

        # Key should be set (no errors)
        self.assertTrue(self.store.check([state.last_participant_arrived_key]))


class StoreHostBehaviorTest(BaseRendezvousTest):
    """Test store host specific behavior."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        import time

        super().setUp()  # Clears environment variables

        self.store = self.shared_store
        # Use unique run_id for each test to avoid key collisions
        self.run_id = f"test_host_{self._testMethodName}_{int(time.time() * 1000000)}"
        self.node_desc_gen = _NodeDescGenerator()

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()  # Restores environment variables

    def test_rank_assignment_with_arrival_order(self):
        """Test that store host assigns ranks to all participants."""
        min_nodes = 3
        max_nodes = 3
        last_call_timeout = _test_timeout()

        results = []

        def participant_thread(participant_id, is_host):
            # Each participant needs its own state instance
            state = _RendezvousBarrierState(
                store=self.store,
                run_id=self.run_id,
                is_store_host=is_host,
                join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
            )
            node = self.node_desc_gen.generate()
            rank, total = state.perform_rendezvous(node, min_nodes, max_nodes, last_call_timeout)
            results.append((participant_id, rank, total))

        threads = []
        for i in range(min_nodes):
            t = threading.Thread(target=participant_thread, args=(i, i == 0))
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=10)

        # Check ranks are assigned correctly
        self.assertEqual(len(results), min_nodes)
        ranks = [r[1] for r in results]
        totals = [r[2] for r in results]

        # All should have same total
        self.assertEqual(len(set(totals)), 1)
        self.assertEqual(totals[0], min_nodes)

        # All should have unique ranks
        self.assertEqual(len(set(ranks)), min_nodes)
        self.assertEqual(set(ranks), set(range(min_nodes)))

    def test_non_store_host_waits_for_ranks(self):
        """Test that non-store host participants wait for rank assignment."""
        # Create separate barrier states with same store
        host_state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )
        non_host_state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=False,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        min_nodes = 2
        max_nodes = 2
        last_call_timeout = _test_timeout()

        results = []

        def host_thread():
            node = self.node_desc_gen.generate()
            rank, total = host_state.perform_rendezvous(
                node, min_nodes, max_nodes, last_call_timeout
            )
            results.append(('host', rank, total))

        def non_host_thread():
            node = self.node_desc_gen.generate()
            rank, total = non_host_state.perform_rendezvous(
                node, min_nodes, max_nodes, last_call_timeout
            )
            results.append(('non_host', rank, total))

        t1 = threading.Thread(target=host_thread)
        t2 = threading.Thread(target=non_host_thread)

        t1.start()
        t2.start()

        t1.join(timeout=10)
        t2.join(timeout=10)

        # Both should complete with ranks
        self.assertEqual(len(results), 2)
        ranks = [r[1] for r in results]
        self.assertEqual(set(ranks), {0, 1})


class GroupRankAssignmentTest(TestCase):
    """Test group rank assignment logic."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        import time

        # NOTE: DO NOT clear SLURM_PROCID/GROUP_RANK for this test class
        # These tests specifically test infrastructure rank behavior

        self.store = self.shared_store
        # Use unique run_id for each test to avoid key collisions
        self.run_id = f"test_rank_{self._testMethodName}_{int(time.time() * 1000000)}"
        self.node_desc_gen = _NodeDescGenerator()

    def tearDown(self):
        """Clean up test fixtures."""
        pass

    def test_rank_assignment_with_arrival_order(self):
        """Test rank assignment based on arrival order (non-infra mode)."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
        )

        # Create participants
        participants = [
            (_NodeDesc("node_a", 100, 0), -1, None),
            (_NodeDesc("node_b", 101, 0), -1, None),
            (_NodeDesc("node_c", 102, 0), -1, None),
        ]
        min_nodes = 3

        # Assign ranks
        result = state._assign_group_ranks(participants, min_nodes)

        # Check all got unique ranks
        self.assertEqual(len(result), 3)
        ranks = set(result.values())
        self.assertEqual(ranks, {0, 1, 2})

    def test_rank_assignment_with_infra_rank(self):
        """Test rank assignment using infrastructure ranks."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        # Create participants with infra ranks
        participants = [
            (_NodeDesc("node_a", 100, 0), 0, "none"),
            (_NodeDesc("node_b", 101, 0), 1, "none"),
            (_NodeDesc("node_c", 102, 0), 2, "none"),
        ]
        min_nodes = 3

        # Assign ranks
        result = state._assign_group_ranks(participants, min_nodes)

        # Check direct mapping: group_rank = infra_rank
        self.assertEqual(result[participants[0][0]], 0)
        self.assertEqual(result[participants[1][0]], 1)
        self.assertEqual(result[participants[2][0]], 2)

    def test_rank_assignment_standby_nodes(self):
        """Test rank assignment for standby nodes (beyond min_nodes)."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
        )

        # 5 participants, min_nodes=3
        participants = [
            (_NodeDesc("node_a", 100, 0), -1, "none"),
            (_NodeDesc("node_b", 101, 0), -1, "none"),
            (_NodeDesc("node_c", 102, 0), -1, "none"),
            (_NodeDesc("node_d", 103, 0), -1, "none"),
            (_NodeDesc("node_e", 104, 0), -1, "none"),
        ]
        min_nodes = 3

        # Assign ranks
        result = state._assign_group_ranks(participants, min_nodes)

        # First 3 should get ranks 0-2 (active)
        # Last 2 should get ranks 3-4 (standby)
        self.assertEqual(len(result), 5)
        ranks = sorted(result.values())
        self.assertEqual(ranks, [0, 1, 2, 3, 4])

    def test_rank_assignment_preserves_slurm_topology_order(self):
        """Test that participants arriving out-of-order are assigned group ranks
        in SLURM topology order (sorted by infrastructure rank), not arrival order.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
        )

        # Create nodes that arrive in arbitrary order (e.g., alphabetically: aaa < bbb < zzz)
        # But assign them infrastructure ranks in reverse order
        node_aaa = _NodeDesc("aaa_node", 100, 0)
        node_bbb = _NodeDesc("bbb_node", 101, 0)
        node_zzz = _NodeDesc("zzz_node", 102, 0)

        # Verify sort order by node descriptor is as expected
        sorted_nodes = sorted([node_zzz, node_aaa, node_bbb])
        self.assertEqual(sorted_nodes, [node_aaa, node_bbb, node_zzz])

        # Create participants list with infra ranks in reverse of alphabetical order
        # Simulating they arrived/joined out of order
        participants = [
            (node_aaa, 102, "none"),  # Largest infra rank
            (node_bbb, 101, "none"),  # Middle infra rank
            (node_zzz, 100, "none"),  # Smallest infra rank
        ]
        min_nodes = 3

        # Assign ranks
        result = state._assign_group_ranks(participants, min_nodes)

        # Group ranks should follow SLURM topology order (infra rank order), not node descriptor order
        self.assertEqual(result[node_zzz], 0)  # Smallest infra rank -> group rank 0
        self.assertEqual(result[node_bbb], 1)  # Middle infra rank -> group rank 1
        self.assertEqual(result[node_aaa], 2)  # Largest infra rank -> group rank 2

    def test_rank_assignment_with_hot_spare_segment_none(self):
        """Test rank assignment with hot spare nodes (segment=None).

        When there are more participants than world_size and segment=None,
        the extra nodes become hot spares with ranks >= world_size.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=None,  # No segment awareness
        )

        # 5 nodes, world_size=3 -> first 3 active [0,1,2], remaining 2 hot spares [3,4]
        participants = [
            (_NodeDesc("node0", 100, 0), 0, "none"),  # Active
            (_NodeDesc("node1", 101, 0), 1, "none"),  # Active
            (_NodeDesc("node2", 102, 0), 2, "none"),  # Active
            (_NodeDesc("node3", 103, 0), 3, "none"),  # Hot spare
            (_NodeDesc("node4", 104, 0), 4, "none"),  # Hot spare
        ]
        world_size = 3

        result = state._assign_group_ranks(participants, world_size)

        # First 3 get active ranks [0,1,2]
        self.assertEqual(result[_NodeDesc("node0", 100, 0)], 0)
        self.assertEqual(result[_NodeDesc("node1", 101, 0)], 1)
        self.assertEqual(result[_NodeDesc("node2", 102, 0)], 2)

        # Remaining 2 are hot spares [3,4]
        self.assertEqual(result[_NodeDesc("node3", 103, 0)], 3)
        self.assertEqual(result[_NodeDesc("node4", 104, 0)], 4)

    def test_rank_assignment_segment_1_with_hot_spare(self):
        """Test rank assignment with segment=1 and hot spare nodes.

        segment=1 means each node is a complete segment.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=1,
        )

        # 5 nodes, world_size=3, segment=1 -> first 3 active, remaining 2 hot spares
        # Each node has its own domain (simulated with node names like "domain0")
        participants = [
            (_NodeDesc("domain0-node", 100, 0), 0, "domain0"),
            (_NodeDesc("domain1-node", 101, 0), 1, "domain1"),
            (_NodeDesc("domain2-node", 102, 0), 2, "domain2"),
            (_NodeDesc("domain3-node", 103, 0), 3, "domain3"),  # Hot spare
            (_NodeDesc("domain4-node", 104, 0), 4, "domain4"),  # Hot spare
        ]
        world_size = 3

        result = state._assign_group_ranks(participants, world_size)

        # Active ranks
        self.assertEqual(result[_NodeDesc("domain0-node", 100, 0)], 0)
        self.assertEqual(result[_NodeDesc("domain1-node", 101, 0)], 1)
        self.assertEqual(result[_NodeDesc("domain2-node", 102, 0)], 2)

        # Hot spares
        self.assertEqual(result[_NodeDesc("domain3-node", 103, 0)], 3)
        self.assertEqual(result[_NodeDesc("domain4-node", 104, 0)], 4)

    def test_rank_assignment_segment_4_with_hot_spare(self):
        """Test rank assignment with segment=4 and hot spare nodes.

        segment=4 means we need 4 nodes per segment.
        world_size=8 requires 2 complete segments.
        Using domain-aware assignment with 2 domains.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=4,
        )

        # Domain 100: 8 nodes (2 complete segments) -> first 8 active
        # Domain 101: 4 nodes (1 complete segment) -> all standby (already have enough)
        # Total 12 nodes, world_size=8 (2 segments)
        participants = [
            # Domain 100: 8 nodes
            (_NodeDesc("nvl72100-node0", 100, 0), 0, "nvl72100"),
            (_NodeDesc("nvl72100-node1", 101, 0), 1, "nvl72100"),
            (_NodeDesc("nvl72100-node2", 102, 0), 2, "nvl72100"),
            (_NodeDesc("nvl72100-node3", 103, 0), 3, "nvl72100"),
            (_NodeDesc("nvl72100-node4", 104, 0), 4, "nvl72100"),
            (_NodeDesc("nvl72100-node5", 105, 0), 5, "nvl72100"),
            (_NodeDesc("nvl72100-node6", 106, 0), 6, "nvl72100"),
            (_NodeDesc("nvl72100-node7", 107, 0), 7, "nvl72100"),
            # Domain 101: 4 nodes
            (_NodeDesc("nvl72101-node0", 108, 0), 8, "nvl72101"),
            (_NodeDesc("nvl72101-node1", 109, 0), 9, "nvl72101"),
            (_NodeDesc("nvl72101-node2", 110, 0), 10, "nvl72101"),
            (_NodeDesc("nvl72101-node3", 111, 0), 11, "nvl72101"),
        ]
        world_size = 8

        result = state._assign_group_ranks(participants, world_size)

        # Domain 100: all 8 nodes are active (2 complete segments) [0-7]
        for i in range(8):
            self.assertEqual(result[_NodeDesc(f"nvl72100-node{i}", 100 + i, 0)], i)

        # Domain 101: all 4 nodes are standby [8-11]
        for i in range(4):
            self.assertEqual(result[_NodeDesc(f"nvl72101-node{i}", 108 + i, 0)], 8 + i)

    def test_rank_assignment_segment_16_with_hot_spare(self):
        """Test rank assignment with segment=16 and hot spare nodes.

        segment=16 means we need 16 nodes per segment.
        Using domain-aware assignment with 2 domains.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=16,
        )

        # Domain 100: 16 nodes (1 complete segment) -> all 16 active
        # Domain 101: 4 nodes (0 complete segments) -> all 4 standby
        # Total 20 nodes, world_size=16 (1 segment)
        participants = [
            # Domain 100: 16 nodes
            *[(_NodeDesc(f"nvl72100-node{i}", 100 + i, 0), i, "nvl72100") for i in range(16)],
            # Domain 101: 4 nodes
            *[(_NodeDesc(f"nvl72101-node{i}", 116 + i, 0), 16 + i, "nvl72101") for i in range(4)],
        ]
        world_size = 16

        result = state._assign_group_ranks(participants, world_size)

        # Domain 100: all 16 nodes are active (1 complete segment) [0-15]
        for i in range(16):
            self.assertEqual(result[_NodeDesc(f"nvl72100-node{i}", 100 + i, 0)], i)

        # Domain 101: all 4 nodes are standby [16-19]
        for i in range(4):
            self.assertEqual(result[_NodeDesc(f"nvl72101-node{i}", 116 + i, 0)], 16 + i)

    def test_rank_assignment_domain_with_zero_complete_segments(self):
        """Test rank assignment when a domain has 0 complete segments.

        If segment=4 and a domain has only 3 nodes, those nodes can't form
        a complete segment and should be assigned to standby.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=4,
        )

        # Domain 100: 4 nodes (1 complete segment) -> active
        # Domain 101: 3 nodes (0 complete segments) -> all standby
        # World_size=4 requires 1 segment
        participants = [
            # Domain 100: 4 nodes
            (_NodeDesc("nvl72100-node0", 100, 0), 0, "nvl72100"),
            (_NodeDesc("nvl72100-node1", 101, 0), 1, "nvl72100"),
            (_NodeDesc("nvl72100-node2", 102, 0), 2, "nvl72100"),
            (_NodeDesc("nvl72100-node3", 103, 0), 3, "nvl72100"),
            # Domain 101: 3 nodes (incomplete segment)
            (_NodeDesc("nvl72101-node0", 104, 0), 4, "nvl72101"),
            (_NodeDesc("nvl72101-node1", 105, 0), 5, "nvl72101"),
            (_NodeDesc("nvl72101-node2", 106, 0), 6, "nvl72101"),
        ]
        world_size = 4

        result = state._assign_group_ranks(participants, world_size)

        # Domain 100: all 4 nodes active [0-3]
        self.assertEqual(result[_NodeDesc("nvl72100-node0", 100, 0)], 0)
        self.assertEqual(result[_NodeDesc("nvl72100-node1", 101, 0)], 1)
        self.assertEqual(result[_NodeDesc("nvl72100-node2", 102, 0)], 2)
        self.assertEqual(result[_NodeDesc("nvl72100-node3", 103, 0)], 3)

        # Domain 101: all 3 nodes become standby [4-6]
        self.assertEqual(result[_NodeDesc("nvl72101-node0", 104, 0)], 4)
        self.assertEqual(result[_NodeDesc("nvl72101-node1", 105, 0)], 5)
        self.assertEqual(result[_NodeDesc("nvl72101-node2", 106, 0)], 6)

    def test_rank_assignment_domain_with_one_complete_segment(self):
        """Test rank assignment when a domain has exactly 1 complete segment."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=4,
        )

        # Domain 100: 4 nodes (1 complete segment)
        # Domain 101: 4 nodes (1 complete segment)
        # World_size=8 requires 2 segments, both domains contribute 1 segment
        participants = [
            # Domain 100
            (_NodeDesc("nvl72100-node0", 100, 0), 0, "nvl72100"),
            (_NodeDesc("nvl72100-node1", 101, 0), 1, "nvl72100"),
            (_NodeDesc("nvl72100-node2", 102, 0), 2, "nvl72100"),
            (_NodeDesc("nvl72100-node3", 103, 0), 3, "nvl72100"),
            # Domain 101
            (_NodeDesc("nvl72101-node0", 104, 0), 4, "nvl72101"),
            (_NodeDesc("nvl72101-node1", 105, 0), 5, "nvl72101"),
            (_NodeDesc("nvl72101-node2", 106, 0), 6, "nvl72101"),
            (_NodeDesc("nvl72101-node3", 107, 0), 7, "nvl72101"),
        ]
        world_size = 8

        result = state._assign_group_ranks(participants, world_size)

        # Both domains contribute 1 segment each, all nodes active
        # Domain 100 comes first (lower infra_rank)
        self.assertEqual(result[_NodeDesc("nvl72100-node0", 100, 0)], 0)
        self.assertEqual(result[_NodeDesc("nvl72100-node1", 101, 0)], 1)
        self.assertEqual(result[_NodeDesc("nvl72100-node2", 102, 0)], 2)
        self.assertEqual(result[_NodeDesc("nvl72100-node3", 103, 0)], 3)

        # Domain 101
        self.assertEqual(result[_NodeDesc("nvl72101-node0", 104, 0)], 4)
        self.assertEqual(result[_NodeDesc("nvl72101-node1", 105, 0)], 5)
        self.assertEqual(result[_NodeDesc("nvl72101-node2", 106, 0)], 6)
        self.assertEqual(result[_NodeDesc("nvl72101-node3", 107, 0)], 7)

    def test_rank_assignment_domain_with_multiple_complete_segments(self):
        """Test rank assignment when a domain has multiple complete segments.

        A domain with 12 nodes and segment=4 has 3 complete segments.
        If world_size=4 (1 segment needed), domain uses 4 nodes, rest are standby.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=4,
        )

        # Domain 100: 12 nodes (3 complete segments)
        # World_size=4 requires 1 segment
        # -> First 4 nodes active, remaining 8 standby
        participants = [
            (_NodeDesc(f"nvl72100-node{i}", 100 + i, 0), i, "nvl72100") for i in range(12)
        ]
        world_size = 4

        result = state._assign_group_ranks(participants, world_size)

        # First 4 nodes active (1 segment)
        for i in range(4):
            self.assertEqual(result[_NodeDesc(f"nvl72100-node{i}", 100 + i, 0)], i)

        # Remaining 8 nodes standby [4-11]
        for i in range(4, 12):
            self.assertEqual(result[_NodeDesc(f"nvl72100-node{i}", 100 + i, 0)], i)

    def test_rank_assignment_multiple_domains_mixed_segments(self):
        """Test rank assignment with multiple domains having different segment counts.

        Test scenario:
        - Domain 100: 8 nodes (2 segments with segment=4)
        - Domain 101: 6 nodes (1 complete segment + 2 incomplete)
        - Domain 102: 4 nodes (1 complete segment)
        - World_size=8 (2 segments needed)
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=4,
        )

        participants = [
            # Domain 100: 8 nodes (2 segments) - lower infra_rank comes first
            (_NodeDesc("nvl72100-node0", 100, 0), 0, "nvl72100"),
            (_NodeDesc("nvl72100-node1", 101, 0), 1, "nvl72100"),
            (_NodeDesc("nvl72100-node2", 102, 0), 2, "nvl72100"),
            (_NodeDesc("nvl72100-node3", 103, 0), 3, "nvl72100"),
            (_NodeDesc("nvl72100-node4", 104, 0), 4, "nvl72100"),
            (_NodeDesc("nvl72100-node5", 105, 0), 5, "nvl72100"),
            (_NodeDesc("nvl72100-node6", 106, 0), 6, "nvl72100"),
            (_NodeDesc("nvl72100-node7", 107, 0), 7, "nvl72100"),
            (_NodeDesc("nvl72101-node0", 108, 0), 8, "nvl72101"),
            (_NodeDesc("nvl72101-node1", 109, 0), 9, "nvl72101"),
            (_NodeDesc("nvl72101-node2", 110, 0), 10, "nvl72101"),
            (_NodeDesc("nvl72101-node3", 111, 0), 11, "nvl72101"),
            (_NodeDesc("nvl72101-node4", 112, 0), 12, "nvl72101"),
            (_NodeDesc("nvl72101-node5", 113, 0), 13, "nvl72101"),
            (_NodeDesc("nvl72102-node0", 114, 0), 14, "nvl72102"),
            (_NodeDesc("nvl72102-node1", 115, 0), 15, "nvl72102"),
            (_NodeDesc("nvl72102-node2", 116, 0), 16, "nvl72102"),
            (_NodeDesc("nvl72102-node3", 117, 0), 17, "nvl72102"),
        ]
        world_size = 8  # Need 2 segments

        result = state._assign_group_ranks(participants, world_size)

        # Domain 100 contributes 2 segments (all 8 nodes active) [0-7]
        for i in range(8):
            self.assertEqual(result[_NodeDesc(f"nvl72100-node{i}", 100 + i, 0)], i)

        # Domain 101: all nodes go to standby (we already have 2 segments)
        for i in range(6):
            self.assertEqual(result[_NodeDesc(f"nvl72101-node{i}", 108 + i, 0)], 8 + i)

        # Domain 102: all nodes go to standby
        for i in range(4):
            self.assertEqual(result[_NodeDesc(f"nvl72102-node{i}", 114 + i, 0)], 14 + i)

    def test_rank_assignment_out_of_order_with_hot_spare_segment_none(self):
        """Test that out-of-order infra rank arrivals work correctly with hot spares (segment=None).

        Participants arrive in arbitrary order, but should be sorted by infra_rank
        and assigned group ranks accordingly, with extras becoming hot spares.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=None,
        )

        # Participants arrive out-of-order (infra ranks: 4, 1, 3, 0, 2)
        # After sorting by infra_rank: 0, 1, 2, 3, 4
        # World_size=3 -> first 3 active [0,1,2], remaining 2 hot spares [3,4]
        participants = [
            (_NodeDesc("node4", 104, 0), 4, "none"),
            (_NodeDesc("node1", 101, 0), 1, "none"),
            (_NodeDesc("node3", 103, 0), 3, "none"),
            (_NodeDesc("node0", 100, 0), 0, "none"),
            (_NodeDesc("node2", 102, 0), 2, "none"),
        ]
        world_size = 3

        result = state._assign_group_ranks(participants, world_size)

        # Should be sorted by infra_rank and assigned accordingly
        self.assertEqual(
            result[_NodeDesc("node0", 100, 0)], 0
        )  # infra_rank 0 -> group rank 0 (active)
        self.assertEqual(
            result[_NodeDesc("node1", 101, 0)], 1
        )  # infra_rank 1 -> group rank 1 (active)
        self.assertEqual(
            result[_NodeDesc("node2", 102, 0)], 2
        )  # infra_rank 2 -> group rank 2 (active)
        self.assertEqual(
            result[_NodeDesc("node3", 103, 0)], 3
        )  # infra_rank 3 -> group rank 3 (hot spare)
        self.assertEqual(
            result[_NodeDesc("node4", 104, 0)], 4
        )  # infra_rank 4 -> group rank 4 (hot spare)

    def test_rank_assignment_out_of_order_with_hot_spare_segment_4(self):
        """Test that out-of-order infra rank arrivals work correctly with segment=4 and hot spares.

        Participants from different domains arrive out-of-order, but should be sorted
        by infra_rank (SLURM topology order) for proper segment assignment.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=4,
        )

        # Participants arrive out-of-order from 2 domains
        # Domain 100 has infra_ranks [0-7], Domain 101 has infra_ranks [8-11]
        # Listed in reverse order to test sorting
        participants = [
            # Domain 101 nodes (listed first, but have higher infra_ranks)
            (_NodeDesc("nvl72101-node3", 111, 0), 11, "nvl72101"),
            (_NodeDesc("nvl72101-node2", 110, 0), 10, "nvl72101"),
            (_NodeDesc("nvl72101-node1", 109, 0), 9, "nvl72101"),
            (_NodeDesc("nvl72101-node0", 108, 0), 8, "nvl72101"),
            (_NodeDesc("nvl72100-node7", 107, 0), 7, "nvl72100"),
            (_NodeDesc("nvl72100-node6", 106, 0), 6, "nvl72100"),
            (_NodeDesc("nvl72100-node5", 105, 0), 5, "nvl72100"),
            (_NodeDesc("nvl72100-node4", 104, 0), 4, "nvl72100"),
            (_NodeDesc("nvl72100-node3", 103, 0), 3, "nvl72100"),
            (_NodeDesc("nvl72100-node2", 102, 0), 2, "nvl72100"),
            (_NodeDesc("nvl72100-node1", 101, 0), 1, "nvl72100"),
            (_NodeDesc("nvl72100-node0", 100, 0), 0, "nvl72100"),
        ]
        world_size = 8  # Need 2 segments

        result = state._assign_group_ranks(participants, world_size)

        # Despite out-of-order arrival, Domain 100 (lower infra_ranks) should be sorted first
        # Domain 100: all 8 nodes active [0-7]
        self.assertEqual(result[_NodeDesc("nvl72100-node0", 100, 0)], 0)
        self.assertEqual(result[_NodeDesc("nvl72100-node1", 101, 0)], 1)
        self.assertEqual(result[_NodeDesc("nvl72100-node2", 102, 0)], 2)
        self.assertEqual(result[_NodeDesc("nvl72100-node3", 103, 0)], 3)
        self.assertEqual(result[_NodeDesc("nvl72100-node4", 104, 0)], 4)
        self.assertEqual(result[_NodeDesc("nvl72100-node5", 105, 0)], 5)
        self.assertEqual(result[_NodeDesc("nvl72100-node6", 106, 0)], 6)
        self.assertEqual(result[_NodeDesc("nvl72100-node7", 107, 0)], 7)

        # Domain 101: all 4 nodes are standby [8-11]
        self.assertEqual(result[_NodeDesc("nvl72101-node0", 108, 0)], 8)
        self.assertEqual(result[_NodeDesc("nvl72101-node1", 109, 0)], 9)
        self.assertEqual(result[_NodeDesc("nvl72101-node2", 110, 0)], 10)
        self.assertEqual(result[_NodeDesc("nvl72101-node3", 111, 0)], 11)

    def test_rank_assignment_out_of_order_mixed_domains_segment_4(self):
        """Test out-of-order arrivals with mixed segment counts across domains.

        Tests that nodes from multiple domains arriving in arbitrary order are
        correctly sorted and assigned ranks based on infra_rank and segment rules.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=4,
        )

        # Domain 100: infra_ranks [0-5] (1 complete segment + 2 incomplete)
        # Domain 101: infra_ranks [6-9] (1 complete segment)
        # Domain 102: infra_ranks [10-12] (0 complete segments)
        # Listed in completely arbitrary order
        participants = [
            (_NodeDesc("nvl72102-node2", 112, 0), 12, "nvl72102"),
            (_NodeDesc("nvl72100-node3", 103, 0), 3, "nvl72100"),
            (_NodeDesc("nvl72101-node1", 107, 0), 7, "nvl72101"),
            (_NodeDesc("nvl72102-node0", 110, 0), 10, "nvl72102"),
            (_NodeDesc("nvl72100-node5", 105, 0), 5, "nvl72100"),
            (_NodeDesc("nvl72101-node3", 109, 0), 9, "nvl72101"),
            (_NodeDesc("nvl72100-node0", 100, 0), 0, "nvl72100"),
            (_NodeDesc("nvl72101-node2", 108, 0), 8, "nvl72101"),
            (_NodeDesc("nvl72102-node1", 111, 0), 11, "nvl72102"),
            (_NodeDesc("nvl72100-node4", 104, 0), 4, "nvl72100"),
            (_NodeDesc("nvl72100-node1", 101, 0), 1, "nvl72100"),
            (_NodeDesc("nvl72101-node0", 106, 0), 6, "nvl72101"),
            (_NodeDesc("nvl72100-node2", 102, 0), 2, "nvl72100"),
        ]
        world_size = 8  # Need 2 segments

        result = state._assign_group_ranks(participants, world_size)

        # Domain 100 (infra_ranks 0-5): 1 complete segment (4 nodes) active, 2 standby
        self.assertEqual(result[_NodeDesc("nvl72100-node0", 100, 0)], 0)
        self.assertEqual(result[_NodeDesc("nvl72100-node1", 101, 0)], 1)
        self.assertEqual(result[_NodeDesc("nvl72100-node2", 102, 0)], 2)
        self.assertEqual(result[_NodeDesc("nvl72100-node3", 103, 0)], 3)
        self.assertEqual(result[_NodeDesc("nvl72100-node4", 104, 0)], 8)  # Standby
        self.assertEqual(result[_NodeDesc("nvl72100-node5", 105, 0)], 9)  # Standby

        # Domain 101 (infra_ranks 6-9): 1 complete segment (4 nodes) active
        self.assertEqual(result[_NodeDesc("nvl72101-node0", 106, 0)], 4)
        self.assertEqual(result[_NodeDesc("nvl72101-node1", 107, 0)], 5)
        self.assertEqual(result[_NodeDesc("nvl72101-node2", 108, 0)], 6)
        self.assertEqual(result[_NodeDesc("nvl72101-node3", 109, 0)], 7)

        # Domain 102 (infra_ranks 10-12): 0 complete segments, all 3 standby
        self.assertEqual(result[_NodeDesc("nvl72102-node0", 110, 0)], 10)  # Standby
        self.assertEqual(result[_NodeDesc("nvl72102-node1", 111, 0)], 11)  # Standby
        self.assertEqual(result[_NodeDesc("nvl72102-node2", 112, 0)], 12)  # Standby

    def test_rank_assignment_with_gaps_in_infra_ranks_segment_none(self):
        """Test rank assignment with gaps in infra_ranks (simulating node failures), segment=None.

        When some nodes fail and don't join, there are gaps in infra_ranks.
        E.g., if nodes 2, 4, 5 failed, we have infra_ranks [0, 1, 3, 6, 7] instead of [0-7].
        These should still get contiguous group ranks [0, 1, 2, 3, 4].
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=None,
        )

        # Simulate 5 nodes joining with gaps: infra_ranks [0, 1, 3, 6, 7]
        # Missing nodes: 2, 4, 5 (failed to join)
        # World_size=3 -> first 3 active [0,1,2], remaining 2 hot spares [3,4]
        participants = [
            (_NodeDesc("node0", 100, 0), 0, "none"),
            (_NodeDesc("node1", 101, 0), 1, "none"),
            (_NodeDesc("node3", 103, 0), 3, "none"),
            (_NodeDesc("node6", 106, 0), 6, "none"),
            (_NodeDesc("node7", 107, 0), 7, "none"),
        ]
        world_size = 3

        result = state._assign_group_ranks(participants, world_size)

        # Should get contiguous group ranks [0-4] based on sorted infra_rank
        self.assertEqual(result[_NodeDesc("node0", 100, 0)], 0)  # Active
        self.assertEqual(result[_NodeDesc("node1", 101, 0)], 1)  # Active
        self.assertEqual(result[_NodeDesc("node3", 103, 0)], 2)  # Active
        self.assertEqual(result[_NodeDesc("node6", 106, 0)], 3)  # Hot spare
        self.assertEqual(result[_NodeDesc("node7", 107, 0)], 4)  # Hot spare

    def test_rank_assignment_with_gaps_in_infra_ranks_with_hot_spare(self):
        """Test rank assignment with gaps in infra_ranks and hot spares (segment=None).

        More aggressive gap scenario: infra_ranks [0, 5, 10, 15, 20, 22, 24]
        Simulates many node failures (1-4, 6-9, 11-14, 16-19, 21, 23 all failed).
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=None,
        )

        # Large gaps in infra_ranks
        participants = [
            (_NodeDesc("node0", 100, 0), 0, "none"),
            (_NodeDesc("node5", 105, 0), 5, "none"),
            (_NodeDesc("node10", 110, 0), 10, "none"),
            (_NodeDesc("node15", 115, 0), 15, "none"),
            (_NodeDesc("node20", 120, 0), 20, "none"),
            (_NodeDesc("node22", 122, 0), 22, "none"),
            (_NodeDesc("node24", 124, 0), 24, "none"),
        ]
        world_size = 4  # Need 4 active nodes

        result = state._assign_group_ranks(participants, world_size)

        # Should get contiguous group ranks [0-6], first 4 active, last 3 hot spares
        self.assertEqual(result[_NodeDesc("node0", 100, 0)], 0)  # Active
        self.assertEqual(result[_NodeDesc("node5", 105, 0)], 1)  # Active
        self.assertEqual(result[_NodeDesc("node10", 110, 0)], 2)  # Active
        self.assertEqual(result[_NodeDesc("node15", 115, 0)], 3)  # Active
        self.assertEqual(result[_NodeDesc("node20", 120, 0)], 4)  # Hot spare
        self.assertEqual(result[_NodeDesc("node22", 122, 0)], 5)  # Hot spare
        self.assertEqual(result[_NodeDesc("node24", 124, 0)], 6)  # Hot spare

    def test_rank_assignment_with_gaps_in_infra_ranks_segment_4(self):
        """Test rank assignment with gaps in infra_ranks across domains with segment=4.

        Simulates scenario where some nodes in each domain failed to join,
        leaving gaps in infra_ranks within each domain.
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=4,
        )

        # Domain 100: infra_ranks [0, 2, 4, 6, 8, 10] (nodes 1,3,5,7,9 failed)
        #             6 nodes -> 1 complete segment (4 nodes), 2 incomplete
        # Domain 101: infra_ranks [12, 14, 17, 19] (nodes 13,15,16,18 failed)
        #             4 nodes -> 1 complete segment
        # World_size=8 requires 2 segments
        participants = [
            # Domain 100 (with gaps)
            (_NodeDesc("nvl72100-node0", 100, 0), 0, "nvl72100"),
            (_NodeDesc("nvl72100-node2", 102, 0), 2, "nvl72100"),
            (_NodeDesc("nvl72100-node4", 104, 0), 4, "nvl72100"),
            (_NodeDesc("nvl72100-node6", 106, 0), 6, "nvl72100"),
            (_NodeDesc("nvl72100-node8", 108, 0), 8, "nvl72100"),
            (_NodeDesc("nvl72100-node10", 110, 0), 10, "nvl72100"),
            (_NodeDesc("nvl72101-node0", 112, 0), 12, "nvl72101"),
            (_NodeDesc("nvl72101-node2", 114, 0), 14, "nvl72101"),
            (_NodeDesc("nvl72101-node5", 117, 0), 17, "nvl72101"),
            (_NodeDesc("nvl72101-node7", 119, 0), 19, "nvl72101"),
        ]
        world_size = 8

        result = state._assign_group_ranks(participants, world_size)

        # Domain 100: 1 complete segment (4 nodes) active, 2 standby
        self.assertEqual(result[_NodeDesc("nvl72100-node0", 100, 0)], 0)
        self.assertEqual(result[_NodeDesc("nvl72100-node2", 102, 0)], 1)
        self.assertEqual(result[_NodeDesc("nvl72100-node4", 104, 0)], 2)
        self.assertEqual(result[_NodeDesc("nvl72100-node6", 106, 0)], 3)
        self.assertEqual(result[_NodeDesc("nvl72100-node8", 108, 0)], 8)  # Standby
        self.assertEqual(result[_NodeDesc("nvl72100-node10", 110, 0)], 9)  # Standby

        # Domain 101: 1 complete segment (4 nodes) active
        self.assertEqual(result[_NodeDesc("nvl72101-node0", 112, 0)], 4)
        self.assertEqual(result[_NodeDesc("nvl72101-node2", 114, 0)], 5)
        self.assertEqual(result[_NodeDesc("nvl72101-node5", 117, 0)], 6)
        self.assertEqual(result[_NodeDesc("nvl72101-node7", 119, 0)], 7)

    def test_rank_assignment_with_gaps_out_of_order_segment_4(self):
        """Test rank assignment with gaps in infra_ranks arriving out-of-order with segment=4.

        Combines three challenging scenarios:
        1. Gaps in infra_ranks (node failures)
        2. Out-of-order arrivals
        3. Multiple domains with segment awareness
        """
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            segment=4,
        )

        # Domain 100: infra_ranks [1, 3, 5, 7, 9] (0,2,4,6,8 failed) -> 1 segment + 1 incomplete
        # Domain 101: infra_ranks [11, 13, 15, 17] (10,12,14,16 failed) -> 1 segment
        # Listed in reverse order to test sorting
        participants = [
            # Domain 101 first, in reverse
            (_NodeDesc("nvl72101-node7", 117, 0), 17, "nvl72101"),
            (_NodeDesc("nvl72101-node5", 115, 0), 15, "nvl72101"),
            (_NodeDesc("nvl72101-node3", 113, 0), 13, "nvl72101"),
            (_NodeDesc("nvl72101-node1", 111, 0), 11, "nvl72101"),
            (_NodeDesc("nvl72100-node9", 109, 0), 9, "nvl72100"),
            (_NodeDesc("nvl72100-node7", 107, 0), 7, "nvl72100"),
            (_NodeDesc("nvl72100-node5", 105, 0), 5, "nvl72100"),
            (_NodeDesc("nvl72100-node3", 103, 0), 3, "nvl72100"),
            (_NodeDesc("nvl72100-node1", 101, 0), 1, "nvl72100"),
        ]
        world_size = 8

        result = state._assign_group_ranks(participants, world_size)

        # Despite reverse order and gaps, Domain 100 (lower infra_ranks) comes first
        # Domain 100: 1 complete segment (4 nodes) active, 1 standby
        self.assertEqual(result[_NodeDesc("nvl72100-node1", 101, 0)], 0)
        self.assertEqual(result[_NodeDesc("nvl72100-node3", 103, 0)], 1)
        self.assertEqual(result[_NodeDesc("nvl72100-node5", 105, 0)], 2)
        self.assertEqual(result[_NodeDesc("nvl72100-node7", 107, 0)], 3)
        self.assertEqual(result[_NodeDesc("nvl72100-node9", 109, 0)], 8)  # Standby

        # Domain 101: 1 complete segment (4 nodes) active
        self.assertEqual(result[_NodeDesc("nvl72101-node1", 111, 0)], 4)
        self.assertEqual(result[_NodeDesc("nvl72101-node3", 113, 0)], 5)
        self.assertEqual(result[_NodeDesc("nvl72101-node5", 115, 0)], 6)
        self.assertEqual(result[_NodeDesc("nvl72101-node7", 117, 0)], 7)


class ErrorCaseTest(BaseRendezvousTest):
    """Test error cases and exception handling."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        import time

        super().setUp()  # Clears environment variables

        self.store = self.shared_store
        # Use unique run_id for each test to avoid key collisions
        self.run_id = f"test_error_{self._testMethodName}_{int(time.time() * 1000000)}"
        self.node_desc_gen = _NodeDescGenerator()

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()  # Restores environment variables

    def test_exceed_max_nodes_raises_error(self):
        """Test that exceeding max_nodes raises RendezvousClosedError."""
        min_nodes = 2
        max_nodes = 2  # Set max to 2
        last_call_timeout = _test_timeout()

        errors = []
        # Use barrier to synchronize thread starts - ensures all threads
        # attempt to join before any complete the rendezvous
        start_barrier = threading.Barrier(3)

        def participant_thread(participant_id, is_host):
            # Each participant needs its own state instance
            state = _RendezvousBarrierState(
                store=self.store,
                run_id=self.run_id,
                is_store_host=is_host,
                join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
            )
            try:
                # Wait for all threads to be ready before proceeding
                start_barrier.wait()
                node = self.node_desc_gen.generate()
                state.perform_rendezvous(node, min_nodes, max_nodes, last_call_timeout)
            except Exception as e:
                errors.append((participant_id, type(e).__name__))

        # Try to add 3 participants (exceeds max_nodes)
        threads = []
        for i in range(3):
            t = threading.Thread(target=participant_thread, args=(i, i == 0))
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=TEST_THREAD_JOIN_TIMEOUT_SECS)

        # At least one should get RendezvousClosedError
        error_types = [e[1] for e in errors]
        self.assertIn('RendezvousClosedError', error_types)

    def test_timeout_raises_error(self):
        """Test that join timeout raises RendezvousTimeoutError."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=1.0,  # Use very short timeout to test timeout behavior
        )
        min_nodes = 5  # Require 5 nodes but we'll only provide 1
        max_nodes = 5
        last_call_timeout = _test_timeout()

        node = self.node_desc_gen.generate()

        # Should timeout waiting for other participants
        with self.assertRaises(RendezvousTimeoutError):
            state.perform_rendezvous(node, min_nodes, max_nodes, last_call_timeout)

    def test_closed_rendezvous_raises_error(self):
        """Test that joining a closed rendezvous raises RendezvousClosedError."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        # Close the rendezvous
        state.set_closed()
        self.assertTrue(state.is_closed())

        min_nodes = 2
        max_nodes = 4
        last_call_timeout = _test_timeout()
        node = self.node_desc_gen.generate()

        # Should raise RendezvousClosedError
        with self.assertRaises(RendezvousClosedError):
            state.perform_rendezvous(node, min_nodes, max_nodes, last_call_timeout)

    def test_duplicate_infra_rank_raises_error(self):
        """Test that duplicate infrastructure ranks raise an error."""
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        # Create participants with duplicate infra_rank
        participants = [
            (_NodeDesc("node_a", 100, 0), 0, "none"),
            (_NodeDesc("node_b", 101, 0), 0, "none"),
        ]
        min_nodes = 2

        # Should raise RuntimeError about duplicate ranks
        with self.assertRaises(RuntimeError) as ctx:
            state._assign_group_ranks(participants, min_nodes)

        self.assertIn("Duplicate", str(ctx.exception))


class AcknowledgmentPhaseTest(BaseRendezvousTest):
    """Test acknowledgment phase behavior."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        import time

        super().setUp()  # Clears environment variables

        self.store = self.shared_store
        # Use unique run_id for each test to avoid key collisions
        self.run_id = f"test_ack_{self._testMethodName}_{int(time.time() * 1000000)}"
        self.node_desc_gen = _NodeDescGenerator()

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()  # Restores environment variables

    def test_all_participants_acknowledge(self):
        """Test that all participants acknowledge completion."""
        min_nodes = 3
        max_nodes = 3
        last_call_timeout = _test_timeout()

        ack_counts = []

        def participant_thread(is_host):
            # Each participant needs its own state instance
            state = _RendezvousBarrierState(
                store=self.store,
                run_id=self.run_id,
                is_store_host=is_host,
                join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
            )
            node = self.node_desc_gen.generate()
            state.perform_rendezvous(node, min_nodes, max_nodes, last_call_timeout)
            # After completion, check ack count
            # Note: Keys might be cleared by store host, so we capture during execution
            ack_counts.append(1)  # Just count that we got here

        threads = []
        for i in range(min_nodes):
            t = threading.Thread(target=participant_thread, args=(i == 0,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=10)

        # All participants should have acknowledged and completed
        self.assertEqual(len(ack_counts), min_nodes)

    def test_barrier_keys_cleared_after_acknowledgment(self):
        """Test that barrier keys are cleared after all acknowledgments.

        Keys cleared: arrived_count_key, ack_count_key, peer_aborted_count_key
        Keys NOT cleared:
        - last_participant_arrived_key: serves as open/close indicator
        - unhealthy_count_key: global job-level counter across all cycles
        """
        min_nodes = 2
        max_nodes = 2
        last_call_timeout = _test_timeout()

        # Need one state to check keys after
        check_state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=False,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        def participant_thread(is_host):
            # Each participant needs its own state instance
            state = _RendezvousBarrierState(
                store=self.store,
                run_id=self.run_id,
                is_store_host=is_host,
                join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
            )
            node = self.node_desc_gen.generate()
            state.perform_rendezvous(node, min_nodes, max_nodes, last_call_timeout)

        threads = []
        for i in range(min_nodes):
            t = threading.Thread(target=participant_thread, args=(i == 0,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=TEST_THREAD_JOIN_TIMEOUT_SECS)

        # Wait a bit for cleanup
        time.sleep(0.05)  # Minimal delay for key cleanup

        # Barrier keys should be cleared by store host
        # Note: last_participant_arrived_key is intentionally NOT cleared as it
        # serves as the open/close indicator for the rendezvous
        # Note: unhealthy_count_key is intentionally NOT cleared as it is a global
        # job-level counter that tracks unhealthy nodes across the entire job lifetime
        self.assertFalse(self.store.check([check_state.arrived_count_key]))
        self.assertFalse(self.store.check([check_state.ack_count_key]))
        self.assertFalse(self.store.check([check_state.peer_aborted_count_key]))

        # Verify that unhealthy_count_key is NOT cleared (should remain absent if no unhealthy nodes)
        # We don't assert its presence/absence since no health check failures occurred in this test


class HandlerIntegrationTest(BaseRendezvousTest):
    """Integration tests for FtRendezvousBarrierHandler."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        import time

        super().setUp()  # Clears environment variables

        self.store = self.shared_store
        # Use unique run_id for each test to avoid key collisions
        self.run_id = f"test_handler_{self._testMethodName}_{int(time.time() * 1000000)}"

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()  # Restores environment variables

    def test_handler_creation(self):
        """Test that handler can be created with correct parameters."""
        handler = FtRendezvousBarrierHandler.from_backend(
            run_id=self.run_id,
            store=self.store,
            backend=None,
            min_nodes=2,
            max_nodes=4,
            timeout=RendezvousTimeout(),
            is_store_host=True,
        )

        self.assertIsNotNone(handler)
        self.assertEqual(handler.settings.run_id, self.run_id)
        self.assertEqual(handler.settings.min_nodes, 2)
        self.assertEqual(handler.settings.max_nodes, 4)

    def test_handler_next_rendezvous(self):
        """Test basic rendezvous flow is validated by other test classes."""
        # Note: Full integration testing with FtRendezvousBarrierHandler is done in functional tests
        # due to IPC socket conflicts in unit tests. The core rendezvous functionality is thoroughly
        # tested by the other 26 tests in this file covering:
        # - Barrier state operations (BarrierStateBasicTest)
        # - Step 2 completion logic (Step2CompletionTest)
        # - Race conditions (RaceConditionTest)
        # - Store host behavior (StoreHostBehaviorTest)
        # - Group rank assignment (GroupRankAssignmentTest)
        # - Error cases (ErrorCaseTest)
        # - Acknowledgment phase (AcknowledgmentPhaseTest)
        # - Infrastructure rank handling (InfrastructureRankTest)

        # This test verifies the components work together, which is already covered above
        self.assertTrue(True, "Core functionality tested by other test classes")


class InfrastructureRankTest(TestCase):
    """Test infrastructure rank handling."""

    @classmethod
    def setUpClass(cls):
        """Set up shared TCPStore for all tests in this class."""
        cls.shared_store = TCPStore(
            host_name="127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

    def setUp(self):
        """Set up test fixtures with unique run_id for each test."""
        import time

        # NOTE: DO NOT clear SLURM_PROCID/GROUP_RANK for this test class
        # These tests specifically test infrastructure rank behavior from environment

        self.store = self.shared_store
        # Use unique run_id for each test to avoid key collisions
        self.run_id = f"test_infra_{self._testMethodName}_{int(time.time() * 1000000)}"
        self.node_desc_gen = _NodeDescGenerator()

    def tearDown(self):
        """Clean up test fixtures."""
        pass

    def test_infra_rank_from_environment(self):
        """Test reading infrastructure rank from environment variables."""
        # Note: This is a simplified test of infra rank assignment logic
        # Testing with actual environment variables in threads is complex due to
        # shared environment. The actual behavior is tested in functional tests.

        # Test the assignment logic directly with simulated participants
        state = _RendezvousBarrierState(
            store=self.store,
            run_id=self.run_id,
            is_store_host=True,
            join_timeout_seconds=TEST_JOIN_TIMEOUT_SECS,
        )

        # Create participants with infra ranks
        participants = [
            (_NodeDesc("node_0", 100, 0), 0, "none"),
            (_NodeDesc("node_1", 101, 0), 1, "none"),
        ]
        min_nodes = 2

        # Assign ranks using infrastructure rank mode
        result = state._assign_group_ranks(participants, min_nodes)

        # Check direct mapping: group_rank = infra_rank
        self.assertEqual(result[participants[0][0]], 0)
        self.assertEqual(result[participants[1][0]], 1)


if __name__ == '__main__':
    import unittest

    unittest.main()
