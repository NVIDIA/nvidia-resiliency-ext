# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic unit tests for ``_ProgressiveLogState`` (no LLM, no I/O).

Covers the bounded progressive-analysis state used by the start/end phases of
``NVRxLogAnalyzer``: rolling-window bounding, checkpoint accumulation, the
atomic snapshot, and the per-path lock / poller-close lifecycle. The canonical
dispatch harness (``run_log_analyzer_dispatch_four_cycles.py``) exercises the
same state end-to-end but depends on a live LLM; these tests pin the mechanics
without that flakiness.
"""

import asyncio
import importlib
import unittest

try:
    nvrx_logsage = importlib.import_module(
        "nvidia_resiliency_ext.attribution.log_analyzer.nvrx_logsage"
    )
    IMPORT_ERROR = None
except ImportError as exc:
    nvrx_logsage = None
    IMPORT_ERROR = exc


@unittest.skipIf(nvrx_logsage is None, f"missing optional dependency: {IMPORT_ERROR}")
class TestProgressiveLogState(unittest.IsolatedAsyncioTestCase):
    @property
    def _State(self):
        return nvrx_logsage._ProgressiveLogState

    def test_unused_state_is_falsy(self):
        state = self._State(3)
        self.assertEqual(len(state), 0)
        self.assertFalse(state)  # falsy until a poll lands (preserves old empty-list semantics)
        self.assertEqual(state.latest_offset, 0)
        self.assertFalse(state.checkpoint_seen)
        self.assertFalse(state.checkpoint_first_poll)
        self.assertFalse(state.closed)
        self.assertIsNone(state.poller_task)

    def test_window_floored_at_one(self):
        self.assertEqual(self._State(0).window, 1)
        self.assertEqual(self._State(-5).window, 1)
        self.assertEqual(self._State(4).window, 4)

    def test_reconcile_accumulates_and_becomes_truthy(self):
        state = self._State(3)
        state.reconcile(10, ["a\n"], False)
        state.reconcile(25, ["b\n", "c\n"], False)
        self.assertEqual(len(state), 2)
        self.assertTrue(state)
        self.assertEqual(state.latest_offset, 25)

    def test_recent_lines_bounded_to_window(self):
        # window=2 keeps only the last two polls' lines.
        state = self._State(2)
        state.reconcile(1, ["a\n"], False)
        state.reconcile(2, ["b\n"], False)
        state.reconcile(3, ["c\n"], False)
        offset, lines, _ = state.snapshot()
        self.assertEqual(offset, 3)
        self.assertEqual(lines, ["b\n", "c\n"])  # "a" evicted
        self.assertEqual(state.poll_count, 3)

    def test_checkpoint_first_poll_and_or_reduce_survive_eviction(self):
        # Checkpoint on the very first poll, then enough polls to evict it from
        # the rolling window. checkpoint_seen / checkpoint_first_poll must persist.
        state = self._State(2)
        state.reconcile(1, ["boot-ckpt\n"], True)
        state.reconcile(2, ["x\n"], False)
        state.reconcile(3, ["y\n"], False)
        offset, lines, checkpoint_seen = state.snapshot()
        self.assertEqual(lines, ["x\n", "y\n"])  # checkpoint line scrolled out of window
        self.assertTrue(checkpoint_seen)  # but the flag is preserved
        self.assertTrue(state.checkpoint_first_poll)

    def test_empty_polls_do_not_evict_window_content(self):
        # Regression: after the writer goes idle the start phase keeps polling and
        # gets empty reads. Those empty polls must NOT push real content (the last
        # log chunk, which carries the terminal error) out of the bounded window.
        state = self._State(2)
        state.reconcile(10, ["step 1\n"], False)
        state.reconcile(20, ["OOM error\n"], False)  # terminal chunk
        state.reconcile(20, [], False)  # idle poll (writer done)
        state.reconcile(20, [], False)  # idle poll
        state.reconcile(20, [], False)  # idle poll
        offset, lines, _ = state.snapshot()
        self.assertEqual(offset, 20)
        self.assertEqual(lines, ["step 1\n", "OOM error\n"])  # OOM retained, not evicted
        self.assertEqual(state.poll_count, 5)  # poll_count still counts every poll

    def test_only_empty_polls_leaves_window_empty(self):
        state = self._State(2)
        state.reconcile(0, [], False)
        state.reconcile(0, [], False)
        _, lines, _ = state.snapshot()
        self.assertEqual(lines, [])
        self.assertEqual(state.poll_count, 2)
        self.assertTrue(state)  # truthy via poll_count (preserves prior semantics)

    def test_checkpoint_not_on_first_poll(self):
        state = self._State(5)
        state.reconcile(1, ["x\n"], False)
        state.reconcile(2, ["late-ckpt\n"], True)
        self.assertTrue(state.checkpoint_seen)
        self.assertFalse(state.checkpoint_first_poll)  # not the first poll

    def test_snapshot_returns_consistent_triple(self):
        state = self._State(3)
        state.reconcile(7, ["l1\n"], False)
        state.reconcile(9, ["l2\n"], True)
        offset, lines, checkpoint_seen = state.snapshot()
        self.assertEqual(offset, 9)
        self.assertEqual(lines, ["l1\n", "l2\n"])
        self.assertTrue(checkpoint_seen)

    async def test_lock_guards_reconcile(self):
        state = self._State(5)
        async with state.lock():
            state.reconcile(1, ["a\n"], False)
        # Same lock instance is reused within one running loop.
        self.assertIs(state.lock(), state.lock())
        self.assertEqual(state.poll_count, 1)

    def test_lock_usable_across_separate_event_loops(self):
        # The dispatch harness drives the start and end phases in *separate*
        # asyncio.run loops. A lock bound to the first (now-closed) loop would
        # raise on reuse; lock() must rebind to the running loop instead.
        state = self._State(3)

        async def use():
            async with state.lock():
                state.reconcile(state.poll_count + 1, ["x\n"], False)

        asyncio.run(use())
        asyncio.run(use())  # would RuntimeError if the lock were bound to loop #1
        self.assertEqual(state.poll_count, 2)

    async def test_request_close_sets_flag_and_cancels_task(self):
        state = self._State(2)

        cancelled = asyncio.Event()

        async def poll():
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.ensure_future(poll())
        await asyncio.sleep(0)  # let the task start and reach the sleep
        state.poller_task = task

        async with state.lock():
            state.request_close()

        self.assertTrue(state.closed)
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(cancelled.is_set())
        self.assertTrue(task.cancelled())

    async def test_request_close_without_task_is_safe(self):
        state = self._State(2)
        async with state.lock():
            state.request_close()  # poller_task is None -> just sets the flag
        self.assertTrue(state.closed)

    async def test_final_attribution_wins_lock_then_poller_sees_closed(self):
        # Models the reviewer's invariant: while final attribution holds the
        # per-path lock (snapshot + close), a concurrent poller must block, and
        # once it acquires the lock it sees ``closed`` and bails WITHOUT
        # reconciling — so it cannot mutate state the finalizer already consumed.
        state = self._State(5)
        async with state.lock():
            state.reconcile(1, ["seed\n"], False)

        order = []

        async def final():
            async with state.lock():
                order.append("final-enter")
                await asyncio.sleep(0.05)  # hold the lock so the poller must wait
                state.request_close()
                order.append("final-exit")

        async def poller():
            await asyncio.sleep(0.01)  # ensure final acquires the lock first
            async with state.lock():
                if state.closed:
                    order.append("poller-saw-closed")
                    return
                state.reconcile(2, ["b\n"], False)
                order.append("poller-reconciled")

        await asyncio.gather(final(), poller())

        self.assertEqual(order, ["final-enter", "final-exit", "poller-saw-closed"])
        self.assertTrue(state.closed)
        self.assertEqual(state.poll_count, 1)  # poller did not reconcile after close


if __name__ == "__main__":
    unittest.main()
