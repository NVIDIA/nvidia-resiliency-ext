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

import multiprocessing
import random
import tempfile
import time
import unittest

import nvidia_resiliency_ext.inprocess as inprocess


class Store:
    def __init__(self, states, keys):
        self.states = states
        self.keys = keys

    def send_state(self, state, rank):
        pass

    def get_states(self, ranks):
        return [self.states[r] for r in ranks]

    def send_key(self, key, rank):
        pass

    def get_keys(self, ranks):
        return [self.keys[r] for r in ranks]


class TestTree(unittest.TestCase):
    def build(self, world_size, layers):
        states = {}
        for rank in range(world_size):
            states[rank] = inprocess.State(rank, world_size)

        keys = {}
        for rank, state in states.items():
            key = [
                (layer.key_or_fn(state) if callable(layer.key_or_fn) else layer.key_or_fn)
                for layer in layers
            ]
            keys[rank] = key

        store = Store(states, keys)
        rank_assignment = inprocess.rank_assignment.Tree(layers)
        return rank_assignment, states, store

    def compare_ranks(self, rank_assignment, reference):
        initial_world_size = len(reference)
        active_world_size = len([rank for rank in reference if rank is not None])

        test_ranks = [
            rank_assignment.init_rank_map[i].state.active_rank for i in range(initial_world_size)
        ]
        self.assertEqual(test_ranks, reference)
        test_active_world_size = [
            rank_assignment.init_rank_map[i].state.active_world_size
            for i in range(initial_world_size)
        ]
        self.assertEqual(test_active_world_size, [active_world_size] * initial_world_size)

    def perf(self, world_size, num_terminated, flag, timelimit):
        layers = [
            inprocess.rank_assignment.Layer(
                flag=flag,
            ),
            inprocess.rank_assignment.Layer(
                min_ranks=32,
                max_ranks=64,
                key_or_fn=lambda state: state.rank // 64,
                flag=flag,
            ),
            inprocess.rank_assignment.Layer(
                min_ranks=8,
                max_ranks=8,
                key_or_fn=lambda state: state.rank // 8,
                flag=flag,
            ),
        ]

        rank_assignment, states, store = self.build(world_size, layers)

        ctx = inprocess.rank_assignment.RankAssignmentCtx(states[0], store, set([]))
        start = time.perf_counter()
        ctx = rank_assignment(ctx)
        stop = time.perf_counter()
        self.assertLess(stop - start, timelimit)

        ctx.terminated_ranks = set(random.sample(range(world_size), num_terminated))
        start = time.perf_counter()
        try:
            ctx = rank_assignment(ctx)
        except inprocess.rank_assignment.RankDiscarded:
            pass
        stop = time.perf_counter()
        self.assertLess(stop - start, timelimit)

    def test_perf_backfill(self):
        self.perf(
            world_size=2**14,
            num_terminated=2**10,
            flag=inprocess.rank_assignment.LayerFlag.BACKFILL,
            timelimit=1.0,
        )

    def test_perf_reserve(self):
        self.perf(
            world_size=2**14,
            num_terminated=2**10,
            flag=inprocess.rank_assignment.LayerFlag.RESERVE,
            timelimit=1.0,
        )

    def test_perf_reserve_backfill(self):
        self.perf(
            world_size=2**14,
            num_terminated=2**10,
            flag=inprocess.rank_assignment.LayerFlag.RESERVE
            | inprocess.rank_assignment.LayerFlag.BACKFILL,
            timelimit=1.0,
        )

    def test_multiple_roots(self):
        world_size = 8
        layers = [
            inprocess.rank_assignment.Layer(key_or_fn=lambda state: state.rank % 2),
        ]
        rank_assignment, states, store = self.build(world_size, layers)
        ctx = inprocess.rank_assignment.RankAssignmentCtx(states[0], store, set([]))
        with self.assertRaises(RuntimeError):
            ctx = rank_assignment(ctx)

    def test_mismatched_topo(self):
        world_size = 8
        layers = [
            inprocess.rank_assignment.Layer(),
            inprocess.rank_assignment.Layer(
                key_or_fn=lambda state: state.rank % 2,
            ),
        ]

        rank_assignment, states, store = self.build(world_size, layers)
        ctx = inprocess.rank_assignment.RankAssignmentCtx(states[0], store, set([]))
        with self.assertWarns(UserWarning):
            ctx = rank_assignment(ctx)

    def backfill_test(self, flags, reference):
        world_size = 8
        layers = [
            inprocess.rank_assignment.Layer(
                flag=flags[0],
            ),
            inprocess.rank_assignment.Layer(
                key_or_fn=lambda state: state.rank // 4,
                flag=flags[1],
            ),
        ]
        rank_assignment, states, store = self.build(world_size, layers)
        ctx = inprocess.rank_assignment.RankAssignmentCtx(states[0], store, set([1]))
        ctx = rank_assignment(ctx)
        self.compare_ranks(rank_assignment, reference)

    def test_full_backfill(self):
        self.backfill_test(
            [
                inprocess.rank_assignment.LayerFlag.BACKFILL,
                inprocess.rank_assignment.LayerFlag.BACKFILL,
            ],
            [0, None, 2, 3, 4, 5, 6, 1],
        )

    def test_partial_backfill(self):
        self.backfill_test(
            [
                None,
                inprocess.rank_assignment.LayerFlag.BACKFILL,
            ],
            [0, None, 2, 1, 3, 4, 5, 6],
        )

    def test_no_backfill(self):
        self.backfill_test(
            [
                None,
                None,
            ],
            [0, None, 1, 2, 3, 4, 5, 6],
        )

    def test_group_backfill_order(self):
        world_size = 8

        layers = [
            inprocess.rank_assignment.Layer(
                flag=inprocess.rank_assignment.LayerFlag.BACKFILL,
            ),
            inprocess.rank_assignment.Layer(
                key_or_fn=lambda state: state.rank // 2,
                flag=inprocess.rank_assignment.LayerFlag.BACKFILL,
            ),
        ]
        rank_assignment, states, store = self.build(world_size, layers)

        ctx = inprocess.rank_assignment.RankAssignmentCtx(states[0], store, set([2, 3]))
        ctx = rank_assignment(ctx)
        self.compare_ranks(rank_assignment, [0, 1, None, None, 4, 5, 2, 3])

    def test_reserve(self):
        world_size = 8

        layers = [
            inprocess.rank_assignment.Layer(
                flag=inprocess.rank_assignment.LayerFlag.RESERVE,
            ),
            inprocess.rank_assignment.Layer(
                max_ranks=3,
                key_or_fn=lambda state: state.rank // 4,
                flag=inprocess.rank_assignment.LayerFlag.RESERVE,
            ),
        ]
        rank_assignment, states, store = self.build(world_size, layers)

        ctx = inprocess.rank_assignment.RankAssignmentCtx(states[0], store, set([1]))
        ctx = rank_assignment(ctx)
        self.compare_ranks(rank_assignment, [0, None, 2, 1, 3, 4, 5, None])

        ctx.terminated_ranks = set([3])
        ctx = rank_assignment(ctx)
        self.compare_ranks(rank_assignment, [0, None, 2, 1, None, 4, 5, 3])

        ctx.terminated_ranks = set([1])
        ctx = rank_assignment(ctx)
        self.compare_ranks(rank_assignment, [0, None, None, 1, None, 3, 4, 2])

    def test_propagation(self):
        world_size = 8
        layers = [
            inprocess.rank_assignment.Layer(),
            inprocess.rank_assignment.Layer(
                min_ranks=3,
                key_or_fn=lambda state: state.rank // 4,
            ),
        ]
        rank_assignment, states, store = self.build(world_size, layers)

        ctx = inprocess.rank_assignment.RankAssignmentCtx(states[0], store, set([0, 2]))
        with self.assertRaises(inprocess.rank_assignment.RankDiscarded):
            ctx = rank_assignment(ctx)
        self.compare_ranks(rank_assignment, [None, None, None, None, 0, 1, 2, 3])

    def test_root_propagation(self):
        world_size = 8
        layers = [
            inprocess.rank_assignment.Layer(
                min_ranks=7,
            ),
            inprocess.rank_assignment.Layer(
                min_ranks=1,
                key_or_fn=lambda state: state.rank // 4,
            ),
        ]
        rank_assignment, states, store = self.build(world_size, layers)

        ctx = inprocess.rank_assignment.RankAssignmentCtx(states[0], store, set([0, 5]))
        with self.assertRaises(inprocess.rank_assignment.RankDiscarded):
            ctx = rank_assignment(ctx)

        self.compare_ranks(rank_assignment, [None, None, None, None, None, None, None, None])

    def test_no_termination(self):
        world_size = 8
        layers = [
            inprocess.rank_assignment.Layer(),
            inprocess.rank_assignment.Layer(
                key_or_fn=lambda state: state.rank // 4,
            ),
        ]
        rank_assignment, states, store = self.build(world_size, layers)

        ctx = inprocess.rank_assignment.RankAssignmentCtx(states[0], store, set())
        ctx = rank_assignment(ctx)
        self.compare_ranks(rank_assignment, list(range(world_size)))

    def test_all_terminated(self):
        world_size = 8
        layers = [
            inprocess.rank_assignment.Layer(),
            inprocess.rank_assignment.Layer(
                key_or_fn=lambda state: state.rank // 4,
            ),
        ]
        rank_assignment, states, store = self.build(world_size, layers)

        ctx = inprocess.rank_assignment.RankAssignmentCtx(
            states[0], store, set(list(range(world_size)))
        )
        with self.assertRaises(inprocess.rank_assignment.RankDiscarded):
            ctx = rank_assignment(ctx)
        self.compare_ranks(rank_assignment, [None] * world_size)

    def test_propagate_reserve_backfill(self):
        world_size = 12
        layers = [
            inprocess.rank_assignment.Layer(
                flag=inprocess.rank_assignment.LayerFlag.RESERVE
                | inprocess.rank_assignment.LayerFlag.BACKFILL,
            ),
            inprocess.rank_assignment.Layer(
                min_ranks=2,
                max_ranks=3,
                key_or_fn=lambda state: state.rank // 4,
                flag=inprocess.rank_assignment.LayerFlag.RESERVE
                | inprocess.rank_assignment.LayerFlag.BACKFILL,
            ),
        ]
        rank_assignment, states, store = self.build(world_size, layers)

        ctx = inprocess.rank_assignment.RankAssignmentCtx(states[0], store, set())
        ctx = rank_assignment(ctx)

        ctx.terminated_ranks = set([0, 1, 2])

        with self.assertRaises(inprocess.rank_assignment.RankDiscarded):
            ctx = rank_assignment(ctx)
        self.compare_ranks(rank_assignment, [None, None, None, None, 3, 4, 5, 1, 6, 7, 2, 0])

    def random_test(self):
        exponent = random.randint(4, 8)
        depth = random.randint(1, exponent)
        initial_world_size = 2**exponent

        boundaries = sorted(random.sample(range(1, exponent), depth - 1))
        partitions = []
        prev = 0
        for b in boundaries:
            partitions.append(b - prev)
            prev = b

        layers = [inprocess.rank_assignment.Layer(key_or_fn='root')]
        leaf_counts = [initial_world_size]
        total = 0
        for p in partitions:
            total += p
            leaf_counts.append(initial_world_size // 2**total)
            layer = inprocess.rank_assignment.Layer(
                key_or_fn=lambda state, total=total: state.rank // (initial_world_size // 2**total)
            )
            layers.append(layer)

        reserve_up_to = random.randint(0, len(layers))
        backfill_up_to = random.randint(0, len(layers))

        for idx, layer in enumerate(reversed(layers)):
            if idx < reserve_up_to:
                if layer.flag:
                    layer.flag |= inprocess.rank_assignment.LayerFlag.RESERVE
                else:
                    layer.flag = inprocess.rank_assignment.LayerFlag.RESERVE
                leaf_count = leaf_counts[-idx - 1]
                layer.max_ranks = random.randint(leaf_count // 2, leaf_count)
            if idx < backfill_up_to:
                if layer.flag:
                    layer.flag |= inprocess.rank_assignment.LayerFlag.BACKFILL
                else:
                    layer.flag = inprocess.rank_assignment.LayerFlag.BACKFILL

        rank_assignment, states, store = self.build(initial_world_size, layers)

        ctx = inprocess.rank_assignment.RankAssignmentCtx(states[0], store, set())
        ctx = rank_assignment(ctx)

        current_world_size = initial_world_size
        n_faults = random.randint(1, 16)
        for _ in range(n_faults):
            if current_world_size == 1:
                break

            num_terminated_ranks = min(random.randint(1, 32), current_world_size - 1)
            current_world_size -= num_terminated_ranks
            ctx.terminated_ranks = set(
                random.sample(range(1, ctx.state.world_size), num_terminated_ranks)
            )
            try:
                ctx = rank_assignment(ctx)
            except inprocess.rank_assignment.RankDiscarded:
                pass

            self.assertEqual(len(rank_assignment.init_rank_map), initial_world_size)

            world_sizes = [node.state.world_size for node in rank_assignment.init_rank_map.values()]
            active_world_sizes = [
                node.state.active_world_size for node in rank_assignment.init_rank_map.values()
            ]

            self.assertEqual(len(set(world_sizes)), 1)
            self.assertEqual(len(set(active_world_sizes)), 1)

            active_world_size = active_world_sizes[0]
            world_size = world_sizes[0]

            self.assertEqual(world_size, current_world_size)
            if reserve_up_to == 0:
                self.assertEqual(active_world_size, current_world_size)
            else:
                self.assertLessEqual(active_world_size, current_world_size)

            active_ranks = {
                leaf.state.active_rank
                for leaf in rank_assignment.init_rank_map.values()
                if leaf.state.mode == inprocess.state.Mode.ACTIVE
            }

            self.assertEqual(active_ranks, set(range(active_world_size)))

    def test_random(self):
        random.seed(1)
        for idx in range(128):
            self.random_test()


class TestShiftRanks(unittest.TestCase):
    def test(self):
        world_size = 8
        store = None
        terminated_ranks = {1, 4, 5}
        ranks = {i: i for i in range(world_size)}
        for rank in ranks:
            state = inprocess.state.State(rank, world_size)
            ctx = inprocess.rank_assignment.RankAssignmentCtx(state, store, terminated_ranks)
            try:
                new_ctx = inprocess.rank_assignment.ShiftRanks()(ctx)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(rank in terminated_ranks)
                ranks[rank] = None
                continue
            self.assertEqual(new_ctx.state.world_size, world_size - len(terminated_ranks))
            self.assertEqual(new_ctx.terminated_ranks, set())
            ranks[rank] = new_ctx.state.rank
        self.assertEqual(ranks, {0: 0, 1: None, 2: 1, 3: 2, 4: None, 5: None, 6: 3, 7: 4})

    def test_empty(self):
        world_size = 8
        store = None
        terminated_ranks = {}
        ranks = {i: i for i in range(world_size)}
        for rank in ranks:
            state = inprocess.state.State(rank, world_size)
            ctx = inprocess.rank_assignment.RankAssignmentCtx(state, store, terminated_ranks)
            new_ctx = inprocess.rank_assignment.ShiftRanks()(ctx)
            self.assertEqual(new_ctx.state.world_size, world_size)
            self.assertEqual(new_ctx.terminated_ranks, set())
            ranks[rank] = state.rank
        self.assertEqual(ranks, {i: i for i in range(world_size)})

    def test_all(self):
        world_size = 8
        store = None
        terminated_ranks = set(range(world_size))
        ranks = {i: i for i in range(world_size)}
        for rank in ranks:
            state = inprocess.state.State(rank, world_size)
            ctx = inprocess.rank_assignment.RankAssignmentCtx(state, store, terminated_ranks)
            try:
                new_ctx = inprocess.rank_assignment.ShiftRanks()(ctx)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(rank in terminated_ranks)
                ranks[rank] = None
                continue
            self.assertEqual(new_ctx.state.world_size, 0)
            self.assertEqual(new_ctx.terminated_ranks, set())
            ranks[rank] = new_ctx.state.rank
        self.assertEqual(ranks, {i: None for i in range(world_size)})

    def test_compose(self):
        world_size = 8
        store = None
        terminated_ranks = {1, 2, 7}
        ranks = {i: i for i in range(world_size)}
        for rank in ranks:
            state = inprocess.state.State(rank, world_size)
            ctx1 = inprocess.rank_assignment.RankAssignmentCtx(state, store, terminated_ranks)
            try:
                ctx2 = inprocess.rank_assignment.ShiftRanks()(ctx1)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(rank in ctx1.terminated_ranks)
                ranks[rank] = None
                continue
            self.assertEqual(ctx2.state.world_size, 5)
            self.assertEqual(ctx2.terminated_ranks, set())

            rank2 = ctx2.state.rank
            world_size2 = ctx2.state.world_size

            try:
                ctx3 = inprocess.rank_assignment.ShiftRanks()(
                    ctx2,
                )
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(rank2 in ctx2.terminated_ranks)
                ranks[rank2] = None
                continue
            self.assertEqual(ctx3.state.rank, rank2)
            self.assertEqual(ctx3.state.world_size, world_size2)
            self.assertEqual(ctx3.terminated_ranks, set())
            ranks[rank] = ctx3.state.rank

        self.assertEqual(ranks, {0: 0, 1: None, 2: None, 3: 1, 4: 2, 5: 3, 6: 4, 7: None})


class TestFillGaps(unittest.TestCase):
    def test(self):
        world_size = 8
        store = None
        terminated_ranks = {1, 4, 5}
        ranks = {i: i for i in range(world_size)}
        for rank in ranks:
            state = inprocess.state.State(rank, world_size)
            ctx = inprocess.rank_assignment.RankAssignmentCtx(state, store, terminated_ranks)
            try:
                new_ctx = inprocess.rank_assignment.FillGaps()(ctx)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(rank in terminated_ranks)
                ranks[rank] = None
                continue
            self.assertEqual(new_ctx.state.world_size, world_size - len(terminated_ranks))
            self.assertEqual(new_ctx.terminated_ranks, set())
            ranks[rank] = new_ctx.state.rank
        self.assertEqual(ranks, {0: 0, 1: None, 2: 2, 3: 3, 4: None, 5: None, 6: 4, 7: 5})

    def test_empty(self):
        world_size = 8
        store = None
        terminated_ranks = {}
        ranks = {i: i for i in range(world_size)}
        for rank in ranks:
            state = inprocess.state.State(rank, world_size)
            ctx = inprocess.rank_assignment.RankAssignmentCtx(state, store, terminated_ranks)
            try:
                new_ctx = inprocess.rank_assignment.FillGaps()(ctx)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(rank in terminated_ranks)
                ranks[rank] = None
                continue
            self.assertEqual(new_ctx.state.world_size, world_size)
            self.assertEqual(new_ctx.terminated_ranks, set())
            ranks[rank] = new_ctx.state.rank
        self.assertEqual(ranks, {i: i for i in range(world_size)})

    def test_all(self):
        world_size = 8
        store = None
        terminated_ranks = set(range(world_size))
        ranks = {i: i for i in range(world_size)}
        for rank in ranks:
            state = inprocess.state.State(rank, world_size)
            ctx = inprocess.rank_assignment.RankAssignmentCtx(state, store, terminated_ranks)
            try:
                new_ctx = inprocess.rank_assignment.FillGaps()(ctx)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(rank in terminated_ranks)
                ranks[rank] = None
                continue
            self.assertEqual(new_ctx.state.world_size, 0)
            self.assertEqual(new_ctx.terminated_ranks, set())
            ranks[rank] = new_ctx.state.rank
        self.assertEqual(ranks, {i: None for i in range(world_size)})

    def test_compose(self):
        world_size = 8
        store = None
        terminated_ranks = {1, 2, 7}
        ranks = {i: i for i in range(world_size)}
        for rank in ranks:
            state = inprocess.state.State(rank, world_size)
            ctx1 = inprocess.rank_assignment.RankAssignmentCtx(state, store, terminated_ranks)
            try:
                ctx2 = inprocess.rank_assignment.FillGaps()(ctx1)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(rank in ctx1.terminated_ranks)
                ranks[rank] = None
                continue
            self.assertEqual(ctx2.state.world_size, 5)
            self.assertEqual(ctx2.terminated_ranks, set())

            rank2 = ctx2.state.rank
            world_size2 = ctx2.state.world_size

            try:
                ctx3 = inprocess.rank_assignment.FillGaps()(
                    ctx2,
                )
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(rank2 in ctx2.terminated_ranks)
                ranks[rank2] = None
                continue
            self.assertEqual(ctx3.state.rank, rank2)
            self.assertEqual(ctx3.state.world_size, world_size2)
            self.assertEqual(ctx3.terminated_ranks, set())
            ranks[rank] = ctx3.state.rank

        self.assertEqual(ranks, {0: 0, 1: None, 2: None, 3: 3, 4: 4, 5: 1, 6: 2, 7: None})


class TestFilterCountGroupedByKey(unittest.TestCase):
    def setUp(self):
        self.tmp_file = tempfile.NamedTemporaryFile()
        self.store = inprocess.store.FileStore(self.tmp_file.name)

    def test(self):
        def run(state, store, terminated_ranks):
            filter_grouped_by_key = inprocess.rank_assignment.FilterCountGroupedByKey(
                key_or_fn=state.rank // 2,
                condition=lambda count: count == 2,
            )
            ctx = inprocess.rank_assignment.RankAssignmentCtx(state, store, terminated_ranks)
            ctx2 = filter_grouped_by_key(ctx)
            assert ctx2.state == ctx.state
            assert ctx2.terminated_ranks == {0, 1, 4, 5, 6, 7}

        world_size = 8
        terminated_ranks = {1, 4, 5, 6}
        ctx = multiprocessing.get_context('fork')
        procs = []
        for rank in range(world_size):
            state = inprocess.state.State(rank, world_size)

            if rank not in terminated_ranks:
                procs.append(
                    ctx.Process(
                        target=run,
                        args=(state, self.store, terminated_ranks),
                    )
                )

        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
            self.assertEqual(proc.exitcode, 0)

    def test_composed(self):
        def run(state, store, terminated_ranks):
            filter_grouped_by_key_div2 = inprocess.rank_assignment.FilterCountGroupedByKey(
                key_or_fn=lambda state: state.rank // 2,
                condition=lambda count: count == 2,
            )
            filter_grouped_by_key_div3 = inprocess.rank_assignment.FilterCountGroupedByKey(
                key_or_fn=lambda state: state.rank // 3,
                condition=lambda count: count == 3,
            )

            rank = state.rank

            ctx1 = inprocess.rank_assignment.RankAssignmentCtx(state, store, terminated_ranks)
            ctx2 = filter_grouped_by_key_div2(ctx1)
            assert ctx2.state == ctx1.state
            assert ctx2.terminated_ranks == {0, 1, 6, 7}

            try:
                ctx3 = inprocess.rank_assignment.ShiftRanks()(ctx2)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(ctx2.state.rank in ctx2.terminated_ranks)
                return

            assert 0 <= ctx3.state.rank < 4
            assert ctx3.state.world_size == 4
            assert ctx3.terminated_ranks == set()

            ctx4 = filter_grouped_by_key_div3(ctx3)
            assert ctx4.state == ctx3.state
            assert ctx4.terminated_ranks == {3}

            try:
                ctx5 = inprocess.rank_assignment.ShiftRanks()(ctx4)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(ctx4.state.rank in ctx4.terminated_ranks)
                return

            assert 0 <= ctx5.state.rank < 3, ctx5.state.rank
            assert ctx5.state.world_size == 3
            assert ctx5.terminated_ranks == set()

            ref_ranks = {2: 0, 3: 1, 4: 2}
            assert ctx5.state.rank == ref_ranks.get(rank, None)

        world_size = 8
        terminated_ranks = {1, 6}
        ctx = multiprocessing.get_context('fork')
        procs = []
        for rank in range(world_size):
            state = inprocess.state.State(rank, world_size)

            if rank not in terminated_ranks:
                procs.append(
                    ctx.Process(
                        target=run,
                        args=(state, self.store, terminated_ranks),
                    )
                )

        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
            self.assertEqual(proc.exitcode, 0)
