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
import tempfile
import unittest


import nvidia_resiliency_ext.inprocess as inprocess


class TestShiftRanks(unittest.TestCase):
    def test(self):
        world_size = 8
        terminated_ranks = {1, 4, 5}
        ranks = {i: i for i in range(world_size)}
        for rank in ranks:
            state = inprocess.state.State()
            state.rank = rank
            state.world_size = world_size
            try:
                (
                    state,
                    new_terminated_ranks,
                ) = inprocess.rank_assignment.ShiftRanks()(state, terminated_ranks)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(rank in terminated_ranks)
                ranks[rank] = None
                continue
            self.assertEqual(state.world_size, world_size - len(terminated_ranks))
            self.assertEqual(new_terminated_ranks, set())
            ranks[rank] = state.rank
        self.assertEqual(ranks, {0: 0, 1: None, 2: 1, 3: 2, 4: None, 5: None, 6: 3, 7: 4})

    def test_empty(self):
        world_size = 8
        terminated_ranks = {}
        ranks = {i: i for i in range(world_size)}
        for rank in ranks:
            state = inprocess.state.State()
            state.rank = rank
            state.world_size = world_size
            state, new_terminated_ranks = inprocess.rank_assignment.ShiftRanks()(
                state, terminated_ranks
            )
            self.assertEqual(state.world_size, world_size)
            self.assertEqual(new_terminated_ranks, set())
            ranks[rank] = state.rank
        self.assertEqual(ranks, {i: i for i in range(world_size)})

    def test_all(self):
        world_size = 8
        terminated_ranks = set(range(world_size))
        ranks = {i: i for i in range(world_size)}
        for rank in ranks:
            state = inprocess.state.State()
            state.rank = rank
            state.world_size = world_size
            try:
                (
                    state,
                    new_terminated_ranks,
                ) = inprocess.rank_assignment.ShiftRanks()(state, terminated_ranks)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(rank in terminated_ranks)
                ranks[rank] = None
                continue
            self.assertEqual(state.world_size, 0)
            self.assertEqual(new_terminated_ranks, set())
            ranks[rank] = state.rank
        self.assertEqual(ranks, {i: None for i in range(world_size)})

    def test_compose(self):
        world_size = 8
        terminated_ranks = {1, 2, 7}
        ranks = {i: i for i in range(world_size)}
        for rank in ranks:
            state = inprocess.state.State()
            state.rank = rank
            state.world_size = world_size
            try:
                (
                    state,
                    terminated_ranks_1,
                ) = inprocess.rank_assignment.ShiftRanks()(state, terminated_ranks)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(rank in terminated_ranks)
                ranks[rank] = None
                continue
            self.assertEqual(state.world_size, 5)
            self.assertEqual(terminated_ranks_1, set())
            new_rank_1 = state.rank
            new_world_size_1 = state.world_size

            try:
                (
                    state,
                    terminated_ranks_2,
                ) = inprocess.rank_assignment.ShiftRanks()(state, terminated_ranks_1)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(new_rank_1 in terminated_ranks_1)
                ranks[new_rank_1] = None
                continue
            self.assertEqual(state.rank, new_rank_1)
            self.assertEqual(state.world_size, new_world_size_1)
            self.assertEqual(terminated_ranks_2, set())
            ranks[rank] = state.rank

        self.assertEqual(ranks, {0: 0, 1: None, 2: None, 3: 1, 4: 2, 5: 3, 6: 4, 7: None})


class TestFillGaps(unittest.TestCase):
    def test(self):
        world_size = 8
        terminated_ranks = {1, 4, 5}
        ranks = {i: i for i in range(world_size)}
        for rank in ranks:
            state = inprocess.state.State()
            state.rank = rank
            state.world_size = world_size
            try:
                (
                    state,
                    new_terminated_ranks,
                ) = inprocess.rank_assignment.FillGaps()(state, terminated_ranks)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(rank in terminated_ranks)
                ranks[rank] = None
                continue
            self.assertEqual(state.world_size, world_size - len(terminated_ranks))
            self.assertEqual(new_terminated_ranks, set())
            ranks[rank] = state.rank
        self.assertEqual(ranks, {0: 0, 1: None, 2: 2, 3: 3, 4: None, 5: None, 6: 4, 7: 5})

    def test_empty(self):
        world_size = 8
        terminated_ranks = {}
        ranks = {i: i for i in range(world_size)}
        for rank in ranks:
            state = inprocess.state.State()
            state.rank = rank
            state.world_size = world_size
            try:
                (
                    state,
                    new_terminated_ranks,
                ) = inprocess.rank_assignment.FillGaps()(state, terminated_ranks)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(rank in terminated_ranks)
                ranks[rank] = None
                continue
            self.assertEqual(state.world_size, world_size)
            self.assertEqual(new_terminated_ranks, set())
            ranks[rank] = state.rank
        self.assertEqual(ranks, {i: i for i in range(world_size)})

    def test_all(self):
        world_size = 8
        terminated_ranks = set(range(world_size))
        ranks = {i: i for i in range(world_size)}
        for rank in ranks:
            state = inprocess.state.State()
            state.rank = rank
            state.world_size = world_size
            try:
                (
                    state,
                    new_terminated_ranks,
                ) = inprocess.rank_assignment.FillGaps()(state, terminated_ranks)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(rank in terminated_ranks)
                ranks[rank] = None
                continue
            self.assertEqual(state.world_size, 0)
            self.assertEqual(new_terminated_ranks, set())
            ranks[rank] = state.rank
        self.assertEqual(ranks, {i: None for i in range(world_size)})

    def test_compose(self):
        world_size = 8
        terminated_ranks = {1, 2, 7}
        ranks = {i: i for i in range(world_size)}
        for rank in ranks:
            state = inprocess.state.State()
            state.rank = rank
            state.world_size = world_size
            try:
                (
                    state,
                    terminated_ranks_1,
                ) = inprocess.rank_assignment.FillGaps()(state, terminated_ranks)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(rank in terminated_ranks)
                ranks[rank] = None
                continue
            self.assertEqual(state.world_size, 5)
            self.assertEqual(terminated_ranks_1, set())
            new_rank_1 = state.rank
            new_world_size_1 = state.world_size

            try:
                (
                    state,
                    terminated_ranks_2,
                ) = inprocess.rank_assignment.FillGaps()(state, terminated_ranks_1)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(new_rank_1 in terminated_ranks_1)
                ranks[new_rank_1] = None
                continue
            self.assertEqual(state.rank, new_rank_1)
            self.assertEqual(state.world_size, new_world_size_1)
            self.assertEqual(terminated_ranks_2, set())
            ranks[rank] = state.rank

        self.assertEqual(ranks, {0: 0, 1: None, 2: None, 3: 3, 4: 4, 5: 1, 6: 2, 7: None})


class TestFilterGroupedByKey(unittest.TestCase):
    def setUp(self):
        self.tmp_file = tempfile.NamedTemporaryFile()
        self.store = inprocess.store.FileStore(self.tmp_file.name)

    def test(self):
        def run(state, terminated_ranks):
            rank = state.rank
            world_size = state.world_size
            filter_grouped_by_key = inprocess.rank_assignment.FilterGroupedByKey(
                key_or_fn=state.rank // 2,
                condition=lambda count: count == 2,
            )
            state, terminated_ranks = filter_grouped_by_key(state, terminated_ranks)
            assert state.rank == rank
            assert state.world_size == world_size
            assert terminated_ranks == {0, 1, 4, 5, 6, 7}

        world_size = 8
        terminated_ranks = {1, 4, 5, 6}
        ctx = multiprocessing.get_context("fork")
        procs = []
        for rank in range(world_size):
            state = inprocess.state.State()
            state.world_size = world_size
            state.rank = rank
            state.store = self.store

            if rank not in terminated_ranks:
                procs.append(
                    ctx.Process(
                        target=run,
                        args=(state, terminated_ranks),
                    )
                )

        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
            self.assertEqual(proc.exitcode, 0)

    def test_composed(self):
        def run(state, terminated_ranks):
            filter_grouped_by_key_div2 = inprocess.rank_assignment.FilterGroupedByKey(
                key_or_fn=lambda rank, _: rank // 2,
                condition=lambda count: count == 2,
            )
            filter_grouped_by_key_div3 = inprocess.rank_assignment.FilterGroupedByKey(
                key_or_fn=lambda rank, _: rank // 3,
                condition=lambda count: count == 3,
            )

            rank = state.rank
            world_size = state.world_size
            state, terminated_ranks_div2 = filter_grouped_by_key_div2(state, terminated_ranks)
            assert state.rank == rank
            assert state.world_size == world_size
            assert terminated_ranks_div2 == {0, 1, 6, 7}
            try:
                (
                    state,
                    terminated_ranks_shift,
                ) = inprocess.rank_assignment.ShiftRanks()(state, terminated_ranks_div2)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(state.rank in terminated_ranks_div2)
                return

            assert 0 <= state.rank < 4
            assert state.world_size == 4
            assert terminated_ranks_shift == set()
            rank_shift = state.rank
            world_size_shift = state.world_size

            state, terminated_ranks_div3 = filter_grouped_by_key_div3(state, terminated_ranks_shift)
            assert state.rank == rank_shift
            assert state.world_size == world_size_shift
            assert terminated_ranks_div3 == {3}
            rank_div3 = state.rank

            try:
                (
                    state,
                    terminated_ranks_shift_2,
                ) = inprocess.rank_assignment.ShiftRanks()(state, terminated_ranks_div3)
            except inprocess.rank_assignment.RankDiscarded:
                self.assertTrue(rank_div3 in terminated_ranks_div3)
                return

            assert 0 <= state.rank < 3, state.rank
            assert state.world_size == 3
            assert terminated_ranks_shift_2 == set()

            ref_ranks = {2: 0, 3: 1, 4: 2}
            assert state.rank == ref_ranks.get(rank, None)

        world_size = 8
        terminated_ranks = {1, 6}
        ctx = multiprocessing.get_context("fork")
        procs = []
        for rank in range(world_size):
            state = inprocess.state.State()
            state.world_size = world_size
            state.rank = rank
            state.store = self.store

            if rank not in terminated_ranks:
                procs.append(
                    ctx.Process(
                        target=run,
                        args=(state, terminated_ranks),
                    )
                )

        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
            self.assertEqual(proc.exitcode, 0)
