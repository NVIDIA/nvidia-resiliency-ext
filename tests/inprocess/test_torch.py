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
import os
import signal
import sys
import threading
import unittest
import weakref

import torch
import torch.distributed as c10d

from . import common


class TestTCPStore(unittest.TestCase):
    def tearDown(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    @unittest.mock.patch.dict(
        os.environ,
        {
            "RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(common.find_free_port()),
        },
    )
    def test_fn_reinit_clears_keys(self):
        def fn():
            torch.distributed.init_process_group(backend="gloo")

            store = torch.distributed.distributed_c10d._get_default_store()
            once = store.add("key", 1)
            self.assertEqual(once, 1)

            torch.distributed.destroy_process_group()

        for _ in range(5):
            fn()

    @unittest.mock.patch.dict(
        os.environ,
        {
            "RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(common.find_free_port()),
        },
    )
    def test_fn_exc_reinit_clears_keys(self):
        def fn():
            torch.distributed.init_process_group(backend="gloo")

            store = torch.distributed.distributed_c10d._get_default_store()
            once = store.add("key", 1)
            self.assertEqual(once, 1)

            torch.distributed.destroy_process_group()
            raise ZeroDivisionError

        for _ in range(5):
            try:
                fn()
            except ZeroDivisionError:
                pass

    @unittest.mock.patch.dict(
        os.environ,
        {
            "RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(common.find_free_port()),
        },
    )
    def test_ref_reinit_clears_keys(self):
        for _ in range(5):
            torch.distributed.init_process_group(backend="gloo")

            store_ref = weakref.ref(
                torch.distributed.distributed_c10d._get_default_store()
            )
            once = store_ref().add("key", 1)
            self.assertEqual(once, 1)

            torch.distributed.destroy_process_group()

    @unittest.expectedFailure
    @unittest.mock.patch.dict(
        os.environ,
        {
            "RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(common.find_free_port()),
        },
    )
    def test_ref_ddp_reinit_clears_keys(self):
        for _ in range(5):
            torch.distributed.init_process_group(backend="gloo")
            model = torch.nn.Linear(1, 1)
            model = torch.nn.parallel.DistributedDataParallel(model)

            store_ref = weakref.ref(
                torch.distributed.distributed_c10d._get_default_store()
            )
            once = store_ref().add("key", 1)
            self.assertEqual(once, 1)

            torch.distributed.destroy_process_group()

    @unittest.mock.patch.dict(
        os.environ,
        {
            "RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(common.find_free_port()),
        },
    )
    def test_del_ref_reinit_clears_keys(self):
        for _ in range(5):
            torch.distributed.init_process_group(backend="gloo")

            store = torch.distributed.distributed_c10d._get_default_store()
            once = store.add("key", 1)
            self.assertEqual(once, 1)

            torch.distributed.destroy_process_group()
            del store

    def test_double_host_raises(self):
        port = common.find_free_port()
        torch.distributed.TCPStore(
            host_name="localhost", port=port, world_size=1, is_master=True
        )
        with self.assertRaises(RuntimeError):
            torch.distributed.TCPStore(
                host_name="localhost", port=port, world_size=1, is_master=True
            )


@unittest.skipIf(not torch.distributed.is_nccl_available(), "nccl not available")
class ProcessGroupNCCLTest(common.MultiProcessTestCase):
    def setUp(self):
        super().setUp()

        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING
        # hence tests that use TORCH_NCCL_BLOCKING_WAIT will test it as
        # expected.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        # self.num_gpus = torch.cuda.device_count()
        self._spawn_processes()

        self.skip_return_code_checks = [
            self.test_saturated_queue_killed.__wrapped__,
        ]

    def _create_process_group_nccl(self, store, opts, world_size=None, device_id=None):
        if world_size is None:
            world_size = self.world_size
        # create nccl processgroup with opts
        c10d.init_process_group(
            "nccl",
            world_size=world_size,
            rank=self.rank,
            store=store,
            pg_options=opts,
            device_id=device_id,
        )
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    def opts(self, high_priority_stream=False):
        opts = c10d.ProcessGroupNCCL.Options()
        opts.is_high_priority_stream = high_priority_stream
        return opts

    # test borrowed from PyTorch tessuite
    def test_close_pg(self):
        # Disable ASYNC_ERROR_HANDLING for this test to ensure we can
        # programmatically abort the process group.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

        self.assertNotEqual(os.getenv("TORCH_NCCL_ABORT_IN_DESTROY_PG", "0"), "1")

        device = torch.device(self.rank)
        torch.cuda.set_device(device)

        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())

        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize state.
        pg.allreduce(t)

        # Destroy pg and validate pg is no longer valid
        torch.distributed.destroy_process_group()
        with self.assertRaises(torch.distributed.DistBackendError):
            pg.allreduce([t])

        del pg

    # test borrowed from PyTorch tessuite
    @unittest.mock.patch.dict(os.environ, {"TORCH_NCCL_ABORT_IN_DESTROY_PG": "1"})
    def test_destruct_before_terminate_pg(self):
        # Disable ASYNC_ERROR_HANDLING for this test to ensure we can
        # programmatically abort the process group.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        self.assertEqual(os.getenv("TORCH_NCCL_ABORT_IN_DESTROY_PG", "0"), "1")

        size = 1024
        device = torch.device(self.rank)
        torch.cuda.set_device(device)
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())

        t = torch.ones(size, dtype=torch.int64, device=device)
        # First allreduce to initialize state.
        pg.allreduce(t)
        # force destruction before terminating comms, destructor would
        # terminate comms
        del pg

    # test borrowed from PyTorch tessuite
    @unittest.mock.patch.dict(os.environ, {"TORCH_NCCL_ABORT_IN_DESTROY_PG": "1"})
    def test_abort_in_destroy_pg(self):
        # Disable ASYNC_ERROR_HANDLING for this test to ensure we can
        # programmatically abort the process group.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        self.assertEqual(os.getenv("TORCH_NCCL_ABORT_IN_DESTROY_PG", "0"), "1")

        size = 1024
        device = torch.device(self.rank)
        torch.cuda.set_device(device)

        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())

        t = torch.ones(size, dtype=torch.int64, device=device)
        # First allreduce to initialize state.
        pg.allreduce(t)

        # Destroy pg and validate pg is NOT in working condition since
        # we have shutdown comms
        torch.distributed.destroy_process_group()

        with self.assertRaises(torch.distributed.DistBackendError):
            pg.allreduce([t])

    # test borrowed from PyTorch tessuite
    @unittest.mock.patch.dict(os.environ, {"TORCH_NCCL_ABORT_IN_DESTROY_PG": "1"})
    def test_abort_in_destroy_multi_pgs(self):
        self.assertEqual(os.getenv("TORCH_NCCL_ABORT_IN_DESTROY_PG", "0"), "1")
        size = 1024
        device = torch.device(self.rank)
        torch.cuda.set_device(device)
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        t = torch.ones(size, dtype=torch.int64, device=device)
        # First allreduce to initialize default PG's communicator.
        pg.allreduce(t).wait()
        new_pg1 = c10d.new_group([0, 1])
        new_pg2 = c10d.new_group([0, 1])
        t1 = torch.ones(size, dtype=torch.int64, device=device)
        t2 = torch.ones(size, dtype=torch.int64, device=device)
        new_pg1.allreduce(t1).wait()
        new_pg2.allreduce(t2).wait()
        backend = pg._get_backend(torch.device(device))
        # default PG's backend should have a split count of 2
        self.assertEqual(backend.comm_split_count(), 2)
        # shutdown all NCCL PGs in one shot
        torch.distributed.destroy_process_group()

    # test borrowed from PyTorch tessuite
    @unittest.mock.patch.dict(os.environ, {"TORCH_NCCL_ABORT_IN_DESTROY_PG": "1"})
    def test_abort_in_destroy_mixed_empty_pgs(self):
        self.assertEqual(os.getenv("TORCH_NCCL_ABORT_IN_DESTROY_PG", "0"), "1")
        size = 1024
        device = torch.device(self.rank)
        torch.cuda.set_device(device)

        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        t = torch.ones(size, dtype=torch.int64, device=device)
        # First allreduce to initialize default PG's communicator.
        pg.allreduce(t).wait()
        # PG1 is an PG without comms initialized, since we don't call
        # collective on it
        c10d.new_group([0, 1])
        new_pg2 = c10d.new_group([0, 1])
        t2 = torch.ones(size, dtype=torch.int64, device=device)

        new_pg2.allreduce(t2).wait()
        backend = pg._get_backend(torch.device(device))
        # default PG's backend should have a split count of 1
        self.assertEqual(backend.comm_split_count(), 1)
        # shutdown all NCCL PGs in one shot
        torch.distributed.destroy_process_group()

    @common.subtests([{"pass_device": pd} for pd in [True, False]])
    def test_destroy_reinit(self, pass_device):
        size = 10

        device = torch.device(self.rank)
        torch.cuda.set_device(device)

        if pass_device:
            init_process_group_device = device
        else:
            init_process_group_device = None

        self.assertTrue(torch.distributed.is_available())

        for i in range(2):
            store = common.wrap_store(
                c10d.FileStore(self.file_name, self.world_size), pass_device, i
            )
            self._create_process_group_nccl(
                store, self.opts(), device_id=init_process_group_device
            )
            self.assertTrue(torch.distributed.is_initialized())

            t = torch.ones(size, dtype=torch.int64, device=device)
            torch.distributed.all_reduce(t)
            self.assertTrue((t == self.world_size).all())

            torch.distributed.destroy_process_group()
            self.assertFalse(torch.distributed.is_initialized())

    @common.subtests([{"pass_device": pd} for pd in [True, False]])
    def test_unmatched_collective(self, pass_device):
        size = 10

        device = torch.device(self.rank)
        torch.cuda.set_device(device)

        if pass_device:
            init_process_group_device = device
        else:
            init_process_group_device = None

        self.assertTrue(torch.distributed.is_available())

        for i in range(2):
            store = common.wrap_store(
                c10d.FileStore(self.file_name, self.world_size), pass_device, i
            )
            self._create_process_group_nccl(
                store, self.opts(), device_id=init_process_group_device
            )
            self.assertTrue(torch.distributed.is_initialized())

            t = torch.ones(size, dtype=torch.int64, device=device)
            torch.distributed.all_reduce(t)
            self.assertTrue((t == self.world_size).all())

            if self.rank == 0:
                torch.distributed.all_reduce(t)

            torch.distributed.destroy_process_group()
            self.assertFalse(torch.distributed.is_initialized())

    @unittest.mock.patch.dict(os.environ, {"TORCH_NCCL_ABORT_IN_DESTROY_PG": "dummy"})
    @common.subtests([{"nccl_abort": na} for na in ["0", "1"]])
    def test_saturated_queue(self, nccl_abort, pass_device=True):
        os.environ["TORCH_NCCL_ABORT_IN_DESTROY_PG"] = nccl_abort
        size = 2**26
        iters = 30

        device = torch.device(self.rank)
        torch.cuda.set_device(device)

        if pass_device:
            init_process_group_device = device
        else:
            init_process_group_device = None

        self.assertTrue(torch.distributed.is_available())

        for i in range(2):
            store = common.wrap_store(
                c10d.FileStore(self.file_name, self.world_size), nccl_abort, i
            )
            self._create_process_group_nccl(
                store, self.opts(), device_id=init_process_group_device
            )
            self.assertTrue(torch.distributed.is_initialized())

            t = torch.ones(size, dtype=torch.int64, device=device)
            for i in range(iters):
                torch.distributed.all_reduce(t)

            torch.distributed.destroy_process_group()
            self.assertFalse(torch.distributed.is_initialized())

            self.assertTrue((t == 2**iters).all())

    @unittest.mock.patch.dict(os.environ, {"TORCH_NCCL_ABORT_IN_DESTROY_PG": "1"})
    def test_saturated_queue_killed(self, rank_to_kill=0, pass_device=True):
        self.assertEqual(os.environ["TORCH_NCCL_ABORT_IN_DESTROY_PG"], "1")
        size = 2**26
        iters = 30

        device = torch.device(self.rank)
        torch.cuda.set_device(device)

        if pass_device:
            init_process_group_device = device
        else:
            init_process_group_device = None

        self.assertTrue(torch.distributed.is_available())
        store = common.wrap_store(
            c10d.FileStore(self.file_name, self.world_size), rank_to_kill
        )
        self._create_process_group_nccl(
            store, self.opts(), device_id=init_process_group_device
        )
        self.assertTrue(torch.distributed.is_initialized())

        t = torch.ones(size, dtype=torch.int64, device=device)
        for i in range(iters):
            torch.distributed.all_reduce(t)

        if self.rank == rank_to_kill:
            os.kill(os.getpid(), signal.SIGKILL)

        torch.distributed.destroy_process_group()
        self.assertTrue((t < 2**iters).all())


class ProcessGroupGLOOTest(common.MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()
        self.skip_return_code_checks = []

    def _create_process_group_gloo(
        self, store, timeout, world_size=None, device_id=None
    ):
        if world_size is None:
            world_size = self.world_size
        # create nccl processgroup with opts
        c10d.init_process_group(
            "gloo",
            world_size=world_size,
            rank=self.rank,
            store=store,
            timeout=timeout,
        )
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    @unittest.expectedFailure
    def test_destroy_process_group(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        self._create_process_group_gloo(store, datetime.timedelta(seconds=2))

        size = 1024 * 1024
        tensor = torch.tensor(size)

        timer = threading.Timer(
            interval=1, function=torch.distributed.destroy_process_group
        )
        timer.start()

        try:
            if self.rank == 0:
                torch.distributed.all_reduce(tensor)

            torch.distributed.barrier()
        except RuntimeError:
            sys.exit(1)
