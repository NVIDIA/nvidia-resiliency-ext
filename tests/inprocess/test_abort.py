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
import multiprocessing
import unittest

from nvidia_resiliency_ext.common.device_utils import get_current_device, get_distributed_backend, get_distributed_init_method, get_xla_model
import torch

import nvidia_resiliency_ext.inprocess as inprocess

from . import common

xm = get_xla_model()

@unittest.skipIf(
    not torch.distributed.is_nccl_available(), 'nccl not available'
)
class TestAbort(unittest.TestCase):
    @staticmethod
    def launch(fn, world_size=2, timeout=datetime.timedelta(seconds=10)):
        procs = []
        ctx = multiprocessing.get_context('fork')
        barrier = ctx.Barrier(world_size)
        for rank in range(world_size):
            p = ctx.Process(target=fn, args=(rank, world_size, barrier))
            p.start()
            procs.append(p)

        for p in procs:
            p.join(timeout.total_seconds())
            if p.exitcode != 0:
                for p in procs:
                    p.kill()
        exitcodes = [p.exitcode for p in procs]
        return exitcodes

    def test_multi_group(self):

        def run(rank, world_size, barrier):
            abort = inprocess.abort.AbortTorchDistributed()
            device = get_current_device()
            store = torch.distributed.TCPStore(
                host_name='localhost',
                port=29500,
                is_master=(rank == 0),
                timeout=datetime.timedelta(seconds=5),
            ) if torch.cuda.is_available() else None
            if store is not None:
                torch.distributed.init_process_group(
                    backend='nccl', store=store, rank=rank, world_size=world_size
                )
            else:
                torch.distributed.init_process_group(
                    backend=get_distributed_backend(), 
                    init_method=get_distributed_init_method(),
                    rank=rank, world_size=world_size
                )
            barrier.wait()
            size = 128
            t1 = torch.ones(size, device=device)
            t2 = torch.ones(size, device=device)
            default_group = torch.distributed.group.WORLD
            torch.distributed.all_reduce(t1, group=default_group)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            new_group = torch.distributed.new_group([0], backend=get_distributed_backend())
            if rank == 0:
                torch.distributed.all_reduce(t2, group=new_group)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            for i in range(3):
                if rank == 0:
                    torch.distributed.all_reduce(t2, group=new_group)
                torch.distributed.all_reduce(t1, group=default_group)
                if i == 1 and rank == 1:
                    abort(None)
                    break
                if i == 2 and rank == 0:
                    abort(None)
                    break
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        exitcodes = self.launch(run)
        self.assertTrue(all(ec == 0 for ec in exitcodes), exitcodes)
