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

import os
from datetime import timedelta
from typing import Tuple

import torch
from torch._C._distributed_c10d import PrefixStore
from torch.distributed import rendezvous


class Utils:
    world_size = torch.cuda.device_count()
    rank = int(os.environ["LOCAL_RANK"])
    inited = False
    store = None

    @staticmethod
    def initialize_distributed():
        if not torch.distributed.is_initialized() and Utils.rank >= 0:
            print(
                f"Initializing torch.distributed with rank: {Utils.rank}, "
                f"world_size: {Utils.world_size}"
            )
            torch.cuda.set_device(Utils.rank % torch.cuda.device_count())
            init_method = "tcp://"
            master_ip = os.getenv("MASTER_ADDR", "localhost")
            master_port = os.getenv("MASTER_PORT", "6000")
            init_method += master_ip + ":" + master_port
            rendezvous_iterator = rendezvous(
                init_method, Utils.rank, Utils.world_size, timeout=timedelta(minutes=1)
            )
            store, rank, world_size = next(rendezvous_iterator)
            store.set_timeout(timedelta(minutes=1))

            # Use a PrefixStore to avoid accidental overrides of keys used by
            # different systems (e.g. RPC) in case the store is multi-tenant.
            store = PrefixStore("default_pg", store)
            Utils.store = store

            torch.distributed.init_process_group(
                backend="nccl",
                world_size=Utils.world_size,
                rank=Utils.rank,
                store=store,
            )

            torch.distributed.barrier()
        Utils.inited = True

    @staticmethod
    def set_world_size(world_size=None, rank=None):
        Utils.world_size = torch.cuda.device_count() if world_size is None else world_size
        if (
            torch.distributed.is_initialized()
            and Utils.world_size != torch.distributed.get_world_size()
        ):
            torch.distributed.destroy_process_group()

        if rank is None:
            Utils.rank = int(os.environ["LOCAL_RANK"])
            if Utils.rank >= Utils.world_size:
                Utils.rank = -1
        else:
            Utils.rank = rank


class TestModel(torch.nn.Module):
    def __init__(self, size: Tuple, ntensor: int) -> None:
        super().__init__()
        for i in range(ntensor):
            self.register_parameter(
                f"param_{i}",
                torch.nn.Parameter(
                    torch.rand(size, device=torch.device(f"cuda:{torch.cuda.current_device()}"))
                ),
            )
