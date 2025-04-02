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

from nvidia_resiliency_ext.common.device_utils import (
    get_current_device, 
    get_distributed_backend, 
    get_distributed_init_method
)

import torch
from typing import Tuple
from datetime import timedelta
from typing import Tuple

import torch
from torch._C._distributed_c10d import PrefixStore
from torch.distributed import rendezvous

from nvidia_resiliency_ext.checkpointing.local.base_state_dict import TensorAwareStateDict



class Utils:

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    inited = False

    @staticmethod
    def initialize_distributed():
        if not torch.distributed.is_initialized():
            print(f'Initializing torch.distributed with rank: {Utils.rank}, world_size: {Utils.world_size}')
            
            init_method = get_distributed_init_method()
            backend = get_distributed_backend()  
 
            torch.distributed.init_process_group(backend=backend, 
                                                 world_size=Utils.world_size, 
                                                 rank=Utils.rank, init_method=init_method)

            torch.distributed.barrier()
        Utils.inited = True

    @staticmethod
    def set_world_size(world_size=None, rank=None):
        Utils.world_size = int(os.environ['WORLD_SIZE']) if world_size is None else world_size
        if (
            torch.distributed.is_initialized()
            and Utils.world_size != torch.distributed.get_world_size()
        ):
            torch.distributed.destroy_process_group()

        if rank is None:
            Utils.rank = int(os.environ['RANK'])
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
                    torch.rand(size, device=get_current_device())
                ),
            )


class SimpleTensorAwareStateDict(TensorAwareStateDict):
    def __init__(self, iteration, tensor_num=1000):
        self._tensors = [
            torch.empty((128, 128), device='cuda').random_() for _ in range(tensor_num)
        ]
        self.iteration = iteration

    def pop_tensors(self):
        raise NotImplementedError

    @property
    def tensors(self):
        raise NotImplementedError

    def tensors_to_orig_device(self):
        raise NotImplementedError

    def is_hollow(self) -> bool:
        raise NotImplementedError

    def insert_tensors(self, tensor_data):
        raise NotImplementedError

    def init_tensors(self):
        raise NotImplementedError

    def copy_tensors_to_cpu(self, non_blocking=False):
        for i, ten in enumerate(self._tensors):
            self._tensors[i] = ten.to("cpu")

    def restore_tensor_device(self, non_blocking=False):
        for i, ten in enumerate(self._tensors):
            self._tensors[i] = ten.to(device=get_current_device())

    def to_state_dict(self):
        raise NotImplementedError

    def __eq__(self, other):
        if len(self._tensors) != len(other._tensors):
            return False
        for self_ten, other_ten in zip(self._tensors, other._tensors):
            if not torch.equal(self_ten, other_ten):
                return False
        return self.iteration == other.iteration
