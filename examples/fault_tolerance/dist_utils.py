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

import itertools
import os
from contextlib import contextmanager

import torch

#
# This module is set of low level utility functions for distributed training
#


def broadcast(tensor, src, group=torch.distributed.group.WORLD, async_op=False):
    """
    Call torch.distributed.broadcast() if distributed is in use
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.broadcast(tensor, src, group, async_op)


def all_gather(tensor_list, tensor, group=torch.distributed.group.WORLD, async_op=False):
    """
    Call torch.distributed.all_gather() if distributed is in use
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.all_gather(tensor_list, tensor, group, async_op)
    else:
        tensor_list[0].copy_(tensor)
        return None


@contextmanager
def sync_workers():
    """
    Yields distributed rank and synchronizes all workers on exit.
    """
    rank = get_rank()
    yield rank
    barrier()


def barrier():
    """
    Call torch.distributed.barrier() if distritubed is in use
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank


def all_reduce_item(value, op='sum'):
    """
    All-reduces single scalar value if distributed is in use
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if op == 'sum' or op == 'mean':
            dop = torch.distributed.ReduceOp.SUM
        elif op == 'min':
            dop = torch.distributed.ReduceOp.MIN
        elif op == 'max':
            dop = torch.distributed.ReduceOp.MAX
        elif op == 'product':
            dop = torch.distributed.ReduceOp.PRODUCT
        else:
            raise RuntimeError('Unsupported reduce op')

        backend = torch.distributed.get_backend()
        if backend == torch.distributed.Backend.NCCL:
            device = torch.device('cuda')
        elif backend == torch.distributed.Backend.GLOO:
            device = torch.device('cpu')
        else:
            raise RuntimeError('Unsupported distributed backend')

        tensor = torch.as_tensor(value).to(device)
        torch.distributed.all_reduce(tensor, dop)
        if op == 'mean':
            tensor /= get_world_size()
        ret = tensor.item()
    else:
        ret = value
    return ret


def get_world_size():
    """
    Gets total number of distributed workers or returns one if distributed is
    not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    return world_size


def init_distributed_with_tcp_store(device):
    """
    Initializes distributed backend.
    """
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = world_size > 1
    if distributed:
        if device.type == 'cuda':
            backend = 'nccl'
        elif device.type == 'cpu':
            backend = 'gloo'
        else:
            raise RuntimeError('Unknown device')
        torch.distributed.init_process_group(backend=backend, init_method='env://')
        assert torch.distributed.is_initialized()
    return distributed


def init_distributed_with_file_store(device, store_file_dir="/tmp"):
    """
    Initializes distributed backend.
    """
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = world_size > 1
    if distributed:
        if device.type == 'cuda':
            backend = 'nccl'
        elif device.type == 'cpu':
            backend = 'gloo'
        else:
            raise RuntimeError('Unknown device')
        # store file should be deleted by PyTorch
        store_filename = f'dist_store_slurm_job_id_{os.getenv("SLURM_JOB_ID","none")}.bin'
        store_file_path = os.path.join(store_file_dir, store_filename)
        store_file_path = os.path.abspath(store_file_path)
        torch.distributed.init_process_group(
            backend,
            init_method=f'file://{store_file_path}',
            world_size=world_size,
            rank=int(os.environ['RANK']),
        )
        assert torch.distributed.is_initialized()
    return distributed


def gather_tensors(tensor, device):
    tensor = tensor.to(device)
    world_size = get_world_size()
    if world_size == 1:
        tensors = [tensor]
    else:
        tensors = [torch.empty_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(tensors, tensor)

    tensors = [tensor.to(torch.device('cpu')) for tensor in tensors]
    return tensors


def gather_objects(obj, device):
    world_size = get_world_size()
    if world_size == 1:
        objs = [obj]
    else:
        objs = [None] * world_size
        torch.distributed.all_gather_object(objs, obj)
    return objs


class ResumableDistributedSampler(torch.utils.data.DistributedSampler):
    """
    Modification of DistributedSampler which allows to skip the first
    `self.start_sample_idx` samples at the beginning of each training epoch.

    To skip samples set self.start_sample_idx before a call to a
    DataLoader.__iter__ method (i.e. before starting an iteration over an
    instance of DataLoader which uses an instance of
    ResumableDistributedSampler).
    """

    def __iter__(self):
        iterator = super().__iter__()
        return itertools.islice(iterator, self.start_sample_idx, None)

    def __len__(self):
        return self.num_samples
