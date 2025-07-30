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

import torch


def all_gather_object(obj, group):
    """Gathers an object from all processes in a given group.

    Args:
        obj (Any): The object to gather.
        group (torch.distributed.ProcessGroup): The process group to gather from.

    Returns:
        list: A list containing the gathered objects from each process.
    """
    world_size = get_world_size(group)
    objs = [None] * world_size
    if world_size > 1:
        torch.distributed.all_gather_object(objs, obj, group)
    else:
        objs[0] = obj
    return objs


def get_world_size(group):
    """Returns the world size (number of processes) in the given group.

    Args:
        group (torch.distributed.ProcessGroup)

    Returns:
        int: The world size of the group
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size(group)
    else:
        world_size = 1
    return world_size


def get_rank(group):
    """Returns the rank (process ID) in the given group.

    Args:
        group (torch.distributed.ProcessGroup)

    Returns:
        int: The rank of the calling process in the specified group
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank(group)
    else:
        rank = 0
    return rank


def get_device_for_backend(group):
    """Find the device that should be used with given distributed group backend."""
    dist_device = torch.device("cpu")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_backend(group) == torch.distributed.Backend.NCCL:
            dist_device = torch.device("cuda")
    return dist_device


def all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
    """All reduce or no-op if the world size is 1."""
    if get_world_size(group) > 1:
        torch.distributed.all_reduce(tensor=tensor, op=op, group=group, async_op=async_op)


def gather_on_rank0(tensor, group=None):
    """Gather tensors on rank0,
    returns a list of gathered tensors existing on the same device as the input tensor.
    """
    gather_list = None
    world_size = get_world_size(group)
    if world_size > 1:
        rank = get_rank(group)
        orig_device = tensor.device
        gather_device = get_device_for_backend(group)
        tensor = tensor.to(gather_device)
        if rank == 0:
            gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
        torch.distributed.gather(tensor=tensor, gather_list=gather_list, dst=0, group=group)
        if rank == 0:
            for i in range(world_size):
                gather_list[i] = gather_list[i].to(orig_device)
    else:
        gather_list = [tensor]
    return gather_list


def is_all_true(flag: bool, group=None) -> bool:
    """Check if a boolean flag is true on all processes in the group."""
    ret = flag
    if get_world_size(group) > 1:
        device = get_device_for_backend(group)
        flag_tensor = torch.tensor([1.0 if flag else 0], dtype=torch.float32, device=device)
        torch.distributed.all_reduce(flag_tensor, op=torch.distributed.ReduceOp.MIN, group=group)
        ret = bool(flag_tensor.item() > 0)
    return ret
