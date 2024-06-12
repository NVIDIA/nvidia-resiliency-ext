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

from __future__ import annotations

import logging
import typing
from dataclasses import dataclass, field
from itertools import islice
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch.distributed as dist

from ...utils import debug_msg, debug_time
from ..base_state_dict import TensorAwareStateDict
from ._torch_future import recv_object_list, send_object_list
from .torch_device_utils import TensorPlaceholder

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExchangePlanEntry:
    """
    Represents an entry in an exchange plan.

    Attributes:
        sender (int): The rank of the sender.
        receiver (int): The rank of the receiver.
        id_ (str): The unique identifier for the exchange entry.
    """

    sender: int
    receiver: int
    id_: str


@dataclass
class ExchangePlan:
    """
    Represents an exchange plan for managing exchanges between ranks.

    Attributes:
        group (GroupWrapper): Contains information about global ranks.
        _entries (List[ExchangePlanEntry]): List of exchange entries with sender-receiver pairs.
    """

    group: "GroupWrapper"
    _entries: List[ExchangePlanEntry] = field(default_factory=list)

    def plan(self, *args, **kwargs):
        """Adds a new exchange entry to the plan."""
        self._entries.append(ExchangePlanEntry(*args, **kwargs))

    @property
    def entries(self) -> Tuple[ExchangePlanEntry]:
        """
        Retrieves all exchange entries as an immutable tuple.

        Returns:
            Tuple[ExchangePlanEntry]: A tuple containing all exchange entries.
        """
        return tuple(self._entries)

    def required_ids(self, rank=None):
        """
        Retrieves the set of exchange entry IDs required by the specified rank.

        Args:
            rank (int, optional): The rank for which to get the required IDs.
                                  If None, the global rank is used.

        Returns:
            set: A set of IDs for the required exchange entries for the specified rank.
        """
        if rank is None:
            rank = self.group.get_global_rank()
        return set(e.id_ for e in self._entries if e.sender == rank)

    # TODO: what sanity checks should be done and how detailed should the error message be
    # Ideas:
    #   - Check for duplicates (deduplicate or raise an error?)
    #   - Check for shape mismatch
    #   - Check if plans are consistent across ranks:
    #       - Two plans are consistent if their operations are in the same order.
    #    - We don't want for user to rely on the order of operations.
    #      We could check if e.g.: (notation: [sender]-[id]->[recv])
    #           1-A->2-A->3
    #      In this case, user relies on the assumption that `1-A->2` happens before 2-A->3


def batched(iterable, n):
    """
    Batch data from the iterable into tuples of length n. The last batch may be shorter than n.
    """
    # Available in itertools from version 3.12, adding here for backwards compatibility.
    # Source: https://docs.python.org/3/library/itertools.html#itertools.batched
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def parse_group_sequence(replication_jump, replication_factor, world_size):
    """
    Creates groups based on the specified replication jump, replication factor, and world size.

    Args:
        replication_jump (int): The jump size for replication.
        replication_factor (int): The factor by which to replicate.
        world_size (int): The total size of the world.

    Returns:
        list: A list of groups formed based on the provided parameters.
    """
    assert replication_jump > 0, "Rank cannot store a replica of itself!"
    assert replication_factor > 0, "Tried to create empty replication groups!"
    # TODO: assert replication_factor > 1
    assert world_size % (replication_jump * replication_factor) == 0, (
        f"Cannot split {world_size} ranks into replication groups! "
        f"Ranks {replication_jump - 1}, {2 * replication_jump - 1}, ... contains "
        f"{world_size // replication_jump} ranks, but this cannot be split into "
        f"groups of size {replication_factor}."
    )
    result = []
    for modulus in range(replication_jump):
        seq_to_slice = list(range(modulus, world_size, replication_jump))
        result.extend(batched(seq_to_slice, replication_factor))
    result.sort()
    return result


class GroupWrapper:
    """
    Wrapper class for managing a distributed process group.

    Attributes:
        _group (ProcessGroupLike): The underlying process group object.
    """

    def __init__(self, group=None):
        """Initializes the GroupWrapper with an optional process group."""
        self._group = group

    @staticmethod
    def wrap(group: ProcessGroupLike):
        """Wraps a given process group into a GroupWrapper.

        Args:
            group (ProcessGroupLike): The process group to wrap.

        Returns:
            GroupWrapper: A wrapped process group.

        Raises:
            ValueError: If the group is of an unsupported type.
        """
        if isinstance(group, dist.ProcessGroup):
            return GroupWrapper(group)
        elif isinstance(group, GroupWrapper):  # type: ignore
            return group
        else:
            raise ValueError(f"Unsupported type: {type(group)}!")

    @staticmethod
    def from_list_of_groups(list_of_groups: List[dist.ProcessGroup]) -> GroupWrapper:
        """Creates a GroupWrapper from a list of process groups, identifying the current group.

        Args:
            list_of_groups (List[dist.ProcessGroup]): A list of process groups.

        Returns:
            GroupWrapper: A GroupWrapper for the current process group.

        Raises:
            AssertionError: If the rank is in more than one group or in none.
        """
        my_rank = dist.get_rank()
        my_process_group = [
            g for g in list_of_groups if my_rank in dist.get_process_group_ranks(g)  # type: ignore
        ]
        assert (
            len(my_process_group) <= 1
        ), f"Rank {my_rank} is in more groups than one! Groups: {my_process_group}"
        assert len(my_process_group) >= 1, f"Rank {my_rank} not in any process group!"
        return GroupWrapper(my_process_group[0])

    @property
    def group(self):
        """Returns the underlying process group.

        Returns:
            ProcessGroupLike: The underlying process group.
        """
        return self._group

    @property
    def backend(self):
        """Returns the backend of the process group.

        Returns:
            str: The backend used by the process group.
        """
        return dist.get_backend(self.group)

    @property
    def supported_devices(self):
        """Lists the devices supported by the backend.

        Returns:
            List[torch.device]: A list of supported devices.
        """
        return [
            torch.device(device)
            for device in dist.Backend.backend_capability[self.backend]  # type: ignore
        ]

    def get_device(self, wanted_device=None):
        """Gets the appropriate device for operations.

        Args:
            wanted_device (str, optional): The desired device name.

        Returns:
            torch.device: The selected device.

        Raises:
            AssertionError: If the desired device is not supported.
        """
        if wanted_device is None:
            wanted_device = self.supported_devices[0]
        assert (
            torch.device(wanted_device) in self.supported_devices
        ), f"Selected backend {self.backend} does not support the selected device {wanted_device}!"
        return wanted_device

    def get_group_rank(self, global_rank=None):
        """Gets the local group rank corresponding to a global rank.

        Args:
            global_rank (int, optional): The global rank to look up.

        Returns:
            int: The local group rank.
        """
        if global_rank is None:
            return dist.get_rank(self.group)
        if self.group is None:
            return global_rank
        return dist.get_group_rank(self.group, global_rank)  # type: ignore

    @property
    def my_group_rank(self):
        """Gets the local rank of the current process in its group.

        Returns:
            int: The local group rank of the current process.
        """
        return self.get_group_rank(None)

    def get_global_rank(self, group_rank=None):
        """Converts a local group rank to a global rank.

        Args:
            group_rank (int, optional): The local group rank.

        Returns:
            int: The corresponding global rank.
        """
        if group_rank is None or self.group is None:
            return dist.get_rank()
        return dist.get_global_rank(self.group, group_rank)  # type: ignore

    @property
    def my_global_rank(self):
        """Gets the global rank of the current process.

        Returns:
            int: The global rank of the current process.
        """
        return self.get_global_rank(None)

    @property
    def ranks(self):
        """Gets the ranks of all processes in the group.

        Returns:
            range: A range of ranks in the process group.
        """
        if self.group is None:
            return range(dist.get_world_size())
        return dist.get_process_group_ranks(self.group)  # type: ignore

    @property
    def world_size(self):
        """Gets the total number of processes in the group.

        Returns:
            int: The total number of processes in the group.
        """
        return dist.get_world_size(self.group)

    def __repr__(self):
        return (
            f"<ProcessGroup of size {self.world_size}, rank: "
            f"local={self.my_group_rank}, global={self.my_global_rank}>"
        )

    def all_gather_object(self, my_obj: T) -> List[T]:
        """Gathers an object from all processes in the group.

        Args:
            my_obj (T): The object to gather.

        Returns:
            List[T]: A list of gathered objects from all ranks.
        """
        result: List[Optional[T]] = [None] * self.world_size
        dist.all_gather_object(result, my_obj, group=self.group)
        return typing.cast(List[T], result)

    def broadcast(self, *args, **kwargs):
        """Broadcasts data from the current process to all processes in the group."""
        return dist.broadcast(*args, **kwargs, group=self.group)

    def all_gather_batch(
        self, my_tensors: List[torch.Tensor], target_device=None
    ) -> List[List[torch.Tensor]]:
        """
        Perform an all-gather on a list of tensors.
        Use `comm_device` for communication (will throw error if unsupported for selected group).
        If not specified, use any supported device for the given group.
        """
        tensor_devices = list(set([ten.device for ten in my_tensors]))
        debug_msg(f"{tensor_devices=}")
        debug_msg(f"{target_device=}")

        # We gather metadata on the tensors being received, including shape and original device type
        with debug_time("all_gather_placeholders"):
            group_tensor_placeholders: List[List[TensorPlaceholder]] = self.all_gather_object(
                [TensorPlaceholder(ten) for ten in my_tensors]
            )
        result = []
        for broadcasting_rank, tensor_placeholders in enumerate(group_tensor_placeholders):
            i_am_broadcasting = broadcasting_rank == self.my_group_rank
            global_broadcasting_rank = self.get_global_rank(broadcasting_rank)
            result.append([])

            for i, tp in enumerate(tensor_placeholders):
                if i_am_broadcasting:
                    ten = my_tensors[i]
                else:
                    ten = tp.empty_like()
                self.broadcast(ten, src=global_broadcasting_rank)
                if target_device is not None:
                    ten = ten.to(target_device, non_blocking=True)
                result[-1].append(ten)

        return result

    # TODO: use dist.batch_isend_irecv
    def isend_state_dict(self, state_dict: TensorAwareStateDict, dst: int) -> Dict[str, float]:
        """Sends the state dictionary to a specified destination.

        This method pops tensor data from the state dictionary, sends the state
        dictionary object to the destination, and then sends each tensor
        individually to the specified destination.

        Args:
            state_dict (TensorAwareStateDict): The state dictionary to send.
            dst (int): The destination rank to which the state dictionary is sent.
        """
        tensor_data = state_dict.pop_tensors()
        log_data = {"data_sent": sum(ten.nbytes for ten in tensor_data)}
        # TODO: count object size without repickling
        self.send_object(state_dict, dst)
        for ten in tensor_data:
            ten = ten.cuda()
            dist.send(ten, dst, group=self.group)
        state_dict.insert_tensors(tensor_data)
        return log_data

    def recv_object(self, src):
        """Receives an object from a specified source.

        This method receives an object from the source rank and returns it.

        Args:
            src (int): The source rank from which to receive the object.

        Returns:
            The received object.
        """
        result = [None]
        recv_object_list(result, src, group=self.group)
        return result[0]

    def send_object(self, obj, dst):
        """Sends an object to a specified destination.

        This method wraps the object in a list and sends it to the destination.

        Args:
            obj: The object to send.
            dst (int): The destination rank to which the object is sent.
        """

        send_object_list([obj], dst, group=self.group)

    # TODO: use dist.batch_isend_irecv
    def irecv_state_dict(self, src: int):
        """Receives a state dictionary from a specified source.

        This method initializes an empty state dictionary and receives tensor data
        from the specified source, populating the state dictionary with the received
        tensor data.

        Args:
            src (int): The source rank from which to receive the state dictionary.

        Returns:
            TensorAwareStateDict: The received state dictionary populated with tensors.
        """
        hollow_ckpt: TensorAwareStateDict = self.recv_object(src)
        hollow_ckpt.init_tensors()
        for ten in hollow_ckpt.tensors:
            if ten.device.type == 'cpu':
                cuda_tensor = torch.empty_like(ten, device='cuda')
                dist.recv(cuda_tensor, src, group=self.group)
                ten.copy_(cuda_tensor.cpu())
            else:
                dist.recv(ten, src, group=self.group)
        log_data = {"data_recv": sum(ten.nbytes for ten in hollow_ckpt.tensors)}
        return hollow_ckpt, log_data

    def execute_plan(
        self, exchange_plan: ExchangePlan, my_data: Dict[str, TensorAwareStateDict]
    ) -> Tuple[Dict[str, TensorAwareStateDict], Dict[str, float]]:
        """Executes an exchange plan, sending and receiving state dictionaries.

        This method processes the provided exchange plan, ensuring that all required
        data is available and executing the send and receive operations according to
        the plan. It returns a dictionary of received state dictionaries.

        Args:
            exchange_plan (ExchangePlan): The plan detailing send and receive operations.
            my_data (Dict[str, TensorAwareStateDict]): The local data available for sending.

        Returns:
            Dict[str, TensorAwareStateDict]: A dictionary of received state dictionaries.

        Raises:
            AssertionError: If not all required data is provided in my_data.
        """
        sent_bytes = 0
        recv_bytes = 0
        # TODO: deduplicate or raise an error if found duplicates?
        # assert not exchange_plan.contains_duplicates(), "Exchange plan contains duplicates"
        missing_ids = exchange_plan.required_ids().difference(my_data.keys())
        assert not missing_ids, f"Not all required data provided! Missing ids: {missing_ids}"

        recv_buf = {}
        for entry in exchange_plan.entries:
            # Sending to oneself
            if entry.sender == self.get_global_rank() == entry.receiver:
                # Get result without any communication
                recv_buf[entry.id_] = my_data[entry.id_]
            # I'm sending to someone else
            elif entry.sender == self.get_global_rank():
                send_log_data = self.isend_state_dict(my_data[entry.id_], entry.receiver)
                sent_bytes += send_log_data["data_sent"]
            # I'm receiving from someone else
            elif entry.receiver == self.get_global_rank():
                recv_buf[entry.id_], recv_log_data = self.irecv_state_dict(entry.sender)
                recv_bytes += recv_log_data["data_recv"]
            # else:
            #  Do nothing - entry not relevant to me

        debug_msg(f"{sent_bytes=}")
        debug_msg(f"{recv_bytes=}")
        return recv_buf


ProcessGroupLike = Union[GroupWrapper, dist.ProcessGroup]
