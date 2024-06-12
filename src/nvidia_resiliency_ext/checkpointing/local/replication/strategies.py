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

import logging
import random
from abc import ABC, abstractmethod
from typing import Generic, List, Mapping, Optional, Sequence, Tuple, TypeVar

import torch

from ...utils import debug_msg, debug_time
from ..base_state_dict import TensorAwareStateDict
from .group_utils import ExchangePlan, GroupWrapper, ProcessGroupLike, parse_group_sequence
from .utils import zip_strict

logger = logging.getLogger(__name__)


class NoReplicasAvailableError(Exception):
    """Exception raised when no replicas are available for a requested ID."""

    pass


class ReplicationStrategy(ABC):
    """Abstract base class defining the interface for replication strategies."""

    @abstractmethod
    def replicate(
        self, local_ckpt: TensorAwareStateDict, id_: str
    ) -> Tuple[List[TensorAwareStateDict], List[str]]:
        """Replicates the local checkpoint.

        Args:
            local_ckpt (TensorAwareStateDict): The local checkpoint to be replicated.
            id_ (str): Identifier for the checkpoint.

        Returns:
            A list of replicated checkpoints together with correspinding IDs
        """
        pass

    @abstractmethod
    def retrieve_plan(
        self, globally_available_ids: Mapping[int, List[str]], wanted: Sequence[str]
    ) -> ExchangePlan:
        """Generates a retrieval plan based on globally available IDs.

        Args:
            globally_available_ids (Mapping[int, List[str]]): Mapping of ranks to available IDs.
            wanted (Sequence[str]): List of IDs to retrieve.

        Returns:
            ExchangePlan: A plan detailing how to retrieve the requested IDs.
        """
        pass

    @abstractmethod
    def retrieve_execute(self, *args, **kwargs):
        """Executes the retrieval plan."""
        pass


class CliqueReplicationStrategy(ReplicationStrategy):
    """Implements a replication strategy where all participants are in a single group.

    This strategy replicates local checkpoints among all ranks in the local process group,
    enabling efficient retrieval and communication of tensor data.
    """

    def __init__(self, local_group: ProcessGroupLike, target_device="cpu"):
        self.local_group: GroupWrapper = GroupWrapper.wrap(local_group)
        self.target_device = target_device

    @debug_time('CliqueReplicationStrategy.replicate', logger)
    def replicate(
        self, local_ckpt: TensorAwareStateDict, id_: str
    ) -> Tuple[List[TensorAwareStateDict], List[str]]:
        """Replicates the local checkpoint and returns the replicated checkpoints with IDs.

        This method splits the local checkpoint into a hollow state dictionary and its tensor data,
        gathers replicated copies from other ranks, and reconstructs the state dictionaries.

        Args:
            local_ckpt (TensorAwareStateDict): The local checkpoint to replicate.
            id_ (str): Identifier for the state dict.

        Returns:
            Tuple[List[TensorAwareStateDict], List[str]]:
                - List[TensorAwareStateDict]: A list of replicated checkpoints.
                - List[str]: A list of identifiers for the replicated checkpoints.
        """
        sent_bytes = 0
        recv_bytes = 0

        # Note: it makes the original local_ckpt hollow
        # Split local_ckpt into a list of tensors and a picklable hollow state dict
        my_tensor_data = local_ckpt.pop_tensors()
        # Send hollow state dicts and tensors separately
        with debug_time("all_gather_hollow_ckpt"):
            others_local_ckpts = self.local_group.all_gather_object(local_ckpt)

        assert all(lch.is_hollow for lch in others_local_ckpts)
        my_tensor_data_nbytes = sum(ten.nbytes for ten in my_tensor_data)

        with debug_time("all_gather_others_tensor_data"):
            others_tensor_data = self.local_group.all_gather_batch(
                my_tensor_data, target_device=self.target_device
            )

        others_tensor_data_nbytes = sum(
            [sum(ten.nbytes for ten in tensor_list) for tensor_list in others_tensor_data]
        )
        sent_bytes += my_tensor_data_nbytes
        recv_bytes += others_tensor_data_nbytes - my_tensor_data_nbytes
        # Assemble hollow state dicts and tensors back into whole state dicts
        for lch, td in zip_strict(others_local_ckpts, others_tensor_data):
            lch.insert_tensors(td)
        assert all(not lch.is_hollow for lch in others_local_ckpts)

        # Label obtained state dicts with ids
        with debug_time("all_gather_other_ids"):
            other_ids = self.local_group.all_gather_object(id_)

        debug_msg(f"{sent_bytes=}")
        debug_msg(f"{recv_bytes=}")
        assert local_ckpt.is_hollow
        return others_local_ckpts, other_ids

    @debug_time('CliqueReplicationStrategy.retrieve_plan', logger)
    def retrieve_plan(
        self, globally_available_ids: Mapping[int, List[str]], wanted: Sequence[str]
    ) -> ExchangePlan:
        """Creates a plan for retrieving the specified IDs from globally available replicas.

        Args:
            globally_available_ids (Mapping[int, List[str]]): Mapping of ranks to available IDs.
            wanted (Sequence[str]): List of IDs to retrieve.

        Returns:
            ExchangePlan: A plan detailing how to retrieve the requested IDs.

        Raises:
            NoReplicasAvailableError: If no replicas are found for a requested ID.
        """
        # TODO: expand the function to multiple wanted IDs, and with smarter "routing"
        rng = random.Random(0)
        with debug_time("all_gather_wanted_ids"):
            globally_wanted = self.local_group.all_gather_object(wanted)
        result = ExchangePlan(group=self.local_group)
        for receiver, currently_wanted in zip(self.local_group.ranks, globally_wanted):
            for wanted_id in currently_wanted:
                available = set(
                    rank
                    for rank in self.local_group.ranks
                    if wanted_id in globally_available_ids[rank]
                )
                if not available:
                    raise NoReplicasAvailableError(
                        f"No replicated copies for id={wanted_id} found!"
                    )
                if receiver in available:
                    sender = receiver
                else:
                    sender = rng.choice(sorted(list(available)))
                result.plan(sender=sender, receiver=receiver, id_=wanted_id)
        return result

    @debug_time('CliqueReplicationStrategy.retrieve_execute', logger)
    def retrieve_execute(self, *args, **kwargs):
        """Executes the retrieval plan using the local group.

        Returns:
            The result of executing the retrieval plan.
        """
        return self.local_group.execute_plan(*args, **kwargs)

    @classmethod
    @debug_time('CliqueReplicationStrategy.from_replication_params', logger)
    def from_replication_params(
        cls, replication_jump: int = torch.cuda.device_count(), replication_factor: int = 2
    ) -> 'CliqueReplicationStrategy':
        """Instantiates process groups necessary for checkpoint replication.

        Training ranks are divided into `W // F` distinct groups of size `F`, where
        `W` is the world size
        and `F` is the `replication_factor`.
        Each group consists of ranks:

        `n`, `n + J`, `n + 2J`, ..., `n + (F - 1)J`,

        where `J` is the `replication_jump` and `n = aJF + b`, with:
            - `a = 0, 1, ..., (W / (JF)) - 1`
            - `b = 0, 1, ..., J - 1`.

        Checkpoint shards are exchanged and fully replicated within each group.

        **Important:** The world size (`W`) must be divisible by `J * F`.

        This grouping enables replication across different failure domains by specifying
        `J` equal to the failure blast radius.

        **Example:**
        For a world size of 32, `replication_jump = 8`, and `replication_factor = 2`,
        the replication groups (cliques) are:

        0-8, 1-9, 2-10, 3-11, 4-12, 5-13, 6-14, 7-15,
        16-24, 17-25, 18-26, 19-27, 20-28, 21-29, 22-30, 23-31

        Args:
            replication_jump (int, optional): `J` in the formula above. Represents the gap between
                successive ranks storing replicas of a given rank's data.
            replication_factor (int, optional): `F` in the formula above. Denotes the number of
                ranks storing replicas of a given rank's data.
        """
        logger.debug(f'Initializing {cls.__name__}')
        repl_process_groups_ranks: List[List[int]] = parse_group_sequence(
            replication_jump=replication_jump,
            replication_factor=replication_factor,
            world_size=torch.distributed.get_world_size(),
        )
        repl_process_groups: List[torch.distributed.ProcessGroup] = [
            torch.distributed.new_group(g) for g in repl_process_groups_ranks
        ]
        my_process_group = GroupWrapper.from_list_of_groups(repl_process_groups)
        return cls(my_process_group, target_device="cpu")


EagerT = TypeVar('EagerT')


class LazyReplicationStrategyBuilder(ReplicationStrategy, ABC, Generic[EagerT]):
    """Represents an uninitialized replication strategy.

    Replication strategy needs process groups which can be impossible to initialize
    and the time of instantiation of the ReplicationStrategy class.

    This class allows for a lazy initialization of an instance of `EagerT` type:
    >>> lazy_repl_strategy = LazyReplicationStrategyBuilder()
    >>> ...
    >>> lazy_repl_strategy.replicate(...)  # performs lazy init transparently
    >>> lazy_repl_strategy.retrieve_execute(...)  # reuses previously initialized instance transparently
    """

    def __init__(self):
        self._replication_strategy: Optional[EagerT] = None

    @property
    def replication_strategy(self) -> EagerT:
        """Lazy build on demand."""
        if self._replication_strategy is None:
            self._replication_strategy = self._eager_build()
        return self._replication_strategy

    def replicate(
        self, local_ckpt: TensorAwareStateDict, id_: str
    ) -> Tuple[List[TensorAwareStateDict], List[str]]:
        """Delegate to the underlying replication strategy."""
        return self.replication_strategy.replicate(local_ckpt, id_)

    def retrieve_plan(
        self, globally_available_ids: Mapping[int, List[str]], wanted: Sequence[str]
    ) -> ExchangePlan:
        """Delegate to the underlying replication strategy."""
        return self.replication_strategy.retrieve_plan(globally_available_ids, wanted)

    def retrieve_execute(self, *args, **kwargs):
        """Delegate to the underlying replication strategy."""
        return self.replication_strategy.retrieve_execute(*args, **kwargs)

    @abstractmethod
    def _eager_build(self) -> EagerT:
        """Instantiates the eager class."""


class LazyCliqueReplicationStrategy(LazyReplicationStrategyBuilder[CliqueReplicationStrategy]):
    """Lazy version of CliqueReplicationStrategy allowing to delay process group formation.

    Training ranks are divided into `W // F` distinct groups of size `F`, where
    `W` is the world size
    and `F` is the `replication_factor`.
    Each group consists of ranks:

    `n`, `n + J`, `n + 2J`, ..., `n + (F - 1)J`,

    where `J` is the `replication_jump` and `n = aJF + b`, with:
        - `a = 0, 1, ..., (W / (JF)) - 1`
        - `b = 0, 1, ..., J - 1`.

    Checkpoint shards are exchanged and fully replicated within each group.

    **Important:** The world size (`W`) must be divisible by `J * F`.

    This grouping enables replication across different failure domains by specifying
    `J` equal to the failure blast radius.

    **Example:**
    For a world size of 32, `replication_jump = 8`, and `replication_factor = 2`,
    the replication groups (cliques) are:

    0-8, 1-9, 2-10, 3-11, 4-12, 5-13, 6-14, 7-15,
    16-24, 17-25, 18-26, 19-27, 20-28, 21-29, 22-30, 23-31

    Args:
        replication_jump (int, optional): `J` in the formula above. Represents the gap between
            successive ranks storing replicas of a given rank's data.
        replication_factor (int, optional): `F` in the formula above. Denotes the number of
            ranks storing replicas of a given rank's data.
    """

    def __init__(
        self, replication_jump: int = torch.cuda.device_count(), replication_factor: int = 2
    ):
        super().__init__()
        self.replication_jump = replication_jump
        self.replication_factor = replication_factor

    def _eager_build(self):
        return CliqueReplicationStrategy.from_replication_params(
            self.replication_jump, self.replication_factor
        )
