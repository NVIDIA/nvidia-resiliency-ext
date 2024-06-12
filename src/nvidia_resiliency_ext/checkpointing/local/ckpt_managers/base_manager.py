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

""" BaseCheckpointManager defines interface for managing local checkpoints.

Each CheckpointManager handles tasks such as:
    - cleaning up old checkpoints
    - tracking the iteration of the latest valid checkpoint
    - saving and loading checkpoints using the implemented backend.

It uses a state_dict interface, requiring users to adjust the state_dict as needed,
with MCore facilitating these modifications.
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Optional, Tuple

import torch

from ...async_ckpt.core import AsyncRequest
from ...utils import _disable_gc, debug_time
from ..base_state_dict import TensorAwareStateDict
from ..replication.group_utils import GroupWrapper
from ..replication.strategies import ReplicationStrategy

logger = logging.getLogger(__name__)

CkptID = Tuple[int, int, Any]


class CheckpointingException(Exception):
    """Base checkpointing related exception"""

    pass


class SameMachineReplicationException(CheckpointingException):
    """
    Exception raised when an attempt is made to override a file during replication.

    Inherits from `CheckpointingException`.
    """

    def __init__(self, ckpt_id):
        message = f"Checkpoint '{ckpt_id}' already exists on the same machine."
        super().__init__(message)


class BaseCheckpointManager(ABC):
    """
    The Base Checkpoint Manager provides an interface for integrating different checkpoint managers,
    abstracting replication mechanisms from the underlying implementations.
    """

    def __init__(self, session_id, repl_strategy: ReplicationStrategy = None):
        self.latest_iteration = -1
        self.repl_strategy = repl_strategy
        self.session_id = session_id
        self._rank = None

    @property
    def rank(self):
        if self._rank is None:
            if torch.distributed.is_initialized():
                self._rank = torch.distributed.get_rank()
            else:
                logger.warning("Torch distributed backend has not been initialized.")
                self._rank = 0
        return self._rank

    def _ckpt_id(self, iteration: int) -> CkptID:
        """
        Generates a unique checkpoint ID from the iteration number.
        Each rank assigns its own distinct ID.

        Args:
            iteration (int): The global iteration number.

        Returns:
            A unique checkpoint ID.
        """
        if iteration < 0:
            raise CheckpointingException(
                f"Invalid iteration: expected a non-negative value, got {iteration}."
            )
        return (iteration, self.rank, self.session_id)

    @abstractmethod
    def _my_ckpt_ids(self) -> Iterable[CkptID]:
        """Collect all locally available checkpoint IDs."""
        pass

    @abstractmethod
    def _load(self, ckpt_id: CkptID) -> TensorAwareStateDict:
        """Load of the checkpoint identified by ckpt_id.
        Should raise a CheckpointingException if failed"""
        pass

    @abstractmethod
    def _save(self, state_dict: TensorAwareStateDict, ckpt_id: CkptID):
        """Save of the tensor_aware_state_dict identified by ckpt_id.
        Should raise a SameMachineReplicationException if the checkpoint already exists"""
        pass

    @abstractmethod
    def _cleanup(self, iteration):
        """Removes outdated or invalid checkpoints after successfully saving the checkpoint
        for the specified iteration.

        Args:
            iteration : The global iteration number for which the checkpoint was successfully saved
        """
        pass

    @abstractmethod
    def _cleanup_failed_save(self, iteration):
        """Removes invalid checkpoints that could not be saved due to a failure.

        Args:
            iteration : The global iteration number for which the checkpoint failed to save.
        """

    @debug_time('BaseCheckpointManager._load_fn', logger)
    def _load_fn(self, ckpt_id: CkptID) -> TensorAwareStateDict:
        state_dict = self._load(ckpt_id)
        state_dict.restore_tensor_device(non_blocking=False)
        logger.debug(f'Finish loading {ckpt_id}')
        return state_dict

    @debug_time('BaseCheckpointManager._save_fn', logger)
    @_disable_gc()
    def _save_fn(self, id_to_state_dict):
        for ckpt_id, state_dict in id_to_state_dict.items():
            try:
                self._save(state_dict, ckpt_id)
            except Exception as e:
                logging.error(f"Exception caught during saving {ckpt_id}: {e}", exc_info=True)
                raise
        logger.debug(f'Finish saving {ckpt_id}')

    @debug_time('BaseCheckpointManager.find_latest', logger)
    def find_latest(self):
        """
        Searches for the most recent complete checkpoint and returns its global iteration number.

        If no complete checkpoints are found, the method returns -1.

        All training ranks have to call this method at once.

        Returns:
            int: The global iteration number of the most recent complete checkpoint,
            or -1 if no checkpoints are available.
        """
        if self.latest_iteration != -1:
            # Use cache to optimize performance in case of two-step loading.
            # Assumes the cache remains valid unless a new save occurs,
            # as no other operations should invalidate the most recent iteration.
            logger.debug(f'Using cached latest_iteration: {self.latest_iteration} in find_latest')
            return self.latest_iteration
        group_wrapper = GroupWrapper()

        locally_available_ids = self._my_ckpt_ids()
        if self.repl_strategy is None:
            # if replication is disabled we cannot send other ranks' shards,
            # so for our purposes they are considered unavailable
            locally_available_ids = [id_ for id_ in locally_available_ids if id_[1] == self.rank]
        # TODO: filter available IDs in case replication is enabled (but for example its' parameters have changed since)

        self.globally_available_ids = group_wrapper.all_gather_object(locally_available_ids)

        # Maps each iteration to a corresponding set of ranks
        checkpoint_coverage_map = defaultdict(set)
        for ids in self.globally_available_ids:
            for ckpt_id in ids:
                iteration, rank, session_id = ckpt_id
                assert type(iteration) is int
                assert session_id == self.session_id
                checkpoint_coverage_map[iteration].add(rank)

        self.latest_iteration = max(
            [
                iteration
                for iteration, rank_set in checkpoint_coverage_map.items()
                if rank_set == set(group_wrapper.ranks)
            ],
            default=-1,
        )
        return self.latest_iteration

    @debug_time('BaseCheckpointManager.load', logger)
    def load(self) -> Tuple[TensorAwareStateDict, str]:
        """Loads the most recent complete checkpoint.

        Ensure that `find_latest()` has been called first to identify the latest checkpoint.

        All training ranks have to call this method at once.

        Returns:
            Tuple[TensorAwareStateDict, str]
                - `state_dict`: The state dictionary loaded from the most recent complete checkpoint.
                - `ckpt_id`: The identifier of the checkpoint that was successfully loaded.
        """
        if self.latest_iteration == -1:
            raise CheckpointingException(
                "The 'find_latest' method must be called before invoking the 'load' function."
            )
        ckpt_id = self._ckpt_id(self.latest_iteration)
        logger.debug(f'Loading checkpoint from {self.latest_iteration} iteration')
        if self.repl_strategy is not None:
            plan = self.repl_strategy.retrieve_plan(self.globally_available_ids, [ckpt_id])
            my_data = {k: self._load_fn(k) for k in plan.required_ids()}
            execute_result = list(self.repl_strategy.retrieve_execute(plan, my_data).items())
            # TODO: refactor
            assert len(execute_result) == 1, f"Got {len(execute_result)} IDs, but requested only 1!"
            assert (
                execute_result[0][0] == ckpt_id
            ), f"Retrieved different ID ({execute_result[0][0]}) than requested ({ckpt_id})?"
            return execute_result[0][1], ckpt_id
        return self._load_fn(ckpt_id), ckpt_id

    @debug_time("BaseCheckpointManager.save", logger)
    def save(
        self, state_dict: TensorAwareStateDict, iteration: int, is_async: bool = False
    ) -> Optional[AsyncRequest]:
        """
        Saves the `state_dict` associated with the specified `iteration` number.

        If `is_async` is set to `True`, the save operation is performed asynchronously,
        and the function returns an `AsyncRequest` object. Otherwise, the save operation
        is completed synchronously.

        All training ranks have to call this method at once.

        Args:
            state_dict (dict): The state dictionary to be saved.
            iteration (int): The global iteration (global_step) number identifying the checkpoint.
            is_async (bool): Whether to perform the save operation asynchronously.

        Returns:
            AsyncRequest or None: An `AsyncRequest` object if `is_async` is True;
            otherwise, None as the operation completes synchronously.
        """
        assert (
            self.latest_iteration < iteration
        ), f'A newer checkpoint is already available: {self.latest_iteration} (saving {iteration})'
        if self.repl_strategy:
            save_arg = {
                ckpt_id: s_dict
                for s_dict, ckpt_id in zip(
                    *self.repl_strategy.replicate(state_dict, self._ckpt_id(iteration))
                )
            }  # TODO consider D2H (below) during replicate, and add more stuff in async save_fn
            save_arg[self._ckpt_id(iteration)].copy_tensors_to_cpu(non_blocking=True)
            save_args = (save_arg,)
        else:
            state_dict.copy_tensors_to_cpu(non_blocking=True)
            save_args = ({self._ckpt_id(iteration): state_dict},)
        save_fn = self._save_fn
        self.latest_iteration = -1  # invalidate latest_iteration

        @debug_time("finalize_fn", logger)
        def finalize_fn():
            # ThreadPoolExecutor creation takes ~0.0001s, so recreating it per save is fine.
            executor = ThreadPoolExecutor(max_workers=1)
            validated_latest_iteration = self.find_latest()  # TODO optimize
            self.latest_iteration = -1  # invalidate latest_iteration
            if validated_latest_iteration < iteration:
                if is_async:
                    executor.submit(self._cleanup_failed_save, iteration)
                    executor.shutdown(wait=False)  # Do not wait for the task to finish
                else:
                    self._cleanup_failed_save(iteration)
                raise CheckpointingException(
                    f"Failure during saving local checkpoint from iteration {iteration}"
                    f" (last valid iteration is {validated_latest_iteration})"
                )
            else:
                if validated_latest_iteration == iteration:
                    logging.info(f"Successfully saved local checkpoint from iteration {iteration}")
                else:
                    logger.warning(
                        f"WARNING: during saving iteration {iteration} "
                        f"found valid checkpoint from iteration {validated_latest_iteration}"
                    )
                if is_async:
                    executor.submit(self._cleanup, iteration)
                    executor.shutdown(wait=False)  # Do not wait for the task to finish
                else:
                    self._cleanup(iteration)

        if is_async:
            # we must wait for D2H to complete before returning control to the training
            with debug_time("ckpt_D2H_synchronize", logger):
                torch.cuda.synchronize()
            return AsyncRequest(save_fn, save_args, [finalize_fn])

        assert not is_async
        save_fn(*save_args)
        # Wait so everyone is done (necessary)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        finalize_fn()
