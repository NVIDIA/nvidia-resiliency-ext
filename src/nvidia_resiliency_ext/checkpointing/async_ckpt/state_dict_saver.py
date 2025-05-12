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

""" State dict saver for PyT Distributed format allowing asynchronous save. """

from dataclasses import fields
from logging import getLogger
from time import time
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from nvidia_resiliency_ext.common.device_utils import get_current_device
import torch
import torch.distributed as dist
from torch.distributed.checkpoint import CheckpointException
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata
from torch.distributed.checkpoint.planner import SavePlan, SavePlanner
from torch.distributed.checkpoint.utils import _DistWrapper, _get_failure_dict

if TYPE_CHECKING:
    from .filesystem_async import FileSystemWriterAsync


logger = getLogger(__name__)


def _compare_dataclasses(obj1, obj2):
    if type(obj1) is not type(obj2):
        return f"Objects are of different types: {type(obj1)} and {type(obj2)}"

    differences = []
    for field in fields(obj1):
        value1 = getattr(obj1, field.name)
        value2 = getattr(obj2, field.name)
        if value1 != value2:
            differences.append(f"{field.name}: {value1} != {value2}")

    return differences if differences else "All fields are equal"


class CheckpointMetadataCache:
    """
    Cache of metadata for checkpoint saving.

    This class maintains a cache of metadata used during distributed checkpoint saving operations.
    It stores various components of the save plan and metadata to optimize subsequent checkpoint
    saves by avoiding redundant planning and metadata generation when the checkpoint structure
    remains consistent across iterations.

    The cache stores:
    - cached_central_plan: The aggregated global save plan from all ranks
    - cached_local_plan: The local save plan describing how the local state_dict is written
    - cached_global_metadata: The global metadata (only held by the coordinator rank)
    - validated_cache_reuse: Flag indicating if checkpoint structures are consistent
    - validated_loaded_metadata_reuse: Flag indicating the metadata loaded from the prev checkpoint
                                       is validated to reuse, which skips all metadata communications
    - loaded_all_plans: Cached local plans from the previous checkpoint's metadata file

    This caching mechanism helps optimize checkpoint saving by:
    1. Avoiding redundant planning when checkpoint structures are consistent
    2. Reusing global metadata when possible
    3. Enabling decentralized planning when supported by the planner and storage writer
    """

    def __init__(self):
        # Cached SavePlans to skip plan in `save_state_dict_async_plan`
        # cached outcome of `SavePlan.prepare_global_plan`,
        # which aggregates local plans from all ranks
        self.cached_central_plan: SavePlan = None
        # cached outcome of `SavePlan.prepare_local_plan` describes how local state_dict is written
        self.cached_local_plan: SavePlan = None
        # Cached global metadata, only `coordinator` for dist-ckpt holds
        # if central plans are consistent over iters
        self.cached_global_metadata: Metadata = None
        # This variable records if the ckpt structures are consistent
        # so the following checkpoint savings reuse `cached_global_metadata`
        self.validated_cache_reuse: bool = False
        # The knob to enable cached metadata communication in saving
        self.validated_loaded_metadata_reuse = False
        # The cached all_local_plans from the loaded metadata file of the previous checkpoint
        self.loaded_all_plans: List[SavePlan] = None

    def set_cached_global_metadata(self, cached_global_metadata):
        """
        Sets the cached global metadata and extracts local plans from it.

        This method stores the global metadata from a previous checkpoint and attempts to extract
        the local plans from it. The local plans are used to verify if the global metadata can
        be reused in subsequent checkpoint saves.

        Args:
            cached_global_metadata (Metadata): The global metadata from a previous checkpoint
                that contains information about the checkpoint structure and local plans.

        Note:
            If the metadata does not contain local plans, a debug message is logged indicating
            that global metadata reuse verification will not be possible.
        """
        self.cached_global_metadata = cached_global_metadata
        self.loaded_all_plans = getattr(self.cached_global_metadata, "all_local_plans", None)
        if self.loaded_all_plans is None:
            logger.debug("no all_local_plans in metadata - can't verify global metadata reuse...")

    def set_cache_metadata(
        self, central_plan: SavePlan, local_plan: SavePlan, global_md_verify_reuse: bool
    ):
        """
        Sets the cached metadata and updates the cache flags.

        This method updates the cache with the latest central plan, local plan, and metadata reuse
        validation results. It also checks if the central plan is consistent with the cached plan.

        Args:
            central_plan (SavePlan): The latest central plan
            local_plan (SavePlan): The latest local plan
            global_md_verify_reuse (bool): Flag indicating if global metadata reuse is valid
        """
        self.validated_loaded_metadata_reuse = global_md_verify_reuse
        self.validated_cache_reuse = bool(central_plan == self.cached_central_plan)
        logger.debug(f"validated: {self.validated_cache_reuse}")
        self.cached_central_plan = central_plan
        self.cached_local_plan = local_plan

    def prepare_save_state_dict_ret(
        self,
        rank: int,
        coordinator: int,
        save_state_dict_ret: Tuple['FileSystemWriterAsync', Union[Metadata, None]],
    ) -> Tuple['FileSystemWriterAsync', Union[Metadata, None]]:
        """
        Prepares the save state dict return value based on the cached metadata.

        This method checks if the global metadata can be reused from the previous checkpoint.
        If so, it updates the save state dict return value with the cached global metadata.

        Args:
            rank (int): The rank of the current process
            coordinator (int): The coordinator rank
            save_state_dict_ret (Tuple[FileSystemWriterAsync, Union[Metadata, None]]):
                                The return value of the save state dict

        Returns:
            Tuple[FileSystemWriterAsync, Union[Metadata, None]]:
            The updated save state dict return value with the cached global metadata
            if it can be reused.
        """
        if (
            self.loaded_all_plans
            and self.cached_global_metadata
            and self.validated_loaded_metadata_reuse
        ):
            if coordinator == rank:
                logger.debug(
                    f"rank: {rank}, reuse global metadata from loaded"
                    f" .metadata, {save_state_dict_ret[1]}"
                )
                save_state_dict_ret = list(save_state_dict_ret)
                save_state_dict_ret[1] = self.cached_global_metadata

        elif self.validated_cache_reuse:
            logger.debug(f"rank: {rank}, cache validated")
            if save_state_dict_ret[1]:  # when global_metadata is not cached
                self.cached_global_metadata = save_state_dict_ret[1]  # Cache Metadata
            # Only Coordinator rank holds cached global_metadata
            # (None is returned for global_metadata)
            elif coordinator == rank:
                logger.debug(
                    f"rank: {rank}, reuse global metadata cached from previous"
                    f" save iteration, {save_state_dict_ret[1]}"
                )
                save_state_dict_ret = list(save_state_dict_ret)
                save_state_dict_ret[1] = self.cached_global_metadata
        return save_state_dict_ret

    def get_cache_metadata(self) -> Optional[Tuple[SavePlan, SavePlan, bool, List[SavePlan]]]:
        """
        Retrieves the cached metadata components.

        This method returns a tuple containing the cached central plan, local plan, cache reuse
        validation, and all local plans from the previous checkpoint's metadata file.
        """
        return (
            self.cached_central_plan,
            self.cached_local_plan,
            self.validated_cache_reuse,
            self.loaded_all_plans,
        )

    def get_metadata_caching_status(self):
        """
        Retrieves the current caching status

        This function returns the current caching status of the checkpoint metadata
        """
        return self.validated_cache_reuse, self.validated_loaded_metadata_reuse


_checkpoint_metadata_cache = None


def init_checkpoint_metadata_cache(cached_global_metadata: Metadata):
    """
    Initializes the checkpoint metadata cache.

    This function creates a new CheckpointMetadataCache instance and
    sets the cached global metadata from the previous checkpoint
    """
    global _checkpoint_metadata_cache
    if _checkpoint_metadata_cache is None:
        _checkpoint_metadata_cache = CheckpointMetadataCache()
    _checkpoint_metadata_cache.set_cached_global_metadata(cached_global_metadata)


def get_metadata_caching_status():
    """
    Retrieves the current caching status

    This function returns the current caching status of the checkpoint metadata
    """
    global _checkpoint_metadata_cache
    if _checkpoint_metadata_cache is not None:
        return _checkpoint_metadata_cache.get_metadata_caching_status()


def save_state_dict_async_plan(
    state_dict: STATE_DICT_TYPE,
    storage_writer: 'FileSystemWriterAsync',
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    planner: Optional[Union[SavePlanner, DefaultSavePlanner]] = None,
    enable_cache: bool = False,
    metadata_cache: Optional[CheckpointMetadataCache] = None,
) -> Tuple['FileSystemWriterAsync', Union[Metadata, None], _DistWrapper]:
    """
    First stage of saving a state dict to storage.

    This is an async adjustment of torch.distributed.checkpoint.state_dict_saver.
    In order to support async save, saving should be split into three parts:
    1. Planning
    2. Actual saving
    3. Finalization

    Out of these, step (2) *must* happen asynchronously.
    The first step is realized with this function.

    The planning part consists of several steps, described here:
    https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner

    Args:
        state_dict (STATE_DICT_TYPE): state dict to save
        storage_writer (FileSystemWriterAsync): in current version only an instance of
            FileSystemWriterAsync
        process_group (dist.ProcessGroup, optional): process group used for save planning
        coordinator_rank (int, optional): coordinator rank for planning. Defaults to 0.
        planner (SavePlanner, optional): save planner for torch.distributed.checkpoint format
        cached_ckpt_structure (Tuple[SavePlan, SavePlan, bool], Optional):
            Each object of this tuple will be used in the order as following
            cached_central_plan (SavePlan): a globally coordinated save plan
                cached in the previous iteration
            cached_local_plan (SavePlan): a local plan
                cached in the previous iteration
            validated_cache_reuse (bool): boolean value to tell global_metadata and planning dict
                is consistent over iterations

    Returns: Tuple of:
        - storage writer (the one passed as input)
        - metadata from planning (or None if we reuse cached global metadata)
        - distributed wrapper used for planning
    The return value of this function should be passed as an input to
    `save_state_dict_async_finalize` and cached_plan to skip `reduce_scatter` at planning.
    """
    cached_central_plan, cached_local_plan, validated_cache_reuse, loaded_all_plans = (
        None,
        None,
        False,
        None,
    )
    global _checkpoint_metadata_cache
    metadata_cache = metadata_cache if metadata_cache is not None else _checkpoint_metadata_cache
    if enable_cache and metadata_cache:
        cached_central_plan, cached_local_plan, validated_cache_reuse, loaded_all_plans = (
            metadata_cache.get_cache_metadata()
        )

    dist_wrapper = _DistWrapper(process_group, True, coordinator_rank)
    rank = dist_wrapper.get_rank()
    if planner is None:
        planner = DefaultSavePlanner()
    assert planner is not None

    global_metadata = None
    logger.debug(f"rank: {rank}, starting state dict save")
    local_plan = cached_local_plan
    global_md_verify_reuse = False

    def local_step():
        nonlocal local_plan
        assert planner is not None
        # PyTorch 2.4 introduced additional `metadata` argument,
        # we have to reference `is_coordinator` args by name
        planner.set_up_planner(state_dict, is_coordinator=dist_wrapper.is_coordinator)
        storage_writer.set_up_storage_writer(dist_wrapper.is_coordinator)
        if not validated_cache_reuse and local_plan is None:
            local_plan = planner.create_local_plan()
        local_plan = storage_writer.prepare_local_plan(local_plan)
        return local_plan

    def global_step(all_local_plans):
        nonlocal global_metadata
        assert planner is not None
        all_local_plans, global_metadata = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
        return all_local_plans

    # Execute local and global planning
    # Ideally we want to use the cached plan. Otherwise if the planner and storage_writer
    # allow it (`can_run_decentralized_global_plan`) we gather the plans to create
    # the metadata but prepare the plans independently on each rank.
    # In the worst case we have to reduce_scatter all the plans.
    start_plan = time()
    if validated_cache_reuse and cached_central_plan:
        logger.debug(f"rank: {rank}, Passed cache reusable")
        local_step()
        central_plan = cached_central_plan
    elif getattr(planner, 'can_run_decentralized_global_plan', False) and getattr(
        storage_writer, 'can_run_decentralized_global_plan', False
    ):
        local_plan = local_step()
        global_md_verify_reuse = verify_global_md_reuse(
            loaded_all_plans, local_plan, rank, dist_wrapper
        )

        if not loaded_all_plans or not global_md_verify_reuse:
            logger.debug(f"rank: {rank}, Passed cache non-reusable")
            all_local_plans = dist_wrapper.gather_object(local_plan)
            if dist_wrapper.is_coordinator:
                _, global_metadata = planner.create_global_plan(all_local_plans)
                global_metadata.all_local_plans = all_local_plans
        else:
            logger.debug(f"rank: {rank}, Passed cached global metadata, {global_md_verify_reuse}")
            global_metadata = None
        local_plan = planner.create_decentralized_global_plan(local_plan)
        local_plan = storage_writer.prepare_decentralized_global_plan(local_plan)
        central_plan = local_plan
    else:
        central_plan = dist_wrapper.reduce_scatter("plan", local_step, global_step)

    central_plan = planner.finish_plan(central_plan)
    end_plan = time()
    logger.debug(f"rank: {rank}, plan time: {end_plan - start_plan}")
    # Prepare async writing of tensors.
    # The `storage_writer` will store the information about tensors it needs to save
    start = time()
    storage_writer.prepare_write_data(central_plan, planner)
    end = time()
    logger.debug(f"{time()} rank: {rank}, write(async) time: {end - start}")
    save_state_dict_ret = (storage_writer, global_metadata, dist_wrapper)
    if enable_cache and metadata_cache:
        logger.debug(f"{time()} rank: {rank}, setting metadata caching")
        metadata_cache.set_cache_metadata(central_plan, local_plan, global_md_verify_reuse)
        save_state_dict_ret = metadata_cache.prepare_save_state_dict_ret(
            rank, coordinator_rank, save_state_dict_ret
        )
    return save_state_dict_ret


def verify_global_md_reuse(
    loaded_all_plans: List[SavePlan], local_plan: SavePlan, rank: int, dist_wrapper: _DistWrapper
) -> bool:
    """
    Verifies that global metadata reuse is possible by checking the loaded plans from the
     checkpoint are consistent, which means we have the same settings when resuming training.

    Args:
        loaded_all_plans: List[SavePlan], The loaded plans from the checkpoint
         (stored in checkpoint metadata).
        local_plan: SavePlan, The local save plan.
        rank: Current process rank.
        dist_wrapper (_DistWrapper): distributed wrapper created during planning

    Returns: True iff the global metadata reuse is possible.

    """
    logger.debug("verifying reuse of global metadata")
    if not loaded_all_plans:
        global_md_verify_reuse = False
        logger.debug("loaded global metadata reuse verification: no loaded plans passed")

    elif len(loaded_all_plans) == dist_wrapper.get_world_size():
        local_verify_reuse = all(
            getattr(local_plan, f.name) == getattr(loaded_all_plans[rank], f.name)
            for f in fields(local_plan)
            if f.name != 'storage_data'
        )

        if not local_verify_reuse:
            logger.debug(
                f"local_verify_reuse is False: diffs -"
                f" {_compare_dataclasses(local_plan, loaded_all_plans[rank])}"
            )
            
        all_results = torch.tensor([local_verify_reuse], dtype=torch.int, device=get_current_device())
        torch.distributed.all_reduce(all_results, op=torch.distributed.ReduceOp.MIN)
        # Check if all reduced results are True
        global_md_verify_reuse = all_results.item() == 1
    else:
        global_md_verify_reuse = False
    return global_md_verify_reuse


def save_state_dict_async_finalize(
    storage_writer: 'FileSystemWriterAsync', global_metadata: Metadata, dist_wrapper: _DistWrapper
) -> None:
    """
    Finalization of save_state_dict_async_plan.

    The input arguments are the same as the save_state_dict_async_plan output,
    the `write_results` are retrieved from the storage_writer.

    Args:
        storage_writer (FileSystemWriterAsync): storage writer used for planning
        global_metadata (Metadata): metadata created during planning
        dist_wrapper (_DistWrapper): distributed wrapper created during planning

    Returns: None
    """
    write_results = storage_writer.retrieve_write_results()

    # Gather the write results that will be saved to the metadata file.
    gather_start = time()
    all_results = dist_wrapper.gather_object(write_results)
    gather_end = time()
    logger.debug(f"{gather_end}, {dist_wrapper.get_rank()}, gather: {gather_end-gather_start}")

    # Store the metadata on coordinator rank
    if dist_wrapper.is_coordinator:
        node_failures = _get_failure_dict(all_results)
        if len(node_failures) == 0:
            assert global_metadata is not None
            write_start = time()
            storage_writer.finish(global_metadata, all_results)
            write_end = time()
            logger.debug(f"{write_end}, metadata_write: {write_end - write_start}")
        else:
            raise CheckpointException("write", node_failures)
