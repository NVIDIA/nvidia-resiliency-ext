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

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import CheckpointException
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata
from torch.distributed.checkpoint.planner import SavePlan, SavePlanner
from torch.distributed.checkpoint.storage import WriteResult
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
    """Cache of metadata for checkpoint saving.

    This class maintains a cache of metadata used during distributed checkpoint saving operations.
    It stores various components of the save plan and metadata to optimize subsequent checkpoint
    saves by avoiding redundant planning and metadata generation when the checkpoint structure
    remains consistent across iterations.

    This caching mechanism helps optimize checkpoint saving by:
    1. Avoiding redundant planning when checkpoint structures are consistent
    2. Reusing global metadata when possible
    3. Enabling decentralized planning when supported by the planner and storage writer

    Args:
        cached_central_plan (SavePlan): The aggregated global save plan from all ranks
        cached_local_plan (SavePlan): The local save plan describing how the local state_dict is written
        cached_global_metadata (Metadata): The global metadata (only held by the coordinator rank)
        validated_cache_reuse (bool): Flag indicating if checkpoint structures are consistent
        validated_loaded_metadata_reuse (bool): Flag indicating the metadata loaded from the prev checkpoint
                                         is validated to reuse, which skips all metadata communications
        loaded_all_plans (List[SavePlan]): Cached local plans from the previous checkpoint's metadata file

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
        self.validated_loaded_metadata_reuse: bool = False
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


def init_checkpoint_metadata_cache(cached_global_metadata: Metadata = None):
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
    reuse_metadata_obj: Optional[Metadata] = None,
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
        enable_cache (bool, optional): Flag to enable caching of checkpoint metadata. When True,
            previously saved metadata can be reused to speed up subsequent saves.
        metadata_cache (CheckpointMetadataCache, optional): Custom metadata cache instance to use
            for storing and retrieving checkpoint metadata. If not provided, the global cache will be used.

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

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    dist_wrapper = _DistWrapper(process_group, True, coordinator_rank)
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
        if reuse_metadata_obj is not None:
            # `--ckpt-metadata`: the full global metadata is supplied, so there is
            # nothing to build. Skip BOTH the `verify_global_md_reuse` all_reduce
            # and the `gather_object(local_plan)` entirely — planning becomes
            # purely local. The prepared metadata is written verbatim in finalize.
            logger.debug(f"rank: {rank}, reusing prepared metadata - skipping plan gather")
            global_metadata = None
        else:
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
                logger.debug(
                    f"rank: {rank}, Passed cached global metadata, {global_md_verify_reuse}"
                )
                global_metadata = None
        local_plan = planner.create_decentralized_global_plan(local_plan)
        local_plan = storage_writer.prepare_decentralized_global_plan(local_plan)
        central_plan = local_plan
    else:
        if reuse_metadata_obj is not None:
            # reuse_metadata_obj is only honored in the decentralized-plan branch
            # above. Here (no cache reuse, no decentralized planning) we cannot skip
            # the planning collective. The prepared metadata is still written and
            # validated at finalize, so this is a performance no-op rather than a
            # correctness problem -- but warn so the degradation is not silent.
            logger.warning(
                "reuse_metadata_obj was provided but the planner/storage_writer do "
                "not support decentralized global planning; the planning collective "
                "cannot be skipped this save (the prepared metadata is still written "
                "and validated at finalize)."
            )
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
        all_results = torch.tensor([local_verify_reuse], dtype=torch.int, device='cuda')
        torch.distributed.all_reduce(all_results, op=torch.distributed.ReduceOp.MIN)
        # Check if all reduced results are True
        global_md_verify_reuse = all_results.item() == 1
    else:
        global_md_verify_reuse = False
    return global_md_verify_reuse


def _write_metadata_only(storage_writer: 'FileSystemWriterAsync', metadata: Metadata) -> None:
    """Write a *complete* Metadata (already carrying ``storage_data``) to ``.metadata``.

    Used by the prepared-metadata reuse path: we bypass ``storage_writer.finish``
    because that rebuilds ``storage_data`` from the gathered per-rank write
    results — exactly the gather we are skipping. The prepared metadata already
    describes the (constant) on-disk layout, so we serialize it atomically
    (temp file + rename).
    """
    import os
    import pickle

    if getattr(storage_writer, "use_msc", False):
        import multistorageclient as msc

        path = os.path.join(storage_writer.checkpoint_dir, ".metadata")
        with msc.open(path, "wb") as f:
            pickle.dump(metadata, f)
        return

    fs = storage_writer.fs
    tmp_path = fs.concat_path(storage_writer.path, ".metadata.tmp")
    with fs.create_stream(tmp_path, "wb") as f:
        pickle.dump(metadata, f)
        # Drain Python's userspace buffer before fsync: os.fsync on the fd flushes
        # the OS page cache but not the file object's own buffer, so the pickle
        # tail could otherwise reach disk only on close() -- with no fsync after.
        # (matches the sibling _atomic_pickle_write.)
        f.flush()
        try:
            os.fsync(f.fileno())
        except (AttributeError, OSError):
            os.sync()
    # POSIX rename atomically replaces an existing destination, so we rename
    # directly rather than unlink-then-rename (the latter opens a window in
    # which `.metadata` is absent if the process dies between the two calls).
    metadata_path = storage_writer._get_metadata_path()
    fs.rename(tmp_path, metadata_path)


def _atomic_pickle_write(obj, path: str) -> None:
    """Atomically pickle `obj` to an arbitrary local `path` (temp + os.replace).

    Used by the `--ckpt-metadata` *create* mode to persist the just-built
    complete metadata to a container-local path so future jobs can reuse it.
    """
    import os
    import pickle

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_written_metadata(storage_writer: 'FileSystemWriterAsync') -> Metadata:
    """Read back the complete `.metadata` that `storage_writer.finish` just wrote.

    `finish` fills `storage_data` on an internal copy of the metadata, so the
    only reliable source of the *complete* metadata is the file it produced.
    Used by `--ckpt-metadata` create mode to capture/persist it.
    """
    import pickle

    if getattr(storage_writer, "use_msc", False):
        import os

        import multistorageclient as msc

        with msc.open(os.path.join(storage_writer.checkpoint_dir, ".metadata"), "rb") as f:
            return pickle.load(f)
    with open(storage_writer._get_metadata_path(), "rb") as f:
        return pickle.load(f)


def _find_layout_mismatch(
    write_results: List[WriteResult], prepared_metadata: Metadata
) -> Optional[Tuple[str, str]]:
    """Return ``(fqn, detail)`` for the first divergence between what this rank
    actually wrote and ``prepared_metadata``, or ``None`` if this rank's slice of
    the layout matches exactly (i.e. ``local_layout_matches_prepared``).

    Why this is needed. On the reuse path each rank rewrites its data with offsets
    recomputed from the bytes actually written (``stream.tell()`` in
    ``_write_item``), yet the *prepared* metadata is published verbatim as
    ``.metadata``. That is only correct if this rank's data files still match the
    prepared metadata exactly, so we check BOTH directions:

    * forward -- every item this rank wrote must exist in the prepared metadata at
      the same ``(relative_path, offset, length)``. Tensors are raw-stored so their
      length is fixed by ``(shape, dtype)``, but non-tensor (BYTE_IO) items are
      ``torch.save``-pickled and can drift: e.g. the distributed optimizer stores
      ``param_groups[*].step`` as a Python int (``DistributedOptimizer.state_dict``),
      so when ``step`` crosses a pickle size band (256 / 65536 / 2**31 ...) that
      item -- and everything after it in the file -- shifts.
    * reverse -- every item the prepared metadata places in a file THIS rank wrote
      must actually have been written this save. Otherwise a removed / resharded
      item leaves a *dangling* entry in the published metadata pointing at bytes
      this save never wrote. (A removal in the *middle* of a file is already caught
      by the forward check via its shifted survivors; the reverse check adds the
      cases that shift nothing this rank wrote -- a removal at the *end* of a file,
      or an item resharded to another rank.)

    Purely local (no collective). Its one blind spot -- a rank writing an entirely
    different *set* of files than the prepared metadata expects of it -- would need
    the global write-result gather this path deliberately skips; such gross
    structural divergence is outside the documented reuse contract (same structure /
    world size / dist-ckpt-workers).
    """
    prepared = prepared_metadata.storage_data
    written_indices = set()
    owned_paths = set()
    for write_result in write_results:
        written_indices.add(write_result.index)
        owned_paths.add(write_result.storage_data.relative_path)
        expected = prepared.get(write_result.index)
        actual = write_result.storage_data
        if expected is None:
            return (
                str(write_result.index.fqn),
                "written this save but absent from the prepared metadata "
                "(checkpoint structure changed)",
            )
        # (relative_path, offset, length) is the byte locator the loader uses; a
        # difference in any of them means the prepared metadata mis-locates data.
        if (expected.relative_path, expected.offset, expected.length) != (
            actual.relative_path,
            actual.offset,
            actual.length,
        ):
            return (
                str(write_result.index.fqn),
                f"prepared (path={expected.relative_path}, offset={expected.offset}, "
                f"length={expected.length}) != written (path={actual.relative_path}, "
                f"offset={actual.offset}, length={actual.length})",
            )
    # Reverse direction: an item the prepared metadata locates in a file this rank
    # just wrote, but which this save did NOT write, would remain a dangling entry
    # in the published metadata and load from the wrong bytes.
    for index, sinfo in prepared.items():
        if sinfo.relative_path in owned_paths and index not in written_indices:
            return (
                str(index.fqn),
                f"present in the prepared metadata for this rank's file "
                f"{sinfo.relative_path} (offset={sinfo.offset}, length={sinfo.length}) "
                f"but NOT written this save (item removed or resharded)",
            )
    return None


def save_state_dict_async_finalize(
    storage_writer: 'FileSystemWriterAsync',
    global_metadata: Metadata,
    dist_wrapper: _DistWrapper,
    reuse_metadata_obj: Optional[Metadata] = None,
    create_metadata_path: Optional[str] = None,
) -> Optional[Metadata]:
    """
    Finalization of save_state_dict_async_plan.

    The input arguments are the same as the save_state_dict_async_plan output,
    the `write_results` are retrieved from the storage_writer.

    Args:
        storage_writer (FileSystemWriterAsync): storage writer used for planning
        global_metadata (Metadata): metadata created during planning
        dist_wrapper (_DistWrapper): distributed wrapper created during planning
        reuse_metadata_obj (Metadata, optional): a prepared, *complete* Metadata
            (carrying storage_data) supplied via Megatron's ``--ckpt-metadata``.
            When provided, the finalize SKIPS the ``gather_object(write_results)``
            collective (its only purpose is to rebuild storage_data on the
            coordinator, which this metadata already has) and writes this object
            verbatim as the checkpoint's ``.metadata``. Before publishing, each
            rank cheaply validates that its freshly-written byte layout still
            matches the prepared metadata (``_find_layout_mismatch``); this and
            write-failure detection are folded into a single ``all_reduce`` of a
            3-valued status flag (no per-rank write-result gather). Valid only when
            the structure / world size / dist-ckpt-workers match the run that
            produced the prepared metadata (the caller owns this guarantee), but a
            STALE prepared metadata (a non-tensor item whose serialized length
            drifted, e.g. the optimizer ``step`` crossing a pickle size band) is
            now caught and raises a ``CheckpointException("metadata_reuse", ...)``
            instead of silently writing a mislocating ``.metadata``. On a write
            failure the failing rank raises a ``CheckpointException`` carrying its
            own traceback; because the per-rank result gather is skipped, peer
            ranks raise without that detail (consult the failing rank's logs).
        create_metadata_path (str, optional): when set, persist the complete
            metadata produced by this (collective) save to this path so later
            jobs can reuse it via ``reuse_metadata_obj``.

    Returns:
        Optional[Metadata]: the complete ``Metadata`` object (on all ranks) when
        ``create_metadata_path`` is set, otherwise ``None``.
    """
    # Reuse and create are mutually exclusive: reuse skips the collective save
    # that create needs to build the complete metadata. Reject the combination
    # explicitly rather than silently taking the reuse path (which would drop the
    # create/persist step and return None).
    if reuse_metadata_obj is not None and create_metadata_path is not None:
        raise ValueError(
            "reuse_metadata_obj and create_metadata_path are mutually exclusive: "
            "reuse skips the collective save that create needs to build the metadata."
        )

    write_results = storage_writer.retrieve_write_results()

    if reuse_metadata_obj is not None:
        # `retrieve_write_results` returns a list[WriteResult] on success or a
        # WRAPPED_EXCEPTION (a tuple) if this rank's write raised. We fold TWO local
        # checks into a single 3-valued status reduced with the (tiny) all_reduce
        # this path already does, so the fast path stays collective-cheap:
        #   2 = wrote OK *and* the local on-disk layout matches the prepared metadata
        #   1 = wrote OK *but* the layout drifted from the prepared metadata (stale)
        #   0 = local write failed
        # MIN gives the right precedence across ranks: write failure (0) beats a
        # stale metadata (1) beats all-good (2).
        layout_mismatch: Optional[Tuple[str, str]] = None
        if not isinstance(write_results, list):
            local_status = 0
        else:
            layout_mismatch = _find_layout_mismatch(write_results, reuse_metadata_obj)
            local_status = 2 if layout_mismatch is None else 1
        flag = torch.tensor(
            [local_status], dtype=torch.int, device=torch.cuda.current_device()
        )
        torch.distributed.all_reduce(
            flag, op=torch.distributed.ReduceOp.MIN, group=dist_wrapper.group
        )
        global_status = int(flag.item())

        if global_status == 0:
            # A rank's write failed (pre-existing behavior). The failing rank raises
            # with its own detail; peers raise detail-less (the write-result gather
            # that would carry the traceback is intentionally skipped here).
            if local_status != 0:
                logger.warning(
                    "Prepared-metadata save: a peer rank failed its checkpoint "
                    "write while this rank succeeded; see the failing rank's logs "
                    "for the traceback."
                )
                node_failures = {}
            else:
                node_failures = {dist_wrapper.get_rank(): write_results}
            raise CheckpointException("write", node_failures)

        if global_status == 1:
            # No write failures, but at least one rank's freshly-written byte layout
            # no longer matches the prepared metadata. Publishing it verbatim would
            # mis-address data and load corrupt, so fail fast and do NOT write a
            # stale `.metadata`. The detecting rank reports which item drifted.
            if layout_mismatch is not None:
                fqn, detail = layout_mismatch
                logger.error(
                    f"Prepared metadata is STALE for this save: item '{fqn}' {detail}. "
                    "The prepared `.metadata` no longer describes the on-disk byte "
                    "layout (a non-tensor item's serialized length changed, e.g. the "
                    "optimizer 'step' crossing a pickle size band). Regenerate it with "
                    "--ckpt-metadata-create, or disable --ckpt-metadata."
                )
                node_failures = {dist_wrapper.get_rank(): RuntimeError(detail)}
            else:
                logger.error(
                    "Prepared metadata is STALE: a peer rank detected a byte-layout "
                    "mismatch against the prepared `.metadata`; see that rank's logs. "
                    "Regenerate it with --ckpt-metadata-create, or disable "
                    "--ckpt-metadata."
                )
                node_failures = {}
            raise CheckpointException("metadata_reuse", node_failures)

        # global_status == 2: every rank's layout matches -> safe to publish verbatim.
        if dist_wrapper.is_coordinator:
            write_start = time()
            _write_metadata_only(storage_writer, reuse_metadata_obj)
            logger.debug(f"{time()}, metadata_write (prepared/reused): {time() - write_start}")
        return None

    # Gather the write results that will be saved to the metadata file.
    gather_start = time()
    all_results = dist_wrapper.gather_object(write_results)
    gather_end = time()
    logger.debug(f"{gather_end}, {torch.distributed.get_rank()}, gather: {gather_end-gather_start}")

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
        node_failures = {}

    # Broadcast failure status to all ranks to raise exceptions everywhere if needed.
    # The failure details are only raised on the coordinator.
    failures_occurred = torch.tensor(
        [int(len(node_failures) > 0)],
        dtype=torch.int,
        device=torch.cuda.current_device(),
    )
    torch.distributed.broadcast(
        failures_occurred, src=dist_wrapper.coordinator_rank, group=dist_wrapper.group
    )
    if failures_occurred:
        raise CheckpointException("write", node_failures)

    # `--ckpt-metadata` *create* mode: this was a full (collective) save; the
    # coordinator now holds the complete metadata (storage_data filled by
    # `finish`). Share it with every rank (one-time broadcast) so subsequent
    # saves in this job can reuse it (the caller stashes the returned object as
    # the strategy's `prepared_metadata`), and persist it to the container-local
    # path so future jobs can `--ckpt-metadata` it. Returns the complete metadata
    # on all ranks (or None when not creating).
    if create_metadata_path is not None:
        # NOTE: `storage_writer.finish` fills `storage_data` on an internal
        # `dataclasses.replace` copy, not on our `global_metadata` variable, so
        # we must take the *complete* metadata from the file finish just wrote
        # (the coordinator) rather than from `global_metadata`.
        if dist_wrapper.is_coordinator:
            complete_metadata = _read_written_metadata(storage_writer)
        else:
            complete_metadata = None
        obj = [complete_metadata]
        torch.distributed.broadcast_object_list(
            obj, src=dist_wrapper.coordinator_rank, group=dist_wrapper.group
        )
        complete_metadata = obj[0]
        if dist_wrapper.is_coordinator:
            _atomic_pickle_write(complete_metadata, create_metadata_path)
            logger.debug(f"{time()}, created prepared metadata at {create_metadata_path}")
        return complete_metadata
    return None
