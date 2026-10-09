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

"""State dict saver for PyT Distributed format allowing asynchronous save."""

import dataclasses

# Issue: [B403:blacklist] Consider possible security implications associated with pickle module.
# Severity: Low   Confidence: High
# CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
# More Info: https://bandit.readthedocs.io/en/1.8.3/blacklists/blacklist_imports.html#b403-import-pickle
import pickle  # nosec
from dataclasses import fields
from logging import getLogger
from time import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.checkpoint import CheckpointException
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata
from torch.distributed.checkpoint.planner import SavePlan, SavePlanner
from torch.distributed.checkpoint.utils import _DistWrapper, _is_wrapped_exception

from nvidia_resiliency_ext.shared_utils import semconv, telemetry

from . import _metadata_pickler, _metadata_reuse
from ._metadata_pickler import table as _table

if TYPE_CHECKING:
    from .filesystem_async import FileSystemWriterAsync


logger = getLogger(__name__)


class CheckpointMetadataCache:
    """Global metadata and plans carried from one save to the next, to reuse the metadata.

    Building the global metadata gathers every rank's local plan on the coordinator. With
    `enable_cache`, `save_state_dict_async_plan` avoids that:

    * The first save in a process reuses the metadata of a loaded checkpoint (see
      `set_cached_global_metadata`) if all ranks' local plans write exactly the chunks it lists;
      `verify_global_metadata_reuse` checks that with one all_reduce.
    * Later saves reuse the plans and metadata of the previous save without communicating:
      `enable_cache` promises that the checkpoint structure does not change. Each rank still
      creates its local plan and raises if it differs from the previous one.

    The coordinator writes a shallow copy of the cached metadata on every save, so the cache never
    holds a save's storage_data and never modifies the metadata it was given.

    Attributes:
        global_metadata (Metadata): The global metadata to reuse: the loaded checkpoint's, without
            its storage_data, on every rank until the first save, the previous save's on the
            coordinator afterwards.
        local_plan (SavePlan): This rank's local plan of the previous save, as the planner created
            it; None before the first save in this process.
        central_plan (SavePlan): This rank's global plan of the previous save.
    """

    def __init__(self):
        self.global_metadata: Optional[Metadata] = None
        self.local_plan: Optional[SavePlan] = None
        self.central_plan: Optional[SavePlan] = None

    def set_cached_global_metadata(self, cached_global_metadata: Optional[Metadata]):
        """
        Sets the global metadata of a loaded checkpoint, to be reused if it still applies.

        Every rank must set it. The next save checks whether all ranks' local plans write exactly
        the chunks this metadata lists, and if so reuses it instead of gathering the plans and
        building the global metadata again. The metadata is not modified.

        Args:
            cached_global_metadata (Metadata): The global metadata from a previous checkpoint.
        """
        # Without the loaded storage_data: no save needs it, as finish rebuilds it every save.
        self.global_metadata = (
            dataclasses.replace(cached_global_metadata, storage_data=None)
            if cached_global_metadata is not None
            else None
        )
        self.local_plan = None
        self.central_plan = None

    def _update(
        self,
        local_plan: SavePlan,
        central_plan: SavePlan,
        global_metadata: Optional[Metadata],
        reused: bool,
        is_coordinator: bool,
    ) -> Optional[Metadata]:
        """Record a planned save; return the global metadata the coordinator writes for it."""
        self.local_plan = local_plan
        self.central_plan = central_plan
        if not is_coordinator:
            self.global_metadata = None
            return None
        if not reused:
            self.global_metadata = global_metadata
        elif self.global_metadata is None:
            raise RuntimeError(
                "the coordinator holds no global metadata to reuse; with enable_cache, the "
                "coordinator rank must stay the same between saves"
            )
        # `finish` populates `storage_data` in the `Metadata` object it's given. Create a fresh copy
        # so that the `storage_data` does not persist in memory after `finish`.
        return dataclasses.replace(self.global_metadata)


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


def save_state_dict_async_plan(
    state_dict: STATE_DICT_TYPE,
    storage_writer: "FileSystemWriterAsync",
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    planner: Optional[Union[SavePlanner, DefaultSavePlanner]] = None,
    enable_cache: bool = False,
    metadata_cache: Optional[CheckpointMetadataCache] = None,
) -> Tuple["FileSystemWriterAsync", Union[Metadata, None], _DistWrapper]:
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
        coordinator_rank (int): coordinator rank for planning. Defaults to 0.
        planner (SavePlanner, optional): save planner for torch.distributed.checkpoint format
        enable_cache (bool): Reuse global metadata instead of building it: the first save reuses
            a loaded checkpoint's metadata if it still applies, later saves reuse the previous
            save's plans and metadata, as the checkpoint structure must not change between them
            (a rank whose local plan changed raises). See CheckpointMetadataCache. Defaults to
            False.
        metadata_cache (CheckpointMetadataCache, optional): Custom metadata cache instance to use
            for storing and retrieving checkpoint metadata. If not provided, the global cache will be used.

    Returns:
        tuple: Contains:

            - storage writer (the one passed as input)
            - global metadata to write (on the coordinator; None on other ranks)
            - distributed wrapper used for planning

    The return value of this function should be passed as an input to
    `save_state_dict_async_finalize`.
    """
    global _checkpoint_metadata_cache
    metadata_cache = metadata_cache if metadata_cache is not None else _checkpoint_metadata_cache
    if not enable_cache:
        metadata_cache = None

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    dist_wrapper = _DistWrapper(process_group, True, coordinator_rank)
    if planner is None:
        planner = DefaultSavePlanner()
    assert planner is not None

    global_metadata = None
    logger.debug(f"rank: {rank}, starting state dict save")
    reused = False

    def local_step():
        """Set up the planner and storage writer for this save and create the local plan."""
        assert planner is not None
        # PyTorch 2.4 introduced additional `metadata` argument,
        # we have to reference `is_coordinator` args by name
        planner.set_up_planner(state_dict, is_coordinator=dist_wrapper.is_coordinator)
        storage_writer.set_up_storage_writer(dist_wrapper.is_coordinator)
        return storage_writer.prepare_local_plan(planner.create_local_plan())

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
    if metadata_cache is not None and metadata_cache.local_plan is not None:
        # A previous save in this process planned the same structure: `enable_cache` promises
        # it does not change, so reuse its plan and metadata without communication. The local
        # plan is still created (cheap) to catch a broken promise on this rank.
        logger.debug(f"rank: {rank}, Passed cache reusable")
        local_plan = local_step()
        _check_plan_unchanged(metadata_cache.local_plan, local_plan, rank)
        central_plan = metadata_cache.central_plan
        reused = True
    elif getattr(planner, "can_run_decentralized_global_plan", False) and getattr(
        storage_writer, "can_run_decentralized_global_plan", False
    ):
        local_plan = local_step()
        if metadata_cache is not None:
            # The first save in this process: every rank takes part, with or without metadata.
            reused = verify_global_metadata_reuse(
                metadata_cache.global_metadata, local_plan, dist_wrapper
            )

        if not reused:
            logger.debug(f"rank: {rank}, Passed cache non-reusable")
            with telemetry.span(semconv.SPAN_GROUP_CKPT_PROFILING, "nv.nvrx.ckpt.save.plan_gather"):
                all_local_plans = dist_wrapper.gather_object(local_plan)
            if dist_wrapper.is_coordinator:
                with telemetry.span(
                    semconv.SPAN_GROUP_CKPT_PROFILING, "nv.nvrx.ckpt.save.global_metadata_create"
                ):
                    _, global_metadata = planner.create_global_plan(all_local_plans)
        else:
            logger.debug(f"rank: {rank}, Passed cached global metadata")
        central_plan = storage_writer.prepare_decentralized_global_plan(
            planner.create_decentralized_global_plan(local_plan)
        )
    else:
        # Keep the local plan, so that the next save can compare against it.
        local_plan = None

        def local_step_kept():
            """Run local_step and keep its plan."""
            nonlocal local_plan
            local_plan = local_step()
            return local_plan

        with telemetry.span(
            semconv.SPAN_GROUP_CKPT_PROFILING, "nv.nvrx.ckpt.save.plan_reduce_scatter"
        ):
            central_plan = dist_wrapper.reduce_scatter("plan", local_step_kept, global_step)

    final_plan = planner.finish_plan(central_plan)
    end_plan = time()
    logger.debug(f"rank: {rank}, plan time: {end_plan - start_plan}")
    # Prepare async writing of tensors.
    # The `storage_writer` will store the information about tensors it needs to save
    start = time()
    storage_writer.prepare_write_data(final_plan, planner)
    end = time()
    logger.debug(f"{time()} rank: {rank}, write(async) time: {end - start}")
    if metadata_cache is not None:
        global_metadata = metadata_cache._update(
            local_plan, central_plan, global_metadata, reused, dist_wrapper.is_coordinator
        )
    return storage_writer, global_metadata, dist_wrapper


@telemetry.trace_fn(semconv.SPAN_GROUP_CKPT_PROFILING, "nv.nvrx.ckpt.save.check_plan_unchanged")
def _check_plan_unchanged(previous: SavePlan, current: SavePlan, rank: int) -> None:
    """Raise if this rank's local plan differs from the one of the previous save.

    With `enable_cache`, saves after the first reuse the first save's plans and metadata without
    communicating, trusting that the checkpoint structure does not change. A changed plan would
    otherwise be written with stale metadata.

    Args:
        previous (SavePlan): local plan of the previous save
        current (SavePlan): local plan of this save
        rank (int): this rank, for the error message
    """
    differing = [
        f.name
        for f in fields(current)
        if f.name != "storage_data" and getattr(previous, f.name) != getattr(current, f.name)
    ]
    if not differing:
        return
    detail = ", ".join(differing)
    if "items" in differing:
        new_fqns = {item.index.fqn for item in current.items}
        old_fqns = {item.index.fqn for item in previous.items}
        first_changed = next(
            (new.index.fqn for old, new in zip(previous.items, current.items) if old != new), None
        )
        detail = (
            f"{detail}; {len(previous.items)} items before, {len(current.items)} now; "
            f"added {sorted(new_fqns - old_fqns)[:5]}, removed {sorted(old_fqns - new_fqns)[:5]}, "
            f"first changed {first_changed!r}"
        )
    raise RuntimeError(
        f"rank {rank}: the local save plan differs from the previous save's ({detail}). "
        "enable_cache requires the checkpoint structure to stay the same between saves."
    )


@telemetry.trace_fn(
    semconv.SPAN_GROUP_CKPT_PROFILING, "nv.nvrx.ckpt.save.verify_global_metadata_reuse"
)
def verify_global_metadata_reuse(
    loaded_metadata: Optional[Metadata],
    local_plan: SavePlan,
    dist_wrapper: _DistWrapper,
) -> bool:
    """
    Verifies that the metadata of a loaded checkpoint can be reused as this save's global metadata.

    It can if all ranks' local plans write exactly the chunks the metadata lists, with the same
    types, global sizes and properties; see `_metadata_reuse`. Collective: every rank of the
    planning group must call it, also ranks without loaded metadata, which vote against reuse.

    Args:
        loaded_metadata (Metadata, optional): The global metadata of the loaded checkpoint.
        local_plan (SavePlan): This rank's local save plan.
        dist_wrapper (_DistWrapper): distributed wrapper created during planning

    Returns: True iff the global metadata reuse is possible.

    """
    rank, world_size = dist_wrapper.rank, dist_wrapper.get_world_size()
    if loaded_metadata is None:
        votes = [1] + [0] * _metadata_reuse.LANES
    else:
        votes = _metadata_reuse.votes(local_plan, loaded_metadata, rank, world_size)
    summed = torch.tensor(votes, dtype=torch.int64, device=torch.cuda.current_device())
    with telemetry.span(
        semconv.SPAN_GROUP_CKPT_PROFILING,
        "nv.nvrx.ckpt.save.verify_global_metadata_reuse_all_reduce",
    ):
        torch.distributed.all_reduce(summed, group=dist_wrapper.group)
    reuse = _metadata_reuse.can_reuse(summed.tolist())
    if loaded_metadata is not None and dist_wrapper.is_coordinator:
        logger.info(
            "reusing the loaded checkpoint's global metadata"
            if reuse
            else "the loaded checkpoint's global metadata doesn't match the save plans; "
            "building new global metadata"
        )
    return reuse


def _encode_write_results(write_results) -> bytes:
    """This rank's write results for the coordinator: a table if it can, else pickled."""
    if isinstance(write_results, list) and _metadata_pickler.writes_tables():
        try:
            return _table.encode(write_results)
        except _table.Unencodable:
            logger.debug(
                "write results unencodable as a table; sending them pickled", exc_info=True
            )
        except Exception:
            logger.warning("encoding the write results as a table failed", exc_info=True)
    return pickle.dumps(write_results)


def _gather_payloads(payload: bytes, dist_wrapper: _DistWrapper) -> Optional[np.ndarray]:
    """Collective: the ranks' payloads on the coordinator, one zero-padded row per rank.

    Rows are a multiple of 8 bytes wide, so a table's int64s are aligned and every row holds the
    8 bytes that tell a table from a pickle.
    """
    width = len(payload)
    device = torch.cuda.current_device() if dist_wrapper.use_dist else None
    if dist_wrapper.use_dist:
        widest = torch.tensor([width], dtype=torch.int64, device=device)
        torch.distributed.all_reduce(
            widest, op=torch.distributed.ReduceOp.MAX, group=dist_wrapper.group
        )
        width = int(widest.item())
    width = -(-width // 8) * 8
    row = np.zeros(width, dtype=np.uint8)
    row[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    if not dist_wrapper.use_dist:
        return row[None, :]
    rows = None
    if dist_wrapper.is_coordinator:
        rows = torch.empty((dist_wrapper.get_world_size(), width), dtype=torch.uint8, device=device)
    torch.distributed.gather(
        torch.from_numpy(row).to(device),
        list(rows) if rows is not None else None,
        dst=getattr(dist_wrapper, "global_coordinator_rank", dist_wrapper.coordinator_rank),
        group=dist_wrapper.group,
    )
    return rows.cpu().numpy() if rows is not None else None


def _decode_payloads(rows: np.ndarray) -> Tuple[np.ndarray, Dict[int, Any]]:
    """Which ranks sent a table, and what each other rank pickled: its write results or exception."""
    is_table = rows[:, :8].copy().view("<i8")[:, 0] == _table.MAGIC
    # Unpickles only what this job's ranks pickled, as dist.gather_object does.
    pickled = {
        int(rank): pickle.loads(rows[rank].tobytes())  # nosec B301
        for rank in np.flatnonzero(~is_table)
    }
    return is_table, pickled


def _writes_rows(storage_writer) -> bool:
    """Whether storage_writer can write storage_data from the gathered tables: a fast .metadata
    writer, and a FileSystemWriterAsync whose finish a subclass doesn't override."""
    from .filesystem_async import FileSystemWriterAsync

    return (
        isinstance(storage_writer, FileSystemWriterAsync)
        and type(storage_writer).finish is FileSystemWriterAsync.finish
        and _metadata_pickler.writes_tables()
    )


def save_state_dict_async_finalize(
    storage_writer: "FileSystemWriterAsync",
    global_metadata: Metadata,
    dist_wrapper: _DistWrapper,
) -> None:
    """
    Finalization of save_state_dict_async_plan.

    The input arguments are the same as the save_state_dict_async_plan output,
    the `write_results` are retrieved from the storage_writer. Each rank sends them to the
    coordinator as a table (see `_metadata_pickler.table`) if it can, else pickled.

    Args:
        storage_writer (FileSystemWriterAsync): storage writer used for planning
        global_metadata (Metadata): metadata created during planning
        dist_wrapper (_DistWrapper): distributed wrapper created during planning

    Returns: None
    """
    write_results = storage_writer.retrieve_write_results()

    # Gather the write results that will be saved to the metadata file.
    gather_start = time()
    with telemetry.span(semconv.SPAN_GROUP_CKPT_PROFILING, "nv.nvrx.ckpt.save.finalize_gather"):
        rows = _gather_payloads(_encode_write_results(write_results), dist_wrapper)
    gather_end = time()
    logger.debug(
        f"{gather_end}, {torch.distributed.get_rank()}, gather: {gather_end - gather_start}"
    )

    # Store the metadata on coordinator rank
    if dist_wrapper.is_coordinator:
        is_table, pickled = _decode_payloads(rows)
        node_failures = {rank: r for rank, r in pickled.items() if _is_wrapped_exception(r)}
        if len(node_failures) == 0:
            assert global_metadata is not None
            with telemetry.span(
                semconv.SPAN_GROUP_CKPT_PROFILING, "nv.nvrx.ckpt.save.metadata_write"
            ):
                write_start = time()
                if is_table.all() and _writes_rows(storage_writer):
                    storage_writer._finish(global_metadata, [], storage_rows=rows)
                else:
                    results = [
                        _table.to_write_results(_table.decode(row)) if sent_table else pickled[rank]
                        for rank, (row, sent_table) in enumerate(zip(rows, is_table))
                    ]
                    storage_writer.finish(global_metadata, results)
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
    with telemetry.span(semconv.SPAN_GROUP_CKPT_PROFILING, "nv.nvrx.ckpt.save.broadcast_failures"):
        torch.distributed.broadcast(
            failures_occurred, src=dist_wrapper.coordinator_rank, group=dist_wrapper.group
        )
    if failures_occurred:
        raise CheckpointException("write", node_failures)
