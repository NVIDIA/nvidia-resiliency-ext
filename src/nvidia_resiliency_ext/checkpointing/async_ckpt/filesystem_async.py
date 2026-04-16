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

""" Storage writer for PyT Distributed format allowing asynchronous save. """
import dataclasses
import hashlib
import inspect
import logging
import os

# Issue: [B403:blacklist] Consider possible security implications associated with pickle module.
# Severity: Low   Confidence: High
# CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
# More Info: https://bandit.readthedocs.io/en/1.8.3/blacklists/blacklist_imports.html#b403-import-pickle
import pickle  # nosec
import queue
import threading
from functools import partial
from heapq import heappop, heappush
from itertools import chain
from operator import itemgetter
from time import time
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import torch
from torch import multiprocessing as mp
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.api import WRAPPED_EXCEPTION, _wrap_exception
from torch.distributed.checkpoint.filesystem import DEFAULT_SUFFIX, _StoragePrefix, _write_item
from torch.distributed.checkpoint.metadata import Metadata

try:
    from torch.distributed.checkpoint.filesystem import _StorageWriterTransforms
except ImportError:
    _StorageWriterTransforms = Any

from torch.distributed.checkpoint.planner import SavePlan, SavePlanner, WriteItem, WriteItemType
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future

try:
    import psutil

    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

from ..utils import _disable_gc
from .core import PersistentAsyncCaller

logger = logging.getLogger(__name__)

WriteBucket = Tuple[str, str, Tuple[list, list]]  # represents writes to a single file

_results_queue = None


class ConsistentDataIdentifier:
    """Identifier for consistent data structure stored in worker cache.

    This allows passing a lightweight identifier instead of pickling
    the entire data structure (which includes IPC handles) across process boundaries.
    """

    def __init__(self, key: str):
        self.key = key


def _compute_data_structure_key_from_plan(items: List[WriteItem]) -> str:
    """Compute a hash key based on plan items only (no data resolution needed).

    This creates a deterministic key from plan metadata that's available without
    resolving the actual tensor data.

    Args:
        items: List of WriteItem from the plan

    Returns:
        Hex-digest string key representing the data structure
    """
    structure_info = []

    for item in items:
        # Include item metadata that defines the structure
        item_info = (
            item.index.fqn,  # Fully qualified name
            item.type,  # WriteItemType (BYTE_IO or TENSOR)
        )

        # Include metadata from plan (available without resolving data)
        if item.tensor_data is not None:
            # Use tensor metadata from the plan
            data_info = (
                tuple(item.tensor_data.chunk.sizes),  # Tensor chunk shape
                str(item.tensor_data.properties.dtype),  # Data type
            )
        else:
            # For non-tensor data (BYTE_IO), use placeholder
            data_info = (("BYTE_IO",), "BYTE_IO")
        structure_info.append((item_info, data_info))

    # Use SHA-256 for collision resistance and cross-process stability
    # (Python's built-in hash() is randomized per-process and collision-prone)
    return hashlib.sha256(str(structure_info).encode()).hexdigest()


@_disable_gc()
def get_write_results_queue(mp_mode: str = 'spawn') -> mp.Queue:
    """Get or create a multiprocessing queue for write results.

    Args:
        mp_mode (str): Multiprocessing context mode. Defaults to 'spawn'.

    Returns:
        mp.Queue: Queue for collecting write results.
    """
    global _results_queue
    if _results_queue is None:
        ctx = mp.get_context(mp_mode)
        _results_queue = ctx.Manager().Queue()
    return _results_queue


class FileSystemWriterAsync(FileSystemWriter):
    """
    Async-enabled implementation of FileSystemWriter using file I/O.

    This class does not spawn the async process itself but relies on an external async mechanism.

    **Flow:**

    1. Call `write_data`
    2. Externally start an async process with `get_save_function_and_args` and its arguments.
    3. The async function calls `write_preloaded_data_multithread` (threads) or
       `write_preloaded_data_multiproc` (processes) across multiple workers.
    4. Once saving is finalized on all ranks, call `super().finish` with the results stored in `self.writer_result`.

    **Note:** Step (3) can also be executed synchronously.

    Currently, it is assumed that a separate writer is created for each ckpt save
    (intermediate state is stored as writer attributes).
    """

    # Class-level cache to track identifiers that have been sent to worker across instances
    _cached_identifiers: set = set()

    # Training-side shm tensor cache: keeps shm tensors alive (and reuses allocations) across
    # checkpoints.  Key: same SHA-256 as ConsistentDataIdentifier.
    # Value: (gpu_items, shm_tensors) where shm_tensors are individual CPU shared-memory
    # tensors (one per GPU tensor, each with its own independent storage).
    _shm_tensor_cache: ClassVar[Dict[str, Tuple[List, List]]] = {}

    # Blocking drain registered by AsyncCallsQueue when cpu_shm_mode=True.
    # Called in prepare_write_data before the first copy_() into a reused shm tensor,
    # so any prior write that is still reading from those tensors completes first.
    _shm_drain_callback: ClassVar[Optional[Callable[[], None]]] = None

    def __init__(
        self,
        path: Union[str, os.PathLike],
        *args,
        separation_hint: Optional[str] = None,
        use_msc: bool = False,
        is_multiproc_io: bool = False,
        use_cached_data_structure: bool = False,
        use_cpu_shm_for_gpu_tensors: bool = False,
        **kwargs,
    ):
        self.checkpoint_dir = path
        self.use_msc = use_msc
        self.open_file = kwargs.pop("open_file", open)  # for overriding in tests

        super().__init__(path, *args, **kwargs)
        if not self.single_file_per_rank:
            raise NotImplementedError(
                'single_file_per_rank flag not supported for FileSystemWriterAsync'
            )

        self.can_run_decentralized_global_plan: bool = True

        # Intermediate state between preparation and finalization
        self.has_data_to_write: bool = False
        self.results_queue: Optional[mp.Queue] = None
        self.separation_hint = separation_hint
        self.use_cached_data_structure = use_cached_data_structure
        self.consistent_data_identifier: Optional[ConsistentDataIdentifier] = None
        # When this flag is True, the FileWriter can create multiple child processes
        # to parallelize File IO in the background async checkpoint process.
        # Setting this flag to False (default) uses multi-threading to parallelize File IO.
        # Note: multi-proc IO requires is_daemon=False on PersistentAsyncCaller (AsyncCallsQueue),
        # whereas the default multithreaded IO is compatible with is_daemon=True (the default).
        self.is_multi_proc_io = is_multiproc_io
        # Use CPU shared-memory tensors instead of GPU IPC fabric handles.
        # Avoids cuMemImportFromShareableHandle on MNNVL systems where NVLink fabric
        # resources are exhausted by the training ranks.
        self.use_cpu_shm_for_gpu_tensors = use_cpu_shm_for_gpu_tensors

    def prepare_write_data(self, plan: SavePlan, planner: SavePlanner) -> None:
        """
        First stage of async saving. Resolve data and store in compact format.

        Separates data into GPU tensors (potentially cacheable), CPU tensors (always fresh),
        and ByteIO (always fresh). Bucket creation is deferred to `preload_tensors` so that
        it can run in the persistent worker process and take advantage of the data cache.

        Args:
            plan (SavePlan): save plan generated by the PyT Distributed compatible planner
            planner (SavePlanner): save planner used to resolve the bytes and tensor data

        Returns: None, but stores the resolved plan data in instance attributes
        """
        start = time()
        logger.debug(f"thread_count: {self.thread_count}, time: {start}")
        if self.separation_hint:
            assert (
                self.thread_count > 1
            ), "thread_count must be at least 2 if separation_hint is provided"

        def _clone_or_dequantize_if_needed(ten: torch.Tensor):
            """Clone if we detect incontiguous storage for CPU tensors.

            Makes sure we perform a `clone` only if we detect incontiguous storage,
            so that we don't blow up host memory unnecessarily.

            For GPU tensors, returns as-is since they'll be moved to CPU in preload_tensors.

            Returns:
                (tensor, was_dequantized): the processed tensor and a bool indicating
                    whether dequantize() was called. Tracked explicitly because some
                    frameworks (e.g. TransformerEngine MXFP8) use a fake bfloat16 dtype
                    for quantized tensors, making dtype-based detection unreliable.
            """
            ten = ten.detach()
            if ten.device.type != "cpu":
                # We call ``dequantize`` if we detect a quantized tensor on GPU.
                # This is a workaround to avoid the issue of quantized tensors not being supported by the async writer.
                if ten.device.type == "cuda" and "dequantize" in type(ten).__dict__:
                    ten = ten.dequantize()
                    # GPU tensors will be moved to CPU in preload_tensors
                    return ten, True
                # GPU tensors will be moved to CPU in preload_tensors
                return ten, False
            # For CPU tensors, clone if they are views to ensure contiguous storage
            is_view = ten.untyped_storage().size() != ten.numel() * ten.itemsize
            return (ten.clone() if is_view else ten), False

        def resolve_data(items):
            resolved = []
            dequantized_flags = []
            for item in items:
                data = planner.resolve_data(item)
                # Apply cloning/dequantize logic during resolution
                if isinstance(data, torch.Tensor):
                    data, was_dequantized = _clone_or_dequantize_if_needed(data)
                else:
                    was_dequantized = False
                resolved.append(data)
                dequantized_flags.append(was_dequantized)
            return resolved, dequantized_flags

        # Separate items by type: only GPU tensors can be cached via IPC
        # CPU tensors and ByteIO must be resolved fresh (cannot use IPC)
        tensor_items = [item for item in plan.items if item.type != WriteItemType.BYTE_IO]
        byte_io_items = [item for item in plan.items if item.type == WriteItemType.BYTE_IO]

        # Helper to separate resolved tensors into cacheable (GPU) vs uncached buckets.
        # Dequantized tensors are tracked explicitly via dequantized_flags because
        # some frameworks (e.g. TransformerEngine MXFP8) report bfloat16 as the dtype
        # for quantized tensors, making dtype-based detection unreliable.
        def separate_cacheable(items, resolved_data, dequantized_flags, include_dequantized=False):
            """Separate tensor items into cacheable (GPU) and uncached categories.

            For the GPU IPC path (include_dequantized=False): dequantized tensors are
            excluded because dequantize() produces a new temporary GPU allocation each
            checkpoint, so a cached IPC handle from the previous step would be stale.

            For the CPU shm path (include_dequantized=True): dequantized GPU tensors are
            included — we copy values in from whatever GPU tensor the current step produces,
            regardless of whether it was dequantized.
            """
            gpu_items, gpu_data = [], []
            uncached_items, uncached_data = [], []

            for item, data, was_dequantized in zip(items, resolved_data, dequantized_flags):
                if isinstance(data, torch.Tensor) and data.device.type == "cpu":
                    uncached_items.append(item)
                    uncached_data.append(data)
                elif was_dequantized and not include_dequantized:
                    uncached_items.append(item)
                    uncached_data.append(data)
                else:
                    gpu_items.append(item)
                    gpu_data.append(data)

            return (gpu_items, gpu_data), (uncached_items, uncached_data)

        # Handle GPU tensor caching (only GPU tensors can benefit from IPC or shm)
        # Uncached tensors: CPU tensors always; dequantized GPU tensors only on GPU IPC path
        # (on the CPU shm path, dequantized GPU tensors ARE included)
        if (self.use_cached_data_structure or self.use_cpu_shm_for_gpu_tensors) and tensor_items:
            key = _compute_data_structure_key_from_plan(tensor_items)
            cache_exists = key in FileSystemWriterAsync._cached_identifiers

            # Always resolve tensors to separate uncached tensors (which can't be cached)
            resolved_tensors, dequantized_flags = resolve_data(tensor_items)
            (gpu_items, gpu_data), (uncached_items, uncached_data) = separate_cacheable(
                tensor_items,
                resolved_tensors,
                dequantized_flags,
                include_dequantized=self.use_cpu_shm_for_gpu_tensors,
            )

            if gpu_items and self.use_cpu_shm_for_gpu_tensors:
                # --- CPU shared-memory path ---
                # D2H is done here (training side) into per-tensor CPU shared-memory tensors
                # so the worker subprocess never needs CUDA IPC / fabric handles.
                # Each tensor gets its own independent share_memory_() allocation.
                if key in FileSystemWriterAsync._shm_tensor_cache:
                    # Drain any in-flight write BEFORE overwriting these shm tensors.
                    # The worker for checkpoint N may still be iterating over the same
                    # shared-memory buffers; without this drain the D2H copy below races
                    # with that read and silently corrupts checkpoint N.
                    # AsyncCallsQueue.register_shm_drain_callback sets this callback when
                    # cpu_shm_mode=True; callers that bypass AsyncCallsQueue must drain
                    # themselves before calling prepare_write_data.
                    if FileSystemWriterAsync._shm_drain_callback is not None:
                        FileSystemWriterAsync._shm_drain_callback()
                    _, existing_shm = FileSystemWriterAsync._shm_tensor_cache[key]
                    for i, (shm_t, gpu_t) in enumerate(zip(existing_shm, gpu_data)):
                        shm_t.copy_(gpu_t, non_blocking=True)
                        # Periodically sync to avoid exhausting CUDA's pageable staging
                        # pages (shm tensors are not pinned so copy_ is not a pinned DMA).
                        if (i + 1) % 8 == 0:
                            torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    logger.debug(
                        f"D2H'd {len(existing_shm)} shm tensors into reused allocations "
                        f"(key={key})"
                    )
                else:
                    # First checkpoint: allocate one independent shm tensor per GPU tensor
                    # and D2H-copy the current values in.
                    shm_tensors = []
                    total_bytes = 0
                    for i, t in enumerate(gpu_data):
                        tc = t.contiguous()
                        shm_t = torch.empty(tc.shape, dtype=tc.dtype).share_memory_()
                        shm_t.copy_(tc, non_blocking=True)
                        shm_tensors.append(shm_t)
                        total_bytes += shm_t.nbytes
                        # Periodically sync to avoid exhausting CUDA's pageable staging pages
                        if (i + 1) % 8 == 0:
                            torch.cuda.synchronize()

                    FileSystemWriterAsync._shm_tensor_cache[key] = (gpu_items, shm_tensors)
                    logger.debug(
                        f"Allocated {len(shm_tensors)} shm tensors ({total_bytes} bytes total) "
                        f"and D2H'd GPU values (key={key})"
                    )

                _, shm_tensors = FileSystemWriterAsync._shm_tensor_cache[key]
                self.consistent_data_identifier = ConsistentDataIdentifier(key)
                if cache_exists:
                    # Subsequent checkpoint: D2H already done into reused shm buffer above.
                    # Worker already has the shm tensor references cached; send None to
                    # signal it should reuse them (values are fresh after the D2H above).
                    self.cached_tensor_data = None
                    logger.debug(
                        f"Reusing worker-cached shm tensors (key={key}), "
                        f"{len(uncached_items)} uncached tensors passed fresh"
                    )
                else:
                    # First checkpoint: D2H already done above, synchronize now.
                    torch.cuda.synchronize()
                    # Send shm tensors to worker so it can cache them.
                    self.cached_tensor_data = (gpu_items, shm_tensors)
                    FileSystemWriterAsync._cached_identifiers.add(key)
                    logger.debug(
                        f"Sending {len(shm_tensors)} shm tensors to worker (key={key}), "
                        f"{len(uncached_items)} uncached tensors passed fresh"
                    )
            elif cache_exists:
                # --- original GPU IPC path, reuse ---
                self.consistent_data_identifier = ConsistentDataIdentifier(key)
                self.cached_tensor_data = None  # Signal to reuse cached data
                logger.debug(
                    f"Reusing cached GPU tensors (key={key}), "
                    f"resolved {len(uncached_items)} uncached tensors fresh"
                )
            elif gpu_items:
                # --- original GPU IPC path, first time ---
                self.consistent_data_identifier = ConsistentDataIdentifier(key)
                self.cached_tensor_data = (gpu_items, gpu_data)
                FileSystemWriterAsync._cached_identifiers.add(key)
                logger.debug(
                    f"Caching {len(gpu_items)} GPU tensors (key={key}), "
                    f"{len(uncached_items)} uncached tensors passed fresh"
                )
            else:
                # No GPU tensors to cache; skip caching entirely
                self.consistent_data_identifier = None
                self.cached_tensor_data = None
                logger.debug(
                    f"No GPU tensors to cache (key={key}), "
                    f"{len(uncached_items)} uncached tensors passed fresh"
                )

            # When using CPU shm path, also D2H-copy any dequantized GPU tensors in
            # uncached_data so they don't require CUDA IPC / fabric handles in the worker.
            if self.use_cpu_shm_for_gpu_tensors and uncached_items:
                needs_sync = False
                new_uncached_data = []
                for t in uncached_data:
                    if isinstance(t, torch.Tensor) and t.is_cuda:
                        new_uncached_data.append(t.to("cpu", non_blocking=True))
                        needs_sync = True
                    else:
                        new_uncached_data.append(t)
                if needs_sync:
                    torch.cuda.synchronize()
                uncached_data = new_uncached_data

            # Uncached tensors are always passed fresh (never cached)
            self.uncached_tensor_data = (uncached_items, uncached_data) if uncached_items else None
        else:
            # No caching - resolve and separate all tensors
            self.consistent_data_identifier = None

            if tensor_items:
                resolved_tensors, dequantized_flags = resolve_data(tensor_items)
                (gpu_items, gpu_data), (uncached_items, uncached_data) = separate_cacheable(
                    tensor_items, resolved_tensors, dequantized_flags
                )
                self.cached_tensor_data = (gpu_items, gpu_data) if gpu_items else None
                self.uncached_tensor_data = (
                    (uncached_items, uncached_data) if uncached_items else None
                )
            else:
                self.cached_tensor_data = None
                self.uncached_tensor_data = None

        # Always resolve ByteIO fresh (cannot use IPC)
        self.byte_io_data = (
            (byte_io_items, resolve_data(byte_io_items)[0]) if byte_io_items else None
        )
        self.storage_plan = plan.storage_data

        # Setup results queue if there's data to write
        self.has_data_to_write = len(plan.items) > 0
        self.results_queue = get_write_results_queue() if self.has_data_to_write else None
        end = time()
        logger.debug(f"prepare_write_data, time: {end - start}")

    @classmethod
    def cleanup_tensor_caches(cls) -> None:
        """Release training-side tensor caches and invalidate the identifier set.

        Must be called whenever the worker process is restarted.  A fresh worker
        has an empty ``_worker_data_cache``, so both paths need re-priming:

        - **CPU shm path** (``use_cpu_shm_for_gpu_tensors=True``): clears
          ``_shm_tensor_cache`` so the next checkpoint re-allocates shm tensors
          and sends them to the worker instead of sending ``None`` (which would
          cause a worker-side cache miss).
        - **GPU IPC path** (``use_cached_data_structure=True``): clearing
          ``_cached_identifiers`` forces the next checkpoint to re-send the GPU
          tensor IPC handles rather than assuming the worker already has them.
        """
        if cls._shm_tensor_cache:
            logger.info(f"Clearing shm tensor cache ({len(cls._shm_tensor_cache)} entries)")
            cls._shm_tensor_cache.clear()
        cls._cached_identifiers.clear()

    @classmethod
    def register_shm_drain_callback(cls, fn: Optional[Callable[[], None]]) -> None:
        """Register (or clear with None) the blocking drain called before reusing shm tensors.

        AsyncCallsQueue registers this when cpu_shm_mode=True so that
        prepare_write_data drains any in-flight write before overwriting shm tensors.
        """
        cls._shm_drain_callback = fn

    def get_save_function_and_args(self) -> Tuple[Optional[Callable], Optional[Callable], List]:
        """
        Get function that saves the data to storage along with its arguments.
        Allows the external caller to apply the save function synchronously or asynchronously.

        Returns: None (if there is nothing to write on this rank) or a tuple of:
            1) the function that saves the data.
            2) the function that stages the GPU tensors to a destination for async checkpointing.
               This function should be self-contained.
            3) arguments to that function in 1).
        """
        if not self.has_data_to_write:
            return None, None, []

        if self.use_msc:
            import multistorageclient as msc

            open_file = msc.open
        else:
            open_file = self.open_file

        transform_list = [self.transforms] if hasattr(self, 'transforms') else []

        # Format: (identifier, (separation_hint, cached_tensor_data,
        # uncached_tensor_data, byte_io_data, thread_count, storage_plan))
        # identifier is None when caching is disabled
        # uncached_tensor_data is always passed fresh (like ByteIO), never cached
        data_to_pass = (
            self.consistent_data_identifier,
            (
                self.separation_hint,
                self.cached_tensor_data,
                self.uncached_tensor_data,
                self.byte_io_data,
                self.thread_count,
                self.storage_plan,
            ),
        )

        # Select write function based on IO mode
        if self.is_multi_proc_io:
            write_func = partial(
                self.write_preloaded_data_multiproc, transform_list, self.use_msc, open_file
            )
        else:
            write_func = partial(
                self.write_preloaded_data_multithread, transform_list, self.use_msc, open_file
            )

        preload_fn = partial(self.preload_tensors, (str(self.checkpoint_dir), data_to_pass), True)

        return (
            write_func,
            preload_fn,
            [torch.distributed.get_rank(), None, self.results_queue],
        )

    @staticmethod
    def preload_tensors(resolved_plan_data: Tuple, non_blocking=True) -> List[WriteBucket]:
        """
        Creates write_buckets and preloads tensors to host memory.

        This runs in the persistent worker process. Bucket creation is done here
        (not in prepare_write_data) so that cached GPU tensor data stored in the
        worker process can be retrieved and reused without re-pickling.

        Args:
            resolved_plan_data (Tuple): Tuple containing
                (checkpoint_dir, (identifier, data_structure)) where:
                - identifier: ConsistentDataIdentifier (caching) or None
                - data_structure: (separation_hint, cached_tensor_data,
                  uncached_tensor_data, byte_io_data, thread_count, storage_plan)
            non_blocking (bool, optional): Enable pinned D2H memcpy. Default is True.

        Returns:
            List[WriteBucket]: List of write buckets with tensors moved to CPU
        """
        start = time()
        logger = logging.getLogger(__name__)

        checkpoint_dir, data_or_identifier = resolved_plan_data

        # Helper to combine GPU tensor, uncached tensor, and ByteIO data
        def combine_data(gpu_tensor_data, uncached_tensor_data, byte_io_data):
            items, resolved = [], []
            for data in [gpu_tensor_data, uncached_tensor_data, byte_io_data]:
                if data:
                    items.extend(data[0])
                    resolved.extend(data[1])
            return items, resolved

        # Parse data structure: (identifier, (separation_hint, cached_tensor_data,
        # uncached_tensor_data, byte_io_data, thread_count, storage_plan))
        # identifier is None when disabled, ConsistentDataIdentifier when enabled
        identifier, data_structure = data_or_identifier
        (
            separation_hint,
            cached_tensor_data,
            uncached_tensor_data,
            byte_io_data,
            thread_count,
            storage_plan,
        ) = data_structure

        if isinstance(identifier, ConsistentDataIdentifier):
            # Caching enabled: get or cache GPU tensor data in the worker process
            # Uncached tensors (CPU tensors, or dequantized on the GPU IPC path) are NOT cached
            key = identifier.key
            if cached_tensor_data is not None:
                PersistentAsyncCaller._worker_data_cache[key] = cached_tensor_data
                logger.debug(f"Worker cached GPU tensors (key={key})")
            elif key in PersistentAsyncCaller._worker_data_cache:
                cached_tensor_data = PersistentAsyncCaller._worker_data_cache[key]
                logger.debug(f"Worker retrieved cached GPU tensors (key={key})")
            else:
                raise RuntimeError(f"Worker cache miss for key {key}. Worker may have restarted.")
        # else: identifier is None, no caching needed

        items, resolved_data = combine_data(cached_tensor_data, uncached_tensor_data, byte_io_data)

        logger.debug(f"preload_tensors: thread_count: {thread_count}, time: {start}")

        # Create buckets from items
        bins = thread_count // 2 if separation_hint is not None else thread_count
        item_buckets = _split_by_size_and_type(bins, items)
        logger.debug(f"preload_tensors: bucket_prep, time: {time() - start}")

        # Create a mapping from items to resolved data
        item_to_data = {id(item): data for item, data in zip(items, resolved_data)}

        file_count = 0

        def gen_file(prefix=""):
            nonlocal file_count
            file_name = f"{prefix}{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
            file_count += 1
            return file_name

        # Build write_buckets with items grouped by file, assigning one per worker
        write_buckets = []
        for group_name, group_buckets in _split_by_separation_hint(
            item_buckets, separation_hint
        ).items():
            for bucket in group_buckets:
                bytes_data = []
                tensor_data = []
                for item in bucket:
                    data = item_to_data[id(item)]
                    if item.type == WriteItemType.BYTE_IO:
                        bytes_data.append((item, data))
                    else:
                        # Tensor data (GPU or CPU) - already cloned if needed
                        tensor_data.append((item, data))

                if len(bytes_data) > 0 or len(tensor_data) > 0:
                    file_name = gen_file(prefix=group_name)
                    write_buckets.append(
                        (
                            os.path.join(checkpoint_dir, file_name),
                            file_name,
                            (bytes_data, tensor_data),
                        )
                    )

        # Move GPU tensors to CPU.  For the shm path, tensors are already on CPU so the
        # .to("cpu") branch is skipped entirely — no D2H, no sync needed.
        result: List[WriteBucket] = []
        needs_sync = False
        for bucket in write_buckets:
            bucket_path, bucket_key, bucket_data = bucket
            bytes_data, tensor_data = bucket_data
            tensor_list = []
            for item, tensor in tensor_data:
                if tensor.is_cuda:
                    needs_sync = True
                    tensor_list.append((item, tensor.to("cpu", non_blocking=non_blocking)))
                else:
                    tensor_list.append((item, tensor))
            result.append((bucket_path, bucket_key, (bytes_data, tensor_list)))

        if non_blocking and needs_sync:
            torch.cuda.synchronize()

        end = time()
        logger.debug(f"preload_tensors: D2H and bucket creation, time: {end - start}")
        return result

    @staticmethod
    def _initialize_write_execution(rank: int) -> Tuple[logging.Logger, float, dict]:
        """
        Common initialization for write execution.

        Args:
            rank (int): training rank

        Returns:
            Tuple[logging.Logger, float, dict]: logger, start time, and initialized results dict
        """
        logger = logging.getLogger(__name__)
        w_start = time()
        write_results_or_exc: Union[dict, Exception] = dict()
        return logger, w_start, write_results_or_exc

    @staticmethod
    def _build_worker_kwargs(
        worker_idx: int, write_bucket: WriteBucket, use_msc: bool, worker_type: str, **extra_kwargs
    ) -> dict:
        """
        Build kwargs for worker (thread or process).

        Args:
            worker_idx (int): index of the worker
            write_bucket (WriteBucket): data to write
            use_msc (bool): flag to indicate use of multi storage client
            worker_type (str): 'thread' or 'proc'
            **extra_kwargs: additional worker-specific kwargs

        Returns:
            dict: kwargs for the worker
        """
        idx_key = f'local_{worker_type}_idx'
        kwargs = {
            idx_key: worker_idx,
            'write_bucket': write_bucket,
            'use_fsync': True,
        }
        if use_msc:
            kwargs['use_msc'] = use_msc
        kwargs.update(extra_kwargs)
        return kwargs

    @staticmethod
    def _finalize_write_execution(
        global_results_queue: mp.Queue,
        write_results_or_exc: Union[dict, Exception],
        rank: int,
        w_start: float,
        worker_type: str,
        logger: logging.Logger,
    ) -> None:
        """
        Common finalization for write execution.

        Args:
            global_results_queue (mp.Queue): queue to put results
            write_results_or_exc (Union[dict, Exception]): results or exception
            rank (int): training rank
            w_start (float): start time
            worker_type (str): 'MultiProc' or 'MultiThread'
            logger (logging.Logger): logger instance
        """
        global_results_queue.put(write_results_or_exc)
        w_end = time()
        logger.debug(
            f"{worker_type} Background Async worker time to persist: {w_end - w_start} s for rank={rank}"
        )

    @staticmethod
    @_disable_gc()
    def write_preloaded_data_multiproc(
        transform_list: List[_StorageWriterTransforms],
        use_msc: bool,
        open_file: Callable,
        rank: int,
        write_buckets: List[WriteBucket],
        global_results_queue: mp.Queue,
    ) -> None:
        """
        Performs saving data to storage with multiple processes.

        Starts predefined number of processes and uses 2 queues to make sure the results
        are complete:
        - local_results_queue - to send the actual results
        - count_queue - small queue to mark worker as completed

        Using just one queue disallowed proper exception handling.

        This method is meant to be run in a forked subprocess.
        Triggering GC during execution leads to CUDA errors
        (cleaning up tensors owned by the parent process).
        To prevent this, we disable the GC explicitly for this function with _disable_gc.

        Note: requires is_daemon=False on the PersistentAsyncCaller, because daemon
        processes cannot spawn child processes.

        Args:
            transform_list (List[_StorageWriterTransforms]): streaming transforms list
            use_msc (bool): flag to indicate use of multi storage client
            open_file (Callable): file open callable
            rank (int): training rank
            write_buckets (List[WriteBucket]): write plan
            global_results_queue (mp.Queue): mp.Queue to collect Dict[List[WriteResults]]
                (or an Exception) from parallel write processes to the main training process
        Returns: None
        """
        logger, w_start, write_results_or_exc = FileSystemWriterAsync._initialize_write_execution(
            rank
        )

        ctx = mp.get_context('fork')
        local_results_queue = ctx.Queue()
        count_queue = ctx.JoinableQueue()
        p_list = []
        for i, write_bucket in enumerate(write_buckets):
            try:
                current_process = mp.current_process()
                if current_process.daemon:
                    err_msg = "Invalid Setup! User cannot establish a daemon Async worker and then use Multi-Proc File IO."
                    logger.error(err_msg)
                    raise RuntimeError(err_msg)

                count_queue.put(i)
                kwargs = FileSystemWriterAsync._build_worker_kwargs(
                    worker_idx=i,
                    write_bucket=write_bucket,
                    use_msc=use_msc,
                    worker_type='proc',
                    results_queue=local_results_queue,
                    count_queue=count_queue,
                )

                p_list.append(
                    ctx.Process(
                        target=partial(
                            FileSystemWriterAsync.write_preloaded_data_proc,
                            transform_list,
                            open_file,
                        ),
                        kwargs=kwargs,
                    )
                )
            except Exception as e:
                err_msg = f'An error is caught while a proc {i} is created, error: {e}'
                logger.error(err_msg)
                write_results_or_exc = RuntimeError(err_msg)

        if not isinstance(write_results_or_exc, Exception):
            for p in p_list:
                p.start()

            logger.debug('FileSystemWriterAsync: collecting worker results...')

            # To make sure all nodes are completed
            count_queue.join()
            # At this point, all workers completed, so the queue should have exactly
            # `len(write_buckets)` items
            for proc_idx in range(len(write_buckets)):
                try:
                    local_proc_idx, local_results_or_exc = local_results_queue.get()
                except queue.Empty:
                    write_results_or_exc = RuntimeError(
                        'Unexpected empty `local_results_queue`'
                        f' (got only {proc_idx}/{len(write_buckets)} items)'
                    )
                    break
                else:
                    if isinstance(local_results_or_exc, Exception):
                        err_msg = (
                            f"Local process {local_proc_idx} encountered"
                            f" an error: {local_results_or_exc}"
                        )
                        logger.error(err_msg)
                        write_results_or_exc = local_results_or_exc
                        break
                    assert isinstance(local_results_or_exc, list), type(local_results_or_exc)
                    write_results_or_exc[local_proc_idx] = local_results_or_exc
                    p_list[local_proc_idx].join()

            logger.debug('FileSystemWriterAsync: collected worker results successfully')

        FileSystemWriterAsync._finalize_write_execution(
            global_results_queue, write_results_or_exc, rank, w_start, "MultiProc", logger
        )

    @staticmethod
    def _write_bucket_to_storage(
        transform_list: List[_StorageWriterTransforms],
        open_file: Callable,
        write_bucket: WriteBucket,
        use_fsync: bool,
        use_msc: bool,
    ) -> List[WriteResult]:
        """
        Core logic for writing a bucket to storage.

        Args:
            transform_list (List[_StorageWriterTransforms]): streaming transforms list
            open_file (Callable): file open callable
            write_bucket (WriteBucket): data to write to storage
            use_fsync (bool): if True, calls os.fsync at the end of saving
            use_msc (bool): flag to indicate use of multi storage client

        Returns:
            List[WriteResult]: list of write results
        """
        file_name, storage_key, (bytes_data, tensor_data) = write_bucket
        extra_kwargs = {}
        write_fn = _write_item
        if "serialization_format" in inspect.signature(_write_item).parameters:
            from torch.distributed.checkpoint.filesystem import SerializationFormat

            extra_kwargs['serialization_format'] = SerializationFormat.TORCH_SAVE

        if "transforms" in inspect.signature(_write_item).parameters:
            assert len(transform_list) <= 1
            write_fn = partial(_write_item, *transform_list)

        local_results = []
        with open_file(file_name, "wb") as stream:
            for write_item, data in bytes_data:
                local_results.append(
                    write_fn(stream, data, write_item, storage_key, **extra_kwargs)
                )

            for write_item, tensor in tensor_data:
                assert tensor.is_cpu
                local_results.append(
                    write_fn(stream, tensor, write_item, storage_key, **extra_kwargs)
                )

            if use_fsync:
                if use_msc:
                    stream.fsync()
                else:
                    os.fsync(stream.fileno())

        return local_results

    @staticmethod
    @_disable_gc()
    def write_preloaded_data_proc(
        transform_list: List[_StorageWriterTransforms],
        open_file: Callable,
        local_proc_idx: int,
        write_bucket: WriteBucket,
        results_queue: mp.SimpleQueue,
        count_queue: mp.JoinableQueue,
        use_fsync: bool,
        **kwargs,
    ) -> None:
        """
        Performs actual data saving to storage (used by worker processes in multiproc mode).

        Args:
            local_proc_idx (int): index of a local process that performs writing
            write_bucket (WriteBucket): data to write to storage
            results_queue (mp.Queue): queue to return the write results
                to the proxy checkpoint process.
            count_queue (mp.JoinableQueue): queue to marks worker task as completed
            use_fsync (bool): if True, calls os.fsync at the end of saving

        Returns: None, the write results are put into the `results_queue`
        """
        logger = logging.getLogger(__name__)
        logger.debug(f'{local_proc_idx} started')
        mem_before = _process_memory()
        use_msc = kwargs.get('use_msc', False)

        try:
            local_results = FileSystemWriterAsync._write_bucket_to_storage(
                transform_list, open_file, write_bucket, use_fsync, use_msc
            )
            local_output = (local_proc_idx, local_results)
        except Exception as e:
            logger.debug(f'{local_proc_idx} failed')
            local_output = (local_proc_idx, e)

        results_queue.put(local_output)
        # Signal this process is done.
        count_queue.get()
        count_queue.task_done()

        mem_after = _process_memory()
        logger.debug(
            f"{local_proc_idx} consumed: {mem_after - mem_before},"
            f" before: {mem_before}, after: {mem_after}"
        )

    @staticmethod
    @_disable_gc()
    def write_preloaded_data_multithread(
        transform_list: List[_StorageWriterTransforms],
        use_msc: bool,
        open_file: Callable,
        rank: int,
        write_buckets: List[WriteBucket],
        global_results_queue: mp.Queue,
    ) -> None:
        """
        Performs saving data to storage with multiple threads.

        Uses threads (not processes) so that this can run safely inside a daemon process
        without spawning child processes. The last bucket runs on the calling thread to
        avoid thread creation overhead. Uses two queues for worker coordination:
        - local_results_queue - to collect write results from worker threads
        - count_queue - to signal worker completion (get + task_done / join).

        Triggering GC during execution can lead to CUDA errors when tensors are shared.
        To prevent this, we disable the GC explicitly for this function with _disable_gc.

        Args:
            transform_list (List[_StorageWriterTransforms]): streaming transforms list
            use_msc (bool): flag to indicate use of multi storage client for storage access
            open_file (Callable): file open callable
            rank (int): training rank
            write_buckets (List[WriteBucket]): write plan
            global_results_queue (mp.Queue): queue to send Dict[List[WriteResults]]
                (or an Exception) back to the main training process
        Returns: None
        """
        logger = logging.getLogger(__name__)
        w_start = time()
        write_results_or_exc: Union[dict, Exception] = dict()
        local_results_queue: queue.Queue = queue.Queue()
        count_queue: queue.Queue = queue.Queue()
        thread_list: List[threading.Thread] = []

        def check_local_output(local_results_or_exc, local_worker_idx):
            if isinstance(local_results_or_exc, Exception):
                err_msg = (
                    f"Local worker {local_worker_idx} encountered"
                    f" an error: {local_results_or_exc}"
                )
                logger.error(err_msg)
                raise local_results_or_exc

        for i, write_bucket in enumerate(write_buckets):
            try:
                kwargs = {
                    "local_thread_idx": i,
                    "write_bucket": write_bucket,
                    "results_queue": local_results_queue,
                    "count_queue": count_queue,
                    "use_fsync": True,
                }
                if use_msc:
                    kwargs["use_msc"] = use_msc

                # Parallel writers: spawn threads for all but the last bucket
                if i < len(write_buckets) - 1:
                    count_queue.put(i)
                    t = threading.Thread(
                        target=partial(
                            FileSystemWriterAsync.write_preloaded_data, transform_list, open_file
                        ),
                        kwargs=kwargs,
                    )
                    thread_list.append(t)
                else:
                    # Run last bucket on the calling thread (no thread overhead)
                    kwargs['count_queue'] = None
                    kwargs['results_queue'] = None
                    logger.debug('FileSystemWriterAsync: main worker started')
                    local_output = FileSystemWriterAsync.write_preloaded_data(
                        transform_list, open_file, **kwargs
                    )
                    if local_output is not None:
                        logger.debug(
                            'FileSystemWriterAsync: main worker results successfully collected'
                        )
                        check_local_output(local_output[1], local_output[0])
                        write_results_or_exc[local_output[0]] = local_output[1]

            except Exception as e:
                err_msg = f"An error is caught while starting worker {i}, error: {e}"
                logger.error(err_msg)
                write_results_or_exc = RuntimeError(err_msg)

        if not isinstance(write_results_or_exc, Exception) and len(thread_list) > 0:
            for t in thread_list:
                t.start()

            logger.debug("FileSystemWriterAsync: collecting worker results...")

            count_queue.join()
            for _ in range(len(write_buckets) - 1):
                try:
                    local_thread_idx, local_results_or_exc = local_results_queue.get()
                except queue.Empty:
                    write_results_or_exc = RuntimeError(
                        "Unexpected empty `local_results_queue`"
                        f" (expected {len(write_buckets) - 1} items)"
                    )
                    break
                else:
                    try:
                        check_local_output(local_results_or_exc, local_thread_idx)
                    except Exception as worker_exc:
                        write_results_or_exc = worker_exc
                        break
                    write_results_or_exc[local_thread_idx] = local_results_or_exc
            for t in thread_list:
                t.join()
            logger.debug('FileSystemWriterAsync: collected worker results successfully')

        if isinstance(write_results_or_exc, dict) and len(write_results_or_exc) != len(
            write_buckets
        ):
            write_results_or_exc = RuntimeError(
                f"Incomplete write results: expected {len(write_buckets)} buckets,"
                f" got {len(write_results_or_exc)}"
            )
        global_results_queue.put(write_results_or_exc)

        w_end = time()
        logger.debug(f"{w_end}, rank: {rank}, write(sync,threads): {w_end - w_start}")

    @staticmethod
    @_disable_gc()
    def write_preloaded_data(
        transform_list: List[_StorageWriterTransforms],
        open_file: Callable,
        local_thread_idx: int,
        write_bucket: WriteBucket,
        results_queue: Optional[queue.Queue],
        count_queue: Optional[queue.Queue],
        use_fsync: bool,
        **kwargs,
    ) -> Optional[Tuple[int, Union[List[WriteResult], Exception]]]:
        """
        Performs actual data saving to storage (used by worker threads in multithread mode).

        Args:
            transform_list (List[_StorageWriterTransforms]): streaming transforms list
            open_file (Callable): file open callable
            local_thread_idx (int): index of the worker thread that performs writing
            write_bucket (WriteBucket): data to write to storage
            results_queue (queue.Queue): queue to return the write results.
                If None (main-thread worker), result is returned directly.
            count_queue (queue.Queue): queue to signal worker task completion.
                If None (main-thread worker), skipped.
            use_fsync (bool): if True, calls os.fsync at the end of saving

        Returns: None when running in a thread (results put in queue);
                 result tuple when running as main-thread worker (results_queue is None)
        """
        logger = logging.getLogger(__name__)
        logger.debug(f'{local_thread_idx} started')
        mem_before = _process_memory()
        use_msc = kwargs.get('use_msc', False)

        try:
            local_results = FileSystemWriterAsync._write_bucket_to_storage(
                transform_list, open_file, write_bucket, use_fsync, use_msc
            )
            local_output = (local_thread_idx, local_results)
        except Exception as e:
            logger.debug(f'{local_thread_idx} failed with exception {e}')
            local_output = (local_thread_idx, e)

        if results_queue is not None:
            results_queue.put(local_output)
        if count_queue is not None:
            # Signal this thread is done.
            count_queue.get()
            count_queue.task_done()

        mem_after = _process_memory()
        logger.debug(
            f"{local_thread_idx} consumed: {mem_after - mem_before},"
            f" before: {mem_before}, after: {mem_after}"
        )
        return local_output

    def write_data(self, plan: SavePlan, planner: SavePlanner) -> Future[List[WriteResult]]:
        """Write all items from ``plan``."""
        raise NotImplementedError('write_data not implemented for FileSystemWriterAsync')

    def retrieve_write_results(self) -> Union[List[WriteResult], WRAPPED_EXCEPTION]:
        """
        Turn the latest dict including write results from `self.results_queue`
            into a single results lists. Includes error check.

        Returns (Union(List[WriteResult], WRAPPED_EXCEPTION): the list of write results
            from all local workers performing the save, or a WRAPPED_EXCEPTION if
            an exception was raised during the writing process.
        """
        if self.results_queue is None:
            write_results_or_exc = {}
        else:
            try:
                write_results_or_exc = self.results_queue.get_nowait()
            except queue.Empty:
                return _wrap_exception(RuntimeError('results_queue should not be empty'))

        if isinstance(write_results_or_exc, Exception):
            # Worker failed — its data cache may have been lost (e.g. after a restart).
            # Drop any identifier we recorded as cached so the next save re-populates it.
            if self.consistent_data_identifier is not None:
                FileSystemWriterAsync._cached_identifiers.discard(
                    self.consistent_data_identifier.key
                )
            try:
                raise RuntimeError(
                    f'Worker failure: {write_results_or_exc}'
                ) from write_results_or_exc
            except Exception as e:
                return _wrap_exception(e)
        write_results: dict = write_results_or_exc
        if self.has_data_to_write and len(write_results) == 0:
            return _wrap_exception(
                RuntimeError(
                    'Worker returned empty results despite having data to write.'
                    ' This probably indicates a worker failure.'
                )
            )
        return list(chain.from_iterable(write_results.values()))

    def prepare_decentralized_global_plan(self, local_plan: SavePlan) -> SavePlan:
        """Instead of assigning indices by plan order, uses PyT rank (same outcome).

        Args:
            local_plan (SavePlan): local plan to turn to a global plan
                (without interactions with other ranks)

        Returns:
            SavePlan - locally transformed plan equivalent to the plan that would be
                created by the coordinator
        """
        return dataclasses.replace(
            local_plan, storage_data=_StoragePrefix(f"__{torch.distributed.get_rank()}_")
        )

    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        """
        Finish the checkpointing process.

        Args:
            metadata (Metadata): metadata to save
            results (List[List[WriteResult]]): results to save
        """
        if self.use_msc:
            import multistorageclient as msc

            storage_md = dict()
            for wr_list in results:
                storage_md.update({wr.index: wr.storage_data for wr in wr_list})

            metadata.storage_data = storage_md

            # storage_meta was introduced since PyTorch 2.4
            if "storage_meta" in inspect.signature(Metadata).parameters:
                metadata.storage_meta = self.storage_meta()

            path = os.path.join(self.checkpoint_dir, ".metadata")

            with msc.open(path, "wb") as metadata_file:
                # Issue: [B301:blacklist] Pickle and modules that wrap it can be unsafe when used to deserialize untrusted data, possible security issue.
                # Severity: Medium   Confidence: High
                # CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
                # More Info: https://bandit.readthedocs.io/en/1.8.3/blacklists/blacklist_calls.html#b301-pickle
                pickle.dump(metadata, metadata_file)  # nosec
        else:
            super().finish(metadata, results)

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        """
        Prepare the local plan for the checkpointing process.
        """
        if self.use_msc:
            import multistorageclient as msc

            msc.os.makedirs(str(self.checkpoint_dir), exist_ok=True)
        else:
            super().prepare_local_plan(plan)

        return plan

    @property
    def checkpoint_id(self) -> Union[str, os.PathLike]:
        """
        return the checkpoint_id that will be used to save the checkpoint.
        """
        return str(self.checkpoint_dir)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """
        Validate the checkpoint_id that will be used to save the checkpoint.

        This method is available in PyTorch 2.3 and above.
        """
        checkpoint_id_str = str(checkpoint_id)
        if checkpoint_id_str.startswith("msc://"):
            return True

        if hasattr(FileSystemWriter, "validate_checkpoint_id"):
            return FileSystemWriter.validate_checkpoint_id(checkpoint_id)

        return False


# Register cleanup hook so that when PersistentAsyncCaller spawns a new worker process
# (e.g., after an abort/restart), the training-side shm tensor cache is invalidated and
# the next checkpoint re-sends actual shm tensors to the fresh worker.
PersistentAsyncCaller.register_worker_restart_callback(FileSystemWriterAsync.cleanup_tensor_caches)


def _split_by_size_and_type(bins: int, items: List[WriteItem]) -> List[List[WriteItem]]:
    """
    Splits write items according to item size into close to uniform bins.

    Same as torch.distributed.checkpoint.filesystem._split_by_size_and_type,
    but with a fixed _item_size function.

    Args:
        bins (int): numbers of bins to split to
        items (List[WriteItem]): list of write items

    Returns (List[List[WriteItem]]): write items split to bins
    """
    if bins == 1:
        return [items]

    bytes_items: List[WriteItem] = []
    tensor_items: List[WriteItem] = []
    for wi in items:
        container = bytes_items if wi.type == WriteItemType.BYTE_IO else tensor_items
        container.append(wi)

    buckets: List[List[WriteItem]] = [[] for _ in range(bins)]
    bucket_sizes = [0 for _ in range(bins)]

    # Assign bytes with a simple round-robin
    for i, item in enumerate(bytes_items):
        buckets[i % bins].append(item)

    # Sort tensor items by size in decreasing order once and store the size with item
    sized_tensors = [(item, _item_size(item)) for item in tensor_items]
    sized_tensors.sort(key=itemgetter(1), reverse=True)

    # Use a min heap for bin assignment
    # Store (total_size_of_bin, bin_index) tuples
    heap: List[Tuple[int, int]] = [(0, i) for i in range(bins)]

    # Assign tensors using heap
    for item, size in sized_tensors:
        total_bin_size, bin_idx = heappop(heap)
        buckets[bin_idx].append(item)
        heappush(heap, (total_bin_size + size, bin_idx))

    return buckets


def _split_by_separation_hint(
    buckets: List[List[WriteItem]], separation_hint: Optional[str] = None
) -> Dict[str, List[List[WriteItem]]]:
    """
    Splits buckets into those whose keys begin with the separation_hint and those whose keys do not

    Args:
        buckets (List[List[WriteItem]]): buckets to split
        separation_hint (Optional[str]): optional prefix to split on

    Returns (Dict[str, List[List[WriteItem]]]): a dictionary
        mapping the prefix to the relevant buckets
    """
    bins = len(buckets)
    buckets_with_separation_hint: Dict[str, List[List[WriteItem]]] = {}
    if separation_hint is not None:
        buckets_default: List[List[WriteItem]] = [[] for _ in range(bins)]
        buckets_hint: List[List[WriteItem]] = [[] for _ in range(bins)]
        for i in range(bins):
            for item in buckets[i]:
                if item.index.fqn.startswith(separation_hint):
                    buckets_hint[i].append(item)
                else:
                    buckets_default[i].append(item)
        buckets_with_separation_hint[""] = buckets_default
        buckets_with_separation_hint[separation_hint] = buckets_hint
    else:
        buckets_with_separation_hint[""] = buckets
    return buckets_with_separation_hint


def _item_size(item: WriteItem) -> int:
    """
    Calculates size (in bytes) of a single write item.

    Same as torch.distributed.checkpoint.filesystem._item_size,
    but fixes computing chunk size (with item.tensor_data.chunk.sizes)

    Args:
        item (WriteItem): write item to compute the size of

    Returns (int): size of an item in bytes
    """
    size = 1
    assert item.tensor_data is not None
    # can't use math.prod as PT needs to support older python
    for s in item.tensor_data.chunk.sizes:
        size *= s

    dtype = item.tensor_data.properties.dtype
    return size * torch._utils._element_size(dtype)


def _process_memory() -> int:
    """
    Get memory used by current process.

    Returns (int): memory used by current process
    """
    if not HAVE_PSUTIL:
        raise RuntimeError("psutil is not installed, please install it with `pip install psutil`")
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss
