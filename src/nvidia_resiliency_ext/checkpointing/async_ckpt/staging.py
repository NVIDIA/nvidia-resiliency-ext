# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Staging: giving the checkpoint worker its own copy of the tensors a rank writes.

Staging runs after planning, on the tensors resolved for this rank's write items. Once a
save is staged, training may modify its tensors. A stager also decides what is sent to the
persistent worker, which keeps tensors it has received under a key so that later saves with
the same write items need not send them again.

- `ShmStager` copies the tensors into shared CPU memory in the training process.
- `IpcStager` sends GPU tensors to the worker as CUDA IPC handles, and the worker copies
  them to host memory (`IpcStager.copy_to_host`).
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple

import torch
from torch.distributed.checkpoint.planner import WriteItem

from nvidia_resiliency_ext.shared_utils import semconv, telemetry

from .core import PersistentAsyncCaller

logger = logging.getLogger(__name__)

# Write items and their staged data, in the same order
ItemsAndData = Tuple[List[WriteItem], List[Any]]


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


def _compute_tensor_data_ptrs(items: List[WriteItem], tensors: List[Any]) -> Tuple[Tuple, ...]:
    """Compute a storage identity fingerprint for tensors cached via CUDA IPC."""
    ptrs = []
    for item, tensor in zip(items, tensors):
        chunk = item.tensor_data.chunk if item.tensor_data is not None else None
        chunk_info = (tuple(chunk.offsets), tuple(chunk.sizes)) if chunk is not None else None

        if isinstance(tensor, torch.Tensor):
            storage = tensor.untyped_storage()
            tensor_info = (
                str(tensor.device),
                tensor.data_ptr(),
                storage.data_ptr(),
                tensor.storage_offset(),
                tuple(tensor.size()),
                tuple(tensor.stride()),
                str(tensor.dtype),
            )
        else:
            tensor_info = (type(tensor).__qualname__,)
        ptrs.append((item.index.fqn, chunk_info, tensor_info))
    return tuple(ptrs)


class StagedTensors(NamedTuple):
    """What a stager sends to the worker for one save.

    Args:
        identifier: the key the worker keeps `cached` under, or None if nothing is kept.
        cached: tensors the worker keeps under `identifier`. None if the worker already
            has them, or if there are none.
        uncached: tensors sent with every save, or None if there are none.
    """

    identifier: Optional[ConsistentDataIdentifier]
    cached: Optional[ItemsAndData]
    uncached: Optional[ItemsAndData]


class Stager(ABC):
    """Stages the tensors resolved for a rank's write items, after planning."""

    # Whether staging finishes in the worker (`AsyncRequest.preload_fn`), rather than in the
    # training process when `stage` returns
    stages_in_worker: bool = False

    def _prepare(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Prepare one resolved tensor for staging, as it is resolved.

        Returns:
            (tensor, was_dequantized): the tensor, and whether dequantize() was called on it.
                Tracked explicitly because some frameworks (e.g. TransformerEngine MXFP8) use a
                fake bfloat16 dtype for quantized tensors, making dtype-based detection
                unreliable.
        """
        tensor = tensor.detach()
        if tensor.device.type == "cuda" and "dequantize" in type(tensor).__dict__:
            return tensor.dequantize(), True
        return tensor, False

    def stage(self, items: List[WriteItem], tensors: Iterable[torch.Tensor]) -> StagedTensors:
        """Stage the tensors of `items`, in the training process.

        Args:
            items: this rank's tensor write items, from the final plan.
            tensors: the resolved tensor of each item, in order. Each is prepared as it is
                taken, so an iterable that resolves them lazily holds one at a time.
        """
        prepared = [
            self._prepare(tensor) if isinstance(tensor, torch.Tensor) else (tensor, False)
            for tensor in tensors
        ]
        return self._stage(
            items, [tensor for tensor, _ in prepared], [dequantized for _, dequantized in prepared]
        )

    @abstractmethod
    def _stage(
        self, items: List[WriteItem], tensors: List[torch.Tensor], dequantized: List[bool]
    ) -> StagedTensors:
        """Stage the prepared tensors of `items`."""

    @abstractmethod
    def forget(self, key: str) -> None:
        """Forget that the worker holds the tensors of `key`, so the next save sends them."""

    @abstractmethod
    def clear(self) -> None:
        """Forget everything the worker holds. Called when a new worker starts."""


# The drain for shared-memory buffers that weren't given their own: a blocking call that waits
# for the writes in flight. Registered by AsyncCallsQueue(cpu_shm_mode=True); see set_shm_drain.
_shm_drain: Optional[Callable[[], None]] = None


class ShmBuffers(ABC):
    """The shared-memory buffers a `ShmStager` copies a save's tensors into.

    Args:
        drain: blocking call that waits for the writes in flight, or None for the one registered
            with `set_shm_drain`.
    """

    def __init__(self, drain: Optional[Callable[[], None]] = None):
        self.drain = drain

    def _drain(self) -> None:
        drain = self.drain if self.drain is not None else _shm_drain
        if drain is not None:
            with telemetry.span(semconv.SPAN_GROUP_CKPT_PHASES, "nv.nvrx.ckpt.save.shm_drain"):
                drain()

    @abstractmethod
    def acquire(
        self, key: str, tensors: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], Optional[ConsistentDataIdentifier], bool]:
        """Buffers to copy `tensors` into, once no write is still reading them.

        Returns:
            (buffers, identifier, worker_has_them): the buffers, the key the worker keeps them
                under (None if it doesn't keep them), and whether it already has them.
        """

    def forget(self, key: str) -> None:
        pass

    def clear(self) -> None:
        pass


def _allocate_shm(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """One shared-memory tensor of the same shape and dtype for each of `tensors`."""
    return [torch.empty(t.shape, dtype=t.dtype).share_memory_() for t in tensors]


class PerSaveShmBuffers(ShmBuffers):
    """New buffers for every save, handed to the worker with it.

    Before allocating, `drain` waits for the writes in flight. The new buffers don't need it, but
    without it, saves scheduled faster than they are written would each hold a full set of
    buffers, and shared memory would grow without bound.
    """

    def acquire(self, key, tensors):
        self._drain()
        return _allocate_shm(tensors), None, False


class PersistentShmBuffers(ShmBuffers):
    """Buffers allocated once per write-item structure and reused by later saves.

    The first save sends the buffers to the worker, which keeps references to them, so later
    saves send only the key. Before buffers are reused, `drain` waits for any write still
    reading them.
    """

    def __init__(self, drain: Optional[Callable[[], None]] = None):
        super().__init__(drain)
        # The buffers the worker holds, allocated by an earlier save.
        # Key: SHA-256 of the tensor write items (same as ConsistentDataIdentifier).
        # Value: one managed share_memory_() tensor per write item.
        self._buffers: Dict[str, List[torch.Tensor]] = {}

    def acquire(self, key, tensors):
        if key in self._buffers:
            # The worker for checkpoint N may still be reading these buffers while we are
            # about to overwrite them with checkpoint N+1 values. Drain before looking them
            # up, because a drain that finds the worker dead clears them.
            self._drain()
        buffers = self._buffers.get(key)
        if buffers is None:
            buffers = self._buffers[key] = _allocate_shm(tensors)
            logger.debug(f"Allocated {len(buffers)} shm tensors (key={key})")
            return buffers, ConsistentDataIdentifier(key), False
        return buffers, ConsistentDataIdentifier(key), True

    def forget(self, key):
        # The next save allocates new buffers and sends them.
        self._buffers.pop(key, None)

    def clear(self):
        if self._buffers:
            logger.info(f"Clearing shm tensor cache ({len(self._buffers)} entries)")
        self._buffers.clear()


class ShmStager(Stager):
    """Copies every tensor into shared CPU memory in the training process.

    The worker then never needs CUDA IPC / fabric handles or references to live training
    tensors. Every save copies every tensor; `buffers` decides whether into new buffers or
    into the ones the previous save used.
    """

    def __init__(self, buffers: ShmBuffers):
        self.buffers = buffers

    def _stage(self, items, tensors, dequantized):
        key = _compute_data_structure_key_from_plan(items)
        shm_tensors, identifier, worker_has_them = self.buffers.acquire(key, tensors)
        with telemetry.span(semconv.SPAN_GROUP_CKPT_PHASES, "nv.nvrx.ckpt.save.stage"):
            pending_cuda_copies = 0
            for shm_t, source_t in zip(shm_tensors, tensors):
                shm_t.copy_(source_t, non_blocking=True)
                # Periodically sync to avoid exhausting CUDA's pageable staging
                # pages (shm tensors are not pinned so copy_ is not a pinned DMA).
                pending_cuda_copies += int(source_t.device.type == "cuda")
                if pending_cuda_copies == 8:
                    torch.cuda.synchronize()
                    pending_cuda_copies = 0
            if pending_cuda_copies:
                torch.cuda.synchronize()
        logger.debug(f"Copied {len(shm_tensors)} tensors into shm (key={key})")
        # Send the buffers unless the worker kept them from an earlier save: their values are
        # fresh after the copy above.
        cached = None if worker_has_them else (items, shm_tensors)
        return StagedTensors(identifier, cached, None)

    def forget(self, key):
        self.buffers.forget(key)

    def clear(self):
        self.buffers.clear()


class IpcHandles(ABC):
    """Whether the worker keeps the CUDA IPC handles of a save's GPU tensors."""

    @abstractmethod
    def acquire(
        self, key: str, items: List[WriteItem], tensors: List[torch.Tensor]
    ) -> Tuple[Optional[ConsistentDataIdentifier], bool]:
        """The key the worker keeps the tensors' handles under, and whether it has them."""

    def forget(self, key: str) -> None:
        pass

    def clear(self) -> None:
        pass


class PerSaveIpcHandles(IpcHandles):
    """Every save sends the handles of its GPU tensors."""

    def acquire(self, key, items, tensors):
        return None, False


class PersistentIpcHandles(IpcHandles):
    """The worker keeps the handles, and they are sent again only if the storage has moved."""

    def __init__(self):
        # The data/storage pointer fingerprint of the GPU tensors whose handles the worker
        # holds. Key: SHA-256 of the tensor write items (same as ConsistentDataIdentifier).
        self._sent_data_ptrs: Dict[str, Tuple[Tuple, ...]] = {}

    def acquire(self, key, items, tensors):
        data_ptrs = _compute_tensor_data_ptrs(items, tensors)
        worker_has_them = self._sent_data_ptrs.get(key) == data_ptrs
        self._sent_data_ptrs[key] = data_ptrs
        return ConsistentDataIdentifier(key), worker_has_them

    def forget(self, key):
        self._sent_data_ptrs.pop(key, None)

    def clear(self):
        self._sent_data_ptrs.clear()


class IpcStager(Stager):
    """Sends GPU tensors to the worker as CUDA IPC handles; the worker copies them to host.

    CPU tensors are cloned in the training process, and quantized GPU tensors are dequantized
    into new tensors. Neither has a stable IPC handle, so they are sent with every save;
    `handles` decides whether the other GPU tensors' handles are too.
    """

    # The worker copies the GPU tensors to host memory
    stages_in_worker = True

    def __init__(self, handles: IpcHandles):
        self.handles = handles

    def _prepare(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        # CPU tensors must be cloned before an ``AsyncRequest`` is sent to a persistent worker.
        if tensor.device.type == "cpu":
            return tensor.detach().clone(), False
        return super()._prepare(tensor)

    def _stage(self, items, tensors, dequantized):
        stable_items, stable_data = [], []
        uncached_items, uncached_data = [], []
        for item, data, was_dequantized in zip(items, tensors, dequantized):
            if isinstance(data, torch.Tensor) and data.device.type == "cpu" or was_dequantized:
                uncached_items.append(item)
                uncached_data.append(data)
            else:
                stable_items.append(item)
                stable_data.append(data)
        # Uncached tensors are always passed fresh (never cached)
        uncached = (uncached_items, uncached_data) if uncached_items else None
        if not stable_items:
            return StagedTensors(None, None, uncached)

        key = _compute_data_structure_key_from_plan(items)
        identifier, worker_has_them = self.handles.acquire(key, stable_items, stable_data)
        logger.debug(
            f"{'Reusing' if worker_has_them else 'Sending'} {len(stable_items)} GPU tensors "
            f"(key={key}), {len(uncached_items)} uncached tensors passed fresh"
        )
        cached = None if worker_has_them else (stable_items, stable_data)
        return StagedTensors(identifier, cached, uncached)

    def forget(self, key):
        self.handles.forget(key)

    def clear(self):
        self.handles.clear()

    @staticmethod
    def copy_to_host(staged: StagedTensors, non_blocking: bool) -> StagedTensors:
        """Copy a save's GPU tensors to host memory, in the worker.

        Returns:
            StagedTensors: every tensor of the save, in host memory, kept by nothing.
        """
        items, tensors = receive(staged)
        host_tensors = []
        needs_sync = False
        for tensor in tensors:
            if tensor.is_cuda:
                needs_sync = True
                host_tensors.append(tensor.to("cpu", non_blocking=non_blocking))
            else:
                host_tensors.append(tensor)
        if non_blocking and needs_sync:
            torch.cuda.synchronize()
        return StagedTensors(None, (items, host_tensors), None)


def make_stager(cpu_shm: bool, persistent: bool) -> Stager:
    """A stager that copies into shared memory, or sends CUDA IPC handles if not `cpu_shm`.

    Args:
        cpu_shm: copy tensors into shared CPU memory in the training process.
        persistent: whether the write items stay the same between saves, so that the stager
            reuses its buffers or handles and the worker keeps them.
    """
    if cpu_shm:
        return ShmStager(PersistentShmBuffers() if persistent else PerSaveShmBuffers())
    return IpcStager(PersistentIpcHandles() if persistent else PerSaveIpcHandles())


# One stager per configuration for this process, for writers that are not given one. The
# persistent worker that they send to is one per process too.
_default_stagers: Dict[Tuple[bool, bool], Stager] = {}


def default_stager(cpu_shm: bool, persistent: bool) -> Stager:
    """This process's stager for the configuration. See `make_stager`."""
    key = (cpu_shm, persistent)
    if key not in _default_stagers:
        _default_stagers[key] = make_stager(cpu_shm, persistent)
    return _default_stagers[key]


def clear_default_stagers() -> None:
    """Make the default stagers send everything again. Called when a new worker starts.

    A fresh worker has an empty ``_worker_data_cache``.
    """
    for stager in _default_stagers.values():
        stager.clear()


def set_shm_drain(drain: Optional[Callable[[], None]]) -> None:
    """Set (or clear with None) the drain of the shared-memory buffers that weren't given one."""
    global _shm_drain
    _shm_drain = drain


def receive(staged: StagedTensors) -> ItemsAndData:
    """A save's tensors and their write items, in the worker.

    The cached tensors as sent, or kept from an earlier save, then the uncached ones.
    """
    items, tensors = [], []
    for part in (receive_cached(staged.identifier, staged.cached), staged.uncached):
        if part:
            items.extend(part[0])
            tensors.extend(part[1])
    return items, tensors


def receive_cached(
    identifier: Optional[ConsistentDataIdentifier], cached: Optional[ItemsAndData]
) -> Optional[ItemsAndData]:
    """Return a save's cached tensors, in the worker: as sent, or kept from an earlier save."""
    if not isinstance(identifier, ConsistentDataIdentifier):
        return cached
    # On the GPU IPC path, CPU and dequantized tensors remain uncached.
    key = identifier.key
    if cached is not None:
        PersistentAsyncCaller._worker_data_cache[key] = cached
        logger.debug(f"Worker cached tensors (key={key})")
        return cached
    if key in PersistentAsyncCaller._worker_data_cache:
        logger.debug(f"Worker retrieved cached tensors (key={key})")
        return PersistentAsyncCaller._worker_data_cache[key]
    raise RuntimeError(f"Worker cache miss for key {key}. Worker may have restarted.")
