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
import inspect
import logging
import os
import queue
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from heapq import heappop, heappush
from io import UnsupportedOperation
from itertools import chain, count
from operator import itemgetter
from pathlib import Path
from time import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    cast,
)

import torch
from torch import multiprocessing as mp
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.api import WRAPPED_EXCEPTION, _wrap_exception
from torch.distributed.checkpoint.filesystem import (
    CURRENT_DCP_VERSION,
    DEFAULT_SUFFIX,
    _metadata_fn,
    _StoragePrefix,
    _write_item,
)
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
from . import _metadata_pickler
from ._metadata_template import MetadataTemplate
from .core import PersistentAsyncCaller
from .staging import (  # noqa: F401 (ConsistentDataIdentifier is re-exported)
    ConsistentDataIdentifier,
    IpcStager,
    StagedTensors,
    Stager,
    clear_default_stagers,
    default_stager,
    receive,
    set_shm_drain,
)

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

WriteBucket = Tuple[str, str, Tuple[list, list]]  # represents writes to a single file

_results_queue = None


def _wrap_exception_for_gather(exc: BaseException) -> WRAPPED_EXCEPTION:
    wrapped_exception = _wrap_exception(exc)
    _, stack_summary = wrapped_exception

    # Python 3.13+ traceback FrameSummary may hold a code object in _code.
    # Code objects are not pickleable, but dist.gather_object pickles the
    # wrapped exception before sending it to the coordinator.
    for frame in stack_summary:
        if hasattr(frame, "_code"):
            object.__setattr__(frame, "_code", None)

    return wrapped_exception


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


# Backward-compat alias for the former private name
_get_write_results_queue = get_write_results_queue


class _WriteData(NamedTuple):
    """What the worker writes for one save.

    Args:
        checkpoint_dir: the directory to write the files into.
        separation_hint: prefix of the items to write to separate files, or None.
        tensors: the tensors as staged.
        byte_io: BytesIO items and their data, or None.
        thread_count: the number of threads or processes to write with.
        storage_plan: this rank's storage data from the plan, with the file prefix.
    """

    checkpoint_dir: str
    separation_hint: Optional[str]
    tensors: StagedTensors
    byte_io: Optional[Tuple[list, list]]
    thread_count: int
    storage_plan: Any

    def received_here(self) -> '_WriteData':
        """This write data with the kept tensors filled in, from this process's worker cache.

        For a forked worker, which keeps nothing between saves; see `TemporalAsyncCaller`.
        """
        return self._replace(tensors=StagedTensors(None, receive(self.tensors), None))


def _make_write_buckets(write_data: _WriteData) -> List[WriteBucket]:
    """Group a save's items into one write bucket per file, in the worker."""
    start = time()
    items, resolved_data = receive(write_data.tensors)
    if write_data.byte_io:
        items = items + list(write_data.byte_io[0])
        resolved_data = resolved_data + list(write_data.byte_io[1])
    separation_hint = write_data.separation_hint
    storage_plan = write_data.storage_plan
    logger.debug(f"_make_write_buckets: thread_count: {write_data.thread_count}, time: {start}")

    # Create buckets from items
    bins = write_data.thread_count // 2 if separation_hint is not None else write_data.thread_count
    item_buckets = _split_by_size_and_type(bins, items)
    logger.debug(f"_make_write_buckets: bucket_prep, time: {time() - start}")

    # Create a mapping from items to resolved data
    item_to_data = {id(item): data for item, data in zip(items, resolved_data)}

    # Files are numbered across all groups
    file_numbers = count()

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
                    # Tensor data, in host memory
                    tensor_data.append((item, data))

            if len(bytes_data) > 0 or len(tensor_data) > 0:
                file_name = f"{group_name}{storage_plan.prefix}{next(file_numbers)}{DEFAULT_SUFFIX}"
                write_buckets.append(
                    (
                        os.path.join(write_data.checkpoint_dir, file_name),
                        file_name,
                        (bytes_data, tensor_data),
                    )
                )
    return write_buckets


class FileSystemWriterAsync(FileSystemWriter):
    """
    Async-enabled implementation of FileSystemWriter using file I/O.

    This class does not spawn the async process itself but relies on an external async mechanism.

    **Flow:**

    1. Call `write_data`
    2. Externally start an async process with `get_save_function_and_args` and its arguments.
    3. The async function calls `write_preloaded_data_multithread`, which writes the buckets
       on worker threads.
    4. Once saving is finalized on all ranks, call `super().finish` with the results stored in `self.writer_result`.

    **Note:** Step (3) can also be executed synchronously.

    Currently, it is assumed that a separate writer is created for each ckpt save
    (intermediate state is stored as writer attributes).
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        *args,
        separation_hint: Optional[str] = None,
        use_msc: bool = False,
        is_multiproc_io: bool = False,
        use_cached_data_structure: bool = False,
        use_cpu_shm_for_gpu_tensors: bool = False,
        stager: Optional[Stager] = None,
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
        if is_multiproc_io:
            warnings.warn(
                "is_multiproc_io is deprecated and ignored: FileSystemWriterAsync always writes"
                " with threads. The argument will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
        # Use CPU shared-memory tensors instead of GPU IPC fabric handles.
        # Avoids cuMemImportFromShareableHandle on MNNVL systems where NVLink fabric
        # resources are exhausted by the training ranks.
        self.use_cpu_shm_for_gpu_tensors = use_cpu_shm_for_gpu_tensors
        # Stages the tensors after planning. Without one, the process's default stager for
        # `use_cpu_shm_for_gpu_tensors` and `use_cached_data_structure`.
        self._stager: Stager = (
            stager
            if stager is not None
            else default_stager(use_cpu_shm_for_gpu_tensors, use_cached_data_structure)
        )

    def prepare_write_data(self, plan: SavePlan, planner: SavePlanner) -> None:
        """
        First stage of async saving. Resolve data and store in compact format.

        Separates tensor data into cacheable and uncached groups based on the transfer mode.
        ByteIO data is always resolved fresh. Bucket creation is deferred to `preload_tensors`
        so that it can run in the persistent worker process and use the data cache.

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

        def resolve_data(items, prepare=None):
            resolved = []
            dequantized_flags = []
            for item in items:
                data = planner.resolve_data(item)
                if prepare is not None and isinstance(data, torch.Tensor):
                    data, was_dequantized = prepare(data)
                else:
                    was_dequantized = False
                resolved.append(data)
                dequantized_flags.append(was_dequantized)
            return resolved, dequantized_flags

        # Separate tensor data from ByteIO data. The stager stages the tensors.
        tensor_items = [item for item in plan.items if item.type != WriteItemType.BYTE_IO]
        byte_io_items = [item for item in plan.items if item.type == WriteItemType.BYTE_IO]

        if tensor_items:
            resolved_tensors, dequantized_flags = resolve_data(tensor_items, self._stager.prepare)
            staged = self._stager.stage(tensor_items, resolved_tensors, dequantized_flags)
        else:
            staged = StagedTensors(None, None, None)
        self.consistent_data_identifier = staged.identifier
        self.cached_tensor_data = staged.cached
        self.uncached_tensor_data = staged.uncached

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
        """Make the default stagers send everything again. Called when a new worker starts."""
        clear_default_stagers()

    @classmethod
    def register_shm_drain_callback(cls, fn: Optional[Callable[[], None]]) -> None:
        """Register (or clear with None) the blocking drain called before reusing shm tensors.

        AsyncCallsQueue registers this when cpu_shm_mode=True so that
        prepare_write_data drains any in-flight write before overwriting shm tensors.
        """
        set_shm_drain(fn)

    def get_save_function_and_args(self) -> Tuple[Optional[Callable], Optional[Callable], List]:
        """
        Get function that saves the data to storage along with its arguments.
        Allows the external caller to apply the save function synchronously or asynchronously.

        Returns: None (if there is nothing to write on this rank) or a tuple of:
            1) the function that saves the data.
            2) the function that finishes staging the tensors in the worker, or None if the
               stager has finished staging them. Its result replaces the 2nd argument of 1).
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

        write_data = _WriteData(
            str(self.checkpoint_dir),
            self.separation_hint,
            StagedTensors(
                self.consistent_data_identifier,
                self.cached_tensor_data,
                self.uncached_tensor_data,
            ),
            self.byte_io_data,
            self.thread_count,
            self.storage_plan,
        )

        write_buckets_func = partial(
            self.write_preloaded_data_multithread, transform_list, self.use_msc, open_file
        )
        write_func = partial(self.write_staged_data, write_buckets_func)

        preload_fn = (
            partial(self.preload_tensors, write_data, True)
            if self._stager.stages_in_worker
            else None
        )

        return (
            write_func,
            preload_fn,
            [torch.distributed.get_rank(), write_data, self.results_queue],
        )

    @staticmethod
    def preload_tensors(write_data: '_WriteData', non_blocking=True) -> '_WriteData':
        """
        Copies the GPU tensors of a save to host memory, in the worker.

        The worker step of `IpcStager`: with CPU shared memory, the tensors are copied to
        host in the training process and there is no worker step.

        Args:
            write_data (_WriteData): the save's write data, with the tensors as staged.
            non_blocking (bool, optional): Enable pinned D2H memcpy. Default is True.

        Returns:
            _WriteData: the write data with every tensor in host memory
        """
        start = time()
        tensors = IpcStager.copy_to_host(write_data.tensors, non_blocking)
        logger.debug(f"preload_tensors: D2H, time: {time() - start}")
        return write_data._replace(tensors=tensors)

    @staticmethod
    def write_staged_data(
        write_buckets_func: Callable,
        rank: int,
        write_data: '_WriteData',
        global_results_queue: mp.Queue,
    ) -> None:
        """
        Writes a save's staged data, in the worker.

        Groups the items into one write bucket per file, then writes the buckets with
        `write_buckets_func` (`write_preloaded_data_multithread`). The buckets are made here,
        in the worker, so that tensors the worker keeps between saves need not be sent again.

        Args:
            write_buckets_func (Callable): writes the buckets: called with `rank`, the buckets
                and `global_results_queue`.
            rank (int): training rank
            write_data (_WriteData): the save's write data, with tensors in host memory.
            global_results_queue (mp.Queue): queue to pass the write results to.
        """
        try:
            write_buckets = _make_write_buckets(write_data)
        except Exception as e:  # reported to finalize like a failed write
            logger.error(f"Making the write buckets failed: {e}")
            global_results_queue.put(e)
            return
        write_buckets_func(rank, write_buckets, global_results_queue)

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
    def write_preloaded_data_multithread(
        transform_list: List[_StorageWriterTransforms],
        use_msc: bool,
        open_file: Callable,
        rank: int,
        write_buckets: List[WriteBucket],
        global_results_queue: mp.Queue,
    ) -> None:
        """
        Performs saving data to storage with one thread per write bucket, all running concurrently.

        Uses threads (not processes) so that this can run safely inside a daemon process
        without spawning child processes.

        Triggering GC during execution can lead to CUDA errors when tensors are shared.
        To prevent this, we disable the GC explicitly for this function with _disable_gc.
        The worker threads run inside it, so GC stays disabled until they have all finished.

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
        write_results_or_exc: Union[dict, Exception]
        try:  # also catches a failure to start a thread, which must not kill the worker
            with ThreadPoolExecutor(max_workers=max(1, len(write_buckets))) as executor:
                futures = [
                    executor.submit(
                        FileSystemWriterAsync.write_preloaded_data,
                        transform_list,
                        open_file,
                        i,
                        write_bucket,
                        use_fsync=True,
                        use_msc=use_msc,
                    )
                    for i, write_bucket in enumerate(write_buckets)
                ]
            write_results_or_exc = {i: future.result() for i, future in enumerate(futures)}
        except Exception as e:
            write_results_or_exc = e
        FileSystemWriterAsync._finalize_write_execution(
            global_results_queue, write_results_or_exc, rank, w_start, "MultiThread", logger
        )

    @staticmethod
    def write_preloaded_data(
        transform_list: List[_StorageWriterTransforms],
        open_file: Callable,
        local_thread_idx: int,
        write_bucket: WriteBucket,
        use_fsync: bool,
        use_msc: bool,
    ) -> List[WriteResult]:
        """
        Performs actual data saving to storage (used by worker threads in multithread mode).

        Args:
            transform_list (List[_StorageWriterTransforms]): streaming transforms list
            open_file (Callable): file open callable
            local_thread_idx (int): index of the worker thread that performs writing
            write_bucket (WriteBucket): data to write to storage
            use_fsync (bool): if True, calls os.fsync at the end of saving
            use_msc (bool): flag to indicate use of multi storage client for storage access

        Returns: the write results of this bucket. Raises if writing fails.
        """
        logger = logging.getLogger(__name__)
        logger.debug(f'{local_thread_idx} started')
        mem_before = _process_memory()
        try:
            local_results = FileSystemWriterAsync._write_bucket_to_storage(
                transform_list, open_file, write_bucket, use_fsync, use_msc
            )
        except Exception as e:
            logger.error(f"Local worker {local_thread_idx} encountered an error: {e}")
            raise
        mem_after = _process_memory()
        logger.debug(
            f"{local_thread_idx} consumed: {mem_after - mem_before},"
            f" before: {mem_before}, after: {mem_after}"
        )
        return local_results

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
                return _wrap_exception_for_gather(RuntimeError('results_queue should not be empty'))

        if isinstance(write_results_or_exc, Exception):
            # Worker failed — its data cache may have been lost (e.g. after a restart).
            # Drop any identifier we recorded as cached so the next save re-populates it.
            if self.consistent_data_identifier is not None:
                self._stager.forget(self.consistent_data_identifier.key)
            try:
                raise RuntimeError(
                    f'Worker failure: {write_results_or_exc}'
                ) from write_results_or_exc
            except Exception as e:
                return _wrap_exception_for_gather(e)
        write_results: dict = write_results_or_exc
        if self.has_data_to_write and len(write_results) == 0:
            return _wrap_exception_for_gather(
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
        Finish the checkpointing process by writing the global metadata file.

        Follows ``FileSystemWriter.finish`` of the torch versions in
        ``_metadata_pickler.TESTED_TORCH_VERSIONS``, but writes ``.metadata`` with
        ``_metadata_pickler.dump_metadata``: the same pickle format as ``pickle.dump``, written
        much faster. On other torch versions, or with ``NVRX_FAST_METADATA_PICKLE=0``, it uses
        ``FileSystemWriter.finish`` (or, for MSC, ``pickle.dump``).

        Args:
            metadata (Metadata): metadata to save
            results (List[List[WriteResult]]): results to save
        """
        self._finish(metadata, results)

    def _finish(
        self,
        metadata: Union[Metadata, "MetadataTemplate"],
        results: List[List[WriteResult]],
        storage_rows: Optional["np.ndarray"] = None,
    ) -> None:
        """Write .metadata as finish does, also from a MetadataTemplate, which it doesn't modify.

        With storage_rows (the ranks' write results as tables, one zero-padded table per row of a
        2-D uint8 array, see ``_metadata_pickler.write_result_table``), storage_data is written
        from them instead of results. Finalize passes rows only when this process writes from tables, which
        needs the fast writer.
        """
        fast = _metadata_pickler.fast_metadata_enabled()
        assert storage_rows is None or fast, "write-result tables need the fast .metadata writer"
        if not fast and not self.use_msc:
            if isinstance(metadata, MetadataTemplate):
                metadata = metadata.to_metadata()
            super().finish(metadata, results)
            return

        fields = {}
        if fast:
            fields["version"] = CURRENT_DCP_VERSION
        fields["storage_meta"] = self.storage_meta()
        storage_md = None
        if storage_rows is None:
            storage_md = {}
            for wr_list in results:
                storage_md.update({wr.index: wr.storage_data for wr in wr_list})

        if isinstance(metadata, MetadataTemplate):
            metadata = metadata.with_fields(**fields)

            def dump(stream) -> None:
                metadata.dump(stream, storage_rows, storage_md)

        else:
            # As torch's finish does: on the Metadata given, which torch's save returns.
            if "version" in fields:
                metadata.version = fields["version"]
            if storage_md is not None:
                metadata.storage_data = storage_md
            metadata.storage_meta = fields["storage_meta"]

            def dump(stream) -> None:
                _metadata_pickler.dump_metadata(metadata, stream, storage_rows)

        if self.use_msc:
            import multistorageclient as msc

            path = os.path.join(self.checkpoint_dir, ".metadata")
            with msc.open(path, "wb") as metadata_file:
                dump(metadata_file)
            return

        # Without collectives, each rank writes its own metadata file.
        if not self.use_collectives and self.rank is not None:
            tmp_filename = f"__{self.rank}{_metadata_fn}.tmp"
            metadata_path = self._get_metadata_path(self.rank)
        else:
            tmp_filename = f"{_metadata_fn}.tmp"
            metadata_path = self._get_metadata_path()
        tmp_path = cast(Path, self.fs.concat_path(self.path, tmp_filename))
        with self.fs.create_stream(tmp_path, "wb") as metadata_file:
            dump(metadata_file)
            if self.sync_files:
                # Flush Python-level buffers (OS for local files, network for cloud storage) before fsync.
                metadata_file.flush()
                try:
                    os.fsync(metadata_file.fileno())
                except (AttributeError, UnsupportedOperation):
                    os.sync()  # as PyTorch < 2.14 does; 2.14 only warns

        # delete in-case other checkpoints were present.
        if self.fs.exists(metadata_path):
            self.fs.rm_file(metadata_path)

        self.fs.rename(tmp_path, metadata_path)

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
