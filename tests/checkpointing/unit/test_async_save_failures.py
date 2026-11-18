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

"""Async saves must not hang the trainer, lose a checkpoint or write a wrong one when the
persistent worker dies, is aborted or is replaced.

The tests use only what applications use: AsyncCallsQueue, AsyncRequest, FileSystemWriterAsync
and the state_dict_saver functions. Requests that the spawned worker runs without the test code
use builtins: `abs` is a call the worker finishes, `os._exit` one it dies in, and `time.sleep`
one it is still running.
"""

import os
import time
import warnings
from typing import IO, Any
from unittest import mock

import pytest
import torch
from torch.distributed.checkpoint import (
    CheckpointException,
    DefaultLoadPlanner,
    DefaultSavePlanner,
    FileSystemReader,
    load,
)
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import SavePlan, WriteItem, WriteItemType

from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue, AsyncRequest
from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import (
    FileSystemWriterAsync,
    get_write_results_queue,
)
from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import (
    save_state_dict_async_finalize,
    save_state_dict_async_plan,
)
from nvidia_resiliency_ext.checkpointing.utils import diff

from . import TempNamedDir
from .test_utilities import Utils


def _worker(async_queue: AsyncCallsQueue):
    """The queue's persistent worker process."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return async_queue.persistent_caller.process


@pytest.fixture
def warm_queue():
    """A persistent queue whose worker was started before the first save, as Megatron does."""
    AsyncCallsQueue.warmup_persistent_caller(rank=0, cpu_shm_mode=True)
    async_queue = AsyncCallsQueue(persistent=True)
    yield async_queue
    async_queue.close(abort=True)


@pytest.mark.timeout(30)
def test_a_rank_with_nothing_to_write_finishes_with_a_started_worker(warm_queue):
    finalize = mock.Mock()
    warm_queue.schedule_async_request(AsyncRequest(None, (), [finalize]))

    assert warm_queue.maybe_finalize_async_calls(blocking=True, no_dist=True) == [0]
    finalize.assert_called_once_with()


@pytest.mark.timeout(30)
def test_a_save_the_worker_dies_in_finishes(warm_queue):
    warm_queue.schedule_async_request(AsyncRequest(os._exit, (1,), [mock.Mock()]))

    assert warm_queue.maybe_finalize_async_calls(blocking=True, no_dist=True) == [0]


@pytest.mark.timeout(30)
def test_close_returns_when_the_worker_has_died(warm_queue):
    warm_queue.schedule_async_request(AsyncRequest(abs, (0,), [mock.Mock()]))
    warm_queue.maybe_finalize_async_calls(blocking=True, no_dist=True)
    worker = _worker(warm_queue)
    worker.kill()
    worker.join()

    warm_queue.close()


def _prepare_shm_save(checkpoint_dir, tensor):
    """Plan a save of `tensor` through CPU shared memory, as a rank's planner resolves it."""
    planner = mock.Mock()
    planner.resolve_data.return_value = tensor
    writer = FileSystemWriterAsync(
        checkpoint_dir, use_cpu_shm_for_gpu_tensors=True, use_cached_data_structure=True
    )
    item = WriteItem(index=MetadataIndex(fqn="tensor"), type=WriteItemType.TENSOR)
    writer.prepare_write_data(SavePlan([item]), planner)
    return writer


@pytest.mark.timeout(30)
def test_reusing_shm_buffers_waits_for_the_write_without_finalizing(tmp_path):
    """Finalizing is collective, and ranks reuse their buffers at different points: only the
    next maybe_finalize_async_calls, which every rank calls together, may finalize."""
    AsyncCallsQueue.warmup_persistent_caller(rank=0, cpu_shm_mode=True)
    async_queue = AsyncCallsQueue(persistent=True, cpu_shm_mode=True)
    finalize = mock.Mock()
    try:
        _prepare_shm_save(tmp_path, torch.tensor(1.0))
        # A write that is still running when the next save reuses the buffers
        async_queue.schedule_async_request(AsyncRequest(time.sleep, (0.5,), [finalize]))

        _prepare_shm_save(tmp_path, torch.tensor(2.0))

        finalize.assert_not_called()
        assert async_queue.maybe_finalize_async_calls(blocking=True, no_dist=True) == [0]
        finalize.assert_called_once_with()
    finally:
        FileSystemWriterAsync.register_shm_drain_callback(None)
        async_queue.close(abort=True)


class _SlowWrites:
    """A file on storage that writes 1 MiB/s, so that a save's write is long."""

    def __init__(self, path: str, mode: str):
        self._file = open(path, mode)

    def write(self, data):
        time.sleep(memoryview(data).nbytes / 2**20)
        return self._file.write(data)

    def __getattr__(self, name):
        return getattr(self._file, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._file.close()


def _slow_write_open(path: str, mode: str = "rb") -> IO[Any]:
    return _SlowWrites(path, mode)


@pytest.mark.distributed
class TestSavesAcrossWorkers:
    def plan(self, checkpoint_dir, state_dict, cpu_shm=False, kept=False, open_file=open):
        """Plan a save as an application does, and return its async request."""
        writer = FileSystemWriterAsync(
            checkpoint_dir,
            open_file=open_file,
            use_cached_data_structure=kept,
            use_cpu_shm_for_gpu_tensors=cpu_shm,
        )
        pending = save_state_dict_async_plan(
            state_dict, writer, None, 0, planner=DefaultSavePlanner()
        )
        save_fn, preload_fn, save_args = writer.get_save_function_and_args()
        return AsyncRequest(
            save_fn,
            save_args,
            [lambda: save_state_dict_async_finalize(pending)],
            preload_fn=preload_fn,
            async_fn_kwargs={},
        )

    def save(self, async_queue, checkpoint_dir, state_dict, **plan_kwargs):
        async_queue.schedule_async_request(self.plan(checkpoint_dir, state_dict, **plan_kwargs))
        async_queue.maybe_finalize_async_calls(blocking=True, no_dist=False)

    def state_dict(self, ntensor=4):
        """A different value in each tensor, the same on every rank, so that its replicated
        items load back."""
        return {
            f"param_{i}": torch.full((64, 64), float(i + 1), device="cuda") for i in range(ntensor)
        }

    def assert_loads(self, checkpoint_dir, state_dict):
        """The checkpoint holds `state_dict`'s values. Loaded into zeros, so that every value
        compared comes from the checkpoint."""
        loaded = {name: torch.zeros_like(tensor) for name, tensor in state_dict.items()}
        load(loaded, storage_reader=FileSystemReader(checkpoint_dir), planner=DefaultLoadPlanner())
        only_loaded, only_saved, mismatched = diff(loaded, state_dict)
        assert not (only_loaded or only_saved or mismatched), (
            f"the checkpoint differs from the saved values: {only_loaded=} {only_saved=} "
            f"{mismatched=}"
        )

    def test_the_second_save_reuses_the_lazily_started_workers_shm_buffers(
        self, tmp_path_dist_ckpt
    ):
        Utils.initialize_distributed()
        FileSystemWriterAsync.cleanup_tensor_caches()
        state_dict = self.state_dict()
        async_queue = AsyncCallsQueue(persistent=True)  # no warmup: the first save starts it
        try:
            with TempNamedDir(tmp_path_dist_ckpt / "first", sync=True) as ckpt_dir:
                self.save(async_queue, ckpt_dir, state_dict, cpu_shm=True, kept=True)
            with (
                TempNamedDir(tmp_path_dist_ckpt / "second", sync=True) as ckpt_dir,
                mock.patch.object(
                    torch.Tensor,
                    "share_memory_",
                    autospec=True,
                    side_effect=torch.Tensor.share_memory_,
                ) as share_memory,
            ):
                request = self.plan(ckpt_dir, state_dict, cpu_shm=True, kept=True)
                assert share_memory.call_count == 0, "the second save allocated new shm buffers"
                async_queue.schedule_async_request(request)
                async_queue.maybe_finalize_async_calls(blocking=True, no_dist=False)
                self.assert_loads(ckpt_dir, state_dict)
        finally:
            async_queue.close()

    @pytest.mark.parametrize("cpu_shm", [True, False], ids=["shm", "ipc"])
    def test_a_save_after_an_abort_reaches_the_new_worker(self, tmp_path_dist_ckpt, cpu_shm):
        """The aborted worker held the save's tensors; the new worker gets them with it."""
        Utils.initialize_distributed()
        FileSystemWriterAsync.cleanup_tensor_caches()
        state_dict = self.state_dict()
        aborted_queue = AsyncCallsQueue(persistent=True)
        try:
            for i in range(2):  # the second save is to a worker that has started already
                with TempNamedDir(tmp_path_dist_ckpt / f"before_{i}", sync=True) as ckpt_dir:
                    self.save(aborted_queue, ckpt_dir, state_dict, cpu_shm=cpu_shm, kept=True)
        finally:
            aborted_queue.close(abort=True)

        for tensor in state_dict.values():
            tensor.add_(10)  # values that neither the old buffers nor the checkpoint have
        async_queue = AsyncCallsQueue(persistent=True)
        try:
            with TempNamedDir(tmp_path_dist_ckpt / "after", sync=True) as ckpt_dir:
                self.save(async_queue, ckpt_dir, state_dict, cpu_shm=cpu_shm, kept=True)
                self.assert_loads(ckpt_dir, state_dict)
        finally:
            async_queue.close()

    @pytest.mark.parametrize("cpu_shm", [True, False], ids=["shm", "ipc"])
    def test_a_save_planned_for_a_worker_that_then_died_reaches_the_new_worker(
        self, tmp_path_dist_ckpt, cpu_shm
    ):
        """The worker dies after the save is planned and before it is sent."""
        Utils.initialize_distributed()
        FileSystemWriterAsync.cleanup_tensor_caches()
        state_dict = self.state_dict()
        async_queue = AsyncCallsQueue(persistent=True)
        try:
            for i in range(2):  # the second save is to a worker that has started already
                with TempNamedDir(tmp_path_dist_ckpt / f"first_{i}", sync=True) as ckpt_dir:
                    self.save(async_queue, ckpt_dir, state_dict, cpu_shm=cpu_shm, kept=True)

            for tensor in state_dict.values():
                tensor.add_(10)  # values that neither the old buffers nor the checkpoint have
            with TempNamedDir(tmp_path_dist_ckpt / "second", sync=True) as ckpt_dir:
                request = self.plan(ckpt_dir, state_dict, cpu_shm=cpu_shm, kept=True)
                worker = _worker(async_queue)
                worker.kill()
                worker.join()
                async_queue.schedule_async_request(request)
                async_queue.maybe_finalize_async_calls(blocking=True, no_dist=False)
                self.assert_loads(ckpt_dir, state_dict)
        finally:
            async_queue.close()

    @pytest.mark.min_world_size(2)
    def test_a_worker_killed_mid_save_fails_the_save_on_every_rank(self, tmp_path_dist_ckpt):
        """Every rank raises CheckpointException for the save, also when the next save is planned
        first and waits for it: no rank hangs or fails with another error."""
        Utils.initialize_distributed()
        FileSystemWriterAsync.cleanup_tensor_caches()
        rank = torch.distributed.get_rank()
        # 8 tensors of 1 MiB: rank 1's write takes 8 s
        state_dict = {
            f"param_{i}": torch.full((512, 512), float(i), device="cuda") for i in range(8)
        }
        async_queue = AsyncCallsQueue(persistent=True, cpu_shm_mode=True)
        try:
            with TempNamedDir(tmp_path_dist_ckpt / "first", sync=True) as ckpt_dir:
                self.save(async_queue, ckpt_dir, state_dict, cpu_shm=True, kept=True)
            with (
                TempNamedDir(tmp_path_dist_ckpt / "killed", sync=True) as ckpt_dir,
                TempNamedDir(tmp_path_dist_ckpt / "next", sync=True) as next_dir,
            ):
                open_file = _slow_write_open if rank == 1 else open
                async_queue.schedule_async_request(
                    self.plan(ckpt_dir, state_dict, cpu_shm=True, kept=True, open_file=open_file)
                )
                if rank == 1:
                    time.sleep(1)  # the worker is in the middle of the write
                    worker = _worker(async_queue)
                    worker.kill()
                    worker.join()
                # Reuses the shm buffers, so it waits for the killed save first
                self.plan(next_dir, state_dict, cpu_shm=True, kept=True)

                with pytest.raises(CheckpointException):
                    async_queue.maybe_finalize_async_calls(blocking=True, no_dist=False)
        finally:
            FileSystemWriterAsync.register_shm_drain_callback(None)
            async_queue.close(abort=True)

    def test_finalize_reads_its_own_write_results(self, tmp_path_dist_ckpt):
        """A save that is never finalized must not leave its write results to the next save,
        which must write a .metadata that describes itself."""
        Utils.initialize_distributed()
        abandoned_queue = AsyncCallsQueue(persistent=True)
        async_queue = AsyncCallsQueue(persistent=True)
        try:
            with (
                TempNamedDir(tmp_path_dist_ckpt / "abandoned", sync=True) as abandoned_dir,
                TempNamedDir(tmp_path_dist_ckpt / "finalized", sync=True) as finalized_dir,
            ):
                abandoned_queue.schedule_async_request(
                    self.plan(abandoned_dir, self.state_dict(ntensor=8))
                )
                while get_write_results_queue().empty():  # the worker has sent its results
                    time.sleep(0.1)
                abandoned_queue.close(abort=True)

                state_dict = self.state_dict(ntensor=10)
                self.save(async_queue, finalized_dir, state_dict)
                self.assert_loads(finalized_dir, state_dict)
        finally:
            abandoned_queue.close(abort=True)
            async_queue.close()
