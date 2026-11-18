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

import time
from unittest import mock

import pytest
import torch
from torch.distributed.checkpoint import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
    FileSystemReader,
    load,
)

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
