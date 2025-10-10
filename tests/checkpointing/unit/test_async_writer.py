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
import filecmp
import pickle
from copy import deepcopy
from dataclasses import fields
from typing import IO, Any

import pytest
import torch
from torch.distributed.checkpoint import (
    CheckpointException,
    DefaultLoadPlanner,
    DefaultSavePlanner,
    FileSystemReader,
    FileSystemWriter,
    load,
    save,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from nvidia_resiliency_ext.checkpointing.async_ckpt.core import (
    AsyncCallsQueue,
    AsyncRequest,
    abort_nvrx_checkpoint,
)
from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync
from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import (
    save_state_dict_async_finalize,
    save_state_dict_async_plan,
)
from nvidia_resiliency_ext.checkpointing.async_ckpt.torch_ckpt import TorchAsyncCheckpoint
from nvidia_resiliency_ext.checkpointing.utils import diff
from tests.checkpointing.unit import TempNamedDir
from tests.checkpointing.unit.test_utilities import Model, Utils


def mock_open(
    self,
    path: str,
    mode: str = "rb",
) -> IO[Any]:
    """Function matching the system open() signature that always raises an error."""
    raise OSError('worker critical failure during open()')


class TestAsyncSave:
    def get_async_save_request(self, writer, save_state_dict_ret) -> AsyncRequest:
        """Creates an async save request with a finalization step."""
        save_fn, preload_fn, save_args = writer.get_save_function_and_args()

        def finalize_fn():
            """Finalizes async checkpointing and synchronizes processes."""
            save_state_dict_async_finalize(*save_state_dict_ret)

        return AsyncRequest(save_fn, save_args, [finalize_fn], preload_fn=preload_fn)

    def async_save_checkpoint(
        self,
        checkpoint_dir,
        state_dict,
        planner,
        async_queue: AsyncCallsQueue,
        thread_count=1,
        caching=False,
        open_file=open,
    ):
        """Performs an asynchronous model checkpoint save."""
        writer = FileSystemWriterAsync(
            checkpoint_dir, thread_count=thread_count, open_file=open_file
        )
        coordinator_rank = 0

        save_state_dict_ret = save_state_dict_async_plan(
            state_dict, writer, None, coordinator_rank, planner=planner, enable_cache=caching
        )
        async_request = self.get_async_save_request(writer, save_state_dict_ret)
        async_queue.schedule_async_request(async_request)

    def sync_save_checkpoint(self, checkpoint_dir, state_dict, planner):
        """Performs a synchronous model checkpoint save using FileSystemWriter."""
        save(
            state_dict=state_dict,
            storage_writer=FileSystemWriter(checkpoint_dir),
            planner=planner,
        )

    def async_save_checkpoint_on_rank0(self, checkpoint_dir, state_dict, torch_ckpt_impl):
        if torch.distributed.get_rank() == 0:
            torch_ckpt_impl.async_save(state_dict, checkpoint_dir / 'test')

    def load_checkpoint(self, checkpoint_dir, state_dict):
        """Loads a checkpoint into the given state_dict."""
        load(
            state_dict=state_dict,
            storage_reader=FileSystemReader(checkpoint_dir),
            planner=DefaultLoadPlanner(),
        )
        return state_dict

    def test_async_is_equivalent_to_sync(self, tmp_path_dist_ckpt, async_queue):
        """Verifies that async checkpointing produces the same results as sync checkpointing."""
        Utils.initialize_distributed()
        model = FSDP(Model((1024, 1024), 8))
        with (
            TempNamedDir(tmp_path_dist_ckpt / 'async_checkpoint', sync=True) as async_ckpt_dir,
            TempNamedDir(tmp_path_dist_ckpt / 'sync_checkpoint', sync=True) as sync_ckpt_dir,
        ):
            state_dict = model.state_dict()
            planner = DefaultSavePlanner()

            # Perform async and sync saves
            self.async_save_checkpoint(async_ckpt_dir, state_dict, planner, async_queue)
            self.sync_save_checkpoint(sync_ckpt_dir, state_dict, planner)

            # Finalize async saves
            async_queue.maybe_finalize_async_calls(blocking=True, no_dist=False)

            # Compare saved files
            comparison = filecmp.dircmp(async_ckpt_dir, sync_ckpt_dir)
            assert (
                not comparison.left_only
            ), f"Extra files in async checkpoint: {comparison.left_only}"
            assert (
                not comparison.right_only
            ), f"Extra files in sync checkpoint: {comparison.right_only}"
            assert not comparison.diff_files or comparison.diff_files == [
                '.metadata'
            ], f"Differences found in saved files: {comparison.diff_files}"

            # Load and compare state dicts
            loaded_async_state_dict = self.load_checkpoint(async_ckpt_dir, deepcopy(state_dict))
            loaded_sync_state_dict = self.load_checkpoint(sync_ckpt_dir, deepcopy(state_dict))
            assert loaded_sync_state_dict.keys() == state_dict.keys()
            for key in loaded_sync_state_dict.keys():
                assert key in loaded_async_state_dict, f"Missing key in async checkpoint: {key}"
                assert torch.equal(
                    loaded_async_state_dict[key], loaded_sync_state_dict[key]
                ), f"Mismatch for key '{key}' between async and sync checkpoints."
                assert torch.equal(
                    loaded_sync_state_dict[key], state_dict[key]
                ), f"Mismatch for key '{key}' between async checkpoint and original state_dict."
            async_queue.close()

    def test_errors_are_reported(self, tmp_path_dist_ckpt, async_queue):
        Utils.initialize_distributed()
        rank = torch.distributed.get_rank()
        model = FSDP(Model((1024, 1024), 8))
        state_dict = model.state_dict()
        planner = DefaultSavePlanner()

        if rank == 1:
            open_file = mock_open
        else:
            open_file = open

        with TempNamedDir(tmp_path_dist_ckpt / 'test_errors_are_reported', sync=True) as ckpt_dir:
            self.async_save_checkpoint(
                ckpt_dir, state_dict, planner, async_queue, open_file=open_file
            )
            with pytest.raises(CheckpointException) as exc_info:
                async_queue.maybe_finalize_async_calls(blocking=True, no_dist=False)
            if rank == 0:
                assert 'Worker failure' in str(exc_info.value)
            else:
                assert 'Worker failure' not in str(exc_info.value)

    def test_cached_metadata(self, tmp_path_dist_ckpt, async_queue):
        Utils.initialize_distributed()
        model = FSDP(Model((1024, 1024), 8))
        state_dict_non_cached = model.state_dict()
        state_dict_cached = deepcopy(state_dict_non_cached)
        loaded_non_cached, loaded_cached = None, None
        md_non_cached, md_cached = None, None
        planner = DefaultSavePlanner()

        with TempNamedDir(tmp_path_dist_ckpt / 'ckpt_dir', sync=True) as ckpt_path:
            self.async_save_checkpoint(
                ckpt_path, state_dict_non_cached, planner, async_queue, caching=True
            )
            async_queue.maybe_finalize_async_calls(blocking=True, no_dist=False)
            loaded_non_cached = self.load_checkpoint(ckpt_path, state_dict_non_cached)
            md_path = ckpt_path / '.metadata'
            with md_path.open('rb') as f:
                md_non_cached = pickle.load(f)

        # Run over 3 iterations with cached metadata enabled
        # The 3rd iteration will run with cached metadata
        # `ckpt_dir` at the 3rd iteration 2 will be maintained for comparison
        for i in range(3):
            ckpt_dir = TempNamedDir(tmp_path_dist_ckpt / f'ckpt_dir_{i}_cached', sync=True)
            self.async_save_checkpoint(
                ckpt_dir, state_dict_cached, planner, async_queue, caching=True
            )
            async_queue.maybe_finalize_async_calls(blocking=True, no_dist=False)
            if i < 2:
                ckpt_dir.cleanup()
        loaded_cached = self.load_checkpoint(ckpt_dir, state_dict_cached)
        md_path = ckpt_dir.path() / '.metadata'

        with md_path.open('rb') as f:
            md_cached = pickle.load(f)

        # Check loaded state dict
        diffs = diff(loaded_non_cached, loaded_cached)

        assert not any(
            len(x) for x in diffs
        ), 'Cached metadata doesn\'t produce the same state_dict in loading'
        # Check metadata recorded in .metadata, torch.distributed.metadata.Metadata
        for field in fields(md_non_cached):
            if field.name not in ['storage_data', 'storage_meta']:
                diffs = diff(getattr(md_non_cached, field.name), getattr(md_cached, field.name))
                assert not any(
                    len(x) for x in diffs
                ), f'{field.name} is different in metadata from non-cached, cached metadata impls'
        ckpt_dir.cleanup()
        async_queue.close()

    def test_async_cp_with_multiple_queue_and_abort(self, tmp_path_dist_ckpt):
        """
        Verifies that async checkpointing backend can be used with multiple async queues.
        For example, user may want to save 2 checkpoints i.e. one sharded state and one only on rank-0.
        Verify the abort CP functionality and the ability to resume after an abort operation
        """
        Utils.initialize_distributed()
        model = FSDP(Model((1024, 1024), 8))
        async_queue_dist = AsyncCallsQueue()
        ckpt_impl = TorchAsyncCheckpoint(persistent_queue=True)
        with (
            TempNamedDir(
                tmp_path_dist_ckpt / 'async_checkpoint_dist', sync=True
            ) as async_ckpt_dir_dist,
            TempNamedDir(
                tmp_path_dist_ckpt / 'async_checkpoint_no_dist', sync=True
            ) as async_ckpt_dir_no_dist,
        ):
            state_dict = model.state_dict()
            planner = DefaultSavePlanner()

            # Perform async saves for both dist CP and non-dict CP use cases.
            self.async_save_checkpoint(async_ckpt_dir_dist, state_dict, planner, async_queue_dist)
            self.async_save_checkpoint_on_rank0(async_ckpt_dir_no_dist, state_dict, ckpt_impl)
            async_queue_dist.maybe_finalize_async_calls(blocking=True, no_dist=False)
            ckpt_impl.finalize_async_save(blocking=True, no_dist=True)

            # Abort the CP workers to mock the action of inprocess restarts
            abort_nvrx_checkpoint()

            # validate state of the Async CP workers after abort operation
            async_calls_queue_no_dist = ckpt_impl._get_async_calls_queue()
            assert (
                async_calls_queue_no_dist is not None
            ), "We expect a valid state of AsyncCallsQueue"
            async_process_no_dist = async_calls_queue_no_dist._get_async_caller()
            if async_process_no_dist is not None:
                assert (
                    async_process_no_dist._debug_is_async_process_running() is False
                ), "After abort async process must stop"

            async_process_dist = async_queue_dist._get_async_caller()
            if async_process_dist is not None:
                assert (
                    async_process_dist._debug_is_async_process_running() is False
                ), "After abort async process must stop"

            # Perform async saves for both dist CP and non-dist CP use cases.
            # Validate that operations seamlessly resume after an abort operation
            self.async_save_checkpoint(async_ckpt_dir_dist, state_dict, planner, async_queue_dist)
            self.async_save_checkpoint_on_rank0(async_ckpt_dir_no_dist, state_dict, ckpt_impl)
            async_queue_dist.maybe_finalize_async_calls(blocking=True, no_dist=False)
            ckpt_impl.finalize_async_save(blocking=True, no_dist=True)

            # validate state of the Async CP workers after resume operation
            async_calls_queue_no_dist = ckpt_impl._get_async_calls_queue()
            assert (
                async_calls_queue_no_dist is not None
            ), "We expect a valid state of AsyncCallsQueue object in TorchAsyncCheckpoint after a CP event"
            async_process_no_dist = async_calls_queue_no_dist._get_async_caller()
            # for the non_dist CP use case, only rank-0 is expected to trigger an async process
            if torch.distributed.get_rank() == 0:
                assert (
                    async_process_no_dist is not None
                ), "We expect a valid state of AsyncCaller after a CP event"
                assert (
                    async_process_no_dist._debug_is_async_process_running() is True
                ), "After resume, we expect async process to be running on rank 0 for non dist async save"

            async_process_dist = async_queue_dist._get_async_caller()
            assert (
                async_process_dist is not None
            ), "We expect a valid state of AsyncCaller after a CP event"
            assert (
                async_process_dist._debug_is_async_process_running() is True
            ), "After resume, we expect async process to be running on all ranks for dist async save"

            async_queue_dist.close()
            ckpt_impl.close()

    def test_async_cp_with_multiple_queue_and_abort_followed_by_delete(self, tmp_path_dist_ckpt):
        """
        Test that persistent async CP worker shuts down cleanly after an abort operation.
        This test mocks the behavior of training exiting after an abort triggered by an inprocess restart.
        """
        Utils.initialize_distributed()
        model = FSDP(Model((1024, 1024), 8))
        async_queue_dist = AsyncCallsQueue(persistent=True)
        with (
            TempNamedDir(
                tmp_path_dist_ckpt / 'async_checkpoint_dist', sync=True
            ) as async_ckpt_dir_dist,
        ):
            state_dict = model.state_dict()
            planner = DefaultSavePlanner()

            try:
                # Raise an exception in training process right after async CP request is submitted
                with pytest.raises(RuntimeError) as exc_info:
                    self.async_save_checkpoint(
                        async_ckpt_dir_dist, state_dict, planner, async_queue_dist
                    )
                    raise RuntimeError("Fake exception to mock training process exception")
                    async_queue_dist.maybe_finalize_async_calls(blocking=True, no_dist=False)
            finally:
                # Mock behavior of an abort operation triggered by inprocess restart when exception occurs.
                # Abort the CP workers to mock the action of inprocess restarts
                abort_nvrx_checkpoint()
        # Mock training loop exit which would invoke a __del__ on async queue object
        async_queue_dist.__del__()
