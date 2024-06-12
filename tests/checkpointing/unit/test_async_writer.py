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

import torch
from torch.distributed.checkpoint import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
    FileSystemReader,
    FileSystemWriter,
    load,
    state_dict_saver,
)

from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue, AsyncRequest
from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync
from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import (
    save_state_dict_async_finalize,
    save_state_dict_async_plan,
)

from . import TempNamedDir
from .test_utilities import TestModel, Utils


class TestAsyncSave:
    def get_async_save_request(self, writer, save_state_dict_ret) -> AsyncRequest:
        """Creates an async save request with a finalization step."""
        save_fn, preload_fn, save_args = writer.get_save_function_and_args()

        def finalize_fn():
            """Finalizes async checkpointing and synchronizes processes."""
            save_state_dict_async_finalize(*save_state_dict_ret)
            torch.distributed.barrier()

        return AsyncRequest(save_fn, save_args, [finalize_fn], preload_fn=preload_fn)

    def async_save_checkpoint(
        self, checkpoint_dir, state_dict, planner, async_queue, thread_count=1
    ):
        """Performs an asynchronous model checkpoint save."""
        writer = FileSystemWriterAsync(checkpoint_dir, thread_count=thread_count)
        coordinator_rank = 0

        save_state_dict_ret, *_ = save_state_dict_async_plan(
            state_dict, writer, None, coordinator_rank, planner=planner
        )
        async_request = self.get_async_save_request(writer, save_state_dict_ret)
        async_queue.schedule_async_request(async_request)

    def sync_save_checkpoint(self, checkpoint_dir, state_dict, planner):
        """Performs a synchronous model checkpoint save using FileSystemWriter."""
        state_dict_saver.save(
            state_dict=state_dict,
            storage_writer=FileSystemWriter(checkpoint_dir),
            planner=planner,
        )

    def load_checkpoint(self, checkpoint_dir, state_dict):
        """Loads a checkpoint into the given state_dict."""
        load(
            state_dict=state_dict,
            storage_reader=FileSystemReader(checkpoint_dir),
            planner=DefaultLoadPlanner(),
        )
        return state_dict

    def test_async_is_equivalent_to_sync(self, tmp_path_dist_ckpt):
        """Verifies that async checkpointing produces the same results as sync checkpointing."""
        Utils.initialize_distributed()
        model = TestModel((1024, 1024), 10)
        async_queue = AsyncCallsQueue()

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
            loaded_async_state_dict = self.load_checkpoint(async_ckpt_dir, state_dict.copy())
            loaded_sync_state_dict = self.load_checkpoint(sync_ckpt_dir, state_dict.copy())

            for key in loaded_sync_state_dict.keys():
                assert key in loaded_async_state_dict, f"Missing key in async checkpoint: {key}"
                assert torch.equal(
                    loaded_async_state_dict[key], loaded_sync_state_dict[key]
                ), f"Mismatch for key '{key}' between async and sync checkpoints."
                assert torch.equal(
                    loaded_async_state_dict[key], state_dict[key]
                ), f"Mismatch for key '{key}' between async checkpoint and original state_dict."
