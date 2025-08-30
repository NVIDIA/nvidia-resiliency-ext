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

import torch
from torch.distributed.checkpoint import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
    FileSystemReader,
    FileSystemWriter,
    load,
    save,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue, AsyncRequest
from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync
from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import (
    save_state_dict_async_finalize,
    save_state_dict_async_plan,
)
from nvidia_resiliency_ext.checkpointing.utils import diff
from tests.checkpointing.unit import TempNamedDir
from tests.checkpointing.unit.test_utilities import TestModel, Utils


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
        self, checkpoint_dir, state_dict, planner, async_queue, thread_count=1, caching=False
    ):
        """Performs an asynchronous model checkpoint save."""
        writer = FileSystemWriterAsync(checkpoint_dir, thread_count=thread_count)
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
        model = FSDP(TestModel((1024, 1024), 8))
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

    def test_cached_metadata(self, tmp_path_dist_ckpt):
        Utils.initialize_distributed()
        async_queue = AsyncCallsQueue()

        model = FSDP(TestModel((1024, 1024), 8))
        state_dict_non_cached = model.state_dict()
        state_dict_cached = deepcopy(state_dict_non_cached)
        loaded_non_cached, loaded_cached = None, None
        md_non_cached, md_cached = None, None
        planner = DefaultSavePlanner()

        with TempNamedDir(tmp_path_dist_ckpt / 'ckpt_dir', sync=True) as ckpt_dir:
            self.async_save_checkpoint(
                ckpt_dir, state_dict_non_cached, planner, async_queue, caching=True
            )
            async_queue.maybe_finalize_async_calls(blocking=True, no_dist=False)
            loaded_non_cached = self.load_checkpoint(ckpt_dir, state_dict_non_cached)
            md_path = ckpt_dir.__enter__() / '.metadata'
            with md_path.open('rb') as f:
                md_non_cached = pickle.load(f)

        # Run over 3 iterations with cached metadata enabled
        # The 3rd iteration will run with cached metadata
        # `ckpt_dir` at the 3rd iteration 2 will be maintained for comparison
        ckpt_dir = None
        for i in range(3):
            ckpt_dir = TempNamedDir(tmp_path_dist_ckpt / f'ckpt_dir_{i}_cached', sync=True)
            self.async_save_checkpoint(
                ckpt_dir.__enter__(), state_dict_cached, planner, async_queue, caching=True
            )
            async_queue.maybe_finalize_async_calls(blocking=True, no_dist=False)
            if i < 2:
                ckpt_dir.cleanup()
        loaded_cached = self.load_checkpoint(ckpt_dir.__enter__(), state_dict_cached)
        md_path = ckpt_dir.__enter__() / '.metadata'

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
