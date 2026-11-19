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

import pytest
import torch
from torch.distributed.checkpoint import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
    FileSystemReader,
    FileSystemWriter,
    load,
    state_dict_saver,
)

from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncRequest
from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync
from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import (
    save_state_dict_async_finalize,
    save_state_dict_async_plan,
)

from . import TempNamedDir
from .test_utilities import Model, Utils

pytestmark = pytest.mark.distributed


class TestAsyncSaveWithMSC:

    def get_async_save_request(self, writer, pending) -> AsyncRequest:
        """Create an async save request with a finalization step."""
        save_fn, preload_fn, save_args = writer.get_save_function_and_args()

        def finalize_fn():
            """Finalizes async checkpointing and synchronizes processes."""
            save_state_dict_async_finalize(pending)
            torch.distributed.barrier()

        return AsyncRequest(
            save_fn, save_args, [finalize_fn], preload_fn=preload_fn, async_fn_kwargs={}
        )

    def async_save_checkpoint(
        self, checkpoint_dir, state_dict, planner, async_queue, thread_count=1
    ):
        """Performs an asynchronous model checkpoint save."""
        writer = FileSystemWriterAsync(checkpoint_dir, thread_count=thread_count, use_msc=True)
        coordinator_rank = 0

        pending = save_state_dict_async_plan(
            state_dict, writer, None, coordinator_rank, planner=planner
        )
        async_request = self.get_async_save_request(writer, pending)
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

    def test_writer_options_require_msc(self, tmp_path_dist_ckpt):
        with pytest.raises(ValueError, match='use_msc'):
            FileSystemWriterAsync(tmp_path_dist_ckpt, writer_options=object())

    def test_writer_options_require_streaming_msc(self, tmp_path_dist_ckpt, monkeypatch):
        msc = pytest.importorskip('multistorageclient')
        monkeypatch.delattr(msc, 'StreamingWriterOptions', raising=False)
        with pytest.raises(ValueError, match='StreamingWriterOptions'):
            FileSystemWriterAsync(tmp_path_dist_ckpt, use_msc=True, writer_options=object())

    def test_writer_options_reach_msc_open(self, tmp_path_dist_ckpt, monkeypatch):
        """Verifies that shard files are opened through msc.open with the writer options."""
        msc = pytest.importorskip('multistorageclient')
        Utils.initialize_distributed()
        # Same values on every rank: DCP saves each replicated tensor from one rank only.
        torch.manual_seed(0)
        model = Model((1024, 1024), 10)
        options = object()
        opened = []
        msc_open = msc.open

        def recording_open(path, mode, writer_options=None, **kwargs):
            opened.append((str(path), mode, writer_options))
            return msc_open(path, mode, **kwargs)

        monkeypatch.setattr(msc, 'open', recording_open)
        # Lets the test run on MSC releases that predate StreamingWriterOptions.
        monkeypatch.setattr(msc, 'StreamingWriterOptions', object, raising=False)
        with TempNamedDir(tmp_path_dist_ckpt / 'writer_options', sync=True) as ckpt_dir:
            writer = FileSystemWriterAsync(ckpt_dir, use_msc=True, writer_options=options)
            pending = save_state_dict_async_plan(
                model.state_dict(), writer, None, 0, planner=DefaultSavePlanner()
            )
            save_fn, preload_fn, (rank, _, results_queue) = writer.get_save_function_and_args()
            buckets = preload_fn()
            torch.cuda.synchronize()
            save_fn(rank, buckets, results_queue)
            save_state_dict_async_finalize(pending)
            torch.distributed.barrier()

            shards = [entry for entry in opened if entry[0].endswith('.distcp')]
            assert shards and all(entry[1:] == ('wb', options) for entry in shards)
            empty = {key: torch.zeros_like(tensor) for key, tensor in model.state_dict().items()}
            loaded = self.load_checkpoint(ckpt_dir, empty)
            for key, tensor in model.state_dict().items():
                assert torch.equal(loaded[key], tensor), key

    def test_async_is_equivalent_to_sync(self, tmp_path_dist_ckpt, async_queue):
        """Verifies that async checkpointing produces the same results as sync checkpointing."""
        Utils.initialize_distributed()
        model = Model((1024, 1024), 10)

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
            async_queue.close()
