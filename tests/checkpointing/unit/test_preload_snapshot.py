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

"""nvbug 6439229 regression tests: CPU tensors received by the async worker share
memory with the trainer's, so they must be copied before training resumes."""

import torch
from torch.distributed.checkpoint.filesystem import _StoragePrefix
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import WriteItem, WriteItemType

from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync
from nvidia_resiliency_ext.checkpointing.utils import preload_tensors


def _tensor_write_items(count):
    return [
        WriteItem(index=MetadataIndex(fqn=f"tensor_{i}"), type=WriteItemType.TENSOR)
        for i in range(count)
    ]


class TestFileSystemWriterPreloadSnapshot:
    def _preload(self, checkpoint_dir, items, tensors):
        # (identifier, data_structure) layout as produced by get_save_function_and_args,
        # with caching disabled and all tensors uncached
        data_to_pass = (None, (None, None, (items, tensors), None, 1, _StoragePrefix("__0_")))
        return FileSystemWriterAsync.preload_tensors(
            (str(checkpoint_dir), data_to_pass), non_blocking=False
        )

    def test_shared_cpu_tensors_are_snapshotted(self, tmp_path):
        """Shared-memory CPU tensors (as received via the worker queue) must be cloned."""
        step = torch.tensor(1.0).share_memory_()
        items = _tensor_write_items(1)

        write_buckets = self._preload(tmp_path, items, [step])

        (bucket,) = write_buckets
        _, _, (_, tensor_data) = bucket
        preloaded = {item.index.fqn: tensor for item, tensor in tensor_data}

        assert (
            preloaded["tensor_0"].untyped_storage().data_ptr() != step.untyped_storage().data_ptr()
        ), "preloaded tensor must not alias the shared-memory buffer"

        # A post-preload in-place mutation must not leak into the snapshot
        step.fill_(2.0)
        assert torch.equal(preloaded["tensor_0"], torch.tensor(1.0))

    def test_private_cpu_tensors_are_not_copied(self, tmp_path):
        """Non-shared CPU tensors (fork-based or synchronous save paths) pass by reference."""
        private = torch.arange(4, dtype=torch.float32)
        items = _tensor_write_items(1)

        write_buckets = self._preload(tmp_path, items, [private])

        (bucket,) = write_buckets
        _, _, (_, tensor_data) = bucket
        preloaded = {item.index.fqn: tensor for item, tensor in tensor_data}

        assert (
            preloaded["tensor_0"].untyped_storage().data_ptr()
            == private.untyped_storage().data_ptr()
        ), "private CPU tensors must not be needlessly copied"


class TestUtilsPreloadSnapshot:
    def test_cpu_tensors_are_cloned(self):
        """preload_tensors must decouple the result from the live CPU tensors."""
        state_dict = {"step": torch.tensor(3.0), "nested": [torch.ones(2)]}

        preloaded = preload_tensors(state_dict, non_blocking=False)

        assert preloaded["step"].data_ptr() != state_dict["step"].data_ptr()
        assert preloaded["nested"][0].data_ptr() != state_dict["nested"][0].data_ptr()

        state_dict["step"].add_(1.0)
        state_dict["nested"][0].add_(1.0)
        assert torch.equal(preloaded["step"], torch.tensor(3.0))
        assert torch.equal(preloaded["nested"][0], torch.ones(2))

    def test_preload_detaches_from_autograd(self):
        state_dict = {"weight": torch.ones(2, requires_grad=True)}

        preloaded = preload_tensors(state_dict, non_blocking=False)

        assert not preloaded["weight"].requires_grad
        assert torch.equal(preloaded["weight"], torch.ones(2))
