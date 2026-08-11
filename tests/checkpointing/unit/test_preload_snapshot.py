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

"""nvbug 6439229 regression tests for async CPU tensor snapshots."""

from unittest.mock import Mock

import torch
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import SavePlan, WriteItem, WriteItemType

from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync
from nvidia_resiliency_ext.checkpointing.utils import preload_tensors


def _tensor_write_items(count):
    return [
        WriteItem(index=MetadataIndex(fqn=f"tensor_{i}"), type=WriteItemType.TENSOR)
        for i in range(count)
    ]


class TestFileSystemWriterPrepareSnapshot:
    def _prepare(self, checkpoint_dir, tensors):
        planner = Mock()
        planner.resolve_data.side_effect = tensors
        writer = FileSystemWriterAsync(checkpoint_dir)
        writer.prepare_write_data(SavePlan(_tensor_write_items(len(tensors))), planner)
        _, snapshotted = writer.uncached_tensor_data
        return snapshotted

    def test_cpu_tensors_are_snapshotted_before_queueing(self, tmp_path):
        """Resolved CPU tensors must not retain aliases to live training state."""
        step = torch.tensor(1.0)
        counter = torch.arange(4, dtype=torch.float32)

        snapshotted = self._prepare(tmp_path, [step, counter])

        assert snapshotted[0].data_ptr() != step.data_ptr()
        assert snapshotted[1].data_ptr() != counter.data_ptr()

        step.fill_(2.0)
        counter.add_(1.0)
        assert torch.equal(snapshotted[0], torch.tensor(1.0))
        assert torch.equal(snapshotted[1], torch.arange(4, dtype=torch.float32))

    def test_prepare_snapshot_detaches_from_autograd(self, tmp_path):
        tensor = torch.ones(2, requires_grad=True)

        (snapshotted,) = self._prepare(tmp_path, [tensor])

        assert not snapshotted.requires_grad
        assert torch.equal(snapshotted, tensor)


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
