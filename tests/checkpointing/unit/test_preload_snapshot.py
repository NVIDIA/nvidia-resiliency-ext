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


def _tensor_write_items(count):
    return [
        WriteItem(index=MetadataIndex(fqn=f"tensor_{i}"), type=WriteItemType.TENSOR)
        for i in range(count)
    ]


class TestFileSystemWriterPrepareSnapshot:
    def _prepare_writer(self, checkpoint_dir, tensors, **writer_kwargs):
        planner = Mock()
        planner.resolve_data.side_effect = tensors
        writer = FileSystemWriterAsync(checkpoint_dir, **writer_kwargs)
        writer.prepare_write_data(SavePlan(_tensor_write_items(len(tensors))), planner)
        return writer

    def _prepare(self, checkpoint_dir, tensors):
        writer = self._prepare_writer(checkpoint_dir, tensors)
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

    def test_cpu_shm_snapshots_and_reuses_cpu_tensor_storage(self, tmp_path, monkeypatch):
        """CPU tensors use the reusable snapshot cache in CPU shared-memory mode."""
        cuda_synchronize = Mock()
        monkeypatch.setattr(torch.cuda, "synchronize", cuda_synchronize)
        drain = Mock()
        FileSystemWriterAsync.cleanup_tensor_caches()
        FileSystemWriterAsync.register_shm_drain_callback(drain)
        source = torch.tensor(1.0)

        try:
            first_writer = self._prepare_writer(
                tmp_path,
                [source],
                use_cached_data_structure=True,
                use_cpu_shm_for_gpu_tensors=True,
            )

            assert first_writer.uncached_tensor_data is None
            _, first_snapshots = first_writer.cached_tensor_data
            (first_snapshot,) = first_snapshots
            assert first_snapshot.untyped_storage().is_shared()
            assert first_snapshot.data_ptr() != source.data_ptr()

            source.fill_(2.0)
            assert torch.equal(first_snapshot, torch.tensor(1.0))

            second_writer = self._prepare_writer(
                tmp_path,
                [source],
                use_cached_data_structure=True,
                use_cpu_shm_for_gpu_tensors=True,
            )

            assert second_writer.uncached_tensor_data is None
            assert second_writer.cached_tensor_data is None
            _, cached_snapshots = next(iter(FileSystemWriterAsync._shm_tensor_cache.values()))
            (reused_snapshot,) = cached_snapshots
            assert reused_snapshot.data_ptr() == first_snapshot.data_ptr()
            assert torch.equal(reused_snapshot, torch.tensor(2.0))
            drain.assert_called_once_with()
            cuda_synchronize.assert_not_called()
        finally:
            FileSystemWriterAsync.register_shm_drain_callback(None)
            FileSystemWriterAsync.cleanup_tensor_caches()
