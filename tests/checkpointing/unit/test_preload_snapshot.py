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

import queue
from unittest.mock import Mock

import torch
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import SavePlan, WriteItem, WriteItemType

from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import (
    FileSystemWriterAsync,
    _WriteData,
)
from nvidia_resiliency_ext.checkpointing.async_ckpt.staging import (
    ConsistentDataIdentifier,
    PerSaveShmBuffers,
    PersistentShmBuffers,
    ShmStager,
    StagedTensors,
)


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
        _, snapshotted = writer._write_data.tensors.uncached
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
        buffers = PersistentShmBuffers(drain)
        stager = ShmStager(buffers)
        source = torch.tensor(1.0)

        first_writer = self._prepare_writer(tmp_path, [source], stager=stager)

        assert first_writer._write_data.tensors.uncached is None
        _, first_snapshots = first_writer._write_data.tensors.cached
        (first_snapshot,) = first_snapshots
        assert first_snapshot.untyped_storage().is_shared()
        assert first_snapshot.data_ptr() != source.data_ptr()

        source.fill_(2.0)
        assert torch.equal(first_snapshot, torch.tensor(1.0))

        second_writer = self._prepare_writer(tmp_path, [source], stager=stager)

        assert second_writer._write_data.tensors.uncached is None
        assert second_writer._write_data.tensors.cached is None
        (cached_snapshots,) = buffers._buffers.values()
        (reused_snapshot,) = cached_snapshots
        assert reused_snapshot.data_ptr() == first_snapshot.data_ptr()
        assert torch.equal(reused_snapshot, torch.tensor(2.0))
        drain.assert_called_once_with()
        cuda_synchronize.assert_not_called()

    def test_a_given_stager_uses_the_registered_drain(self, tmp_path):
        """Buffers a writer is given drain like the default ones before they are reused."""
        drain = Mock()
        stager = ShmStager(PersistentShmBuffers())
        FileSystemWriterAsync.register_shm_drain_callback(drain)
        try:
            for _ in range(2):
                self._prepare_writer(tmp_path, [torch.tensor(1.0)], stager=stager)
        finally:
            FileSystemWriterAsync.register_shm_drain_callback(None)
        drain.assert_called_once_with()  # before the second save reuses the buffers

    def test_new_buffers_for_every_save_wait_for_the_writes_in_flight(self, tmp_path):
        """Otherwise saves scheduled faster than they are written would pile up in shared
        memory."""
        drain = Mock()
        stager = ShmStager(PerSaveShmBuffers(drain))
        for _ in range(2):
            self._prepare_writer(tmp_path, [torch.tensor(1.0)], stager=stager)
        assert drain.call_count == 2


def test_a_save_that_cant_be_made_into_buckets_is_reported_as_failed(tmp_path):
    """The error reaches finalize like a failed write, rather than ending the worker."""
    # Only the key of tensors this process never received
    missing = StagedTensors(ConsistentDataIdentifier('never sent'), None, None)
    write_data = _WriteData(str(tmp_path), None, missing, None, 1, None, save_id=7)
    results, write_buckets = queue.Queue(), Mock()

    FileSystemWriterAsync.write_staged_data(write_buckets, 0, write_data, results)

    save_id, error = results.get_nowait()
    assert save_id == 7 and isinstance(error, RuntimeError)
    write_buckets.assert_not_called()
