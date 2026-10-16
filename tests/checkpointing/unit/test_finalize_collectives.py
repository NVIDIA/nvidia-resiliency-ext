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

"""The collectives of save_state_dict_async_finalize on two ranks: which tables the ranks send, the
delivery of the coordinator's fqn list, and failures raised on every rank. The saves are planned
with the plan's own _pending_save, without writing tensors."""

import warnings

import numpy as np
import pytest
import torch
from torch.distributed.checkpoint import CheckpointException, FileSystemReader
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, Metadata, MetadataIndex
from torch.distributed.checkpoint.storage import WriteResult
from torch.distributed.checkpoint.utils import _DistWrapper

from nvidia_resiliency_ext.checkpointing.async_ckpt import state_dict_saver
from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler import (
    write_result_table,
    writer,
)
from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync
from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import (
    CheckpointMetadataCache,
    save_state_dict_async_finalize,
)
from tests.checkpointing.unit import TempNamedDir
from tests.checkpointing.unit.test_utilities import Utils

NEEDS_TABLES = pytest.mark.skipif(
    not writer.writes_tables(), reason="needs the fast .metadata writer"
)


def write_results(rank: int):
    """Return the write results of a rank that writes one bytes entry."""
    return [
        WriteResult(
            index=MetadataIndex(f"r{rank}.w"),
            size_in_bytes=4,
            storage_data=_StorageInfo(f"__{rank}_0.distcp", 0, 4),
        )
    ]


def global_metadata() -> Metadata:
    """Return the global metadata of the ranks' bytes entries."""
    return Metadata(
        state_dict_metadata={
            f"r{rank}.w": BytesStorageMetadata() for rank in range(Utils.world_size)
        }
    )


class StagedWriter(FileSystemWriterAsync):
    """A storage writer whose save wrote this rank's write results, without writing tensors."""

    def retrieve_write_results(self):
        return write_results(torch.distributed.get_rank())


def plan(path, cache, reused, group=None):
    """Plan a save into path as save_state_dict_async_plan ends it, over group, building the
    global metadata on the coordinator unless reused."""
    dist_wrapper = _DistWrapper(group, True, 0)
    storage_writer = StagedWriter(path)
    storage_writer.set_up_storage_writer(dist_wrapper.is_coordinator)
    built = global_metadata() if dist_wrapper.is_coordinator and not reused else None
    return state_dict_saver._pending_save(
        storage_writer, dist_wrapper, cache, None, None, built, reused
    )


@pytest.fixture
def modes(monkeypatch):
    """Record the fqn mode of the table each finalize on this rank sends."""
    Utils.initialize_distributed()
    recorded = []
    encode = state_dict_saver._encode_write_results

    def recording(*args, **kwargs):
        payload = encode(*args, **kwargs)
        recorded.append(write_result_table.decode(np.frombuffer(payload, dtype=np.uint8)).fqn_mode)
        return payload

    monkeypatch.setattr(state_dict_saver, "_encode_write_results", recording)
    return recorded


def assert_written(path):
    """Assert that path's .metadata holds every rank's write results."""
    written = FileSystemReader(path).read_metadata()
    expected = [r for rank in range(Utils.world_size) for r in write_results(rank)]
    assert written.storage_data == {r.index: r.storage_data for r in expected}


@NEEDS_TABLES
def test_build_then_reuse(tmp_path_dist_ckpt, modes):
    """The other ranks get the coordinator's list from the build's finalize, then send index
    tables."""
    cache = CheckpointMetadataCache()
    for i, reused in enumerate([False, True]):
        with TempNamedDir(tmp_path_dist_ckpt / f"build_then_reuse_{i}") as ckpt_dir:
            save_state_dict_async_finalize(plan(ckpt_dir, cache, reused))
            assert_written(ckpt_dir)
    if torch.distributed.get_rank() == 0:
        assert modes == [write_result_table.FQN_INDICES] * 2
    else:
        assert modes == [write_result_table.FQN_STRINGS, write_result_table.FQN_INDICES]


@NEEDS_TABLES
def test_failed_build_still_delivers(tmp_path_dist_ckpt, modes, monkeypatch):
    """A build whose .metadata write fails raises on every rank, and still delivers its fqn list,
    which the next save, reusing its metadata, encodes against."""
    cache = CheckpointMetadataCache()
    write = state_dict_saver._write_metadata

    def failing(*args):
        raise OSError("disk full")

    monkeypatch.setattr(state_dict_saver, "_write_metadata", failing)
    with TempNamedDir(tmp_path_dist_ckpt / "failed_build") as ckpt_dir:
        with pytest.raises(CheckpointException):
            save_state_dict_async_finalize(plan(ckpt_dir, cache, reused=False))
    monkeypatch.setattr(state_dict_saver, "_write_metadata", write)
    with TempNamedDir(tmp_path_dist_ckpt / "after_failed_build") as ckpt_dir:
        save_state_dict_async_finalize(plan(ckpt_dir, cache, reused=True))
        assert_written(ckpt_dir)
    assert modes[-1] == write_result_table.FQN_INDICES


@NEEDS_TABLES
def test_legacy_finalize_with_cache(tmp_path_dist_ckpt, modes):
    """Unpacked into finalize's old arguments, a save with a cache writes .metadata, with every
    rank sending fqn strings.

    TODO: remove with the deprecated PendingSave.__iter__ and finalize arguments.
    """
    cache = CheckpointMetadataCache()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for i, reused in enumerate([False, True]):
            with TempNamedDir(tmp_path_dist_ckpt / f"legacy_{i}") as ckpt_dir:
                save_state_dict_async_finalize(*plan(ckpt_dir, cache, reused))
                assert_written(ckpt_dir)
    assert modes == [write_result_table.FQN_STRINGS] * 2


@NEEDS_TABLES
def test_coordinator_failure_raises_everywhere(tmp_path_dist_ckpt, modes, monkeypatch):
    """A failure on the coordinator after the gather raises on every rank."""

    def unreadable(rows):
        raise RuntimeError("unreadable rows")

    monkeypatch.setattr(state_dict_saver, "_decode_payloads", unreadable)
    with TempNamedDir(tmp_path_dist_ckpt / "unreadable_rows") as ckpt_dir:
        with pytest.raises(CheckpointException):
            save_state_dict_async_finalize(plan(ckpt_dir, None, reused=False))


def test_group_whose_coordinator_isnt_global_rank_0(tmp_path_dist_ckpt):
    """Saving over a group of only the last rank: the collectives address its coordinator by its
    global rank."""
    Utils.initialize_distributed()
    last = Utils.world_size - 1
    group = torch.distributed.new_group([last])
    with TempNamedDir(tmp_path_dist_ckpt / "group") as ckpt_dir:
        if torch.distributed.get_rank() == last:
            save_state_dict_async_finalize(plan(ckpt_dir, None, reused=False, group=group))
            written = FileSystemReader(ckpt_dir).read_metadata()
            assert written.storage_data == {r.index: r.storage_data for r in write_results(last)}
        torch.distributed.barrier()
