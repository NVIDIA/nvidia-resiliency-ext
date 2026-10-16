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

"""The collectives of save_state_dict_async_finalize on two CPU ranks (gloo): which tables the
ranks send, the delivery of the coordinator's fqn list, and failures raised on every rank. The
saves are planned with the plan's own _pending_save, without writing tensors."""

import datetime
import os
import warnings

import numpy as np
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.checkpoint import CheckpointException, FileSystemReader
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, Metadata, MetadataIndex
from torch.distributed.checkpoint.storage import WriteResult
from torch.distributed.checkpoint.utils import _DistWrapper

from nvidia_resiliency_ext.checkpointing.async_ckpt import state_dict_saver
from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler import table, writer
from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync
from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import (
    CheckpointMetadataCache,
    save_state_dict_async_finalize,
)

pytestmark = pytest.mark.skipif(
    not writer.writes_tables(), reason="needs the fast .metadata writer"
)

WORLD_SIZE = 2


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
        state_dict_metadata={f"r{rank}.w": BytesStorageMetadata() for rank in range(WORLD_SIZE)}
    )


class StagedWriter(FileSystemWriterAsync):
    """A storage writer whose save wrote this rank's write results, without writing tensors."""

    def retrieve_write_results(self):
        return write_results(dist.get_rank())


def plan(path, cache, reused):
    """Plan a save into path as save_state_dict_async_plan ends it, building the global
    metadata on the coordinator unless reused."""
    os.makedirs(path, exist_ok=True)
    dist_wrapper = _DistWrapper(None, True, 0)
    storage_writer = StagedWriter(path)
    storage_writer.set_up_storage_writer(dist_wrapper.is_coordinator)
    built = global_metadata() if dist_wrapper.is_coordinator and not reused else None
    return state_dict_saver._pending_save(
        storage_writer, dist_wrapper, cache, None, None, built, reused
    )


def record_fqn_modes(modes):
    """Record in modes the fqn mode of the table each finalize on this rank sends."""
    encode = state_dict_saver._encode_write_results

    def recording(*args, **kwargs):
        payload = encode(*args, **kwargs)
        modes.append(table.decode(np.frombuffer(payload, dtype=np.uint8)).fqn_mode)
        return payload

    state_dict_saver._encode_write_results = recording


def assert_written(path):
    """Assert that path's .metadata holds both ranks' write results."""
    written = FileSystemReader(path).read_metadata()
    expected = [r for rank in range(WORLD_SIZE) for r in write_results(rank)]
    assert written.storage_data == {r.index: r.storage_data for r in expected}


def build_then_reuse(tmp_path, modes):
    """The other rank gets the coordinator's list from the build's finalize, then sends index
    tables."""
    cache = CheckpointMetadataCache()
    for i, reused in enumerate([False, True]):
        save_state_dict_async_finalize(plan(tmp_path / f"save_{i}", cache, reused))
        assert_written(tmp_path / f"save_{i}")
    if dist.get_rank() == 0:
        assert modes == [table.FQN_INDICES] * 2
    else:
        assert modes == [table.FQN_STRINGS, table.FQN_INDICES]


def failed_build_still_delivers(tmp_path, modes):
    """A build whose .metadata write fails raises on every rank, and still delivers its fqn list,
    which the next save, reusing its metadata, encodes against."""
    cache = CheckpointMetadataCache()
    write = state_dict_saver._write_metadata

    def failing(*args):
        raise OSError("disk full")

    state_dict_saver._write_metadata = failing
    with pytest.raises(CheckpointException):
        save_state_dict_async_finalize(plan(tmp_path / "failed", cache, reused=False))
    state_dict_saver._write_metadata = write
    save_state_dict_async_finalize(plan(tmp_path / "next", cache, reused=True))
    assert_written(tmp_path / "next")
    assert modes[-1] == table.FQN_INDICES


def legacy_finalize_with_cache(tmp_path, modes):
    """Unpacked into finalize's old arguments, as Megatron-LM passes it, a save with a cache
    writes .metadata, with every rank sending fqn strings.

    TODO: remove with the deprecated PendingSave.__iter__ and finalize arguments.
    """
    cache = CheckpointMetadataCache()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for i, reused in enumerate([False, True]):
            save_state_dict_async_finalize(*plan(tmp_path / f"legacy_{i}", cache, reused))
            assert_written(tmp_path / f"legacy_{i}")
    assert modes == [table.FQN_STRINGS] * 2


def coordinator_failure_raises_everywhere(tmp_path, modes):
    """A failure on the coordinator after the gather raises on every rank."""

    def unreadable(rows):
        raise RuntimeError("unreadable rows")

    state_dict_saver._decode_payloads = unreadable
    with pytest.raises(CheckpointException):
        save_state_dict_async_finalize(plan(tmp_path / "unreadable", None, reused=False))


def run(rank, scenario, tmp_path, init_file):
    """Run scenario on this rank of a gloo group."""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        modes = []
        record_fqn_modes(modes)
        scenario(tmp_path, modes)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "scenario",
    [
        build_then_reuse,
        failed_build_still_delivers,
        legacy_finalize_with_cache,
        coordinator_failure_raises_everywhere,
    ],
    ids=lambda scenario: scenario.__name__,
)
def test_finalize_on_two_ranks(tmp_path, scenario):
    mp.spawn(run, args=(scenario, tmp_path, tmp_path / "init"), nprocs=WORLD_SIZE, join=True)
