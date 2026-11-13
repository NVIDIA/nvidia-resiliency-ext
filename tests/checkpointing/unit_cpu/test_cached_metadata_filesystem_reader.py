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

"""CachedMetadataFileSystemReader on single-process CPU checkpoints.

The cache shows by overwriting a checkpoint on disk: a cached read returns the old keys.
"""

import os

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter

from nvidia_resiliency_ext.checkpointing.async_ckpt.cached_metadata_filesystem_reader import (
    CachedMetadataFileSystemReader as Reader,
)


@pytest.fixture(autouse=True)
def empty_cache():
    Reader.clear_metadata_cache()
    yield
    Reader.clear_metadata_cache()


@pytest.fixture
def checkpoints(tmp_path):
    """Checkpoint a holds x and checkpoint b holds y."""
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    dcp.save({"x": torch.arange(4.0)}, checkpoint_id=a)
    dcp.save({"y": torch.ones(2)}, checkpoint_id=b)
    return a, b


def overwrite(path):
    dcp.save({"z": torch.zeros(1)}, storage_writer=FileSystemWriter(path, overwrite=True))


def keys(reader):
    return set(reader.read_metadata().state_dict_metadata)


def test_cached_reads_ignore_disk_changes(checkpoints, monkeypatch):
    a, _ = checkpoints
    assert keys(Reader(a)) == {"x"}
    overwrite(a)
    monkeypatch.chdir(os.path.dirname(a))
    for reader in (Reader(a), Reader(os.path.basename(a))):
        metadata = reader.read_metadata()
        assert set(metadata.state_dict_metadata) == {"x"}
        assert metadata.storage_meta.load_id == reader.load_id


def test_clear_metadata_cache(checkpoints):
    a, _ = checkpoints
    assert keys(Reader(a)) == {"x"}
    overwrite(a)
    Reader.clear_metadata_cache()
    assert keys(Reader(a)) == {"z"}


def test_uncached_reader_reads_disk(checkpoints):
    a, _ = checkpoints
    reader = Reader(a, cache_metadata=False)
    assert keys(reader) == {"x"}
    overwrite(a)
    assert keys(reader) == {"z"}


def test_reset_reads_new_checkpoint(checkpoints):
    a, b = checkpoints
    reader = Reader(a)
    assert keys(reader) == {"x"}
    reader.reset(b)
    metadata = reader.read_metadata()
    assert set(metadata.state_dict_metadata) == {"y"}
    assert metadata.storage_meta.load_id == reader.load_id


def test_dcp_load(checkpoints):
    a, b = checkpoints
    reader = Reader(a)
    for _ in range(2):
        state = {"x": torch.zeros(4)}
        dcp.load(state, storage_reader=reader)
        assert torch.equal(state["x"], torch.arange(4.0))
    state = {"y": torch.zeros(2)}
    dcp.load(state, checkpoint_id=b, storage_reader=reader)
    assert torch.equal(state["y"], torch.ones(2))
