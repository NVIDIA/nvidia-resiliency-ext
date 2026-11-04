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

"""Shared metadata reads, reader isolation, and invalidation between completed loads."""

import gc
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest
from torch.distributed.checkpoint import BytesStorageMetadata, FileSystemReader, Metadata

from nvidia_resiliency_ext.checkpointing.async_ckpt.cached_metadata_filesystem_reader import (
    CachedMetadataFileSystemReader,
)


@pytest.fixture(autouse=True)
def clear_reader_cache():
    """Keep test files and metadata references isolated between tests."""
    CachedMetadataFileSystemReader.clear_metadata_cache()
    yield
    CachedMetadataFileSystemReader.clear_metadata_cache()


def _write_metadata(directory, label="first", filename=".metadata"):
    directory.mkdir(parents=True, exist_ok=True)
    metadata = Metadata(
        state_dict_metadata={label: BytesStorageMetadata()}, planner_data={}, storage_data={}
    )
    path = directory / filename
    with path.open("wb") as metadata_file:
        pickle.dump(metadata, metadata_file)
    return path


@pytest.fixture
def base_reads():
    """Count actual deserializations without replacing the filesystem implementation."""
    with mock.patch.object(
        FileSystemReader, "read_metadata", autospec=True, side_effect=FileSystemReader.read_metadata
    ) as reads:
        yield reads


def test_metadata_copies_preserve_reader_identity(tmp_path, base_reads):
    """Reader IDs and save-side attribute replacement cannot change other readers' metadata."""
    _write_metadata(tmp_path)
    first_reader = CachedMetadataFileSystemReader(tmp_path)
    second_reader = CachedMetadataFileSystemReader(str(tmp_path), cache_metadata=True)
    first = first_reader.read_metadata()
    second = second_reader.read_metadata()
    assert base_reads.call_count == 1
    assert first is not second
    assert first.state_dict_metadata is second.state_dict_metadata
    assert first.storage_meta.load_id == first_reader.load_id
    assert second.storage_meta.load_id == second_reader.load_id
    first.storage_data = {"replacement": None}
    first.storage_meta.load_id = "modified"
    again = second_reader.read_metadata()
    assert again.storage_data == {}
    assert again.storage_meta.load_id == second_reader.load_id


def test_replacement_between_loads_with_same_size_and_mtime(tmp_path, base_reads):
    """A later load must see a completed replacement, even with unchanged size and mtime."""
    path = _write_metadata(tmp_path, "first")
    reader = CachedMetadataFileSystemReader(tmp_path)
    assert "first" in reader.read_metadata().state_dict_metadata
    original_stat = path.stat()
    replacement = _write_metadata(tmp_path, "other", filename=".metadata.new")
    assert replacement.stat().st_size == original_stat.st_size
    os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    replacement.replace(path)
    assert "other" in CachedMetadataFileSystemReader(tmp_path).read_metadata().state_dict_metadata
    assert base_reads.call_count == 2


def test_cache_is_bounded_and_can_be_cleared_by_directory(tmp_path, base_reads):
    """Switching checkpoints evicts the old entry; clearing another directory is harmless."""
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    _write_metadata(first_dir)
    _write_metadata(second_dir)
    first, second = map(CachedMetadataFileSystemReader, (first_dir, second_dir))
    first.read_metadata()
    CachedMetadataFileSystemReader.clear_metadata_cache(second_dir)
    first.read_metadata()
    assert base_reads.call_count == 1
    second.read_metadata()
    first.read_metadata()
    assert base_reads.call_count == 3
    CachedMetadataFileSystemReader.clear_metadata_cache(first_dir)
    first.read_metadata()
    assert base_reads.call_count == 4
    first.reset(second_dir)
    first.read_metadata()
    assert base_reads.call_count == 5


@pytest.mark.skipif(
    not hasattr(FileSystemReader, "_get_metadata_path"), reason="Requires rank-local metadata API"
)
def test_rank_local_metadata(tmp_path, base_reads):
    """Different rank-local metadata files must not alias the directory's cache entry."""
    _write_metadata(tmp_path, "first", "__0.metadata")
    _write_metadata(tmp_path, "other", "__1.metadata")
    reader = CachedMetadataFileSystemReader(tmp_path)
    assert "first" in reader.read_metadata(rank=0).state_dict_metadata
    assert "first" in reader.read_metadata(rank=0).state_dict_metadata
    assert base_reads.call_count == 1
    assert "other" in reader.read_metadata(rank=1).state_dict_metadata
    CachedMetadataFileSystemReader.clear_metadata_cache(tmp_path)
    reader.read_metadata(rank=1)
    assert base_reads.call_count == 3


def test_concurrent_readers(tmp_path, base_reads):
    """Concurrent readers deserialize once and retain separate load identities."""
    _write_metadata(tmp_path)
    readers = [CachedMetadataFileSystemReader(tmp_path) for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        metadatas = list(pool.map(lambda reader: reader.read_metadata(), readers))
    assert base_reads.call_count == 1
    for reader, metadata in zip(readers, metadatas):
        assert metadata.storage_meta.load_id == reader.load_id


@pytest.mark.parametrize("cache_metadata", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_gc_restored_on_success_and_failure(tmp_path, enabled, fail, cache_metadata):
    """Restore the caller's GC setting, including when unpickling raises."""
    _write_metadata(tmp_path)
    original = FileSystemReader.read_metadata

    def read(reader):
        assert not gc.isenabled()
        if fail:
            raise pickle.UnpicklingError("invalid metadata")
        return original(reader)

    previous = gc.isenabled()
    try:
        (gc.enable if enabled else gc.disable)()
        with mock.patch.object(FileSystemReader, "read_metadata", read):
            if fail:
                with pytest.raises(pickle.UnpicklingError):
                    CachedMetadataFileSystemReader(tmp_path, cache_metadata).read_metadata()
            else:
                CachedMetadataFileSystemReader(tmp_path, cache_metadata).read_metadata()
        assert gc.isenabled() == enabled
    finally:
        (gc.enable if previous else gc.disable)()


def test_disabled_cache_reads_fresh_metadata(tmp_path, base_reads):
    """The opt-out bypasses an existing entry and does not evict it."""
    _write_metadata(tmp_path)
    cached = CachedMetadataFileSystemReader(tmp_path)
    cached.read_metadata()
    uncached = CachedMetadataFileSystemReader(tmp_path, cache_metadata=False)
    first = uncached.read_metadata()
    second = uncached.read_metadata()
    assert first is not second
    assert first.state_dict_metadata is not second.state_dict_metadata
    cached.read_metadata()
    assert base_reads.call_count == 3


def test_deleted_metadata_is_not_served_from_cache(tmp_path, base_reads):
    """A later load must report a missing checkpoint instead of returning a stale entry."""
    path = _write_metadata(tmp_path)
    CachedMetadataFileSystemReader(tmp_path).read_metadata()
    path.unlink()
    with pytest.raises(FileNotFoundError):
        CachedMetadataFileSystemReader(tmp_path).read_metadata()
    _write_metadata(tmp_path, "other")
    assert "other" in CachedMetadataFileSystemReader(tmp_path).read_metadata().state_dict_metadata
    assert base_reads.call_count == 3
