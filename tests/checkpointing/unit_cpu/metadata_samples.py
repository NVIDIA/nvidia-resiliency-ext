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

"""Fixed DCP metadata samples and ways to make unusual metadata, shared by the .metadata tests."""

from dataclasses import fields

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)

from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler import writer

HAS_TRANSFORMS = "transform_descriptors" in {f.name for f in fields(_StorageInfo)}  # torch 2.8+


# A Metadata field a test can set with with_fields: version where torch has it (2.9+), else
# planner_data, which every version has.
SETTABLE_FIELD = "version" if "version" in {f.name for f in fields(Metadata)} else "planner_data"


def large_metadata() -> Metadata:
    """Return metadata that crosses the writers' size boundaries: batches of 1000 items, memo
    indices past 255, strings longer than 255 bytes, and integers of every width.
    """
    props = TensorProperties(dtype=torch.bfloat16)
    many_chunks = [
        ChunkStorageMetadata(torch.Size([i, 0]), torch.Size([1, 2**40])) for i in range(2500)
    ]
    state_dict_metadata = {
        "w" * 300: TensorStorageMetadata(props, torch.Size([2500, 2**40]), many_chunks),
        **{f"layers.{i}.ünïcode.bias": BytesStorageMetadata() for i in range(1500)},
    }
    lengths = [0, 255, 256, 65535, 65536, 2**31 - 1, 2**31, 2**63 - 1, -1, -(2**31), -(2**63)]
    storage_data = {
        MetadataIndex("w" * 300, torch.Size([i, 0]), i): _StorageInfo(
            f"__{i % 7}_0.distcp", i * 2**33, lengths[i % len(lengths)]
        )
        for i in range(2500)
    }
    return Metadata(state_dict_metadata=state_dict_metadata, storage_data=storage_data)


def dcp_saved_metadata(tmp_path) -> Metadata:
    """Save a state dict of tensors and objects with torch, and return the Metadata it wrote."""
    state_dict = {
        "scalar": torch.tensor(3.0),
        "weight": torch.randn(4, 5, dtype=torch.bfloat16),
        "ids": torch.arange(10),
        "stats": {"step": 7, "name": "run-1"},
    }
    dcp.save(state_dict, storage_writer=dcp.FileSystemWriter(tmp_path))  # no process group
    md = dcp.FileSystemReader(tmp_path).read_metadata()
    assert md.storage_data, "expected storage entries"
    return md


def subclassed_metadata() -> Metadata:
    """Return metadata with objects of subclasses of the metadata classes, which must keep their
    class.
    """

    md = writer._sample_metadata()
    w = md.state_dict_metadata["w"]
    chunks = [SubclassedChunk(torch.Size([0, 0, 0, 0]), torch.Size([1, 1, 1, 1])), *w.chunks]
    md.state_dict_metadata["w2"] = SubclassedTensor(w.properties, w.size, chunks)
    md.storage_data[SubclassedIndex("w2", torch.Size([0, 0, 0, 0]), 0)] = SubclassedInfo(
        "__9_0.distcp", 5, 6
    )
    return md


class SubclassedMetadata(Metadata):
    pass


class SubclassedTensor(TensorStorageMetadata):
    pass


class SubclassedChunk(ChunkStorageMetadata):
    pass


class SubclassedIndex(MetadataIndex):
    pass


class SubclassedInfo(_StorageInfo):
    pass


def properties_of_equal_values() -> Metadata:
    """Return metadata with TensorProperties whose values are equal but of different types: True
    and 1.
    """
    md = writer._sample_metadata()
    w = md.state_dict_metadata["w"]
    md.state_dict_metadata["w1"] = TensorStorageMetadata(
        TensorProperties(dtype=torch.float32, requires_grad=True), w.size, w.chunks
    )
    md.state_dict_metadata["w2"] = TensorStorageMetadata(
        TensorProperties(dtype=torch.float32, requires_grad=1), w.size, w.chunks
    )
    return md


FIXED_METADATA = [
    pytest.param(lambda tmp_path: writer._sample_metadata(), id="sample"),
    pytest.param(lambda tmp_path: large_metadata(), id="large"),
    pytest.param(dcp_saved_metadata, id="dcp-save"),
    pytest.param(lambda tmp_path: subclassed_metadata(), id="subclasses"),
    pytest.param(lambda tmp_path: properties_of_equal_values(), id="equal-properties"),
]


def set_index(md, value):
    object.__setattr__(next(iter(md.storage_data)), "index", value)  # MetadataIndex is frozen


class SubclassedStr(str):
    pass


def rename_first(md, rename):
    fqn = next(iter(md.state_dict_metadata))
    md.state_dict_metadata[rename(fqn)] = md.state_dict_metadata.pop(fqn)


# Objects whose pickling changes the metadata being written, through a class attribute set by the
# test. The native writer must raise or write a loadable pickle, never crash or write garbage.
class EmptiesChunkList(ChunkStorageMetadata):
    target = None

    def __reduce_ex__(self, protocol):
        type(self).target.clear()
        return super().__reduce_ex__(protocol)


class ReplacesChunkList(TensorProperties):
    target = None

    def __reduce_ex__(self, protocol):
        type(self).target.chunks = []  # frees the list being written
        return super().__reduce_ex__(protocol)


class DeletesEntries(BytesStorageMetadata):
    target = None

    def __reduce_ex__(self, protocol):
        for fqn in [fqn for fqn in type(self).target if fqn.startswith("obj.")][:250]:
            del type(self).target[fqn]
        return super().__reduce_ex__(protocol)


class ReplacesStorageInfo(MetadataIndex):
    target = None

    def __reduce_ex__(self, protocol):
        type(self).target[self] = _StorageInfo("__9_9.distcp", 0, 1)  # frees the old value
        return super().__reduce_ex__(protocol)


def metadata_changed_while_pickled(case):
    md = writer._sample_metadata()
    w = md.state_dict_metadata["w"]
    if case == "chunk-list-emptied":
        w.chunks = [EmptiesChunkList(torch.Size([0]), torch.Size([1])), *w.chunks * 1000]
        EmptiesChunkList.target = w.chunks
    elif case == "chunk-list-replaced":
        w.properties = ReplacesChunkList(dtype=torch.float32)
        w.chunks = w.chunks * 2500
        ReplacesChunkList.target = w
    elif case == "entries-deleted":
        md.state_dict_metadata = {"first": DeletesEntries(), **md.state_dict_metadata}
        DeletesEntries.target = md.state_dict_metadata
    elif case == "storage-info-replaced":
        index = ReplacesStorageInfo("w2", torch.Size([0, 0, 0, 0]), 0)
        md.storage_data = {index: _StorageInfo("__0_0.distcp", 0, 1), **md.storage_data}
        ReplacesStorageInfo.target = md.storage_data
    return md


CHANGED_WHILE_PICKLED = [
    "chunk-list-emptied",
    "chunk-list-replaced",
    "entries-deleted",
    "storage-info-replaced",
]
