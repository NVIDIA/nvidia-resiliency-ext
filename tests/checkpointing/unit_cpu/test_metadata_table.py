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

"""Write results sent as tables: encoding, decoding, and writing storage_data from them.

The .metadata written from tables must be byte-identical to the one written from the dict that
finish builds out of the same write results, rank by rank.
"""

import dataclasses

import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex
from torch.distributed.checkpoint.storage import WriteResult

from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler import table, writer

from .test_metadata_pickler import dcp_saved_metadata, large_metadata, metadatas

DUMPS = [pytest.param(writer._python_dumps, id="python")]


def split_into_ranks(md: Metadata, ranks: int):
    """md.storage_data as the write results of `ranks` ranks, in order (some may be empty)."""
    items = list(md.storage_data.items())
    bounds = [len(items) * r // ranks for r in range(ranks + 1)]
    return [
        [
            WriteResult(index=idx, size_in_bytes=0, storage_data=info)
            for idx, info in items[bounds[r] : bounds[r + 1]]
        ]
        for r in range(ranks)
    ]


def tables_of(results, pad=0):
    """Each rank's table as received: encoded, padded, decoded from the padded buffer."""
    return [table.decode(table.encode(r) + b"\0" * pad) for r in results]


def assert_tables_write_like_dict(dumps, md: Metadata, ranks: int, pad=0):
    """Writing storage_data from the ranks' tables gives the bytes of writing md itself."""
    tables = tables_of(split_into_ranks(md, ranks), pad)
    from_tables = dumps(dataclasses.replace(md, storage_data=None), storage_tables=tables)
    assert from_tables == dumps(md)
    assert table.to_storage_data(tables) == md.storage_data


@pytest.mark.parametrize("dumps", DUMPS)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(md=metadatas())
def test_tables_write_like_the_dict(dumps, md):
    assume(all(getattr(i, "transform_descriptors", None) is None for i in md.storage_data.values()))
    for ranks in (1, 3):
        assert_tables_write_like_dict(dumps, md, ranks, pad=5)


@pytest.mark.parametrize("dumps", DUMPS)
@pytest.mark.parametrize("make", [large_metadata, dcp_saved_metadata], ids=["large", "dcp-save"])
def test_tables_write_like_the_dict_on(dumps, make, tmp_path):
    md = make(tmp_path) if make is dcp_saved_metadata else make()
    assert_tables_write_like_dict(dumps, md, ranks=7)


@pytest.mark.parametrize("dumps", DUMPS)
def test_index_with_offset_none(dumps):
    """A MetadataIndex whose offset attribute is None, as unpickled from an older checkpoint."""
    idx = MetadataIndex("w", index=2)
    idx.__dict__["offset"] = None
    md = Metadata(state_dict_metadata={}, storage_data={idx: _StorageInfo("__0_0.distcp", 0, 9)})
    assert_tables_write_like_dict(dumps, md, ranks=1)
    (back,) = table.to_storage_data(tables_of(split_into_ranks(md, 1)))
    assert vars(back) == vars(idx)


def _result(idx=None, info=None):
    return WriteResult(
        index=MetadataIndex("w", torch.Size([0])) if idx is None else idx,
        size_in_bytes=1,
        storage_data=_StorageInfo("__0_0.distcp", 0, 1) if info is None else info,
    )


class _Str(str):
    pass


UNENCODABLE = {
    "index is not a MetadataIndex": lambda: _result(idx="w"),
    "fqn is a str subclass": lambda: _result(idx=MetadataIndex(_Str("w"))),
    "offset beyond int64": lambda: _result(info=_StorageInfo("__0_0.distcp", 2**63, 1)),
}
if "transform_descriptors" in {f.name for f in dataclasses.fields(_StorageInfo)}:
    UNENCODABLE["transform descriptors"] = lambda: _result(
        info=_StorageInfo("__0_0.distcp", 0, 1, transform_descriptors=["zstd"])
    )


@pytest.mark.parametrize("make", UNENCODABLE.values(), ids=UNENCODABLE.keys())
def test_unencodable_write_results(make):
    with pytest.raises(table.Unencodable):
        table.encode([make()])
