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
import io

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, Metadata, MetadataIndex
from torch.distributed.checkpoint.storage import WriteResult

from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler import pickler, table, writer
from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler.state_dict_table import (
    StateDictTable,
)
from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync

from .metadata_samples import dcp_saved_metadata, large_metadata
from .metadata_strategies import metadatas

DUMPS = [
    pytest.param(pickler.dumps, id="python"),
    pytest.param(
        writer.native.dumps if writer.native else None,
        id="native",
        marks=pytest.mark.skipif(writer.native is None, reason="native writer not built"),
    ),
]


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


def rows_of(results, pad=0):
    """The ranks' tables as the coordinator receives them: one zero-padded row per rank."""
    encoded = [table.encode(r) for r in results]
    width = max(len(e) for e in encoded) + pad
    return np.stack([np.frombuffer(e.ljust(width, b"\0"), dtype=np.uint8) for e in encoded])


def assert_tables_write_like_dict(dumps, md: Metadata, ranks: int, pad=0):
    """Writing storage_data from the ranks' tables gives the bytes of writing md itself."""
    rows = rows_of(split_into_ranks(md, ranks), pad)
    from_tables = dumps(dataclasses.replace(md, storage_data=None), storage_rows=rows)
    assert from_tables == dumps(md)
    assert table.to_storage_data(table.decode_rows(rows)) == md.storage_data


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
    (back,) = table.to_storage_data(table.decode_rows(rows_of(split_into_ranks(md, 1))))
    assert vars(back) == vars(idx)


def _result(idx=None, info=None):
    return WriteResult(
        index=MetadataIndex("w", torch.Size([0])) if idx is None else idx,
        size_in_bytes=1,
        storage_data=_StorageInfo("__0_0.distcp", 0, 1) if info is None else info,
    )


def _fields(results):
    """Everything in each write result, index.index too, which MetadataIndex equality ignores."""
    return [(vars(r.index), r.size_in_bytes, vars(r.storage_data)) for r in results]


def test_to_write_results_gives_them_back():
    """Each write result, in order, also two with equal indexes (from one rank, as a planner that
    doesn't deduplicate may give)."""
    results = [
        _result(),
        _result(
            idx=MetadataIndex("w", torch.Size([0]), 3), info=_StorageInfo("__0_1.distcp", 5, 1)
        ),
        _result(idx=MetadataIndex("b")),
    ]
    assert _fields(table.to_write_results(table.decode(table.encode(results)))) == _fields(results)


def test_torch_write_results_fit_the_table(tmp_path):
    """What the table assumes of torch: the write results of FileSystemWriterAsync's write path
    (torch's _write_item) have no transform descriptors, and size_in_bytes is the storage length,
    so they come back from a table unchanged. Check it passes on a torch release before adding it
    to TESTED_TORCH_VERSIONS."""
    state_dict = {"t": torch.arange(12.0).reshape(3, 4), "b": io.BytesIO(b"bytes"), "n": 7}
    planner = DefaultSavePlanner()
    planner.set_up_planner(state_dict, is_coordinator=True)
    plan = planner.create_local_plan()
    items = [(item, planner.resolve_data(item)) for item in plan.items]
    tensors = [(item, data) for item, data in items if isinstance(data, torch.Tensor)]
    others = [(item, data) for item, data in items if not isinstance(data, torch.Tensor)]
    writer = FileSystemWriterAsync(tmp_path)
    results = FileSystemWriterAsync._write_bucket_to_storage(
        [writer.transforms] if hasattr(writer, "transforms") else [],
        open,
        (str(tmp_path / "__0_0.distcp"), "__0_0.distcp", (others, tensors)),
        False,
        False,
    )
    assert len(results) == 3
    assert _fields(table.to_write_results(table.decode(table.encode(results)))) == _fields(results)


class SubclassedStr(str):
    pass


UNENCODABLE = {
    "index is not a MetadataIndex": lambda: _result(idx="w"),
    "fqn is a str subclass": lambda: _result(idx=MetadataIndex(SubclassedStr("w"))),
    "offset beyond int64": lambda: _result(info=_StorageInfo("__0_0.distcp", 2**63, 1)),
    "fqn not UTF-8": lambda: _result(idx=MetadataIndex("w\ud800", torch.Size([0]))),
}
if "transform_descriptors" in {f.name for f in dataclasses.fields(_StorageInfo)}:
    UNENCODABLE["transform descriptors"] = lambda: _result(
        info=_StorageInfo("__0_0.distcp", 0, 1, transform_descriptors=["zstd"])
    )


@pytest.mark.parametrize("make", UNENCODABLE.values(), ids=UNENCODABLE.keys())
def test_unencodable_write_results(make):
    with pytest.raises(table.Unencodable):
        table.encode([make()])


def _corrupt(mutate, results=None):
    """The row of a table of results (one write result by default), after mutate(header,
    string_len, entry) changes those sections of it in place."""
    raw = np.frombuffer(bytearray(table.encode(results or [_result()])), dtype=np.uint8)
    header = raw[: 8 * table.HEADER].view("<i8")
    n_entries, n_strings = int(header[2]), int(header[3])
    pos = 8 * table.HEADER
    string_len = raw[pos : pos + 8 * n_strings].view("<i8")
    pos += 8 * n_strings
    entry = raw[pos : pos + 8 * table.COLUMNS * n_entries].view("<i8").reshape(n_entries, -1)
    mutate(header, string_len, entry)
    return raw[None, :]


def _set(array, index, value):
    array[index] = value


# Offsets that wrap around int64 when summed: 4 * 2**62 + 5 == 5 (mod 2**64), the 5 dims there are.
_FIVE_DIMS = [_result(idx=MetadataIndex("w", torch.Size([i]))) for i in range(5)]

CORRUPT = {
    "fqn-id": lambda: _corrupt(lambda h, s, e: _set(e, (0, table.FQN), 5)),
    "path-id": lambda: _corrupt(lambda h, s, e: _set(e, (0, table.PATH), -1)),
    "offset-ndim": lambda: _corrupt(lambda h, s, e: _set(e, (0, table.OFFSET_NDIM), 9)),
    "offset-ndim-wraps": lambda: _corrupt(
        lambda h, s, e: _set(e, (slice(None), table.OFFSET_NDIM), [2**62] * 4 + [5]), _FIVE_DIMS
    ),
    "unknown-flag": lambda: _corrupt(lambda h, s, e: _set(e, (0, table.FLAGS), 0x8 | 0x7)),
    "size-without-offset": lambda: _corrupt(
        lambda h, s, e: _set(e, (0, table.FLAGS), table.FLAG_OFFSET_SIZE)
    ),
    "string-len": lambda: _corrupt(lambda h, s, e: _set(s, 0, 1000)),
    "entries-beyond-row": lambda: _corrupt(lambda h, s, e: _set(h, 2, 2**40)),
    "magic": lambda: _corrupt(lambda h, s, e: _set(h, 0, 1)),
    "shorter-than-header": lambda: _corrupt(lambda h, s, e: None)[:, :20],
}


@pytest.mark.parametrize("dumps", DUMPS)
@pytest.mark.parametrize("corrupt", CORRUPT.values(), ids=CORRUPT.keys())
def test_corrupt_tables_are_rejected(dumps, corrupt):
    """Both writers reject a corrupt table with ValueError, before reading past its row or
    misplacing a value."""
    with pytest.raises(ValueError):
        dumps(Metadata(state_dict_metadata={}, storage_data=None), storage_rows=corrupt())


# Index tables: fqns named by entry in the coordinator's fqn list.


def _fqn_list(md: Metadata) -> table.FqnList:
    """Return the fqn list of md's state dict table."""
    return StateDictTable.from_metadata(dataclasses.replace(md, storage_data=None)).fqn_list


def test_fqn_list_of_the_metadata_matches_its_state_dict_table(tmp_path):
    md = dcp_saved_metadata(tmp_path)
    expected = _fqn_list(md)
    fqn_list = table.FqnList.of(md.state_dict_metadata)
    assert (fqn_list.fqns, dict(fqn_list.index), fqn_list.fingerprint) == (
        expected.fqns,
        dict(expected.index),
        expected.fingerprint,
    )


@given(st.lists(st.text(), unique=True))
def test_fqn_list_round_trips(fqns):
    fqn_list = table.FqnList.of(fqns)
    assert table.FqnList.decode(fqn_list.encode()) == fqn_list


def _mixed_rows(md: Metadata, ranks: int, indices):
    """Encode md.storage_data as `ranks` ranks' tables, rank r an index table if indices[r] and
    every fqn it writes is in md's fqn list, else a string table."""
    fqn_list = _fqn_list(md)
    encoded = []
    for r, results in enumerate(split_into_ranks(md, ranks)):
        try:
            if not indices[r]:
                raise table.Unencodable("a string table")
            encoded.append(table.encode(results, fqn_list))
        except table.Unencodable:
            encoded.append(table.encode(results))
    width = -(-max(len(e) for e in encoded) // 8) * 8
    return np.stack([np.frombuffer(e.ljust(width, b"\0"), dtype=np.uint8) for e in encoded])


@pytest.mark.parametrize("dumps", DUMPS)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(md=metadatas(), data=st.data())
def test_index_and_string_tables_write_like_the_dict(dumps, md, data):
    """Any mix of index and string tables writes the bytes of writing md itself."""
    assume(all(getattr(i, "transform_descriptors", None) is None for i in md.storage_data.values()))
    # fqns the storage writes that state_dict_metadata lacks make those ranks send strings
    ranks = data.draw(st.integers(1, 4))
    rows = _mixed_rows(
        md, ranks, data.draw(st.lists(st.booleans(), min_size=ranks, max_size=ranks))
    )
    from_tables = dumps(dataclasses.replace(md, storage_data=None), storage_rows=rows)
    assert from_tables == dumps(md)
    assert table.to_storage_data(table.decode_rows(rows), _fqn_list(md)) == md.storage_data


def _index_table_metadata() -> Metadata:
    """Return metadata whose storage writes fqns of its state dict: w and b."""
    return Metadata(
        state_dict_metadata={"w": BytesStorageMetadata(), "b": BytesStorageMetadata()},
        storage_data={
            MetadataIndex("w"): _StorageInfo("__0_0.distcp", 0, 1),
            MetadataIndex("b"): _StorageInfo("__0_0.distcp", 1, 1),
        },
    )


def _corrupt_index_table(mutate):
    """Return the row of an index table of _index_table_metadata, after mutate(header, entry)."""
    md = _index_table_metadata()
    rows = _mixed_rows(md, 1, [True])
    raw = rows[0].copy()
    header = raw[: 8 * table.HEADER].view("<i8")
    assert header[table.FQN_MODE] == table.FQN_INDICES
    pos = 8 * table.HEADER + 8 * int(header[3])
    entry = (
        raw[pos : pos + 8 * table.COLUMNS * int(header[2])].view("<i8").reshape(-1, table.COLUMNS)
    )
    mutate(header, entry)
    return raw[None, :]


CORRUPT_INDEX = {
    "unknown-fqn-mode": lambda: _corrupt_index_table(lambda h, e: _set(h, table.FQN_MODE, 7)),
    "other-fqn-list": lambda: _corrupt_index_table(
        lambda h, e: _set(h, table.FINGERPRINT, h[table.FINGERPRINT] ^ 1)
    ),
    "fqn-entry-past-the-list": lambda: _corrupt_index_table(
        lambda h, e: _set(e, (0, table.FQN), 2)
    ),
    "negative-fqn-entry": lambda: _corrupt_index_table(lambda h, e: _set(e, (0, table.FQN), -1)),
}


@pytest.mark.parametrize("dumps", DUMPS)
@pytest.mark.parametrize("corrupt", CORRUPT_INDEX.values(), ids=CORRUPT_INDEX.keys())
def test_corrupt_index_tables_are_rejected(dumps, corrupt):
    """Both writers reject an index table that doesn't fit the fqn list, with ValueError."""
    md = dataclasses.replace(_index_table_metadata(), storage_data=None)
    with pytest.raises(ValueError):
        dumps(md, storage_rows=corrupt())


@pytest.mark.parametrize("dumps", DUMPS)
def test_index_table_written_like_the_dict(dumps):
    md = _index_table_metadata()
    rows = _mixed_rows(md, 1, [True])
    assert dumps(dataclasses.replace(md, storage_data=None), storage_rows=rows) == dumps(md)


def test_decoding_an_index_table_needs_its_fqn_list():
    md = _index_table_metadata()
    (decoded,) = table.decode_rows(_mixed_rows(md, 1, [True]))
    assert table.to_write_results(decoded, _fqn_list(md))
    with pytest.raises(ValueError):
        table.to_write_results(decoded)
    with pytest.raises(ValueError):
        table.to_write_results(decoded, table.FqnList.of(("b", "w")))


def test_encoding_an_fqn_the_list_lacks_is_unencodable():
    """A rank writing an fqn the coordinator's list lacks sends its write results another way."""
    with pytest.raises(table.Unencodable):
        table.encode([_result()], table.FqnList.of(["other"]))
