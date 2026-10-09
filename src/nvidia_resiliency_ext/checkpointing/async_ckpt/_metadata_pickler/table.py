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

"""A rank's write results as a compact table of columns, for the coordinator's .metadata writer.

The coordinator only needs each write result's storage index and storage info to write
storage_data. Sent as a table, it receives them without unpickling a WriteResult per chunk, and
the writer encodes storage_data from the columns without creating its objects.

Layout of an encoded table (little-endian):
    header      int64[8]: magic, version, entries, strings, string bytes, offset values, 0, 0
    string_len  int64[strings]       byte length of each string (fqns and relative paths)
    entry       int64[entries, 7]    fqn, index, offset_ndim, path, offset, length, flags
    offsets     int64[offset values] all entries' MetadataIndex offsets, concatenated
    strings     uint8[string bytes]  the strings, UTF-8 (surrogatepass), concatenated

Entry columns: fqn and path are string ids. index is MetadataIndex.index, valid if FLAG_INDEX.
offset_ndim is the number of MetadataIndex.offset dims, valid if FLAG_OFFSET (the index has an
offset attribute) and FLAG_OFFSET_SIZE (it is a torch.Size, not None). offset and length are the
_StorageInfo's.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.storage import WriteResult

MAGIC = 0x4E56525854424C31  # "NVRXTBL1"
VERSION = 1
HEADER = 8
COLUMNS = 7
FQN, INDEX, OFFSET_NDIM, PATH, OFFSET, LENGTH, FLAGS = range(COLUMNS)
FLAG_INDEX, FLAG_OFFSET, FLAG_OFFSET_SIZE = 0x1, 0x2, 0x4
_INT64 = (-(2**63), 2**63 - 1)


class Unencodable(TypeError):
    """A write result the table can't represent; the save falls back to pickled write results."""


@dataclass
class Table:
    """A decoded table: numpy views into the received buffer, and its strings."""

    entry: np.ndarray  # int64[entries, COLUMNS]
    offsets: np.ndarray  # int64[offset values]
    strings: List[str]

    def __len__(self) -> int:
        return len(self.entry)


def _check_int(v) -> int:
    """v, if it is an int that fits int64."""
    if type(v) is not int or not _INT64[0] <= v <= _INT64[1]:
        raise Unencodable(f"expected an int64, got {v!r}")
    return v


def encode(results: List[WriteResult]) -> bytes:
    """One rank's write results as a table. Raises Unencodable for anything it can't represent."""
    string_ids: dict = {}
    strings: List[bytes] = []

    def string_id(s) -> int:
        if type(s) is not str:
            raise Unencodable(f"expected str, got {type(s).__name__}")
        i = string_ids.get(s)
        if i is None:
            i = string_ids[s] = len(strings)
            strings.append(s.encode("utf-8", "surrogatepass"))
        return i

    entry = np.zeros((len(results), COLUMNS), dtype="<i8")
    offsets: List[int] = []
    for row, wr in enumerate(results):
        idx, info = wr.index, wr.storage_data
        if type(idx) is not MetadataIndex or type(info) is not _StorageInfo:
            raise Unencodable(f"unexpected types {type(idx).__name__}, {type(info).__name__}")
        state = vars(idx)
        offset = state.get("offset", None)
        has_offset = "offset" in state
        if set(state) != ({"fqn", "index", "offset"} if has_offset else {"fqn", "index"}):
            raise Unencodable(f"unexpected MetadataIndex attributes: {list(state)}")
        info_state = vars(info)
        if info_state.get("transform_descriptors") is not None or not {
            "relative_path",
            "offset",
            "length",
        } <= set(info_state) <= {"relative_path", "offset", "length", "transform_descriptors"}:
            raise Unencodable(f"unexpected _StorageInfo state: {info_state}")
        flags = 0
        e = entry[row]
        e[FQN] = string_id(state["fqn"])
        if state["index"] is not None:
            e[INDEX] = _check_int(state["index"])
            flags |= FLAG_INDEX
        if has_offset:
            flags |= FLAG_OFFSET
            if offset is not None:
                if type(offset) is not torch.Size:
                    raise Unencodable(f"expected torch.Size offset, got {type(offset).__name__}")
                flags |= FLAG_OFFSET_SIZE
                e[OFFSET_NDIM] = len(offset)
                offsets.extend(_check_int(d) for d in offset)
        e[PATH] = string_id(info_state["relative_path"])
        e[OFFSET] = _check_int(info_state["offset"])
        e[LENGTH] = _check_int(info_state["length"])
        e[FLAGS] = flags
    string_len = np.fromiter((len(s) for s in strings), dtype="<i8", count=len(strings))
    blob = b"".join(strings)
    header = np.array(
        [MAGIC, VERSION, len(results), len(strings), len(blob), len(offsets), 0, 0], dtype="<i8"
    )
    return b"".join(
        (
            header.tobytes(),
            string_len.tobytes(),
            entry.tobytes(),
            np.asarray(offsets, dtype="<i8").tobytes(),
            blob,
        )
    )


def is_table(buf) -> bool:
    """Whether buf (bytes or a uint8 array) starts like a table; a pickle never does."""
    return len(buf) >= 8 and int(np.frombuffer(buf, dtype="<i8", count=1)[0]) == MAGIC


def decode(buf) -> Table:
    """The table in buf (bytes, or a uint8 array possibly padded at the end), without copying."""
    raw = np.frombuffer(buf, dtype=np.uint8)
    header = raw[: HEADER * 8].view("<i8")
    if header[0] != MAGIC or header[1] != VERSION:
        raise ValueError("not a write-result table of this version")
    n_entries, n_strings, n_bytes, n_offsets = (int(x) for x in header[2:6])
    pos = HEADER * 8
    string_len = raw[pos : pos + 8 * n_strings].view("<i8")
    pos += 8 * n_strings
    entry = raw[pos : pos + 8 * COLUMNS * n_entries].view("<i8").reshape(n_entries, COLUMNS)
    pos += 8 * COLUMNS * n_entries
    offsets = raw[pos : pos + 8 * n_offsets].view("<i8")
    pos += 8 * n_offsets
    blob = raw[pos : pos + n_bytes].tobytes()
    if len(blob) != n_bytes or len(entry) != n_entries:
        raise ValueError("write-result table is truncated")
    ids = entry[:, [FQN, PATH]]
    ndim = entry[:, OFFSET_NDIM]
    has_size = (entry[:, FLAGS] & FLAG_OFFSET_SIZE) != 0
    if (ids < 0).any() or (ids >= n_strings).any() or (ndim < 0).any():
        raise ValueError("write-result table: string id or offset dims out of range")
    if int(ndim[has_size].sum()) != n_offsets:
        raise ValueError("write-result table: offset dims don't match the offsets")
    strings, start = [], 0
    for n in string_len.tolist():
        strings.append(blob[start : start + n].decode("utf-8", "surrogatepass"))
        start += n
    return Table(entry=entry, offsets=offsets, strings=strings)


def to_storage_data(tables: List[Table]) -> dict:
    """The storage_data dict the tables describe, as finish builds it from write results."""
    storage_data = {}
    for table in tables:
        offsets = table.offsets.tolist()
        pos = 0
        for fqn, index, ndim, path, offset, length, flags in table.entry.tolist():
            if flags & FLAG_OFFSET_SIZE:
                idx = MetadataIndex(
                    table.strings[fqn],
                    torch.Size(offsets[pos : pos + ndim]),
                    index if flags & FLAG_INDEX else None,
                )
                pos += ndim
            elif flags & FLAG_OFFSET:
                idx = MetadataIndex(table.strings[fqn], None, index if flags & FLAG_INDEX else None)
                idx.__dict__["offset"] = None
            else:
                idx = MetadataIndex(table.strings[fqn], index=index if flags & FLAG_INDEX else None)
            storage_data[idx] = _StorageInfo(table.strings[path], offset, length)
    return storage_data
