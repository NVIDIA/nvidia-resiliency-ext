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

The coordinator receives each rank's table zero-padded to a common width; the padding is ignored.
See WriteResultTable for what the entries hold.
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
class WriteResultTable:
    """A decoded table: numpy views into the received buffer, and its strings.

    Each entry (a row of ``entry``) is one write result, ``WriteResult(index, size_in_bytes,
    storage_data)``, which the coordinator writes as the ``storage_data`` item ``index ->
    storage_data``. Only those two are kept: ``size_in_bytes`` is the storage length.

    The table has three parts:

    - ``entry``: ``int64[entries, COLUMNS]``, one row per write result, columns below.
    - ``offsets``: ``int64[offset values]``, the dims of every entry's ``index.offset``,
      concatenated in entry order. An offset has one value per dim of its tensor, so it doesn't fit
      a fixed number of columns; each entry's OFFSET_NDIM says how many of these values are its. A
      reader walks the entries in order with a running position into ``offsets``, starting at 0;
      at the end, the position must equal ``len(offsets)``. For example, entries with offsets
      ``(0, 64)``, None and ``(128,)`` have OFFSET_NDIM 2, 0 and 1, and ``offsets`` is
      ``[0, 64, 128]``.
    - ``strings``: the distinct fqns and relative paths, which FQN and PATH index.

    ===========  ===================================================================================
    Column       Value
    ===========  ===================================================================================
    FQN          ``index.fqn``: an id into ``strings``.
    INDEX        ``index.index`` if FLAG_INDEX is set, else 0 (the index is None).
    OFFSET_NDIM  The number of dims of ``index.offset`` if FLAG_OFFSET_SIZE is set, else 0: how many
                 values of ``offsets``, from the running position, are its dims.
    PATH         ``storage_data.relative_path``: an id into ``strings``.
    OFFSET       ``storage_data.offset``.
    LENGTH       ``storage_data.length``.
    FLAGS        A bitmask of the FLAG_* values below.
    ===========  ===================================================================================

    Flags, for the attributes a value can't express:

    - FLAG_INDEX (0x1): ``index.index`` is an int, not None.
    - FLAG_OFFSET (0x2): ``index`` has an ``offset`` attribute. A MetadataIndex created without
      an offset has none, and pickles without it.
    - FLAG_OFFSET_SIZE (0x4): ``index.offset`` is a ``torch.Size``, not None; set only with
      FLAG_OFFSET.

    Strings are stored once per table, so an fqn or path shared by several entries has one id.
    ``storage_data`` is a ``_StorageInfo`` without ``transform_descriptors``; ``encode`` raises
    Unencodable for anything else, and that rank's write results are sent pickled.
    """

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


def decode(buf) -> WriteResultTable:
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
    if (string_len < 0).any() or int(string_len.sum()) != n_bytes:
        raise ValueError("write-result table: string lengths don't match the string bytes")
    strings, start = [], 0
    for n in string_len.tolist():
        strings.append(blob[start : start + n].decode("utf-8", "surrogatepass"))
        start += n
    return WriteResultTable(entry=entry, offsets=offsets, strings=strings)


def decode_rows(rows: np.ndarray) -> List[WriteResultTable]:
    """The tables in the rows of a 2-D uint8 array, one zero-padded table per row."""
    return [decode(row) for row in rows]


def to_storage_data(tables: List[WriteResultTable]) -> dict:
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


def to_write_results(t: WriteResultTable) -> List[WriteResult]:
    """The write results a table holds, as torch's finish takes them. size_in_bytes is the storage
    length, as torch's writer reports it."""
    return [
        WriteResult(index=index, size_in_bytes=info.length, storage_data=info)
        for index, info in to_storage_data([t]).items()
    ]
