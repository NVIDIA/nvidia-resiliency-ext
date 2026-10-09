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
    strings     uint8[string bytes]  the strings, UTF-8, concatenated

The coordinator receives each rank's table zero-padded to a common width; the padding is ignored.
See WriteResultTable for what the entries hold.

Tables come from this job's ranks, and both writers check everything that could make them read
outside a table or misplace a value: counts, string ids, offset dims, string lengths and flags,
rejecting the same tables with ValueError. They trust one thing: that the string bytes are UTF-8,
as encode wrote them. The native writer copies them into the pickle unchecked, since checking would
cost about as much as writing them; corrupt string bytes would make a .metadata that doesn't load.
decode, which builds Python strings, raises UnicodeDecodeError for them instead.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.storage import WriteResult

MAGIC = int.from_bytes(b"NVRXTBL1", "little")  # reads "NVRXTBL1" on the wire
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

    Entries with equal indexes (MetadataIndex compares fqn and offset, not index) are all kept: the
    same chunk written by two ranks, when the save planner doesn't deduplicate it. It is legal, and
    torch's finish keeps one item, with the first key and the last value. The writers write every
    entry as an item of storage_data, which unpickles to the same dict, but the .metadata bytes
    differ from those written from torch's dict.
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
            try:
                strings.append(s.encode("utf-8"))
            except UnicodeEncodeError as e:  # a lone surrogate: left to the stdlib pickler
                raise Unencodable(f"string not encodable as UTF-8: {s!r}") from e
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


def _bad(what: str) -> ValueError:
    """The error for a malformed table."""
    return ValueError(f"write-result table: {what}")


def decode(buf) -> WriteResultTable:
    """The table in buf (bytes, or a uint8 array possibly padded at the end), without copying.

    Rejects with ValueError the same malformed tables as the native writer (native_src/native.cpp)
    does; see the module docstring for the one exception, invalid UTF-8.
    """
    raw = np.frombuffer(buf, dtype=np.uint8)
    if len(raw) < 8 * HEADER:
        raise _bad("truncated")
    header = raw[: 8 * HEADER].view("<i8").tolist()
    if header[0] != MAGIC or header[1] != VERSION:
        raise _bad("not a table of this version")
    pos = 8 * HEADER

    def take(count: int, item_size: int) -> np.ndarray:
        """The next section, count items of item_size bytes, checked to fit in the row."""
        nonlocal pos
        if count < 0 or count > (len(raw) - pos) // item_size:
            raise _bad("truncated")
        section = raw[pos : pos + count * item_size]
        pos += count * item_size
        return section

    n_entries, n_strings, n_bytes, n_offsets = header[2:6]
    string_len = take(n_strings, 8).view("<i8")
    entry = take(n_entries, 8 * COLUMNS).view("<i8").reshape(n_entries, COLUMNS)
    offsets = take(n_offsets, 8).view("<i8")
    blob = take(n_bytes, 1).tobytes()

    flags = entry[:, FLAGS]
    if (flags & ~(FLAG_INDEX | FLAG_OFFSET | FLAG_OFFSET_SIZE)).any():
        raise _bad("unknown flags")
    has_size = (flags & FLAG_OFFSET_SIZE) != 0
    if (has_size & ((flags & FLAG_OFFSET) == 0)).any():
        raise _bad("an offset size without an offset")
    ids = entry[:, [FQN, PATH]]
    if (ids < 0).any() or (ids >= n_strings).any():
        raise _bad("string id out of range")
    # Summed as Python ints, which can't wrap around.
    ndim = entry[has_size, OFFSET_NDIM]
    if (ndim < 0).any() or sum(ndim.tolist()) != n_offsets:
        raise _bad("offset dims don't match the offsets")
    lengths = string_len.tolist()
    if (string_len < 0).any() or sum(lengths) != n_bytes:
        raise _bad("string lengths don't match the string bytes")
    strings, start = [], 0
    for n in lengths:
        strings.append(blob[start : start + n].decode("utf-8"))
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
