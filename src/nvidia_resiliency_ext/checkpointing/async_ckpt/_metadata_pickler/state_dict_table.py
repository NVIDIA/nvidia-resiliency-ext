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

"""A Metadata without its storage_data, as a few flat arrays.

A DCP ``Metadata`` holds a Python object per chunk, each with a ``torch.Size`` of offsets and one
of sizes: millions of small objects for a large checkpoint, and every full garbage collection visits
all of them. A ``StateDictTable`` holds the same ``state_dict_metadata`` in numpy arrays, which the
garbage collector does not track, and the other fields as pickle opcodes. The ``.metadata`` writers
write from it, with storage_data from the write results.

Entries are kept in dict order, as one of three kinds:

- BYTES: a ``BytesStorageMetadata`` without attributes.
- TENSOR: a ``TensorStorageMetadata`` with exactly the attributes properties, size and chunks,
  whose size is a ``torch.Size``, chunks a list of ``ChunkStorageMetadata`` with exactly the
  attributes offsets and sizes, each a ``torch.Size`` with as many dims as size. A ``torch.Size``
  holds only ints within int64.
- OTHER: anything else, for example an object of a subclass, kept as the stdlib pickler's
  opcodes for it (``small_pickle``) and written as they are.

Arrays, for entries ``i`` (``n`` of them):

==================  ==============  ===============================================================
Array               Type            Value
==================  ==============  ===============================================================
kind                uint8[n]        BYTES, TENSOR or OTHER.
ref                 int64[n]        TENSOR: an index into ``properties``; OTHER: into ``others``.
ndim                int64[n]        A tensor's number of dims; 0 for BYTES and OTHER.
chunk_start         int64[n + 1]    Entry i's chunks are numbers ``chunk_start[i]`` up to
                                    ``chunk_start[i+1]``, counting all entries' chunks.
dims_start          int64[n + 1]    Where entry i's values start in ``dims``.
dims                int64[...]      Each tensor's values in one range: its size, then each chunk's
                                    offsets and sizes. Chunk k of entry i starts at
                                    ``dims_start[i] + ndim[i] + 2 * ndim[i] * k``.
==================  ==============  ===============================================================

``chunk_start`` is kept rather than derived from ``dims``, because a 0-dim tensor has no values
whatever its number of chunks.

``properties`` holds the opcodes of the distinct ``TensorProperties``, compared by those
opcodes: equal values of different types, such as ``True`` and ``1``, pickle differently and are
kept apart. ``fields`` holds Metadata's fields in order, with their opcodes, except
state_dict_metadata and storage_data, which the writers write themselves.
"""

import dataclasses
import functools
from array import array
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    TensorStorageMetadata,
)

from . import table
from .opcodes import small_pickle, small_unpickle

BYTES, TENSOR, OTHER = 0, 1, 2
# The fields the writers write themselves; fields holds None for them.
STATE_DICT_METADATA, STORAGE_DATA = "state_dict_metadata", "storage_data"


@dataclass(frozen=True, eq=False)
class StateDictTable:
    """A Metadata without storage_data, as arrays; see the module docstring for the layout.

    Create it with from_metadata. It is immutable and checked when created: its sequences are
    tuples, its index a read-only mapping and its arrays read-only, so a writer can read it
    without further checks.
    """

    # The names of the array fields.
    ARRAYS: ClassVar[Tuple[str, ...]] = ("kind", "ref", "ndim", "chunk_start", "dims_start", "dims")

    fields: Tuple[Tuple[str, Optional[bytes]], ...]
    fqns: Tuple[str, ...]
    kind: np.ndarray
    ref: np.ndarray
    ndim: np.ndarray
    chunk_start: np.ndarray
    dims_start: np.ndarray
    dims: np.ndarray
    properties: Tuple[bytes, ...]
    others: Tuple[bytes, ...]
    index: Mapping[str, int]  # fqn -> entry

    def __post_init__(self):
        """Freeze what the fields hold, then check them."""
        for name in ("fields", "fqns", "properties", "others"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "index", MappingProxyType(dict(self.index)))
        for name in self.ARRAYS:
            value = getattr(self, name)
            if isinstance(value, np.ndarray):
                value = value.view()
                value.flags.writeable = False
                object.__setattr__(self, name, value)
        self._check()

    @classmethod
    def from_metadata(cls, metadata: Metadata) -> "StateDictTable":
        """Convert metadata, without its storage_data, to a table.

        Raise TypeError for what the writers can't write from a table: a subclass of Metadata, a
        state_dict_metadata that isn't a dict, or an fqn that isn't a str.
        """
        if type(metadata) is not Metadata:
            raise TypeError(f"expected Metadata, got {type(metadata).__name__}")
        entries = metadata.state_dict_metadata
        if type(entries) is not dict:
            raise TypeError(f"expected dict for state_dict_metadata, got {type(entries).__name__}")
        fields = [
            (name, None if name in (STATE_DICT_METADATA, STORAGE_DATA) else small_pickle(value))
            for name, value in vars(metadata).items()
        ]
        fqns, kind, ref, ndims = [], array("B"), array("q"), array("q")
        chunk_start, dims_start, dims = array("q", [0]), array("q", [0]), array("q")
        properties, others = [], []
        property_ids: Dict[bytes, int] = {}  # by their opcodes
        for fqn, entry in entries.items():
            if type(fqn) is not str:
                raise TypeError(f"expected str fqn, got {type(fqn).__name__}")
            fqns.append(fqn)
            # Counted before small_pickle runs Python code, which could change the chunks.
            counts = _tensor_entry(entry, dims) if type(entry) is TensorStorageMetadata else None
            ndim, chunks = counts if counts is not None else (0, 0)
            if counts is not None:
                opcodes = small_pickle(entry.properties)
                property_id = property_ids.setdefault(opcodes, len(properties))
                if property_id == len(properties):
                    properties.append(opcodes)
                kind.append(TENSOR)
                ref.append(property_id)
            elif type(entry) is BytesStorageMetadata and not vars(entry):
                kind.append(BYTES)
                ref.append(0)
            else:
                kind.append(OTHER)
                ref.append(len(others))
                others.append(small_pickle(entry))
            ndims.append(ndim)
            chunk_start.append(chunk_start[-1] + chunks)
            dims_start.append(len(dims))
        if len(fqns) != len(entries):
            raise RuntimeError("state_dict_metadata changed size while being converted")
        return cls(
            fields=fields,
            fqns=fqns,
            kind=np.frombuffer(kind, dtype=np.uint8),
            ref=np.frombuffer(ref, dtype=np.int64),
            ndim=np.frombuffer(ndims, dtype=np.int64),
            chunk_start=np.frombuffer(chunk_start, dtype=np.int64),
            dims_start=np.frombuffer(dims_start, dtype=np.int64),
            dims=np.frombuffer(dims, dtype=np.int64),
            properties=properties,
            others=others,
            index={fqn: i for i, fqn in enumerate(fqns)},
        )

    def to_metadata(self, storage_data: Optional[dict] = None) -> Metadata:
        """Build a new Metadata equal to the one this table was converted from.

        Its storage_data is the one given, and its fields are set in the order the original had
        them, so that it pickles alike.
        """
        metadata = Metadata.__new__(Metadata)
        for name, opcodes in self.fields:
            if name == STATE_DICT_METADATA:
                value = {fqn: self.entry(i) for i, fqn in enumerate(self.fqns)}
            elif name == STORAGE_DATA:
                value = storage_data
            else:
                value = small_unpickle(opcodes)
            setattr(metadata, name, value)
        if STORAGE_DATA not in dict(self.fields):
            metadata.storage_data = storage_data
        return metadata

    def with_fields(self, **values) -> "StateDictTable":
        """Return a copy of this table with these Metadata fields (e.g. version) set.

        The copy shares the table's arrays. A field the table doesn't have yet is added last, as
        setting it on the Metadata would. Raise TypeError for a name that isn't a field of Metadata
        a save sets.
        """
        known = {f.name for f in dataclasses.fields(Metadata)} - {STATE_DICT_METADATA, STORAGE_DATA}
        unknown = set(values) - known
        if unknown:
            raise TypeError(f"not fields of Metadata that a save sets: {sorted(unknown)}")
        fields = [
            (name, small_pickle(values.pop(name)) if name in values else opcodes)
            for name, opcodes in self.fields
        ]
        fields.extend((name, small_pickle(value)) for name, value in values.items())
        return dataclasses.replace(self, fields=fields)

    def __len__(self) -> int:
        return len(self.fqns)

    @functools.cached_property
    def fqns_fingerprint(self) -> int:
        """The fingerprint of the fqn list (see table.fingerprint), computed once."""
        return table.fingerprint(self.fqns)

    @property
    def nbytes(self) -> int:
        """Bytes held by the arrays."""
        return sum(getattr(self, name).nbytes for name in self.ARRAYS)

    def chunk_count(self, i: int) -> int:
        """Return the number of chunks of entry i (0 unless a tensor)."""
        return int(self.chunk_start[i + 1] - self.chunk_start[i])

    def size(self, i: int) -> np.ndarray:
        """Return entry i's size (empty unless a tensor)."""
        start = self.dims_start[i]
        return self.dims[start : start + self.ndim[i]]

    def chunks_dims(self, i: int) -> np.ndarray:
        """Return entry i's chunks as rows: a chunk's offsets, then its sizes."""
        ndim = int(self.ndim[i])
        start = self.dims_start[i] + ndim
        return self.dims[start : self.dims_start[i + 1]].reshape(self.chunk_count(i), 2 * ndim)

    def entry(self, i: int):
        """Build entry i as an object, equal to the one it was converted from."""
        kind = self.kind[i]
        if kind == BYTES:
            return BytesStorageMetadata()
        if kind == OTHER:
            return small_unpickle(self.others[self.ref[i]])
        ndim = int(self.ndim[i])
        chunks = [
            ChunkStorageMetadata(torch.Size(row[:ndim]), torch.Size(row[ndim:]))
            for row in self.chunks_dims(i).tolist()
        ]
        size = torch.Size(self.size(i).tolist())
        return TensorStorageMetadata(small_unpickle(self.properties[self.ref[i]]), size, chunks)

    def _check(self) -> None:
        """Raise ValueError unless the arrays describe the entries consistently. The C++ writer
        checks the same before reading a table (StateDictTableView)."""

        def bad(what: str):
            return ValueError(f"state dict table: {what}")

        arrays = {"kind": np.uint8, **{name: np.int64 for name in self.ARRAYS if name != "kind"}}
        for name, dtype in arrays.items():
            array = getattr(self, name)
            if (
                type(array) is not np.ndarray
                or array.dtype != dtype
                or array.ndim != 1
                or not array.flags.c_contiguous
            ):
                raise bad(f"{name} is not a 1-D array of the expected type")
        for field in self.fields:
            if type(field) is not tuple or len(field) != 2:
                raise bad("a field is not a (name, opcodes) tuple")
            name, opcodes = field
            if type(name) is not str:
                raise TypeError(f"expected str field name, got {type(name).__name__}")
            written_here = name in (STATE_DICT_METADATA, STORAGE_DATA)
            if opcodes is not None if written_here else type(opcodes) is not bytes:
                raise bad(f"unexpected opcodes of field {name}")
        n = len(self.kind)
        if len(self.fqns) != n or len(self.ref) != n or len(self.ndim) != n:
            raise bad("arrays of different lengths")
        for starts, pool in ((self.chunk_start, None), (self.dims_start, len(self.dims))):
            if (
                len(starts) != n + 1
                or starts[0] != 0
                or (starts[1:] < starts[:-1]).any()
                or (pool is not None and starts[-1] != pool)
            ):
                raise bad("start offsets out of order or range")
        ndim, chunks, values = self.ndim, np.diff(self.chunk_start), np.diff(self.dims_start)
        kind, ref = self.kind, self.ref

        def refers(pickles, mask):
            return all(
                0 <= r < len(pickles) and type(pickles[r]) is bytes for r in ref[mask].tolist()
            )

        tensor, other = kind == TENSOR, kind == OTHER
        if not np.isin(kind, (BYTES, TENSOR, OTHER)).all():
            raise bad("unknown entry kind")
        if (ndim < 0).any():
            raise bad("a negative number of dims")
        # Python ints: no overflow.
        if any(
            count != dims * (1 + 2 * chunk_count)
            for count, dims, chunk_count in zip(
                values[tensor].tolist(), ndim[tensor].tolist(), chunks[tensor].tolist()
            )
        ) or not refers(self.properties, tensor):
            raise bad("a tensor entry's dims don't match its chunks")
        if not refers(self.others, other):
            raise bad("an entry refers to no pickle")
        non_tensor = ~tensor
        if (ndim[non_tensor] | chunks[non_tensor] | values[non_tensor]).any():
            raise bad("a non-tensor entry has dims")


def _tensor_entry(entry, dims: array) -> Optional[Tuple[int, int]]:
    """Append a TENSOR entry's values to dims and return its number of dims and of chunks; None,
    with nothing appended, if it isn't one."""
    state = vars(entry)
    size, chunks = state.get("size"), state.get("chunks")
    if (
        len(state) != 3
        or "properties" not in state
        or type(size) is not torch.Size
        or type(chunks) is not list
    ):
        return None
    ndim = len(size)
    values = array("q", size)
    for chunk in chunks:
        if type(chunk) is not ChunkStorageMetadata:
            return None
        chunk_state = vars(chunk)
        offsets, sizes = chunk_state.get("offsets"), chunk_state.get("sizes")
        if (
            len(chunk_state) != 2
            or type(offsets) is not torch.Size
            or type(sizes) is not torch.Size
            or len(offsets) != ndim
            or len(sizes) != ndim
        ):
            return None
        values.extend(offsets)
        values.extend(sizes)
    dims.extend(values)
    return ndim, len(chunks)
