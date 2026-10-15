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

import io

# Issue: [B403:blacklist] Consider possible security implications associated with pickle module.
# Severity: Low   Confidence: High
# CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
# More Info: https://bandit.readthedocs.io/en/1.8.3/blacklists/blacklist_imports.html#b403-import-pickle
import pickle  # nosec
from array import array
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    TensorStorageMetadata,
)

BYTES, TENSOR, OTHER = 0, 1, 2
# The fields the writers write themselves; fields holds None for them.
STATE_DICT_METADATA, STORAGE_DATA = "state_dict_metadata", "storage_data"


def small_pickle(obj) -> bytes:
    """Opcodes for a small object, from the stdlib pickler without memo, to splice into the stream.

    Protocol 3 emits no FRAME opcodes, and fast mode emits no memo PUT/GET, so the body can be
    embedded anywhere without disturbing the writer's memo numbering.
    """
    buf = io.BytesIO()
    p = pickle.Pickler(buf, protocol=3)
    p.fast = True
    p.dump(obj)
    data = buf.getvalue()
    assert data[:2] == b"\x80\x03" and data[-1:] == b"."
    return data[2:-1]


def small_unpickle(opcodes: bytes):
    """The object small_pickle wrote opcodes for."""
    return pickle.loads(b"\x80\x03" + opcodes + b".")  # nosec - our own bytes


@dataclass
class StateDictTable:
    """A Metadata without storage_data, as arrays; see the module docstring for the layout."""

    fields: List[Tuple[str, Optional[bytes]]]
    fqns: List[str]
    kind: np.ndarray
    ref: np.ndarray
    ndim: np.ndarray
    chunk_start: np.ndarray
    dims_start: np.ndarray
    dims: np.ndarray
    properties: List[bytes]
    others: List[bytes]
    index: Dict[str, int]  # fqn -> entry

    def __len__(self) -> int:
        return len(self.fqns)

    @property
    def nbytes(self) -> int:
        """Bytes held by the arrays."""
        arrays = (self.kind, self.ref, self.ndim, self.chunk_start, self.dims_start, self.dims)
        return sum(a.nbytes for a in arrays)

    def chunk_count(self, i: int) -> int:
        """The number of chunks of entry i (0 unless a tensor)."""
        return int(self.chunk_start[i + 1] - self.chunk_start[i])

    def size(self, i: int) -> np.ndarray:
        """Entry i's size (empty unless a tensor)."""
        start = self.dims_start[i]
        return self.dims[start : start + self.ndim[i]]

    def chunks_dims(self, i: int) -> np.ndarray:
        """Entry i's chunks as rows: a chunk's offsets, then its sizes."""
        ndim = int(self.ndim[i])
        start = self.dims_start[i] + ndim
        return self.dims[start : self.dims_start[i + 1]].reshape(self.chunk_count(i), 2 * ndim)

    def entry(self, i: int):
        """Entry i as an object, equal to the one it was converted from."""
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


def _tensor(v, dims: array) -> Optional[Tuple[int, int]]:
    """Append a TENSOR entry's values to dims and return its number of dims and of chunks; None,
    with nothing appended, if it isn't one."""
    state = vars(v)
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
    for c in chunks:
        if type(c) is not ChunkStorageMetadata:
            return None
        cstate = vars(c)
        offsets, sizes = cstate.get("offsets"), cstate.get("sizes")
        if (
            len(cstate) != 2
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


def from_metadata(md: Metadata) -> StateDictTable:
    """md without its storage_data, as a table.

    Raises TypeError for what the writers can't write from a table: md of a subclass of Metadata,
    a state_dict_metadata that isn't a dict, or an fqn that isn't a str.
    """
    if type(md) is not Metadata:
        raise TypeError(f"expected Metadata, got {type(md).__name__}")
    entries = md.state_dict_metadata
    if type(entries) is not dict:
        raise TypeError(f"expected dict for state_dict_metadata, got {type(entries).__name__}")
    fields = [
        (name, None if name in (STATE_DICT_METADATA, STORAGE_DATA) else small_pickle(value))
        for name, value in vars(md).items()
    ]
    fqns, kind, ref, ndims = [], array("B"), array("q"), array("q")
    chunk_start, dims_start, dims = array("q", [0]), array("q", [0]), array("q")
    properties, others = [], []
    property_ids: Dict[bytes, int] = {}  # by their opcodes
    for fqn, v in entries.items():
        if type(fqn) is not str:
            raise TypeError(f"expected str fqn, got {type(fqn).__name__}")
        fqns.append(fqn)
        # Counted before small_pickle runs Python code, which could change the chunks.
        counts = _tensor(v, dims) if type(v) is TensorStorageMetadata else None
        ndim, chunks = counts if counts is not None else (0, 0)
        if counts is not None:
            opcodes = small_pickle(v.properties)
            pid = property_ids.setdefault(opcodes, len(properties))
            if pid == len(properties):
                properties.append(opcodes)
            kind.append(TENSOR)
            ref.append(pid)
        elif type(v) is BytesStorageMetadata and not vars(v):
            kind.append(BYTES)
            ref.append(0)
        else:
            kind.append(OTHER)
            ref.append(len(others))
            others.append(small_pickle(v))
        ndims.append(ndim)
        chunk_start.append(chunk_start[-1] + chunks)
        dims_start.append(len(dims))
    if len(fqns) != len(entries):
        raise RuntimeError("state_dict_metadata changed size while being converted")
    return StateDictTable(
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
