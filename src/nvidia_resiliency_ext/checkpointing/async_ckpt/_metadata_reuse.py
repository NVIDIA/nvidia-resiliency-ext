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

"""Internal: decide whether a loaded checkpoint's metadata can be reused for a new save.

The global metadata is built from the write items of all ranks' local plans, so it can be reused
if the new plans write exactly the chunks it lists. Every rank holds the loaded metadata and checks
its share; the shares are summed across ranks with one all_reduce:

* each tensor item's state_dict_metadata entry is a tensor with the item's global size and
  properties (compared with ==);
* the chunks written now and the chunks listed in the metadata are the same multiset: each rank
  adds a 128-bit hash of every chunk it writes (fqn, bytes or tensor, offsets, sizes) and
  subtracts the hashes of its 1/N slice of the metadata's chunks. The sum over all ranks is zero
  if, and (up to 2^-127) only if, they match.

Which rank writes a chunk does not matter: the metadata does not record it, and the storage data
is rebuilt from this save's write results.

The check reads the metadata's entries through a view: ObjectEntries for a Metadata, TableEntries
for a state table (see _metadata_pickler.state_table).
"""

import struct
from dataclasses import fields
from hashlib import blake2b
from logging import getLogger
from typing import Any, Iterable, List, Sequence, Tuple, Union

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import SavePlan, WriteItemType

from ._metadata_pickler.opcodes import small_unpickle
from ._metadata_pickler.state_table import OTHER, TENSOR, StateDictTable

logger = getLogger(__name__)

HASH_BITS = 128
LANE_BITS = 32
LANES = HASH_BITS // LANE_BITS
HASH_MASK = (1 << HASH_BITS) - 1
LANE_MASK = (1 << LANE_BITS) - 1


def _fqn_hash(fqn: str, kind: bytes) -> "blake2b":
    """Hash state of an fqn and its kind (b"B" bytes, b"T" tensor), length-prefixed."""
    encoded = fqn.encode("utf-8", "surrogatepass")
    return blake2b(struct.pack("<Q", len(encoded)) + encoded + kind, digest_size=HASH_BITS // 8)


def _tensor_chunk_hash(fqn_hash: "blake2b", offsets: Sequence[int], sizes: Sequence[int]) -> int:
    """Hash of one tensor chunk: its fqn, offsets and sizes."""
    h = fqn_hash.copy()
    h.update(struct.pack(f"<Q{len(offsets)}q{len(sizes)}q", len(offsets), *offsets, *sizes))
    return int.from_bytes(h.digest(), "little")


def _bytes_hash(fqn: str) -> int:
    """Hash of one bytes entry: its fqn."""
    return int.from_bytes(_fqn_hash(fqn, b"B").digest(), "little")


def plan_hash(plan: SavePlan) -> int:
    """Sum of the hashes of the chunks a local plan writes, mod 2^128."""
    total = 0
    for item in plan.items:
        if item.type == WriteItemType.BYTE_IO:
            total += _bytes_hash(item.index.fqn)
        else:
            chunk = item.tensor_data.chunk
            total += _tensor_chunk_hash(_fqn_hash(item.index.fqn, b"T"), chunk.offsets, chunk.sizes)
    return total & HASH_MASK


Chunk = Tuple[Sequence[int], Sequence[int]]  # offsets, sizes


class ObjectEntries:
    """The entries of a Metadata, as the check reads them."""

    def __init__(self, metadata: Metadata):
        self._entries = metadata.state_dict_metadata
        self._fqns = list(self._entries)
        self.planner_data = metadata.planner_data

    def __len__(self) -> int:
        return len(self._fqns)

    def fqn(self, i: int) -> str:
        return self._fqns[i]

    def counts(self) -> List[int]:
        """Each entry's number of chunks, 1 for an entry that isn't a tensor."""
        return [
            len(e.chunks) if isinstance(e, TensorStorageMetadata) else 1
            for e in self._entries.values()
        ]

    def is_tensor(self, i: int) -> bool:
        return isinstance(self._entries[self._fqns[i]], TensorStorageMetadata)

    def chunks(self, i: int, lo: int, hi: int) -> Iterable[Chunk]:
        """Chunks lo up to hi of tensor entry i."""
        return [(c.offsets, c.sizes) for c in self._entries[self._fqns[i]].chunks[lo:hi]]

    def tensor(self, fqn: str) -> Union[Tuple[torch.Size, Any], str]:
        """The global size and properties of fqn's entry if it is a tensor, else what it is."""
        entry = self._entries.get(fqn)
        if not isinstance(entry, TensorStorageMetadata):
            return type(entry).__name__
        return entry.size, entry.properties


class TableEntries:
    """The entries of a state table, as the check reads them. An entry the table keeps pickled is
    unpickled when the check reads it."""

    def __init__(self, t: StateDictTable):
        self._t = t
        self._objects = {}  # entry -> object, for entries kept pickled
        self._properties = {}  # properties id -> TensorProperties
        planner_data = dict(t.fields).get("planner_data")
        self.planner_data = None if planner_data is None else small_unpickle(planner_data)

    def __len__(self) -> int:
        return len(self._t)

    def fqn(self, i: int) -> str:
        return self._t.fqns[i]

    def _object(self, i: int):
        if i not in self._objects:
            self._objects[i] = self._t.entry(i)
        return self._objects[i]

    def counts(self) -> List[int]:
        """Each entry's number of chunks, 1 for an entry that isn't a tensor."""
        t = self._t
        counts = (t.chunk_start[1:] - t.chunk_start[:-1]).tolist()
        for i, kind in enumerate(t.kind.tolist()):
            if kind == OTHER:
                e = self._object(i)
                counts[i] = len(e.chunks) if isinstance(e, TensorStorageMetadata) else 1
            elif kind != TENSOR:
                counts[i] = 1
        return counts

    def is_tensor(self, i: int) -> bool:
        kind = self._t.kind[i]
        return kind == TENSOR or (
            kind == OTHER and isinstance(self._object(i), TensorStorageMetadata)
        )

    def chunks(self, i: int, lo: int, hi: int) -> Iterable[Chunk]:
        """Chunks lo up to hi of tensor entry i."""
        if self._t.kind[i] == OTHER:
            return [(c.offsets, c.sizes) for c in self._object(i).chunks[lo:hi]]
        ndim = int(self._t.ndim[i])
        return [(row[:ndim], row[ndim:]) for row in self._t.chunks_dims(i)[lo:hi].tolist()]

    def tensor(self, fqn: str) -> Union[Tuple[torch.Size, Any], str]:
        """The global size and properties of fqn's entry if it is a tensor, else what it is."""
        t = self._t
        i = t.index.get(fqn)
        if i is None:
            return "NoneType"
        if t.kind[i] == OTHER:
            e = self._object(i)
            if not isinstance(e, TensorStorageMetadata):
                return type(e).__name__
            return e.size, e.properties
        if t.kind[i] != TENSOR:
            return "BytesStorageMetadata"
        ref = int(t.ref[i])
        if ref not in self._properties:
            self._properties[ref] = small_unpickle(t.properties[ref])
        size = torch.Size(t.size(i).tolist())
        return size, self._properties[ref]


Entries = Union[ObjectEntries, TableEntries]


def metadata_share_hash(entries: Entries, rank: int, world_size: int) -> int:
    """Sum of the hashes of this rank's contiguous 1/N slice of the metadata's chunks, mod 2^128."""
    counts = entries.counts()
    n_chunks = sum(counts)
    lo, hi = n_chunks * rank // world_size, n_chunks * (rank + 1) // world_size
    total, pos = 0, 0
    for i, count in enumerate(counts):
        if pos >= hi:
            break
        if pos + count > lo:
            fqn = entries.fqn(i)
            if entries.is_tensor(i):
                fqn_hash = _fqn_hash(fqn, b"T")
                for offsets, sizes in entries.chunks(i, max(lo - pos, 0), min(hi - pos, count)):
                    total += _tensor_chunk_hash(fqn_hash, offsets, sizes)
            else:
                total += _bytes_hash(fqn)
        pos += count
    return total & HASH_MASK


def mismatch(plan: SavePlan, entries: Entries) -> str:
    """Why this rank's plan can't reuse the metadata, judged locally; empty if it may."""
    chunk_fields = [f.name for f in fields(ChunkStorageMetadata)]
    if chunk_fields != ["offsets", "sizes"]:
        return f"unknown ChunkStorageMetadata fields {chunk_fields}"
    if plan.planner_data or entries.planner_data:
        return "planner data is not checked"
    for item in plan.items:
        # Bytes entries have no attributes; their fqn and kind are compared by the chunk hash.
        if item.type == WriteItemType.BYTE_IO:
            continue
        fqn = item.index.fqn
        entry = entries.tensor(fqn)
        if isinstance(entry, str):
            return f"{fqn!r}: tensor, metadata has {entry}"
        size, properties = entry
        data = item.tensor_data
        if size != data.size:
            return f"{fqn!r}: size {tuple(data.size)}, metadata has {tuple(size)}"
        if properties != data.properties:
            return f"{fqn!r}: properties {data.properties}, metadata has {properties}"
    return ""


def votes(plan: SavePlan, entries: Entries, rank: int, world_size: int) -> List[int]:
    """This rank's contribution to the reuse all_reduce (sum): [failures, hash lanes...].

    The hash difference is split into 32-bit lanes, so the sum fits in int64 for any world size
    below 2^31 and no lane relies on overflow. Reuse is possible if, after summing over all ranks,
    `can_reuse` holds. A rank that can't compute its share votes against reuse.
    """
    try:
        reason = mismatch(plan, entries)
        if reason:
            logger.debug(f"rank {rank}: can't reuse the loaded metadata: {reason}")
            return [1] + [0] * LANES
        diff = (plan_hash(plan) - metadata_share_hash(entries, rank, world_size)) & HASH_MASK
    except Exception as e:  # noqa: BLE001 - any failure here means "don't reuse"
        logger.warning(f"rank {rank}: can't check the loaded metadata for reuse: {e!r}")
        return [1] + [0] * LANES
    return [0] + [(diff >> (LANE_BITS * k)) & LANE_MASK for k in range(LANES)]


def can_reuse(summed_votes: List[int]) -> bool:
    """Whether the votes summed over all ranks allow reusing the metadata."""
    failures, lanes = summed_votes[0], summed_votes[1:]
    # Each lane sum is exact (32-bit values summed in int64) but can exceed 32 bits. Python ints
    # have arbitrary precision, so recombining the lane sums at their 32-bit offsets carries the
    # excess into the next lane, as adding the 128-bit values directly would; then reduce mod 2^128.
    total = sum(lane << (LANE_BITS * k) for k, lane in enumerate(lanes))
    return failures == 0 and total & HASH_MASK == 0
