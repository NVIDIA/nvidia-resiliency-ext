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
"""

import struct
from dataclasses import fields
from hashlib import blake2b
from logging import getLogger
from typing import List

from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import SavePlan, WriteItemType

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


def _tensor_chunk_hash(fqn_hash: "blake2b", chunk: ChunkStorageMetadata) -> int:
    """Hash of one tensor chunk: its fqn, offsets and sizes."""
    h = fqn_hash.copy()
    offsets, sizes = chunk.offsets, chunk.sizes
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
            total += _tensor_chunk_hash(_fqn_hash(item.index.fqn, b"T"), item.tensor_data.chunk)
    return total & HASH_MASK


def metadata_share_hash(metadata: Metadata, rank: int, world_size: int) -> int:
    """Sum of the hashes of this rank's contiguous 1/N slice of the metadata's chunks, mod 2^128."""
    entries = metadata.state_dict_metadata
    counts = [
        len(entry.chunks) if isinstance(entry, TensorStorageMetadata) else 1
        for entry in entries.values()
    ]
    n_chunks = sum(counts)
    lo, hi = n_chunks * rank // world_size, n_chunks * (rank + 1) // world_size
    total, pos = 0, 0
    for (fqn, entry), count in zip(entries.items(), counts):
        if pos >= hi:
            break
        if pos + count > lo:
            if isinstance(entry, TensorStorageMetadata):
                fqn_hash = _fqn_hash(fqn, b"T")
                for chunk in entry.chunks[max(lo - pos, 0) : min(hi - pos, count)]:
                    total += _tensor_chunk_hash(fqn_hash, chunk)
            else:
                total += _bytes_hash(fqn)
        pos += count
    return total & HASH_MASK


def mismatch(plan: SavePlan, metadata: Metadata) -> str:
    """Why this rank's plan can't reuse the metadata, judged locally; empty if it may."""
    chunk_fields = [f.name for f in fields(ChunkStorageMetadata)]
    if chunk_fields != ["offsets", "sizes"]:
        return f"unknown ChunkStorageMetadata fields {chunk_fields}"
    if plan.planner_data or metadata.planner_data:
        return "planner data is not checked"
    entries = metadata.state_dict_metadata
    for item in plan.items:
        # Bytes entries carry nothing to compare; bytes vs tensor is part of the chunk hash.
        if item.type == WriteItemType.BYTE_IO:
            continue
        fqn = item.index.fqn
        entry = entries.get(fqn)
        if not isinstance(entry, TensorStorageMetadata):
            return f"{fqn!r}: tensor, metadata has {type(entry).__name__}"
        data = item.tensor_data
        if entry.size != data.size:
            return f"{fqn!r}: size {tuple(data.size)}, metadata has {tuple(entry.size)}"
        if entry.properties != data.properties:
            return f"{fqn!r}: properties {data.properties}, metadata has {entry.properties}"
    return ""


def votes(plan: SavePlan, metadata: Metadata, rank: int, world_size: int) -> List[int]:
    """This rank's contribution to the reuse all_reduce (sum): [failures, hash lanes...].

    The hash difference is split into 32-bit lanes, so the sum fits in int64 for any world size
    below 2^31 and no lane relies on overflow. Reuse is possible if, after summing over all ranks,
    `can_reuse` holds. A rank that can't compute its share votes against reuse.
    """
    try:
        reason = mismatch(plan, metadata)
        if reason:
            logger.debug(f"rank {rank}: can't reuse the loaded metadata: {reason}")
            return [1] + [0] * LANES
        diff = (plan_hash(plan) - metadata_share_hash(metadata, rank, world_size)) & HASH_MASK
    except Exception as e:  # noqa: BLE001 - any failure here means "don't reuse"
        logger.warning(f"rank {rank}: can't check the loaded metadata for reuse: {e!r}")
        return [1] + [0] * LANES
    return [0] + [(diff >> (LANE_BITS * k)) & LANE_MASK for k in range(LANES)]


def can_reuse(summed_votes: List[int]) -> bool:
    """Whether the votes summed over all ranks allow reusing the metadata."""
    failures, lanes = summed_votes[0], summed_votes[1:]
    total = sum(lane << (LANE_BITS * k) for k, lane in enumerate(lanes))
    return failures == 0 and total & HASH_MASK == 0
