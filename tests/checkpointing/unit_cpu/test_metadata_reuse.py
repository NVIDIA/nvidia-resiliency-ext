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

"""Reusing global metadata: the check of a save's local plans against a loaded checkpoint's
metadata, and the CheckpointMetadataCache that carries metadata and plans between saves.

Ranks are simulated: each plan's votes are computed as its rank would, and summed here as the
all_reduce would.
"""

import dataclasses
import pickle
from dataclasses import fields

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    create_default_global_save_plan,
)
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import SavePlan, TensorWriteData, WriteItem, WriteItemType

from nvidia_resiliency_ext.checkpointing.async_ckpt import _metadata_reuse
from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import CheckpointMetadataCache


def tensor_item(fqn, offsets, sizes, size, dtype=torch.float32, requires_grad=False):
    """Write item of one chunk of a tensor."""
    return WriteItem(
        index=MetadataIndex(fqn, torch.Size(offsets)),
        type=WriteItemType.SHARD,
        tensor_data=TensorWriteData(
            chunk=ChunkStorageMetadata(torch.Size(offsets), torch.Size(sizes)),
            properties=TensorProperties(dtype=dtype, requires_grad=requires_grad),
            size=torch.Size(size),
        ),
    )


def bytes_item(fqn):
    """Write item of a bytes entry."""
    return WriteItem(index=MetadataIndex(fqn), type=WriteItemType.BYTE_IO)


def sample_plans():
    """Four ranks' plans: a row-sharded tensor, two-chunk-per-rank experts, bytes, one scalar."""
    plans = []
    for rank in range(4):
        items = [tensor_item("w", (2 * rank, 0), (2, 4), (8, 4))]
        items += [
            tensor_item("experts", (2 * rank + e, 0, 0), (1, 3, 3), (8, 3, 3)) for e in (0, 1)
        ]
        items.append(bytes_item(f"rank{rank}.state"))
        if rank == 0:
            items.append(tensor_item("step", (), (), ()))
        plans.append(SavePlan(items))
    return plans


def metadata_of(plans):
    """Global metadata as the default planner builds it, through a pickle round trip like a load."""
    _, metadata = create_default_global_save_plan(plans)
    return pickle.loads(pickle.dumps(metadata))


def reusable(plans, metadata):
    """Whether the simulated ranks' summed votes allow reuse."""
    votes = [_metadata_reuse.votes(p, metadata, r, len(plans)) for r, p in enumerate(plans)]
    return _metadata_reuse.can_reuse([sum(column) for column in zip(*votes)])


def with_items(plans, rank, items):
    """Plans with rank's items replaced."""
    return plans[:rank] + [dataclasses.replace(plans[rank], items=items)] + plans[rank + 1 :]


def changed_chunk(item, **changes):
    """item with its chunk changed."""
    data = item.tensor_data
    return dataclasses.replace(
        item,
        tensor_data=dataclasses.replace(data, chunk=dataclasses.replace(data.chunk, **changes)),
    )


def changed_data(item, **changes):
    """item with its tensor data changed."""
    return dataclasses.replace(item, tensor_data=dataclasses.replace(item.tensor_data, **changes))


def changed_properties(item, **changes):
    """item with its tensor properties changed."""
    properties = dataclasses.replace(item.tensor_data.properties, **changes)
    return changed_data(item, properties=properties)


def test_same_plans_reuse():
    plans = sample_plans()
    assert reusable(plans, metadata_of(plans))


def test_metadata_from_a_real_save_reuses():
    """Plans and metadata as DefaultSavePlanner creates them for a state dict."""
    state_dict = {"a": torch.ones(4, 4), "b": torch.arange(6.0), "extra": {"k": 1}}
    planner = DefaultSavePlanner(flatten_state_dict=False)
    planner.set_up_planner(state_dict, is_coordinator=True)
    plan = planner.create_local_plan()
    _, metadata = planner.create_global_plan([plan])
    assert reusable([plan], pickle.loads(pickle.dumps(metadata)))


CHANGES = {
    "drop a chunk": lambda plans: with_items(plans, 1, plans[1].items[1:]),
    "drop the only chunk of a tensor": lambda plans: with_items(plans, 0, plans[0].items[:-1]),
    "drop a bytes entry": lambda plans: with_items(plans, 2, plans[2].items[:-1]),
    "write a chunk twice": lambda plans: with_items(plans, 3, plans[3].items + [plans[0].items[0]]),
    "drop one chunk, write another twice": lambda plans: with_items(
        with_items(plans, 1, plans[1].items[1:]), 3, plans[3].items + [plans[0].items[0]]
    ),
    "add a tensor": lambda plans: with_items(
        plans, 2, plans[2].items + [tensor_item("new", (0,), (2,), (2,))]
    ),
    "rename a tensor": lambda plans: with_items(
        plans,
        0,
        [dataclasses.replace(plans[0].items[-1], index=MetadataIndex("renamed", ()))]
        + plans[0].items[:-1],
    ),
    "change chunk sizes": lambda plans: with_items(
        plans, 1, [changed_chunk(plans[1].items[0], sizes=torch.Size((1, 4)))] + plans[1].items[1:]
    ),
    "change chunk offsets": lambda plans: with_items(
        plans,
        1,
        [changed_chunk(plans[1].items[0], offsets=torch.Size((3, 0)))] + plans[1].items[1:],
    ),
    "change global size": lambda plans: with_items(
        plans, 1, [changed_data(plans[1].items[0], size=torch.Size((9, 4)))] + plans[1].items[1:]
    ),
    "change dtype": lambda plans: with_items(
        plans,
        1,
        [changed_properties(plans[1].items[0], dtype=torch.bfloat16)] + plans[1].items[1:],
    ),
    "change requires_grad": lambda plans: with_items(
        plans,
        1,
        [changed_properties(plans[1].items[0], requires_grad=True)] + plans[1].items[1:],
    ),
    "bytes become a tensor": lambda plans: with_items(
        plans, 2, plans[2].items[:-1] + [tensor_item("rank2.state", (0,), (1,), (1,))]
    ),
    "a tensor becomes bytes": lambda plans: with_items(
        plans, 0, plans[0].items[:-1] + [bytes_item("step")]
    ),
}


@pytest.mark.parametrize("change", CHANGES.values(), ids=CHANGES.keys())
def test_changed_plans_dont_reuse(change):
    plans = sample_plans()
    metadata = metadata_of(plans)
    assert not reusable(change(plans), metadata)


def test_chunk_moved_to_another_rank_reuses():
    """The metadata doesn't record which rank writes a chunk."""
    plans = sample_plans()
    moved = plans[1].items[0]
    plans_after = with_items(with_items(plans, 1, plans[1].items[1:]), 2, plans[2].items + [moved])
    assert reusable(plans_after, metadata_of(plans))


@pytest.mark.parametrize("world_size", [1, 2, 3, 7, 40])
def test_same_chunks_on_another_world_size_reuse(world_size):
    """Any distribution of the same chunks over any number of ranks, also ranks writing nothing."""
    plans = sample_plans()
    items = [item for plan in plans for item in plan.items]
    new_plans = [SavePlan(items[r::world_size]) for r in range(world_size)]
    assert reusable(new_plans, metadata_of(plans))


def test_planner_data_doesnt_reuse():
    """Planner data isn't checked, so metadata or plans carrying it are never reused."""
    plans = sample_plans()
    metadata = metadata_of(plans)
    assert not reusable(plans, dataclasses.replace(metadata, planner_data={"w": ("w",)}))
    assert not reusable(
        [dataclasses.replace(p, planner_data={"w": ("w",)}) for p in plans], metadata
    )


def test_unknown_chunk_fields_dont_reuse(monkeypatch):
    """A field torch adds to ChunkStorageMetadata would not be compared, so refuse reuse."""

    @dataclasses.dataclass
    class ChunkWithStride:
        offsets: torch.Size
        sizes: torch.Size
        stride: torch.Size = None

    monkeypatch.setattr(_metadata_reuse, "ChunkStorageMetadata", ChunkWithStride)
    plans = sample_plans()
    assert not reusable(plans, metadata_of(plans))


def test_unreadable_metadata_votes_against_reuse():
    """A rank that can't check the metadata votes against reuse instead of raising."""
    plans = sample_plans()
    metadata = metadata_of(plans)
    metadata.state_dict_metadata["experts"].chunks = None
    votes = _metadata_reuse.votes(plans[0], metadata, 0, len(plans))
    assert votes[0] == 1


def test_hash_lanes_carry():
    """Lane sums past 32 bits carry into the next lane, as the 128-bit sum does."""
    lanes = _metadata_reuse.LANES
    minus_one = [0] + [_metadata_reuse.LANE_MASK] * lanes  # -1 mod 2^128
    plus_one = [0, 1] + [0] * (lanes - 1)
    assert _metadata_reuse.can_reuse([a + b for a, b in zip(minus_one, plus_one)])
    assert not _metadata_reuse.can_reuse(minus_one)
    assert not _metadata_reuse.can_reuse([1] + [0] * lanes)


@st.composite
def chunked_tensors(draw):
    """Tensors split into row blocks: (fqn, rows, cols, block rows) per tensor."""
    names = draw(st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=5, unique=True))
    return [
        (name, draw(st.integers(1, 12)), draw(st.integers(1, 4)), draw(st.integers(1, 4)))
        for name in names
    ]


def all_items(tensors):
    """Every chunk's write item."""
    return [
        tensor_item(name, (row, 0), (min(block, rows - row), cols), (rows, cols))
        for name, rows, cols, block in tensors
        for row in range(0, rows, block)
    ]


@settings(max_examples=50, deadline=None)
@given(tensors=chunked_tensors(), data=st.data())
def test_any_distribution_of_the_same_chunks_reuses(tensors, data):
    items = all_items(tensors)
    old_ranks = data.draw(st.integers(1, 6))
    new_ranks = data.draw(st.integers(1, 6))
    old_owner = data.draw(
        st.lists(st.integers(0, old_ranks - 1), min_size=len(items), max_size=len(items))
    )
    new_owner = data.draw(st.permutations(range(len(items))))
    old_plans = [
        SavePlan([i for i, o in zip(items, old_owner) if o == r]) for r in range(old_ranks)
    ]
    shuffled = [items[i] for i in new_owner]
    new_plans = [SavePlan(shuffled[r::new_ranks]) for r in range(new_ranks)]
    assert reusable(new_plans, metadata_of(old_plans))


@settings(max_examples=50, deadline=None)
@given(tensors=chunked_tensors(), data=st.data())
def test_any_missing_chunk_doesnt_reuse(tensors, data):
    items = all_items(tensors)
    dropped = data.draw(st.integers(0, len(items) - 1))
    plans = [SavePlan(items[r::3]) for r in range(3)]
    kept = items[:dropped] + items[dropped + 1 :]
    assert not reusable([SavePlan(kept[r::3]) for r in range(3)], metadata_of(plans))


def test_set_cached_global_metadata_leaves_the_metadata_alone():
    """The caller (e.g. Megatron's load strategy) keeps using the metadata it passes."""
    metadata = metadata_of(sample_plans())
    metadata.all_local_plans = sample_plans()  # as written by older versions
    before = dict(vars(metadata))
    cache = CheckpointMetadataCache()
    cache.set_cached_global_metadata(metadata)
    assert vars(metadata) == before
    assert cache.metadata is metadata


def test_reused_metadata_is_a_copy_without_local_plans():
    """The coordinator writes a copy: finish may modify it, and old local plans aren't written."""
    plans = sample_plans()
    metadata = metadata_of(plans)
    metadata.all_local_plans = plans
    storage_data = metadata.storage_data
    cache = CheckpointMetadataCache()
    cache.set_cached_global_metadata(metadata)
    written = cache._update(plans[0], plans[0], None, reused=True, is_coordinator=True)
    assert written is not metadata
    assert "all_local_plans" not in vars(written)
    assert all(getattr(written, f.name) == getattr(metadata, f.name) for f in fields(metadata))
    written.storage_data = {"rewritten": None}  # as finish does
    assert cache.metadata.storage_data is storage_data
    again = cache._update(plans[0], plans[0], None, reused=True, is_coordinator=True)
    assert again is not written and again.storage_data is storage_data


def test_only_the_coordinator_keeps_metadata_after_a_save():
    """Other ranks drop the loaded metadata once the first save checked it."""
    plans = sample_plans()
    metadata = metadata_of(plans)
    other = CheckpointMetadataCache()
    other.set_cached_global_metadata(metadata)
    assert other._update(plans[1], plans[1], None, reused=True, is_coordinator=False) is None
    assert other.metadata is None and other.local_plan is plans[1]

    built = metadata_of(plans)
    coordinator = CheckpointMetadataCache()
    coordinator.set_cached_global_metadata(metadata)
    assert coordinator._update(plans[0], plans[0], built, False, is_coordinator=True) is built
    assert coordinator.metadata is built


def test_reuse_without_metadata_on_the_coordinator_raises():
    """E.g. a different coordinator rank than in the previous save."""
    plans = sample_plans()
    cache = CheckpointMetadataCache()
    with pytest.raises(RuntimeError, match="coordinator rank must stay the same"):
        cache._update(plans[0], plans[0], None, reused=True, is_coordinator=True)


def test_set_cached_global_metadata_starts_over():
    """After loading a checkpoint, the next save is checked against its metadata again."""
    plans = sample_plans()
    cache = CheckpointMetadataCache()
    cache._update(plans[0], plans[0], metadata_of(plans), reused=False, is_coordinator=True)
    loaded = metadata_of(plans)
    cache.set_cached_global_metadata(loaded)
    assert cache.metadata is loaded and cache.local_plan is None and cache.central_plan is None
