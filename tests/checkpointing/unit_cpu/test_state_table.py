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

"""Unit tests of StateDictTable, a Metadata without storage_data as arrays (CPU only)."""

import gc
from collections import OrderedDict

import pytest
import torch
from hypothesis import given

from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler import state_table, writer
from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler.state_table import (
    BYTES,
    OTHER,
    TENSOR,
)

from .test_metadata_pickler import (
    FIXED_METADATA,
    _Metadata,
    _metadata_changed_while_pickled,
    _rename_first,
    _Str,
    assert_identical,
    hypothesis_settings,
    large_metadata,
    metadatas,
    properties_of_equal_values,
    stock_round_trip,
    subclassed_metadata,
)


def assert_consistent(t: state_table.StateDictTable) -> None:
    """The arrays describe len(t) entries and use exactly the dims they hold."""
    n = len(t)
    assert len(t.kind) == len(t.ref) == len(t.ndim) == n
    for starts in (t.chunk_start, t.dims_start):
        assert len(starts) == n + 1 and starts[0] == 0 and (starts[1:] >= starts[:-1]).all()
    assert t.dims_start[-1] == len(t.dims)
    for i in range(n):
        values = t.dims_start[i + 1] - t.dims_start[i]
        assert values == t.ndim[i] * (1 + 2 * t.chunk_count(i))
        if t.kind[i] != TENSOR:
            assert t.ndim[i] == 0 and t.chunk_count(i) == 0


def assert_converted(md) -> state_table.StateDictTable:
    """md converts to a consistent table whose entries and fields are md's as stock pickle
    round-trips them."""
    t = state_table.from_metadata(md)
    md = stock_round_trip(md)
    assert_consistent(t)
    assert t.fqns == list(md.state_dict_metadata)
    assert t.index == {fqn: i for i, fqn in enumerate(t.fqns)}
    for i, (fqn, entry) in enumerate(md.state_dict_metadata.items()):
        assert_identical(t.entry(i), entry, fqn)
    assert [name for name, _ in t.fields] == list(vars(md))
    for name, opcodes in t.fields:
        if name in (state_table.STATE_DICT_METADATA, state_table.STORAGE_DATA):
            assert opcodes is None
        else:
            assert_identical(state_table.small_unpickle(opcodes), getattr(md, name), name)
    return t


@hypothesis_settings
@given(metadatas())
def test_converts(md):
    assert_converted(md)


@pytest.mark.parametrize("make", FIXED_METADATA)
def test_converts_on(make, tmp_path):
    assert_converted(make(tmp_path))


def test_kinds():
    md = subclassed_metadata()
    t = assert_converted(md)
    kinds = dict(zip(t.fqns, t.kind.tolist()))
    assert kinds["w"] == kinds["s"] == TENSOR
    assert kinds["w2"] == OTHER  # a subclass
    assert kinds["obj.0/shard_0"] == BYTES


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda w: setattr(w, "chunks", tuple(w.chunks)), id="tuple-chunks"),
        pytest.param(lambda w: setattr(w.chunks[0], "offsets", (0, 0, 0, 0)), id="tuple-size"),
        pytest.param(lambda w: setattr(w, "size", torch.Size([4, 2])), id="chunk-ndim-differs"),
        pytest.param(lambda w: object.__setattr__(w, "extra", 1), id="tensor-extra-attribute"),
        pytest.param(
            lambda w: object.__setattr__(w.chunks[0], "extra", 1), id="chunk-extra-attribute"
        ),
    ],
)
def test_unusual_tensors_are_kept_as_pickles(mutate):
    """A tensor entry the arrays don't describe exactly is kept as the stdlib pickler's opcodes."""
    md = writer._sample_metadata()
    mutate(md.state_dict_metadata["w"])
    t = assert_converted(md)
    assert t.kind[t.index["w"]] == OTHER and t.kind[t.index["s"]] == TENSOR


def test_properties_are_kept_once_per_pickle():
    """Equal properties are kept once; equal values of different types (True, 1) are not equal."""
    md = properties_of_equal_values()
    t = assert_converted(md)
    ref = dict(zip(t.fqns, t.ref.tolist()))
    assert ref["w1"] != ref["w2"]
    assert len(t.properties) == len(set(t.properties)) == 4  # w, s, w1, w2


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda md: setattr(md, "__class__", _Metadata), id="metadata-subclass"),
        pytest.param(
            lambda md: setattr(md, "state_dict_metadata", OrderedDict(md.state_dict_metadata)),
            id="ordered-dict",
        ),
        pytest.param(lambda md: _rename_first(md, _Str), id="str-subclass"),
    ],
)
def test_rejects(mutate):
    md = writer._sample_metadata()
    mutate(md)
    with pytest.raises(TypeError):
        state_table.from_metadata(md)


@pytest.mark.parametrize("case", ["chunk-list-emptied", "chunk-list-replaced", "entries-deleted"])
def test_metadata_changed_while_converted(case):
    """Python code run during the conversion can change the metadata: the conversion raises, or
    returns a consistent table."""
    md = _metadata_changed_while_pickled(case)
    try:
        t = state_table.from_metadata(md)
    except RuntimeError:
        return
    assert_consistent(t)


def test_few_objects_for_the_gc():
    """However large the metadata, the table holds a handful of objects the garbage collector
    visits: its arrays, lists and fields, not one per entry or chunk."""
    t = state_table.from_metadata(large_metadata())
    seen, stack = {id(t)}, [t]
    while stack:
        for r in gc.get_referents(stack.pop()):
            if gc.is_tracked(r) and id(r) not in seen and not isinstance(r, type):
                seen.add(id(r))
                stack.append(r)
    assert len(t) == 1501 and len(seen) < 40
