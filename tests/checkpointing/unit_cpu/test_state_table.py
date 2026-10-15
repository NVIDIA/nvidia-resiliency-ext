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

import dataclasses
import gc
import io
import logging
import pickle
from collections import OrderedDict

import numpy as np
import pytest
import torch
from hypothesis import given
from torch.distributed.checkpoint.storage import WriteResult

from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler import (
    pickler,
    state_table,
    table,
    writer,
)
from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler.state_table import (
    BYTES,
    OTHER,
    TENSOR,
)

from .test_metadata_pickler import (
    FIXED_METADATA,
    NEEDS_FAST,
    NEEDS_NATIVE,
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
    state_table.check(t)
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


def assert_same_tables(a: state_table.StateDictTable, b: state_table.StateDictTable) -> None:
    """a and b hold the same values, in arrays of the same types."""
    for f in dataclasses.fields(state_table.StateDictTable):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if isinstance(va, np.ndarray):
            assert va.dtype == vb.dtype and np.array_equal(va, vb), f.name
        else:
            assert va == vb, f.name


@NEEDS_NATIVE
@hypothesis_settings
@given(metadatas())
def test_native_converts_like_python(md):
    assert_same_tables(writer.native.from_metadata(md), state_table.from_metadata(md))


@NEEDS_NATIVE
@pytest.mark.parametrize("make", FIXED_METADATA)
def test_native_converts_like_python_on(make, tmp_path):
    md = make(tmp_path)
    assert_same_tables(writer.native.from_metadata(md), state_table.from_metadata(md))


@NEEDS_NATIVE
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
def test_native_rejects(mutate):
    md = writer._sample_metadata()
    mutate(md)
    with pytest.raises(TypeError):
        writer.native.from_metadata(md)


def _replace(t, **arrays):
    return dataclasses.replace(t, **arrays)


def _bump(a: np.ndarray, i: int, by: int) -> np.ndarray:
    a = a.copy()
    a[i] += by
    return a


# Malformed tables, which both writers must reject before reading their arrays.
MALFORMED = [
    pytest.param(lambda t: _replace(t, kind=t.kind.astype(np.int64)), id="kind-dtype"),
    pytest.param(lambda t: _replace(t, dims=t.dims[::2]), id="not-contiguous"),
    pytest.param(lambda t: _replace(t, ref=t.ref[:-1]), id="short-ref"),
    pytest.param(lambda t: _replace(t, fqns=t.fqns[:-1]), id="short-fqns"),
    pytest.param(lambda t: _replace(t, ndim=t.ndim[:-1]), id="short-ndim"),
    pytest.param(lambda t: _replace(t, dims_start=_bump(t.dims_start, -1, 1)), id="dims-range"),
    pytest.param(lambda t: _replace(t, dims_start=_bump(t.dims_start, 1, -1)), id="dims-mismatch"),
    pytest.param(lambda t: _replace(t, ndim=_bump(t.ndim, 0, 1)), id="ndim-mismatch"),
    pytest.param(lambda t: _replace(t, ndim=_bump(t.ndim, 0, -5)), id="negative-ndim"),
    pytest.param(lambda t: _replace(t, chunk_start=_bump(t.chunk_start, 1, 1)), id="chunk-count"),
    pytest.param(lambda t: _replace(t, kind=_bump(t.kind, 0, 7)), id="unknown-kind"),
    pytest.param(lambda t: _replace(t, ref=_bump(t.ref, 0, 99)), id="ref-out-of-range"),
    pytest.param(lambda t: _replace(t, properties=[1, *t.properties[1:]]), id="properties-type"),
    pytest.param(lambda t: _replace(t, fields=[*t.fields, ("x", None)]), id="field-opcodes"),
    pytest.param(lambda t: _replace(t, fields=[["x", b"N"]]), id="field-not-tuple"),
    pytest.param(
        lambda t: _replace(t, kind=np.where(t.kind == TENSOR, BYTES, t.kind).astype(np.uint8)),
        id="bytes-with-dims",
    ),
]


@pytest.mark.parametrize(
    "dumps_table",
    [pickler.dumps_table, *([writer.native.dumps_table] if writer.native else [])],
    ids=lambda f: f.__module__.rsplit(".", 1)[-1],
)
@pytest.mark.parametrize("corrupt", MALFORMED)
def test_writers_reject_malformed_tables(dumps_table, corrupt):
    t = corrupt(state_table.from_metadata(writer._sample_metadata()))
    with pytest.raises(ValueError):
        dumps_table(t)


def _storage_rows(md) -> np.ndarray:
    """md.storage_data as gathered write-result tables: one rank's table, in one row."""
    payload = table.encode(
        [WriteResult(index, info.length, info) for index, info in md.storage_data.items()]
    )
    return np.frombuffer(payload + bytes(-len(payload) % 8), dtype=np.uint8)[None, :]


@pytest.mark.parametrize("make", FIXED_METADATA)
def test_dump_metadata_of_table(make, tmp_path):
    """A table is written as its Metadata, with the storage_data given, or None."""
    md = make(tmp_path)
    t = state_table.from_metadata(md)
    for storage_data in (md.storage_data, None):
        buf = io.BytesIO()
        writer.dump_metadata(t, buf, storage_data=storage_data)
        expected = dataclasses.replace(md, storage_data=storage_data)
        assert_identical(pickle.loads(buf.getvalue()), stock_round_trip(expected))


def test_dump_metadata_of_table_with_rows():
    md = writer._sample_metadata()
    rows = _storage_rows(md)
    a, b = io.BytesIO(), io.BytesIO()
    writer.dump_metadata(state_table.from_metadata(md), a, storage_rows=rows)
    writer.dump_metadata(md, b, storage_rows=rows)
    assert a.getvalue() == b.getvalue()


def test_with_fields():
    """Fields set on a table are written as if set on the Metadata; a new one comes last."""
    md = writer._sample_metadata()
    t = state_table.with_fields(state_table.from_metadata(md), version="9.9", extra=[1, 2])
    md.version, md.extra = "9.9", [1, 2]
    buf = io.BytesIO()
    writer.dump_metadata(t, buf, storage_data=md.storage_data)
    assert_identical(pickle.loads(buf.getvalue()), stock_round_trip(md))
    with pytest.raises(ValueError):
        state_table.with_fields(t, storage_data={})


def test_to_metadata():
    md = writer._sample_metadata()
    back = writer.to_metadata(state_table.from_metadata(md), md.storage_data)
    assert back.storage_data is md.storage_data
    assert_identical(
        dataclasses.replace(back, storage_data=None),
        stock_round_trip(dataclasses.replace(md, storage_data=None)),
    )


@NEEDS_FAST
def test_to_table_warns_and_returns_none(caplog):
    md = writer._sample_metadata()
    md.__class__ = _Metadata
    with caplog.at_level(logging.WARNING, logger=writer.__name__):
        assert writer.to_table(md) is None
    assert "can't convert" in caplog.text


def test_no_table_without_the_fast_writers(monkeypatch):
    monkeypatch.setenv("NVRX_FAST_METADATA_PICKLE", "0")
    assert writer.to_table(writer._sample_metadata()) is None


@NEEDS_FAST
@NEEDS_NATIVE
def test_dump_metadata_of_table_falls_back_to_python(caplog):
    """A table the native writer can't write (a lone surrogate) is written by the Python one."""
    md = writer._sample_metadata()
    _rename_first(md, lambda s: s + "\ud800")
    t = writer.native.from_metadata(md)
    buf = io.BytesIO()
    with caplog.at_level(logging.WARNING, logger=writer.__name__):
        writer.dump_metadata(t, buf, storage_data=md.storage_data)
    assert "failed to write" in caplog.text
    assert_identical(pickle.loads(buf.getvalue()), stock_round_trip(md))


def test_to_metadata_without_a_writer(monkeypatch):
    """to_metadata builds the metadata itself when no writer can write the table."""
    md = writer._sample_metadata()
    t = state_table.from_metadata(md)

    def fail(*args, **kwargs):
        raise RuntimeError("no")

    monkeypatch.setattr(pickler, "dumps_table", fail)
    if writer.native:
        monkeypatch.setattr(writer.native, "dumps_table", fail)
    back = writer.to_metadata(t, md.storage_data)
    assert back == md and list(vars(back)) == list(vars(md))
