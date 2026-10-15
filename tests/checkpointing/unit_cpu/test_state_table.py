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
import types
from collections import OrderedDict

import numpy as np
import pytest
import torch
from hypothesis import assume, given
from hypothesis import strategies as st
from torch.distributed.checkpoint.storage import WriteResult

from nvidia_resiliency_ext.checkpointing.async_ckpt import _metadata_reuse
from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler import (
    pickler,
    state_table,
    table,
    writer,
)
from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler.opcodes import small_unpickle
from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler.state_table import (
    BYTES,
    OTHER,
    TENSOR,
    StateDictTable,
)
from nvidia_resiliency_ext.checkpointing.async_ckpt.metadata_template import MetadataTemplate

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
    int64s,
    large_metadata,
    metadatas,
    properties_of_equal_values,
    stock_round_trip,
    subclassed_metadata,
    texts,
)


def assert_consistent(table: state_table.StateDictTable) -> None:
    """The arrays describe len(table) entries and use exactly the dims they hold."""
    n = len(table)
    assert len(table.kind) == len(table.ref) == len(table.ndim) == n
    for starts in (table.chunk_start, table.dims_start):
        assert len(starts) == n + 1 and starts[0] == 0 and (starts[1:] >= starts[:-1]).all()
    assert table.dims_start[-1] == len(table.dims)
    for i in range(n):
        values = table.dims_start[i + 1] - table.dims_start[i]
        assert values == table.ndim[i] * (1 + 2 * table.chunk_count(i))
        if table.kind[i] != TENSOR:
            assert table.ndim[i] == 0 and table.chunk_count(i) == 0


def assert_converted(metadata) -> state_table.StateDictTable:
    """metadata converts to a consistent table whose entries and fields are metadata's as stock pickle
    round-trips them."""
    table = StateDictTable.from_metadata(metadata)
    metadata = stock_round_trip(metadata)
    assert_consistent(table)
    assert list(table.fqns) == list(metadata.state_dict_metadata)
    assert table.index == {fqn: i for i, fqn in enumerate(table.fqns)}
    for i, (fqn, entry) in enumerate(metadata.state_dict_metadata.items()):
        assert_identical(table.entry(i), entry, fqn)
    assert [name for name, _ in table.fields] == list(vars(metadata))
    for name, opcodes in table.fields:
        if name in (state_table.STATE_DICT_METADATA, state_table.STORAGE_DATA):
            assert opcodes is None
        else:
            assert_identical(small_unpickle(opcodes), getattr(metadata, name), name)
    return table


@hypothesis_settings
@given(metadatas())
def test_converts(metadata):
    assert_converted(metadata)


@pytest.mark.parametrize("make", FIXED_METADATA)
def test_converts_on(make, tmp_path):
    assert_converted(make(tmp_path))


def test_kinds():
    metadata = subclassed_metadata()
    table = assert_converted(metadata)
    kinds = dict(zip(table.fqns, table.kind.tolist()))
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
    metadata = writer._sample_metadata()
    mutate(metadata.state_dict_metadata["w"])
    table = assert_converted(metadata)
    assert table.kind[table.index["w"]] == OTHER and table.kind[table.index["s"]] == TENSOR


def test_properties_are_kept_once_per_pickle():
    """Equal properties are kept once; equal values of different types (True, 1) are not equal."""
    metadata = properties_of_equal_values()
    table = assert_converted(metadata)
    ref = dict(zip(table.fqns, table.ref.tolist()))
    assert ref["w1"] != ref["w2"]
    assert len(table.properties) == len(set(table.properties)) == 4  # w, s, w1, w2


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda metadata: setattr(metadata, "__class__", _Metadata), id="metadata-subclass"
        ),
        pytest.param(
            lambda metadata: setattr(
                metadata, "state_dict_metadata", OrderedDict(metadata.state_dict_metadata)
            ),
            id="ordered-dict",
        ),
        pytest.param(lambda metadata: _rename_first(metadata, _Str), id="str-subclass"),
    ],
)
def test_rejects(mutate):
    metadata = writer._sample_metadata()
    mutate(metadata)
    with pytest.raises(TypeError):
        StateDictTable.from_metadata(metadata)


@pytest.mark.parametrize("case", ["chunk-list-emptied", "chunk-list-replaced", "entries-deleted"])
def test_metadata_changed_while_converted(case):
    """Python code run during the conversion can change the metadata: the conversion raises, or
    returns a consistent table."""
    metadata = _metadata_changed_while_pickled(case)
    try:
        table = StateDictTable.from_metadata(metadata)
    except RuntimeError:
        return
    assert_consistent(table)


def test_few_objects_for_the_gc():
    """However large the metadata, the table holds a handful of objects the garbage collector
    visits: its arrays, lists and fields, not one per entry or chunk."""
    table = StateDictTable.from_metadata(large_metadata())
    seen, stack = {id(table)}, [table]
    while stack:
        for r in gc.get_referents(stack.pop()):
            if gc.is_tracked(r) and id(r) not in seen and not isinstance(r, type):
                seen.add(id(r))
                stack.append(r)
    assert len(table) == 1501 and len(seen) < 40


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
def test_native_converts_like_python(metadata):
    assert_same_tables(
        writer.native.StateDictTable.from_metadata(metadata), StateDictTable.from_metadata(metadata)
    )


@NEEDS_NATIVE
@pytest.mark.parametrize("make", FIXED_METADATA)
def test_native_converts_like_python_on(make, tmp_path):
    metadata = make(tmp_path)
    assert_same_tables(
        writer.native.StateDictTable.from_metadata(metadata), StateDictTable.from_metadata(metadata)
    )


@NEEDS_NATIVE
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda metadata: setattr(metadata, "__class__", _Metadata), id="metadata-subclass"
        ),
        pytest.param(
            lambda metadata: setattr(
                metadata, "state_dict_metadata", OrderedDict(metadata.state_dict_metadata)
            ),
            id="ordered-dict",
        ),
        pytest.param(lambda metadata: _rename_first(metadata, _Str), id="str-subclass"),
    ],
)
def test_native_rejects(mutate):
    metadata = writer._sample_metadata()
    mutate(metadata)
    with pytest.raises(TypeError):
        writer.native.StateDictTable.from_metadata(metadata)


def _bump(a: np.ndarray, i: int, by: int) -> np.ndarray:
    a = a.copy()
    a[i] += by
    return a


def _attributes(table: StateDictTable, **changes) -> dict:
    """A table's attributes, with these changed."""
    return {**{f.name: getattr(table, f.name) for f in dataclasses.fields(table)}, **changes}


# Malformed tables, which a StateDictTable refuses to be and the C++ writer refuses to read.
MALFORMED = [
    pytest.param(lambda table: dict(kind=table.kind.astype(np.int64)), id="kind-dtype"),
    pytest.param(lambda table: dict(dims=table.dims[::2]), id="not-contiguous"),
    pytest.param(lambda table: dict(ref=table.ref[:-1]), id="short-ref"),
    pytest.param(lambda table: dict(fqns=table.fqns[:-1]), id="short-fqns"),
    pytest.param(lambda table: dict(ndim=table.ndim[:-1]), id="short-ndim"),
    pytest.param(lambda table: dict(dims_start=_bump(table.dims_start, -1, 1)), id="dims-range"),
    pytest.param(lambda table: dict(dims_start=_bump(table.dims_start, 1, -1)), id="dims-mismatch"),
    pytest.param(lambda table: dict(ndim=_bump(table.ndim, 0, 1)), id="ndim-mismatch"),
    pytest.param(lambda table: dict(ndim=_bump(table.ndim, 0, -5)), id="negative-ndim"),
    pytest.param(lambda table: dict(chunk_start=_bump(table.chunk_start, 1, 1)), id="chunk-count"),
    pytest.param(lambda table: dict(kind=_bump(table.kind, 0, 7)), id="unknown-kind"),
    pytest.param(lambda table: dict(ref=_bump(table.ref, 0, 99)), id="ref-out-of-range"),
    pytest.param(lambda table: dict(properties=(1, *table.properties[1:])), id="properties-type"),
    pytest.param(lambda table: dict(fields=(*table.fields, ("x", None))), id="field-opcodes"),
    pytest.param(lambda table: dict(fields=(["x", b"N"],)), id="field-not-tuple"),
    pytest.param(
        lambda table: dict(kind=np.where(table.kind == TENSOR, BYTES, table.kind).astype(np.uint8)),
        id="bytes-with-dims",
    ),
]


@pytest.mark.parametrize("corrupt", MALFORMED)
def test_malformed_tables_are_rejected_when_created(corrupt):
    table = StateDictTable.from_metadata(writer._sample_metadata())
    with pytest.raises(ValueError):
        StateDictTable(**_attributes(table, **corrupt(table)))


@NEEDS_NATIVE
@pytest.mark.parametrize("corrupt", MALFORMED)
def test_native_writer_rejects_malformed_tables(corrupt):
    """The C++ writer checks what it reads itself, also from an object that isn't a
    StateDictTable but has its attributes."""
    table = StateDictTable.from_metadata(writer._sample_metadata())
    with pytest.raises(ValueError):
        writer.native.dumps_table(types.SimpleNamespace(**_attributes(table, **corrupt(table))))


def test_python_writer_takes_only_state_dict_tables():
    table = StateDictTable.from_metadata(writer._sample_metadata())
    with pytest.raises(TypeError):
        pickler.dumps_table(types.SimpleNamespace(**_attributes(table)))


def test_tables_are_immutable():
    table = StateDictTable.from_metadata(writer._sample_metadata())
    with pytest.raises(dataclasses.FrozenInstanceError):
        table.dims = table.dims
    with pytest.raises(ValueError):
        table.dims[0] = 1
    with pytest.raises(TypeError):
        table.index["w"] = 0
    assert all(type(getattr(table, name)) is tuple for name in ("fields", "fqns", "others"))


def _storage_rows(metadata) -> np.ndarray:
    """metadata.storage_data as gathered write-result tables: one rank's table, in one row."""
    payload = table.encode(
        [WriteResult(index, info.length, info) for index, info in metadata.storage_data.items()]
    )
    return np.frombuffer(payload + bytes(-len(payload) % 8), dtype=np.uint8)[None, :]


@pytest.mark.parametrize("make", FIXED_METADATA)
def test_dump_metadata_of_table(make, tmp_path):
    """A table is written as its Metadata, with the storage_data given, or None."""
    metadata = make(tmp_path)
    table = StateDictTable.from_metadata(metadata)
    for storage_data in (metadata.storage_data, None):
        buf = io.BytesIO()
        writer.dump_metadata(table, buf, storage_data=storage_data)
        expected = dataclasses.replace(metadata, storage_data=storage_data)
        assert_identical(pickle.loads(buf.getvalue()), stock_round_trip(expected))


def test_dump_metadata_of_table_with_rows():
    metadata = writer._sample_metadata()
    rows = _storage_rows(metadata)
    a, b = io.BytesIO(), io.BytesIO()
    writer.dump_metadata(StateDictTable.from_metadata(metadata), a, storage_rows=rows)
    writer.dump_metadata(metadata, b, storage_rows=rows)
    assert a.getvalue() == b.getvalue()


def test_with_fields():
    """Fields set on a table are written as if set on the Metadata; one the table doesn't have yet
    comes last. Names that aren't fields a save sets are refused."""
    metadata = writer._sample_metadata()
    del metadata.version
    table = StateDictTable.from_metadata(metadata).with_fields(version="9.9")
    metadata.version = "9.9"
    buf = io.BytesIO()
    writer.dump_metadata(table, buf, storage_data=metadata.storage_data)
    assert_identical(pickle.loads(buf.getvalue()), stock_round_trip(metadata))
    for name in ("extra", "storage_data", "state_dict_metadata"):
        with pytest.raises(TypeError):
            table.with_fields(**{name: None})


def test_to_metadata():
    metadata = writer._sample_metadata()
    back = StateDictTable.from_metadata(metadata).to_metadata(metadata.storage_data)
    assert back.storage_data is metadata.storage_data
    assert_identical(
        dataclasses.replace(back, storage_data=None),
        stock_round_trip(dataclasses.replace(metadata, storage_data=None)),
    )


@NEEDS_FAST
def test_state_dict_table_if_fast_warns_and_returns_none(caplog):
    metadata = writer._sample_metadata()
    metadata.__class__ = _Metadata
    with caplog.at_level(logging.WARNING, logger=writer.__name__):
        assert writer.state_dict_table_if_fast(metadata) is None
    assert "can't convert" in caplog.text


def test_no_table_without_the_fast_writers(monkeypatch):
    monkeypatch.setenv("NVRX_FAST_METADATA_PICKLE", "0")
    assert writer.state_dict_table_if_fast(writer._sample_metadata()) is None


@NEEDS_FAST
@NEEDS_NATIVE
def test_dump_metadata_of_table_falls_back_to_python(caplog):
    """A table the native writer can't write (a lone surrogate) is written by the Python one."""
    metadata = writer._sample_metadata()
    _rename_first(metadata, lambda s: s + "\ud800")
    table = writer.native.StateDictTable.from_metadata(metadata)
    buf = io.BytesIO()
    with caplog.at_level(logging.WARNING, logger=writer.__name__):
        writer.dump_metadata(table, buf, storage_data=metadata.storage_data)
    assert "failed to write" in caplog.text
    assert_identical(pickle.loads(buf.getvalue()), stock_round_trip(metadata))


def test_dump_metadata_of_table_without_a_writer(monkeypatch):
    """dump_metadata writes the Metadata built from the table when no writer can write it."""
    metadata = writer._sample_metadata()
    table = StateDictTable.from_metadata(metadata)

    def fail(*args, **kwargs):
        raise RuntimeError("no")

    monkeypatch.setattr(pickler, "dumps_table", fail)
    if writer.native:
        monkeypatch.setattr(writer.native, "dumps_table", fail)
    buf = io.BytesIO()
    writer.dump_metadata(table, buf, storage_data=metadata.storage_data)
    back = pickle.loads(buf.getvalue())
    assert back == metadata and list(vars(back)) == list(vars(metadata))


# Properties over random metadata, next to the directed tests above.

WRITER_MODULES = [pickler, *([writer.native] if writer.native else [])]


def _random_rows(metadata, data):
    """metadata.storage_data as write-result tables of a random number of ranks, each a contiguous
    run of its items, so that the rows list the items in the dict's order; None if an item doesn't
    fit a table."""
    items = list(metadata.storage_data.items())
    cuts = sorted(data.draw(st.lists(st.integers(0, len(items)), max_size=4)))
    parts = [items[a:b] for a, b in zip([0, *cuts], [*cuts, len(items)])]
    try:
        payloads = [
            table.encode([WriteResult(index, info.length, info) for index, info in part])
            for part in parts
        ]
    except table.Unencodable:
        return None
    width = -(-max(map(len, payloads)) // 8) * 8
    rows = np.zeros((len(payloads), width), dtype=np.uint8)
    for rank, payload in enumerate(payloads):
        rows[rank, : len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return rows


@hypothesis_settings
@given(metadatas(), st.data())
def test_writing_a_table_matches_writing_the_metadata(metadata, data):
    """Both writers write a table, with storage_data or with rows of it, as the metadata."""
    state_dict_table = StateDictTable.from_metadata(metadata)
    expected = pickler.dumps(metadata)
    rows = _random_rows(metadata, data)
    for module in WRITER_MODULES:
        assert module.dumps_table(state_dict_table, metadata.storage_data) == expected
        if rows is not None:
            assert module.dumps_table(state_dict_table, storage_rows=rows) == expected


@hypothesis_settings
@given(metadatas(), st.none() | texts)
def test_table_round_trips(metadata, version):
    """to_metadata gives the metadata back, with the fields with_fields set, as stock pickle
    round-trips it; MetadataTemplate does in both its forms."""
    expected = stock_round_trip(metadata)
    expected.version, expected.storage_data = version, None
    state_dict_table = StateDictTable.from_metadata(metadata).with_fields(version=version)
    assert_identical(state_dict_table.to_metadata(), expected)
    as_table = MetadataTemplate(StateDictTable.from_metadata(metadata)).with_fields(version=version)
    assert_identical(as_table.to_metadata(), expected)
    as_metadata = MetadataTemplate(dataclasses.replace(metadata, storage_data=None))
    back = as_metadata.with_fields(version=version).to_metadata()
    assert back == dataclasses.replace(metadata, version=version, storage_data=None)
    buf = io.BytesIO()
    as_table.dump(buf, storage_data=metadata.storage_data)
    expected.storage_data = stock_round_trip(metadata).storage_data
    assert_identical(pickle.loads(buf.getvalue()), expected)


def _chunks(chunks):
    return [(tuple(offsets), tuple(sizes)) for offsets, sizes in chunks]


@hypothesis_settings
@given(metadatas(), st.data())
def test_the_reuse_views_agree(metadata, data):
    """The reuse check reads the same entries from a Metadata and from its table."""
    objects = _metadata_reuse.ObjectEntries(metadata)
    tables = _metadata_reuse.TableEntries(StateDictTable.from_metadata(metadata))
    assert len(objects) == len(tables) and objects.counts() == tables.counts()
    assert objects.planner_data == tables.planner_data
    for i, count in enumerate(objects.counts()):
        assert objects.fqn(i) == tables.fqn(i) and objects.is_tensor(i) == tables.is_tensor(i)
        if objects.is_tensor(i):
            lo = data.draw(st.integers(0, count))
            hi = data.draw(st.integers(lo, count))
            assert _chunks(objects.chunks(i, lo, hi)) == _chunks(tables.chunks(i, lo, hi))
    missing = data.draw(texts.filter(lambda fqn: fqn not in metadata.state_dict_metadata))
    for fqn in [*metadata.state_dict_metadata, missing]:
        expected, got = objects.tensor(fqn), tables.tensor(fqn)
        assert (
            (expected == got)
            if isinstance(expected, str)
            else (
                not isinstance(got, str)
                and tuple(expected[0]) == tuple(got[0])
                and expected[1] == got[1]
            )
        )


@NEEDS_NATIVE
@hypothesis_settings
@given(metadatas(), st.data())
def test_both_writers_treat_a_corrupted_table_alike(metadata, data):
    """Change one value of one array of a valid table: either a StateDictTable refuses to be it
    and the C++ writer refuses to read it, or both writers write the same bytes."""
    state_dict_table = StateDictTable.from_metadata(metadata)
    name = data.draw(st.sampled_from(state_table._ARRAYS))
    array = getattr(state_dict_table, name).copy()
    assume(len(array) > 0)
    i = data.draw(st.integers(0, len(array) - 1))
    if array.dtype == np.uint8:
        array[i] = data.draw(st.integers(0, 255))
    else:
        array[i] = data.draw(st.integers(-3, 20) | int64s)
    attributes = _attributes(state_dict_table, **{name: array})
    try:
        python = StateDictTable(**attributes)
    except ValueError:
        python = None
    try:
        native = writer.native.dumps_table(types.SimpleNamespace(**attributes))
    except ValueError:
        native = None
    assert (python is None) == (native is None)
    if python is not None:
        assert pickler.dumps_table(python) == native
