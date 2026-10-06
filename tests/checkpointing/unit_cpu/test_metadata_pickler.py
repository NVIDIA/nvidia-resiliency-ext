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

"""Unit tests of the fast .metadata writers (CPU only).

The writers must produce a pickle that unpickles to exactly what stock pickle round-trips to:
same values and same types. Writing through FileSystemWriterAsync.finish is tested in
tests/checkpointing/unit/test_async_writer.py, on GPUs.
"""

import io
import pickle
from collections import OrderedDict
from dataclasses import fields

import pytest
import torch
import torch.distributed.checkpoint as dcp
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)

from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler import writer

HAS_TRANSFORMS = "transform_descriptors" in {f.name for f in fields(_StorageInfo)}  # torch 2.8+
NEEDS_NATIVE = pytest.mark.skipif(writer.native is None, reason="native extension not built")
WRITERS = [
    pytest.param(writer._python_dumps, id="python"),
    pytest.param(writer._native_dumps, id="native", marks=NEEDS_NATIVE),
]


@pytest.fixture(autouse=True)
def fresh_selection(monkeypatch):
    """Each test selects the writer anew, from the default environment."""
    monkeypatch.delenv("NVRX_FAST_METADATA_PICKLE", raising=False)
    writer.fast_metadata_enabled.cache_clear()
    writer._select_dumps.cache_clear()
    yield
    writer.fast_metadata_enabled.cache_clear()
    writer._select_dumps.cache_clear()


def assert_identical(a, b, path="metadata"):
    """a and b are equal, have the same types throughout, and their dicts the same order."""
    assert type(a) is type(b), f"{path}: {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        assert len(a) == len(b), f"{path}: {len(a)} != {len(b)} items"
        for i, ((ka, va), (kb, vb)) in enumerate(zip(a.items(), b.items())):
            assert_identical(ka, kb, f"{path}.key[{i}]")
            assert_identical(va, vb, f"{path}[{ka!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: {len(a)} != {len(b)} items"
        for i, (xa, xb) in enumerate(zip(a, b)):
            assert_identical(xa, xb, f"{path}[{i}]")
    elif hasattr(a, "__dict__"):
        assert_identical(vars(a), vars(b), f"{path}.__dict__")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def stock_round_trip(md: Metadata) -> Metadata:
    return pickle.loads(pickle.dumps(md))


def assert_written_like_stock(dumps, md: Metadata) -> None:
    """dumps(md) unpickles exactly as stock pickle round-trips md."""
    assert_identical(pickle.loads(dumps(md)), stock_round_trip(md))


# Strategies for Metadata the writers encode themselves. Integers stay within int64, which the
# native writer is limited to; fqns and paths have no lone surrogates, which it cannot encode.
# Both cases are tested as fallbacks below.
int64s = st.integers(min_value=-(2**63), max_value=2**63 - 1)
dims = st.integers(min_value=0, max_value=2**63 - 1)
sizes = st.lists(dims, max_size=6).map(torch.Size)
texts = st.text(max_size=300)
properties = st.builds(
    TensorProperties,
    dtype=st.sampled_from([torch.float32, torch.bfloat16, torch.int64, torch.uint8]),
    requires_grad=st.booleans(),
)
chunks = st.builds(ChunkStorageMetadata, offsets=sizes, sizes=sizes)


@st.composite
def metadatas(draw):
    # Draw fqns and paths from small pools, so that strings repeat as in real checkpoints.
    fqns = draw(st.lists(texts, min_size=1, max_size=20, unique=True))
    paths = draw(st.lists(texts, min_size=1, max_size=5, unique=True))
    tensor = st.builds(
        TensorStorageMetadata,
        properties=properties,
        size=sizes,
        chunks=st.lists(chunks, max_size=5),
    )
    state_dict_metadata = draw(
        st.dictionaries(st.sampled_from(fqns), st.one_of(tensor, st.builds(BytesStorageMetadata)))
    )
    index = st.builds(
        MetadataIndex,
        fqn=st.sampled_from(fqns),
        offset=st.none() | sizes,
        index=st.none() | int64s,
    )
    transforms = st.none()
    if HAS_TRANSFORMS:
        transforms = st.none() | st.lists(texts, min_size=1, max_size=2)

    def storage_info(path, offset, length, transform_descriptors):
        if transform_descriptors is None:
            return _StorageInfo(path, offset, length)
        return _StorageInfo(path, offset, length, transform_descriptors=transform_descriptors)

    info = st.builds(storage_info, st.sampled_from(paths), int64s, int64s, transforms)
    storage_data = draw(st.dictionaries(index, info, max_size=30))
    planner_data = draw(st.none() | st.dictionaries(texts, st.tuples(texts), max_size=3))
    return Metadata(
        state_dict_metadata=state_dict_metadata,
        planner_data=planner_data,
        storage_data=storage_data,
    )


def large_metadata() -> Metadata:
    """Crosses the writers' size boundaries: batches of 1000 items, memo indices past 255,
    strings longer than 255 bytes, and integers of every width."""
    props = TensorProperties(dtype=torch.bfloat16)
    many_chunks = [
        ChunkStorageMetadata(torch.Size([i, 0]), torch.Size([1, 2**40])) for i in range(2500)
    ]
    state_dict_metadata = {
        "w" * 300: TensorStorageMetadata(props, torch.Size([2500, 2**40]), many_chunks),
        **{f"layers.{i}.ünïcode.bias": BytesStorageMetadata() for i in range(1500)},
    }
    lengths = [0, 255, 256, 65535, 65536, 2**31 - 1, 2**31, 2**63 - 1, -1, -(2**31), -(2**63)]
    storage_data = {
        MetadataIndex("w" * 300, torch.Size([i, 0]), i): _StorageInfo(
            f"__{i % 7}_0.distcp", i * 2**33, lengths[i % len(lengths)]
        )
        for i in range(2500)
    }
    return Metadata(state_dict_metadata=state_dict_metadata, storage_data=storage_data)


def dcp_saved_metadata(tmp_path) -> Metadata:
    """Metadata as torch writes it for a state dict of tensors and objects."""
    state_dict = {
        "scalar": torch.tensor(3.0),
        "weight": torch.randn(4, 5, dtype=torch.bfloat16),
        "ids": torch.arange(10),
        "stats": {"step": 7, "name": "run-1"},
    }
    dcp.save(state_dict, storage_writer=dcp.FileSystemWriter(tmp_path))  # no process group
    md = dcp.FileSystemReader(tmp_path).read_metadata()
    assert md.storage_data, "expected storage entries"
    return md


def subclassed_metadata() -> Metadata:
    """Objects of subclasses of the metadata classes, which must keep their class."""

    md = writer._sample_metadata()
    w = md.state_dict_metadata["w"]
    chunks = [_Chunk(torch.Size([0, 0, 0, 0]), torch.Size([1, 1, 1, 1])), *w.chunks]
    md.state_dict_metadata["w2"] = _Tensor(w.properties, w.size, chunks)
    md.storage_data[_Index("w2", torch.Size([0, 0, 0, 0]), 0)] = _Info("__9_0.distcp", 5, 6)
    return md


class _Tensor(TensorStorageMetadata):
    pass


class _Chunk(ChunkStorageMetadata):
    pass


class _Index(MetadataIndex):
    pass


class _Info(_StorageInfo):
    pass


FIXED_METADATA = [
    pytest.param(lambda tmp_path: writer._sample_metadata(), id="sample"),
    pytest.param(lambda tmp_path: large_metadata(), id="large"),
    pytest.param(dcp_saved_metadata, id="dcp-save"),
    pytest.param(lambda tmp_path: subclassed_metadata(), id="subclasses"),
]

# Examples are random on purpose, to cover more inputs over time; failures found locally are
# replayed from the example database in .hypothesis/.
# TODO: persist the example database across CI runs.
hypothesis_settings = settings(
    max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


@pytest.mark.parametrize("dumps", WRITERS)
@hypothesis_settings
@given(metadatas())
def test_writer_matches_stock_pickle(dumps, md):
    assert_written_like_stock(dumps, md)


@NEEDS_NATIVE
@hypothesis_settings
@given(metadatas())
def test_native_matches_python(md):
    assert writer._native_dumps(md) == writer._python_dumps(md)


@pytest.mark.parametrize("dumps", WRITERS)
@pytest.mark.parametrize("make", FIXED_METADATA)
def test_writer_matches_stock_pickle_on(dumps, make, tmp_path):
    assert_written_like_stock(dumps, make(tmp_path))


@NEEDS_NATIVE
@pytest.mark.parametrize("make", FIXED_METADATA)
def test_native_matches_python_on(make, tmp_path):
    md = make(tmp_path)
    assert writer._native_dumps(md) == writer._python_dumps(md)


def _set_index(md, value):
    object.__setattr__(next(iter(md.storage_data)), "index", value)  # MetadataIndex is frozen


class _Str(str):
    pass


def _rename_first(md, rename):
    fqn = next(iter(md.state_dict_metadata))
    md.state_dict_metadata[rename(fqn)] = md.state_dict_metadata.pop(fqn)


# (mutation, whether the Python writer rejects it too). The native writer rejects them all.
UNEXPECTED_VALUES = pytest.mark.parametrize(
    ("mutate", "python_rejects"),
    [
        pytest.param(
            lambda md: setattr(
                md.state_dict_metadata["w"], "chunks", tuple(md.state_dict_metadata["w"].chunks)
            ),
            True,
            id="tuple-chunks",
        ),
        pytest.param(
            lambda md: setattr(md.state_dict_metadata["w"].chunks[0], "offsets", (0, 0, 0, 0)),
            True,
            id="tuple-size",
        ),
        pytest.param(lambda md: _set_index(md, True), True, id="bool-int"),
        pytest.param(lambda md: _rename_first(md, _Str), True, id="str-subclass"),
        pytest.param(
            lambda md: setattr(md, "storage_data", OrderedDict(md.storage_data)),
            True,
            id="ordered-dict",
        ),
        pytest.param(
            lambda md: object.__setattr__(next(iter(md.storage_data)), "extra", 1),
            True,
            id="metadata-index-extra-attribute",
        ),
        pytest.param(
            lambda md: object.__setattr__(md.state_dict_metadata["w"], "extra", 1),
            True,
            id="tensor-extra-attribute",
        ),
        pytest.param(
            lambda md: object.__setattr__(md.state_dict_metadata["w"].chunks[0], "extra", 1),
            True,
            id="chunk-extra-attribute",
        ),
        pytest.param(
            lambda md: setattr(next(iter(md.storage_data.values())), "extra", 1),
            True,
            id="storage-info-extra-attribute",
        ),
        pytest.param(lambda md: _set_index(md, 2**70), False, id="int-beyond-int64"),
        pytest.param(
            lambda md: _rename_first(md, lambda s: s + "\ud800"), False, id="lone-surrogate"
        ),
    ],
)


@pytest.mark.parametrize("dumps", WRITERS)
@UNEXPECTED_VALUES
def test_writer_rejects_unexpected_values(dumps, mutate, python_rejects):
    """A writer rejects a value it does not encode exactly, rather than write something wrong."""
    md = writer._sample_metadata()
    mutate(md)
    if dumps is writer._native_dumps or python_rejects:
        with pytest.raises((TypeError, OverflowError, UnicodeError)):
            dumps(md)
    else:
        assert_written_like_stock(dumps, md)


@UNEXPECTED_VALUES
def test_dump_metadata_falls_back(mutate, python_rejects):
    """dump_metadata writes a correct pickle when the selected writer rejects a value."""
    md = writer._sample_metadata()
    mutate(md)
    buf = io.BytesIO()
    writer.dump_metadata(md, buf)
    assert_identical(pickle.loads(buf.getvalue()), stock_round_trip(md))


def test_changed_pickled_state_disables_fast_writers(monkeypatch):
    monkeypatch.setattr(
        ChunkStorageMetadata, "__getstate__", lambda self: {"offsets": self.offsets}, raising=False
    )
    assert not writer._layout_supported()
    assert writer._select_dumps() is None
    md = writer._sample_metadata()
    buf = io.BytesIO()
    writer.dump_metadata(md, buf)
    assert buf.getvalue() == pickle.dumps(md)


def test_moved_class_disables_fast_writers(monkeypatch):
    paths = dict(writer._CLASS_PATHS)
    paths[MetadataIndex] = ("torch.distributed.checkpoint.metadata", "MovedMetadataIndex")
    monkeypatch.setattr(writer, "_CLASS_PATHS", paths)
    assert not writer._layout_supported()


def test_layout_supported_on_this_torch():
    assert writer._layout_supported()


_FIRST, _LAST = writer.TESTED_TORCH_VERSIONS
_TESTED = f"{_LAST[0]}.{_LAST[1]}.0"  # whatever torch runs the tests
_FAST = writer._native_dumps if writer.native is not None else writer._python_dumps


@pytest.mark.parametrize(
    ("mode", "version", "expected"),
    [
        (None, _TESTED, _FAST),
        ("1", _TESTED, _FAST),
        ("python", _TESTED, writer._python_dumps),
        ("0", _TESTED, None),
        ("off", _TESTED, None),
        (None, f"{_FIRST[0]}.{_FIRST[1] - 1}.1", None),
        (None, f"{_LAST[0]}.{_LAST[1] + 1}.0", None),
        ("force", f"{_LAST[0]}.{_LAST[1] + 1}.0", _FAST),
        (None, f"{_LAST[0]}.{_LAST[1]}.0a0+git1234567", _FAST),
        (None, f"{_FIRST[0]}.{_FIRST[1]}.0+cu121", _FAST),
    ],
)
def test_writer_selection(monkeypatch, mode, version, expected):
    if mode is not None:
        monkeypatch.setenv("NVRX_FAST_METADATA_PICKLE", mode)
    monkeypatch.setattr(torch, "__version__", version)
    assert writer.fast_metadata_enabled() is (expected is not None)
    assert writer._select_dumps() is expected


def test_python_writer_without_native(monkeypatch):
    monkeypatch.setattr(torch, "__version__", _TESTED)
    monkeypatch.setattr(writer, "native", None)
    assert writer._select_dumps() is writer._python_dumps
