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
import logging
import os
import pickle
from collections import OrderedDict

import numpy as np
import pytest
import torch
from hypothesis import given
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, MetadataIndex

from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler import pickler, writer

from .metadata_assertions import (
    NEEDS_NATIVE,
    assert_identical,
    assert_written_like_stock,
    stock_round_trip,
)
from .metadata_samples import (
    CHANGED_WHILE_PICKLED,
    FIXED_METADATA,
    SubclassedMetadata,
    SubclassedStr,
    metadata_changed_while_pickled,
    rename_first,
    set_index,
)
from .metadata_strategies import hypothesis_settings, metadatas

writer.fast_metadata_enabled.cache_clear()
WRITERS = [
    pytest.param(pickler.dumps, id="python"),
    pytest.param(writer.native.dumps if writer.native else None, id="native", marks=NEEDS_NATIVE),
]


@pytest.mark.parametrize("dumps", WRITERS)
@hypothesis_settings
@given(metadatas())
def test_writer_matches_stock_pickle(dumps, md):
    assert_written_like_stock(dumps, md)


@NEEDS_NATIVE
@hypothesis_settings
@given(metadatas())
def test_native_matches_python(md):
    assert writer.native.dumps(md) == pickler.dumps(md)


@pytest.mark.parametrize("dumps", WRITERS)
@pytest.mark.parametrize("make", FIXED_METADATA)
def test_writer_matches_stock_pickle_on(dumps, make, tmp_path):
    assert_written_like_stock(dumps, make(tmp_path))


@NEEDS_NATIVE
@pytest.mark.parametrize("make", FIXED_METADATA)
def test_native_matches_python_on(make, tmp_path):
    md = make(tmp_path)
    assert writer.native.dumps(md) == pickler.dumps(md)


# State-dict entries the table keeps as the stdlib pickler's opcodes (see state_dict_table.py),
# which both writers write exactly as stock pickle does.
KEPT_AS_PICKLES = pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda md: setattr(
                md.state_dict_metadata["w"], "chunks", tuple(md.state_dict_metadata["w"].chunks)
            ),
            id="tuple-chunks",
        ),
        pytest.param(
            lambda md: setattr(md.state_dict_metadata["w"].chunks[0], "offsets", (0, 0, 0, 0)),
            id="tuple-size",
        ),
        pytest.param(
            lambda md: object.__setattr__(md.state_dict_metadata["w"], "extra", 1),
            id="tensor-extra-attribute",
        ),
        pytest.param(
            lambda md: object.__setattr__(md.state_dict_metadata["w"].chunks[0], "extra", 1),
            id="chunk-extra-attribute",
        ),
        pytest.param(
            lambda md: setattr(md.state_dict_metadata["w"], "size", torch.Size([np.int64(4), 2])),
            id="numpy-int-size",
        ),
        pytest.param(
            lambda md: setattr(
                md.state_dict_metadata["w"].chunks[0], "offsets", torch.Size([2**63, 0])
            ),
            id="offset-beyond-int64",
        ),
    ],
)


@pytest.mark.parametrize("dumps", WRITERS)
@KEPT_AS_PICKLES
def test_writer_writes_unusual_entries_like_stock(dumps, mutate):
    md = writer._sample_metadata()
    mutate(md)
    assert_written_like_stock(dumps, md)


@NEEDS_NATIVE
@KEPT_AS_PICKLES
def test_native_matches_python_on_unusual_entries(mutate):
    md = writer._sample_metadata()
    mutate(md)
    assert writer.native.dumps(md) == pickler.dumps(md)


# (mutation, whether the Python writer rejects it too). The native writer rejects them all.
UNEXPECTED_VALUES = pytest.mark.parametrize(
    ("mutate", "python_rejects"),
    [
        pytest.param(lambda md: set_index(md, True), True, id="bool-int"),
        pytest.param(lambda md: rename_first(md, SubclassedStr), True, id="str-subclass"),
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
            lambda md: setattr(next(iter(md.storage_data.values())), "extra", 1),
            True,
            id="storage-info-extra-attribute",
        ),
        pytest.param(
            lambda md: setattr(md, "__class__", SubclassedMetadata), True, id="metadata-subclass"
        ),
        pytest.param(lambda md: set_index(md, 2**70), False, id="int-beyond-int64"),
        pytest.param(
            lambda md: rename_first(md, lambda s: s + "\ud800"), False, id="lone-surrogate"
        ),
    ],
)


@pytest.mark.parametrize("dumps", WRITERS)
@UNEXPECTED_VALUES
def test_writer_rejects_unexpected_values(dumps, mutate, python_rejects):
    """A writer rejects a value it does not encode exactly, rather than write something wrong."""
    md = writer._sample_metadata()
    mutate(md)
    if dumps is not pickler.dumps or python_rejects:
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


@NEEDS_NATIVE
@pytest.mark.parametrize("case", CHANGED_WHILE_PICKLED)
def test_native_survives_metadata_changed_while_pickled(case):
    md = metadata_changed_while_pickled(case)
    try:
        out = writer.native.dumps(md)
    except RuntimeError as e:
        assert "changed size" in str(e)
    else:
        pickle.loads(out)


@pytest.mark.parametrize("case", CHANGED_WHILE_PICKLED)
def test_dump_metadata_raises_or_loads_when_metadata_changes(case):
    md = metadata_changed_while_pickled(case)
    buf = io.BytesIO()
    try:
        writer.dump_metadata(md, buf)
    except RuntimeError:
        return  # as stock pickle does when a dict changes size
    pickle.loads(buf.getvalue())


@pytest.mark.parametrize(
    ("n", "header"),
    [
        (255, b"\x8c\xff"),
        (256, b"X\x00\x01\x00\x00"),
        (2**32 - 1, b"X\xff\xff\xff\xff"),
        (2**32, b"\x8d\x00\x00\x00\x00\x01\x00\x00\x00"),
    ],
)
def test_str_header(n, header):
    """The str opcode for each length boundary, including BINUNICODE8 past 4 GiB."""
    assert pickler._str_header(n) == header
    if n < 2**20:  # small enough to compare with the stdlib pickler
        assert header + b"a" in pickle.dumps("a" * n, protocol=4)


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
    paths = dict(pickler.CLASS_PATHS)
    paths[MetadataIndex] = ("torch.distributed.checkpoint.metadata", "MovedMetadataIndex")
    monkeypatch.setattr(pickler, "CLASS_PATHS", paths)
    assert not writer._layout_supported()


@pytest.mark.skipif(
    os.environ.get("NVRX_REQUIRE_NATIVE_WRITER") != "1", reason="NVRX_REQUIRE_NATIVE_WRITER unset"
)
def test_native_writer_is_built():
    """In CI, where the native writer's tests would otherwise skip silently if it failed to build."""
    assert writer.native is not None


def test_layout_supported_on_this_torch():
    assert writer._layout_supported()


_LAST = max(writer.TESTED_TORCH_VERSIONS)
_TESTED = f"{_LAST[0]}.{_LAST[1]}.0"  # whatever torch runs the tests
_UNTESTED = f"{_LAST[0]}.{_LAST[1] + 1}.0"
_FAST = writer.native.dumps if writer.native is not None else pickler.dumps


@pytest.mark.parametrize(
    ("mode", "version", "expected"),
    [
        (None, _TESTED, _FAST),
        ("1", _TESTED, _FAST),
        ("python", _TESTED, pickler.dumps),
        ("0", _TESTED, None),
        ("off", _TESTED, None),
        (None, "2.2.1", None),
        (None, _UNTESTED, None),
        ("force", _UNTESTED, _FAST),
        ("force", "2.3.1", None),  # in INCOMPATIBLE_TORCH_VERSIONS
        (None, "2.7.0_custom", None),  # not PEP 440: untested
        ("force", "2.7.0_custom", _FAST),
        (None, f"{_LAST[0]}.{_LAST[1]}.0a0+git1234567", _FAST),
        (None, "2.4.0+cu121", _FAST),
    ],
)
def test_writer_selection(monkeypatch, mode, version, expected):
    if mode is not None:
        monkeypatch.setenv("NVRX_FAST_METADATA_PICKLE", mode)
    monkeypatch.setattr(torch, "__version__", version)
    assert writer.fast_metadata_enabled() is (expected is not None)
    assert writer._select_dumps() is expected
    # Tables only on tested versions: the force cases here are all untested.
    assert writer.writes_tables() is (expected is not None and mode != "force")


@pytest.mark.parametrize("mode", [None, "force"])
def test_incompatible_wins_over_tested(monkeypatch, mode):
    """A tested version later found incompatible gets neither the fast writers nor tables."""
    if mode is not None:
        monkeypatch.setenv("NVRX_FAST_METADATA_PICKLE", mode)
    monkeypatch.setattr(torch, "__version__", _TESTED)
    monkeypatch.setattr(writer, "INCOMPATIBLE_TORCH_VERSIONS", frozenset({_LAST}))
    assert not writer._tested_torch()
    assert not writer.fast_metadata_enabled()
    assert not writer.writes_tables()


@pytest.mark.parametrize("version", ["2.3.1", "2.3.0+cu121"])
def test_force_on_incompatible_torch_logs_error(monkeypatch, caplog, version):
    monkeypatch.setenv("NVRX_FAST_METADATA_PICKLE", "force")
    monkeypatch.setattr(torch, "__version__", version)
    with caplog.at_level(logging.ERROR, logger=writer.__name__):
        assert not writer.fast_metadata_enabled()
    assert "known not to work" in caplog.text


def test_python_writer_without_native(monkeypatch):
    monkeypatch.setattr(torch, "__version__", _TESTED)
    monkeypatch.setattr(writer, "native", None)
    assert writer._select_dumps() is pickler.dumps
