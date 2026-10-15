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

"""Unit tests of GlobalMetadata, the global metadata the metadata cache keeps (CPU only)."""

import io
import pickle

import pytest

from nvidia_resiliency_ext.checkpointing.async_ckpt import _metadata_reuse
from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler import writer
from nvidia_resiliency_ext.checkpointing.async_ckpt.global_metadata import GlobalMetadata

from .test_metadata_pickler import NEEDS_FAST, assert_identical, stock_round_trip


@pytest.fixture(params=[pytest.param("fast", marks=NEEDS_FAST), "off"])
def fast(request, monkeypatch):
    """With the fast .metadata writers, and with NVRX_FAST_METADATA_PICKLE=0."""
    if request.param == "off":
        monkeypatch.setenv("NVRX_FAST_METADATA_PICKLE", "0")
    return request.param == "fast"


def test_kept_as_a_table_with_the_fast_writers(fast):
    gm = GlobalMetadata.from_metadata(writer._sample_metadata())
    assert (gm.table is not None) is fast
    expected = _metadata_reuse.TableEntries if fast else _metadata_reuse.ObjectEntries
    assert isinstance(gm.entries(), expected)


def test_to_metadata(fast):
    md = writer._sample_metadata()
    gm = GlobalMetadata.from_metadata(md)
    back = gm.to_metadata(md.storage_data)
    assert back.storage_data is md.storage_data
    assert back == md
    assert gm.to_metadata().storage_data is None


@NEEDS_FAST
def test_a_table_is_independent_of_its_metadata():
    md = writer._sample_metadata()
    gm = GlobalMetadata.from_metadata(md)
    before = pickle.dumps(gm.to_metadata())
    md.state_dict_metadata["w"].chunks.clear()
    del md.state_dict_metadata["s"]
    assert pickle.dumps(gm.to_metadata()) == before


def test_with_fields_refuses_unknown_fields(fast):
    gm = GlobalMetadata.from_metadata(writer._sample_metadata())
    for name in ("extra", "storage_data", "state_dict_metadata"):
        with pytest.raises(TypeError):
            gm.with_fields(**{name: None})


def test_with_fields_leaves_the_original_alone(fast):
    md = writer._sample_metadata()
    gm = GlobalMetadata.from_metadata(md)
    before = pickle.dumps(gm.to_metadata())
    changed = gm.with_fields(version="9.9")
    assert changed.to_metadata().version == "9.9"
    assert pickle.dumps(gm.to_metadata()) == before


def test_dump(fast):
    md = writer._sample_metadata()
    buf = io.BytesIO()
    GlobalMetadata.from_metadata(md).dump(buf, storage_data=md.storage_data)
    assert_identical(pickle.loads(buf.getvalue()), stock_round_trip(md))
