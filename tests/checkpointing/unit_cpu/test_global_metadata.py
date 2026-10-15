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
    global_metadata = GlobalMetadata.from_metadata(writer._sample_metadata())
    assert (global_metadata.table is not None) is fast
    expected = _metadata_reuse.TableEntries if fast else _metadata_reuse.ObjectEntries
    assert isinstance(global_metadata.entries(), expected)


def test_to_metadata(fast):
    metadata = writer._sample_metadata()
    global_metadata = GlobalMetadata.from_metadata(metadata)
    back = global_metadata.to_metadata(metadata.storage_data)
    assert back.storage_data is metadata.storage_data
    assert back == metadata
    assert global_metadata.to_metadata().storage_data is None


@NEEDS_FAST
def test_a_table_is_independent_of_its_metadata():
    metadata = writer._sample_metadata()
    global_metadata = GlobalMetadata.from_metadata(metadata)
    before = pickle.dumps(global_metadata.to_metadata())
    metadata.state_dict_metadata["w"].chunks.clear()
    del metadata.state_dict_metadata["s"]
    assert pickle.dumps(global_metadata.to_metadata()) == before


def test_with_fields_refuses_unknown_fields(fast):
    global_metadata = GlobalMetadata.from_metadata(writer._sample_metadata())
    for name in ("extra", "storage_data", "state_dict_metadata"):
        with pytest.raises(TypeError):
            global_metadata.with_fields(**{name: None})


def test_with_fields_leaves_the_original_alone(fast):
    metadata = writer._sample_metadata()
    global_metadata = GlobalMetadata.from_metadata(metadata)
    before = pickle.dumps(global_metadata.to_metadata())
    changed = global_metadata.with_fields(version="9.9")
    assert changed.to_metadata().version == "9.9"
    assert pickle.dumps(global_metadata.to_metadata()) == before


def test_dump(fast):
    metadata = writer._sample_metadata()
    buf = io.BytesIO()
    GlobalMetadata.from_metadata(metadata).dump(buf, storage_data=metadata.storage_data)
    assert_identical(pickle.loads(buf.getvalue()), stock_round_trip(metadata))
