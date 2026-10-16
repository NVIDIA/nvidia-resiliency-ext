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

"""Unit tests of MetadataTemplate, the metadata the metadata cache keeps (CPU only)."""

import io
import pickle

import pytest

from nvidia_resiliency_ext.checkpointing.async_ckpt import _metadata_reuse
from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler import writer
from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_template import MetadataTemplate

from .metadata_assertions import NEEDS_FAST, assert_identical, stock_round_trip
from .metadata_samples import SETTABLE_FIELD


@pytest.fixture(params=[pytest.param("fast", marks=NEEDS_FAST), "off"])
def fast(request, monkeypatch):
    """Run with the fast .metadata writers, and with NVRX_FAST_METADATA_PICKLE=0."""
    if request.param == "off":
        monkeypatch.setenv("NVRX_FAST_METADATA_PICKLE", "0")
    return request.param == "fast"


def test_kept_as_a_table_with_the_fast_writers(fast):
    metadata_template = MetadataTemplate.from_metadata(writer._sample_metadata())
    assert (metadata_template.table is not None) is fast
    expected = _metadata_reuse.TableEntries if fast else _metadata_reuse.ObjectEntries
    assert isinstance(metadata_template.entries(), expected)


def test_to_metadata(fast):
    metadata = writer._sample_metadata()
    metadata_template = MetadataTemplate.from_metadata(metadata)
    back = metadata_template.to_metadata(metadata.storage_data)
    assert back.storage_data is metadata.storage_data
    assert back == metadata
    assert metadata_template.to_metadata().storage_data is None


@NEEDS_FAST
def test_a_table_is_independent_of_its_metadata():
    metadata = writer._sample_metadata()
    metadata_template = MetadataTemplate.from_metadata(metadata)
    before = pickle.dumps(metadata_template.to_metadata())
    metadata.state_dict_metadata["w"].chunks.clear()
    del metadata.state_dict_metadata["s"]
    assert pickle.dumps(metadata_template.to_metadata()) == before


def test_with_fields_refuses_unknown_fields(fast):
    metadata_template = MetadataTemplate.from_metadata(writer._sample_metadata())
    for name in ("extra", "storage_data", "state_dict_metadata"):
        with pytest.raises(TypeError):
            metadata_template.with_fields(**{name: None})


def test_with_fields_leaves_the_original_alone(fast):
    metadata = writer._sample_metadata()
    metadata_template = MetadataTemplate.from_metadata(metadata)
    before = pickle.dumps(metadata_template.to_metadata())
    changed = metadata_template.with_fields(**{SETTABLE_FIELD: "9.9"})
    assert getattr(changed.to_metadata(), SETTABLE_FIELD) == "9.9"
    assert pickle.dumps(metadata_template.to_metadata()) == before


def test_dump(fast):
    metadata = writer._sample_metadata()
    buf = io.BytesIO()
    MetadataTemplate.from_metadata(metadata).dump(buf, storage_data=metadata.storage_data)
    assert_identical(pickle.loads(buf.getvalue()), stock_round_trip(metadata))
