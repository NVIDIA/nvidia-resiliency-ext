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

"""Hypothesis strategies of DCP metadata, shared by the .metadata tests."""


import numpy as np
import torch
from hypothesis import HealthCheck, settings
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

from .metadata_samples import HAS_TRANSFORMS, SubclassedTensor

# Strategies for Metadata the writers encode themselves. Integers stay within int64, which the
# native writer is limited to; fqns and paths have no lone surrogates, which it cannot encode.
# Both cases are tested as fallbacks in test_metadata_pickler.py.
int64s = st.integers(min_value=-(2**63), max_value=2**63 - 1)


dims_values = st.integers(min_value=0, max_value=2**63 - 1)


sizes = st.lists(dims_values, max_size=6).map(torch.Size)


# Dims of state_dict_metadata entries, also of the kinds a torch.Size keeps but the table doesn't
# (numpy ints, ints beyond int64): such entries are kept as pickles.
entry_dims_values = (
    dims_values | dims_values.map(np.int64) | st.integers(min_value=2**63, max_value=2**70)
)


entry_sizes = st.lists(entry_dims_values, max_size=6).map(torch.Size)


texts = st.text(max_size=300)


properties = st.builds(
    TensorProperties,
    dtype=st.sampled_from([torch.float32, torch.bfloat16, torch.int64, torch.uint8]),
    requires_grad=st.sampled_from([True, False, 1, 0]),  # equal values, different pickles
)


chunks = st.builds(ChunkStorageMetadata, offsets=entry_sizes, sizes=entry_sizes)


@st.composite
def consistent_tensors(draw):
    """Draw a tensor entry whose size and chunks all have the same number of dims, as in a
    checkpoint.
    """
    ndim = draw(st.integers(0, 4))
    dims = st.lists(entry_dims_values, min_size=ndim, max_size=ndim).map(torch.Size)
    chunk = st.builds(ChunkStorageMetadata, offsets=dims, sizes=dims)
    return TensorStorageMetadata(draw(properties), draw(dims), draw(st.lists(chunk, max_size=5)))


@st.composite
def metadatas(draw):
    # Draw fqns and paths from small pools, so that strings repeat as in real checkpoints.
    fqns = draw(st.lists(texts, min_size=1, max_size=20, unique=True))
    paths = draw(st.lists(texts, min_size=1, max_size=5, unique=True))
    tensor = st.builds(
        TensorStorageMetadata,
        properties=properties,
        size=entry_sizes,
        chunks=st.lists(chunks, max_size=5),
    )
    # Unusual entries, which the writers leave to the stdlib pickler: an object of a subclass, a
    # tuple of chunks. (Chunks whose number of dims differs from the size's come from tensor.)
    unusual = st.one_of(
        st.builds(
            SubclassedTensor,
            properties=properties,
            size=entry_sizes,
            chunks=st.lists(chunks, max_size=3),
        ),
        st.builds(
            TensorStorageMetadata,
            properties=properties,
            size=entry_sizes,
            chunks=st.lists(chunks, max_size=3).map(tuple),
        ),
    )
    entry = st.one_of(consistent_tensors(), tensor, st.builds(BytesStorageMetadata), unusual)
    state_dict_metadata = draw(st.dictionaries(st.sampled_from(fqns), entry))
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


# Examples are random on purpose, to cover more inputs over time; failures found locally are
# replayed from the example database in .hypothesis/.
# TODO: persist the example database across CI runs.
hypothesis_settings = settings(
    max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
