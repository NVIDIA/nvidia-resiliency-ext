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

"""Assertions and markers shared by the .metadata tests (CPU only)."""

import pickle

import pytest
from torch.distributed.checkpoint.metadata import Metadata

from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler import writer

NEEDS_NATIVE = pytest.mark.skipif(writer.native is None, reason="native extension not built")


_FAST_HERE = writer.fast_metadata_enabled()


NEEDS_FAST = pytest.mark.skipif(not _FAST_HERE, reason="no fast .metadata writer on this torch")


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
