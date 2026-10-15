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

"""The C++ writer of the ``.metadata`` pickle, with the same interface as pickler.py.

Importing it raises ImportError if the ``_native`` extension (native_src/native.cpp) isn't built.
"""

from typing import Optional

import numpy as np
from torch.distributed.checkpoint.metadata import Metadata

from . import _native
from .opcodes import small_pickle
from .state_table import StateDictTable


def from_metadata(md: Metadata) -> StateDictTable:
    """md without its storage_data as a table, exactly as state_table.from_metadata converts it."""
    return _native.from_metadata(md, small_pickle, StateDictTable)


def dumps_table(
    t: StateDictTable,
    storage_data: Optional[dict] = None,
    storage_rows: Optional[np.ndarray] = None,
) -> bytes:
    """The pickle of the Metadata t was converted from, byte for byte as pickler.dumps_table writes
    it; the extension reads the storage rows itself."""
    return _native.dumps_table(t, small_pickle, storage_data, storage_rows)


def dumps(md: Metadata, storage_rows: Optional[np.ndarray] = None) -> bytes:
    """The pickle of md, byte for byte as pickler.dumps writes it. With storage_rows (the gathered
    write-result tables, see table.py), storage_data is written from them instead of
    md.storage_data."""
    return dumps_table(from_metadata(md), md.storage_data, storage_rows)
