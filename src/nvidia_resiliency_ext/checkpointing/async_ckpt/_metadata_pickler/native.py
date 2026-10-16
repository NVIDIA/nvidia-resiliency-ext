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

from . import _native, state_dict_table
from .opcodes import small_pickle


class StateDictTable(state_dict_table.StateDictTable):
    """state_dict_table.StateDictTable, converted from a Metadata by the C++ producer, which gives
    exactly the table the Python producer gives."""

    @classmethod
    def from_metadata(cls, metadata: Metadata) -> "StateDictTable":
        """Convert metadata, without its storage_data, to a table with the C++ producer."""
        return _native.from_metadata(metadata, small_pickle, cls)


def dumps_table(
    table: state_dict_table.StateDictTable,
    storage_data: Optional[dict] = None,
    storage_rows: Optional[np.ndarray] = None,
) -> bytes:
    """Pickle the Metadata that table was converted from, and return the bytes.

    storage_data is written from storage_rows (the gathered write-result tables, see
    write_result_table.py) if given, else from the storage_data dict, else as None. The bytes equal pickler.dumps_table's; the
    extension reads the rows itself.
    """
    return _native.dumps_table(table, small_pickle, storage_data, storage_rows)


def dumps(metadata: Metadata, storage_rows: Optional[np.ndarray] = None) -> bytes:
    """Pickle metadata and return the bytes.

    storage_data is written from storage_rows (the gathered write-result tables, see
    write_result_table.py) if given, else from metadata.storage_data. The bytes equal pickler.dumps's.
    """
    return dumps_table(StateDictTable.from_metadata(metadata), metadata.storage_data, storage_rows)
