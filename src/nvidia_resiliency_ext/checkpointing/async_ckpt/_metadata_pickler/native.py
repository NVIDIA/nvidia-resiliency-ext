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

from . import _native, pickler


def dumps(md: Metadata, storage_rows: Optional[np.ndarray] = None) -> bytes:
    """The pickle of md, byte for byte as pickler.dumps writes it. With storage_rows (the gathered
    write-result tables, see table.py), storage_data is written from them instead of
    md.storage_data; the extension reads the rows itself."""
    return _native.dumps(md, pickler.small_pickle, storage_rows)
