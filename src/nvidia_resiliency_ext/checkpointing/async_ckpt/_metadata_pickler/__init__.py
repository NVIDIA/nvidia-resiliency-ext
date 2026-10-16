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

"""Internal: fast writer for the ``.metadata`` file of torch distributed checkpoints.

See :mod:`.writer`, which picks the C++ writer (:mod:`.native`, wrapping the optional ``_native``
extension built from ``native_src/native.cpp``) or the Python writer (:mod:`.pickler`).
"""

from . import state_dict_table, write_result_table
from .state_dict_table import StateDictTable
from .writer import (
    TESTED_TORCH_VERSIONS,
    dump_metadata,
    fast_metadata_enabled,
    state_dict_table_if_fast,
    writes_tables,
)
