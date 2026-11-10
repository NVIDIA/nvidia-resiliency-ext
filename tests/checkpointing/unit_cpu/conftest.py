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

import pytest

from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler import writer


@pytest.fixture(autouse=True)
def fresh_selection(monkeypatch):
    """Each test selects the .metadata writer anew, from the default environment."""
    monkeypatch.delenv("NVRX_FAST_METADATA_PICKLE", raising=False)
    writer.fast_metadata_enabled.cache_clear()
    writer._select_dumps.cache_clear()
    yield
    writer.fast_metadata_enabled.cache_clear()
    writer._select_dumps.cache_clear()
