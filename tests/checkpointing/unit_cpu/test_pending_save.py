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

"""Unit tests of PendingSave, what save_state_dict_async_plan returns (CPU only)."""

import pytest

from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import PendingSave


def test_unpacking_is_deprecated():
    """Callers that unpack the PendingSave into finalize's old arguments still get them, with a warning.

    TODO: remove with the deprecated PendingSave.__iter__.
    """
    pending = PendingSave("writer", "metadata", "wrapper")
    with pytest.warns(DeprecationWarning, match="unpacking a PendingSave"):
        assert tuple(pending) == ("writer", "metadata", "wrapper")
    assert (pending.storage_writer, pending.metadata_template, pending.dist_wrapper) == (
        "writer",
        "metadata",
        "wrapper",
    )
