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

"""Pickle opcodes for one small object, to splice into a stream written opcode by opcode.

The ``.metadata`` writers write the metadata classes' opcodes themselves, and leave everything
else (``TensorProperties``, ``StorageMeta``, unusual entries) to the stdlib pickler through
``small_pickle``. ``small_unpickle`` reads such opcodes back.
"""

import io

# Issue: [B403:blacklist] Consider possible security implications associated with pickle module.
# Severity: Low   Confidence: High
# CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
# More Info: https://bandit.readthedocs.io/en/1.8.3/blacklists/blacklist_imports.html#b403-import-pickle
import pickle  # nosec


def small_pickle(obj) -> bytes:
    """Return the stdlib pickler's opcodes for a small object, to splice into a stream.

    Protocol 3 emits no FRAME opcodes, and fast mode emits no memo PUT/GET, so the body can be
    embedded anywhere without disturbing the writer's memo numbering.
    """
    buf = io.BytesIO()
    p = pickle.Pickler(buf, protocol=3)
    p.fast = True
    p.dump(obj)
    data = buf.getvalue()
    assert data[:2] == b"\x80\x03" and data[-1:] == b"."
    return data[2:-1]


def small_unpickle(opcodes: bytes):
    """Return the object small_pickle wrote these opcodes for."""
    return pickle.loads(b"\x80\x03" + opcodes + b".")  # nosec - our own bytes
