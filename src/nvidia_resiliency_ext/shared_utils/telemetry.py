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

"""The seam between NVRx and nemo-lens, and nothing else.

nemo-lens is an optional dependency (the `otel` extra). This module is the only place
in NVRx that names it. Everything that decides *what* to instrument lives with the
subsystem doing the instrumenting:

    fault_tolerance/telemetry.py             restart-cycle spans for the ft_launcher agent
    checkpointing/async_ckpt/telemetry.py    spans for the persistent checkpoint worker

Callers check `HAS_NEMO_LENS` once, when they build their telemetry object, and hold a
disabled object afterwards. Deliberately no no-op stubs for the missing case: a stub
that pretends to work has to fake a tracer, which needs opentelemetry, which may also
be absent - so it would raise from inside the code path meant to be the safe one.
"""

from contextlib import nullcontext

try:
    from nemo.lens import NemoLensConfig, setup_telemetry  # noqa: F401

    # nemo.lens.helpers holds the real, state-driven implementation. nemo.lens.fallbacks
    # (a different module, despite the similar name) is a set of permanently-hardcoded
    # no-ops meant only for consumers that never import nemo.lens at all; importing from
    # there would silently no-op every span regardless of config.
    from nemo.lens.helpers import managed_span  # noqa: F401

    HAS_NEMO_LENS = True

except ImportError:
    NemoLensConfig = None
    setup_telemetry = None
    managed_span = None
    HAS_NEMO_LENS = False


def span(group: str, name: str, **attributes):
    """`managed_span` when nemo-lens is present, a no-op context manager when it is not.

    Lets an instrumented hot loop stay free of conditionals without the caller having to
    keep its own null context around.
    """
    if not HAS_NEMO_LENS:
        return nullcontext()
    return managed_span(group, name, **attributes)


def config_from_env(prefix: str = 'NEMO_LENS'):
    """Build a NemoLensConfig from the environment, or None when nemo-lens is absent."""
    if not HAS_NEMO_LENS:
        return None
    return NemoLensConfig.from_env(prefix=prefix)
