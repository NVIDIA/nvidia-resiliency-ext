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

"""No-op fallbacks for when nemo-lens is not installed.

nemo-lens is an optional dependency (the `otel` extra); most consumers of this
package don't need it. When it is installed the real implementations are used;
when it is not, everything here degrades to a no-op, so callers never have to
check whether telemetry is available before emitting a span.
"""

try:
    from nemo.lens import NemoLensConfig, setup_telemetry  # noqa: F401

    # nemo.lens.helpers holds the real, state-driven implementation.
    # nemo.lens.fallbacks (a different module, despite the similar name) is a set
    # of permanently-hardcoded no-ops meant only for consumers that never import
    # nemo.lens at all; importing from there would silently no-op every span
    # regardless of config.
    from nemo.lens.helpers import managed_span  # noqa: F401

except ImportError:
    from contextlib import contextmanager

    class NemoLensConfig:  # noqa: D101
        enabled = False  # callers read .enabled and return early

        @classmethod
        def from_env(cls, *args, **kwargs):
            """Return a disabled config, so callers can call from_env() unconditionally."""
            return cls()

        def __getattr__(self, name):
            """Any other attribute the caller reads is None."""
            return None

    def setup_telemetry(*args, **kwargs):
        """Return a handle whose .tracer/.meter are no-ops and whose .shutdown() does nothing."""
        return _NoOpTelemetryHandle()

    class _NoOpTelemetryHandle:
        is_exporting = False

        @property
        def tracer(self):
            from opentelemetry import trace

            return trace.get_tracer(__name__)

        @property
        def meter(self):
            from opentelemetry import metrics

            return metrics.get_meter(__name__)

        def shutdown(self, timeout_ms=5000):
            pass

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        """No-op context manager; yields None."""
        yield None
