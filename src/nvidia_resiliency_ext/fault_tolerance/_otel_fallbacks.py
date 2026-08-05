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

nemo-lens is an optional dependency (the `otel` extra) -- most consumers of
this package don't need it. When it IS installed, the real implementations
are used. When it is NOT, everything here degrades to a no-op so the
persistent checkpoint worker's hot path never has to check "is telemetry
available" itself.
"""

try:
    from nemo.lens import NemoLensConfig, setup_telemetry  # noqa: F401

    # nemo.lens.state / nemo.lens.helpers hold the REAL, state-driven
    # implementations -- nemo.lens.fallbacks (a different module, despite the
    # similar name) is a set of permanently-hardcoded no-ops meant only for
    # consumers that never import nemo.lens at all. Importing from the wrong
    # one here would silently no-op every span regardless of config.
    from nemo.lens.helpers import (  # noqa: F401
        managed_span,
        safe_set_span_attributes,
        span_cm,
        trace_fn,
    )
    from nemo.lens.state import is_span_group_enabled  # noqa: F401

    HAS_NEMO_LENS = True
except ImportError:
    from contextlib import contextmanager

    HAS_NEMO_LENS = False

    class NemoLensConfig:  # noqa: D101
        pass

    def setup_telemetry(*args, **kwargs):
        """No-op -- returns a handle whose .tracer/.meter are no-ops and .shutdown() does nothing."""
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

    def trace_fn(group, name, tracer=None):
        """No-op decorator -- returns the function unchanged."""
        def decorator(func):
            return func
        return decorator

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        """No-op context manager -- yields None."""
        yield None

    def is_span_group_enabled(group):
        """Always returns False when nemo-lens is not installed."""
        return False

    def safe_set_span_attributes(span, attributes, redact_keys=None):
        """No-op."""
        pass

    @contextmanager
    def span_cm(name, tracer=None, record_exception=True, **attributes):
        """No-op context manager -- yields None."""
        yield None
