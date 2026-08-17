# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Optional nemo-lens OTel instrumentation.

The only file in NVRx that imports nemo-lens. When nemo-lens is absent every
export here is a no-op, so callers never need to guard their instrumentation.

nemo-lens gates on span groups that default to empty until ``setup_telemetry``
runs, so ``managed_span`` / ``traced`` are also no-ops before (or without)
initialization -- there is nothing for this module to re-implement.
"""

import logging
import os
from contextlib import ExitStack, contextmanager
from typing import ClassVar, Optional

logger = logging.getLogger(__name__)

_NVRX_GROUPS = frozenset(["nvrx.ft", "nvrx.ckpt"])

try:
    from nemo.lens import NemoLensConfig as _NemoLensConfig
    from nemo.lens import SpanGroup as _SpanGroup
    from nemo.lens import managed_span
    from nemo.lens import safe_set_span_attributes as _safe_set_span_attributes
    from nemo.lens import setup_telemetry as _setup_telemetry
    from nemo.lens import trace_fn as traced

    class _NVRxSpanGroup(_SpanGroup):
        """Teaches nemo-lens about the NVRx groups.

        The base ``SpanGroup.resolve()`` only knows its own groups, so without
        this ``NEMO_LENS_SPAN_GROUPS=nvrx.ft`` raises ValueError and the stock
        presets emit no NVRx spans. Every preset gains the NVRx groups so that
        ``NEMO_LENS_ENABLED=1`` alone is sufficient; ``nvrx`` selects them alone.
        """

        FT = "nvrx.ft"
        CKPT = "nvrx.ckpt"
        ALL_GROUPS = _SpanGroup.ALL_GROUPS | _NVRX_GROUPS
        _PRESETS: ClassVar[dict] = {
            **{name: groups | _NVRX_GROUPS for name, groups in _SpanGroup._PRESETS.items()},
            "nvrx": _NVRX_GROUPS,
        }

    _AVAILABLE = True

except ModuleNotFoundError:
    _AVAILABLE = False

except Exception:
    logger.warning("nemo-lens import failed, continuing without telemetry", exc_info=True)
    _AVAILABLE = False


if not _AVAILABLE:

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        """No-op stand-in for ``nemo.lens.managed_span``."""
        yield None

    def traced(group, name, tracer=None):
        """No-op stand-in for ``nemo.lens.trace_fn``."""

        def decorator(func):
            return func

        return decorator


class _NoOpHandle:
    """Stand-in for ``nemo.lens.TelemetryHandle`` when telemetry is unavailable."""

    def shutdown(self, timeout_ms: int = 5000) -> None:
        pass


def setup_telemetry(rank: int, world_size: int):
    """Initialize nemo-lens for this process. Call once, at process start.

    ``rank`` and ``world_size`` identify this process in the OTel resource
    (``dl.rank``, ``dl.world_size``, ``service.instance.id``), so they must be
    distinct per emitter or backends cannot tell the processes apart.

    NVRx defaults to exporting from every process: each node runs its own
    collector and we want per-node visibility. An explicit
    ``NEMO_LENS_EXPORT_STRATEGY`` still wins, so volume remains tunable.
    """
    if not _AVAILABLE:
        return _NoOpHandle()
    try:
        config = _NemoLensConfig.from_env(span_group_cls=_NVRxSpanGroup)
        if not os.environ.get("NEMO_LENS_EXPORT_STRATEGY"):
            config.export_strategy = "all_ranks"
        return _setup_telemetry(config, rank, world_size)
    except Exception:
        logger.warning("nemo-lens init failed, continuing without telemetry", exc_info=True)
        return _NoOpHandle()


class ManualSpan:
    """A span started and ended explicitly, for a lifetime that is not a block.

    `managed_span` covers anything scoped to a `with`. This is for the rest: a
    span opened in one call and closed in another, where the caller cannot hold
    a context manager across the two. It owns the ExitStack bookkeeping, and
    every method is a no-op while no span is open, so callers need no guards.
    """

    def __init__(self) -> None:
        self._stack: Optional[ExitStack] = None
        self._span = None

    def open(self, group: str, name: str, attributes: Optional[dict] = None) -> None:
        """Start a span, closing any span this handle already had open."""
        self.close()
        self._stack = ExitStack()
        self._span = self._stack.enter_context(managed_span(group, name))
        self.set(attributes)

    def set(self, attributes: Optional[dict] = None) -> None:
        """Set attributes on the open span."""
        if self._span is None or not attributes:
            return
        for key, value in attributes.items():
            self._span.set_attribute(key, value)

    def close(self, attributes: Optional[dict] = None) -> None:
        """Set any final attributes and end the span. Idempotent."""
        self.set(attributes)
        if self._stack is not None:
            self._stack.close()
            self._stack = None
        self._span = None


def set_span_attributes(**attributes) -> None:
    """Set attributes on the currently active span.

    For use inside a ``@traced`` function, which owns its span but does not
    hand it to the caller. A no-op when no span is recording.
    """
    if not _AVAILABLE:
        return
    from opentelemetry import trace

    _safe_set_span_attributes(trace.get_current_span(), attributes)
