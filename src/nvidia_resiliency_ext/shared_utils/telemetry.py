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

"""Optional nemo-lens OTel instrumentation. The only file in NVRx that imports it.

Span instrumentation is inert when nemo-lens is unavailable or its span group
is off, so callers need no guards.

Design and rationale: docs/design/telemetry/NEMO_LENS.md.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Optional, ParamSpec, TypeVar

if TYPE_CHECKING:
    from contextvars import Token

    from nemo.lens import TelemetryHandle
    from nemo.lens.resources.attributes import ResourceAttributes, ResourceAttributeValue
    from opentelemetry.context import Context
    from opentelemetry.trace import Span, SpanContext, Tracer

_P = ParamSpec("_P")
_R = TypeVar("_R")

logger = logging.getLogger(__name__)

#: Span groups NVRx emits, and the presets selecting them. nvrx.ckpt is one span
#: per checkpoint request per side; nvrx.ckpt.phases breaks each into its stages
#: and is opt-in, being per-stage cardinality.
_NAMESPACE = "nvrx"
_JOB = "nvrx.job"
_FT = "nvrx.ft"
_CKPT = "nvrx.ckpt"
_CKPT_PHASES = "nvrx.ckpt.phases"
_GROUPS = frozenset([_JOB, _FT, _CKPT, _CKPT_PHASES])
_PRESETS = {
    "default": frozenset([_JOB, _FT, _CKPT]),
    "per_step": frozenset([_JOB, _FT, _CKPT, _CKPT_PHASES]),
    "profiling": _GROUPS,
}

# Captured before anything can extend it. Extensions build from this, never from
# the last extension, or a relaunched cohort accumulates a key per restart.
_INHERITED_RESOURCE_ATTRIBUTES = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")

try:
    # Underscored imports exist only when nemo-lens is installed.
    #
    # OpenTelemetry is imported here, not where it is used, so that NVRx depends on
    # it in its own right rather than on nemo-lens continuing to pull it in. A
    # failure here lands in the same place as a missing nemo-lens: _AVAILABLE goes
    # false and every entry point below no-ops.
    from nemo.lens import NemoLensConfig as _NemoLensConfig
    from nemo.lens import SpanRegistry as _SpanRegistry
    from nemo.lens import get_tracer as _get_tracer
    from nemo.lens import is_span_group_enabled as _is_span_group_enabled
    from nemo.lens import managed_span as _managed_span
    from nemo.lens import setup_telemetry as _setup_telemetry
    from nemo.lens import span_attributes as _span_attributes
    from nemo.lens import trace_fn as _trace_fn
    from nemo.lens.resources.attributes import (
        extend_otel_resource_attributes,
        get_otel_resource_attributes,
        publish_otel_resource_attributes,
    )
    from nemo.lens.semconv.encoding import compose_attributes
    from nemo.lens.span_utilities import emit_span as _emit_span
    from nemo.lens.span_utilities import linux_process_create_time as _process_create_time
    from opentelemetry import context as _otel_context
    from opentelemetry import trace as _otel_trace

    _AVAILABLE = True

except ImportError:
    # nemo-lens absent, or a version whose surface moved: either way, nothing to call.
    _AVAILABLE = False

except Exception:
    logger.warning("nemo-lens import failed, continuing without telemetry", exc_info=True)
    _AVAILABLE = False


if _AVAILABLE:
    try:
        # At import, not in setup_telemetry: the trainer emits NVRx checkpoint
        # spans and never calls setup_telemetry, so its groups would be dark.
        _SpanRegistry.register(_NAMESPACE, _GROUPS, _PRESETS)
    except Exception:
        # A name collision costs these groups, not all telemetry.
        logger.warning("Could not register the NVRx span groups", exc_info=True)


if not _AVAILABLE:

    def get_otel_resource_attributes(
        *, environ: Optional[Mapping[str, str]] = None
    ) -> dict[str, str]:
        """Return no attributes when nemo-lens is unavailable."""
        return {}

    def compose_attributes(
        current: ResourceAttributes,
        *,
        defaults: Optional[ResourceAttributes] = None,
        overrides: Optional[ResourceAttributes] = None,
    ) -> dict[str, Optional[ResourceAttributeValue]]:
        """Return an inert map when nemo-lens is unavailable."""
        return {}

    def extend_otel_resource_attributes(
        text: Optional[str],
        *,
        defaults: Optional[ResourceAttributes] = None,
        overrides: Optional[ResourceAttributes] = None,
    ) -> str:
        """Leave a selected carrier unchanged when nemo-lens is unavailable."""
        return text or ""

    @contextmanager
    def publish_otel_resource_attributes(
        attributes: ResourceAttributes, *, environ: Optional[MutableMapping[str, str]] = None
    ) -> Iterator[None]:
        """Leave the environment unchanged when nemo-lens is unavailable."""
        yield


class _NoOpHandle:
    """Stand-in for ``nemo.lens.TelemetryHandle`` when telemetry is unavailable."""

    def shutdown(self, timeout_ms: int = 5000) -> None:
        pass


def setup_telemetry(
    service_name: str,
    instance_id: Optional[str] = None,
    resource_attributes: Optional[dict[str, Any]] = None,
) -> TelemetryHandle | _NoOpHandle:
    """Initialize nemo-lens. Call once, at process start, only in a process NVRx owns.

    ``service_name`` becomes ``service.name``, overriding ``OTEL_SERVICE_NAME``,
    which names the workload rather than these processes. ``instance_id`` becomes
    ``service.instance.id``; omit it when a parent published one through
    :func:`publish_otel_resource_attributes`. One of the two must supply it -- nemo-lens
    derives its own from ``nv.dl.rank``, which no NVRx process has a usable value for.
    """
    global _AVAILABLE
    if not _AVAILABLE:
        return _NoOpHandle()
    try:
        config = _NemoLensConfig.from_env()
        config.service_name = service_name
        attributes = {"service.instance.id": instance_id} if instance_id else {}
        attributes.update(resource_attributes or {})
        return _setup_telemetry(config, resource_attributes=attributes)
    except Exception:
        # Setup can fail after Lens enables groups; later calls must stay inert.
        _AVAILABLE = False
        logger.warning("nemo-lens init failed, continuing without telemetry", exc_info=True)
        return _NoOpHandle()


def shutdown(handle: TelemetryHandle | _NoOpHandle, timeout_s: float = 2.0) -> None:
    """Flush and shut down, bounded.

    ``TelemetryHandle.shutdown()`` can block for the exporter's whole retry budget
    against a collector that is gone, which outlasts a SIGTERM grace period.
    """
    worker = threading.Thread(target=handle.shutdown, daemon=True)
    worker.start()
    worker.join(timeout_s)


def flush(timeout_ms: int = 1500) -> None:
    """Export what is buffered, for a point where this process may be killed next."""
    if not _AVAILABLE:
        return
    provider = _otel_trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush(timeout_millis=timeout_ms)


def get_inherited_resource_attributes() -> str:
    """Return the Resource carrier captured when this module was imported."""
    return _INHERITED_RESOURCE_ATTRIBUTES


def trace_fn(
    group: str,
    name: str,
    tracer: Optional[Tracer] = None,
    attrs: Optional[Callable[..., Optional[dict[str, Any]]]] = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorate a function with an optional gated attribute callback.

    ``attrs`` receives the decorated function's arguments. It is evaluated only
    when the span group is enabled and supplies attributes directly to the new
    span. Existing callers that omit it retain Lens's ``trace_fn`` behavior.
    """

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        if not _AVAILABLE:
            return func
        traced_func: Callable[_P, _R] = (
            _trace_fn(group, name, tracer)(func) if attrs is None else func
        )

        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            if not _AVAILABLE:
                return func(*args, **kwargs)
            if attrs is None:
                return traced_func(*args, **kwargs)
            if not _is_span_group_enabled(group):
                return func(*args, **kwargs)
            attributes = attrs(*args, **kwargs)
            active_tracer = tracer if tracer is not None else _get_tracer("nemo.lens")
            with _managed_span(group, name, active_tracer, **(attributes or {})):
                return func(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def span(
    group: str,
    name: str,
    attributes: Optional[dict[str, Any]] = None,
    *,
    inherit_attributes: bool = False,
) -> Iterator[Optional[Span]]:
    """A lexical span, yielding it or None when telemetry or the group is off.

    With ``inherit_attributes=True``, entry attributes also apply to nested spans
    until this block exits.
    """
    if not _AVAILABLE or not _is_span_group_enabled(group):
        yield None
        return
    attribute_scope = (
        _span_attributes(attributes) if inherit_attributes and attributes else nullcontext()
    )
    with attribute_scope, _managed_span(group, name, **(attributes or {})) as active:
        yield active


def _emit(
    group: str,
    name: str,
    start: float,
    end: float,
    attributes: Optional[dict[str, Any]] = None,
    context: Optional[Context] = None,
) -> Optional[SpanContext]:
    """Emit one span over an explicit window. Returns its ``SpanContext``, or None.

    ``context`` of None inherits the ambient span, the way an ordinary span does;
    an empty ``Context()`` roots a new trace instead.
    """
    if not _AVAILABLE or not _is_span_group_enabled(group):
        return None
    if end < start:
        logger.warning("Skipping telemetry span %s: end %s precedes start %s", name, end, start)
        return None
    recorded = _emit_span(
        _get_tracer(__name__),
        name,
        start,
        end,
        group=group,
        context=context,
        attributes=attributes,
    )
    return recorded.get_span_context() if recorded is not None else None


def backdated_span(
    group: str,
    name: str,
    start: Optional[float],
    end: Optional[float],
    attributes: Optional[dict[str, Any]] = None,
    parent: Optional[SpanContext] = None,
) -> Optional[SpanContext]:
    """Record a span for a window that elapsed before there was a tracer.

    ``start`` and ``end`` are wall-clock seconds; ``parent`` is usually the
    ``SpanContext`` of the ``mark`` that opened the window, and without one the span
    starts a new trace. Lens validates timestamps, checks whether the group is
    enabled, and ends the span. Zero duration is valid. Return the recorded span's
    context, or None if no span was recorded.
    """
    if start is None or end is None:
        return None
    if not _AVAILABLE or not _is_span_group_enabled(group):
        return None
    # Empty, not the ambient context: this window closed before the call, so the
    # span that happens to be open now is not its parent.
    context = _otel_context.Context()
    if parent is not None:
        context = _otel_trace.set_span_in_context(_otel_trace.NonRecordingSpan(parent), context)
    return _emit(group, name, start, end, attributes, context)


def mark(
    group: str, name: str, attributes: Optional[dict[str, Any]] = None
) -> Optional[SpanContext]:
    """Record an instant: a zero-duration span pinning a moment in time.

    Returns its ``SpanContext``, or None when the group is off. A mark ends
    immediately; export may be buffered. Its context contains IDs and does not
    keep a span open.
    Inherits the ambient span, so a mark nests where an ordinary span would.
    """
    if not _AVAILABLE or not _is_span_group_enabled(group):
        return None
    now = time.time()
    return _emit(group, name, now, now, attributes)


def record_process_startup(
    group: str,
    imports_started: float,
    imports_finished: float,
    attributes: Optional[dict[str, Any]] = None,
) -> None:
    """Record how long this process took to become able to run.

    Two backdated windows: process creation to the entry module's first statement,
    and that module's top-level imports. Both root their own trace.
    """
    created = None
    if _AVAILABLE:
        try:
            created = _process_create_time()
        except Exception:
            logger.debug("Process create time unavailable", exc_info=True)
    backdated_span(group, "nv.nvrx.ftl.python.startup", created, imports_started, attributes)
    backdated_span(
        group, "nv.nvrx.ftl.python.imports", imports_started, imports_finished, attributes
    )


class Phase:
    """A long window, recorded as a start anchor and a duration summary.

    For a window too long to hold a span open across, since a span exports only
    when it ends. ``open()`` emits ``<name>_start`` as a zero-duration anchor and
    makes its context active, so spans on this thread nest under the phase.
    ``close()`` emits ``<name>`` as a duration summary from the saved start time.
    A phase that never closes still leaves the anchor and its recorded children.

    ``open`` attributes go on both records; ``set`` and ``close`` reach only the
    span and override by name.

    ORDERING CONTRACT, from ``contextvars``: ``open()`` and ``close()`` must run on
    the same thread, and anything opened after this one must close before it does.
    """

    def __init__(self) -> None:
        self._window: Optional[tuple[str, str, float]] = None
        self._parent: Optional[SpanContext] = None
        self._token: Optional[Token[Context]] = None
        self._attributes: dict[str, Any] = {}

    def open(self, group: str, name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        """Emit the start anchor, closing any phase this handle had open."""
        # Without nemo-lens nothing downstream can record anything, so leave _window
        # unset: that is the flag set() and close() already bail on, which makes the
        # whole handle inert for the cost of one module-global read.
        if not _AVAILABLE:
            return
        self.close()
        self._window = (group, name, time.time())
        # Seeded, not emptied: a consumer filtering spans never sees the mark's
        # attributes, so a key left only there cannot be grouped on.
        self._attributes = dict(attributes or {})
        try:
            self._parent = mark(group, f"{name}_start", attributes)
        except BaseException:
            self._window = self._parent = None
            self._attributes = {}
            raise
        if self._parent is None:  # group off; the phase still spans nothing to nest in
            return
        try:
            self._token = _otel_context.attach(
                _otel_trace.set_span_in_context(_otel_trace.NonRecordingSpan(self._parent))
            )
        except Exception:
            # Losing the ambient context costs nesting, not spans.
            logger.debug("Could not make %s the active context", name, exc_info=True)

    def set(self, attributes: Optional[dict[str, Any]] = None) -> None:
        """Update attributes saved for the duration summary."""
        if self._window is None:
            return
        if attributes:
            self._attributes.update(attributes)

    def close(self, attributes: Optional[dict[str, Any]] = None) -> None:
        """Emit the backdated span covering the phase. Idempotent."""
        self.set(attributes)
        if self._window is None:
            return
        group, name, start = self._window
        if self._token is not None:
            try:
                _otel_context.detach(self._token)
            except Exception:
                # A phase opened after this one outlived it, so the token is not the
                # top of the stack. The span is still correct; the next open() fixes
                # the stale ambient context.
                logger.debug("Out-of-order close for phase %s", name, exc_info=True)
            self._token = None
        try:
            backdated_span(
                group,
                name,
                start,
                time.time(),
                self._attributes,
                parent=self._parent,
            )
        finally:
            self._window = self._parent = None
            self._attributes = {}
