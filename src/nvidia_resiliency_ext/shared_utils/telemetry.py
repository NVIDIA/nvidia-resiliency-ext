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

"""Optional Lens instrumentation; all Lens imports stay in this module.

Span calls are inert without Lens or when their group is disabled.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager, nullcontext
from contextvars import Token
from typing import Any, ParamSpec, TypeVar

from nvidia_resiliency_ext.shared_utils import semconv

_P = ParamSpec("_P")
_R = TypeVar("_R")

logger = logging.getLogger(__name__)

#: NVRx span groups and presets; checkpoint phases are opt-in.
_NAMESPACE = "nvrx"
_GROUPS = frozenset(
    [
        semconv.SPAN_GROUP_STARTUP,
        semconv.SPAN_GROUP_FT,
        semconv.SPAN_GROUP_CKPT,
        semconv.SPAN_GROUP_CKPT_PHASES,
    ]
)
_PRESETS = {
    "default": frozenset(
        [semconv.SPAN_GROUP_STARTUP, semconv.SPAN_GROUP_FT, semconv.SPAN_GROUP_CKPT]
    ),
    "per_step": _GROUPS,
    "profiling": _GROUPS,
}

# Capture the initial worker Resource carrier before telemetry setup.
_INHERITED_RESOURCE_ATTRIBUTES = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")

try:
    # Optional dependencies share one import-failure boundary.
    from nemo.lens import NemoLensConfig as _NemoLensConfig
    from nemo.lens import SpanRegistry as _SpanRegistry
    from nemo.lens import TelemetryHandle
    from nemo.lens import get_tracer as _get_tracer
    from nemo.lens import is_span_group_enabled as _is_span_group_enabled
    from nemo.lens import managed_span as _managed_span
    from nemo.lens import setup_telemetry as _setup_telemetry
    from nemo.lens import span_attributes as _span_attributes
    from nemo.lens import trace_fn as _trace_fn
    from nemo.lens.resources.attributes import (
        ResourceAttributes,
        ResourceAttributeValue,
        extend_otel_resource_attributes,
        get_otel_resource_attributes,
        publish_otel_resource_attributes,
    )
    from nemo.lens.semconv.encoding import compose_attributes
    from nemo.lens.span_utilities import emit_span as _emit_span
    from nemo.lens.span_utilities import linux_process_create_time as _process_create_time
    from opentelemetry import context as _otel_context
    from opentelemetry import trace as _otel_trace
    from opentelemetry.context import Context
    from opentelemetry.trace import Span, SpanContext, Tracer

    _AVAILABLE = True

except ImportError:
    # nemo-lens absent, or a version whose surface moved: either way, nothing to call.
    _AVAILABLE = False

except Exception:
    logger.warning("nemo-lens import failed, continuing without telemetry", exc_info=True)
    _AVAILABLE = False


if _AVAILABLE:
    try:
        # Register here so trainers can use a framework-owned provider.
        _SpanRegistry.register(_NAMESPACE, _GROUPS, _PRESETS)
    except Exception:
        # A name collision costs these groups, not all telemetry.
        logger.warning("Could not register the NVRx span groups", exc_info=True)


if not _AVAILABLE:
    # Resource helpers leave the environment unchanged without Lens.

    def get_otel_resource_attributes(*, environ: Mapping[str, str] | None = None) -> dict[str, str]:
        return {}

    def compose_attributes(
        current: ResourceAttributes,
        *,
        defaults: ResourceAttributes | None = None,
        overrides: ResourceAttributes | None = None,
    ) -> dict[str, ResourceAttributeValue | None]:
        return {}

    def extend_otel_resource_attributes(
        text: str | None,
        *,
        defaults: ResourceAttributes | None = None,
        overrides: ResourceAttributes | None = None,
    ) -> str:
        return text or ""

    @contextmanager
    def publish_otel_resource_attributes(
        attributes: ResourceAttributes, *, environ: MutableMapping[str, str] | None = None
    ) -> Iterator[None]:
        yield


class _NoOpHandle:
    """Stand-in for ``nemo.lens.TelemetryHandle`` when telemetry is unavailable."""

    def shutdown(self, timeout_ms: int = 5000) -> None:
        pass


def setup_telemetry(
    service_name: str,
    instance_id: str | None = None,
    resource_attributes: dict[str, Any] | None = None,
) -> TelemetryHandle | _NoOpHandle:
    """Initialize Lens once in an NVRx-owned process.

    ``service_name`` overrides ``OTEL_SERVICE_NAME``. Supply ``instance_id`` here
    or publish ``service.instance.id`` in the parent before spawning.
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
    """Bound shutdown waiting; exporter retries can outlast the termination grace period."""
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
    tracer: Tracer | None = None,
    attrs: Callable[..., dict[str, Any] | None] | None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorate with a span, evaluating ``attrs(*args, **kwargs)`` only when enabled.

    Without ``attrs``, use Lens's ``trace_fn`` behavior.
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


class _NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        pass


_NO_OP_SPAN = _NoOpSpan()


@contextmanager
def span(
    group: str,
    name: str,
    attributes: dict[str, Any] | None = None,
    *,
    inherit_attributes: bool = False,
) -> Iterator[Span | _NoOpSpan]:
    """A lexical span; disabled spans accept attribute updates as no-ops.

    With ``inherit_attributes=True``, entry attributes also apply to nested spans
    until this block exits.
    """
    if not _AVAILABLE or not _is_span_group_enabled(group):
        yield _NO_OP_SPAN
        return
    attribute_scope = (
        _span_attributes(attributes) if inherit_attributes and attributes else nullcontext()
    )
    with attribute_scope, _managed_span(group, name, **(attributes or {})) as active:
        yield active if active is not None else _NO_OP_SPAN


def _emit(
    group: str,
    name: str,
    start: float,
    end: float,
    attributes: dict[str, Any] | None = None,
    context: Context | None = None,
) -> SpanContext | None:
    """Emit a completed interval and return its context, or None if disabled.

    ``context=None`` inherits the active span; an empty Context roots a trace.
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
    start: float | None,
    end: float | None,
    attributes: dict[str, Any] | None = None,
    parent: SpanContext | None = None,
) -> SpanContext | None:
    """Emit a completed window in wall-clock seconds and return its span context.

    Without ``parent``, the span starts a new trace. Missing timestamps or a
    disabled group produce no span. Zero duration is valid.
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


def mark(group: str, name: str, attributes: dict[str, Any] | None = None) -> SpanContext | None:
    """Emit a zero-duration span under the active parent and return its context.

    The span ends immediately; export may be buffered. Returns None if disabled.
    """
    if not _AVAILABLE or not _is_span_group_enabled(group):
        return None
    now = time.time()
    return _emit(group, name, now, now, attributes)


def record_process_startup(
    group: str,
    imports_started: float,
    imports_finished: float,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Emit separate root spans for process startup and timed module imports."""
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
    """Emit an ended ``<name>_start`` anchor and a backdated ``<name>`` summary.

    The anchor supplies the active context; the summary is its child.
    Opening attributes go on both records; later updates affect only the summary.
    An interrupted phase can leave its anchor and children without a summary.

    Open and close on the same thread, in reverse nesting order.
    """

    def __init__(self) -> None:
        self._window: tuple[str, str, float] | None = None
        self._parent: SpanContext | None = None
        self._token: Token[Context] | None = None
        self._attributes: dict[str, Any] = {}

    def open(self, group: str, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Emit the start anchor, closing any phase this handle had open."""
        if not _AVAILABLE:
            return
        self.close()
        self._window = (group, name, time.time())
        self._attributes = dict(attributes or {})
        try:
            self._parent = mark(group, f"{name}_start", attributes)
        except BaseException:
            self._window = self._parent = None
            self._attributes = {}
            raise
        if self._parent is None:  # Group disabled.
            return
        try:
            self._token = _otel_context.attach(
                _otel_trace.set_span_in_context(_otel_trace.NonRecordingSpan(self._parent))
            )
        except Exception:
            # Losing the ambient context costs nesting, not spans.
            logger.debug("Could not make %s the active context", name, exc_info=True)

    def set(self, attributes: dict[str, Any] | None = None) -> None:
        """Update attributes saved for the duration summary."""
        if self._window is None:
            return
        if attributes:
            self._attributes.update(attributes)

    def close(self, attributes: dict[str, Any] | None = None) -> None:
        """Emit the backdated span covering the phase. Idempotent."""
        self.set(attributes)
        if self._window is None:
            return
        group, name, start = self._window
        if self._token is not None:
            try:
                _otel_context.detach(self._token)
            except Exception:
                # Context restoration failed; still attempt the summary.
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
