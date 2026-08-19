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
runs, so every span here is a no-op before (or without) initialization -- there
is nothing for this module to re-implement.

Attributes are passed as a dict to every entry point, because NVRx attribute
names are dotted and so cannot be Python keywords.
"""

import logging
import os
import threading
import time
from contextlib import ExitStack, contextmanager
from typing import ClassVar, Optional

logger = logging.getLogger(__name__)

_NVRX_GROUPS = frozenset(["nvrx.ft", "nvrx.ckpt"])

# The OTEL_RESOURCE_ATTRIBUTES this process inherited, captured before anything
# can extend it. Every extension is built from this value rather than from the
# last extended one: the agent launches a fresh worker cohort every cycle, and
# extending the extension would append another nvrx.cycle each time until the
# variable grew without bound across a long-running job.
_INHERITED_RESOURCE_ATTRIBUTES = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")

try:
    # Names NVRx re-exports keep their nemo-lens spelling, so that searching for
    # one finds every use across nemo-lens and its consumers. The underscored
    # ones are internal: they only exist when nemo-lens is installed, and
    # _setup_telemetry would otherwise collide with the wrapper defined below.
    from nemo.lens import NemoLensConfig as _NemoLensConfig
    from nemo.lens import SpanGroup as _SpanGroup
    from nemo.lens import get_tracer as _get_tracer
    from nemo.lens import is_span_group_enabled as _is_span_group_enabled
    from nemo.lens import managed_span as _managed_span
    from nemo.lens import safe_set_span_attributes as _safe_set_span_attributes
    from nemo.lens import setup_telemetry as _setup_telemetry
    from nemo.lens import trace_fn

    class _NVRxSpanGroup(_SpanGroup):
        """Teaches nemo-lens about the NVRx groups, and about groups nobody declared.

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

        @classmethod
        def resolve(cls, spec: str) -> frozenset:
            """Resolve a span-group spec, passing unknown names through as groups.

            Upstream raises on any name it does not recognise. Since NVRx catches
            that and degrades to a no-op handle, one undeclared name would take
            *all* telemetry down rather than just enabling that group -- and the
            failure is silent. It also means adding a span under a new group, for
            a one-off investigation you intend to filter on in the collector,
            could not be done without editing this file.

            So an unrecognised name is its own group. A typo then costs that one
            group rather than the entire signal, and the log line says which.
            """
            known = cls.ALL_GROUPS | set(cls._PRESETS)
            parts = [p.strip().lower() for p in spec.split(",") if p.strip()]
            declared = ",".join(p for p in parts if p in known)
            adhoc = frozenset(p for p in parts if p not in known)
            if adhoc:
                logger.info("Enabling span groups not declared by NVRx: %s", sorted(adhoc))
            return (super().resolve(declared) if declared else frozenset()) | adhoc

    _AVAILABLE = True

except ModuleNotFoundError:
    _AVAILABLE = False

except Exception:
    logger.warning("nemo-lens import failed, continuing without telemetry", exc_info=True)
    _AVAILABLE = False


if not _AVAILABLE:

    @contextmanager
    def _managed_span(group, name, tracer=None, **attributes):
        """No-op stand-in for ``nemo.lens.managed_span``."""
        yield None

    def trace_fn(group, name, tracer=None):
        """No-op stand-in for ``nemo.lens.trace_fn``."""

        def decorator(func):
            return func

        return decorator


class _NoOpHandle:
    """Stand-in for ``nemo.lens.TelemetryHandle`` when telemetry is unavailable."""

    def shutdown(self, timeout_ms: int = 5000) -> None:
        pass


def setup_telemetry(rank: int, world_size: int, resource_attributes: Optional[dict] = None):
    """Initialize nemo-lens for this process. Call once, at process start.

    ``rank`` and ``world_size`` identify this process in the OTel resource
    (``dl.rank``, ``dl.world_size``, ``service.instance.id``), so they must be
    distinct per emitter or backends cannot tell the processes apart.
    ``resource_attributes`` is merged over those defaults, which is how a
    process whose rank is not a trainer rank overrides ``service.instance.id``
    rather than colliding with the trainer that happens to share its number.

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
        return _setup_telemetry(config, rank, world_size, resource_attributes=resource_attributes)
    except Exception:
        logger.warning("nemo-lens init failed, continuing without telemetry", exc_info=True)
        return _NoOpHandle()


def shutdown(handle, timeout_s: float = 2.0) -> None:
    """Flush and shut down, bounded.

    ``TelemetryHandle.shutdown()`` flushes synchronously and can block for as
    long as the exporter's own retry budget against a collector that is gone --
    longer than the SIGTERM-to-SIGKILL grace a launcher gets. Run it on a daemon
    thread and stop waiting after ``timeout_s``; a missed flush is preferable to
    being killed mid-teardown.
    """
    worker = threading.Thread(target=handle.shutdown, daemon=True)
    worker.start()
    worker.join(timeout_s)


def flush(timeout_ms: int = 1500) -> None:
    """Export what is buffered, without shutting the providers down.

    For the handful of points where this process may be killed moments later --
    a detected fault, a health-check exclusion -- so those spans reach the
    collector rather than dying in the batch processor's queue. Everywhere else
    the batch processor's own schedule is enough.
    """
    if not _AVAILABLE:
        return
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush(timeout_millis=timeout_ms)


class ManualSpan:
    """A span started and ended explicitly, for a lifetime that is not a block.

    `managed_span` covers anything scoped to a `with`. This is for the rest: a
    span opened in one call and closed in another, where the caller cannot hold
    a context manager across the two. It owns the ExitStack bookkeeping, and
    every method is a no-op while no span is open, so callers need no guards.

    ORDERING CONTRACT. `managed_span` attaches an OTel context token on entry
    and detaches it on exit, and contextvar tokens must be reset in the reverse
    order they were set, on the thread that set them. So:

    * open() and close() must run on the same thread.
    * Any span opened after this one -- including a `with managed_span(...)` or
      a `@trace_fn` method -- must finish before close() is called.

    Violating either does not raise. It silently restores a stale OTel context,
    so later spans are parented to a span that has already ended.
    """

    def __init__(self) -> None:
        self._stack: Optional[ExitStack] = None
        self._span = None

    def open(self, group: str, name: str, attributes: Optional[dict] = None) -> None:
        """Start a span, closing any span this handle already had open."""
        self.close()
        self._stack = ExitStack()
        self._span = self._stack.enter_context(span(group, name, attributes))

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


def span(group: str, name: str, attributes: Optional[dict] = None):
    """A span around a block, yielding it (or None when the group is off).

    Adapter over ``nemo.lens.managed_span``, which takes attributes as keyword
    arguments. Every attribute NVRx sets is dotted -- ``nvrx.cycle``,
    ``nvrx.call_idx`` -- and a dotted name is not a Python identifier, so it can
    never be a keyword argument: ``f(nvrx.cycle=3)`` is a SyntaxError. A dict is
    the only way to carry these names, and taking it here rather than making
    every caller write ``**{...}`` keeps attributes one shape across ``span``,
    ``mark``, ``set_span_attributes`` and ``ManualSpan``.

    Returns the upstream context manager rather than wrapping it in another one,
    so the only cost over calling ``managed_span`` directly is rebuilding the
    dict that ``**`` unpacks -- about 200ns against 40us for an enabled span.
    """
    return _managed_span(group, name, **(attributes or {}))


def backdated_span(
    group: str,
    name: str,
    start: Optional[float],
    end: Optional[float],
    attributes: Optional[dict] = None,
    parent=None,
) -> None:
    """Record a span for a window that elapsed before there was a tracer.

    Startup phases are only measurable in hindsight: the job was queued and the
    launch script ran before this process existed, so the span has to be created
    with both timestamps rather than wrapped around live code. ``start`` and
    ``end`` are wall-clock seconds.

    ``parent`` is the ``SpanContext`` this span belongs under, normally the
    ``SpanContext`` returned by the ``mark`` that opened the window. Passing it
    explicitly is what puts the backdated span in the same trace as the spans
    that ran inside the window, even though nothing was held open across it.

    Without a ``parent`` the span is emitted with an empty context and roots its
    own trace, which is what a window that predates every other span in the
    process wants.

    A no-op if the window is not a positive interval, so callers can pass
    whatever timestamps they found without pre-checking.
    """
    if start is None or end is None or end <= start:
        return
    if not _AVAILABLE or not _is_span_group_enabled(group):
        return
    from opentelemetry import trace
    from opentelemetry.context import Context

    context = Context()
    if parent is not None:
        context = trace.set_span_in_context(trace.NonRecordingSpan(parent), context)
    tracer = _get_tracer(__name__)
    span = tracer.start_span(
        name, context=context, start_time=int(start * 1e9), attributes=attributes or {}
    )
    span.end(end_time=int(end * 1e9))


def mark(group: str, name: str, attributes: Optional[dict] = None):
    """Record an instant: a zero-duration span pinning a moment in time.

    For a boundary worth a timestamp but with no duration of its own, where the
    surrounding spans start too late to pin it -- a detected fault sits between
    the run span ending and the teardown span starting.

    Returns the ``SpanContext`` of the instant, or None when the group is off.
    A mark exports the moment it is made, so its context outlives it: it is a
    trace and span id, not a handle to anything still running. That is what lets
    a long window be opened by a mark and closed by a backdated span in the same
    trace, without a span being held open across it.
    """
    with span(group, name, attributes) as recorded:
        return recorded.get_span_context() if recorded is not None else None


def set_span_attributes(attributes: dict) -> None:
    """Set attributes on the currently active span.

    For use inside a ``@trace_fn`` function, which owns its span but does not
    hand it to the caller. A no-op when no span is recording.
    """
    if not _AVAILABLE:
        return
    from opentelemetry import trace

    _safe_set_span_attributes(trace.get_current_span(), attributes)


def extended_resource_attributes(attributes: dict) -> str:
    """Extend the inherited ``OTEL_RESOURCE_ATTRIBUTES`` with more pairs.

    The variable is comma-separated ``key=value`` with percent-encoded values.
    An OTel SDK reads it into the process Resource with no code at all, and it
    is inherited across every process spawn, which makes it the way to tell a
    process something about itself that it cannot work out on its own -- and it
    costs nothing per span, since a Resource is serialized once per export batch
    rather than once per span.

    NVRx reads this variable but never parses it. Whatever a launching
    environment put in it is an opaque string here: NVRx looks up no key in it,
    and behaves identically whether it holds nothing or twenty keys from some
    scheduler NVRx has never heard of. Reading a string in order to extend it
    creates no coupling; branching on a key in it would.

    Values are percent-encoded. NVRx's own are integers and a fixed enum, so
    they would survive unencoded, but encoding makes that a property of the code
    rather than a property of the current set of attributes.

    Callers must not pass a key the inherited value already sets, since which
    occurrence wins is not something to depend on.
    """
    from urllib.parse import quote

    added = ",".join(f"{key}={quote(str(value), safe='')}" for key, value in attributes.items())
    if not _INHERITED_RESOURCE_ATTRIBUTES:
        return added
    if not added:
        return _INHERITED_RESOURCE_ATTRIBUTES
    return f"{_INHERITED_RESOURCE_ATTRIBUTES},{added}"


def traceparent(context) -> Optional[str]:
    """Format a ``SpanContext`` as a W3C ``traceparent``, for another process.

    Returns None when there is no context to pass on, or when telemetry is not
    available to format it.

    What the receiving process does with it is the receiver's decision, and the
    intended one is a **link** rather than a parent. A link references a span
    without inheriting its trace id or its lifetime, so a trainer's spans stay
    in the trainer's own per-rank trace and merely record which cycle they ran
    under. Parenting would pull every rank of every cycle into one trace, which
    no viewer can open and no query needs.

    Nothing reads this variable automatically -- no OTel SDK looks for
    TRACEPARENT in the environment -- so the receiver extracts it explicitly.
    """
    if not _AVAILABLE or context is None:
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

        carrier: dict = {}
        TraceContextTextMapPropagator().inject(
            carrier, context=trace.set_span_in_context(trace.NonRecordingSpan(context))
        )
        return carrier.get("traceparent")
    except Exception:
        logger.debug("Could not format a traceparent", exc_info=True)
        return None


def context_carrier() -> Optional[dict]:
    """Serialize the active trace context and Baggage into a picklable dict.

    For handing a span's identity to a process that does not share memory with
    this one. The result is a small dict of W3C header strings -- ``traceparent``
    and, when anything is in Baggage, ``baggage`` -- so it survives pickling onto
    a multiprocessing queue, which a live ``SpanContext`` does not.

    Uses the globally configured propagators, so whatever the deployment has set
    up rides along rather than only the two NVRx happens to know about.

    Returns None when there is nothing to carry, which is also what a caller
    should store when telemetry is off: the receiving side then behaves exactly
    as it did before any of this existed.
    """
    if not _AVAILABLE:
        return None
    try:
        from opentelemetry import propagate

        carrier: dict = {}
        propagate.inject(carrier)
        return carrier or None
    except Exception:
        logger.debug("Could not capture a context carrier", exc_info=True)
        return None


def carrier_baggage(carrier: Optional[dict]) -> dict:
    """Read the Baggage out of a carrier, as a plain dict.

    Baggage is ambient key/value context that rides alongside the trace context.
    It is not telemetry by itself and lands on no span automatically -- reading a
    value out of it and setting it as a span attribute is a deliberate act, which
    is why this returns the values rather than applying them.
    """
    if not _AVAILABLE or not carrier:
        return {}
    try:
        from opentelemetry import baggage, propagate

        return dict(baggage.get_all(propagate.extract(carrier)))
    except Exception:
        logger.debug("Could not read baggage from a carrier", exc_info=True)
        return {}


@contextmanager
def linked_span(group: str, name: str, carrier: Optional[dict], attributes: Optional[dict] = None):
    """A span in this process's own trace, linking to a span in another.

    A **link** references another span without inheriting its trace id or its
    lifetime. That is the difference that matters here: parenting would pull
    every worker span into the trainer's trace, and with it every rank of every
    checkpoint, producing one trace no viewer can open. A link leaves the worker
    in its own trace and still records what caused the work, so a consumer can
    navigate from a slow write back to the training step that asked for it.

    Falls back to an ordinary span when there is no carrier or no telemetry, so
    a caller needs no branch of its own.
    """
    if not _AVAILABLE or not carrier:
        with span(group, name, attributes) as recorded:
            yield recorded
        return
    try:
        from opentelemetry import propagate, trace

        link_context = trace.get_current_span(propagate.extract(carrier)).get_span_context()
        links = [trace.Link(link_context)] if link_context.is_valid else None
    except Exception:
        logger.debug("Could not build a link from a carrier", exc_info=True)
        links = None
    if links is None or not _is_span_group_enabled(group):
        with span(group, name, attributes) as recorded:
            yield recorded
        return
    from opentelemetry import trace

    tracer = _get_tracer(__name__)
    started = tracer.start_span(name, links=links, attributes=attributes or {})
    with trace.use_span(started, end_on_exit=True):
        yield started


def record_process_startup(
    group: str,
    imports_started: float,
    imports_finished: float,
    attributes: Optional[dict] = None,
) -> None:
    """Record how long this process took to become able to run.

    Two windows, both over before there was a tracer to measure them, so both
    are backdated:

    * ``python.startup`` -- the process being created to its entry module's
      first statement. Interpreter start, and on a shared filesystem the cost of
      finding and reading the interpreter and the standard library at all.
    * ``python.imports`` -- the entry module's top-level imports. Importing
      torch alone is seconds, and on a cold page cache across thousands of nodes
      it is a large share of the time before a job does any work.

    ``imports_started`` and ``imports_finished`` are wall-clock seconds stamped
    around the entry module's import block; the process create time comes from
    the OS. Neither span carries an ``nvrx.`` prefix -- they describe Python, not
    NVRx, and ``service.name`` already says which process emitted them, so one
    query answers "how long did imports take" across every service that records
    them.

    Both root their own trace. They precede every cycle, so there is nothing for
    them to belong to.
    """
    try:
        import psutil

        created = psutil.Process().create_time()
    except Exception:
        logger.debug("Process create time unavailable", exc_info=True)
        created = None
    backdated_span(group, "python.startup", created, imports_started, attributes)
    backdated_span(group, "python.imports", imports_started, imports_finished, attributes)


class Phase:
    """A long window, recorded as a start marker now and a backdated span later.

    A span exports only when it ends, so a span held open across a restart cycle
    -- which on a healthy job is the whole job -- is invisible for as long as the
    thing it measures is interesting. Worse, if the process dies the span is
    never exported at all, and the window that most wanted recording is the one
    that goes missing.

    So a phase is two spans. ``open()`` records a zero-duration ``<name>_start``
    mark, which exports immediately; ``close()`` records a ``<name>`` span
    backdated to that instant, carrying the duration and the outcome. A phase
    that never closes leaves its marker and every span that ran inside it, and
    the absence of the backdated span is itself the signal that it did not
    finish.

    Between the two, the marker's ``SpanContext`` is the active OTel context, so
    spans opened anywhere on this thread nest under the phase and land in its
    trace without anything being passed to them. That is ``ManualSpan``'s
    grouping property without ``ManualSpan``'s lifetime.

    Attributes given to ``open`` go on the marker; attributes accumulated by
    ``set`` and ``close`` go on the backdated span, since there is no live span
    to carry them in between.

    ORDERING CONTRACT, inherited from ``contextvars``: tokens must be reset in
    the reverse order they were set, on the thread that set them. So ``open()``
    and ``close()`` must run on the same thread, and any phase or span opened
    after this one must close before it does.
    """

    def __init__(self) -> None:
        self._group: Optional[str] = None
        self._name: Optional[str] = None
        self._start: Optional[float] = None
        self._parent = None
        self._token = None
        self._attributes: dict = {}

    @property
    def context(self):
        """The start mark's ``SpanContext``, or None while no phase is open.

        This is what another process is given in order to link to the phase.
        It stays valid for the phase's whole lifetime precisely because the mark
        it names has already ended.
        """
        return self._parent

    def open(self, group: str, name: str, attributes: Optional[dict] = None) -> None:
        """Mark the start of the phase, closing any phase this handle had open."""
        self.close()
        self._group, self._name = group, name
        self._start = time.time()
        self._attributes = {}
        self._parent = mark(group, f"{name}_start", attributes)
        if not _AVAILABLE or self._parent is None:
            return
        try:
            from opentelemetry import context as otel_context
            from opentelemetry import trace

            self._token = otel_context.attach(
                trace.set_span_in_context(trace.NonRecordingSpan(self._parent))
            )
        except Exception:
            # Losing the ambient context costs nesting, not spans, and must not
            # cost the workload anything at all.
            logger.debug("Could not make %s the active context", self._name, exc_info=True)

    def set(self, attributes: Optional[dict] = None) -> None:
        """Record attributes to be emitted on the backdated span at close."""
        if self._start is None or not attributes:
            return
        self._attributes.update(attributes)

    def close(self, attributes: Optional[dict] = None) -> None:
        """Emit the backdated span covering the phase. Idempotent."""
        self.set(attributes)
        if self._start is None:
            return
        if self._token is not None:
            try:
                from opentelemetry import context as otel_context

                otel_context.detach(self._token)
            except Exception:
                # A phase opened after this one outlived it, so the token is no
                # longer the top of this thread's context stack. The span is
                # still correct; only the ambient context for whatever runs next
                # is stale, and the next open() replaces it.
                logger.debug("Out-of-order close for phase %s", self._name, exc_info=True)
            self._token = None
        backdated_span(
            self._group,
            self._name,
            self._start,
            time.time(),
            self._attributes,
            parent=self._parent,
        )
        self._group = self._name = self._start = self._parent = None
        self._attributes = {}
