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

"""Restart-cycle telemetry for the ft_launcher agent.

The agent runs in its own process, separate from the training workers, so it stands up
its own nemo-lens tracer and emits injob restart-cycle spans to the node's collector.

Three pieces, innermost first:

`_SpanTree`
    How spans nest, open and close. Best-effort by construction: an error inside OTel is
    logged at debug and swallowed, because a dropped span must never break the launcher.
`RestartCycleRecorder`
    When, as a table mapping each `ProfilingEvent` to actions on the tree. It observes the
    fault-tolerance profiler rather than being called by the launcher, so adding an event
    means adding a row to `_PLAN`.
`LauncherTelemetry`
    What `launcher.py` talks to. Disabled instances answer every call harmlessly, so no
    call site needs a guard.

`shared_utils/telemetry.py` owns the nemo-lens seam; nothing here imports nemo directly.
"""

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from ..shared_utils import telemetry as backend
from ..shared_utils.log_manager import LogConfig
from ..shared_utils.profiling import ProfilingEvent, ProfilingEventRecord, set_profiling_observer

log = logging.getLogger(LogConfig.name)

try:  # torchelastic delivers the FT death signal as SignalException (an Exception subclass)
    from torch.distributed.elastic.multiprocessing import SignalException as _SignalException
except Exception:  # torchelastic absent -> sentinel that never matches a real signal

    class _SignalException(Exception):
        pass


#: Presence of this variable is how a workload opts the agent into lens telemetry.
EXPORTER_ENV = 'NEMO_LENS_EXPORTER'
#: Bounds on export at teardown. The agent may be between SIGTERM and SIGKILL here.
FLUSH_TIMEOUT_MS = 1500
SHUTDOWN_TIMEOUT_S = 1.5

_PREFIX = 'nvrx.restart.'
_NODE_DESC_SUFFIX = re.compile(r'_\d+_\d+$')  # node_desc appends _<pid>_<local_rank>


class CycleOutcome:
    """Values of the `nvrx.cycle_outcome` attribute on a cycle span."""

    COMPLETED = 'completed'  # workers torn down after a failure; a restart follows
    EXCLUDED = 'excluded'  # this node bailed at its rendezvous health check
    STANDBY = 'standby'  # this node was not selected and went back to waiting
    SUCCEEDED = 'succeeded'  # the worker group finished cleanly
    FAILED = 'failed'  # this node's workers failed and it is driving the restart
    PEER_RESTART = 'peer_restart'  # a peer failed and this node is joining its restart
    TERMINATED = 'terminated'  # the agent is exiting with a cycle still open


def _clean_node(node):
    """Normalize node_desc ('host.cm.cluster_<pid>_<local>') to the clean host form, so
    phase spans and the cycle span carry the same nvrx.node value."""
    return _NODE_DESC_SUFFIX.sub('', node) if isinstance(node, str) else node


def _now_ns():
    return int(time.time() * 1e9)


def _safe(fn, what):
    """Call `fn`, returning None instead of raising: every attribute here is optional."""
    try:
        return fn()
    except Exception as e:
        log.debug("nvrx telemetry: %s unavailable: %s", what, e)
        return None


# --------------------------------------------------------------------------- span tree


class _SpanTree:
    """The span tree one node emits for one restart cycle.

    A parent `nvrx.restart.cycle` span, a *sweep* of phase children (at most one open at a
    time, each ending where the next begins), zero-duration marks, and side spans tracked
    by key that overlap the sweep or precede any cycle. The sweep is what keeps the tree
    leak-free on partial sequences: an evicted node never reaches the later boundaries, but
    whatever it had open is closed by the cycle close.
    """

    def __init__(self, tracer):
        self._tracer = tracer
        self._cycle = None
        self._ctx = None
        self._cycle_start_ns = None
        self._phase = None
        self._sides: Dict[str, Any] = {}

    @property
    def cycle_start_ns(self):
        return self._cycle_start_ns

    def _start(self, name, ns, attributes, parented=True):
        # Attributes are set at creation so they are visible to the sampler and to any
        # SpanProcessor.on_start, and it is one call instead of N.
        attrs = {k: v for k, v in attributes.items() if v is not None}
        try:
            try:
                return self._tracer.start_span(
                    name, start_time=ns, context=self._ctx if parented else None, attributes=attrs
                )
            except TypeError:  # tracers without a context kwarg
                return self._tracer.start_span(name, start_time=ns, attributes=attrs)
        except Exception as e:
            log.debug("nvrx telemetry: failed to start span %s: %s", name, e)
            return None

    @staticmethod
    def _end(span, ns):
        if span is None:
            return
        try:
            span.end(end_time=ns if ns is not None else _now_ns())
        except Exception as e:
            log.debug("nvrx telemetry: failed to end span: %s", e)

    @staticmethod
    def _annotate(span, attributes):
        if span is None:
            return
        for key, value in attributes.items():
            if value is None:
                continue
            try:
                span.set_attribute(key, value)
            except Exception as e:
                log.debug("nvrx telemetry: failed to set %s: %s", key, e)

    def open_cycle(self, ns, attributes):
        """Open the cycle span. Returns False when one is already open."""
        if self._cycle is not None:
            return False
        self._cycle = self._start(_PREFIX + 'cycle', ns, attributes)
        self._cycle_start_ns = ns
        try:
            from opentelemetry import trace

            self._ctx = trace.set_span_in_context(self._cycle)
        except Exception:  # opentelemetry absent -> children become root spans
            self._ctx = None
        return True

    def close_cycle(self, ns, attributes):
        """Close the cycle span and everything still open under it.

        An unended span is an unexported span, so this also sweeps up the open phase and
        any side span: otherwise a spare killed while waiting, or a node killed
        mid-attribution, loses those spans.
        """
        end_ns = ns if ns is not None else _now_ns()
        for key in list(self._sides):
            self.close_side(key, end_ns)
        self.end_phase(end_ns)
        span, self._cycle = self._cycle, None
        self._ctx = None
        self._cycle_start_ns = None  # cleared so a closed cycle cannot leak its start
        self._annotate(span, attributes)
        self._end(span, end_ns)

    def annotate_cycle(self, attributes):
        self._annotate(self._cycle, attributes)

    def start_phase(self, name, ns, attributes):
        self.end_phase(ns)  # sweep: the previous phase ends where this one begins
        self._phase = self._start(_PREFIX + name, ns, attributes)

    def end_phase(self, ns):
        span, self._phase = self._phase, None
        self._end(span, ns)

    def interval(self, name, start_ns, end_ns, attributes, parented=True):
        """Emit an already-finished span covering [start_ns, end_ns]."""
        self._end(self._start(name, start_ns, attributes, parented=parented), end_ns)

    def mark(self, name, ns, attributes):
        self.interval(_PREFIX + name, ns, ns, attributes)

    def open_side(self, key, ns, attributes, parented=True):
        """Open a span tracked by `key`, outside the phase sweep. `parented=False` forces a
        root span, as the standby round-wait needs: it precedes any cycle."""
        self.close_side(key, ns)
        self._sides[key] = self._start(_PREFIX + key, ns, attributes, parented=parented)

    def close_side(self, key, ns):
        self._end(self._sides.pop(key, None), ns)


# ----------------------------------------------------------------------------- recorder


@dataclass(frozen=True)
class ColdStartAnchors:
    """Wall-clock anchors for the part of the job that precedes any Python we control.

    Both come from the batch environment: by the time the agent starts, the queue wait and
    the launch script have already happened.
    """

    launch_script_start: Optional[float] = None  # batch script's first line, outside the srun
    slurm_job_start: Optional[float] = None  # Slurm's own job start (queue/prolog boundary)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "ColdStartAnchors":
        env = os.environ if env is None else env

        def _float(name):
            try:
                return float(env[name])
            except (KeyError, TypeError, ValueError):
                return None

        return cls(_float('LENS_LAUNCH_SCRIPT_START_TIME'), _float('SLURM_JOB_START_TIME'))


class RestartCycleRecorder:
    """Renders the profiling event stream as this node's restart-cycle span tree.

    Per-cycle event order (each node runs its own rendezvous, so every node sees these)::

        RENDEZVOUS_STARTED -> HEALTH_CHECK_COMPLETED -> RENDEZVOUS_COMPLETED
          -> WORKER_START_STARTED -> WORKER_START_COMPLETED -> (training)
          -> FAILURE_DETECTED -> WORKER_TERMINATED

    An evicted node bails right after HEALTH_CHECK_COMPLETED (UnhealthyNodeException in the
    health check) and emits NODE_EXCLUDED instead of the rest.

    Outcomes the event stream cannot express (succeeded, failed, peer_restart, terminated)
    are staged by the agent through `stage_outcome()` / `finish_cycle()`.
    """

    #: event -> ((action, argument), ...). This table is the specification of the tree.
    _PLAN = {
        ProfilingEvent.RENDEZVOUS_STARTED: (('cycle_open', None), ('phase', 'health_check')),
        ProfilingEvent.HEALTH_CHECK_COMPLETED: (('phase', 'rendezvous'),),
        ProfilingEvent.RENDEZVOUS_COMPLETED: (('end_phase', None),),
        # Launching workers means this node was selected active this cycle, so stamp it at
        # selection time: a node killed after selection but before the close still shows
        # membership (complementing cycle_outcome, where standby means not selected).
        ProfilingEvent.WORKER_START_STARTED: (
            ('phase', 'worker_launch'),
            ('annotate', {'nvrx.membership': 'active'}),
        ),
        ProfilingEvent.WORKER_START_COMPLETED: (('phase', 'run'),),
        ProfilingEvent.FAILURE_DETECTED: (('mark', 'fault'), ('phase', 'teardown')),
        ProfilingEvent.WORKER_TERMINATED: (
            ('end_phase', None),
            ('cycle_close', CycleOutcome.COMPLETED),
        ),
        ProfilingEvent.NODE_EXCLUDED: (
            ('end_phase', None),
            ('mark', 'excluded'),
            ('cycle_close', CycleOutcome.EXCLUDED),
        ),
        ProfilingEvent.ATTRIBUTION_GET_STARTED: (('attr_open', None),),
        ProfilingEvent.ATTRIBUTION_GET_COMPLETED: (('attr_close', None),),
        # Standby / round-open wait. A hot spare not in the active rendezvous sits here for
        # a whole cycle and would otherwise be invisible. It re-enters the wait still
        # holding the previous round's cycle (it started that round but was never selected,
        # so no WORKER_TERMINATED closed it), so close the whole cycle first: otherwise
        # open_cycle's already-open guard keeps the stale one and a promoted spare reuses
        # the earlier round's start. A no-op for a normal node.
        ProfilingEvent.AWAIT_ROUND_STARTED: (
            ('cycle_close', CycleOutcome.STANDBY),
            ('await_open', None),
        ),
        ProfilingEvent.AWAIT_ROUND_COMPLETED: (('await_close', None),),
    }

    #: Events after which this node may be killed, so its spans are force-flushed
    #: synchronously; routine events rely on BatchSpanProcessor's normal export.
    _FLUSH_EVENTS = frozenset(
        {
            ProfilingEvent.FAILURE_DETECTED,
            ProfilingEvent.WORKER_TERMINATED,
            ProfilingEvent.NODE_EXCLUDED,
        }
    )

    def __init__(self, tracer, flush=None, cold_start: Optional[ColdStartAnchors] = None):
        self._tree = _SpanTree(tracer)
        self._flush = flush
        self._cold_start = cold_start or ColdStartAnchors()
        self._cold_start_done = False
        # Guards recorder state, mutated by the launcher's main thread and by the
        # attribution poller daemon (health_check _poll_loop -> ATTRIBUTION_GET_*).
        self._lock = threading.RLock()
        self._outcome = None
        self._extra: Dict[str, Any] = {}

    # ---------------------------------------------------------------- observer API

    def on_event(self, record: ProfilingEventRecord) -> None:
        """Apply this boundary event to the span tree, flushing kill-adjacent events
        immediately so an evicted node's spans survive its kill."""
        ns = int(record.timestamp * 1e9)
        node = _clean_node(record.node_id)
        base = {'is_goodput_span': True, 'nvrx.cycle': record.cycle, 'nvrx.node': node}
        ranked = dict(base, **{'nvrx.rank': record.rank})

        with self._lock:
            try:
                self._emit_cold_start_once(ns, node)
                for action, arg in self._PLAN.get(record.event, ()):
                    if action == 'cycle_open':
                        if self._tree.open_cycle(ns, base):
                            self._clear_staged()
                    elif action == 'cycle_close':
                        self._close_cycle(ns, arg)
                    elif action == 'phase':
                        self._tree.start_phase(arg, ns, ranked)
                    elif action == 'end_phase':
                        self._tree.end_phase(ns)
                    elif action == 'mark':
                        self._tree.mark(arg, ns, base)
                    elif action == 'annotate':
                        self._tree.annotate_cycle(arg)
                    elif action == 'attr_open':
                        self._tree.open_side('attribution', ns, ranked)
                    elif action == 'attr_close':
                        self._tree.close_side('attribution', ns)
                    elif action == 'await_open':
                        # root span: the standby wait precedes any cycle
                        self._tree.open_side(
                            'await_round',
                            ns,
                            dict(base, **{'nvrx.membership': 'standby'}),
                            parented=False,
                        )
                    elif action == 'await_close':
                        self._tree.close_side('await_round', ns)
                if record.event in self._FLUSH_EVENTS:
                    self.flush()
            except (_SignalException, KeyboardInterrupt, SystemExit):
                raise  # never swallow the FT death signal (torchelastic raises it here)
            except Exception as e:
                log.debug("nvrx telemetry: event %s not recorded: %s", record.event.value, e)

    # ------------------------------------------------------------ agent-facing API

    def annotate_cycle(self, **attributes) -> None:
        """Enrich the open cycle span with metadata only the agent knows."""
        with self._lock:
            self._tree.annotate_cycle(attributes)

    def stage_outcome(self, outcome: str, **attributes) -> None:
        """Record how this cycle is ending, ahead of the event that closes it."""
        with self._lock:
            self._outcome = outcome
            self._extra = dict(attributes)

    def finish_cycle(self, outcome: str = CycleOutcome.COMPLETED, **attributes) -> None:
        """Close a cycle whose terminal state has no teardown event to close it."""
        with self._lock:
            self._outcome = outcome
            if attributes:
                self._extra = dict(attributes)
            self._close_cycle(None, outcome)
            self.flush()

    def cycle_start_seconds(self) -> Optional[float]:
        """Wall-clock start of the open cycle (this round's rendezvous start), or None."""
        with self._lock:
            ns = self._tree.cycle_start_ns
            return None if ns is None else ns / 1e9

    def flush(self) -> None:
        if self._flush is None:
            return
        try:
            self._flush()
        except Exception as e:
            log.debug("nvrx telemetry: flush failed: %s", e)

    # ---------------------------------------------------------------------- internals

    def _clear_staged(self):
        self._outcome = None
        self._extra = {}

    def _close_cycle(self, ns, outcome):
        """Close the cycle, stamping the staged outcome in preference to the plan's."""
        attributes = {'nvrx.cycle_outcome': self._outcome or outcome or CycleOutcome.COMPLETED}
        attributes.update(self._extra)
        self._tree.close_cycle(ns, attributes)
        self._clear_staged()

    def _emit_cold_start_once(self, ns, node):
        """Emit the pre-agent spans, backdated and closed immediately, once per node.

        `nvrx.cold_start` covers the launch script up to this agent's first recorded event.
        `pre_startup` covers the queue/prolog gap before it; under NVRx the agent owns that
        window (megatron suppresses its own on any ft_launcher cohort), so the whole
        pre-Python envelope has one owner. Its name matches the bare-path span so the
        goodput taxonomy classifies it identically.
        """
        launch = self._cold_start.launch_script_start
        if self._cold_start_done or launch is None:
            return
        self._cold_start_done = True
        attributes = {'is_goodput_span': True, 'nvrx.node': node}
        job_start = self._cold_start.slurm_job_start
        if job_start is not None and job_start < launch:
            self._tree.interval(
                'pre_startup', int(job_start * 1e9), int(launch * 1e9), attributes, parented=False
            )
        self._tree.interval('nvrx.cold_start', int(launch * 1e9), ns, attributes, parented=False)


# ------------------------------------------------------------------------ agent facade


#: rdzv handler getters describing who was active or standby this cycle. Optional: not
#: every rendezvous backend implements them.
_TOPOLOGY_GETTERS = (
    ('nvrx.active_nodes', 'get_active_node_addrs'),
    ('nvrx.standby_nodes', 'get_standby_node_addrs'),
    ('nvrx.active_ranks', 'get_active_ranks'),
)


def _infrastructure_rank():
    def _get():
        from .utils import get_infrastructure_rank

        return get_infrastructure_rank(skip_nodename_logic=True)

    return _safe(_get, 'infrastructure rank')


class LauncherTelemetry:
    """Restart-cycle telemetry for one ft_launcher agent, enabled or not.

    A disabled instance answers every call harmlessly, so `launcher.py` needs no guards
    and no try/except at any call site. The agent's role in the span tree is small: the
    recorder owns the spans, and the agent only enriches the open cycle, stages the
    outcome, closes terminal cycles, and hands each worker cohort its anchors.
    """

    def __init__(self, node_id, env: Optional[Mapping[str, str]] = None):
        env = os.environ if env is None else env
        self._node_id = node_id
        self._handle = None
        self._recorder = None
        if not env.get(EXPORTER_ENV) or not backend.HAS_NEMO_LENS:
            return
        try:
            config = backend.config_from_env()
            config.enabled = True
            self._handle = backend.setup_telemetry(
                config,
                rank=0,
                world_size=1,
                resource_attributes={
                    'nvrx.role': 'ft_launcher_agent',
                    'nvrx.node': str(node_id),
                },
            )
            self._recorder = RestartCycleRecorder(
                self._handle.tracer,
                flush=_provider_flush(),
                cold_start=ColdStartAnchors.from_env(env),
            )
            # Registering the recorder is what turns the launcher's and the rendezvous
            # handler's pre-placed boundary events into spans.
            set_profiling_observer(self._recorder)
        except Exception as e:  # never let telemetry setup break the launcher
            log.debug("nvrx telemetry: agent tracer setup skipped: %s", e)
            if self._recorder is not None:
                set_profiling_observer(None)
            if self._handle is not None:
                # setup_telemetry() succeeded and something after it failed. Shut the
                # provider down rather than orphaning its exporter thread and connection
                # for the life of the process.
                try:
                    self._handle.shutdown()
                except Exception:
                    pass
            self._handle = None
            self._recorder = None

    def annotate_cycle(self, worker_group, remaining_restarts) -> None:
        """Describe the open cycle: restart budget, group rank, rendezvous topology."""
        if self._recorder is None:
            return
        spec = worker_group.spec
        attributes = {
            'nvrx.remaining_restarts': remaining_restarts,
            'nvrx.max_restarts': _safe(lambda: spec.max_restarts, 'max_restarts'),
            'nvrx.node': str(self._node_id),
            'nvrx.membership': 'active',  # annotate runs once this node launched workers
            'nvrx.group_rank': _safe(lambda: worker_group.group_rank, 'group_rank'),
            'nvrx.group_world_size': _safe(
                lambda: worker_group.group_world_size, 'group_world_size'
            ),
            'nvrx.rdzv_run_id': _safe(lambda: spec.rdzv_handler.get_run_id(), 'run id'),
            # Same source as cycle_info, so a cycle span can answer who was active or
            # standby and which physical rank this node held.
            'nvrx.infra_rank': _infrastructure_rank(),
        }
        for key, getter in _TOPOLOGY_GETTERS:
            values = _safe(lambda g=getter: getattr(spec.rdzv_handler, g)(), getter)
            if values is not None:
                attributes[key] = ",".join(str(v) for v in values)
        self._recorder.annotate_cycle(**attributes)

    def stage_outcome(self, outcome: str, **attributes) -> None:
        """Record how the current cycle is ending, before the teardown event closes it."""
        if self._recorder is not None:
            self._recorder.stage_outcome(outcome, **attributes)

    def finish_cycle(self, outcome: str = CycleOutcome.COMPLETED, **attributes) -> None:
        """Close a cycle that ends without a teardown event (clean finish, agent exit)."""
        if self._recorder is not None:
            self._recorder.finish_cycle(outcome, **attributes)

    def worker_env(self, cycle: int) -> Dict[str, str]:
        """Telemetry environment for the worker cohort the agent is about to launch.

        Emitted whether or not the agent itself is recording, because it is the trainer
        that consumes these. One launch stamp per cohort, so every restart hands its
        workers a fresh anchor: the batch script's launch_script_start is set once,
        outside the single srun, and is stale by then.
        """
        env = {
            'NVRX_LAUNCH_TIME': repr(time.time()),
            'NVRX_CYCLE': str(cycle),
            'NVRX_MEMBERSHIP': 'active',  # a launched worker is active this cycle
        }
        infra_rank = _infrastructure_rank()
        if infra_rank is not None:
            env['NVRX_INFRA_RANK'] = str(infra_rank)
        # For any cycle after the first (an in-place restart or a promoted spare, both of
        # which launch at cycle > 0) anchor pre_startup at this round's rendezvous start
        # rather than at the stale sbatch job start. Cycle 0 leaves it unset so pre_startup
        # falls back to the real sbatch queue time.
        if cycle and self._recorder is not None:
            start = self._recorder.cycle_start_seconds()
            if start is not None:
                env['NVRX_CYCLE_START_TIME'] = repr(start)
        return env

    def shutdown(self) -> None:
        """Close the open cycle, then bound the provider shutdown.

        BatchSpanProcessor only flushes on a timer or on shutdown, so without this the last
        cycle's spans are dropped on exit. finish_cycle() already force-flushed them, and
        handle.shutdown() can block on a dead collector for longer than the SIGTERM-to-
        SIGKILL grace, so the shutdown itself runs on a thread we wait on only briefly.
        """
        if self._handle is None:
            return
        try:
            self.finish_cycle(CycleOutcome.TERMINATED)
            set_profiling_observer(None)
            thread = threading.Thread(
                target=self._handle.shutdown, daemon=True, name='nvrx-otel-shutdown'
            )
            thread.start()
            thread.join(SHUTDOWN_TIMEOUT_S)
        except Exception as e:
            log.debug("nvrx telemetry: agent shutdown incomplete: %s", e)


def _provider_flush():
    """A bounded force_flush callable for the active tracer provider, or None."""

    def _get():
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if not hasattr(provider, 'force_flush'):
            return None
        return lambda: provider.force_flush(timeout_millis=FLUSH_TIMEOUT_MS)

    return _safe(_get, 'tracer provider flush')
