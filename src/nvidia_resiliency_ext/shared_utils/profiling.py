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

# This file adds time profiling capabilities for fault tolerance (cycle and event logging).

import logging
import re
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from ..shared_utils.log_manager import LogConfig

try:  # torchelastic delivers the FT death signal as SignalException (an Exception subclass)
    from torch.distributed.elastic.multiprocessing import SignalException as _SignalException
except Exception:  # torchelastic absent -> sentinel that never matches a real signal

    class _SignalException(Exception):
        pass


_NODE_DESC_SUFFIX = re.compile(r'_\d+_\d+$')  # node_desc appends _<pid>_<local_rank>


def _clean_node(n):
    """Normalize node_desc ('host.cm.cluster_<pid>_<local>') to the clean host form, so phase
    spans and the cycle span carry the same nvrx.node value."""
    return _NODE_DESC_SUFFIX.sub('', n) if isinstance(n, str) else n


class ProfilingEvent(Enum):
    """Enumeration of profiling events for fault tolerance metrics."""

    FAILURE_DETECTED = "failure_detected"
    WORKER_TERMINATED = "worker_terminated"
    RENDEZVOUS_STARTED = "rendezvous_started"
    HEALTH_CHECK_COMPLETED = "health_check_completed"
    RENDEZVOUS_COMPLETED = "rendezvous_completed"
    WORKER_START_STARTED = "worker_start_started"
    WORKER_START_COMPLETED = "worker_start_completed"
    ATTRIBUTION_GET_STARTED = "attribution_get_started"
    ATTRIBUTION_GET_COMPLETED = "attribution_get_completed"
    NODE_EXCLUDED = "node_excluded"  # bailed at the rendezvous health check (evicted/unhealthy)
    AWAIT_ROUND_STARTED = "await_round_started"  # entered the Step-0 wait for the round
    AWAIT_ROUND_COMPLETED = "await_round_completed"  # round opened (or shutdown); node moves on


class FaultToleranceProfiler:
    """Profiler for measuring fault tolerance timing metrics (cycle and event logging)."""

    def __init__(self):
        self._current_cycle = 0
        self._logger = logging.getLogger(LogConfig.name)
        # OTel per-cycle span-tree state, set by the ft_launcher agent via attach_otel.
        self._otel_tracer = None
        self._otel_flush = None
        # Guards the recorder span state, which is mutated by the launcher's main thread
        # and by the attribution poller daemon (health_check _poll_loop -> ATTRIBUTION_GET_*).
        self._otel_lock = threading.RLock()
        self._otel_cycle_span = None  # per-cycle parent span ('nvrx.restart.cycle')
        self._otel_cycle_ctx = None  # parent context, so phases nest as children (one trace/cycle)
        self._otel_phase = None  # (name, span) of the single currently-open phase (sweep model)
        self._otel_attr = None  # attribution span (a nested lookup, tracked off the sweep)
        self._otel_await = None  # standby / round-open wait span (root; makes a spare visible)
        self._otel_launch_start = None  # batch launch_script_start, for the cold-start span
        self._otel_slurm_job_start = None  # Slurm job-start, for the pre_startup span
        self._otel_cycle_start_ns = None  # current cycle span start (this round's rendezvous start)
        self._otel_cold_done = False  # cold_start + pre_startup emitted (once per node)
        self._otel_outcome = None  # outcome the agent staged for the current cycle
        self._otel_extra = {}  # extra attrs the agent staged for the current cycle

    # Per-cycle event order (each node runs its own rendezvous, so every node sees these):
    #   RENDEZVOUS_STARTED -> HEALTH_CHECK_COMPLETED -> RENDEZVOUS_COMPLETED -> WORKER_START_STARTED
    #   -> WORKER_START_COMPLETED -> (training) -> FAILURE_DETECTED -> WORKER_TERMINATED.
    # An evicted node bails right after HEALTH_CHECK_COMPLETED (UnhealthyNodeException in the
    # health check), so the exclusion catch also emits NODE_EXCLUDED. Phases are a sweep: one open
    # phase at a time, each boundary ending the prior phase and starting the next, so nothing
    # leaks open even on the partial evicted-node sequence, and every phase is a child of the
    # cycle span (one trace per cycle).
    #   value -> list of (action, arg): 'cycle_open' | 'cycle_close'(outcome) | 'phase'(name)
    #            | 'end' | 'mark'(name) | 'attr_open' | 'attr_close'
    _OTEL_SEQ = {
        'rendezvous_started': [('cycle_open', None), ('phase', 'health_check')],
        'health_check_completed': [('phase', 'rendezvous')],
        'rendezvous_completed': [('end', None)],
        'worker_start_started': [('phase', 'worker_launch')],
        'worker_start_completed': [('phase', 'run')],
        'failure_detected': [('mark', 'fault'), ('phase', 'teardown')],
        'worker_terminated': [('end', None), ('cycle_close', 'completed')],
        'node_excluded': [('end', None), ('mark', 'excluded'), ('cycle_close', 'excluded')],
        'attribution_get_started': [('attr_open', None)],
        'attribution_get_completed': [('attr_close', None)],
        # standby / round-open wait: a hot spare (or late node) not in the active rendezvous sits
        # here for the whole cycle. Emitted as a root span so the node shows up even though it
        # never joined the round; long for a spare, short for an active node just joining.
        'await_round_started': [('await_open', None)],
        'await_round_completed': [('await_close', None)],
    }

    # Events after which this node may be killed, so its spans must be force-flushed
    # synchronously before the kill; routine events rely on BatchSpanProcessor's normal export.
    _OTEL_FLUSH_EVENTS = frozenset({'failure_detected', 'worker_terminated', 'node_excluded'})

    def attach_otel(self, tracer, flush=None):
        """Register the ft_launcher agent's nemo-lens tracer so profiling events become a
        per-cycle span tree. Called once, from the agent process, after setup_telemetry()."""
        self._otel_tracer = tracer
        self._otel_flush = flush

    def otel_set_launch_start(self, ts):
        """Batch-script launch_script_start, captured outside the single srun. Used once, to emit
        the nvrx.cold_start span spanning launch_script_start to this node's first nvrx event."""
        self._otel_launch_start = ts

    def otel_set_slurm_job_start(self, ts):
        """Slurm's own job-start time (SLURM_JOB_START_TIME). Used once, to emit the pre_startup
        span spanning the SLURM job-start -> launch_script_start queue/prolog gap. Under NVRx the
        agent owns this window (megatron suppresses its own pre_startup when ft_launcher is present),
        so the whole pre-Python scheduling envelope has a single owner."""
        self._otel_slurm_job_start = ts

    def otel_cycle_start_seconds(self):
        """Wall-clock start of the current cycle span, meaning this round's rendezvous start,
        or None. The agent stamps it as NVRX_CYCLE_START_TIME on a restart or a promotion, so
        pre_startup anchors here.
        """
        return None if self._otel_cycle_start_ns is None else self._otel_cycle_start_ns / 1e9

    def otel_annotate_cycle(self, **attrs):
        """Agent enriches the currently-open cycle span with restart-budget/rendezvous metadata
        (the recorder opens the cycle at rendezvous; the agent is who knows the budget)."""
        with self._otel_lock:
            sp = self._otel_cycle_span
            if sp is None:
                return
            try:
                for k, v in attrs.items():
                    if v is not None:
                        sp.set_attribute(k, v)
            except Exception:
                pass

    def otel_stage_outcome(self, outcome, **attrs):
        """Agent stages the cycle outcome (failed/peer_restart) before teardown; the recorder
        stamps it on the cycle span when WORKER_TERMINATED closes the cycle."""
        self._otel_outcome = outcome
        self._otel_extra = attrs or {}

    def otel_finish_cycle(self, outcome='completed', **attrs):
        """Close the cycle on a terminal state with no teardown event (success/terminated)."""
        with self._otel_lock:
            self._otel_outcome = outcome
            if attrs:
                self._otel_extra = attrs
            self._otel_cycle_close(None, outcome)
            if self._otel_flush is not None:
                try:
                    self._otel_flush()
                except Exception:
                    pass

    def _otel_span(self, name, ns, node_id_str, rank=None, parent=True):
        """Start a span, parented to the current cycle span when one is open. parent=False forces
        a root span, as used by the standby wait, which happens before any cycle opens."""
        tr = self._otel_tracer
        ctx = self._otel_cycle_ctx if parent else None
        # Set at creation rather than after: attributes passed to start_span are visible to the
        # sampler and to any SpanProcessor.on_start, and it is one call instead of five.
        attrs = {
            'is_goodput_span': True,  # every restart span is resiliency overhead
            'nvrx.cycle': self._current_cycle,
        }
        if node_id_str is not None:
            attrs['nvrx.node'] = node_id_str
        if rank is not None:
            attrs['nvrx.rank'] = rank
        try:
            return tr.start_span(name, start_time=ns, context=ctx, attributes=attrs)
        except TypeError:
            return tr.start_span(name, start_time=ns, attributes=attrs)

    def _otel_cycle_open(self, ns, node_id_str):
        if self._otel_cycle_span is not None:
            return  # cycle already open
        self._otel_outcome = None
        self._otel_extra = {}
        sp = self._otel_span('nvrx.restart.cycle', ns, node_id_str)
        self._otel_cycle_span = sp
        self._otel_cycle_start_ns = ns
        try:
            from opentelemetry import trace as _t

            self._otel_cycle_ctx = _t.set_span_in_context(sp)
        except Exception:
            self._otel_cycle_ctx = None

    def _otel_cycle_close(self, ns, outcome):
        _ns = ns if ns is not None else int(time.time() * 1e9)
        # Terminal/signal safety net: an unended span is an unexported span, so on a
        # SIGTERM-driven close, sweep up any still-open await (standby round-wait) or
        # attribution span too; otherwise a spare killed while waiting, or a node killed
        # mid-attribution, leaks that span. No-op on a normal close (already None).
        if self._otel_await is not None:
            try:
                self._otel_await.end(end_time=_ns)
            except Exception:
                pass
            self._otel_await = None
        if self._otel_attr is not None:
            try:
                self._otel_attr.end(end_time=_ns)
            except Exception:
                pass
            self._otel_attr = None
        self._otel_end_phase(ns)
        sp = self._otel_cycle_span
        if sp is not None:
            try:
                sp.set_attribute('nvrx.cycle_outcome', self._otel_outcome or outcome or 'completed')
                for k, v in (self._otel_extra or {}).items():
                    if v is not None:
                        sp.set_attribute(k, v)
                sp.end(end_time=_ns)
            except Exception:
                pass
        self._otel_cycle_span = None
        self._otel_cycle_ctx = None
        self._otel_cycle_start_ns = None  # cleared so a closed cycle can't leak its start timestamp
        self._otel_outcome = None
        self._otel_extra = {}

    def _otel_start_phase(self, name, ns, node_id_str, rank):
        self._otel_end_phase(ns)  # sweep: close the prior phase before opening the next
        sp = self._otel_span('nvrx.restart.' + name, ns, node_id_str, rank)
        self._otel_phase = (name, sp)

    def _otel_end_phase(self, ns):
        if self._otel_phase is not None:
            _, sp = self._otel_phase
            self._otel_phase = None
            try:
                sp.end(end_time=ns if ns is not None else int(time.time() * 1e9))
            except Exception:
                pass

    def _otel_mark(self, name, ns, node_id_str):
        sp = self._otel_span('nvrx.restart.' + name, ns, node_id_str)
        try:
            sp.end(end_time=ns)
        except Exception:
            pass

    def _otel_on_event(self, event, timestamp, node_id_str, rank):
        """Drive the per-cycle span tree from this boundary event, flushing the kill-adjacent
        events immediately so an evicted node's spans survive its kill."""
        if self._otel_tracer is None:
            return
        node_id_str = _clean_node(node_id_str)
        ns = int(timestamp * 1e9)
        self._otel_lock.acquire()
        try:
            # One-time "outside srun to first nvrx" cold start: batch launch_script_start up to
            # this agent's first recorded event. Backdated, closed immediately, once per node.
            if not self._otel_cold_done and self._otel_launch_start is not None:
                self._otel_cold_done = True
                # pre_startup (SLURM job-start -> launch_script_start): the queue/prolog gap before
                # the launch script's first line. The agent owns it under NVRx (megatron suppresses
                # its own on any ft_launcher cohort), so the pre-Python envelope has one owner. Only
                # when Slurm's stamp precedes launch_start; name matches the bare-path span so the
                # goodput taxonomy classifies it identically (BAD;scheduling;pre_startup).
                if (
                    self._otel_slurm_job_start is not None
                    and self._otel_slurm_job_start < self._otel_launch_start
                ):
                    ps = self._otel_tracer.start_span(
                        'pre_startup',
                        start_time=int(self._otel_slurm_job_start * 1e9),
                        attributes={
                            'is_goodput_span': True,
                            'nvrx.node': node_id_str,
                        },
                    )
                    ps.end(end_time=int(self._otel_launch_start * 1e9))
                cs = self._otel_tracer.start_span(
                    'nvrx.cold_start',
                    start_time=int(self._otel_launch_start * 1e9),
                    attributes={
                        'is_goodput_span': True,
                        'nvrx.node': node_id_str,
                    },
                )
                cs.end(end_time=ns)
            for action, arg in self._OTEL_SEQ.get(event.value, ()):
                if action == 'cycle_open':
                    self._otel_cycle_open(ns, node_id_str)
                elif action == 'cycle_close':
                    self._otel_cycle_close(ns, arg)
                elif action == 'phase':
                    self._otel_start_phase(arg, ns, node_id_str, rank)
                    # Launching a worker means this node was selected active this
                    # cycle, so stamp the cycle span at selection time; a node killed
                    # after selection but before close then still shows membership=active
                    # (complements cycle_outcome: standby=not selected, excluded=evicted).
                    if arg == 'worker_launch' and self._otel_cycle_span is not None:
                        try:
                            self._otel_cycle_span.set_attribute('nvrx.membership', 'active')
                        except Exception:
                            pass
                elif action == 'end':
                    self._otel_end_phase(ns)
                elif action == 'mark':
                    self._otel_mark(arg, ns, node_id_str)
                elif action == 'attr_open':
                    self._otel_attr = self._otel_span(
                        'nvrx.restart.attribution', ns, node_id_str, rank
                    )
                elif action == 'attr_close':
                    if self._otel_attr is not None:
                        try:
                            self._otel_attr.end(end_time=ns)
                        except Exception:
                            pass
                        self._otel_attr = None
                elif action == 'await_open':
                    # A standby or looping node re-enters the Step-0 wait still holding the prior
                    # round's open cycle and phase (it started that round's rendezvous but was not
                    # selected, so no WORKER_TERMINATED closed it). Close the whole cycle, not just
                    # the phase, so the next RENDEZVOUS_STARTED opens a fresh cycle with the right
                    # start timestamp. Otherwise _otel_cycle_open's "already open" guard keeps the
                    # stale cycle, and a promoted spare reuses the earlier round's cycle start.
                    # No-op for a normal node: its cycle is already closed before it loops.
                    self._otel_cycle_close(
                        ns, 'standby'
                    )  # also ends the open phase (via _otel_end_phase)
                    self._otel_await = self._otel_span(
                        'nvrx.restart.await_round', ns, node_id_str, parent=False
                    )  # root: precedes any cycle
                    # The await span's whole lifetime is the unselected/standby wait: it
                    # opens at the Step-0 wait and closes when the round opens and the
                    # node proceeds. Tag it so a spare is distinguishable by membership.
                    try:
                        self._otel_await.set_attribute('nvrx.membership', 'standby')
                    except Exception:
                        pass
                elif action == 'await_close':
                    if self._otel_await is not None:
                        try:
                            self._otel_await.end(end_time=ns)
                        except Exception:
                            pass
                        self._otel_await = None
            if self._otel_flush is not None and event.value in self._OTEL_FLUSH_EVENTS:
                self._otel_flush()
        except (_SignalException, KeyboardInterrupt, SystemExit):
            raise  # never swallow the FT death signal (torchelastic raises it here)
        except Exception:
            pass  # telemetry must never break the launcher
        finally:
            self._otel_lock.release()

    def _timestamp_to_utc_datetime(self, timestamp: float) -> str:
        """Convert timestamp to UTC datetime string."""
        utc_datetime = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return utc_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ]  # Remove last 3 digits for milliseconds

    def set_cycle(self, cycle: int) -> None:
        """Set the current cycle number.

        Called by the rendezvous handler when a newly joining node syncs its cycle number
        via _sync_from_per_round_state() which scans per-round round_done keys. This ensures
        newly joining nodes (e.g., replacement array tasks) continue with the correct cycle
        number instead of starting from 0.

        Args:
            cycle: The cycle number to set. Only sets if >= current cycle to prevent backward jumps.
        """
        if cycle >= self._current_cycle:
            self._current_cycle = cycle
        else:
            self._logger.warning(
                f"Attempted to set profiler cycle to {cycle}, which is less than "
                f"current cycle {self._current_cycle}. Ignoring to prevent backward cycle jumps."
            )

    def record_event(
        self,
        event: ProfilingEvent,
        node_id: Optional[Any] = None,
        rank: Optional[int] = None,
    ) -> str:
        """Record a profiling event and return a unique event ID."""
        timestamp = time.time()
        # Convert node_id to string for event ID and logging
        node_id_str = str(node_id) if node_id is not None else 'unknown'
        event_id = f"{event.value}_{timestamp}_{node_id_str}_{rank or 'unknown'}"

        # Turn this pre-placed boundary event into a nemo-lens phase span (rendezvous,
        # health_check, worker_launch, run, teardown, attribution), so that an evicted node's
        # restart sequence survives its kill. No-op unless the agent attached a tracer.
        self._otel_on_event(event, timestamp, node_id_str, rank)

        # Increment cycle count for failure detection events
        if event == ProfilingEvent.FAILURE_DETECTED:
            self._current_cycle += 1

        # Format log message with cycle count and UTC time
        utc_time = self._timestamp_to_utc_datetime(timestamp)
        self._logger.info(
            f"  - Cycle: {self._current_cycle} Event: {event.value} Node: {node_id_str} Rank: {rank} "
            f"Time: {utc_time} UTC"
        )
        return event_id


# Global profiler instance (lazy-initialized to avoid stdout output at import time)
_global_profiler: Optional[FaultToleranceProfiler] = None
_global_profiler_lock = threading.Lock()


def _get_global_profiler() -> FaultToleranceProfiler:
    """Get or create the global profiler instance (thread-safe)."""
    global _global_profiler
    if _global_profiler is None:
        with _global_profiler_lock:
            # Double-check pattern to avoid race conditions
            if _global_profiler is None:
                _global_profiler = FaultToleranceProfiler()
    return _global_profiler


def record_profiling_event(
    event: ProfilingEvent,
    node_id: Optional[Any] = None,
    rank: Optional[int] = None,
) -> str:
    """Convenience function to record a profiling event.

    Args:
        event: The profiling event to record
        node_id: Node identifier (can be any type, will be converted to string)
        rank: Rank identifier

    Returns:
        Event ID string
    """
    return _get_global_profiler().record_event(event, node_id, rank)


def get_profiling_cycle() -> int:
    """Return the current profiling cycle number."""
    return _get_global_profiler()._current_cycle


def set_profiling_cycle(cycle: int) -> None:
    """Set the current cycle number in the global profiler.

    Called by the rendezvous handler when a newly joining node syncs its cycle number
    from the global_cycle_key in the store. This ensures newly joining nodes (e.g.,
    replacement array tasks) continue with the correct cycle number instead of starting from 0.

    Args:
        cycle: The cycle number to set. Only sets if >= current cycle to prevent backward jumps.
    """
    _get_global_profiler().set_cycle(cycle)
