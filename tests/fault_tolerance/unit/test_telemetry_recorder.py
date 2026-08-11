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

"""Unit tests for the restart-cycle telemetry recorder.

These run without opentelemetry or nemo-lens installed: the recorder only needs an
object with a `start_span()`, so a fake tracer records exactly what it was asked for.
"""

from unittest.mock import MagicMock

import pytest

from nvidia_resiliency_ext.fault_tolerance.telemetry import (
    ColdStartAnchors,
    CycleOutcome,
    LauncherTelemetry,
    RestartCycleRecorder,
)
from nvidia_resiliency_ext.shared_utils.profiling import (
    FaultToleranceProfiler,
    ProfilingEvent,
    ProfilingEventRecord,
    profiling_phase,
    set_profiling_observer,
)
from nvidia_resiliency_ext.shared_utils.telemetry import HAS_NEMO_LENS


class FakeSpan:
    def __init__(self, name, start_time, attributes):
        self.name = name
        self.start_time = start_time
        self.end_time = None
        self.attributes = dict(attributes)

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def end(self, end_time=None):
        assert self.end_time is None, f"span {self.name} ended twice"
        self.end_time = end_time

    @property
    def ended(self):
        return self.end_time is not None


class FakeTracer:
    def __init__(self):
        self.spans = []

    def start_span(self, name, start_time=None, context=None, attributes=None):
        span = FakeSpan(name, start_time, attributes or {})
        self.spans.append(span)
        return span

    def named(self, name):
        return [s for s in self.spans if s.name == name]

    def one(self, name):
        matches = self.named(name)
        assert len(matches) == 1, f"expected exactly one {name}, got {len(matches)}"
        return matches[0]

    @property
    def names(self):
        return [s.name for s in self.spans]


@pytest.fixture
def tracer():
    return FakeTracer()


@pytest.fixture
def flushes():
    return []


@pytest.fixture
def recorder(tracer, flushes):
    return RestartCycleRecorder(tracer, flush=lambda: flushes.append(True))


def feed(recorder, event, timestamp, node='host_1234_0', rank=None, cycle=0):
    recorder.on_event(
        ProfilingEventRecord(event=event, timestamp=timestamp, node_id=node, rank=rank, cycle=cycle)
    )


HEALTHY_CYCLE = [
    (ProfilingEvent.RENDEZVOUS_STARTED, 100.0),
    (ProfilingEvent.HEALTH_CHECK_COMPLETED, 101.0),
    (ProfilingEvent.RENDEZVOUS_COMPLETED, 102.0),
    (ProfilingEvent.WORKER_START_STARTED, 103.0),
    (ProfilingEvent.WORKER_START_COMPLETED, 104.0),
    (ProfilingEvent.FAILURE_DETECTED, 200.0),
    (ProfilingEvent.WORKER_TERMINATED, 201.0),
]


def run_cycle(recorder, events=HEALTHY_CYCLE, **kwargs):
    for event, timestamp in events:
        feed(recorder, event, timestamp, **kwargs)


class TestCycleSpanTree:
    def test_full_cycle_emits_the_expected_phases(self, recorder, tracer):
        run_cycle(recorder)
        assert tracer.names == [
            'nvrx.restart.cycle',
            'nvrx.restart.health_check',
            'nvrx.restart.rendezvous',
            'nvrx.restart.worker_launch',
            'nvrx.restart.run',
            'nvrx.restart.fault',
            'nvrx.restart.teardown',
        ]
        assert all(span.ended for span in tracer.spans)

    def test_phases_are_a_contiguous_sweep(self, recorder, tracer):
        run_cycle(recorder)
        health = tracer.one('nvrx.restart.health_check')
        rendezvous = tracer.one('nvrx.restart.rendezvous')
        assert health.start_time == int(100.0 * 1e9)
        # each phase ends exactly where the next begins
        assert health.end_time == rendezvous.start_time == int(101.0 * 1e9)
        assert rendezvous.end_time == int(102.0 * 1e9)
        assert tracer.one('nvrx.restart.run').end_time == int(200.0 * 1e9)

    def test_cycle_span_bounds_and_attributes(self, recorder, tracer):
        run_cycle(recorder, cycle=4)
        cycle = tracer.one('nvrx.restart.cycle')
        assert (cycle.start_time, cycle.end_time) == (int(100.0 * 1e9), int(201.0 * 1e9))
        assert cycle.attributes['nvrx.cycle'] == 4
        assert cycle.attributes['is_goodput_span'] is True
        assert cycle.attributes['nvrx.cycle_outcome'] == CycleOutcome.COMPLETED
        # launching workers marks this node as selected for the cycle
        assert cycle.attributes['nvrx.membership'] == 'active'

    def test_node_desc_suffix_is_stripped(self, recorder, tracer):
        run_cycle(recorder, node='host.cm.cluster_98765_0')
        assert tracer.one('nvrx.restart.cycle').attributes['nvrx.node'] == 'host.cm.cluster'

    def test_rank_is_recorded_on_phases_only(self, recorder, tracer):
        run_cycle(recorder, rank=3)
        assert tracer.one('nvrx.restart.run').attributes['nvrx.rank'] == 3
        assert 'nvrx.rank' not in tracer.one('nvrx.restart.cycle').attributes

    def test_a_second_rendezvous_opens_a_second_cycle(self, recorder, tracer):
        run_cycle(recorder)
        feed(recorder, ProfilingEvent.RENDEZVOUS_STARTED, 300.0, cycle=1)
        cycles = tracer.named('nvrx.restart.cycle')
        assert len(cycles) == 2
        assert cycles[1].start_time == int(300.0 * 1e9)
        assert not cycles[1].ended

    def test_repeated_rendezvous_does_not_reopen_a_cycle(self, recorder, tracer):
        feed(recorder, ProfilingEvent.RENDEZVOUS_STARTED, 100.0)
        feed(recorder, ProfilingEvent.RENDEZVOUS_STARTED, 105.0)
        assert len(tracer.named('nvrx.restart.cycle')) == 1


class TestOutcomes:
    def test_staged_outcome_wins_over_the_default(self, recorder, tracer):
        run_cycle(recorder, events=HEALTHY_CYCLE[:-1])
        recorder.stage_outcome(CycleOutcome.FAILED, **{'nvrx.failures': 2})
        feed(recorder, ProfilingEvent.WORKER_TERMINATED, 201.0)
        cycle = tracer.one('nvrx.restart.cycle')
        assert cycle.attributes['nvrx.cycle_outcome'] == CycleOutcome.FAILED
        assert cycle.attributes['nvrx.failures'] == 2

    def test_staged_outcome_does_not_leak_into_the_next_cycle(self, recorder, tracer):
        run_cycle(recorder, events=HEALTHY_CYCLE[:-1])
        recorder.stage_outcome(CycleOutcome.FAILED)
        feed(recorder, ProfilingEvent.WORKER_TERMINATED, 201.0)
        run_cycle(recorder, events=[(e, t + 200.0) for e, t in HEALTHY_CYCLE])
        second = tracer.named('nvrx.restart.cycle')[1]
        assert second.attributes['nvrx.cycle_outcome'] == CycleOutcome.COMPLETED

    def test_finish_cycle_closes_a_cycle_with_no_teardown_event(self, recorder, tracer):
        run_cycle(recorder, events=HEALTHY_CYCLE[:5])
        recorder.finish_cycle(CycleOutcome.SUCCEEDED)
        cycle = tracer.one('nvrx.restart.cycle')
        assert cycle.ended
        assert cycle.attributes['nvrx.cycle_outcome'] == CycleOutcome.SUCCEEDED
        # the open 'run' phase is swept up with it
        assert tracer.one('nvrx.restart.run').ended

    def test_finish_cycle_without_an_open_cycle_is_harmless(self, recorder, tracer):
        recorder.finish_cycle(CycleOutcome.TERMINATED)
        assert tracer.spans == []


class TestExcludedNode:
    def test_eviction_closes_the_partial_cycle(self, recorder, tracer):
        feed(recorder, ProfilingEvent.RENDEZVOUS_STARTED, 100.0)
        feed(recorder, ProfilingEvent.HEALTH_CHECK_COMPLETED, 101.0)
        feed(recorder, ProfilingEvent.NODE_EXCLUDED, 102.0)
        assert 'nvrx.restart.excluded' in tracer.names
        assert all(span.ended for span in tracer.spans)
        cycle = tracer.one('nvrx.restart.cycle')
        assert cycle.attributes['nvrx.cycle_outcome'] == CycleOutcome.EXCLUDED
        assert cycle.end_time == int(102.0 * 1e9)

    def test_kill_adjacent_events_flush(self, recorder, flushes):
        run_cycle(recorder)
        # FAILURE_DETECTED and WORKER_TERMINATED both flush; routine boundaries do not
        assert len(flushes) == 2

    def test_routine_events_do_not_flush(self, recorder, flushes):
        run_cycle(recorder, events=HEALTHY_CYCLE[:5])
        assert flushes == []


class TestStandbyNode:
    def test_await_span_is_emitted_for_a_waiting_node(self, recorder, tracer):
        feed(recorder, ProfilingEvent.AWAIT_ROUND_STARTED, 50.0)
        feed(recorder, ProfilingEvent.AWAIT_ROUND_COMPLETED, 90.0)
        await_span = tracer.one('nvrx.restart.await_round')
        assert (await_span.start_time, await_span.end_time) == (int(50.0 * 1e9), int(90.0 * 1e9))
        assert await_span.attributes['nvrx.membership'] == 'standby'

    def test_reentering_the_wait_closes_the_unselected_cycle(self, recorder, tracer):
        # this node started a round but was never selected, so nothing closed its cycle
        feed(recorder, ProfilingEvent.RENDEZVOUS_STARTED, 100.0)
        feed(recorder, ProfilingEvent.HEALTH_CHECK_COMPLETED, 101.0)
        feed(recorder, ProfilingEvent.AWAIT_ROUND_STARTED, 150.0)
        cycle = tracer.one('nvrx.restart.cycle')
        assert cycle.ended and cycle.attributes['nvrx.cycle_outcome'] == CycleOutcome.STANDBY
        assert tracer.one('nvrx.restart.rendezvous').ended
        # and the next round gets a fresh cycle, not the stale one
        feed(recorder, ProfilingEvent.AWAIT_ROUND_COMPLETED, 160.0)
        feed(recorder, ProfilingEvent.RENDEZVOUS_STARTED, 161.0)
        assert tracer.named('nvrx.restart.cycle')[1].start_time == int(161.0 * 1e9)

    def test_an_open_await_is_swept_up_by_a_cycle_close(self, recorder, tracer):
        feed(recorder, ProfilingEvent.AWAIT_ROUND_STARTED, 50.0)
        feed(recorder, ProfilingEvent.RENDEZVOUS_STARTED, 60.0)
        recorder.finish_cycle(CycleOutcome.TERMINATED)
        assert tracer.one('nvrx.restart.await_round').ended


class TestAttribution:
    def test_attribution_span_overlaps_the_phase_sweep(self, recorder, tracer):
        feed(recorder, ProfilingEvent.RENDEZVOUS_STARTED, 100.0)
        feed(recorder, ProfilingEvent.ATTRIBUTION_GET_STARTED, 110.0, rank=1)
        feed(recorder, ProfilingEvent.HEALTH_CHECK_COMPLETED, 115.0)
        feed(recorder, ProfilingEvent.ATTRIBUTION_GET_COMPLETED, 120.0)
        attribution = tracer.one('nvrx.restart.attribution')
        assert (attribution.start_time, attribution.end_time) == (
            int(110.0 * 1e9),
            int(120.0 * 1e9),
        )
        assert attribution.attributes['nvrx.rank'] == 1

    def test_an_open_attribution_is_swept_up_by_a_cycle_close(self, recorder, tracer):
        feed(recorder, ProfilingEvent.RENDEZVOUS_STARTED, 100.0)
        feed(recorder, ProfilingEvent.ATTRIBUTION_GET_STARTED, 110.0)
        feed(recorder, ProfilingEvent.NODE_EXCLUDED, 120.0)
        assert tracer.one('nvrx.restart.attribution').end_time == int(120.0 * 1e9)


class TestColdStart:
    def test_cold_start_spans_are_backdated_once(self, tracer):
        recorder = RestartCycleRecorder(
            tracer, cold_start=ColdStartAnchors(launch_script_start=40.0, slurm_job_start=10.0)
        )
        run_cycle(recorder)
        pre_startup = tracer.one('pre_startup')
        cold_start = tracer.one('nvrx.cold_start')
        assert (pre_startup.start_time, pre_startup.end_time) == (int(10.0 * 1e9), int(40.0 * 1e9))
        assert (cold_start.start_time, cold_start.end_time) == (int(40.0 * 1e9), int(100.0 * 1e9))
        # a second cycle does not re-emit them
        feed(recorder, ProfilingEvent.RENDEZVOUS_STARTED, 300.0)
        assert len(tracer.named('nvrx.cold_start')) == 1

    def test_pre_startup_needs_a_job_start_that_precedes_the_script(self, tracer):
        recorder = RestartCycleRecorder(
            tracer, cold_start=ColdStartAnchors(launch_script_start=40.0, slurm_job_start=50.0)
        )
        run_cycle(recorder)
        assert tracer.named('pre_startup') == []
        assert tracer.named('nvrx.cold_start') != []

    def test_no_anchors_means_no_cold_start_spans(self, recorder, tracer):
        run_cycle(recorder)
        assert tracer.named('nvrx.cold_start') == []

    def test_anchors_from_env(self):
        anchors = ColdStartAnchors.from_env(
            {'LENS_LAUNCH_SCRIPT_START_TIME': '12.5', 'SLURM_JOB_START_TIME': 'not-a-number'}
        )
        assert anchors.launch_script_start == 12.5
        assert anchors.slurm_job_start is None


class TestCycleStartSeconds:
    def test_reports_the_open_cycle_start(self, recorder):
        assert recorder.cycle_start_seconds() is None
        feed(recorder, ProfilingEvent.RENDEZVOUS_STARTED, 100.0)
        assert recorder.cycle_start_seconds() == pytest.approx(100.0)

    def test_cleared_once_the_cycle_closes(self, recorder):
        run_cycle(recorder)
        assert recorder.cycle_start_seconds() is None


class TestRobustness:
    def test_a_broken_tracer_does_not_break_the_recorder(self):
        class BrokenTracer:
            def start_span(self, *args, **kwargs):
                raise RuntimeError("collector exploded")

        recorder = RestartCycleRecorder(BrokenTracer())
        run_cycle(recorder)  # must not raise

    def test_a_broken_flush_does_not_break_the_recorder(self, tracer):
        def flush():
            raise RuntimeError("collector unreachable")

        run_cycle(RestartCycleRecorder(tracer, flush=flush))  # must not raise


class TestLauncherTelemetry:
    """The agent-facing facade. Without NEMO_LENS_EXPORTER it must be a working no-op."""

    def test_disabled_by_default(self):
        telemetry = LauncherTelemetry('host-a', env={})
        assert type(telemetry) is LauncherTelemetry

    def test_disabled_facade_accepts_every_call(self):
        telemetry = LauncherTelemetry('host-a', env={})
        telemetry.annotate_cycle(object(), 3)
        telemetry.stage_outcome(CycleOutcome.FAILED, **{'nvrx.failures': 1})
        telemetry.finish_cycle(CycleOutcome.SUCCEEDED)
        telemetry.shutdown()

    @pytest.mark.skipif(HAS_NEMO_LENS, reason="needs a backend that cannot be set up")
    def test_setup_failure_degrades_to_the_no_op(self):
        # NEMO_LENS_EXPORTER is set but nemo-lens is absent, so construction fails and must
        # fall back to the no-op rather than propagate out of the agent's constructor.
        telemetry = LauncherTelemetry('host-a', env={'NEMO_LENS_EXPORTER': 'console'})
        assert type(telemetry) is LauncherTelemetry

    def test_failed_setup_shuts_the_provider_down(self, monkeypatch):
        # setup_telemetry() succeeds, then something after it throws: the provider must be
        # shut down rather than orphaned with its exporter thread alive.
        shutdowns = []

        class Handle:
            @property
            def tracer(self):
                raise RuntimeError("tracer unavailable")

            def shutdown(self):
                shutdowns.append(True)

        import nvidia_resiliency_ext.fault_tolerance.telemetry as ft_telemetry

        monkeypatch.setattr(ft_telemetry.backend, 'HAS_NEMO_LENS', True)
        monkeypatch.setattr(ft_telemetry.backend, 'config_from_env', lambda *a, **k: MagicMock())
        monkeypatch.setattr(ft_telemetry.backend, 'setup_telemetry', lambda *a, **k: Handle())

        telemetry = LauncherTelemetry('host-a', env={'NEMO_LENS_EXPORTER': 'console'})

        assert shutdowns == [True]
        assert telemetry._handle is None and telemetry._recorder is None
        telemetry.shutdown()  # must stay harmless afterwards

    def test_worker_env_is_emitted_even_when_disabled(self):
        # the trainer, not the agent, consumes these, so they must not depend on the
        # agent's own telemetry being on
        env = LauncherTelemetry('host-a', env={}).worker_env(0)
        assert env['NVRX_CYCLE'] == '0'
        assert env['NVRX_MEMBERSHIP'] == 'active'
        assert float(env['NVRX_LAUNCH_TIME']) > 0
        # cycle 0 anchors pre_startup on the real sbatch queue time instead
        assert 'NVRX_CYCLE_START_TIME' not in env

    def test_worker_env_carries_the_cycle_start_after_a_restart(self, tracer):
        # a restarted cohort anchors pre_startup on this round's rendezvous start, which
        # is the recorder's open cycle span
        telemetry = LauncherTelemetry('host-a', env={})
        telemetry._recorder = RestartCycleRecorder(tracer)
        feed(telemetry._recorder, ProfilingEvent.RENDEZVOUS_STARTED, 1234.5)

        env = telemetry.worker_env(2)
        assert env['NVRX_CYCLE'] == '2'
        assert float(env['NVRX_CYCLE_START_TIME']) == pytest.approx(1234.5)

    def test_worker_env_omits_cycle_start_when_no_cycle_is_open(self, tracer):
        telemetry = LauncherTelemetry('host-a', env={})
        telemetry._recorder = RestartCycleRecorder(tracer)
        assert 'NVRX_CYCLE_START_TIME' not in telemetry.worker_env(2)


class TestProfilerWiring:
    def test_listeners_see_events_with_the_pre_increment_cycle(self):
        seen = []

        class Listener:
            def on_event(self, record):
                seen.append((record.event, record.cycle))

        profiler = FaultToleranceProfiler()
        listener = Listener()
        profiler._observer = listener
        profiler.record_event(ProfilingEvent.FAILURE_DETECTED, node_id='host')
        profiler.record_event(ProfilingEvent.WORKER_TERMINATED, node_id='host')
        # the failure is attributed to the cycle it ended, not the one it starts
        assert seen == [(ProfilingEvent.FAILURE_DETECTED, 0), (ProfilingEvent.WORKER_TERMINATED, 1)]

        profiler._observer = None
        profiler.record_event(ProfilingEvent.RENDEZVOUS_STARTED, node_id='host')
        assert len(seen) == 2

    def test_profiling_phase_closes_even_when_the_body_raises(self):
        seen = []

        class Listener:
            def on_event(self, record):
                seen.append(record.event)

        listener = Listener()
        set_profiling_observer(listener)
        try:
            with pytest.raises(RuntimeError):
                with profiling_phase(
                    ProfilingEvent.AWAIT_ROUND_STARTED,
                    ProfilingEvent.AWAIT_ROUND_COMPLETED,
                    node_id='host',
                ):
                    raise RuntimeError("shutdown unwound out of the wait")
        finally:
            set_profiling_observer(None)
        assert seen == [
            ProfilingEvent.AWAIT_ROUND_STARTED,
            ProfilingEvent.AWAIT_ROUND_COMPLETED,
        ]

    def test_recorder_consumes_profiler_events_end_to_end(self, tracer):
        profiler = FaultToleranceProfiler()
        profiler._observer = RestartCycleRecorder(tracer)
        profiler.record_event(ProfilingEvent.RENDEZVOUS_STARTED, node_id='host_1_0')
        profiler.record_event(ProfilingEvent.WORKER_START_STARTED, node_id='host_1_0', rank=2)
        assert tracer.names == [
            'nvrx.restart.cycle',
            'nvrx.restart.health_check',
            'nvrx.restart.worker_launch',
        ]
        assert tracer.one('nvrx.restart.worker_launch').attributes['nvrx.rank'] == 2
