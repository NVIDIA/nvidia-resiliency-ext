# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the optional nemo-lens telemetry shim.

These cover the contract NVRx depends on: instrumentation is inert when
nemo-lens is missing or uninitialized, and never propagates a telemetry
failure into the workload. They run with or without nemo-lens installed.
"""

import ast
import pathlib
import threading
import time
import unittest
import unittest.mock

from nvidia_resiliency_ext.shared_utils import telemetry


@unittest.skipUnless(telemetry._AVAILABLE, "requires nemo-lens")
class TestLifecycleSnapshots(unittest.TestCase):
    def setUp(self):
        from nemo.lens import NemoLensConfig
        from nemo.lens.providers import build_providers
        from nemo.lens.state import enabled_span_groups, set_enabled_span_groups
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        self.addCleanup(set_enabled_span_groups, enabled_span_groups())
        set_enabled_span_groups(frozenset({"nvrx.ft"}))
        self.exporter = InMemorySpanExporter()
        with unittest.mock.patch("opentelemetry.trace.set_tracer_provider") as install:
            build_providers(
                NemoLensConfig(enabled=True, metrics_enabled=False),
                span_exporter=self.exporter,
            )
        self.provider = install.call_args.args[0]
        self.addCleanup(self.provider.shutdown)
        self.enterContext(
            unittest.mock.patch.object(telemetry, "_get_tracer", self.provider.get_tracer)
        )
        self.enterContext(
            unittest.mock.patch("opentelemetry.trace.get_tracer", self.provider.get_tracer)
        )

    def _record_group(self, cycle_attributes, rendezvous_attributes):
        phase = telemetry.Phase()
        self.addCleanup(phase.close)
        phase.open("nvrx.ft", "cycle", cycle_attributes)
        with telemetry.span("nvrx.ft", "await_round", cycle_attributes):
            pass
        rendezvous = telemetry.ManualSpan()
        rendezvous.open(
            "nvrx.ft",
            "rendezvous",
            rendezvous_attributes,
            inherit_attributes=True,
        )
        with telemetry.span("nvrx.ft", "health_check"):
            pass
        rendezvous.close()
        phase.close({"nv.nvrx.cycle.outcome": "completed"})
        self.provider.force_flush()

    def test_cycle_parents_wait_and_operation_snapshots_do_not_change(self):
        first = {
            "nv.nvrx.ftl.rdzv.round": 0,
            "nv.nvrx.ftl.profiling.cycle": 0,
        }
        self._record_group(first, first)
        first_spans = {span.name: span for span in self.exporter.get_finished_spans()}

        cycle_start = first_spans["cycle_start"]
        for name in ("await_round", "rendezvous", "cycle"):
            recorded = first_spans[name]
            self.assertEqual(recorded.context.trace_id, cycle_start.context.trace_id)
            self.assertEqual(recorded.parent.span_id, cycle_start.context.span_id)
        self.assertEqual(
            first_spans["health_check"].parent.span_id,
            first_spans["rendezvous"].context.span_id,
        )
        for name in ("cycle_start", "await_round", "rendezvous", "health_check", "cycle"):
            expected = {
                **first,
                **({"nv.nvrx.cycle.outcome": "completed"} if name == "cycle" else {}),
            }
            self.assertEqual(dict(first_spans[name].attributes), expected)

        first_trace = cycle_start.context.trace_id
        self.exporter.clear()
        wait_snapshot = {
            "nv.nvrx.ftl.rdzv.round": 0,
            "nv.nvrx.ftl.profiling.cycle": 1,
        }
        rendezvous_snapshot = {
            "nv.nvrx.ftl.rdzv.round": 1,
            "nv.nvrx.ftl.profiling.cycle": 1,
        }
        self._record_group(wait_snapshot, rendezvous_snapshot)
        restarted = {span.name: span for span in self.exporter.get_finished_spans()}

        self.assertNotEqual(restarted["cycle_start"].context.trace_id, first_trace)
        self.assertEqual(
            restarted["await_round"].parent.span_id, restarted["cycle_start"].context.span_id
        )
        self.assertEqual(dict(restarted["await_round"].attributes), wait_snapshot)
        self.assertEqual(dict(restarted["cycle_start"].attributes), wait_snapshot)
        self.assertEqual(
            dict(restarted["cycle"].attributes),
            {**wait_snapshot, "nv.nvrx.cycle.outcome": "completed"},
        )
        self.assertEqual(dict(restarted["rendezvous"].attributes), rendezvous_snapshot)
        self.assertEqual(dict(restarted["health_check"].attributes), rendezvous_snapshot)


class TestTelemetryIsInert(unittest.TestCase):
    """Instrumentation must be a no-op before/without setup_telemetry()."""

    def test_phase_without_lens(self):
        with unittest.mock.patch.object(telemetry, "_AVAILABLE", False):
            phase = telemetry.Phase()
            phase.open(
                "nvrx.ft",
                "nv.nvrx.ftl.cycle",
                {
                    "nv.nvrx.ftl.rdzv.round": 3,
                    "nv.nvrx.ftl.profiling.cycle": 4,
                },
            )
            self.assertIsNone(phase._start)
            phase.close()
            handle = telemetry.setup_telemetry("nvrx.ft_launcher")
            handle.shutdown()

    def test_managed_span_yields_and_runs_body(self):
        ran = False
        with telemetry.span("nvrx.ft", "nv.nvrx.ftl.cycle") as active:
            ran = True
            self.assertIsNone(active)
        self.assertTrue(ran)

    def test_managed_span_propagates_body_exceptions(self):
        # Telemetry must never swallow a workload error -- notably SignalException,
        # which torch elastic raises out of the launcher's monitor loop.
        with self.assertRaises(ValueError):
            with telemetry.span("nvrx.ft", "nv.nvrx.ftl.cycle"):
                raise ValueError("from the instrumented body")

    def test_managed_span_accepts_attributes(self):
        with telemetry.span("nvrx.ckpt", "nv.nvrx.ckpt.save.request", {"nv.nvrx.ckpt.call_idx": 7}):
            pass

    def test_trace_fn_returns_a_working_decorator(self):
        @telemetry.trace_fn("nvrx.ft", "nv.nvrx.ftl.worker_launch")
        def start(a, b=2):
            return a + b

        self.assertEqual(start(1), 3)
        self.assertEqual(start(1, b=10), 11)

    def test_trace_fn_propagates_exceptions(self):
        @telemetry.trace_fn("nvrx.ft", "nv.nvrx.ftl.teardown")
        def boom():
            raise RuntimeError("worker teardown failed")

        with self.assertRaises(RuntimeError):
            boom()

    def test_trace_fn_does_not_evaluate_attributes_when_disabled(self):
        if not telemetry._AVAILABLE:
            self.skipTest("requires nemo-lens")
        attributes = unittest.mock.Mock(side_effect=AssertionError("attributes evaluated"))

        @telemetry.trace_fn("nvrx.ft", "disabled", attrs=attributes)
        def work(value):
            return value

        with unittest.mock.patch.object(telemetry, "_is_span_group_enabled", return_value=False):
            self.assertEqual(work(7), 7)
        attributes.assert_not_called()

    def test_trace_fn_passes_callback_attributes_to_the_span(self):
        if not telemetry._AVAILABLE:
            self.skipTest("requires nemo-lens")
        attributes = unittest.mock.Mock(return_value={"example.attribute": "value"})

        @telemetry.trace_fn("nvrx.ft", "enabled", attrs=attributes)
        def work(value, *, scale=1):
            return value * scale

        with (
            unittest.mock.patch.object(telemetry, "_is_span_group_enabled", return_value=True),
            unittest.mock.patch.object(telemetry, "_managed_span") as managed,
        ):
            self.assertEqual(work(7, scale=2), 14)
        attributes.assert_called_once_with(7, scale=2)
        managed.assert_called_once_with(
            "nvrx.ft",
            "enabled",
            unittest.mock.ANY,
            **{"example.attribute": "value"},
        )

    def test_trace_fn_callback_is_inert_without_lens(self):
        attributes = unittest.mock.Mock(side_effect=AssertionError("attributes evaluated"))

        with unittest.mock.patch.object(telemetry, "_AVAILABLE", False):

            @telemetry.trace_fn("nvrx.ft", "absent", attrs=attributes)
            def work(value):
                return value

        self.assertEqual(work(7), 7)
        attributes.assert_not_called()


class TestManualSpan(unittest.TestCase):
    """ManualSpan must tolerate every order the launcher can call it in."""

    def test_all_methods_are_safe_before_open(self):
        span = telemetry.ManualSpan()
        span.set(
            {
                "nv.nvrx.ftl.rdzv.round": 0,
                "nv.nvrx.ftl.profiling.cycle": 0,
            }
        )
        span.close({"nv.nvrx.cycle.outcome": "terminated"})
        span.close()

    def test_close_is_idempotent(self):
        span = telemetry.ManualSpan()
        span.open(
            "nvrx.ft",
            "nv.nvrx.ftl.rendezvous",
            {
                "nv.nvrx.ftl.rdzv.round": 0,
                "nv.nvrx.ftl.profiling.cycle": 0,
            },
        )
        span.close({"nv.nvrx.cycle.outcome": "completed"})
        span.close()
        span.close({"nv.nvrx.cycle.outcome": "terminated"})

    def test_reopen_closes_the_previous_span(self):
        # A stale or standby rendezvous may be followed by another operation.
        span = telemetry.ManualSpan()
        span.open(
            "nvrx.ft",
            "nv.nvrx.ftl.rendezvous",
            {
                "nv.nvrx.ftl.rdzv.round": 0,
                "nv.nvrx.ftl.profiling.cycle": 0,
            },
        )
        first_stack = span._stack
        span.set({"nv.nvrx.cycle.outcome": "failed"})
        span.open(
            "nvrx.ft",
            "nv.nvrx.ftl.rendezvous",
            {
                "nv.nvrx.ftl.rdzv.round": 1,
                "nv.nvrx.ftl.profiling.cycle": 1,
            },
        )
        self.assertIsNot(span._stack, first_stack)
        span.close()
        self.assertIsNone(span._stack)

    def test_set_tolerates_none_and_empty(self):
        span = telemetry.ManualSpan()
        span.open("nvrx.ft", "nv.nvrx.ftl.rendezvous")
        span.set(None)
        span.set({})
        span.close()

    def test_open_attributes_are_not_inherited_by_default(self):
        attributes = {
            "nv.nvrx.ftl.rdzv.round": 3,
            "nv.nvrx.ftl.profiling.cycle": 4,
        }
        span = telemetry.ManualSpan()
        with (
            unittest.mock.patch.object(telemetry, "_span_attributes", create=True) as scope,
            unittest.mock.patch.object(telemetry, "span") as span_context,
        ):
            span.open("nvrx.ft", "nv.nvrx.ftl.attribution", attributes)
            span_context.assert_called_once_with("nvrx.ft", "nv.nvrx.ftl.attribution", attributes)
            scope.assert_not_called()
            span.close()

    def test_open_failure_restores_the_short_attribute_scope(self):
        if not telemetry._AVAILABLE:
            self.skipTest("requires nemo-lens")
        scope = unittest.mock.MagicMock()
        span = telemetry.ManualSpan()
        with (
            unittest.mock.patch.object(telemetry, "_is_span_group_enabled", return_value=True),
            unittest.mock.patch.object(telemetry, "_span_attributes", return_value=scope),
            unittest.mock.patch.object(telemetry, "span", side_effect=RuntimeError("open failed")),
            self.assertRaisesRegex(RuntimeError, "open failed"),
        ):
            span.open(
                "nvrx.ft",
                "nv.nvrx.ftl.rendezvous",
                {
                    "nv.nvrx.ftl.rdzv.round": 3,
                    "nv.nvrx.ftl.profiling.cycle": 4,
                },
                inherit_attributes=True,
            )
        scope.__exit__.assert_called_once()
        self.assertIsNone(span._stack)
        self.assertIsNone(span._span)


@unittest.skipUnless(telemetry._AVAILABLE, "requires nemo-lens")
class TestLensTimedEmission(unittest.TestCase):
    def setUp(self):
        self.gate = self.enterContext(
            unittest.mock.patch.object(telemetry, "_is_span_group_enabled", return_value=True)
        )
        self.emit = self.enterContext(unittest.mock.patch.object(telemetry, "_emit_span"))
        self.tracer = self.enterContext(unittest.mock.patch.object(telemetry, "_get_tracer"))

    def test_interval_forwards_group_and_parent_and_returns_context_or_none(self):
        from opentelemetry import trace

        parent = trace.SpanContext(1, 2, False)
        for supplied_parent in (None, parent):
            with self.subTest(parent=supplied_parent):
                result = telemetry.backdated_span(
                    "nvrx.ft", "interval", 1, 2, parent=supplied_parent
                )
                self.assertIs(result, self.emit.return_value.get_span_context.return_value)
                args, kwargs = self.emit.call_args
                self.assertEqual(args, (self.tracer.return_value, "interval", 1, 2))
                self.assertEqual(kwargs["group"], "nvrx.ft")
                context = kwargs["context"]
                self.assertIsNotNone(context)
                self.assertEqual(
                    trace.get_current_span(context).get_span_context(),
                    supplied_parent or trace.INVALID_SPAN_CONTEXT,
                )
        self.emit.return_value = None
        self.assertIsNone(telemetry.backdated_span("nvrx.ft", "interval", 1, 2))
        self.emit.reset_mock()
        for start, end in ((None, 2), (1, None), (None, None)):
            self.assertIsNone(telemetry.backdated_span("nvrx.ft", "missing", start, end))
        self.emit.assert_not_called()

    def test_mark_reads_clock_once(self):
        with unittest.mock.patch.object(telemetry.time, "time", return_value=1000) as clock:
            result = telemetry.mark("nvrx.ft", "instant")
        clock.assert_called_once_with()
        self.assertIs(result, self.emit.return_value.get_span_context.return_value)
        self.emit.assert_called_once_with(
            self.tracer.return_value,
            "instant",
            1000,
            1000,
            group="nvrx.ft",
            context=None,
            attributes=None,
        )

    def test_gate_precedes_clock_context_and_attribute_work(self):
        self.gate.return_value = False
        with (
            unittest.mock.patch.object(telemetry.time, "time", side_effect=AssertionError("clock")),
            unittest.mock.patch.object(
                telemetry._otel_context, "Context", side_effect=AssertionError("context")
            ),
        ):
            self.assertIsNone(telemetry.mark("nvrx.ft", "disabled"))
            self.assertIsNone(telemetry.backdated_span("nvrx.ft", "disabled", 1, 2))
        self.tracer.assert_not_called()
        self.emit.assert_not_called()

    def test_phase_close_restores_context_when_emission_fails(self):
        from opentelemetry import trace

        current = trace.get_current_span()
        self.emit.return_value.get_span_context.return_value = trace.SpanContext(1, 2, False)
        phase = telemetry.Phase()
        self.addCleanup(phase.close)
        phase.open(
            "nvrx.ft",
            "cycle",
            {
                "nv.nvrx.ftl.rdzv.round": 3,
                "nv.nvrx.ftl.profiling.cycle": 4,
            },
        )
        phase.set({"nv.nvrx.ftl.membership": "active"})
        self.emit.side_effect = RuntimeError("emission failed")
        with self.assertRaisesRegex(RuntimeError, "emission failed"):
            phase.close()
        self.assertIs(trace.get_current_span(), current)
        self.assertIsNone(phase._start)
        phase.close()


class TestMarkAndFlush(unittest.TestCase):

    def test_mark_is_inert(self):
        telemetry.mark("nvrx.ft", "nv.nvrx.ftl.fault")
        telemetry.mark(
            "nvrx.ft",
            "nv.nvrx.ftl.fault",
            {"nv.nvrx.ftl.cycle.state": "FAILED", "nv.nvrx.ftl.cycle.failures": 2},
        )

    def test_flush_is_inert(self):
        # Must tolerate a provider with no force_flush (the no-op one) and a
        # provider that was never configured at all.
        telemetry.flush()
        telemetry.flush(timeout_ms=1)

    def test_shutdown_is_bounded_and_never_raises(self):
        class SlowHandle:
            def __init__(self):
                self.entered = threading.Event()

            def shutdown(self, timeout_ms: int = 5000):
                self.entered.set()
                time.sleep(30)  # a collector that is gone

        handle = SlowHandle()
        started = time.monotonic()
        telemetry.shutdown(handle, timeout_s=0.2)
        elapsed = time.monotonic() - started
        self.assertTrue(handle.entered.wait(1), "shutdown() was never called")
        self.assertLess(elapsed, 5, "shutdown was not bounded")


class TestBackdatedSpan(unittest.TestCase):
    """Startup windows are reconstructed from timestamps, so guard the inputs."""

    def test_inert_without_telemetry(self):
        telemetry.backdated_span("job", "nv.nvrx.ftl.python.startup", 1000.0, 1016.7)
        telemetry.backdated_span(
            "job", "nv.nvrx.ftl.python.imports", 1016.7, 1020.9, {"nv.nvrx.ftl.node": "n0"}
        )

    def test_absent_timestamps_are_dropped_not_raised(self):
        # SLURM_JOB_START_TIME is absent off Slurm, so the caller passes None
        # rather than pre-checking; `end > None` would be a TypeError.
        telemetry.backdated_span("job", "pre_startup", None, 1016.7)
        telemetry.backdated_span("job", "pre_startup", 1000.0, None)
        telemetry.backdated_span("job", "pre_startup", None, None)

    def test_disabled_windows_are_inert(self):
        # Disabled instrumentation does not validate or emit these windows.
        telemetry.backdated_span("job", "nv.nvrx.ftl.python.imports", 1016.7, 1000.0)
        telemetry.backdated_span("job", "nv.nvrx.ftl.python.imports", 1000.0, 1000.0)


class TestResourcePublicationWithoutLens(unittest.TestCase):
    def test_fresh_process_without_optional_imports(self):
        import subprocess
        import sys

        code = r"""
import importlib.abc
import importlib.util
import os
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == sys.argv[2] or fullname.startswith(sys.argv[2] + "."):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockOptional())
os.environ['OTEL_RESOURCE_ATTRIBUTES'] = 'job.uid=imported'
spec = importlib.util.spec_from_file_location('isolated_telemetry', sys.argv[1])
telemetry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(telemetry)
assert not telemetry._AVAILABLE
for original in (None, '', 'job.uid=live,nv.dl.rank=7'):
    if original is None:
        os.environ.pop('OTEL_RESOURCE_ATTRIBUTES', None)
    else:
        os.environ['OTEL_RESOURCE_ATTRIBUTES'] = original
    assert telemetry.get_otel_resource_attributes() == {}
    assert telemetry.compose_attributes({}, defaults={'nv.dl.rank': 3}) == {}
    assert telemetry.extend_otel_resource_attributes(
        'job.uid=imported', overrides={'role': 'worker'}
    ) == 'job.uid=imported'
    try:
        with telemetry.publish_otel_resource_attributes({'role': 'worker'}):
            assert os.environ.get('OTEL_RESOURCE_ATTRIBUTES') == original
            raise RuntimeError('workload error')
    except RuntimeError:
        pass
    assert os.environ.get('OTEL_RESOURCE_ATTRIBUTES') == original
"""
        for blocked in ("nemo.lens", "opentelemetry"):
            with self.subTest(blocked=blocked):
                result = subprocess.run(
                    [sys.executable, "-c", code, telemetry.__file__, blocked],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


class TestPhase(unittest.TestCase):
    """A phase is a start anchor and a duration summary; check the two line up.

    nemo-lens and the OTel SDK are optional and usually absent here, so the two
    primitives a phase is built from are replaced and the phase's own logic --
    span naming, the backdated window, where attributes land -- is what is under
    test.
    """

    def setUp(self):
        self.marks = []
        self.spans = []

        def fake_mark(group, name, attributes=None):
            self.marks.append((group, name, attributes))
            return f"ctx-of-{name}"

        def fake_backdated(group, name, start, end, attributes=None, parent=None):
            self.spans.append((group, name, start, end, attributes, parent))

        for target, replacement in (
            ("mark", fake_mark),
            ("backdated_span", fake_backdated),
            # Stands in for nemo-lens being importable, alongside the two primitives
            # it would have supplied. Without it open() takes its unavailable-so-inert
            # path and none of the logic below is reachable.
            ("_AVAILABLE", True),
        ):
            patcher = unittest.mock.patch.object(telemetry, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_marks_the_start_and_backdates_the_span_to_it(self):
        phase = telemetry.Phase()
        before = time.time()
        opening = {
            "nv.nvrx.ftl.rdzv.round": 2,
            "nv.nvrx.ftl.profiling.cycle": 3,
        }
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle", opening)
        phase.close({"nv.nvrx.cycle.outcome": "completed"})
        after = time.time()

        self.assertEqual(self.marks, [("nvrx.ft", "nv.nvrx.ftl.cycle_start", opening)])
        group, name, start, end, attributes, parent = self.spans[0]
        self.assertEqual((group, name), ("nvrx.ft", "nv.nvrx.ftl.cycle"))
        # The span covers the window, rather than being an instant at close.
        self.assertLessEqual(before, start)
        self.assertLessEqual(start, end)
        self.assertLessEqual(end, after)
        # Same trace as the mark, so the spans that ran inside the phase join it.
        self.assertEqual(parent, "ctx-of-nv.nvrx.ftl.cycle_start")
        # Opening attributes carry through to the span; close adds to them.
        self.assertEqual(attributes, {**opening, "nv.nvrx.cycle.outcome": "completed"})

    def test_open_attributes_go_on_the_mark_and_the_span(self):
        # The mark is the only record while the phase runs, and the span is the only
        # one a consumer filters. Both need the attributes.
        phase = telemetry.Phase()
        opening = {
            "nv.nvrx.ftl.rdzv.round": 3,
            "nv.nvrx.ftl.profiling.cycle": 4,
        }
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle", opening)
        phase.close()
        self.assertEqual(self.marks[0][2], opening)
        self.assertEqual(self.spans[0][4], opening)

    def test_close_attributes_override_opening_ones(self):
        # Lets a required attribute be seeded at open rather than on each close path.
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle", {"nv.nvrx.ftl.membership": "unjoined"})
        phase.close({"nv.nvrx.ftl.membership": "standby"})
        self.assertEqual(self.marks[0][2], {"nv.nvrx.ftl.membership": "unjoined"})
        self.assertEqual(self.spans[0][4], {"nv.nvrx.ftl.membership": "standby"})

    def test_set_accumulates_until_close(self):
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle")
        phase.set({"nv.nvrx.ftl.group.rank": 3})
        phase.set({"nv.nvrx.ftl.membership": "active"})
        phase.close({"nv.nvrx.cycle.outcome": "failed"})
        self.assertEqual(
            self.spans[0][4],
            {
                "nv.nvrx.ftl.group.rank": 3,
                "nv.nvrx.ftl.membership": "active",
                "nv.nvrx.cycle.outcome": "failed",
            },
        )

    def test_close_is_idempotent(self):
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle")
        phase.close()
        phase.close({"nv.nvrx.cycle.outcome": "completed"})
        self.assertEqual(len(self.spans), 1)

    def test_close_without_open_is_a_no_op(self):
        telemetry.Phase().close({"nv.nvrx.cycle.outcome": "completed"})
        self.assertEqual(self.spans, [])

    def test_is_inert_without_nemo_lens(self):
        # Nothing downstream can record, so open() does no work at all rather than
        # building attributes and a mark name for primitives that will drop them.
        with unittest.mock.patch.object(telemetry, "_AVAILABLE", False):
            phase = telemetry.Phase()
            phase.open(
                "nvrx.ft",
                "nv.nvrx.ftl.cycle",
                {
                    "nv.nvrx.ftl.rdzv.round": 1,
                    "nv.nvrx.ftl.profiling.cycle": 2,
                },
            )
            phase.set({"nv.nvrx.ftl.group.rank": 0})
            phase.close({"nv.nvrx.cycle.outcome": "completed"})
        self.assertEqual(self.marks, [])
        self.assertEqual(self.spans, [])

    def test_open_closes_the_previous_phase(self):
        # The launcher reuses one handle across cycles and relies on this.
        phase = telemetry.Phase()
        phase.open(
            "nvrx.ft",
            "nv.nvrx.ftl.cycle",
            {
                "nv.nvrx.ftl.rdzv.round": 0,
                "nv.nvrx.ftl.profiling.cycle": 0,
            },
        )
        phase.open(
            "nvrx.ft",
            "nv.nvrx.ftl.cycle",
            {
                "nv.nvrx.ftl.rdzv.round": 1,
                "nv.nvrx.ftl.profiling.cycle": 1,
            },
        )
        self.assertEqual(len(self.spans), 1, "the first cycle was never emitted")
        self.assertEqual(len(self.marks), 2)

    def test_attributes_do_not_leak_between_phases(self):
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle")
        phase.close({"nv.nvrx.cycle.outcome": "failed"})
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle")
        phase.close()
        self.assertEqual(self.spans[1][4], {})


class TestSetupTelemetry(unittest.TestCase):

    def test_returns_handle_with_idempotent_shutdown(self):
        # Disabled is the default (NEMO_LENS_ENABLED is unset), so this exercises
        # the no-op path whether or not nemo-lens is installed.
        handle = telemetry.setup_telemetry("nvrx.test", "nvrx-test0")
        self.assertTrue(hasattr(handle, "shutdown"))
        handle.shutdown()
        handle.shutdown()

    def test_uses_the_current_lens_setup_signature(self):
        original = telemetry._AVAILABLE
        telemetry._AVAILABLE = True
        handle = unittest.mock.MagicMock()

        def strict_setup(config, resource_attributes=None):
            self.assertIsNotNone(config)
            self.assertEqual(resource_attributes, {"service.instance.id": "nvrx-test0"})
            return handle

        try:
            with (
                unittest.mock.patch.object(telemetry, "_setup_telemetry", strict_setup),
                unittest.mock.patch.object(telemetry, "_NemoLensConfig", create=True) as config_cls,
            ):
                config_cls.from_env.return_value = unittest.mock.MagicMock()
                self.assertIs(telemetry.setup_telemetry("nvrx.test", "nvrx-test0"), handle)
        finally:
            telemetry._AVAILABLE = original

    @unittest.skipUnless(telemetry._AVAILABLE, "requires nemo-lens")
    def test_misconfigured_provider_leaves_instrumentation_inert(self):
        import subprocess
        import sys

        try:
            import opentelemetry.sdk.trace  # noqa: F401
        except ImportError:
            self.skipTest("requires the OpenTelemetry SDK")

        # A subprocess isolates OTel's process-global, write-once providers.
        code = r"""
import os
import time
from nvidia_resiliency_ext.shared_utils import telemetry

os.environ.update({
    "NEMO_LENS_ENABLED": "1",
    "NEMO_LENS_TRACES_ENABLED": "0",
    "NEMO_LENS_METRICS_ENABLED": "0",
    "NEMO_LENS_LOGS_ENABLED": "0",
    "OTEL_PYTHON_TRACER_PROVIDER": "missing_provider",
})

@telemetry.trace_fn("nvrx.ft", "work")
def work():
    return "completed"

@telemetry.trace_fn("nvrx.ft", "work_with_attrs", attrs=lambda: {"node": "test"})
def work_with_attrs():
    return "completed"

handle = telemetry.setup_telemetry("nvrx.test", "test")
now = time.time()
telemetry.record_process_startup("nvrx.job", now, now)
telemetry.mark("nvrx.ft", "fault")
phase = telemetry.Phase()
phase.open("nvrx.ft", "cycle")
with telemetry.span("nvrx.ft", "operation"):
    assert work() == "completed"
    assert work_with_attrs() == "completed"
phase.close()
telemetry.flush()
telemetry.shutdown(handle)
"""
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=15
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nemo-lens init failed", result.stderr)

    def test_init_failure_does_not_propagate(self):
        original = telemetry._AVAILABLE
        telemetry._AVAILABLE = True
        try:
            # _NemoLensConfig is undefined when nemo-lens is absent; either way the
            # shim must degrade to a no-op handle rather than raise into the caller.
            with unittest.mock.patch.object(
                telemetry, "_setup_telemetry", side_effect=RuntimeError("boom"), create=True
            ):
                with unittest.mock.patch.object(
                    telemetry, "_NemoLensConfig", create=True
                ) as config_cls:
                    config_cls.from_env.return_value = unittest.mock.MagicMock()
                    handle = telemetry.setup_telemetry("nvrx.test", "nvrx-test0")
            self.assertIsInstance(handle, telemetry._NoOpHandle)
        finally:
            telemetry._AVAILABLE = original


@unittest.skipUnless(telemetry._AVAILABLE, "nemo-lens is not installed")
class TestSpanGroupRegistration(unittest.TestCase):
    """The NVRx groups must be selectable, or every NVRx span is dark.

    nemo-lens ships no group names, so importing the shim is what makes these
    resolvable -- at import, not in ``setup_telemetry``, which the trainer never calls.
    """

    def test_registered_under_the_nvrx_namespace(self):
        from nemo.lens import SpanRegistry

        self.assertIn(telemetry._NAMESPACE, SpanRegistry.namespaces())

    def test_every_group_resolves_by_name(self):
        from nemo.lens import SpanRegistry

        for group in telemetry._GROUPS:
            enabled, pending = SpanRegistry.resolve(group)
            self.assertEqual(enabled, frozenset([group]))
            self.assertEqual(pending, frozenset(), f"{group!r} resolved to nothing")

    def test_presets_resolve_to_their_members(self):
        from nemo.lens import SpanRegistry

        for preset, members in telemetry._PRESETS.items():
            enabled, _ = SpanRegistry.resolve(preset)
            self.assertTrue(
                members <= enabled, f"preset {preset!r} is missing {sorted(members - enabled)}"
            )

    def test_phases_are_a_drill_down_not_a_default(self):
        # Per-request spans are always on; per-stage ones are opted into.
        self.assertIn(telemetry._CKPT, telemetry._PRESETS["default"])
        self.assertNotIn(telemetry._CKPT_PHASES, telemetry._PRESETS["default"])
        self.assertIn(telemetry._CKPT_PHASES, telemetry._PRESETS["per_step"])


class TestEverySpanGroupIsRegistered(unittest.TestCase):
    """No call site may name a group NVRx does not register.

    nemo-lens reports an unregistered group and carries on, which is right for a
    job-wide spec and wrong for a call site, where it is a typo that costs those
    spans silently. Reads the source rather than importing it, so it runs lens-free.
    """

    #: Every call that takes a span group as its first positional argument.
    _CALLS = frozenset(
        ["span", "linked_span", "mark", "trace_fn", "backdated_span", "record_process_startup"]
    )

    def test_no_call_site_names_an_unregistered_group(self):
        root = pathlib.Path(telemetry.__file__).parent.parent
        offenders = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name not in self._CALLS:
                    continue
                group = node.args[0]
                if not isinstance(group, ast.Constant) or not isinstance(group.value, str):
                    continue
                if group.value not in telemetry._GROUPS:
                    offenders.append(
                        f"{path.relative_to(root)}:{group.lineno} {name}({group.value!r})"
                    )
        self.assertEqual(offenders, [], "call sites naming an unregistered span group")

    def test_the_scan_actually_finds_call_sites(self):
        # Guards the test above against passing because it matched nothing.
        root = pathlib.Path(telemetry.__file__).parent.parent
        found = sum(
            1
            for path in root.rglob("*.py")
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
            if isinstance(node, ast.Call)
            and node.args
            and (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", None)
            )
            in self._CALLS
            and isinstance(node.args[0], ast.Constant)
        )
        self.assertGreater(found, 10, "the span-group scan matched almost nothing")


if __name__ == "__main__":
    unittest.main()
