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

These cover optional instrumentation, span lifecycles, and recovery from
telemetry setup failures. They run with or without nemo-lens installed.
"""

import threading
import time
import unittest
from unittest.mock import ANY, MagicMock, Mock, patch

import pytest

from nvidia_resiliency_ext.shared_utils import semconv, telemetry

try:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
except ImportError:
    InMemorySpanExporter = None


@unittest.skipUnless(telemetry._AVAILABLE, "requires nemo-lens")
@unittest.skipUnless(InMemorySpanExporter is not None, "requires the OpenTelemetry SDK")
class TestLifecycleSnapshots(unittest.TestCase):
    def setUp(self):
        from nemo.lens import NemoLensConfig
        from nemo.lens.providers import build_providers
        from nemo.lens.state import enabled_span_groups, set_enabled_span_groups

        self.addCleanup(set_enabled_span_groups, enabled_span_groups())
        set_enabled_span_groups(frozenset({semconv.SPAN_GROUP_FT}))
        self.exporter = InMemorySpanExporter()
        with patch("opentelemetry.trace.set_tracer_provider") as install:
            build_providers(
                NemoLensConfig(enabled=True, metrics_enabled=False),
                span_exporter=self.exporter,
            )
        self.provider = install.call_args.args[0]
        self.addCleanup(self.provider.shutdown)
        self.enterContext(patch.object(telemetry, "_get_tracer", self.provider.get_tracer))
        self.enterContext(patch("opentelemetry.trace.get_tracer", self.provider.get_tracer))

    def _record_group(self, cycle_attributes, rendezvous_attributes):
        phase = telemetry.Phase()
        self.addCleanup(phase.close)
        phase.open(semconv.SPAN_GROUP_FT, "cycle", cycle_attributes)
        with telemetry.span(semconv.SPAN_GROUP_FT, "await_round", cycle_attributes):
            pass
        with telemetry.span(
            semconv.SPAN_GROUP_FT, "rendezvous", rendezvous_attributes, inherit_attributes=True
        ):
            with telemetry.span(semconv.SPAN_GROUP_FT, "health_check"):
                pass
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

    def test_inherited_attributes_restore_after_body_failure(self):
        from opentelemetry import trace

        previous = trace.get_current_span()
        error = RuntimeError("operation failed")
        with self.assertRaises(RuntimeError) as caught:
            with telemetry.span(
                semconv.SPAN_GROUP_FT, "operation", {"entry": "value"}, inherit_attributes=True
            ) as active:
                active.set_attribute("updated", 42)
                active.set_attributes({"final": True})
                with telemetry.span(semconv.SPAN_GROUP_FT, "child"):
                    pass
                raise error
        self.assertIs(caught.exception, error)
        self.assertIs(trace.get_current_span(), previous)
        with telemetry.span(semconv.SPAN_GROUP_FT, "sibling", {"own": "value"}):
            with telemetry.span(semconv.SPAN_GROUP_FT, "sibling_child"):
                pass
        self.provider.force_flush()
        spans = {span.name: span for span in self.exporter.get_finished_spans()}
        self.assertEqual(
            dict(spans["operation"].attributes), {"entry": "value", "updated": 42, "final": True}
        )
        self.assertEqual(dict(spans["child"].attributes), {"entry": "value"})
        self.assertEqual(dict(spans["sibling"].attributes), {"own": "value"})
        self.assertEqual(dict(spans["sibling_child"].attributes), {})


class TestTelemetryIsInert(unittest.TestCase):
    """Instrumentation must be a no-op before/without setup_telemetry()."""

    def test_span_runs_body_and_preserves_its_exception(self):
        with telemetry.span(semconv.SPAN_GROUP_FT, "nv.nvrx.ftl.cycle") as active:
            active.set_attribute("membership", "active")
            active.set_attributes({"rank": 0})
        error = ValueError("from the instrumented body")
        with self.assertRaises(ValueError) as caught:
            with telemetry.span(
                semconv.SPAN_GROUP_FT,
                "nv.nvrx.ftl.cycle",
                {"entry": "value"},
                inherit_attributes=True,
            ):
                raise error
        self.assertIs(caught.exception, error)

    def test_trace_fn_preserves_arguments_results_and_exceptions(self):
        @telemetry.trace_fn(semconv.SPAN_GROUP_FT, "nv.nvrx.ftl.worker_launch")
        def start(a, b=2, *, error=None):
            if error is not None:
                raise error
            return a + b

        self.assertEqual(start(1), 3)
        self.assertEqual(start(1, b=10), 11)
        error = RuntimeError("worker launch failed")
        with self.assertRaises(RuntimeError) as caught:
            start(1, error=error)
        self.assertIs(caught.exception, error)

    @unittest.skipUnless(telemetry._AVAILABLE, "requires nemo-lens")
    def test_trace_fn_does_not_evaluate_attributes_when_disabled(self):
        attributes = Mock(side_effect=AssertionError("attributes evaluated"))

        @telemetry.trace_fn(semconv.SPAN_GROUP_FT, "disabled", attrs=attributes)
        def work(value):
            return value

        with patch.object(telemetry, "_is_span_group_enabled", return_value=False):
            self.assertEqual(work(7), 7)
        attributes.assert_not_called()

    @unittest.skipUnless(telemetry._AVAILABLE, "requires nemo-lens")
    def test_trace_fn_passes_callback_attributes_to_the_span(self):
        attributes = Mock(return_value={"example.attribute": "value"})

        @telemetry.trace_fn(semconv.SPAN_GROUP_FT, "enabled", attrs=attributes)
        def work(value, *, scale=1):
            return value * scale

        with (
            patch.object(telemetry, "_is_span_group_enabled", return_value=True),
            patch.object(telemetry, "_managed_span") as managed,
        ):
            self.assertEqual(work(7, scale=2), 14)
        attributes.assert_called_once_with(7, scale=2)
        managed.assert_called_once_with(
            semconv.SPAN_GROUP_FT,
            "enabled",
            ANY,
            **{"example.attribute": "value"},
        )

    def test_trace_fn_callback_is_inert_without_lens(self):
        attributes = Mock(side_effect=AssertionError("attributes evaluated"))

        with patch.object(telemetry, "_AVAILABLE", False):

            @telemetry.trace_fn(semconv.SPAN_GROUP_FT, "absent", attrs=attributes)
            def work(value):
                return value

        self.assertEqual(work(7), 7)
        attributes.assert_not_called()


@pytest.mark.skipif(not telemetry._AVAILABLE, reason="requires nemo-lens")
class TestLensTimedEmission:
    @pytest.fixture(autouse=True)
    def _mock_emission(self):
        with (
            patch.object(telemetry, "_is_span_group_enabled", return_value=True) as self.gate,
            patch.object(telemetry, "_emit_span") as self.emit,
            patch.object(telemetry, "_get_tracer") as self.tracer,
        ):
            yield

    @pytest.mark.parametrize("has_parent", [False, True], ids=["root", "explicit-parent"])
    def test_interval_forwards_group_and_parent_and_returns_context(self, has_parent):
        from opentelemetry import trace

        parent = trace.SpanContext(1, 2, False) if has_parent else None
        result = telemetry.backdated_span(semconv.SPAN_GROUP_FT, "interval", 1, 2, parent=parent)
        assert result is self.emit.return_value.get_span_context.return_value
        args, kwargs = self.emit.call_args
        assert args == (self.tracer.return_value, "interval", 1, 2)
        assert kwargs["group"] == semconv.SPAN_GROUP_FT
        context = kwargs["context"]
        assert context is not None
        assert trace.get_current_span(context).get_span_context() == (
            parent or trace.INVALID_SPAN_CONTEXT
        )

    def test_interval_returns_none_when_lens_does_not_record(self):
        self.emit.return_value = None
        assert telemetry.backdated_span(semconv.SPAN_GROUP_FT, "interval", 1, 2) is None

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            pytest.param(None, 2, id="missing-start"),
            pytest.param(1, None, id="missing-end"),
            pytest.param(None, None, id="missing-both"),
        ],
    )
    def test_missing_timestamps_do_not_emit(self, start, end):
        assert telemetry.backdated_span(semconv.SPAN_GROUP_FT, "missing", start, end) is None
        self.emit.assert_not_called()

    def test_mark_reads_clock_once(self):
        with patch.object(telemetry.time, "time", return_value=1000) as clock:
            result = telemetry.mark(semconv.SPAN_GROUP_FT, "instant")
        clock.assert_called_once_with()
        assert result is self.emit.return_value.get_span_context.return_value
        self.emit.assert_called_once_with(
            self.tracer.return_value,
            "instant",
            1000,
            1000,
            group=semconv.SPAN_GROUP_FT,
            context=None,
            attributes=None,
        )

    def test_gate_precedes_clock_context_and_attribute_work(self):
        self.gate.return_value = False
        with (
            patch.object(telemetry.time, "time", side_effect=AssertionError("clock")),
            patch.object(telemetry._otel_context, "Context", side_effect=AssertionError("context")),
        ):
            assert telemetry.mark(semconv.SPAN_GROUP_FT, "disabled") is None
            assert telemetry.backdated_span(semconv.SPAN_GROUP_FT, "disabled", 1, 2) is None
        self.tracer.assert_not_called()
        self.emit.assert_not_called()

    def test_phase_close_restores_context_when_emission_fails(self, request):
        from opentelemetry import trace

        current = trace.get_current_span()
        self.emit.return_value.get_span_context.return_value = trace.SpanContext(1, 2, False)
        phase = telemetry.Phase()
        request.addfinalizer(phase.close)
        phase.open(semconv.SPAN_GROUP_FT, "cycle", {"entry": 3})
        phase.set({"updated": 4})
        self.emit.side_effect = RuntimeError("emission failed")
        with pytest.raises(RuntimeError, match="emission failed"):
            phase.close()
        assert trace.get_current_span() is current
        phase.close()


class TestFlushAndShutdown(unittest.TestCase):
    def test_flush_is_inert(self):
        # Tolerate an unconfigured provider without force_flush.
        telemetry.flush()
        telemetry.flush(timeout_ms=1)

    def test_shutdown_returns_while_handle_is_blocked(self):
        class SlowHandle:
            def __init__(self):
                self.entered = threading.Event()
                self.release = threading.Event()
                self.finished = threading.Event()

            def shutdown(self):
                self.entered.set()
                self.release.wait(5)
                self.finished.set()

        handle = SlowHandle()
        try:
            started = time.monotonic()
            telemetry.shutdown(handle, timeout_s=0.2)
            elapsed = time.monotonic() - started
            self.assertTrue(handle.entered.wait(1), "shutdown() was never called")
            self.assertFalse(handle.finished.is_set())
            self.assertLess(elapsed, 2, "shutdown was not bounded")
        finally:
            handle.release.set()
            self.assertTrue(handle.finished.wait(1), "shutdown thread did not finish")


class TestResourcePublicationWithoutLens:
    @pytest.mark.parametrize("blocked", ["nemo.lens", "opentelemetry"], ids=["no-lens", "no-otel"])
    @pytest.mark.parametrize(
        "original",
        [None, "", "job.uid=live,nv.dl.rank=7"],
        ids=["unset", "empty", "populated"],
    )
    def test_fresh_process_without_optional_imports(self, blocked, original):
        import json
        import subprocess
        import sys

        code = r"""
import importlib.abc
import importlib.util
import json
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
original = json.loads(sys.argv[3])
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
else:
    raise AssertionError('workload exception was swallowed')
assert os.environ.get('OTEL_RESOURCE_ATTRIBUTES') == original
"""
        result = subprocess.run(
            [sys.executable, "-c", code, telemetry.__file__, blocked, json.dumps(original)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr


class TestPhase(unittest.TestCase):
    """Phase bookkeeping with mocked emission and context."""

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
            ("_otel_context", MagicMock()),
            ("_otel_trace", MagicMock()),
            # Exercise Phase bookkeeping even without Lens.
            ("_AVAILABLE", True),
        ):
            patcher = patch.object(telemetry, target, replacement, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_phase_preserves_its_window_and_merges_final_attributes(self):
        phase = telemetry.Phase()
        phase.close({"unused": True})
        opening = {
            "nv.nvrx.ftl.rdzv.round": 2,
            "nv.nvrx.ftl.profiling.cycle": 3,
            "nv.nvrx.ftl.membership": "unjoined",
        }
        with patch.object(telemetry, "time") as clock:
            clock.time.side_effect = (100.0, 200.0)
            phase.open(semconv.SPAN_GROUP_FT, "nv.nvrx.ftl.cycle", dict(opening))
            phase.set({"nv.nvrx.ftl.group.rank": 3})
            phase.set({"nv.nvrx.ftl.membership": "active"})
            phase.close({"nv.nvrx.ftl.membership": "standby", "nv.nvrx.cycle.outcome": "completed"})
            phase.close({"unused": True})

        self.assertEqual(self.marks, [(semconv.SPAN_GROUP_FT, "nv.nvrx.ftl.cycle_start", opening)])
        self.assertEqual(len(self.spans), 1)
        group, name, start, end, attributes, parent = self.spans[0]
        self.assertEqual((group, name), (semconv.SPAN_GROUP_FT, "nv.nvrx.ftl.cycle"))
        self.assertEqual((start, end), (100.0, 200.0))
        self.assertEqual(parent, "ctx-of-nv.nvrx.ftl.cycle_start")
        self.assertEqual(
            attributes,
            {
                **opening,
                "nv.nvrx.ftl.group.rank": 3,
                "nv.nvrx.ftl.membership": "standby",
                "nv.nvrx.cycle.outcome": "completed",
            },
        )

    def test_is_inert_without_nemo_lens(self):
        with patch.object(telemetry, "_AVAILABLE", False):
            phase = telemetry.Phase()
            phase.open(semconv.SPAN_GROUP_FT, "cycle", {"entry": 1})
            phase.set({"updated": 2})
            phase.close({"final": 3})
        self.assertEqual(self.marks, [])
        self.assertEqual(self.spans, [])

    def test_reopen_and_explicit_close_reset_attributes(self):
        phase = telemetry.Phase()
        self.addCleanup(phase.close)
        phase.open(semconv.SPAN_GROUP_FT, "cycle", {"entry": 1})
        phase.open(semconv.SPAN_GROUP_FT, "cycle")
        self.assertEqual(len(self.spans), 1, "the first cycle was never emitted")
        self.assertEqual(len(self.marks), 2)
        phase.close({"final": 2})
        self.assertEqual(self.spans[1][4], {"final": 2})
        phase.open(semconv.SPAN_GROUP_FT, "cycle")
        phase.close()
        self.assertEqual(self.spans[2][4], {})


class TestSetupTelemetry(unittest.TestCase):

    def test_returns_handle_with_idempotent_shutdown(self):
        # Disabled is the default (NEMO_LENS_ENABLED is unset), so this exercises
        # the no-op path whether or not nemo-lens is installed.
        handle = telemetry.setup_telemetry("nvrx.test", "nvrx-test0")
        self.assertTrue(hasattr(handle, "shutdown"))
        handle.shutdown()
        handle.shutdown()

    def test_setup_forwards_service_and_resource_attributes(self):
        handle = Mock()
        attributes = {"nv.nvrx.ftl.node": "node0"}
        with (
            patch.object(telemetry, "_AVAILABLE", True),
            patch.object(telemetry, "_NemoLensConfig", create=True) as config_cls,
            patch.object(telemetry, "_setup_telemetry", return_value=handle, create=True) as setup,
        ):
            self.assertIs(telemetry.setup_telemetry("nvrx.test", "nvrx-test0", attributes), handle)
        self.assertEqual(config_cls.from_env.return_value.service_name, "nvrx.test")
        setup.assert_called_once_with(
            config_cls.from_env.return_value,
            resource_attributes={"service.instance.id": "nvrx-test0", **attributes},
        )

    @unittest.skipUnless(telemetry._AVAILABLE, "requires nemo-lens")
    @unittest.skipUnless(InMemorySpanExporter is not None, "requires the OpenTelemetry SDK")
    def test_misconfigured_provider_leaves_instrumentation_inert(self):
        import subprocess
        import sys

        # A subprocess isolates OTel's process-global, write-once providers.
        code = r"""
import os
import time
from nvidia_resiliency_ext.shared_utils import semconv, telemetry

os.environ.update({
    "NEMO_LENS_ENABLED": "1",
    "NEMO_LENS_TRACES_ENABLED": "0",
    "NEMO_LENS_METRICS_ENABLED": "0",
    "NEMO_LENS_LOGS_ENABLED": "0",
    "OTEL_PYTHON_TRACER_PROVIDER": "missing_provider",
})

@telemetry.trace_fn(semconv.SPAN_GROUP_FT, "work")
def work():
    return "completed"

@telemetry.trace_fn(semconv.SPAN_GROUP_FT, "work_with_attrs", attrs=lambda: {"node": "test"})
def work_with_attrs():
    return "completed"

handle = telemetry.setup_telemetry("nvrx.test", "test")
now = time.time()
telemetry.record_process_startup(semconv.SPAN_GROUP_STARTUP, now, now)
telemetry.mark(semconv.SPAN_GROUP_FT, "fault")
phase = telemetry.Phase()
phase.open(semconv.SPAN_GROUP_FT, "cycle")
with telemetry.span(semconv.SPAN_GROUP_FT, "operation"):
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
        with (
            patch.object(telemetry, "_AVAILABLE", True),
            patch.object(telemetry, "_NemoLensConfig", create=True),
            patch.object(
                telemetry, "_setup_telemetry", side_effect=RuntimeError("boom"), create=True
            ),
            self.assertLogs(telemetry.logger, level="WARNING"),
        ):
            handle = telemetry.setup_telemetry("nvrx.test", "nvrx-test0")
            self.assertIsInstance(handle, telemetry._NoOpHandle)
            self.assertFalse(telemetry._AVAILABLE)


@pytest.mark.skipif(not telemetry._AVAILABLE, reason="requires nemo-lens")
class TestSpanGroupRegistration:
    @pytest.mark.parametrize(
        ("group", "name"),
        [
            (semconv.SPAN_GROUP_STARTUP, "nv.nvrx.ftl.python"),
            (semconv.SPAN_GROUP_FT, "nv.nvrx.ftl"),
            (semconv.SPAN_GROUP_CKPT, "nv.nvrx.ckpt"),
            (semconv.SPAN_GROUP_CKPT_PHASES, "nv.nvrx.ckpt.save"),
        ],
    )
    def test_import_registers_group(self, group, name):
        from nemo.lens import SpanRegistry

        assert group == name
        assert SpanRegistry.resolve(group) == (frozenset({name}), frozenset())

    @pytest.mark.parametrize(
        ("preset", "expected"),
        [
            pytest.param(
                "default", {"nv.nvrx.ftl.python", "nv.nvrx.ftl", "nv.nvrx.ckpt"}, id="default"
            ),
            pytest.param(
                "per_step",
                {"nv.nvrx.ftl.python", "nv.nvrx.ftl", "nv.nvrx.ckpt", "nv.nvrx.ckpt.save"},
                id="per-step",
            ),
            pytest.param(
                "profiling",
                {"nv.nvrx.ftl.python", "nv.nvrx.ftl", "nv.nvrx.ckpt", "nv.nvrx.ckpt.save"},
                id="profiling",
            ),
        ],
    )
    def test_preset_selects_expected_nvrx_groups(self, preset, expected):
        from nemo.lens import SpanRegistry

        enabled, pending = SpanRegistry.resolve(preset)
        assert (
            enabled & {"nv.nvrx.ftl.python", "nv.nvrx.ftl", "nv.nvrx.ckpt", "nv.nvrx.ckpt.save"}
            == expected
        )
        assert pending == frozenset()
