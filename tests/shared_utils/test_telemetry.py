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


class TestTelemetryIsInert(unittest.TestCase):
    """Instrumentation must be a no-op before/without setup_telemetry()."""

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

    def test_set_span_attributes_without_active_span(self):
        telemetry.set_span_attributes({"nv.nvrx.cycle.index": 3, "nv.nvrx.ftl.node": "node-0"})


class TestManualSpan(unittest.TestCase):
    """ManualSpan must tolerate every order the launcher can call it in."""

    def test_all_methods_are_safe_before_open(self):
        span = telemetry.ManualSpan()
        span.set({"nv.nvrx.cycle.index": 0})
        span.close({"nv.nvrx.cycle.outcome": "terminated"})
        span.close()

    def test_close_is_idempotent(self):
        span = telemetry.ManualSpan()
        span.open("nvrx.ft", "nv.nvrx.ftl.cycle", {"nv.nvrx.cycle.index": 0})
        span.close({"nv.nvrx.cycle.outcome": "completed"})
        span.close()
        span.close({"nv.nvrx.cycle.outcome": "terminated"})

    def test_reopen_closes_the_previous_span(self):
        # The restart path relies on this: a cycle is left open so teardown lands
        # inside it, and the next rendezvous closes it by opening the next cycle.
        span = telemetry.ManualSpan()
        span.open("nvrx.ft", "nv.nvrx.ftl.cycle", {"nv.nvrx.cycle.index": 0})
        first_stack = span._stack
        span.set({"nv.nvrx.cycle.outcome": "failed"})
        span.open("nvrx.ft", "nv.nvrx.ftl.cycle", {"nv.nvrx.cycle.index": 1})
        self.assertIsNot(span._stack, first_stack)
        span.close()
        self.assertIsNone(span._stack)

    def test_set_tolerates_none_and_empty(self):
        span = telemetry.ManualSpan()
        span.open("nvrx.ft", "nv.nvrx.ftl.cycle")
        span.set(None)
        span.set({})
        span.close()


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

    def test_non_positive_window_is_dropped(self):
        # A coarse clock can make a fast window measure as zero-length or inverted.
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
    """A phase is a mark now and a backdated span later; check the two line up.

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
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle", {"nv.nvrx.cycle.index": 2})
        phase.close({"nv.nvrx.cycle.outcome": "completed"})
        after = time.time()

        self.assertEqual(
            self.marks, [("nvrx.ft", "nv.nvrx.ftl.cycle_start", {"nv.nvrx.cycle.index": 2})]
        )
        group, name, start, end, attributes, parent = self.spans[0]
        self.assertEqual((group, name), ("nvrx.ft", "nv.nvrx.ftl.cycle"))
        # The span covers the window, rather than being an instant at close.
        self.assertLessEqual(before, start)
        self.assertLessEqual(start, end)
        self.assertLessEqual(end, after)
        # Same trace as the mark, so the spans that ran inside the phase join it.
        self.assertEqual(parent, "ctx-of-nv.nvrx.ftl.cycle_start")
        # Opening attributes carry through to the span; close adds to them.
        self.assertEqual(
            attributes, {"nv.nvrx.cycle.index": 2, "nv.nvrx.cycle.outcome": "completed"}
        )

    def test_open_attributes_go_on_the_mark_and_the_span(self):
        # The mark is the only record while the phase runs, and the span is the only
        # one a consumer filters. Both need the attributes.
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle", {"nv.nvrx.cycle.index": 3})
        phase.close()
        self.assertEqual(self.marks[0][2], {"nv.nvrx.cycle.index": 3})
        self.assertEqual(self.spans[0][4], {"nv.nvrx.cycle.index": 3})

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
            phase.open("nvrx.ft", "nv.nvrx.ftl.cycle", {"nv.nvrx.cycle.index": 1})
            phase.set({"nv.nvrx.ftl.group.rank": 0})
            phase.close({"nv.nvrx.cycle.outcome": "completed"})
        self.assertEqual(self.marks, [])
        self.assertEqual(self.spans, [])

    def test_open_closes_the_previous_phase(self):
        # The launcher reuses one handle across cycles and relies on this.
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle", {"nv.nvrx.cycle.index": 0})
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle", {"nv.nvrx.cycle.index": 1})
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
