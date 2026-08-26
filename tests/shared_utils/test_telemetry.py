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
import os
import pathlib
import pickle
import threading
import time
import unittest
import unittest.mock

from nvidia_resiliency_ext.shared_utils import telemetry


class TestTelemetryIsInert(unittest.TestCase):
    """Instrumentation must be a no-op before/without setup_telemetry()."""

    def test_managed_span_yields_and_runs_body(self):
        ran = False
        with telemetry.span("nvrx.ft", "nvrx.ft.cycle") as active:
            ran = True
            self.assertIsNone(active)
        self.assertTrue(ran)

    def test_managed_span_propagates_body_exceptions(self):
        # Telemetry must never swallow a workload error -- notably SignalException,
        # which torch elastic raises out of the launcher's monitor loop.
        with self.assertRaises(ValueError):
            with telemetry.span("nvrx.ft", "nvrx.ft.cycle"):
                raise ValueError("from the instrumented body")

    def test_managed_span_accepts_attributes(self):
        with telemetry.span("nvrx.ckpt", "nvrx.ckpt.save.request", {"nvrx.call_idx": 7}):
            pass

    def test_trace_fn_returns_a_working_decorator(self):
        @telemetry.trace_fn("nvrx.ft", "nvrx.ft.worker_start")
        def start(a, b=2):
            return a + b

        self.assertEqual(start(1), 3)
        self.assertEqual(start(1, b=10), 11)

    def test_trace_fn_propagates_exceptions(self):
        @telemetry.trace_fn("nvrx.ft", "nvrx.ft.teardown")
        def boom():
            raise RuntimeError("worker teardown failed")

        with self.assertRaises(RuntimeError):
            boom()

    def test_set_span_attributes_without_active_span(self):
        telemetry.set_span_attributes({"nvrx.cycle": 3, "nvrx.node": "node-0"})


class TestManualSpan(unittest.TestCase):
    """ManualSpan must tolerate every order the launcher can call it in."""

    def test_all_methods_are_safe_before_open(self):
        span = telemetry.ManualSpan()
        span.set({"nvrx.cycle": 0})
        span.close({"nvrx.cycle_outcome": "terminated"})
        span.close()

    def test_close_is_idempotent(self):
        span = telemetry.ManualSpan()
        span.open("nvrx.ft", "nvrx.ft.cycle", {"nvrx.cycle": 0})
        span.close({"nvrx.cycle_outcome": "succeeded"})
        span.close()
        span.close({"nvrx.cycle_outcome": "terminated"})

    def test_reopen_closes_the_previous_span(self):
        # The restart path relies on this: a cycle is left open so teardown lands
        # inside it, and the next rendezvous closes it by opening the next cycle.
        span = telemetry.ManualSpan()
        span.open("nvrx.ft", "nvrx.ft.cycle", {"nvrx.cycle": 0})
        first_stack = span._stack
        span.set({"nvrx.cycle_outcome": "failed"})
        span.open("nvrx.ft", "nvrx.ft.cycle", {"nvrx.cycle": 1})
        self.assertIsNot(span._stack, first_stack)
        span.close()
        self.assertIsNone(span._stack)

    def test_set_tolerates_none_and_empty(self):
        span = telemetry.ManualSpan()
        span.open("nvrx.ft", "nvrx.ft.cycle")
        span.set(None)
        span.set({})
        span.close()


class TestMarkAndFlush(unittest.TestCase):

    def test_mark_is_inert(self):
        telemetry.mark("nvrx.ft", "nvrx.ft.fault")
        telemetry.mark("nvrx.ft", "nvrx.ft.fault", {"nvrx.state": "FAILED", "nvrx.failures": 2})

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
        telemetry.backdated_span("job", "python.startup", 1000.0, 1016.7)
        telemetry.backdated_span("job", "python.imports", 1016.7, 1020.9, {"nvrx.node": "n0"})

    def test_absent_timestamps_are_dropped_not_raised(self):
        # SLURM_JOB_START_TIME is absent off Slurm, so the caller passes None
        # rather than pre-checking; `end > None` would be a TypeError.
        telemetry.backdated_span("job", "pre_startup", None, 1016.7)
        telemetry.backdated_span("job", "pre_startup", 1000.0, None)
        telemetry.backdated_span("job", "pre_startup", None, None)

    def test_non_positive_window_is_dropped(self):
        # A coarse clock can make a fast window measure as zero-length or inverted.
        telemetry.backdated_span("job", "python.imports", 1016.7, 1000.0)
        telemetry.backdated_span("job", "python.imports", 1000.0, 1000.0)


class TestExtendedResourceAttributes(unittest.TestCase):
    """The agent extends a variable it must never parse, once per cohort."""

    def extend(self, inherited, attributes):
        with unittest.mock.patch.object(telemetry, "_INHERITED_RESOURCE_ATTRIBUTES", inherited):
            return telemetry.extended_resource_attributes(attributes)

    def test_carries_the_inherited_value_through_untouched(self):
        # Whatever the launching environment set is opaque here, including keys
        # NVRx has no notion of.
        inherited = "slurm.job_id=370487,cluster=oci-aga,job.uid=b3f1"
        result = self.extend(inherited, {"nvrx.cycle": 2})
        self.assertTrue(result.startswith(inherited + ","))
        self.assertTrue(result.endswith("nvrx.cycle=2"))

    def test_works_with_nothing_inherited(self):
        self.assertEqual(self.extend("", {"nvrx.cycle": 0}), "nvrx.cycle=0")

    def test_no_attributes_leaves_the_value_alone(self):
        self.assertEqual(self.extend("cluster=oci-aga", {}), "cluster=oci-aga")
        self.assertEqual(self.extend("", {}), "")

    def test_values_are_percent_encoded(self):
        # A value containing a comma or an equals would otherwise be read back as
        # extra pairs, silently rewriting the resource.
        result = self.extend("", {"nvrx.membership": "active,standby=maybe"})
        self.assertEqual(result, "nvrx.membership=active%2Cstandby%3Dmaybe")

    def test_extends_the_inherited_value_not_the_last_one(self):
        # The agent relaunches a cohort every cycle. Extending its own previous
        # output would append another nvrx.cycle each time, without bound.
        inherited = "cluster=oci-aga"
        first = self.extend(inherited, {"nvrx.cycle": 0})
        second = self.extend(inherited, {"nvrx.cycle": 1})
        self.assertEqual(first.count("nvrx.cycle"), 1)
        self.assertEqual(second.count("nvrx.cycle"), 1)
        self.assertEqual(second, "cluster=oci-aga,nvrx.cycle=1")


class TestPublishResourceAttributes(unittest.TestCase):
    """The only channel that reaches a spawned child, so it has to be exact."""

    def setUp(self):
        self.env = unittest.mock.patch.dict(
            "os.environ", {"OTEL_RESOURCE_ATTRIBUTES": "job.uid=abc"}, clear=False
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.inherited = unittest.mock.patch.object(
            telemetry, "_INHERITED_RESOURCE_ATTRIBUTES", "job.uid=abc"
        )
        self.inherited.start()
        self.addCleanup(self.inherited.stop)

    def test_the_child_sees_both_inherited_and_published(self):
        with telemetry.publish_resource_attributes({"dl.rank": 3}):
            published = os.environ["OTEL_RESOURCE_ATTRIBUTES"]
        self.assertIn("job.uid=abc", published)
        self.assertIn("dl.rank=3", published)

    def test_the_parent_is_restored(self):
        # Left set, it would describe this process and every later child of it.
        with telemetry.publish_resource_attributes({"dl.rank": 3}):
            pass
        self.assertEqual(os.environ["OTEL_RESOURCE_ATTRIBUTES"], "job.uid=abc")

    def test_restored_even_when_the_spawn_raises(self):
        with self.assertRaises(RuntimeError):
            with telemetry.publish_resource_attributes({"dl.rank": 3}):
                raise RuntimeError("Process.start() failed")
        self.assertEqual(os.environ["OTEL_RESOURCE_ATTRIBUTES"], "job.uid=abc")

    def test_an_unset_variable_is_removed_again_not_left_empty(self):
        with unittest.mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop("OTEL_RESOURCE_ATTRIBUTES", None)
            with unittest.mock.patch.object(telemetry, "_INHERITED_RESOURCE_ATTRIBUTES", ""):
                with telemetry.publish_resource_attributes({"dl.rank": 3}):
                    self.assertEqual(os.environ["OTEL_RESOURCE_ATTRIBUTES"], "dl.rank=3")
                self.assertNotIn("OTEL_RESOURCE_ATTRIBUTES", os.environ)

    def test_successive_publishes_do_not_accumulate(self):
        # A worker restarts per cycle; building from the last value grows unbounded.
        for _ in range(3):
            with telemetry.publish_resource_attributes({"dl.rank": 3}):
                published = os.environ["OTEL_RESOURCE_ATTRIBUTES"]
        self.assertEqual(published.count("dl.rank"), 1)


class TestContextCarrier(unittest.TestCase):
    """Cause is handed to the checkpoint worker as a picklable dict of headers."""

    def test_carrier_is_none_without_telemetry(self):
        # The AsyncRequest field then defaults to None, and the worker takes the
        # same path it took before any of this existed.
        self.assertIsNone(telemetry.context_carrier())

    def test_carrier_survives_pickling(self):
        # The request is pickled onto a multiprocessing queue, which is the whole
        # reason this is headers rather than a live SpanContext.
        carrier = telemetry.context_carrier() or {
            "traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"
        }
        self.assertEqual(pickle.loads(pickle.dumps(carrier)), carrier)

    def test_baggage_of_nothing_is_empty(self):
        self.assertEqual(telemetry.carrier_baggage(None), {})
        self.assertEqual(telemetry.carrier_baggage({}), {})

    def test_baggage_never_raises_on_a_malformed_carrier(self):
        # The carrier crossed a process boundary and came from another repo's
        # instrumentation; a bad one must cost the attribute, not the checkpoint.
        self.assertEqual(telemetry.carrier_baggage({"traceparent": "nonsense"}), {})
        self.assertEqual(telemetry.carrier_baggage({"baggage": "="}), {})


class TestLinkedSpan(unittest.TestCase):
    """Worker spans link to the trainer's rather than joining its trace."""

    def test_runs_the_body_without_a_carrier(self):
        ran = False
        with telemetry.linked_span("nvrx.ckpt", "nvrx.ckpt.save.request", None) as active:
            ran = True
            self.assertIsNone(active)
        self.assertTrue(ran)

    def test_runs_the_body_with_a_carrier(self):
        ran = False
        carrier = {"traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"}
        with telemetry.linked_span(
            "nvrx.ckpt", "nvrx.ckpt.save.request", carrier, {"nvrx.call_idx": 4}
        ):
            ran = True
        self.assertTrue(ran)

    def test_malformed_carrier_still_runs_the_body(self):
        with telemetry.linked_span("nvrx.ckpt", "nvrx.ckpt.save.write", {"traceparent": "x"}):
            pass

    def test_propagates_body_exceptions(self):
        # Same contract as span(): a checkpoint failure must reach the caller.
        with self.assertRaises(ValueError):
            with telemetry.linked_span("nvrx.ckpt", "nvrx.ckpt.save.request", None):
                raise ValueError("from the instrumented body")


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

        for target, replacement in (("mark", fake_mark), ("backdated_span", fake_backdated)):
            patcher = unittest.mock.patch.object(telemetry, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_marks_the_start_and_backdates_the_span_to_it(self):
        phase = telemetry.Phase()
        before = time.time()
        phase.open("nvrx.ft", "nvrx.ft.cycle", {"nvrx.cycle": 2})
        phase.close({"nvrx.cycle_outcome": "succeeded"})
        after = time.time()

        self.assertEqual(self.marks, [("nvrx.ft", "nvrx.ft.cycle_start", {"nvrx.cycle": 2})])
        (group, name, start, end, attributes, parent) = self.spans[0]
        self.assertEqual((group, name), ("nvrx.ft", "nvrx.ft.cycle"))
        # The span covers the window, rather than being an instant at close.
        self.assertLessEqual(before, start)
        self.assertLessEqual(start, end)
        self.assertLessEqual(end, after)
        # Same trace as the mark, so the spans that ran inside the phase join it.
        self.assertEqual(parent, "ctx-of-nvrx.ft.cycle_start")
        # Opening attributes carry through to the span; close adds to them.
        self.assertEqual(attributes, {"nvrx.cycle": 2, "nvrx.cycle_outcome": "succeeded"})

    def test_open_attributes_go_on_the_mark_and_the_span(self):
        # The mark is the only record while the phase runs, and the span is the only
        # one a consumer filters. Both need the attributes.
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nvrx.ft.cycle", {"nvrx.cycle": 3})
        phase.close()
        self.assertEqual(self.marks[0][2], {"nvrx.cycle": 3})
        self.assertEqual(self.spans[0][4], {"nvrx.cycle": 3})

    def test_close_attributes_override_opening_ones(self):
        # Lets a required attribute be seeded at open rather than on each close path.
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nvrx.ft.cycle", {"nvrx.membership": "unjoined"})
        phase.close({"nvrx.membership": "standby"})
        self.assertEqual(self.marks[0][2], {"nvrx.membership": "unjoined"})
        self.assertEqual(self.spans[0][4], {"nvrx.membership": "standby"})

    def test_set_accumulates_until_close(self):
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nvrx.ft.cycle")
        phase.set({"nvrx.rank": 3})
        phase.set({"nvrx.membership": "active"})
        phase.close({"nvrx.cycle_outcome": "failed"})
        self.assertEqual(
            self.spans[0][4],
            {"nvrx.rank": 3, "nvrx.membership": "active", "nvrx.cycle_outcome": "failed"},
        )

    def test_close_is_idempotent(self):
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nvrx.ft.cycle")
        phase.close()
        phase.close({"nvrx.cycle_outcome": "succeeded"})
        self.assertEqual(len(self.spans), 1)

    def test_close_without_open_is_a_no_op(self):
        telemetry.Phase().close({"nvrx.cycle_outcome": "succeeded"})
        self.assertEqual(self.spans, [])

    def test_open_closes_the_previous_phase(self):
        # The launcher reuses one handle across cycles and relies on this.
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nvrx.ft.cycle", {"nvrx.cycle": 0})
        phase.open("nvrx.ft", "nvrx.ft.cycle", {"nvrx.cycle": 1})
        self.assertEqual(len(self.spans), 1, "the first cycle was never emitted")
        self.assertEqual(len(self.marks), 2)

    def test_attributes_do_not_leak_between_phases(self):
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nvrx.ft.cycle")
        phase.close({"nvrx.cycle_outcome": "failed"})
        phase.open("nvrx.ft", "nvrx.ft.cycle")
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
