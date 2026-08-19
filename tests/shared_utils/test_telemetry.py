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
        telemetry.backdated_span("job", "pre_startup", 1000.0, 1016.7)
        telemetry.backdated_span("job", "nvrx.cold_start", 1016.7, 1020.9, {"nvrx.node": "n0"})

    def test_absent_timestamps_are_dropped_not_raised(self):
        # SLURM_JOB_START_TIME is absent off Slurm, so the caller passes None
        # rather than pre-checking; `end > None` would be a TypeError.
        telemetry.backdated_span("job", "pre_startup", None, 1016.7)
        telemetry.backdated_span("job", "pre_startup", 1000.0, None)
        telemetry.backdated_span("job", "pre_startup", None, None)

    def test_non_positive_window_is_dropped(self):
        # Clock skew between Slurm's stamp and the launch script's can invert these.
        telemetry.backdated_span("job", "pre_startup", 1016.7, 1000.0)
        telemetry.backdated_span("job", "pre_startup", 1000.0, 1000.0)


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
        self.assertEqual(attributes, {"nvrx.cycle_outcome": "succeeded"})

    def test_open_attributes_go_on_the_mark_not_the_span(self):
        # The mark is the only thing that exists while the phase runs, so what
        # identifies the phase has to be on it -- the backdated span may never
        # be emitted at all.
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nvrx.ft.run", {"is_goodput_span": False})
        phase.close()
        self.assertEqual(self.marks[0][2], {"is_goodput_span": False})
        self.assertEqual(self.spans[0][4], {})

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
        handle = telemetry.setup_telemetry(0, 1)
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
                    handle = telemetry.setup_telemetry(0, 1)
            self.assertIsInstance(handle, telemetry._NoOpHandle)
        finally:
            telemetry._AVAILABLE = original


@unittest.skipUnless(telemetry._AVAILABLE, "nemo-lens is not installed")
class TestAdHocSpanGroups(unittest.TestCase):
    """A group nobody declared must cost that group, not the whole signal."""

    def test_unknown_group_resolves_to_itself(self):
        resolved = telemetry._NVRxSpanGroup.resolve("nvrx,debug_issue12345")
        self.assertIn("debug_issue12345", resolved)
        self.assertIn("nvrx.ft", resolved, "declared groups must survive alongside it")

    def test_unknown_group_alone_still_resolves(self):
        self.assertEqual(
            telemetry._NVRxSpanGroup.resolve("debug_issue12345"),
            frozenset(["debug_issue12345"]),
        )

    def test_declared_groups_are_unaffected(self):
        self.assertEqual(
            telemetry._NVRxSpanGroup.resolve("nvrx"),
            frozenset(["nvrx.ft", "nvrx.ckpt"]),
        )


@unittest.skipUnless(telemetry._AVAILABLE, "nemo-lens is not installed")
class TestSpanGroups(unittest.TestCase):
    """NVRx groups must resolve, and must be on by default."""

    def test_nvrx_groups_resolve(self):
        self.assertEqual(
            telemetry._NVRxSpanGroup.resolve("nvrx"),
            frozenset(["nvrx.ft", "nvrx.ckpt"]),
        )

    def test_nvrx_groups_are_in_every_preset(self):
        for preset in telemetry._NVRxSpanGroup._PRESETS:
            resolved = telemetry._NVRxSpanGroup.resolve(preset)
            self.assertIn("nvrx.ft", resolved, f"missing from preset {preset!r}")
            self.assertIn("nvrx.ckpt", resolved, f"missing from preset {preset!r}")

    def test_base_groups_are_preserved(self):
        self.assertIn("job", telemetry._NVRxSpanGroup.resolve("default"))


if __name__ == "__main__":
    unittest.main()
