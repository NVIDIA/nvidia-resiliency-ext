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

import unittest
import unittest.mock

from nvidia_resiliency_ext.shared_utils import telemetry


class TestTelemetryIsInert(unittest.TestCase):
    """Instrumentation must be a no-op before/without setup_telemetry()."""

    def test_managed_span_yields_and_runs_body(self):
        ran = False
        with telemetry.managed_span("nvrx.ft", "nvrx.ft.cycle") as span:
            ran = True
            self.assertIsNone(span)
        self.assertTrue(ran)

    def test_managed_span_propagates_body_exceptions(self):
        # Telemetry must never swallow a workload error -- notably SignalException,
        # which torch elastic raises out of the launcher's monitor loop.
        with self.assertRaises(ValueError):
            with telemetry.managed_span("nvrx.ft", "nvrx.ft.cycle"):
                raise ValueError("from the instrumented body")

    def test_managed_span_accepts_attributes(self):
        with telemetry.managed_span("nvrx.ckpt", "nvrx.ckpt.save.request", **{"nvrx.call_idx": 7}):
            pass

    def test_traced_returns_a_working_decorator(self):
        @telemetry.traced("nvrx.ft", "nvrx.ft.worker_start")
        def start(a, b=2):
            return a + b

        self.assertEqual(start(1), 3)
        self.assertEqual(start(1, b=10), 11)

    def test_traced_propagates_exceptions(self):
        @telemetry.traced("nvrx.ft", "nvrx.ft.teardown")
        def boom():
            raise RuntimeError("worker teardown failed")

        with self.assertRaises(RuntimeError):
            boom()

    def test_set_span_attributes_without_active_span(self):
        telemetry.set_span_attributes(**{"nvrx.cycle": 3, "nvrx.node": "node-0"})


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
