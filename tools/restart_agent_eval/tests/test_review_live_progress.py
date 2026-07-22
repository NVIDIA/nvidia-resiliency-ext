# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incremental review progress events and live-artifact publication tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from _assertions import assert_json_file  # noqa: E402
from restart_agent_eval import panel  # noqa: E402
from restart_agent_eval import review  # noqa: E402


class ReviewLiveProgressTest(unittest.TestCase):
    def test_missing_truncated_and_malformed_event_streams_are_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.jsonl"
            missing = review.read_live_progress(path, 10)

            self.assertEqual(missing.items, ())

            path.write_bytes(b'{"event":"partial"')
            partial = review.read_live_progress(path, 0)
            self.assertEqual(partial.next_offset, 0)
            self.assertEqual(partial.items, ())

            path.write_bytes(b"not-json\n")
            malformed = review.read_live_progress(path, 100)
            self.assertEqual(
                [item.message for item in malformed.items], ["live: malformed lifecycle event"]
            )

    def test_should_read_only_new_complete_live_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            events_path = Path(temp_dir) / "events.jsonl"
            events_path.write_text(
                json.dumps(
                    {
                        "event": "run_started",
                        "route_count": 2,
                        "elapsed_s": 0.01,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            first_batch = review.read_live_progress(events_path, 0)
            self.assertEqual(
                [item.message for item in first_batch.items],
                ["live: analysis started routes=2 t=0.01s"],
            )

            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event": "route_completed",
                            "route_id": "fast",
                            "execution_status": "ok",
                            "decision": "RESTART",
                            "elapsed_s": 1.25,
                        }
                    )
                    + "\n"
                )
                handle.write('{"event":"partial"')
            second_batch = review.read_live_progress(events_path, first_batch.next_offset)
            self.assertEqual(
                [item.message for item in second_batch.items],
                ["live: route fast status=ok decision=RESTART t=1.25s"],
            )
            self.assertGreater(second_batch.next_offset, first_batch.next_offset)
            self.assertLess(second_batch.next_offset, events_path.stat().st_size)

    def test_should_format_shared_l0_readiness_event(self) -> None:
        actual = review.format_live_event(
            {
                "event": "l0_artifacts_ready",
                "l0_wall_clock_s": 8.5,
                "elapsed_s": 8.7,
            }
        )

        self.assertEqual(
            actual,
            "live: L0 artifacts ready l0=8.5s t=8.7s",
        )

    def test_should_format_every_lifecycle_event_and_unknown_fallback(self) -> None:
        cases = {
            "deterministic_fallback_ready": "live: deterministic fallback ready decision=RESTART",
            "route_started": "live: route_started",
            "route_completed": "live: route gpt status=ok decision=RESTART",
            "run_completed": "live: analysis completed routes=1/1",
            "run_failed": "live: analysis failed error=timeout",
            "custom_event": "live: custom_event",
        }
        payloads = {
            "deterministic_fallback_ready": {"decision": "RESTART"},
            "route_started": {"route_id": "gpt", "model": "model-a"},
            "route_completed": {"route_id": "gpt", "execution_status": "ok", "decision": "RESTART"},
            "run_completed": {"completed_routes": 1, "total_routes": 1},
            "run_failed": {"error": "timeout"},
            "custom_event": {},
        }
        for event, expected in cases.items():
            with self.subTest(event=event):
                actual = review.format_live_event({"event": event, **payloads[event]})

                self.assertEqual(actual, expected)

    def test_panel_summary_indexes_live_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            live_dir = run_dir / "live"
            live_dir.mkdir()
            (live_dir / "run_status.json").write_text("{}\n", encoding="utf-8")
            (live_dir / "events.jsonl").write_text("", encoding="utf-8")

            json_path, _ = panel.write_panel_summary(
                run_dir,
                [{"target": "deterministic", "analysis": {}, "artifacts": {}}],
            )

            payload = assert_json_file(
                self,
                json_path,
                required_fields=("artifact_paths",),
            )
            self.assertEqual(
                payload["artifact_paths"]["live_status"],
                str(live_dir / "run_status.json"),
            )
            self.assertEqual(
                payload["artifact_paths"]["live_events"],
                str(live_dir / "events.jsonl"),
            )
            self.assertNotIn("live_routes", payload["artifact_paths"])


if __name__ == "__main__":
    unittest.main()
