# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval import inspect as inspect_trace  # noqa: E402


def _run_inspection(trace: dict, view: str) -> tuple[int, str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "trace.json"
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = inspect_trace.main([str(trace_path), "--view", view])
    return exit_code, stdout.getvalue(), stderr.getvalue()


class TraceInspectionTest(unittest.TestCase):
    def test_should_emit_full_l0_and_decision_evidence_views(self) -> None:
        decision_evidence = {"schema_version": "decision.v1"}
        trace = {
            "l0_bundle": {"line_count": 10},
            "analyzer_trace": {"decision_evidence": decision_evidence},
        }

        full_exit, full_stdout, _ = _run_inspection(trace, "full-l0")
        decision_exit, decision_stdout, _ = _run_inspection(trace, "decision-evidence")

        self.assertEqual(full_exit, 0)
        self.assertEqual(json.loads(full_stdout), {"line_count": 10})
        self.assertEqual(decision_exit, 0)
        self.assertEqual(json.loads(decision_stdout), decision_evidence)

    def test_should_emit_typed_model_view_when_trace_contains_l0b(self) -> None:
        model_view = {
            "schema_version": "restart_agent_l0_model_view.v1",
            "attempt_execution_context": {"scope": "current_log_only"},
            "evidence_bundle": {"context_windows": []},
            "projection_metrics": {
                "view_size": {"estimated_tokens": 12},
                "projection_integrity": {"status": "ok"},
            },
        }
        trace = {
            "analyzer_trace": {
                "l0_model_view": model_view,
                "l1": {
                    "interaction_transcript": [
                        {
                            "event_type": "bundle_snapshot",
                            "bundle": {"context_windows": [{"window_id": "ignored"}]},
                        }
                    ]
                },
            }
        }

        exit_code, stdout, stderr = _run_inspection(trace, "model-l0")

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout), model_view)
        self.assertEqual(stderr, "")

    def test_should_report_shared_decision_evidence_in_comparison_view(self) -> None:
        decision_evidence = {
            "schema_version": "restart_agent_decision_evidence.v1",
            "deterministic_primary_candidate": {
                "fine_class": "observed_exception",
                "line": 12,
            },
        }
        trace = {
            "l0_bundle": {"context_windows": []},
            "analyzer_trace": {
                "decision_evidence": decision_evidence,
                "l0_model_view": {
                    "schema_version": "restart_agent_l0_model_view.v1",
                    "decision_evidence": decision_evidence,
                    "evidence_bundle": {"context_windows": []},
                    "projection_metrics": {},
                },
                "l1": {"interaction_transcript": []},
            },
        }

        exit_code, stdout, stderr = _run_inspection(trace, "comparison")
        comparison = json.loads(stdout)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(
            comparison["decision_evidence"],
            {
                "schema_version": "restart_agent_decision_evidence.v1",
                "present_in_trace": True,
                "present_in_model_view": True,
                "exactly_shared": True,
            },
        )

    def test_should_reject_model_view_when_trace_has_no_typed_l0b(self) -> None:
        trace = {
            "analyzer_trace": {
                "l1": {
                    "interaction_transcript": [
                        {
                            "event_type": "bundle_snapshot",
                            "schema_version": "transcript-only.v1",
                            "bundle": {"context_windows": []},
                        }
                    ]
                }
            }
        }

        exit_code, stdout, stderr = _run_inspection(trace, "model-l0")

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout, "")
        self.assertTrue(stderr)

    def test_should_reject_missing_views_non_object_root_and_invalid_snapshot(self) -> None:
        for trace, view in (
            ({}, "full-l0"),
            ({}, "decision-evidence"),
            ({"analyzer_trace": {}}, "model-l0"),
            (
                {
                    "l0_bundle": {},
                    "analyzer_trace": {"l0_model_view": {"evidence_bundle": []}},
                },
                "comparison",
            ),
        ):
            with self.subTest(view=view, trace=trace):
                exit_code, stdout, stderr = _run_inspection(trace, view)
                self.assertEqual(exit_code, 2)
                self.assertEqual(stdout, "")
                self.assertTrue(stderr)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text("[]", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = inspect_trace.main([str(path)])

            self.assertEqual(exit_code, 2)
            with self.assertRaises(SystemExit):
                inspect_trace.main([str(path), "--snapshot", "0"])

    def test_should_report_missing_and_malformed_trace_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            malformed = root / "malformed.json"
            malformed.write_text("{", encoding="utf-8")
            for path in (root / "missing.json", malformed):
                with self.subTest(path=path):
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        exit_code = inspect_trace.main([str(path)])

                    self.assertEqual(exit_code, 2)
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertTrue(stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
