# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decision stability cohorts, comparability, and report CLI tests."""

from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _bootstrap import configure_test_imports

configure_test_imports()

from _builders import recovery_assessment  # noqa: E402
from restart_agent_eval import stability  # noqa: E402


class DecisionStabilityTest(unittest.TestCase):
    def test_summary_timestamp_uses_injected_clock(self) -> None:
        class _Clock:
            def now_utc(self):
                return dt.datetime(2026, 7, 19, 1, 2, 3, tzinfo=dt.timezone.utc)

        summary = stability.build_stability_summary([], clock=_Clock())

        self.assertEqual(summary["generated_at_utc"], "2026-07-19T01:02:03+00:00")

    def _write_run(
        self,
        root: Path,
        run_id: str,
        *,
        decision: str = "RESTART",
        retry_outlook: str = "may_recover",
        l1_usable: bool = True,
        endpoint: str = "ok",
        l0b_sha256: str = "sha256:l0b",
        request_sha256: str = "sha256:request",
        product_dirty: bool = False,
        tool_calls: int = 0,
    ) -> Path:
        run_dir = root / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "l0_bundle.json").write_text('{"bundle":"same"}\n', encoding="utf-8")
        trace_path = run_dir / "model.route.trace.json"
        assessment = (
            recovery_assessment(
                failure_domain="workload",
                retry_outlook=retry_outlook,
                failure_domain_confidence=80,
                retry_outlook_confidence=80,
                rationale="test",
            )
            if l1_usable
            else {}
        )
        transcript = [
            {
                "event_type": "model_request",
                "model_turn": 1,
                "payload_sha256": request_sha256,
            }
        ]
        if tool_calls:
            transcript.extend(
                {
                    "event_type": "tool_result",
                    "name": "read_window",
                    "model_turn": index + 1,
                }
                for index in range(tool_calls)
            )
        trace_path.write_text(
            json.dumps(
                {
                    "schema_version": "restart_agent_cli_trace.v1",
                    "request": {
                        "analysis_intent": "terminal",
                        "job_id": "job-a",
                    },
                    "analysis_result": {
                        "decision": decision,
                        "primary_failure": {
                            "fine_class": "checkpoint_read",
                            "line": 12,
                        },
                        "root_cause_assessment": {"status": "supported_but_unconfirmed"},
                        "model_recovery_assessment": assessment,
                    },
                    "analyzer_trace": {
                        "l1": {
                            "success": l1_usable,
                            "interaction_transcript": transcript,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        manifest = {
            "run_id": run_id,
            "created_at_utc": f"2026-07-18T00:00:{int(run_id[-2:]):02d}+00:00",
            "source": {"sha256": "source-sha"},
            "repositories": {"product": {"commit": "product-commit", "dirty": product_dirty}},
        }
        panel = {
            "schema_version": "restart_agent_panel.v1",
            "source_log_sha256": "source-sha",
            "product_commit": "product-commit",
            "run_manifest": manifest,
            "restart_agent_config": {
                "config_fingerprint": "sha256:config",
                "effective_config": {
                    "model_routes": [
                        {
                            "route_id": "qwen397b",
                            "model": "nvidia/qwen/397b",
                            "request": {"temperature": 0.2},
                            "tools": {"enabled": True, "max_rounds": 8},
                        }
                    ]
                },
            },
            "rows": [
                {
                    "target": "deterministic",
                    "model": None,
                    "decision": "RESTART",
                },
                {
                    "target": "qwen397b",
                    "model": "nvidia/qwen/397b",
                    "decision": decision,
                    "decision_basis": "general_retry_available",
                    "l1_output_usable": l1_usable,
                    "l1_execution_status": "ok" if l1_usable else "failed",
                    "l1_semantic_primary_class": "checkpoint_read",
                    "l1_semantic_primary_line": 12,
                    "current_root_fingerprint": "root-a",
                    "model_calls": 1 + tool_calls,
                    "tool_calls": tool_calls,
                    "no_new_prompt_line_calls": tool_calls,
                    "l1_wall_clock_s": 10.0 + tool_calls,
                    "total_tokens": 1000 + tool_calls * 500,
                    "endpoint_reliability": endpoint,
                    "failed_endpoint_attempts": 0 if endpoint == "ok" else 1,
                    "retried_model_calls": 0,
                    "timeout_model_calls": 0 if endpoint == "ok" else 1,
                    "gold_l1_core_semantic": l1_usable,
                    "gold_l4_action_correct": decision == "RESTART",
                    "gold_case_id": "case-a",
                    "l0b_projection_metrics": {
                        "projection_integrity": {"deterministic_payload_sha256": l0b_sha256}
                    },
                    "artifacts": {"trace_json": str(trace_path)},
                },
            ],
        }
        (run_dir / "panel_summary.json").write_text(json.dumps(panel), encoding="utf-8")
        return run_dir

    def test_identical_repetitions_are_observed_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [self._write_run(root, f"run-{index:02d}") for index in range(1, 4)]
            summary = stability.build_stability_summary(runs, minimum_samples=3)

        cohort = summary["cohorts"][0]
        self.assertEqual(cohort["status"], "observed_stable")
        self.assertEqual(cohort["decision_stability"]["modal_agreement"], 1.0)
        self.assertEqual(cohort["semantic_stability"]["exact_policy_tuple"]["modal_agreement"], 1.0)
        self.assertEqual(cohort["decision_stability"]["sequential_flips"], 0)

    def test_semantic_and_action_flips_are_measured_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [
                self._write_run(root, "run-01"),
                self._write_run(
                    root,
                    "run-02",
                    decision="STOP",
                    retry_outlook="cannot_recover",
                ),
                self._write_run(root, "run-03"),
            ]
            summary = stability.build_stability_summary(runs, minimum_samples=3)

        cohort = summary["cohorts"][0]
        self.assertEqual(cohort["status"], "observed_unstable")
        self.assertEqual(cohort["decision_stability"]["distribution"], {"RESTART": 2, "STOP": 1})
        self.assertEqual(cohort["decision_stability"]["sequential_flips"], 2)
        retry_outlook = cohort["semantic_stability"]["fields"][
            "retry_outlook_without_workload_change"
        ]
        self.assertEqual(
            retry_outlook["distribution"],
            {"may_recover": 2, "cannot_recover": 1},
        )

    def test_endpoint_failure_does_not_become_semantic_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [
                self._write_run(root, "run-01"),
                self._write_run(root, "run-02"),
                self._write_run(
                    root,
                    "run-03",
                    l1_usable=False,
                    endpoint="failed",
                ),
            ]
            summary = stability.build_stability_summary(runs, minimum_samples=3)

        cohort = summary["cohorts"][0]
        self.assertEqual(cohort["availability"]["usable_l1_count"], 2)
        self.assertEqual(cohort["semantic_stability"]["exact_policy_tuple"]["count"], 2)
        self.assertEqual(
            cohort["endpoint_reliability"]["status_distribution"],
            {"ok": 2, "failed": 1},
        )
        self.assertEqual(cohort["status"], "observed_unstable")

    def test_input_change_creates_a_separate_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [
                self._write_run(root, "run-01"),
                self._write_run(root, "run-02", l0b_sha256="sha256:different"),
            ]
            summary = stability.build_stability_summary(runs, minimum_samples=2)

        self.assertEqual(summary["cohort_count"], 2)
        self.assertEqual({cohort["sample_count"] for cohort in summary["cohorts"]}, {1})

    def test_dirty_product_marks_comparison_provisional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [
                self._write_run(root, "run-01", product_dirty=True),
                self._write_run(root, "run-02", product_dirty=True),
            ]
            summary = stability.build_stability_summary(runs, minimum_samples=2)

        cohort = summary["cohorts"][0]
        self.assertEqual(cohort["comparability"]["status"], "provisional_dirty_product")
        self.assertEqual(cohort["status"], "observed_stable")

    def test_cli_discovers_runs_and_writes_report_without_model_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root, "run-01")
            self._write_run(root, "run-02")
            output = root / "report"
            exit_code = stability.main(
                [
                    "--runs-root",
                    str(root),
                    "--route",
                    "qwen397b",
                    "--minimum-samples",
                    "2",
                    "--output-dir",
                    str(output),
                ]
            )

            payload = json.loads((output / "stability_summary.json").read_text())
            markdown = (output / "stability_summary.md").read_text()

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["schema_version"], stability.SCHEMA_VERSION)
        self.assertIn("Stability and correctness are independent", markdown)

    def test_discovery_deduplicates_explicit_and_root_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._write_run(root, "run-01")

            discovered = stability.discover_run_dirs([run], root)

            self.assertEqual(discovered, [run.resolve()])

    def test_discovery_rejects_missing_explicit_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with self.assertRaises(SystemExit):
                stability.discover_run_dirs([root / "missing-run"], None)

    def test_discovery_rejects_missing_runs_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with self.assertRaises(SystemExit):
                stability.discover_run_dirs([], root / "missing-root")

    def test_summary_rejects_malformed_runs_without_losing_valid_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = self._write_run(root, "run-01")
            malformed = root / "run-02"
            malformed.mkdir()
            (malformed / "panel_summary.json").write_text("{", encoding="utf-8")

            summary = stability.build_stability_summary(
                [valid, malformed],
                route_filters=["qwen397b"],
                minimum_samples=2,
            )

        self.assertEqual(summary["accepted_sample_count"], 1)
        self.assertEqual(summary["cohort_count"], 1)
        self.assertEqual(summary["cohorts"][0]["status"], "insufficient_samples")
        self.assertEqual(len(summary["rejected_runs"]), 1)

    def test_cli_rejects_empty_run_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with self.assertRaises(SystemExit):
                stability.main(["--runs-root", str(root)])

    def test_cli_rejects_unmatched_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root, "run-01")

            with self.assertRaises(SystemExit):
                stability.main(
                    [
                        "--runs-root",
                        str(root),
                        "--route",
                        "missing-route",
                    ]
                )

    def test_cli_rejects_nonpositive_sample_and_latest_limits(self) -> None:
        for arguments in (("--minimum-samples", "0"), ("--latest", "0")):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                stability.main(list(arguments))

    def test_default_output_directory_uses_injected_clock(self) -> None:
        class _Clock:
            def now_utc(self):
                return dt.datetime(2026, 7, 19, 1, 2, 3, tzinfo=dt.timezone.utc)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._write_run(root, "run-01")
            with mock.patch.object(stability, "SYSTEM_CLOCK", _Clock()):
                exit_code = stability.main(
                    [str(run), "--minimum-samples", "1", "--route", "qwen397b"]
                )

            output = root / "stability" / "20260719T010203000000Z"
            self.assertEqual(exit_code, 0)
            self.assertTrue((output / "stability_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
