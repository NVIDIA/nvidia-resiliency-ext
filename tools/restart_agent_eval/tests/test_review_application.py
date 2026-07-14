# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-log review application validation and deterministic publication tests."""

from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from _mocks import process_result  # noqa: E402
from restart_agent_eval import review  # noqa: E402
from restart_agent_eval.product_trace import SINGLE_TRACE_SCHEMA  # noqa: E402


class _Clock:
    def now_utc(self):
        return dt.datetime(2026, 7, 20, 2, 3, 4, tzinfo=dt.timezone.utc)


class _DeterministicExecutor:
    def __init__(self, returncode=0) -> None:
        self.returncode = returncode

    def run(self, command, *, cwd, env=None):
        command = list(command)
        trace_path = Path(command[command.index("--trace-json") + 1])
        analysis = {
            "schema_version": "restart_agent_response.v1",
            "decision": "RESTART",
            "decision_basis": "general_retry_available",
            "primary_failure": {},
        }
        trace_path.write_text(
            json.dumps(
                {
                    "schema_version": SINGLE_TRACE_SCHEMA,
                    "request": {"analysis_mode": "terminal"},
                    "analysis_result": analysis,
                    "analyzer_trace": {"layers": {}},
                    "l0_bundle": {},
                }
            ),
            encoding="utf-8",
        )
        return process_result(
            command,
            returncode=self.returncode,
            stdout=json.dumps(analysis),
            stderr="product warning" if self.returncode else "",
        )


def _product_repo(root: Path) -> Path:
    product_repo = root / "product"
    (product_repo / "src/nvidia_resiliency_ext/attribution/restart_agent").mkdir(parents=True)
    return product_repo


class ReviewApplicationTest(unittest.TestCase):
    def test_rejects_relative_missing_log_and_invalid_product_repo(self) -> None:
        application = review.ReviewApplication(
            environment={},
            process_executor=_DeterministicExecutor(),
            clock=_Clock(),
        )
        relative = review.parse_review_args(
            ["--log", "relative.log", "--run-dir", "/tmp/run"], environ={}
        )
        with self.assertRaises(SystemExit):
            application.run(relative)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = review.parse_review_args(
                ["--log", str(root / "missing.log"), "--run-dir", str(root / "run")],
                environ={},
            )
            with self.assertRaises(SystemExit):
                application.run(missing)

            log_path = root / "input.log"
            log_path.write_text("failure\n", encoding="utf-8")
            invalid_repo = review.parse_review_args(
                [
                    "--log",
                    str(log_path),
                    "--product-repo",
                    str(root / "not-product"),
                    "--run-dir",
                    str(root / "run"),
                ],
                environ={},
            )
            with self.assertRaises(SystemExit):
                application.run(invalid_repo)

    def test_rejects_missing_l0_bundle_replay_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "input.log"
            log_path.write_text("failure\n", encoding="utf-8")
            args = review.parse_review_args(
                [
                    "--log",
                    str(log_path),
                    "--l0-bundle-json-in",
                    str(root / "missing-l0-bundle.json"),
                    "--product-repo",
                    str(_product_repo(root)),
                    "--run-dir",
                    str(root / "run"),
                    "deterministic",
                ],
                environ={},
            )

            with self.assertRaises(SystemExit):
                review.ReviewApplication(
                    environment={},
                    process_executor=_DeterministicExecutor(),
                    clock=_Clock(),
                ).run(args)

    def test_deterministic_review_publishes_complete_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "input.log"
            log_path.write_text("failure\n", encoding="utf-8")
            run_dir = root / "run"
            args = review.parse_review_args(
                [
                    "--log",
                    str(log_path),
                    "--product-repo",
                    str(_product_repo(root)),
                    "--run-dir",
                    str(run_dir),
                    "deterministic",
                ],
                environ={},
            )

            exit_code = review.ReviewApplication(
                environment={},
                process_executor=_DeterministicExecutor(),
                clock=_Clock(),
            ).run(args)

            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            review_index = (run_dir / "review_index.md").read_text(encoding="utf-8")
            published = {
                name: (run_dir / name).is_file()
                for name in (
                    "deterministic.result.json",
                    "deterministic.trace.json",
                    "deterministic.review.json",
                    "deterministic.review.md",
                    "review_index.json",
                    "panel_summary.json",
                    "panel_summary.md",
                    "panel_diagnostics.md",
                )
            }

        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["source"]["relative_path"], "input.log")
        self.assertIn("## Start Here", review_index)
        self.assertIn("## Artifact Map", review_index)
        self.assertIn("[open](deterministic.review.md)", review_index)
        for name, exists in published.items():
            with self.subTest(name=name):
                self.assertTrue(exists)

    def test_product_exit_code_is_preserved_after_artifact_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "input.log"
            log_path.write_text("failure\n", encoding="utf-8")
            run_dir = root / "run"
            args = review.parse_review_args(
                [
                    "--log",
                    str(log_path),
                    "--product-repo",
                    str(_product_repo(root)),
                    "--run-dir",
                    str(run_dir),
                    "deterministic",
                ],
                environ={},
            )

            exit_code = review.ReviewApplication(
                environment={},
                process_executor=_DeterministicExecutor(returncode=7),
                clock=_Clock(),
            ).run(args)
            stderr_exists = (run_dir / "deterministic.stderr.log").is_file()

        self.assertEqual(exit_code, 7)
        self.assertTrue(stderr_exists)


if __name__ == "__main__":
    unittest.main()
