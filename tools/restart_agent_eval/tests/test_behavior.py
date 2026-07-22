# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior-fixture worker failures, normalization boundaries, and drift checks."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _bootstrap import configure_test_imports

configure_test_imports()

from _mocks import process_result  # noqa: E402
from restart_agent_eval import behavior  # noqa: E402


class _Executor:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    def run(self, command, *, cwd, env=None):
        self.calls.append((list(command), cwd, dict(env or {})))
        return self.result


def _fixture_case(root: Path):
    log_root = root / "logs"
    gold_root = root / "gold"
    log_path = log_root / "cases" / "case.log"
    gold_path = gold_root / "cases" / "case.log" / "gold.json"
    log_path.parent.mkdir(parents=True)
    gold_path.parent.mkdir(parents=True)
    log_path.write_text("failure\n", encoding="utf-8")
    gold_path.write_text("{}", encoding="utf-8")
    fixture = {
        "schema_version": behavior.FIXTURE_SCHEMA_VERSION,
        "source": {"relative_path": None, "sha256": "abc", "byte_size": 8},
        "behavior": {"decision": "RESTART"},
    }
    argv = [
        "capture_behavior_fixtures.py",
        "--log-root",
        str(log_root),
        "--gold-root",
        str(gold_root),
        "--product-repo",
        str(root / "product"),
    ]
    return fixture, argv, gold_path.parent / behavior.FIXTURE_NAME


class BehaviorFixtureTest(unittest.TestCase):
    def test_worker_failure_raises_runtime_error(self) -> None:
        failed = _Executor(process_result(returncode=2, stdout="ignored", stderr="worker failed"))

        with self.assertRaises(RuntimeError):
            behavior.build_fixture(Path("input.log"), Path("product"), process_executor=failed)

    def test_worker_failure_falls_back_to_stdout_diagnostic(self) -> None:
        stdout_only = _Executor(process_result(returncode=2, stdout="stdout failure"))

        with self.assertRaises(RuntimeError):
            behavior.build_fixture(Path("input.log"), Path("product"), process_executor=stdout_only)

    def test_worker_rejects_malformed_json_output(self) -> None:
        malformed = _Executor(process_result(stdout="{"))

        with self.assertRaises(json.JSONDecodeError):
            behavior.build_fixture(Path("input.log"), Path("product"), process_executor=malformed)

    def test_worker_rejects_non_object_output(self) -> None:
        non_object = _Executor(process_result(stdout="[]"))

        with self.assertRaises(TypeError):
            behavior.build_fixture(Path("input.log"), Path("product"), process_executor=non_object)

    def test_worker_environment_prepends_harness_source_and_preserves_existing_path(self) -> None:
        executor = _Executor(process_result(stdout='{"schema_version":"fixture.v1"}'))

        behavior.build_fixture(
            Path("input.log"),
            Path("product"),
            process_executor=executor,
            environment={"PYTHONPATH": "/existing"},
            python_executable="python-test",
        )

        command, cwd, environment = executor.calls[0]
        self.assertEqual(command[0], "python-test")
        self.assertEqual(cwd, Path("product").resolve())
        self.assertTrue(environment["PYTHONPATH"].endswith(":/existing"))

    def test_worker_capture_rejects_product_without_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "input.log"
            log_path.write_text("failure\n", encoding="utf-8")

            with self.assertRaises(SystemExit):
                behavior.build_fixture_in_worker(log_path, root / "not-a-product")

    def test_main_writes_behavior_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, argv, fixture_path = _fixture_case(root)
            output = io.StringIO()
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(behavior, "build_fixture", return_value=fixture),
                contextlib.redirect_stdout(output),
            ):
                exit_code = behavior.main()

            written = json.loads(fixture_path.read_text(encoding="utf-8"))

            self.assertEqual(exit_code, 0)
            self.assertEqual(written["source"]["relative_path"], "cases/case.log")

    def test_main_checks_matching_behavior_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, argv, fixture_path = _fixture_case(root)
            expected = json.loads(json.dumps(fixture))
            expected["source"]["relative_path"] = "cases/case.log"
            fixture_path.write_text(json.dumps(expected), encoding="utf-8")
            output = io.StringIO()
            with (
                mock.patch.object(sys, "argv", [*argv, "--check"]),
                mock.patch.object(behavior, "build_fixture", return_value=fixture),
                contextlib.redirect_stdout(output),
            ):
                exit_code = behavior.main()

            self.assertEqual(exit_code, 0)
            self.assertIn("OK", output.getvalue())

    def test_main_reports_missing_behavior_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, argv, _ = _fixture_case(root)
            with (
                mock.patch.object(sys, "argv", [*argv, "--check"]),
                mock.patch.object(behavior, "build_fixture", return_value=fixture),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = behavior.main()

            self.assertEqual(exit_code, 1)

    def test_main_reports_different_behavior_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, argv, fixture_path = _fixture_case(root)
            fixture_path.write_text(
                json.dumps(
                    {
                        "source": {"relative_path": "case.log"},
                        "behavior": {"decision": "STOP"},
                    }
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with (
                mock.patch.object(sys, "argv", [*argv, "--check"]),
                mock.patch.object(behavior, "build_fixture", return_value=fixture),
                contextlib.redirect_stdout(output),
            ):
                exit_code = behavior.main()

            self.assertEqual(exit_code, 1)
            self.assertIn("DIFF", output.getvalue())

    def test_main_rejects_empty_gold_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_root = root / "logs"
            gold_root = root / "gold"
            log_root.mkdir()
            gold_root.mkdir()
            argv = [
                "capture_behavior_fixtures.py",
                "--log-root",
                str(log_root),
                "--gold-root",
                str(gold_root),
                "--product-repo",
                str(root / "product"),
            ]
            with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                behavior.main()

    def test_main_rejects_gold_without_source_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_root = root / "logs"
            gold_root = root / "gold"
            log_root.mkdir()
            gold_root.mkdir()
            gold_path = gold_root / "missing.log" / "gold.json"
            gold_path.parent.mkdir(parents=True)
            gold_path.write_text("{}", encoding="utf-8")
            argv = [
                "capture_behavior_fixtures.py",
                "--log-root",
                str(log_root),
                "--gold-root",
                str(gold_root),
                "--product-repo",
                str(root / "product"),
            ]

            with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                behavior.main()


if __name__ == "__main__":
    unittest.main()
