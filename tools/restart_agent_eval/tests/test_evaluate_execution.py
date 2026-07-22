# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation product execution, corpus lifecycle, and failure-isolation tests."""

from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from _evaluation_fixtures import write_evaluation_case  # noqa: E402
from _mocks import process_result  # noqa: E402
from restart_agent_eval import evaluate  # noqa: E402


class _Clock:
    def __init__(self) -> None:
        self.values = iter(
            (
                dt.datetime(2026, 7, 20, 1, 2, 3, tzinfo=dt.timezone.utc),
                dt.datetime(2026, 7, 20, 1, 2, 4, tzinfo=dt.timezone.utc),
            )
        )

    def now_utc(self):
        return next(self.values)


class _PublishingExecutor:
    def __init__(
        self,
        *,
        verdict=None,
        returncode=0,
        result_count=1,
        malformed_result=False,
        trace_count=1,
    ) -> None:
        self.verdict = verdict or {
            "schema_version": "restart_agent_response.v1",
            "decision": "RESTART",
        }
        self.returncode = returncode
        self.result_count = result_count
        self.malformed_result = malformed_result
        self.trace_count = trace_count
        self.calls = []

    def run(self, command, *, cwd, env=None):
        command = list(command)
        self.calls.append((command, cwd, dict(env or {})))
        run_dir = Path(command[command.index("--run-dir") + 1])
        run_dir.mkdir(parents=True, exist_ok=True)
        for index in range(self.result_count):
            path = run_dir / f"model.route-{index}.result.json"
            path.write_text(
                "{" if self.malformed_result else json.dumps(self.verdict),
                encoding="utf-8",
            )
        for index in range(self.trace_count):
            (run_dir / f"model.route-{index}.trace.json").write_text(
                json.dumps(
                    {
                        "schema_version": "restart_agent_cli_trace.v1",
                        "request": {},
                        "analysis_result": self.verdict,
                        "analyzer_trace": {},
                        "l0_bundle": {},
                    }
                ),
                encoding="utf-8",
            )
        return process_result(
            command,
            returncode=self.returncode,
            stderr="product failed" if self.returncode else "",
        )


def _simplify_gold(gold_root: Path) -> None:
    path = gold_root / "cases" / "case-a.log" / "gold.json"
    label = json.loads(path.read_text(encoding="utf-8"))
    for field in (
        "decision",
        "recovery_assessment_expectation",
        "retry_policy_expectation",
        "l0b_expectation",
        "primary_anchor_expectation",
    ):
        label.pop(field, None)
    path.write_text(json.dumps(label), encoding="utf-8")


def _args(root: Path, log_root: Path, gold_root: Path, *extra: str):
    return evaluate.parse_evaluation_args(
        [
            "--log-root",
            str(log_root),
            "--gold-root",
            str(gold_root),
            "--artifacts-dir",
            str(root / "artifacts"),
            "--product-repo",
            str(root / "product"),
            *extra,
        ],
        environ={},
    )


class ProductRunnerTest(unittest.TestCase):
    def test_runner_publishes_one_result_and_optional_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_root, gold_root = write_evaluation_case(root)
            case = evaluate.discover_cases(log_root, gold_root)[0]
            executor = _PublishingExecutor(trace_count=1)
            runner = evaluate.ProductRunner(
                product_repo=root / "product",
                target="gpt",
                artifacts_dir=root / "artifacts",
                python="python-test",
                process_executor=executor,
                environment={"KEY": "value"},
            )

            verdict, result_path, trace_path = runner(case)

        self.assertEqual(verdict["decision"], "RESTART")
        self.assertEqual(result_path.name, "model.route-0.result.json")
        self.assertEqual(trace_path.name, "model.route-0.trace.json")
        command, cwd, environment = executor.calls[0]
        self.assertEqual(command[0], "python-test")
        self.assertEqual(cwd, evaluate.EVAL_ROOT)
        self.assertEqual(environment, {"KEY": "value"})

    def test_runner_rejects_process_failure_result_count_and_malformed_json(self) -> None:
        scenarios = (
            _PublishingExecutor(returncode=7),
            _PublishingExecutor(result_count=0, trace_count=0),
            _PublishingExecutor(result_count=2, trace_count=0),
            _PublishingExecutor(malformed_result=True, trace_count=0),
        )
        for executor in scenarios:
            with self.subTest(executor=executor.__dict__):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    log_root, gold_root = write_evaluation_case(root)
                    case = evaluate.discover_cases(log_root, gold_root)[0]
                    runner = evaluate.ProductRunner(
                        product_repo=root / "product",
                        target="gpt",
                        artifacts_dir=root / "artifacts",
                        python="python-test",
                        process_executor=executor,
                        environment={},
                    )
                    with self.assertRaises((RuntimeError, json.JSONDecodeError)):
                        runner(case)

    def test_runner_omits_trace_when_multiple_trace_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_root, gold_root = write_evaluation_case(root)
            case = evaluate.discover_cases(log_root, gold_root)[0]
            runner = evaluate.ProductRunner(
                product_repo=root / "product",
                target="gpt",
                artifacts_dir=root / "artifacts",
                python="python-test",
                process_executor=_PublishingExecutor(trace_count=2),
                environment={},
            )

            _, _, trace_path = runner(case)

        self.assertIsNone(trace_path)


class EvaluationApplicationTest(unittest.TestCase):
    def test_scored_run_writes_manifest_results_and_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_root, gold_root = write_evaluation_case(root)
            _simplify_gold(gold_root)
            args = _args(root, log_root, gold_root)
            application = evaluate.EvaluationApplication(
                environment={"KEY": "value"},
                process_executor=_PublishingExecutor(),
                clock=_Clock(),
            )

            exit_code = application.run(args)
            manifest = json.loads((root / "artifacts" / "run_manifest.json").read_text())
            aggregate = json.loads((root / "artifacts" / "aggregate.json").read_text())
            result_lines = (root / "artifacts" / "case_results.jsonl").read_text().splitlines()

        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["run_id"], "20260720T010203Z")
        self.assertEqual(
            manifest["case_status_counts"], {"scored": 1, "unavailable": 0, "analyzer_error": 0}
        )
        self.assertEqual(aggregate["decision_accuracy"], 1.0)
        self.assertEqual(len(result_lines), 1)

    def test_fail_on_mismatch_returns_one_after_publishing_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_root, gold_root = write_evaluation_case(root)
            _simplify_gold(gold_root)
            args = _args(root, log_root, gold_root, "--fail-on-mismatch")
            application = evaluate.EvaluationApplication(
                environment={},
                process_executor=_PublishingExecutor(
                    verdict={
                        "schema_version": "restart_agent_response.v1",
                        "decision": "STOP",
                    }
                ),
                clock=_Clock(),
            )

            exit_code = application.run(args)

        self.assertEqual(exit_code, 1)

    def test_analyzer_error_and_unavailable_case_do_not_abort_corpus(self) -> None:
        scenarios = (
            (True, _PublishingExecutor(returncode=7), "analyzer_error"),
            (False, _PublishingExecutor(), "unavailable"),
        )
        for with_log, executor, status in scenarios:
            with self.subTest(status=status):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    log_root, gold_root = write_evaluation_case(root, with_log=with_log)
                    args = _args(root, log_root, gold_root)
                    exit_code = evaluate.EvaluationApplication(
                        environment={},
                        process_executor=executor,
                        clock=_Clock(),
                    ).run(args)
                    result = json.loads(
                        (root / "artifacts" / "case_results.jsonl").read_text().splitlines()[0]
                    )

                self.assertEqual(exit_code, 2)
                self.assertEqual(result["status"], status)

    def test_missing_roots_and_empty_corpus_are_rejected(self) -> None:
        args = evaluate.parse_evaluation_args([], environ={})
        with self.assertRaises(SystemExit):
            evaluate.EvaluationApplication(
                environment={},
                process_executor=_PublishingExecutor(),
                clock=_Clock(),
            ).run(args)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_root = root / "logs"
            gold_root = root / "gold"
            log_root.mkdir()
            gold_root.mkdir()
            exit_code = evaluate.EvaluationApplication(
                environment={},
                process_executor=_PublishingExecutor(),
                clock=_Clock(),
            ).run(_args(root, log_root, gold_root))

        self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
