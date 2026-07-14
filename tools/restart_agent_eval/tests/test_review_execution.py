# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Review execution, route publication, and process-diagnostic scenarios."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _bootstrap import configure_test_imports

configure_test_imports()

from _assertions import assert_json_file, assert_paths_exist  # noqa: E402
from _mocks import process_result as _process_result  # noqa: E402
from _review_execution_fixtures import (  # noqa: E402
    collect_all_execution,
    collect_all_result,
    collect_all_trace,
    completed_route_result,
    write_route_then_batch,
)
from restart_agent_eval import review as review_log  # noqa: E402
from restart_agent_eval.profiles import expand_targets  # noqa: E402


class ReviewExecutionTest(unittest.TestCase):
    def test_collect_all_invokes_product_once_with_generated_config(self) -> None:
        with collect_all_execution() as (root, _, model_targets, _, run):
            command = run.call_args.kwargs["cmd"]
            config = assert_json_file(
                self,
                root / "restart_agent.json",
                required_fields=("schema_version", "routing", "model_routes"),
            )

            self.assertEqual(run.call_count, 1)
            self.assertIn("--config", command)
            self.assertNotIn("--max-parallel-models", command)
            self.assertEqual(config["schema_version"], "restart_agent_config.v1")
            self.assertEqual(config["routing"]["max_parallel_models"], len(model_targets))
            self.assertEqual(len(config["model_routes"]), len(model_targets))

    def test_collect_all_forwards_l0_bundle_replay_path(self) -> None:
        with collect_all_execution(replay_l0=True) as (root, _, _, _, run):
            command = run.call_args.kwargs["cmd"]

            self.assertEqual(
                command[command.index("--l0-bundle-json-in") + 1],
                str(root / "prior_l0_bundle.json"),
            )
            self.assertEqual(
                command[command.index("--l0-bundle-json-out") + 1],
                str(root / "l0_bundle.json"),
            )

    def test_collect_all_publishes_shared_batch_artifacts(self) -> None:
        with collect_all_execution() as (root, _, _, _, _):
            assert_paths_exist(
                self,
                tuple(
                    root / name
                    for name in (
                        "restart_agent.result.json",
                        "restart_agent.trace.json",
                        "decision_evidence.json",
                        "l0_model_view.json",
                    )
                ),
            )
            self.assertEqual(
                (root / "restart_agent.stderr.log").read_text(encoding="utf-8"),
                "batch warning",
            )
            self.assertFalse((root / "restart_agent.trace.pretty.json").exists())
            self.assertFalse((root / "restart_agent.stdout.raw.txt").exists())

    def test_collect_all_publishes_each_route_contract(self) -> None:
        with collect_all_execution() as (_, targets, _, target_runs, _):
            self.assertEqual(set(target_runs), {target.name for target in targets})
            for target in targets:
                with self.subTest(target=target.name):
                    paths = target_runs[target.name][1]
                    assert_paths_exist(
                        self,
                        (paths["result_json"], paths["trace_json"]),
                    )
                    self.assertNotIn("trace_pretty_json", paths)
                    self.assertNotIn("stdout_raw", paths)
                    self.assertNotIn("stderr_log", paths)

    def test_collect_all_silent_success_has_no_stderr_artifact(self) -> None:
        target = expand_targets(["qwen235b"])[0]
        args = review_log.parse_review_args(["--log", "/tmp/example.log", "qwen235b"])
        batch_result = collect_all_result({}, [target])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "input.log"
            log_path.write_text("RuntimeError: failure\n", encoding="utf-8")

            def fake_run(*, cmd, **kwargs):
                command = cmd
                trace_path = Path(command[command.index("--trace-json") + 1])
                l0_path = Path(command[command.index("--l0-bundle-json-out") + 1])
                trace_path.write_text(
                    json.dumps(
                        collect_all_trace(
                            log_path,
                            batch_result,
                            {"model_routes": {target.name: {"analyzer_trace": {"layers": {}}}}},
                            l0_bundle={},
                        )
                    ),
                    encoding="utf-8",
                )
                l0_path.write_text("{}\n", encoding="utf-8")
                return _process_result(command, stdout=json.dumps(batch_result))

            with mock.patch.object(
                review_log,
                "_run_process_with_live_events",
                side_effect=fake_run,
            ) as run:
                review_log.run_collect_all_targets(
                    targets=[target],
                    model_targets=[target],
                    args=args,
                    log_path=log_path,
                    product_repo=root,
                    run_dir=root,
                    l0_bundle_out=root / "l0_bundle.json",
                )

            self.assertNotIn("--summary", run.call_args.kwargs["cmd"])
            self.assertFalse((root / "restart_agent.stderr.log").exists())

    def test_collect_all_materializes_route_review_before_batch_finishes(self) -> None:
        target = expand_targets(["qwen235b"])[0]
        args = review_log.parse_review_args(["--log", "/tmp/example.log", "qwen235b"])
        analysis_result = {
            "schema_version": "restart_agent_response.v1",
            "decision": "RESTART",
            "decision_basis": "general_retry_available",
        }
        route_result = completed_route_result(target, analysis_result)
        batch_result = collect_all_result(analysis_result, [target])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "input.log"
            log_path.write_text("RuntimeError: failure\n", encoding="utf-8")
            published = []

            def target_ready(ready_target, completed, paths):
                review_log.write_review_summary(
                    target=ready_target,
                    completed=completed,
                    paths=paths,
                    source_log=log_path,
                )
                self.assertFalse((root / "restart_agent.result.json").exists())
                self.assertTrue(paths["result_json"].is_file())
                self.assertTrue(paths["trace_json"].is_file())
                self.assertTrue(paths["review_json"].is_file())
                self.assertTrue(paths["review_md"].is_file())
                published.append(ready_target.name)

            def fake_run(*, cmd, on_event, **kwargs):
                return write_route_then_batch(
                    cmd,
                    log_path=log_path,
                    target=target,
                    analysis_result=analysis_result,
                    batch_result=batch_result,
                    on_event=on_event,
                )

            with mock.patch.object(
                review_log,
                "_run_process_with_live_events",
                side_effect=fake_run,
            ):
                target_runs = review_log.run_collect_all_targets(
                    targets=[target],
                    model_targets=[target],
                    args=args,
                    log_path=log_path,
                    product_repo=root,
                    run_dir=root,
                    l0_bundle_out=root / "l0_bundle.json",
                    on_target_ready=target_ready,
                )

            self.assertEqual(published, [target.name])
            self.assertIn(target.name, target_runs)

    def test_failed_process_without_stderr_gets_diagnostic_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "restart_agent.stderr.log"
            completed = _process_result(["restart-agent"], returncode=7)

            written = review_log.persist_process_diagnostics(completed, path)

            self.assertTrue(written)
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "restart_agent process exited with code 7 without stderr\n",
            )
