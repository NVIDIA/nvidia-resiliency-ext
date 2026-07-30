# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect-all polling, incremental publication, failure, and cancellation tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval import review  # noqa: E402
from restart_agent_eval.product_process import ProcessTimeoutError  # noqa: E402
from restart_agent_eval.product_trace import (  # noqa: E402
    COLLECT_ALL_TRACE_SCHEMA,
    SINGLE_TRACE_SCHEMA,
)
from restart_agent_eval.profiles import expand_targets  # noqa: E402


class _PollingProcess:
    def __init__(self, polls, returncode=0) -> None:
        self.polls = iter(polls)
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self.wait_calls = []

    def poll(self):
        return next(self.polls, self.returncode)

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class _Sleeper:
    def __init__(self) -> None:
        self.durations = []

    def sleep(self, duration):
        self.durations.append(duration)


class _PublishingExecutor:
    def __init__(self, target, process) -> None:
        self.target = target
        self.process = process

    def start(self, command, *, cwd, env, stdout, stderr):
        command = list(command)
        manifest_path = Path(command[command.index("--route-artifact-manifest") + 1])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        route_paths = manifest["routes"][self.target.name]
        result_path = Path(route_paths["result_json"])
        trace_path = Path(route_paths["trace_json"])
        analysis = {"schema_version": "restart_agent_response.v1", "decision": "RESTART"}
        result_path.write_text(json.dumps(analysis), encoding="utf-8")
        trace_path.write_text(
            json.dumps(
                {
                    "schema_version": SINGLE_TRACE_SCHEMA,
                    "request": {},
                    "analysis_result": analysis,
                    "analyzer_trace": {},
                    "l0_bundle": {},
                }
            ),
            encoding="utf-8",
        )
        batch = {
            "schema_version": "restart_agent_collect_all.v1",
            "deterministic_result": analysis,
            "model_results": [],
            "shared_analysis": {"routing_mode": "collect_all"},
        }
        batch_result_path = Path(command[command.index("--result-json") + 1])
        batch_trace_path = Path(command[command.index("--trace-json") + 1])
        batch_result_path.write_text(json.dumps(batch), encoding="utf-8")
        batch_trace_path.write_text(
            json.dumps(
                {
                    "schema_version": COLLECT_ALL_TRACE_SCHEMA,
                    "request": {},
                    "collect_all_result": batch,
                    "analyzer_trace": {},
                    "l0_bundle": {},
                }
            ),
            encoding="utf-8",
        )
        incremental_dir = Path(command[command.index("--incremental-artifact-dir") + 1])
        incremental_dir.mkdir(parents=True)
        (incremental_dir / "events.jsonl").write_text(
            json.dumps(
                {
                    "event": "route_completed",
                    "route_id": self.target.name,
                    "result_artifact": result_path.name,
                    "trace_artifact": trace_path.name,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        stdout.write(json.dumps(batch))
        stdout.flush()
        return self.process


class _FailureExecutor:
    def __init__(self, process, *, poll_error=None) -> None:
        self.process = process
        self.poll_error = poll_error

    def start(self, command, *, cwd, env, stdout, stderr):
        if self.poll_error is not None:
            self.process.poll = self.poll_error
        stderr.write("provider process failed")
        stderr.flush()
        return self.process


def _run_collect_all(root, executor, *, on_target_ready=None, sleeper=None):
    target = expand_targets(["qwen235b"])[0]
    args = review.parse_review_args(["--log", str(root / "input.log"), "qwen235b"])
    (root / "input.log").write_text("failure\n", encoding="utf-8")
    return review.run_collect_all_targets(
        targets=[target],
        model_targets=[target],
        args=args,
        log_path=root / "input.log",
        product_repo=root,
        run_dir=root,
        l0_bundle_out=root / "l0_bundle.json",
        process_executor=executor,
        environment={},
        sleeper=sleeper or _Sleeper(),
        on_target_ready=on_target_ready,
    )


class ReviewProcessLifecycleTest(unittest.TestCase):
    def test_route_is_published_during_process_polling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = expand_targets(["qwen235b"])[0]
            process = _PollingProcess([None, 0])
            sleeper = _Sleeper()
            ready = []

            runs = _run_collect_all(
                root,
                _PublishingExecutor(target, process),
                sleeper=sleeper,
                on_target_ready=lambda route, completed, paths: ready.append(
                    (route.name, completed.returncode, paths["result_json"].is_file())
                ),
            )

        self.assertEqual(ready, [("qwen235b", 0, True)])
        self.assertEqual(set(runs), {"qwen235b"})
        self.assertEqual(sleeper.durations, [0.1])

    def test_nonzero_process_exit_preserves_stderr_and_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = _PollingProcess([7], returncode=7)

            with self.assertRaises(RuntimeError):
                _run_collect_all(root, _FailureExecutor(process))

            self.assertEqual(
                (root / "restart_agent.stderr.log").read_text(encoding="utf-8"),
                "provider process failed",
            )

    def test_polling_exception_terminates_then_kills_unresponsive_process(self) -> None:
        class _UnresponsiveProcess(_PollingProcess):
            def wait(self, timeout=None):
                self.wait_calls.append(timeout)
                if timeout is not None:
                    raise ProcessTimeoutError("still running")
                return self.returncode

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = _UnresponsiveProcess([], returncode=1)

            def fail_poll():
                raise RuntimeError("poll failed")

            with self.assertRaises(RuntimeError):
                _run_collect_all(root, _FailureExecutor(process, poll_error=fail_poll))

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertEqual(process.wait_calls, [5, None])


if __name__ == "__main__":
    unittest.main()
