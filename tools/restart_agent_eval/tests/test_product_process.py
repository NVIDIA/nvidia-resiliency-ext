# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval import product_process  # noqa: E402


class ProductProcessTest(unittest.TestCase):
    def test_product_cli_command_targets_restart_agent_module(self) -> None:
        actual = product_process.product_cli_command("python3", "/logs/input.log")

        self.assertEqual(
            actual,
            [
                "python3",
                "-m",
                "nvidia_resiliency_ext.attribution.restart_agent.cli",
                "/logs/input.log",
            ],
        )

    def test_run_normalizes_subprocess_result(self) -> None:
        completed = subprocess.CompletedProcess(["tool"], 7, "out", "err")
        with mock.patch.object(product_process.subprocess, "run", return_value=completed) as run:
            result = product_process.SubprocessExecutor().run(
                ["tool", "arg"],
                cwd=Path("/work"),
                env={"KEY": "value"},
            )

        expected = product_process.ProcessResult(("tool", "arg"), 7, "out", "err")

        self.assertEqual(
            result,
            expected,
        )
        run.assert_called_once_with(
            ["tool", "arg"],
            cwd=Path("/work"),
            env={"KEY": "value"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_started_process_translates_timeout_and_delegates_shutdown(self) -> None:
        process = mock.Mock()
        process.wait.side_effect = subprocess.TimeoutExpired(["tool"], 2.0)
        with (
            tempfile.TemporaryFile(mode="w+") as stdout,
            tempfile.TemporaryFile(mode="w+") as stderr,
        ):
            with mock.patch.object(product_process.subprocess, "Popen", return_value=process):
                running = product_process.SubprocessExecutor().start(
                    ["tool"],
                    cwd=Path("/work"),
                    env={"KEY": "value"},
                    stdout=stdout,
                    stderr=stderr,
                )

        with self.assertRaises(product_process.ProcessTimeoutError):
            running.wait(timeout=2.0)
        running.terminate()
        running.kill()
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()

    def test_started_process_exposes_poll_and_successful_wait(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        with (
            tempfile.TemporaryFile(mode="w+") as stdout,
            tempfile.TemporaryFile(mode="w+") as stderr,
            mock.patch.object(product_process.subprocess, "Popen", return_value=process) as popen,
        ):
            running = product_process.SubprocessExecutor().start(
                ["tool", "arg"],
                cwd=Path("/work"),
                env={"KEY": "value"},
                stdout=stdout,
                stderr=stderr,
            )

        self.assertIsNone(running.poll())
        self.assertEqual(running.wait(), 0)
        popen.assert_called_once_with(
            ["tool", "arg"],
            cwd=Path("/work"),
            env={"KEY": "value"},
            text=True,
            stdout=stdout,
            stderr=stderr,
        )


if __name__ == "__main__":
    unittest.main()
