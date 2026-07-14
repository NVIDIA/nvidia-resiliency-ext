# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository provenance success, command-failure, and absent-repo tests."""

from __future__ import annotations

import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from _mocks import process_result  # noqa: E402
from restart_agent_eval import repository_identity  # noqa: E402


class _SequenceExecutor:
    def __init__(self, *results) -> None:
        self.results = list(results)
        self.commands = []

    def run(self, command, *, cwd, env=None):
        self.commands.append((list(command), cwd))
        return self.results.pop(0)


class RepositoryIdentityTest(unittest.TestCase):
    def test_absent_repository_has_no_identity(self) -> None:
        actual = repository_identity.git_identity(None)

        self.assertIsNone(actual)

    def test_clean_and_dirty_repositories_are_distinguished(self) -> None:
        for status_output, expected_dirty in (("", False), (" M file.py\n", True)):
            executor = _SequenceExecutor(
                process_result(stdout="abc123\n"),
                process_result(stdout=status_output),
            )

            identity = repository_identity.git_identity(
                Path("/repo"),
                process_executor=executor,
            )

            with self.subTest(status_output=status_output):
                self.assertEqual(
                    identity,
                    {"path": "/repo", "commit": "abc123", "dirty": expected_dirty},
                )
                self.assertEqual(
                    [command for command, _ in executor.commands],
                    [["git", "rev-parse", "HEAD"], ["git", "status", "--porcelain"]],
                )

    def test_failed_git_commands_publish_unknown_fields(self) -> None:
        executor = _SequenceExecutor(
            process_result(returncode=128, stderr="not a repository"),
            process_result(returncode=128, stderr="not a repository"),
        )

        identity = repository_identity.git_identity(
            Path("/repo"),
            process_executor=executor,
        )

        self.assertEqual(identity, {"path": "/repo", "commit": None, "dirty": None})


if __name__ == "__main__":
    unittest.main()
