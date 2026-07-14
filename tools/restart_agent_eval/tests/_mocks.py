# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic infrastructure fakes shared by harness tests."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Mapping
from unittest import mock

from restart_agent_eval.product_process import ProcessResult


def process_result(
    command=(),
    returncode=0,
    stdout="",
    stderr="",
) -> ProcessResult:
    """Build a normalized product-process result without spawning a process."""
    return ProcessResult(tuple(command), returncode, stdout, stderr)


class RecordingExecutor:
    """Record process commands and return one configured result."""

    def __init__(self, result: ProcessResult) -> None:
        self.result = result
        self.calls = []

    def run(self, command, *, cwd, env=None):
        self.calls.append((list(command), cwd, dict(env or {})))
        return self.result


@contextlib.contextmanager
def isolated_environment(
    values: Mapping[str, str] | None = None,
) -> Iterator[None]:
    """Run a test with only the explicitly supplied environment variables."""
    with mock.patch.dict(os.environ, dict(values or {}), clear=True):
        yield
