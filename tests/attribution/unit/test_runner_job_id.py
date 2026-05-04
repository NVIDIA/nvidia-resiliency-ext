# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

PY310_PLUS = sys.version_info >= (3, 10)

if PY310_PLUS:
    from nvidia_resiliency_ext.attribution.orchestration import runner


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):
        return self._value


def _run_coro_immediately(coro, loop):
    return _ImmediateFuture(asyncio.run(coro))


class _FakeController:
    def __init__(self):
        self.config = SimpleNamespace(cache=SimpleNamespace(compute_timeout=None))
        self.submitted_metadata = []

    def validate_path(self, user_path, require_regular_file=True, reject_empty=False):
        return user_path

    async def submit_log(self, log_path, user="unknown", job_id=None):
        self.submitted_metadata.append((user, job_id))

    async def analyze_log(self, log_path, wl_restart=None):
        return SimpleNamespace(result={"ok": True})


@unittest.skipUnless(PY310_PLUS, "attribution tests require Python 3.10+")
class TestRunnerJobId(unittest.TestCase):
    def test_run_log_analysis_sync_uses_controller_metadata_defaults(self):
        controller = _FakeController()

        with (
            patch.object(runner, "_controller", controller),
            patch.object(runner, "_controller_loop", object()),
            patch.object(runner, "_get_or_create_controller", return_value=True),
            patch.object(
                runner.asyncio,
                "run_coroutine_threadsafe",
                side_effect=_run_coro_immediately,
            ),
        ):
            result = runner.run_log_analysis_sync("/tmp/test.log")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(controller.submitted_metadata, [("unknown", None)])

    def test_notify_log_path_sync_uses_controller_metadata_defaults(self):
        controller = _FakeController()

        with (
            patch.object(runner, "_controller", controller),
            patch.object(runner, "_controller_loop", object()),
            patch.object(runner, "_get_or_create_controller", return_value=True),
            patch.object(
                runner.asyncio,
                "run_coroutine_threadsafe",
                side_effect=_run_coro_immediately,
            ),
        ):
            runner.notify_log_path_sync("/tmp/test.log")

        self.assertEqual(controller.submitted_metadata, [("unknown", None)])


if __name__ == "__main__":
    unittest.main()
