# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
import unittest
from unittest import mock

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval import runtime  # noqa: E402


class RuntimeAdapterTest(unittest.TestCase):
    def test_system_clock_returns_utc_and_monotonic_values(self) -> None:
        expected = dt.datetime(2026, 7, 19, tzinfo=dt.timezone.utc)
        with (
            mock.patch.object(runtime.dt, "datetime") as datetime_type,
            mock.patch.object(runtime.time, "monotonic", return_value=12.5),
        ):
            datetime_type.now.return_value = expected
            datetime_type.side_effect = dt.datetime

            clock = runtime.SystemClock()
            self.assertIs(clock.now_utc(), expected)
            self.assertEqual(clock.monotonic(), 12.5)
            datetime_type.now.assert_called_once_with(dt.timezone.utc)

    def test_system_sleeper_delegates_requested_duration(self) -> None:
        with mock.patch.object(runtime.time, "sleep") as sleep:
            runtime.SystemSleeper().sleep(0.25)

        sleep.assert_called_once_with(0.25)


if __name__ == "__main__":
    unittest.main()
