# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replaceable runtime services for harness composition roots."""

from __future__ import annotations

import datetime as dt
import time
from typing import Protocol


class Clock(Protocol):
    def now_utc(self) -> dt.datetime: ...

    def monotonic(self) -> float: ...


class SystemClock:
    def now_utc(self) -> dt.datetime:
        return dt.datetime.now(dt.timezone.utc)

    def monotonic(self) -> float:
        return time.monotonic()


class Sleeper(Protocol):
    def sleep(self, seconds: float) -> None: ...


class SystemSleeper:
    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


SYSTEM_CLOCK = SystemClock()
SYSTEM_SLEEPER = SystemSleeper()
