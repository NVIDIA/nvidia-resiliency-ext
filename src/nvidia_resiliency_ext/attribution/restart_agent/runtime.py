# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replaceable runtime services used by orchestration and provider adapters."""

from __future__ import annotations

import time
from concurrent.futures import Executor, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    """Monotonic and wall-clock time source."""

    def monotonic(self) -> float: ...

    def now_utc(self) -> datetime: ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)


class Sleeper(Protocol):
    def sleep(self, seconds: float) -> None: ...


class SystemSleeper:
    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class ExecutorFactory(Protocol):
    """Create an executor without exposing a concrete thread implementation."""

    def __call__(self, *, max_workers: int, thread_name_prefix: str) -> Executor: ...


class ThreadExecutorFactory:
    def __call__(self, *, max_workers: int, thread_name_prefix: str) -> Executor:
        return ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )


SYSTEM_CLOCK = SystemClock()
SYSTEM_SLEEPER = SystemSleeper()
THREAD_EXECUTOR_FACTORY = ThreadExecutorFactory()
