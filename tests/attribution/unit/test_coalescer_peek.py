# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time

from nvidia_resiliency_ext.attribution.coalescing.coalescer import RequestCoalescer
from nvidia_resiliency_ext.attribution.coalescing.types import CacheEntry, InFlightEntry


def test_peek_returns_cached_result_without_compute():
    async def run():
        coalescer = RequestCoalescer()
        coalescer._cache["/tmp/job.log"] = CacheEntry(
            result={"result": []},
            cached_at=time.monotonic(),
            file_mtime=None,
            file_size=None,
        )

        result = await coalescer.peek("/tmp/job.log")

        assert result == {"status": "completed", "result": {"result": []}}
        stats = await coalescer.get_stats()
        assert stats["cache"]["hits"] == 1
        assert stats["compute"]["total"] == 0

    asyncio.run(run())


def test_peek_returns_in_flight_without_awaiting_future():
    async def run():
        coalescer = RequestCoalescer()
        future = asyncio.get_running_loop().create_future()
        coalescer._in_flight["/tmp/job.log"] = InFlightEntry(
            future=future,
            started_at=time.monotonic(),
        )

        result = await coalescer.peek("/tmp/job.log")

        assert result == {"status": "in_flight", "result": None}
        assert not future.done()
        stats = await coalescer.get_stats()
        assert stats["compute"]["total"] == 0

    asyncio.run(run())


def test_peek_returns_pending_without_starting_compute():
    async def run():
        coalescer = RequestCoalescer()
        await coalescer.track_submission("/tmp/job.log")

        result = await coalescer.peek("/tmp/job.log")

        assert result == {"status": "pending", "result": None}
        stats = await coalescer.get_stats()
        assert stats["cache"]["misses"] == 0
        assert stats["compute"]["total"] == 0

    asyncio.run(run())
