# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time

from nvidia_resiliency_ext.attribution.coalescing.coalesced_cache import LogAnalysisCoalesced
from nvidia_resiliency_ext.attribution.coalescing.coalescer import RequestCoalescer
from nvidia_resiliency_ext.attribution.coalescing.types import CacheEntry


def test_cache_summary_uses_recommendation_envelope():
    coalescer = RequestCoalescer()
    coalescer._cache["/tmp/job.log"] = CacheEntry(
        result={
            "module": "log_analyzer",
            "result_id": "abcdef1234567890",
            "result": [],
            "recommendation": {"action": "STOP", "source": "log_analyzer"},
        },
        cached_at=time.monotonic(),
    )

    cache = asyncio.run(coalescer.get_cache())

    entry = cache["entries"][0]
    assert entry["recommendation"] == {"action": "STOP", "source": "log_analyzer"}
    assert "state" not in entry


def test_cache_summary_synthesizes_fr_only_recommendation():
    coalescer = RequestCoalescer()
    coalescer._cache["/tmp/fr"] = CacheEntry(
        result=LogAnalysisCoalesced(log_result=None, fr_dump_path="/tmp/fr"),
        cached_at=time.monotonic(),
    )

    cache = asyncio.run(coalescer.get_cache())

    entry = cache["entries"][0]
    assert entry["module"] == "fr_only"
    assert entry["recommendation"] == {"action": "UNKNOWN", "source": "fr_only"}
    assert "state" not in entry


def test_cache_summary_source_fallback_does_not_promote_service_reason():
    coalescer = RequestCoalescer()
    coalescer._cache["/tmp/job.log"] = CacheEntry(
        result={
            "result_id": "abcdef1234567890",
            "error": "service generated explanation",
            "result": [],
            "recommendation": {"action": "STOP"},
        },
        cached_at=time.monotonic(),
    )

    cache = asyncio.run(coalescer.get_cache())

    entry = cache["entries"][0]
    assert entry["module"] == "unknown"
    assert entry["recommendation"] == {"action": "STOP", "source": "unknown"}
