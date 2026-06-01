# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
import unittest
from unittest.mock import patch

from nvidia_resiliency_ext.attribution.orchestration.analysis_pipeline import (
    AnalysisPipelineMode,
    FrDumpPathNotFoundError,
    run_attribution_pipeline,
)


class TestAnalysisPipelineTraceOnly(unittest.TestCase):
    def test_log_only_records_analysis_completion_timestamp(self):
        async def _run():
            before_ms = round(time.time() * 1000)
            result = await run_attribution_pipeline(
                "/tmp/log.txt",
                mode=AnalysisPipelineMode.LOG_ONLY,
                run_logsage=lambda: asyncio.sleep(0, result={"result": [], "state": "CONTINUE"}),
            )
            after_ms = round(time.time() * 1000)

            self.assertGreaterEqual(result.processing_time, 0)
            self.assertGreaterEqual(result.analysis_completed_at_ms, before_ms)
            self.assertLessEqual(result.analysis_completed_at_ms, after_ms)

        asyncio.run(_run())

    def test_trace_only_missing_dump_raises_fr_dump_path_not_found(self):
        async def _run():
            with self.assertRaises(FrDumpPathNotFoundError):
                await run_attribution_pipeline(
                    "/tmp/no_fr_hint.log",
                    mode=AnalysisPipelineMode.TRACE_ONLY,
                    discover_fr_dump_path=lambda _p: None,
                )

        asyncio.run(_run())

    def test_merge_llm_omits_unset_overrides(self):
        captured = {}

        async def fake_merge(log_result, fr_analysis, *, llm_api_key, **kwargs):
            captured["log_result"] = log_result
            captured["fr_analysis"] = fr_analysis
            captured["llm_api_key"] = llm_api_key
            captured["kwargs"] = kwargs
            return "merged"

        async def _run():
            with (
                patch(
                    "nvidia_resiliency_ext.attribution.orchestration.analysis_pipeline.load_llm_api_key",
                    return_value="key",
                ),
                patch(
                    "nvidia_resiliency_ext.attribution.combined_log_fr.llm_merge.merge_log_fr_llm",
                    new=fake_merge,
                ),
            ):
                result = await run_attribution_pipeline(
                    "/tmp/log.txt",
                    mode=AnalysisPipelineMode.LOG_AND_TRACE_WITH_LLM,
                    run_logsage=lambda: asyncio.sleep(
                        0,
                        result={
                            "module": "log_analyzer",
                            "result": [],
                            "recommendation": {"action": "UNKNOWN", "source": "log_analyzer"},
                        },
                    ),
                    discover_fr_dump_path=lambda _p: "/tmp/fr",
                    run_fr_analysis=lambda _p: asyncio.sleep(0, result={"fr": "analysis"}),
                )

            self.assertEqual(result.llm_merged_summary, "merged")

        asyncio.run(_run())

        self.assertEqual(captured["llm_api_key"], "key")
        self.assertEqual(captured["kwargs"], {})

    def test_log_and_trace_can_use_combined_mcp_without_merge_summary(self):
        called = False

        async def combined(_log_path, _fr_path):
            nonlocal called
            called = True
            return (
                {
                    "module": "log_fr_analyzer",
                    "result": [],
                    "recommendation": {"action": "UNKNOWN", "source": "log_analyzer"},
                },
                {"fr": "analysis"},
                "unexpected merge summary",
            )

        async def _run():
            result = await run_attribution_pipeline(
                "/tmp/log.txt",
                mode=AnalysisPipelineMode.LOG_AND_TRACE,
                run_logsage=lambda: asyncio.sleep(
                    0,
                    result={
                        "module": "log_analyzer",
                        "result": [],
                        "recommendation": {"action": "UNKNOWN", "source": "log_analyzer"},
                    },
                ),
                discover_fr_dump_path=lambda _p: "/tmp/fr",
                run_fr_analysis=lambda _p: asyncio.sleep(0, result={"fr": "analysis"}),
                run_log_fr_analyzer_mcp=combined,
            )

            self.assertTrue(called)
            self.assertEqual(result.fr_analysis, {"fr": "analysis"})
            self.assertIsNone(result.llm_merged_summary)

        asyncio.run(_run())

    def test_log_and_trace_lib_path_runs_logsage_and_fr_in_parallel_without_merge(self):
        async def fake_merge(*_args, **_kwargs):
            raise AssertionError("merge_log_fr_llm must not run in LOG_AND_TRACE")

        async def _run():
            log_started = asyncio.Event()
            fr_started = asyncio.Event()

            async def run_logsage():
                log_started.set()
                await asyncio.wait_for(fr_started.wait(), timeout=1)
                return {
                    "module": "log_analyzer",
                    "result": [],
                    "recommendation": {"action": "UNKNOWN", "source": "log_analyzer"},
                }

            async def run_fr_analysis(_path):
                fr_started.set()
                await asyncio.wait_for(log_started.wait(), timeout=1)
                return {"fr": "analysis"}

            with patch(
                "nvidia_resiliency_ext.attribution.combined_log_fr.llm_merge.merge_log_fr_llm",
                new=fake_merge,
            ):
                result = await run_attribution_pipeline(
                    "/tmp/log.txt",
                    mode=AnalysisPipelineMode.LOG_AND_TRACE,
                    run_logsage=run_logsage,
                    discover_fr_dump_path=lambda _p: "/tmp/fr",
                    run_fr_analysis=run_fr_analysis,
                )

            self.assertEqual(result.fr_analysis, {"fr": "analysis"})
            self.assertIsNone(result.llm_merged_summary)

        asyncio.run(_run())

    def test_with_llm_does_not_merge_when_fr_analysis_is_missing(self):
        async def fake_merge(*_args, **_kwargs):
            raise AssertionError("merge_log_fr_llm must not run without FR analysis")

        async def _run():
            with patch(
                "nvidia_resiliency_ext.attribution.combined_log_fr.llm_merge.merge_log_fr_llm",
                new=fake_merge,
            ):
                result = await run_attribution_pipeline(
                    "/tmp/log.txt",
                    mode=AnalysisPipelineMode.LOG_AND_TRACE_WITH_LLM,
                    run_logsage=lambda: asyncio.sleep(
                        0,
                        result={
                            "module": "log_analyzer",
                            "result": [],
                            "recommendation": {"action": "UNKNOWN", "source": "log_analyzer"},
                        },
                    ),
                    discover_fr_dump_path=lambda _p: "/tmp/fr",
                    run_fr_analysis=lambda _p: asyncio.sleep(0, result=None),
                )

            self.assertIsNone(result.fr_analysis)
            self.assertIsNone(result.llm_merged_summary)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
