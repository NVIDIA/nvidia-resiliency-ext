# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import unittest

from nvidia_resiliency_ext.attribution.log_analyzer.analysis_pipeline import (
    AnalysisPipelineMode,
    FrDumpPathNotFoundError,
    run_attribution_pipeline,
)


class TestAnalysisPipelineTraceOnly(unittest.TestCase):
    def test_trace_only_missing_dump_raises_fr_dump_path_not_found(self):
        async def _run():
            with self.assertRaises(FrDumpPathNotFoundError):
                await run_attribution_pipeline(
                    "/tmp/no_fr_hint.log",
                    mode=AnalysisPipelineMode.TRACE_ONLY,
                    discover_fr_dump_path=lambda _p: None,
                )

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
