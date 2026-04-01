# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from nvidia_resiliency_ext.attribution.trace_analyzer.fr_support import (
    FRAnalysisResult,
    fr_result_from_mcp_module_response,
)


class TestFrResultFromMcpModuleResponse(unittest.TestCase):
    def test_structured_result(self) -> None:
        self.assertIsNone(
            fr_result_from_mcp_module_response(
                {"result": {"analysis_text": "only text"}},
            ),
        )

        r2 = fr_result_from_mcp_module_response(
            {
                "module": "fr_analyzer",
                "result": {"analysis_text": "table\nhere", "hanging_ranks": "hanging ranks: [1]"},
            }
        )
        self.assertIsInstance(r2, FRAnalysisResult)
        assert r2 is not None
        self.assertEqual(r2.analysis_text, "table\nhere")
        self.assertEqual(r2.hanging_ranks, "hanging ranks: [1]")

    def test_legacy_string_result(self) -> None:
        r = fr_result_from_mcp_module_response(
            {"result": "hanging ranks: [2, 3]"},
        )
        self.assertIsInstance(r, FRAnalysisResult)
        assert r is not None
        self.assertEqual(r.analysis_text, "")
        self.assertEqual(r.hanging_ranks, "hanging ranks: [2, 3]")

    def test_error_returns_none(self) -> None:
        self.assertIsNone(
            fr_result_from_mcp_module_response({"error": "boom", "result": {}}),
        )


if __name__ == "__main__":
    unittest.main()
