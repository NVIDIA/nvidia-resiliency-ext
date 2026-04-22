# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest

PY310_PLUS = sys.version_info >= (3, 10)

if PY310_PLUS:
    from nvidia_resiliency_ext.attribution.base import AttributionState
    from nvidia_resiliency_ext.attribution.combined_log_fr.llm_merge import unpack_run_result


@unittest.skipUnless(PY310_PLUS, "attribution tests require Python 3.10+")
class TestUnpackRunResult(unittest.TestCase):
    def test_log_analyzer_list_of_tuples_joins_text(self):
        """NVRxLogAnalyzer.run_sync returns list[tuple[str, AttributionState]]."""
        raw = [
            ("cycle a summary", AttributionState.CONTINUE),
            ("cycle b summary", AttributionState.CONTINUE),
        ]
        payload, st = unpack_run_result(raw)
        self.assertEqual(payload, "cycle a summary\n\ncycle b summary")
        self.assertEqual(st, AttributionState.CONTINUE)

    def test_log_analyzer_list_stop_if_any_stop(self):
        raw = [
            ("ok", AttributionState.CONTINUE),
            ("stop here", AttributionState.STOP),
        ]
        _payload, st = unpack_run_result(raw)
        self.assertEqual(st, AttributionState.STOP)

    def test_empty_list_yields_empty_string(self):
        payload, st = unpack_run_result([])
        self.assertEqual(payload, "")
        self.assertEqual(st, AttributionState.CONTINUE)

    def test_collective_analyzer_two_tuple_unchanged(self):
        d = {"analysis_text": "x", "hanging_ranks": "y"}
        raw = (d, AttributionState.CONTINUE)
        payload, st = unpack_run_result(raw)
        self.assertIs(payload, d)
        self.assertEqual(st, AttributionState.CONTINUE)


if __name__ == "__main__":
    unittest.main()
