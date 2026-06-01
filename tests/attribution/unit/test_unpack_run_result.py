# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import unittest

PY310_PLUS = sys.version_info >= (3, 10)

if PY310_PLUS:
    from nvidia_resiliency_ext.attribution.base import AttributionState
    from nvidia_resiliency_ext.attribution.combined_log_fr.llm_merge import unpack_run_result
    from nvidia_resiliency_ext.attribution.mcp_integration.registry import serialize_result
    from nvidia_resiliency_ext.attribution.orchestration.types import (
        AttributionRecommendation,
        LogSageAnalysisResult,
        RawAnalysisResultItem,
    )
    from nvidia_resiliency_ext.attribution.orchestration.utils import nvrx_run_result_to_log_dict


def _item(raw_text, action):
    return RawAnalysisResultItem(
        raw_text=raw_text,
        auto_resume=raw_text,
        auto_resume_explanation="",
        attribution_text="",
        checkpoint_saved_flag=0,
        action=action,
    )


def _payload(raw_text, action):
    return _item(raw_text, action).to_payload()


@unittest.skipUnless(PY310_PLUS, "attribution tests require Python 3.10+")
class TestUnpackRunResult(unittest.TestCase):
    def test_log_analyzer_items_join_text(self):
        """NVRxLogAnalyzer.run_sync returns list[RawAnalysisResultItem]."""
        raw = [
            _item("cycle a summary", "CONTINUE"),
            _item("cycle b summary", "CONTINUE"),
        ]
        payload, st = unpack_run_result(raw)
        self.assertEqual(payload, "cycle a summary\n\ncycle b summary")
        self.assertEqual(st, AttributionState.CONTINUE)

    def test_log_analyzer_list_stop_if_any_stop(self):
        raw = [
            _item("ok", "CONTINUE"),
            _item("stop here", "STOP"),
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

    def test_nvrx_run_result_to_log_dict_serializes_raw_items(self):
        raw = (
            LogSageAnalysisResult(
                [_item("cycle summary", "STOP")],
                AttributionRecommendation(action="STOP", source="log_analyzer"),
            ),
            AttributionState.STOP,
        )

        payload = nvrx_run_result_to_log_dict(raw, "/logs/job.out")

        self.assertEqual(payload["module"], "log_analyzer")
        self.assertNotIn("state", payload)
        self.assertEqual(payload["result"], [_payload("cycle summary", "STOP")])
        self.assertEqual(payload["recommendation"], {"action": "STOP", "source": "log_analyzer"})

    def test_mcp_serialization_keeps_nested_raw_items_structured(self):
        payload = nvrx_run_result_to_log_dict(
            ([_item("cycle summary", "STOP")], AttributionState.STOP),
            "/logs/job.out",
        )
        serialized = serialize_result(payload)

        self.assertEqual(
            json.loads(serialized),
            {
                "module": "log_analyzer",
                "result": [_payload("cycle summary", "STOP")],
                "recommendation": {"action": "STOP", "source": "log_analyzer"},
            },
        )


if __name__ == "__main__":
    unittest.main()
