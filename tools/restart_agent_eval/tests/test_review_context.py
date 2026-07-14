# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalized review context across fallback, enriched, and malformed payloads."""

from __future__ import annotations

import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval.review_context import ReviewContext  # noqa: E402


def _trace(analysis, analyzer_trace=None, collect_all_context=None):
    return {
        "schema_version": "restart_agent_cli_trace.v1",
        "request": {},
        "analysis_result": analysis,
        "analyzer_trace": analyzer_trace or {},
        "l0_bundle": {},
        "collect_all_context": collect_all_context,
    }


class ReviewContextTest(unittest.TestCase):
    def test_normalizes_all_stage_payloads_and_candidate_results(self) -> None:
        fallback = {"decision": "RESTART"}
        enriched = {"decision": "STOP"}
        analyzer_trace = {
            "l0_model_view": {"schema_version": "l0b.v1"},
            "decision_evidence": {
                "deterministic_primary_candidate": {"line": 10},
            },
            "layers": {"L1": {"output_status": "usable"}},
            "l1": {
                "parsed_evidence": {"primary_failure": {"line": 11}},
                "model_calls": [{"success": True}],
                "tool_calls": [{"name": "overview"}],
                "transcript_events": [{"event_type": "model_request"}],
            },
            "l2_audit": {"audit_status": "clean"},
            "l2_grounded_semantics": {"primary_failure": {"line": 12}},
            "current_failure_facts": {"observed": True},
            "timing": {"l1_wall_clock_s": 1.0},
            "latency_measurement": {"mode": "terminal"},
            "token_usage": {"total_tokens": 10},
            "token_limit": {"context_window_tokens": 100},
            "decision_candidates": {
                "deterministic_fallback": {"result": fallback},
                "l1_enriched": {"result": enriched},
            },
        }
        analysis = {
            "decision": "STOP",
            "primary_failure": {"line": 12},
            "result_provenance": {"candidate_kind": "l1_enriched"},
        }

        context = ReviewContext.from_payloads(
            {"decision": "STOP"},
            _trace(
                analysis,
                analyzer_trace,
                {"route_id": "gpt", "execution_status": "completed"},
            ),
        )

        self.assertEqual(context.route_execution_status, "completed")
        self.assertEqual(context.l0_primary, {"line": 10})
        self.assertEqual(context.l1_primary, {"line": 11})
        self.assertEqual(context.l2_primary, {"line": 12})
        self.assertEqual(context.fallback_analysis, fallback)
        self.assertEqual(context.enriched_analysis, enriched)
        self.assertEqual(context.interaction_transcript[0]["event_type"], "model_request")
        self.assertEqual(context.primary, {"line": 12})

    def test_selected_candidate_falls_back_to_analysis_when_envelope_is_absent(self) -> None:
        for candidate_kind, expected_field in (
            ("deterministic_fallback", "fallback_analysis"),
            ("l1_enriched", "enriched_analysis"),
        ):
            analysis = {
                "decision": "RESTART",
                "result_provenance": {"candidate_kind": candidate_kind},
            }
            context = ReviewContext.from_payloads({}, _trace(analysis))

            with self.subTest(candidate_kind=candidate_kind):
                self.assertEqual(getattr(context, expected_field), analysis)

    def test_non_mapping_result_and_optional_stage_values_become_empty(self) -> None:
        context = ReviewContext.from_payloads(
            ["not-an-object"],
            _trace({"decision": "RESTART"}, {"l1": {"model_calls": None}}),
        )

        self.assertEqual(context.result, {})
        self.assertEqual(context.model_calls, [])
        self.assertEqual(context.tool_calls, [])
        self.assertEqual(context.interaction_transcript, [])
        self.assertIsNone(context.route_execution_status)

    def test_read_uses_injected_artifact_store(self) -> None:
        class _Store:
            def __init__(self) -> None:
                self.paths = []

            def read_json(self, path):
                self.paths.append(path)
                if path.name == "result.json":
                    return {"decision": "RESTART"}
                return _trace({"decision": "RESTART"})

        store = _Store()
        context = ReviewContext.read(
            {"result_json": Path("result.json"), "trace_json": Path("trace.json")},
            artifact_store=store,
        )

        self.assertEqual(context.analysis["decision"], "RESTART")
        self.assertEqual(store.paths, [Path("result.json"), Path("trace.json")])


if __name__ == "__main__":
    unittest.main()
