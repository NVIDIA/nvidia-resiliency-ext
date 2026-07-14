# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool-context dependency, redundancy, and L0-gap scoring tests."""

from __future__ import annotations

import unittest

from _bootstrap import configure_test_imports

configure_test_imports()

from _panel_fixtures import panel_summary  # noqa: E402
from restart_agent_eval import scoring  # noqa: E402


def _single_tool_turn(tool_name: str, *, result_lines: int) -> dict:
    return {
        "model_calls": [{"model_turn": 1}, {"model_turn": 2}],
        "tool_calls": [
            {
                "tool_call_id": "call-1",
                "model_turn": 1,
                "name": tool_name,
                "result_lines": result_lines,
            }
        ],
    }


def _tool_result_event(lines: list[dict]) -> dict:
    return {
        "event_type": "tool_result",
        "tool_call_id": "call-1",
        "result": {"lines": lines},
    }


class ToolEfficiencyScoringTest(unittest.TestCase):
    def test_tool_dependency_uses_final_cited_lines_not_raw_line_novelty(self) -> None:
        transcript = [
            {
                "event_type": "bundle_snapshot",
                "bundle": {"context_windows": [{"lines": [{"line": 10}]}]},
            },
            _tool_result_event([{"line": 20, "text": "new"}]),
        ]
        summary = scoring.tool_efficiency_summary(
            l1=_single_tool_turn("read_window", result_lines=1),
            timing={},
            interaction_transcript=transcript,
            analysis={
                "primary_failure": {"line": 20},
                "evidence": [{"line": 20, "quote": "new"}],
            },
        )

        self.assertEqual(
            summary["final_context_dependency"],
            "final_evidence_depends_on_tool_only_lines",
        )
        self.assertEqual(summary["final_context_impact"], "decision_critical_primary")
        self.assertTrue(summary["final_primary_from_tool_only_context"])
        self.assertEqual(summary["final_evidence_tool_only_lines"], [20])
        self.assertEqual(summary["model_turns"], 2)
        self.assertEqual(summary["extra_model_turns_after_initial"], 1)
        self.assertEqual(summary["tool_driven_model_turns"], 1)

    def test_l0_gap_requires_final_dependency_on_tool_only_context(self) -> None:
        base_summary = {
            "target": "qwen235b",
            "model": "qwen235b",
            "l0_bundle_kpis": {"tool_calls_added_new_prompt_lines": 12},
            "tool_efficiency": {
                "final_context_dependency": "no_final_citation_dependency_observed"
            },
        }

        concerns = panel_summary([base_summary])["concerns"]
        self.assertNotIn("l0_bundle_gap", {item["category"] for item in concerns})

        dependent_summary = {
            **base_summary,
            "tool_efficiency": {
                "final_context_dependency": "final_evidence_depends_on_tool_only_lines",
                "decision_relevant_tool_only_lines": [20],
            },
        }
        concerns = panel_summary([dependent_summary])["concerns"]
        self.assertIn("l0_bundle_gap", {item["category"] for item in concerns})

    def test_tool_line_for_existing_structured_fact_is_not_an_l0_gap(self) -> None:
        summary = scoring.tool_efficiency_summary(
            l1=_single_tool_turn("grep_log", result_lines=1),
            timing={},
            interaction_transcript=[
                {"event_type": "bundle_snapshot", "bundle": {"context_windows": []}},
                _tool_result_event(
                    [
                        {
                            "line": 666,
                            "text": "world_size ........................ 6144",
                        }
                    ]
                ),
            ],
            analysis={
                "primary_failure": {"line": 1000},
                "evidence": [{"line": 666, "quote": "world_size 6144"}],
            },
            l0_bundle={
                "job_metadata": {
                    "explicit_world_size": 6144,
                    "explicit_world_size_line": 666,
                }
            },
        )

        self.assertEqual(
            summary["final_context_impact"],
            "existing_structured_fact_redundancy",
        )
        self.assertEqual(summary["decision_relevant_tool_only_lines"], [])
        self.assertEqual(summary["structured_fact_redundant_tool_only_lines"], [666])
        route_summary = {
            "target": "nemotron",
            "model": "nemotron",
            "l0_bundle_kpis": {"tool_calls_added_new_prompt_lines": 1},
            "tool_efficiency": summary,
        }
        concerns = panel_summary([route_summary])["concerns"]
        self.assertNotIn("l0_bundle_gap", {item["category"] for item in concerns})

    def test_incidental_tool_only_teardown_is_not_an_l0_gap(self) -> None:
        summary = scoring.tool_efficiency_summary(
            l1=_single_tool_turn("read_window", result_lines=1),
            timing={},
            interaction_transcript=[
                {
                    "event_type": "bundle_snapshot",
                    "bundle": {"context_windows": [{"lines": [{"line": 10}]}]},
                },
                _tool_result_event([{"line": 30, "text": "cancelled"}]),
            ],
            analysis={
                "primary_failure": {"line": 10},
                "secondary_failures": [{"line": 30, "causal_role": "teardown"}],
                "evidence": [
                    {
                        "line": 30,
                        "quote": "cancelled",
                        "supports": "scheduler cancellation is downstream teardown",
                    }
                ],
            },
        )

        self.assertEqual(summary["final_context_impact"], "incidental_downstream_context")
        self.assertEqual(summary["decision_relevant_tool_only_lines"], [])
        self.assertEqual(summary["incidental_tool_only_lines"], [30])
        concerns = panel_summary(
            [
                {
                    "target": "gpt",
                    "model": "gpt",
                    "l0_bundle_kpis": {"tool_calls_added_new_prompt_lines": 1},
                    "tool_efficiency": summary,
                }
            ]
        )["concerns"]
        self.assertNotIn("l0_bundle_gap", {item["category"] for item in concerns})

    def test_tool_context_separates_decision_structured_and_unused_lines(self) -> None:
        summary = scoring.tool_efficiency_summary(
            l1=_single_tool_turn("grep_log", result_lines=3),
            timing={},
            interaction_transcript=[
                {"event_type": "bundle_snapshot", "bundle": {"context_windows": []}},
                _tool_result_event(
                    [
                        {"line": 20, "text": "world_size 6144"},
                        {"line": 30, "text": "deterministic_mode False"},
                        {"line": 40, "text": "unreferenced setup detail"},
                    ]
                ),
            ],
            analysis={
                "primary_failure": {"line": 100},
                "evidence": [
                    {"line": 20, "quote": "world_size 6144"},
                    {"line": 30, "quote": "deterministic_mode False"},
                ],
            },
            l0_bundle={
                "job_metadata": {
                    "explicit_world_size": 6144,
                    "explicit_world_size_line": 20,
                }
            },
        )

        self.assertEqual(summary["unique_new_prompt_excerpt_line_count"], 3)
        self.assertEqual(summary["decision_relevant_tool_only_lines"], [30])
        self.assertEqual(summary["structured_fact_redundant_tool_only_lines"], [20])
        self.assertEqual(summary["incidental_tool_only_lines"], [])
        self.assertEqual(summary["unused_tool_only_lines"], [40])
        route_summary = {
            "target": "nemotron",
            "model": "nemotron",
            "tool_efficiency": summary,
        }
        concerns = panel_summary([route_summary])["concerns"]
        gap = next(item for item in concerns if item["category"] == "l0_bundle_gap")
        self.assertIn("decision-relevant tool-only lines=[30]", gap["summary"])
        self.assertIn("structured-fact repeats=[20]", gap["summary"])
        self.assertIn("unused new lines=1", gap["summary"])
