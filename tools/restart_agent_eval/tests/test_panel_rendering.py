# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Panel summary and diagnostics Markdown rendering behavior."""

from __future__ import annotations

import unittest

from _bootstrap import configure_test_imports

configure_test_imports()

from _assertions import assert_mapping_fields  # noqa: E402
from _panel_fixtures import published_panel  # noqa: E402
from _panel_fixtures import rich_panel_artifacts  # noqa: E402
from _panel_fixtures import rich_route_summary  # noqa: E402


class PanelRenderingTest(unittest.TestCase):
    def test_panel_markdown_separates_summary_from_diagnostics(self) -> None:
        _, _, _, _, summary_markdown, diagnostics_markdown = rich_panel_artifacts()

        for section in (
            "## Run Identity",
            "## Attention Required",
            "## Cross-Route Outcome",
            "## Semantic Comparison",
            "## Operational Comparison",
            "## Shared Deterministic Evidence",
            "## History And Policy",
            "## Artifacts",
        ):
            with self.subTest(section=section):
                self.assertIn(section, summary_markdown)
        self.assertNotIn("## Experimental Failure Identity", summary_markdown)
        self.assertIn("## Experimental Failure Identity", diagnostics_markdown)
        self.assertIn("## L2 Grounding, Identity, And Audit", diagnostics_markdown)
        self.assertIn("## Deterministic Fallback Inputs", diagnostics_markdown)
        self.assertIn("## L3 History", diagnostics_markdown)
        self.assertIn("## L4 Policy Decision", diagnostics_markdown)

    def test_panel_reports_shared_l0_shape_once(self) -> None:
        _, _, panel, _, summary_markdown, diagnostics_markdown = rich_panel_artifacts()
        shape = panel["shared_l0_shape"]

        assert_mapping_fields(
            self,
            shape,
            {
                "line_count": 243184,
                "byte_size": 28038563,
                "context_window_count": 9,
                "candidate_anchor_count": 14,
                "occurrence_group_count": 7,
                "failure_episode_count": 3,
                "consistent_across_models": True,
            },
        )
        markdown = summary_markdown + "\n" + diagnostics_markdown
        self.assertEqual(markdown.count("log_lines=`243184`"), 1)

    def test_panel_reports_progressive_gate_latency(self) -> None:
        summary = rich_route_summary()
        summary["l4_kpis"] = {
            "latency_mode": "progressive_end_to_result",
            "terminal_total_wall_clock_s": 8.0,
            "post_progressive_end_wall_clock_s": 3.0,
        }
        _, _, markdown = published_panel([summary])

        self.assertIn("| post-end |", markdown)
        self.assertNotIn("terminal total", markdown)
        self.assertIn("production progressive decision-gate latency", markdown)

    def test_panel_separates_l1_core_semantics_from_related_failures(self) -> None:
        summary = {
            "target": "gpt",
            "model": "test-gpt",
            "decision": "RESTART",
            "gold_score": {
                "case_id": "checkpoint-a",
                "l1": {
                    "root_cause_correct": True,
                    "recovery_assessment_correct": True,
                    "related_failure_recall": False,
                    "unsupported_claims": [],
                    "core_semantic_pass": True,
                    "overall_semantic_pass": False,
                },
                "l2": {},
                "l4": {
                    "cascade_correct": True,
                    "action_correct": True,
                    "overall_semantic_pass": True,
                },
            },
            "l1_kpis": {
                "successful_model_calls": 1,
                "response_parsed": True,
                "output_usable": True,
            },
            "model_selection_signals": {"endpoint_reliability": "ok"},
            "path_redaction_audit": {"passed": True},
        }

        panel, markdown, _ = published_panel([summary])
        row = panel["rows"][0]

        self.assertTrue(row["gold_l1_core_semantic"])
        self.assertFalse(row["gold_l1_related_failure_recall"])
        self.assertEqual(panel["comparison_axes"]["semantic_quality"][0]["status"], "pass")
        self.assertIn("L1 RCA", markdown)
        self.assertIn("L1 recovery", markdown)
        self.assertIn("L1 related failures", markdown)
        self.assertIn("final cascades", markdown)
        self.assertNotIn(
            "gold_l1_related_failure_recall",
            {item["category"] for item in panel["concerns"]},
        )
        self.assertNotIn(
            "gold_l1_root_cause",
            {item["category"] for item in panel["concerns"]},
        )

    def test_panel_reports_fallback_and_enriched_policy_scores_separately(self) -> None:
        summary = {
            "target": "gpt",
            "model": "test-gpt",
            "decision": "STOP",
            "decision_paths": {
                "deterministic_fallback": {
                    "available": True,
                    "decision": "RESTART",
                    "retry_rule": "general_retry",
                },
                "l1_enriched": {
                    "available": True,
                    "decision": "STOP",
                    "retry_rule": "workload_unrecoverable",
                },
            },
            "gold_score": {
                "case_id": "code-error",
                "l1": {},
                "l2": {},
                "l4": {"action_correct": True, "policy_action_pass": True},
                "fallback_l4": {
                    "action_correct": False,
                    "policy_action_pass": False,
                },
                "enriched_l4": {
                    "action_correct": True,
                    "policy_action_pass": True,
                },
                "l4_path_comparison": {
                    "action_effect": "improved",
                    "policy_action_effect": "improved",
                },
            },
            "l1_kpis": {
                "execution_status": "ok",
                "successful_model_calls": 1,
                "output_usable": True,
            },
            "model_selection_signals": {"endpoint_reliability": "ok"},
            "path_redaction_audit": {"passed": True},
        }

        panel, markdown, _ = published_panel([summary])
        comparison = panel["decision_path_comparison"]

        self.assertEqual(comparison["fallback_consistency"], "consistent")
        self.assertEqual(comparison["shared_fallback"]["decision"], "RESTART")
        self.assertEqual(comparison["action_effect_counts"], {"improved": 1})
        self.assertIn("Fallback Versus L1-Enriched Policy", markdown)
        self.assertIn("| gpt | STOP | workload_unrecoverable | yes | yes | improved |", markdown)

    def test_panel_reports_shared_distributed_incident_without_member_fanout(
        self,
    ) -> None:
        incident = {
            "incident_id": "di-1",
            "incident_kind": "distributed_mechanism",
            "incident_type": "distributed_collective_timeout_wave",
            "primary_observed_line": 34362,
            "sample_lines": [34362, 34373],
            "event_count": 4317,
            "unique_operation_count": 4,
            "observed_rank_count": 1874,
            "root_cause_status": "unknown",
            "history_fingerprint": ("distributed_incident:collective_operation_timeout:steady_mid"),
        }
        summary = {
            "target": "qwen235b",
            "decision": "RESTART",
            "retry_policy": {"rule": "general_retry", "allowed_retries": 3},
            "primary_failure": {
                "fine_class": "collective_operation_timeout",
                "line": 34362,
            },
            "distributed_failure_incidents": [incident],
            "l0_bundle_kpis": {
                "line_count": 49984,
                "byte_size": 9079532,
                "failure_episode_count": 1,
                "distributed_failure_incident_count": 1,
                "distributed_failure_incidents": [incident],
                "first_terminal_incident_line": 34362,
                "first_terminal_incident_timestamp": "19:02:08.000203700",
                "configured_terminal_timeout_seconds": 600.0,
                "seconds_from_last_progress_to_terminal_incident": 605.806,
                "terminal_detection_lag_seconds": 5.806,
            },
        }

        panel, _, markdown = published_panel([summary])

        self.assertEqual(panel["shared_l0_shape"]["distributed_failure_incident_count"], 1)
        self.assertEqual(
            panel["shared_l0_execution"]["distributed_failure_incidents"],
            [incident],
        )
        self.assertIn("### Distributed Failure Incidents", markdown)
        self.assertIn("kind=`distributed_mechanism`", markdown)
        self.assertIn("events=`4317`", markdown)
        self.assertIn("detection_lag_s=`5.806`", markdown)
        self.assertNotIn("member_event_lines", markdown)
