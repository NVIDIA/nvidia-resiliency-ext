# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Panel aggregation and cross-route comparison behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from _panel_fixtures import panel_summary as _panel_summary  # noqa: E402
from _panel_fixtures import rich_panel_artifacts  # noqa: E402
from restart_agent_eval import panel as summarize_review_panel  # noqa: E402


class PanelAggregationTest(unittest.TestCase):
    def test_panel_reads_resolved_restart_agent_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            config = {
                "config_id": "qwen-fast-and-enriched",
                "config_version": 1,
                "config_fingerprint": "sha256:test",
            }
            (run_dir / "restart_agent.result.json").write_text(
                json.dumps({"shared_analysis": {"restart_agent_config": config}}),
                encoding="utf-8",
            )

            json_path, _ = summarize_review_panel.write_panel_summary(
                run_dir,
                [{"target": "deterministic"}],
            )
            panel = json.loads(json_path.read_text(encoding="utf-8"))

            self.assertEqual(panel["restart_agent_config"], config)

    def test_missing_failed_route_does_not_imply_payload_inconsistency(self) -> None:
        evidence = {"schema_version": "restart_agent_decision_evidence.v1"}
        restart_context = {"process_state_recreated": True}
        panel = _panel_summary(
            [
                {
                    "target": "qwen397b",
                    "model": "qwen397b",
                    "decision_evidence": evidence,
                    "l0_kpis": {"restart_environment_context": restart_context},
                },
                {"target": "claude", "model": "claude"},
            ],
        )

        self.assertEqual(
            panel["decision_evidence_consistency"]["status"],
            "consistent_among_available",
        )
        self.assertEqual(panel["shared_decision_evidence"], evidence)
        self.assertEqual(
            panel["restart_environment_context_consistency"]["status"],
            "consistent_among_available",
        )
        self.assertEqual(panel["shared_restart_environment_context"], restart_context)
        self.assertNotIn(
            "l0_decision_evidence_consistency",
            {item["category"] for item in panel["concerns"]},
        )

    def test_should_publish_effective_tool_profile_for_route(self) -> None:
        panel = _panel_summary(
            [
                {
                    "target": "qwen235b",
                    "effective_tool_profile": {
                        "profile_id": "qwen235b.experimental.one_tool_round.v1",
                        "experimental": True,
                        "tools_enabled": True,
                        "max_tool_rounds": 1,
                        "max_model_turns": 2,
                        "source": "target_profile",
                    },
                }
            ]
        )
        row = panel["rows"][0]

        self.assertEqual(row["max_tool_rounds"], 1)
        self.assertEqual(row["max_model_turns"], 2)
        self.assertEqual(row["tool_profile_source"], "target_profile")

    def test_panel_exposes_stage_semantics_and_route_metrics(self) -> None:
        summary, _, panel, row, _, _ = rich_panel_artifacts()

        self.assertEqual(row["causal_role"], "initiating")
        self.assertEqual(
            row["model_recovery_confidence"],
            {"failure_domain": 75, "retry_outlook": 75},
        )
        self.assertEqual(row["l1_execution_status"], "ok")
        self.assertTrue(row["l1_contract_repair_requested"])
        self.assertEqual(row["l1_contract_repair_turns"], 1)
        self.assertEqual(row["l1_primary_relation_to_l0"], "same_failure_episode")
        self.assertEqual(panel["shared_decision_evidence"], summary["decision_evidence"])
        self.assertEqual(panel["decision_evidence_consistency"]["status"], "consistent")
        self.assertEqual(
            set(panel["comparison_axes"]),
            {
                "semantic_quality",
                "behavioral_efficiency",
                "endpoint_reliability",
                "route_outcome",
            },
        )
        self.assertIn("l1_contract_repair", {item["category"] for item in panel["concerns"]})

    def test_panel_reports_fingerprint_stability_and_disagreement_reason(self) -> None:
        summary, second, panel, _, _, _ = rich_panel_artifacts()

        self.assertEqual(panel["l2_root_fingerprint_agreement"]["status"], "stable")
        self.assertTrue(panel["l2_root_fingerprint_agreement"]["all_available_agree"])
        self.assertTrue(panel["client_concrete_agreement"]["all_available_agree"])

        different_primary = json.loads(json.dumps(second))
        different_primary["target"] = "claude"
        different_primary["primary_failure"]["line"] = 12084
        different_primary["primary_failure"]["failure_identity"]["client_concrete"][
            "fingerprint"
        ] = "client_concrete:sha256:different"
        split_panel = _panel_summary([summary, different_primary])
        self.assertEqual(
            split_panel["client_concrete_agreement"]["disagreement_reason"],
            "primary_selection_disagreement",
        )

        different_root = json.loads(json.dumps(second))
        different_root["target"] = "gemini"
        different_root["l2_kpis"]["root_fingerprint"] = "checkpoint_decode_error:different"
        unstable_panel = _panel_summary([summary, different_root])
        self.assertEqual(unstable_panel["l2_root_fingerprint_agreement"]["status"], "unstable")
        self.assertIn(
            "l2_root_fingerprint_stability",
            {item["category"] for item in unstable_panel["concerns"]},
        )

    def test_panel_reports_inconsistent_decision_evidence(self) -> None:
        summary, second, _, _, _, _ = rich_panel_artifacts()
        different_evidence = json.loads(json.dumps(second))
        different_evidence["target"] = "gemini"
        different_evidence["decision_evidence"]["deterministic_primary_candidate"]["line"] = 12081

        panel = _panel_summary([summary, different_evidence])

        self.assertEqual(panel["decision_evidence_consistency"]["status"], "inconsistent")
        self.assertEqual(panel["shared_decision_evidence"], {})
        self.assertIn(
            "l0_decision_evidence_consistency",
            {item["category"] for item in panel["concerns"]},
        )

    def test_comparison_axes_separate_model_endpoint_and_route_outcomes(self) -> None:
        panel = _panel_summary(
            [
                {
                    "target": "timeout-route",
                    "model": "timeout-model",
                    "gold_score": {
                        "case_id": "case-1",
                        "l1": {"core_semantic_pass": False},
                    },
                    "l1_kpis": {
                        "successful_model_calls": 0,
                        "model_calls": 1,
                        "failed_model_calls": 1,
                        "timeout_model_calls": 1,
                        "output_usable": False,
                        "execution_status": "failed",
                        "endpoint_reliability": "failed",
                    },
                    "decision": "RESTART",
                    "l4_kpis": {
                        "result_quality": "degraded",
                        "nvrx_use": "eligible_degraded",
                    },
                },
                {
                    "target": "usable-route",
                    "model": "usable-model",
                    "gold_score": {
                        "case_id": "case-1",
                        "l1": {"core_semantic_pass": True},
                    },
                    "l1_kpis": {
                        "successful_model_calls": 1,
                        "model_calls": 1,
                        "model_turns": 1,
                        "tool_driven_model_turns": 0,
                        "contract_repair_turns": 0,
                        "output_usable": True,
                        "execution_status": "ok",
                        "endpoint_reliability": "ok",
                    },
                    "decision": "RESTART",
                    "l4_kpis": {
                        "result_quality": "normal",
                        "nvrx_use": "eligible",
                    },
                },
            ]
        )
        axes = panel["comparison_axes"]

        self.assertEqual(axes["semantic_quality"][0]["status"], "not_observed")
        self.assertEqual(axes["semantic_quality"][1]["status"], "pass")
        self.assertIsNone(axes["behavioral_efficiency"][0]["first_turn_usable"])
        self.assertTrue(axes["behavioral_efficiency"][1]["first_turn_usable"])
        self.assertEqual(
            axes["route_outcome"][0]["model_contribution"],
            "fallback_only",
        )
        self.assertEqual(axes["route_outcome"][0]["nvrx_use"], "eligible_fallback")
        self.assertEqual(axes["route_outcome"][0]["reason"], "no_model_enrichment")
        self.assertEqual(
            axes["route_outcome"][1]["model_contribution"],
            "model_enriched",
        )

    def test_panel_excludes_deterministic_fallback_from_model_scoring(self) -> None:
        deterministic = {
            "target": "deterministic",
            "model": None,
            "decision": "RESTART",
            "gold_score": {
                "case_id": "checkpoint-a",
                "l1": {
                    "root_cause_correct": False,
                    "recovery_assessment_correct": False,
                    "core_semantic_pass": False,
                },
                "l2": {},
                "l4": {
                    "action_correct": True,
                    "retry_rule_correct": True,
                    "policy_action_pass": True,
                    "overall_semantic_pass": False,
                },
            },
            "l1_kpis": {"execution_status": "in_flight"},
            "model_selection_signals": {"endpoint_reliability": "ok"},
            "path_redaction_audit": {"passed": True},
        }
        model = {
            **deterministic,
            "target": "gpt",
            "model": "test-gpt",
            "l1_kpis": {
                "execution_status": "ok",
                "successful_model_calls": 1,
                "output_usable": True,
            },
            "l2_kpis": {"root_fingerprint": "root-a"},
        }

        panel = _panel_summary([deterministic, model])
        deterministic_row = panel["rows"][0]

        self.assertEqual(deterministic_row["l1_execution_status"], "not_run")
        self.assertIsNone(deterministic_row["gold_l1_core_semantic"])
        self.assertTrue(deterministic_row["gold_l4_policy_action"])
        self.assertEqual(panel["l2_root_fingerprint_agreement"]["total_models"], 1)
        self.assertNotIn(
            "deterministic",
            {
                item["target"]
                for item in panel["concerns"]
                if str(item["category"]).startswith("gold_l1_")
            },
        )

    def test_panel_uses_published_tool_turn_count(self) -> None:
        panel = _panel_summary(
            [
                {
                    "target": "current-review",
                    "model": "current-review",
                    "l1_response_parsed": True,
                    "l1_kpis": {
                        "output_usable": True,
                        "contract_repair_requested": True,
                    },
                    "tool_efficiency": {
                        "tool_driven_model_turns": 2,
                    },
                    "model_selection_signals": {"endpoint_reliability": "ok"},
                }
            ]
        )
        row = panel["rows"][0]

        self.assertEqual(row["l1_kpi_tool_driven_model_turns"], 2)
        self.assertEqual(row["l1_contract_repair_turns"], 1)
