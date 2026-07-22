# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L2 grounding-audit scoring and advisory finding behavior."""

from __future__ import annotations

import unittest

from _bootstrap import configure_test_imports

configure_test_imports()

from _panel_fixtures import panel_summary  # noqa: E402
from restart_agent_eval import scoring  # noqa: E402


class AuditScoringTest(unittest.TestCase):
    def test_l2_citation_resolution_is_not_semantic_normalization(self) -> None:
        summary = scoring.semantic_safety_summary(
            l2_audit={
                "used": True,
                "model": "test-model",
                "model_failure_domain": "unknown",
                "model_failure_domain_status": "unknown",
                "model_retry_outlook_without_workload_change": "unknown",
                "model_retry_outlook_status": "unknown",
                "grounding_adjustments": [
                    {
                        "field": "evidence[1].line",
                        "from": 12072,
                        "to": 12073,
                        "reason": "nearby_unique_quote_match",
                    }
                ],
            },
            l4_policy={"retry_policy": {"rule": "general_retry"}},
        )

        self.assertEqual(summary["semantic_safety"], "ok")
        self.assertEqual(summary["normalization_count"], 1)
        self.assertEqual(summary["semantic_normalization_count"], 0)

    def test_retry_policy_signal_is_reported_without_rewriting_l1(self) -> None:
        semantic = scoring.semantic_safety_summary(
            l2_audit={
                "used": True,
                "model": "test-model",
                "model_failure_domain": "infrastructure",
                "model_failure_domain_status": "supported_but_unconfirmed",
                "model_retry_outlook_without_workload_change": "unknown",
                "model_retry_outlook_status": "unknown",
            },
            l4_policy={
                "retry_policy": {
                    "rule": "general_retry",
                    "retry_budget_exhausted": True,
                }
            },
        )
        signals = scoring.model_selection_signals(
            model_call_summary={},
            tool_efficiency={},
            semantic_safety=semantic,
        )
        self.assertEqual(signals["retry_policy_rule"], "general_retry")
        self.assertTrue(signals["retry_budget_exhausted"])
        self.assertEqual(signals["model_failure_domain"], "infrastructure")
        self.assertEqual(signals["model_failure_domain_status"], "supported_but_unconfirmed")

    def test_recovery_audit_preserves_raw_claim_and_reports_observation(
        self,
    ) -> None:
        semantic = scoring.semantic_safety_summary(
            l2_audit={
                "used": True,
                "model": "test-model",
                "model_retry_outlook_without_workload_change": "cannot_recover",
                "model_retry_outlook_status": "supported_but_unconfirmed",
                "recovery_field_audits": [
                    {
                        "field": "model_recovery_assessment.retry_outlook_without_workload_change",
                        "l1_value": "cannot_recover",
                        "audit_support": "weak",
                        "reason": "recovery mechanism remains unconfirmed",
                        "severity": "policy_material",
                        "applied": False,
                    }
                ],
            },
            l4_policy={"retry_policy": {}},
        )

        self.assertEqual(semantic["semantic_safety"], "recovery_audit_observation")
        self.assertEqual(semantic["model_retry_outlook_without_workload_change"], "cannot_recover")
        self.assertEqual(semantic["model_retry_outlook_status"], "supported_but_unconfirmed")
        self.assertFalse(semantic["l2_recovery_suggestion_applied"])
        self.assertEqual(semantic["recovery_audit_observation_count"], 1)

    def test_l2_kpis_separate_material_and_advisory_findings(self) -> None:
        kpis = scoring.l2_kpis(
            l2_audit={
                "used": True,
                "audit_status": "findings",
                "field_findings": {
                    "model_recovery_assessment": [
                        "missing confidence",
                        "recovery concern",
                    ]
                },
                "findings": [
                    {
                        "severity": "advisory",
                        "policy_material": False,
                    },
                    {
                        "severity": "policy_adjustment",
                        "policy_material": True,
                    },
                ],
            },
            l0_primary={"root_fingerprint": "observed:value_error"},
            timing={"l2_wall_clock_s": 0.01},
        )

        self.assertEqual(kpis["finding_count"], 2)
        self.assertEqual(kpis["material_finding_count"], 1)
        self.assertEqual(
            kpis["finding_severity_counts"],
            {"advisory": 1, "policy_adjustment": 1},
        )
        summaries = [
            {
                "target": "qwen235b",
                "model": "qwen235b",
                "l2_kpis": {
                    "audit_status": "findings",
                    "finding_count": 2,
                    "material_finding_count": 0,
                },
            }
        ]
        self.assertNotIn(
            "l2_audit",
            {item["category"] for item in panel_summary(summaries)["concerns"]},
        )

    def test_l2_audit_gold_is_optional_and_mechanical(self) -> None:
        gold = {
            "case_id": "validator-a",
            "label_version": 1,
            "l2_audit_expectation": [
                {
                    "field": "model_recovery_assessment",
                    "expected": "findings",
                }
            ],
        }
        l2_audit = {"field_audits": {"model_recovery_assessment": {"status": "findings"}}}

        scored = scoring.score_l2_audit(l2_audit, gold)
        unscored = scoring.score_l2_audit(l2_audit, {})

        self.assertTrue(scored)
        self.assertIsNone(unscored)
