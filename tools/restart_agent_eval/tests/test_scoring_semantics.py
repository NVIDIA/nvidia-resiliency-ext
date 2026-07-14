# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gold scoring for L1 root cause, recovery, and related-failure semantics."""

from __future__ import annotations

import unittest

from _bootstrap import configure_test_imports

configure_test_imports()

from _builders import recovery_assessment, retry_policy  # noqa: E402
from restart_agent_eval import scoring  # noqa: E402


class SemanticGoldScoringTest(unittest.TestCase):
    def test_l2_grounding_scores_its_grounded_primary_contract(self) -> None:
        gold = {
            "case_id": "grounded-primary",
            "label_version": 1,
            "primary_anchor_expectation": {
                "accepted_lines": [33464],
                "tolerance_lines": 0,
            },
        }
        l1 = {
            "analysis_status": "primary_identified",
            "primary_failure": {"line": 33464},
            "root_cause_assessment": {"status": "unknown"},
        }
        l2 = {
            "grounding_status": "grounded",
            "grounded_primary_failure": {
                "line": 33464,
                "causal_role": "initiating",
                "root_fingerprint": (
                    "distributed_incident:collective_operation_timeout:steady_mid"
                ),
            },
            "history_identities": {
                "primary": {
                    "ready": True,
                    "identity_kind": "root",
                    "root_fingerprint": (
                        "distributed_incident:collective_operation_timeout:steady_mid"
                    ),
                }
            },
        }

        score = scoring.score_against_gold(
            {},
            gold,
            l1_evidence=l1,
            l2_grounding=l2,
        )

        self.assertEqual(score["l2"]["primary_grounding_effect"], "neutral")
        self.assertTrue(score["l2"]["grounding_score"]["primary_anchor_correct"])
        self.assertEqual(score["l2"]["grounding_score"]["grounded_anchor_line"], 33464)
        self.assertEqual(
            score["l2"]["grounding_score"]["root_fingerprint"],
            "distributed_incident:collective_operation_timeout:steady_mid",
        )
        self.assertNotIn("grounded_primary_projection", score["l2"])

    def test_root_cause_scoring_rejects_a_contradictory_identity_mechanism(self) -> None:
        gold = {
            "primary_anchor_expectation": {"accepted_lines": [10], "tolerance_lines": 0},
            "root_cause_expectation": {
                "required_concept_groups": [["checkpoint"], ["position mismatch"]],
                "accepted_operations": ["checkpoint_save"],
                "rejected_mechanism_terms": ["read", "load"],
            },
        }
        view = {
            "primary_failure": {
                "line": 10,
                "signature": "checkpoint position mismatch",
                "failure_identity": {
                    "operation": "checkpoint_save",
                    "mechanism": "filesystem_read_corruption",
                },
            },
            "root_cause_assessment": {"summary": "checkpoint position mismatch"},
        }

        score = scoring.score_semantic_view(view, gold, include_action=False)

        self.assertFalse(score["root_cause_correct"])
        self.assertTrue(score["root_cause_operation_correct"])
        self.assertTrue(score["root_cause_mechanism_contradiction"])

    def test_gold_scores_recovery_policy_and_unsupported_certainty_separately(
        self,
    ) -> None:
        gold = {
            "case_id": "checkpoint-a",
            "label_version": 1,
            "primary_anchor_expectation": {
                "accepted_lines": [12083],
                "tolerance_lines": 0,
            },
            "root_cause_expectation": {
                "required_concept_groups": [["checkpoint"], ["decode", "unicode"]]
            },
            "recovery_assessment_expectation": {
                "failure_domain": ["workload"],
                "failure_domain_status": ["supported_but_unconfirmed"],
                "retry_outlook_without_workload_change": ["may_recover"],
                "retry_outlook_status": ["supported_but_unconfirmed"],
            },
            "retry_policy_expectation": {
                "accepted_rules": ["concrete_confirmation_retry"],
                "allowed_retries": 1,
                "retry_budget_exhausted": False,
            },
            "action_expectation": {"accepted": ["RESTART"]},
            "cascade_expectation": {"expected_lines": [12123]},
            "unsupported_claims": [
                {"id": "proven_corruption", "text_patterns": ["persistent corruption"]}
            ],
        }
        analysis = {
            "analysis_status": "primary_identified",
            "decision": "STOP",
            "primary_failure": {
                "line": 12083,
                "failure_class": "checkpoint_unicode_decode",
            },
            "root_cause_assessment": {"summary": "checkpoint metadata decode failed"},
            "model_recovery_assessment": recovery_assessment(
                failure_domain="workload",
                failure_domain_status="established_by_current_log",
                retry_outlook="cannot_recover",
                retry_outlook_status="established_by_current_log",
                failure_domain_confidence=95,
                retry_outlook_confidence=95,
                rationale="persistent corruption",
            ),
            "retry_policy": retry_policy(
                base_rule="workload_unrecoverable",
                allowed_retries=0,
                retry_budget_exhausted=True,
            ),
            "cascades": [{"first_line": 12123, "last_line": 12326}],
            "justification": "checkpoint decode failed",
        }

        score = scoring.score_against_gold(
            analysis,
            gold,
            l1_evidence=analysis,
        )

        self.assertTrue(score["l1"]["root_cause_correct"])
        self.assertFalse(score["l1"]["recovery_assessment_correct"])
        self.assertEqual(score["l1"]["unsupported_claims"], ["proven_corruption"])
        self.assertFalse(score["l4"]["retry_rule_correct"])
        self.assertFalse(score["l4"]["allowed_retries_correct"])
        self.assertTrue(score["l4"]["retry_exhaustion_correct"])
        self.assertFalse(score["l4"]["action_correct"])
        self.assertEqual(score["l2"]["primary_grounding_effect"], "not_comparable")
        self.assertIsNone(score["calibration_score"])
        calibration = score["calibration_eligibility"]
        self.assertEqual(calibration["eligible_claim_count"], 2)
        self.assertEqual(
            calibration["claims"]["failure_domain"]["calibration_confidence"],
            95,
        )
        self.assertFalse(calibration["claims"]["failure_domain"]["gold_correct"])

    def test_non_primary_confidence_is_retained_but_not_calibration_eligible(self) -> None:
        analysis = {
            "analysis_status": "no_failure_observed",
            "primary_failure": None,
            "root_cause_assessment": {"status": "unknown"},
            "model_recovery_assessment": recovery_assessment(
                failure_domain="unknown",
                failure_domain_status="unknown",
                retry_outlook="unknown",
                retry_outlook_status="unknown",
                failure_domain_confidence=1,
                retry_outlook_confidence=1,
                rationale="No failure was observed.",
            ),
        }

        score = scoring.score_against_gold(analysis, {}, l1_evidence=analysis)

        calibration = score["calibration_eligibility"]
        self.assertEqual(calibration["eligible_claim_count"], 0)
        for claim in calibration["claims"].values():
            self.assertEqual(claim["reason"], "non_primary_placeholder")
            self.assertEqual(claim["raw_confidence"], 1)
            self.assertIsNone(claim["calibration_confidence"])

    def test_unknown_primary_claim_confidence_is_not_calibration_eligible(self) -> None:
        analysis = {
            "analysis_status": "primary_identified",
            "primary_failure": {"line": 12, "causal_role": "initiating"},
            "root_cause_assessment": {"status": "supported_but_unconfirmed"},
            "model_recovery_assessment": recovery_assessment(
                failure_domain="workload",
                failure_domain_confidence=73,
                retry_outlook="unknown",
                retry_outlook_status="unknown",
                retry_outlook_confidence=94,
            ),
        }

        score = scoring.score_against_gold(analysis, {}, l1_evidence=analysis)

        calibration = score["calibration_eligibility"]
        self.assertEqual(calibration["eligible_claim_count"], 1)
        self.assertTrue(calibration["claims"]["failure_domain"]["eligible"])
        unknown_claim = calibration["claims"]["retry_outlook_without_workload_change"]
        self.assertFalse(unknown_claim["eligible"])
        self.assertEqual(unknown_claim["reason"], "unknown_claim")
        self.assertEqual(unknown_claim["raw_confidence"], 94)
        self.assertIsNone(unknown_claim["calibration_confidence"])

    def test_gold_requires_teardown_role_when_labeled(self) -> None:
        gold = {
            "primary_anchor_expectation": {
                "accepted_lines": [10],
                "tolerance_lines": 0,
            },
            "root_cause_expectation": {},
            "cascade_expectation": {"teardown_lines": [20]},
        }
        view = {
            "primary_failure": {"line": 10, "failure_class": "primary_failure"},
            "root_cause_assessment": {"summary": "observed failure"},
            "model_recovery_assessment": {},
            "cascades": [
                {
                    "first_line": 20,
                    "last_line": 20,
                    "causal_role": "teardown",
                }
            ],
        }

        correct = scoring.score_semantic_view(view, gold, include_action=False)
        view["cascades"][0]["causal_role"] = "cascade"
        wrong_role = scoring.score_semantic_view(view, gold, include_action=False)

        self.assertTrue(correct["teardown_role_correct"])
        self.assertTrue(correct["cascade_correct"])
        self.assertFalse(wrong_role["teardown_role_correct"])
        self.assertFalse(wrong_role["cascade_correct"])

    def test_l1_core_semantics_do_not_fail_on_related_failure_recall(self) -> None:
        gold = {
            "case_id": "checkpoint-a",
            "label_version": 1,
            "primary_anchor_expectation": {
                "accepted_lines": [12083],
                "tolerance_lines": 0,
            },
            "root_cause_expectation": {"required_concept_groups": [["checkpoint"], ["decode"]]},
            "recovery_assessment_expectation": {
                "failure_domain": ["workload"],
                "retry_outlook_without_workload_change": ["may_recover"],
            },
            "cascade_expectation": {"expected_lines": [12123]},
        }
        l1 = {
            "primary_failure": {"line": 12083, "failure_class": "checkpoint_decode"},
            "root_cause_assessment": {"summary": "checkpoint metadata decode failed"},
            "model_recovery_assessment": recovery_assessment(
                failure_domain="workload",
                retry_outlook="may_recover",
            ),
            "related_failures": [],
        }

        score = scoring.score_semantic_view(l1, gold, include_action=False)

        self.assertTrue(score["core_semantic_pass"])
        self.assertFalse(score["related_failure_recall"])
        self.assertFalse(score["overall_semantic_pass"])
