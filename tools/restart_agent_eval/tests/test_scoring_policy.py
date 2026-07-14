# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gold scoring for deterministic fallback and L1-enriched L4 policy."""

from __future__ import annotations

import unittest

from _bootstrap import configure_test_imports

configure_test_imports()

from _builders import recovery_assessment, retry_policy  # noqa: E402
from restart_agent_eval import scoring  # noqa: E402


class PolicyGoldScoringTest(unittest.TestCase):
    def test_gold_scores_fallback_and_enriched_policy_paths_separately(self) -> None:
        gold = {
            "case_id": "code-error",
            "label_version": 1,
            "retry_policy_expectation": {
                "accepted_rules": ["workload_unrecoverable"],
                "allowed_retries": 0,
                "retry_budget_exhausted": True,
            },
            "action_expectation": {"accepted": ["STOP"]},
        }
        fallback = {
            "decision": "RESTART",
            "retry_policy": retry_policy(allowed_retries=1),
        }
        enriched = {
            "decision": "STOP",
            "retry_policy": retry_policy(
                rule="workload_unrecoverable",
                allowed_retries=0,
                retry_budget_exhausted=True,
            ),
        }

        score = scoring.score_against_gold(
            enriched,
            gold,
            fallback_analysis=fallback,
            enriched_analysis=enriched,
        )

        self.assertFalse(score["fallback_l4"]["action_correct"])
        self.assertFalse(score["fallback_l4"]["policy_action_pass"])
        self.assertTrue(score["enriched_l4"]["action_correct"])
        self.assertTrue(score["enriched_l4"]["policy_action_pass"])
        self.assertEqual(score["l4_path_comparison"]["action_effect"], "improved")
        self.assertEqual(score["l4_path_comparison"]["policy_action_effect"], "improved")

    def test_l4_policy_action_pass_is_independent_of_l1_semantics(self) -> None:
        gold = {
            "primary_anchor_expectation": {"accepted_lines": [10]},
            "root_cause_expectation": {"required_concept_groups": [["checkpoint"]]},
            "recovery_assessment_expectation": {"failure_domain": ["workload"]},
            "retry_policy_expectation": {"accepted_rules": ["general_retry"]},
            "action_expectation": {"accepted": ["RESTART"]},
        }
        view = {
            "decision": "RESTART",
            "primary_failure": {"line": 99, "fine_class": "unknown"},
            "root_cause_assessment": {"summary": "unknown failure"},
            "model_recovery_assessment": recovery_assessment(),
            "retry_policy": {"rule": "general_retry"},
        }

        score = scoring.score_semantic_view(view, gold, include_action=True)

        self.assertFalse(score["core_semantic_pass"])
        self.assertTrue(score["policy_action_pass"])
        self.assertFalse(score["overall_semantic_pass"])
