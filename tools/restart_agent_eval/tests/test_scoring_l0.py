# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gold scoring for L0A collection and L0B attention projection."""

from __future__ import annotations

import unittest

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval import scoring  # noqa: E402


class L0GoldScoringTest(unittest.TestCase):
    def test_gold_can_validate_l0_setup_progress(self) -> None:
        score = scoring.score_against_gold(
            {},
            {
                "case_id": "port-conflict",
                "label_version": 1,
                "l0_expectation": {
                    "required_setup_marker_types": [
                        "checkpoint_load",
                        "cuda_graph_build",
                    ],
                    "minimum_setup_marker_count": 2,
                    "required_coverage": {"setup_progress": "found"},
                },
            },
            l0_bundle={
                "progress": {
                    "setup_markers": [
                        {"marker_type": "checkpoint_load", "line": 10},
                        {"marker_type": "cuda_graph_build", "line": 20},
                    ]
                },
                "evidence_coverage": {"setup_progress": "found"},
            },
        )

        self.assertTrue(score["l0a"]["overall_pass"])
        self.assertEqual(
            score["l0a"]["observed_setup_marker_types"],
            ["checkpoint_load", "cuda_graph_build"],
        )

    def test_gold_scores_l0_and_l2_history_identity_independently(self) -> None:
        score = scoring.score_against_gold(
            {},
            {
                "case_id": "identity-a",
                "label_version": 1,
                "l0_expectation": {"accepted_root_fingerprints": ["observed:unicode_decode_error"]},
                "history_identity_expectation": {
                    "operation": "checkpoint_load",
                    "mechanism": "checkpoint_unicode_decode_error",
                    "canonical_anchor_line": 12,
                    "expected_cross_route_identity_count": 1,
                },
            },
            l0_bundle={
                "deterministic_primary_candidate": {
                    "root_fingerprint": "observed:unicode_decode_error"
                },
                "operation_artifact_comparisons": [{"operation": "checkpoint_load"}],
            },
            l2_audit={
                "stable_root_fingerprint": "checkpoint:unicode:decode:error",
                "stable_identity_anchor_line": 12,
            },
        )

        self.assertTrue(score["l0a"]["root_fingerprint_accuracy"])
        self.assertTrue(score["l2"]["history_identity_correct"])
        self.assertTrue(score["l2"]["canonical_anchor_correct"])
        self.assertTrue(score["l2"]["operation_correct"])
        self.assertTrue(score["l2"]["mechanism_correct"])

    def test_gold_scores_observation_only_selection_without_requiring_a_primary(self) -> None:
        score = scoring.score_against_gold(
            {
                "primary_failure": None,
                "selected_observed_failure": {
                    "failure_class": "tcpstore_connection_loss",
                    "line": 20,
                    "causal_role": "cascade",
                },
                "l1_assessment": {
                    "primary_failure": None,
                    "observed_failures": [
                        {
                            "id": "o1",
                            "line": 20,
                            "causal_role": "cascade",
                            "failure_identity": {"mechanism": "tcpstore_connection_loss"},
                        }
                    ],
                    "selected_observed_failure_id": "o1",
                    "root_cause_assessment": {
                        "status": "unknown",
                        "missing_evidence": ["initiating failure is absent"],
                    },
                },
            },
            {
                "case_id": "app-only-transport-surface",
                "label_version": 1,
                "observation_expectation": {
                    "require_primary_absent": True,
                    "accepted_lines": [20],
                    "accepted_failure_classes": ["tcpstore_connection_loss"],
                    "accepted_causal_roles": ["cascade"],
                    "accepted_observation_fingerprints": ["transport:tcpstore_connection_loss"],
                },
                "history_identity_expectation": {
                    "identity_kind": "observation_only",
                    "mechanism": "tcpstore_connection_loss",
                    "canonical_anchor_line": 20,
                    "expected_cross_route_identity_count": 1,
                },
            },
            l0_bundle={
                "deterministic_primary_candidate": None,
                "selected_observed_failure": {
                    "failure_class": "tcpstore_connection_loss",
                    "line": 20,
                    "causal_role": "cascade",
                    "observation_fingerprint": "transport:tcpstore_connection_loss",
                },
            },
            l1_evidence={
                "primary_failure": None,
                "observed_failures": [
                    {
                        "id": "o1",
                        "line": 20,
                        "causal_role": "cascade",
                        "failure_identity": {"mechanism": "tcpstore_connection_loss"},
                    }
                ],
                "selected_observed_failure_id": "o1",
                "root_cause_assessment": {
                    "status": "unknown",
                    "missing_evidence": ["initiating failure is absent"],
                },
            },
            l2_audit={
                "identity_kind": "observation_only",
                "stable_observation_fingerprint": "transport:tcpstore_connection_loss",
                "stable_identity_anchor_line": 20,
            },
        )

        self.assertTrue(score["l0a"]["selected_observation_accuracy"])
        self.assertTrue(score["l0a"]["selected_observation_class_correct"])
        self.assertTrue(score["l0a"]["selected_observation_causal_role_correct"])
        self.assertTrue(score["l0a"]["observation_fingerprint_accuracy"])
        self.assertTrue(score["l0a"]["primary_absence_correct"])
        self.assertTrue(score["l1"]["observation_selection_correct"])
        self.assertTrue(score["l2"]["identity_kind_correct"])
        self.assertTrue(score["l2"]["history_identity_correct"])

    def test_gold_can_validate_operation_artifact_comparison(self) -> None:
        score = scoring.score_against_gold(
            {},
            {
                "case_id": "progress-log",
                "label_version": 1,
                "l0_expectation": {
                    "required_operation_artifact_comparisons": [
                        {
                            "operation": "checkpoint_save",
                            "logical_artifact_id": "ckpt#656375",
                            "minimum_success_count": 2,
                            "current_outcome": "started_not_completed",
                            "comparison_level": "same_operation_different_artifact",
                        }
                    ]
                },
            },
            l0_bundle={
                "operation_artifact_comparisons": [
                    {
                        "operation": "checkpoint_save",
                        "logical_artifact_id": "ckpt#656375",
                        "success_count": 2,
                        "current_outcome": "started_not_completed",
                        "comparison_level": "same_operation_different_artifact",
                    }
                ]
            },
        )

        self.assertTrue(score["l0a"]["overall_pass"])
        self.assertTrue(score["l0a"]["operation_artifact_comparison_checks"][0]["passed"])

    def test_gold_can_validate_terminal_and_recovered_failure_lifecycles(self) -> None:
        score = scoring.score_against_gold(
            {},
            {
                "case_id": "rdma-lifecycle",
                "label_version": 1,
                "l0_expectation": {
                    "required_failure_episodes": [
                        {
                            "lifecycle_family": "rdma_port",
                            "required_source_dialects": ["nccl_net_ib"],
                            "status": "recovered",
                            "accepted_start_lines": [20],
                            "minimum_fault_count": 4,
                            "minimum_recovery_attempt_count": 4,
                            "minimum_recovery_confirmation_count": 4,
                            "required_entities": ["node-a/mlx5_1:1"],
                            "first_progress_after_line": 40,
                        },
                        {
                            "lifecycle_family": "rdma_port",
                            "required_source_dialects": ["nccl_net_ib"],
                            "status": "terminal",
                            "accepted_start_lines": [50],
                            "minimum_fault_count": 2,
                            "maximum_recovery_confirmation_count": 0,
                            "terminal_exception_line": 60,
                        },
                    ]
                },
            },
            l0_bundle={
                "failure_episodes": [
                    {
                        "lifecycle_family": "rdma_port",
                        "lifecycle_source_dialects": ["nccl_net_ib"],
                        "status": "recovered",
                        "start_line": 20,
                        "lifecycle_entities": ["node-a/mlx5_1:1"],
                        "lifecycle_fault_lines": [20, 21, 22, 23],
                        "recovery_attempt_lines": [24, 25, 26, 27],
                        "recovery_confirmation_lines": [28, 29, 30, 31],
                        "first_progress_after": {"line": 40},
                    },
                    {
                        "lifecycle_family": "rdma_port",
                        "lifecycle_source_dialects": ["nccl_net_ib"],
                        "status": "terminal",
                        "start_line": 50,
                        "lifecycle_fault_lines": [50, 51],
                        "recovery_attempt_lines": [],
                        "recovery_confirmation_lines": [],
                        "terminal_exception_line": 60,
                    },
                ]
            },
        )

        self.assertTrue(score["l0a"]["overall_pass"])
        self.assertEqual(
            [item["passed"] for item in score["l0a"]["failure_episode_checks"]],
            [True, True],
        )

    def _l0_projection_score(self):
        score = scoring.score_against_gold(
            {},
            {
                "case_id": "bounded-view",
                "label_version": 1,
                "l0_expectation": {
                    "accepted_primary_lines": [12083],
                    "required_progress_lines": [12000],
                    "required_checkpoint_lines": [11990],
                    "expected_primary_phase": "setup",
                    "expected_checkpoint_load_iteration": 5000,
                    "expected_progress_after_failure_episode": False,
                    "required_cascade_lines": [12123],
                },
                "l0b_expectation": {
                    "accepted_primary_lines": [12083],
                    "required_evidence_lines": [12000, 12083],
                    "required_reference_ids": {"context_window_ids": ["w-primary"]},
                },
            },
            l0_bundle={
                "deterministic_primary_candidate": {"line": 12083, "phase": "setup"},
                "candidate_anchors": [{"anchor_id": "a-primary", "line": 12083}],
                "context_windows": [
                    {
                        "window_id": "w-primary",
                        "start_line": 12000,
                        "end_line": 12083,
                        "lines": [{"line": 12000}, {"line": 12083}],
                    }
                ],
                "progress": {
                    "application_markers": [{"line": 12000}],
                    "setup_markers": [{"marker_type": "checkpoint_load", "line": 11990}],
                },
                "run_progress_summary": {
                    "checkpoint_load_iteration": 5000,
                    "progress_after_failure_episode": False,
                },
                "cascades": [{"first_line": 12123, "last_line": 12180}],
            },
            l0_model_view={
                "evidence_bundle": {
                    "context_windows": [
                        {
                            "window_id": "w-primary",
                            "lines": [
                                {"line": 12000, "text": "iteration completed"},
                                {"line": 12083, "text": "UnicodeDecodeError"},
                            ],
                        }
                    ]
                },
                "projection_metrics": {"projection_integrity": {"status": "ok", "checks": {}}},
            },
        )
        return score

    def test_gold_scores_l0a_primary_quality_dimensions(self) -> None:
        score = self._l0_projection_score()

        for field in (
            "primary_evidence_coverage",
            "selected_primary_accuracy",
            "progress_line_recall",
            "checkpoint_line_recall",
            "primary_phase_correct",
            "checkpoint_load_iteration_correct",
            "progress_after_failure_correct",
            "cascade_line_recall",
        ):
            with self.subTest(field=field):
                self.assertTrue(score["l0a"][field])

    def test_gold_scores_l0b_retention_dimensions(self) -> None:
        score = self._l0_projection_score()

        self.assertTrue(score["l0b"]["primary_retained_from_l0a"])
        self.assertTrue(score["l0b"]["required_evidence_line_recall"])
        self.assertTrue(score["l0b"]["overall_pass"])

    def test_l0b_counts_rendered_candidate_and_occurrence_group_evidence(self) -> None:
        score = scoring.score_against_gold(
            {},
            {
                "case_id": "rendered-group-evidence",
                "label_version": 1,
                "l0_expectation": {"accepted_primary_lines": [100]},
                "l0b_expectation": {
                    "accepted_primary_lines": [100],
                    "required_evidence_lines": [100, 200, 300],
                },
            },
            l0_bundle={
                "candidate_anchors": [
                    {"anchor_id": "a-primary", "line": 100},
                    {"anchor_id": "a-cascade", "line": 200},
                ],
                "occurrence_groups": [{"occurrence_group_id": "og-cascade", "first_line": 300}],
            },
            l0_model_view={
                "evidence_bundle": {
                    "context_windows": [
                        {
                            "window_id": "w-primary",
                            "lines": [{"line": 100, "text": "primary timeout"}],
                        }
                    ],
                    "candidate_anchors": [
                        {"anchor_id": "a-cascade", "line": 200, "quote": "cascade"}
                    ],
                    "registry_candidate_groups": [
                        {
                            "registry_id": "nccl_cascade",
                            "first_line": 300,
                            "representative_quote": "distributed cascade fanout",
                        }
                    ],
                },
                "projection_metrics": {"projection_integrity": {"status": "ok", "checks": {}}},
            },
        )

        self.assertEqual(score["l0b"]["visible_evidence_line_count"], 3)
        self.assertTrue(score["l0b"]["required_evidence_line_recall"])
        self.assertTrue(score["l0b"]["overall_pass"])
