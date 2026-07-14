# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gold-label schema and source-identity contract tests."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval import gold  # noqa: E402


class GoldContractTest(unittest.TestCase):
    def test_scored_label_requires_human_review_and_source_digest(self) -> None:
        label = {
            "schema_version": "restart_agent_eval.v1",
            "case_id": "case-a",
            "label_version": 1,
            "review_status": "human_approved",
            "source_sha256": "0" * 64,
        }

        actual = gold.validate_scored_gold_label(label)

        self.assertIsNone(actual)

    def test_unknown_gold_field_is_rejected(self) -> None:
        with self.assertRaises(gold.GoldSchemaError):
            gold.validate_gold_label({"invented": True})

    def test_observation_only_expectations_are_declared_schema(self) -> None:
        gold.validate_gold_label(
            {
                "observation_expectation": {
                    "require_primary_absent": True,
                    "accepted_lines": [20],
                    "line_tolerance": 2,
                    "accepted_failure_classes": ["tcpstore_connection_loss"],
                    "accepted_causal_roles": ["cascade"],
                    "accepted_observation_fingerprints": ["transport:tcpstore_connection_loss"],
                },
                "history_identity_expectation": {
                    "identity_kind": "observation_only",
                    "canonical_anchor_line": 20,
                    "expected_cross_route_identity_count": 1,
                },
            }
        )

    def test_l0_failure_episode_expectation_is_declared_schema(self) -> None:
        gold.validate_gold_label(
            {
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
                            "maximum_recovery_confirmation_count": 8,
                            "required_entities": ["node-a/mlx5_1:1"],
                            "first_progress_after_line": 40,
                        }
                    ]
                }
            }
        )

    def test_source_sha256_streams_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.log"
            path.write_bytes(b"abcdefgh")

            actual = gold.source_sha256(path, chunk_size=3)

        self.assertEqual(actual, hashlib.sha256(b"abcdefgh").hexdigest())

    def test_scored_label_rejects_review_and_digest_variations(self) -> None:
        base = {
            "schema_version": "restart_agent_eval.v1",
            "case_id": "case-a",
            "label_version": 1,
            "review_status": "human_approved",
            "source_sha256": "0" * 64,
        }
        for field, value in (
            ("review_status", "model_generated"),
            ("source_sha256", "A" * 64),
            ("source_sha256", "too-short"),
            ("source_sha256", None),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(gold.GoldSchemaError):
                    gold.validate_scored_gold_label({**base, field: value})

    def test_nested_gold_fields_must_have_declared_shapes(self) -> None:
        invalid_labels = (
            {"l0_expectation": []},
            {"l0_expectation": {"required_coverage": []}},
            {"l0b_expectation": {"required_reference_ids": []}},
            {"unsupported_claims": {}},
            {"unsupported_claims": ["not-an-object"]},
            {"l2_audit_expectation": [{"unexpected": True}]},
        )
        for label in invalid_labels:
            with self.subTest(label=label):
                with self.assertRaises(gold.GoldSchemaError):
                    gold.validate_gold_label(label)

    def test_gold_source_must_match_reviewed_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.log"
            path.write_text("current bytes\n", encoding="utf-8")

            with self.assertRaises(gold.GoldSchemaError):
                gold.validate_gold_source({"source_sha256": "0" * 64}, path)


if __name__ == "__main__":
    unittest.main()
