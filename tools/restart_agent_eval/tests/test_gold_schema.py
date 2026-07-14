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
