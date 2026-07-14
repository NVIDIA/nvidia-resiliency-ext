# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mirrored corpus discovery and invalid gold-label boundary tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval import corpus  # noqa: E402
from restart_agent_eval import gold  # noqa: E402


def _label(**updates):
    value = {
        "schema_version": corpus.SCHEMA_VERSION,
        "case_id": "case-a",
        "label_version": 1,
        "review_status": "human_approved",
        "source_sha256": "0" * 64,
        "action_expectation": {"accepted": ["RESTART"]},
    }
    value.update(updates)
    return value


class CorpusTest(unittest.TestCase):
    def test_discovery_skips_private_gold_subtrees(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold_root = root / "gold"
            public = gold_root / "cases" / "case.log" / "gold.json"
            private = gold_root / "_scratch" / "case.log" / "gold.json"
            for path in (public, private):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(_label()), encoding="utf-8")

            cases = corpus.discover_cases(root / "logs", gold_root)

        self.assertEqual([case.label_path for case in cases], [public])

    def test_discovery_rejects_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "gold" / "case.log" / "gold.json"
            path.parent.mkdir(parents=True)
            path.write_text("{", encoding="utf-8")

            with self.assertRaises(ValueError):
                corpus.discover_cases(root / "logs", root / "gold")

    def test_case_rejects_non_object_wrong_schema_and_invalid_decisions(self) -> None:
        cases = (
            [],
            _label(schema_version="restart_agent_eval.v99"),
            _label(action_expectation={"accepted": []}),
            _label(action_expectation={"accepted": ["PAUSE"]}),
        )
        for label in cases:
            with self.subTest(label=label):
                with self.assertRaises((ValueError, gold.GoldSchemaError)):
                    corpus.case_from_label(
                        log_path=Path("missing.log"),
                        label_path=Path("gold.json"),
                        label=label,
                        default_case_id="fallback",
                    )

    def test_case_rejects_non_object_recovery_and_retry_expectations(self) -> None:
        for field in ("recovery_assessment_expectation", "retry_policy_expectation"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    corpus.case_from_label(
                        log_path=Path("missing.log"),
                        label_path=Path("gold.json"),
                        label=_label(**{field: ["not-an-object"]}),
                        default_case_id="fallback",
                    )

    def test_legacy_decision_and_default_case_id_are_normalized(self) -> None:
        label = _label(case_id="", action_expectation=None, decision="STOP", label_version=0)

        case = corpus.case_from_label(
            log_path=Path("missing.log"),
            label_path=Path("gold.json"),
            label=label,
            default_case_id="fallback",
        )

        self.assertEqual(case.case_id, "fallback")
        self.assertEqual(case.accepted_decisions, ("STOP",))
        self.assertEqual(case.label_version, 1)
        self.assertFalse(case.available)


if __name__ == "__main__":
    unittest.main()
