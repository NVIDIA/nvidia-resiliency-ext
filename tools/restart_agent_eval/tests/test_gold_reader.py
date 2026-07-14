# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gold-label filesystem loading and obsolete-vocabulary validation tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval import scoring  # noqa: E402


class GoldReaderTest(unittest.TestCase):
    def test_rejects_obsolete_operation_history_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gold.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "restart_agent_eval.v1",
                        "case_id": "stale-operation-history",
                        "label_version": 1,
                        "review_status": "human_approved",
                        "source_sha256": "0" * 64,
                        "l0_expectation": {
                            "required_coverage": {"operation_history": "found"},
                            "required_operation_histories": [],
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit):
                scoring.read_gold_label(path)

    def test_accepts_operation_artifact_comparison_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gold.json"
            expected = {
                "schema_version": "restart_agent_eval.v1",
                "case_id": "current-operation-history",
                "label_version": 1,
                "review_status": "human_approved",
                "source_sha256": "0" * 64,
                "l0_expectation": {
                    "required_coverage": {"operation_artifact_comparisons": "found"},
                    "required_operation_artifact_comparisons": [
                        {
                            "operation": "checkpoint_save",
                            "minimum_success_count": 2,
                            "current_outcome": "started_not_completed",
                        }
                    ],
                },
                "human_assessment": {"free_form_note": "retained as rationale"},
            }
            path.write_text(json.dumps(expected), encoding="utf-8")

            actual = scoring.read_gold_label(path)

            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
