# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Corpus discovery, mirrored-path resolution, and label-validation tests."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from _evaluation_fixtures import write_evaluation_case  # noqa: E402
from _mocks import isolated_environment  # noqa: E402
from restart_agent_eval import evaluate as eval_harness  # noqa: E402


class EvaluationDiscoveryTest(unittest.TestCase):
    def test_discover_preserves_unavailable_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_root, gold_root = write_evaluation_case(root, with_log=False)
            cases = eval_harness.discover_cases(log_root, gold_root)

        self.assertEqual([case.case_id for case in cases], ["case-a"])
        self.assertFalse(cases[0].available)

    def test_product_repo_defaults_to_containing_checkout(self) -> None:
        with isolated_environment():
            args = eval_harness.parse_evaluation_args([])
        self.assertEqual(args.product_repo, eval_harness.REPO_ROOT)

    def test_discover_mirrored_gold_resolves_source_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_root = root / "logs"
            gold_root = root / "gold"
            relative = Path("checkpoint_logs/training.log")
            log_path = log_root / relative
            log_path.parent.mkdir(parents=True)
            log_path.write_text("root\n", encoding="utf-8")
            gold_path = gold_root / relative / "gold.json"
            gold_path.parent.mkdir(parents=True)
            gold_path.write_text(
                json.dumps(
                    {
                        "schema_version": eval_harness.SCHEMA_VERSION,
                        "label_version": 1,
                        "case_id": "checkpoint-training",
                        "review_status": "human_approved",
                        "source_sha256": hashlib.sha256(b"root\n").hexdigest(),
                        "action_expectation": {"accepted": ["RESTART"]},
                    }
                ),
                encoding="utf-8",
            )

            cases = eval_harness.discover_cases(log_root, gold_root)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "checkpoint-training")
        self.assertEqual(cases[0].log_path, log_path.resolve())
        self.assertEqual(cases[0].label_path, gold_path)

    def test_discover_rejects_non_human_gold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_root, gold_root = write_evaluation_case(root)
            gold_path = gold_root / "cases/case-a.log/gold.json"
            label = json.loads(gold_path.read_text(encoding="utf-8"))
            label["review_status"] = "model_generated"
            gold_path.write_text(json.dumps(label), encoding="utf-8")

            with self.assertRaises(ValueError):
                eval_harness.discover_cases(log_root, gold_root)

    def test_discover_rejects_source_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_root, gold_root = write_evaluation_case(root)
            (log_root / "cases/case-a.log").write_text("changed\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                eval_harness.discover_cases(log_root, gold_root)

    def test_documented_product_runner_wrapper_exists(self) -> None:
        self.assertEqual(eval_harness.REVIEW_LOG.name, "review_log.py")
        self.assertTrue(eval_harness.REVIEW_LOG.is_file())


if __name__ == "__main__":
    unittest.main()
