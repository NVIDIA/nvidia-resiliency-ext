# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval import scoring  # noqa: E402


class PathRedactionAuditTest(unittest.TestCase):
    def _source_log(self, root: Path) -> Path:
        source_log = (
            root / "nvrx_log_analyzer" / "elk" / "logs" / "checkpoint_logs" / "training.log"
        )
        source_log.parent.mkdir(parents=True)
        source_log.write_text("training\n", encoding="utf-8")
        return source_log

    def test_generic_logs_word_is_not_a_path_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_log = self._source_log(Path(tmp))
            audit = scoring.path_redaction_audit(
                source_log,
                [{"role": "system", "content": "Analyze distributed training logs."}],
            )

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["source_path_tokens_found"], [])

    def test_distinctive_parent_component_is_a_path_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_log = self._source_log(Path(tmp))
            audit = scoring.path_redaction_audit(
                source_log,
                [{"role": "user", "content": "case came from checkpoint_logs"}],
            )

        self.assertFalse(audit["passed"])
        self.assertEqual(audit["source_path_tokens_found"], ["checkpoint_logs"])

    def test_filename_is_a_path_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_log = self._source_log(Path(tmp))
            audit = scoring.path_redaction_audit(
                source_log,
                [{"role": "user", "content": "open training.log."}],
            )

        self.assertFalse(audit["passed"])
        self.assertEqual(audit["source_path_tokens_found"], ["training.log"])

    def test_path_component_already_in_source_content_is_not_a_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_log = self._source_log(Path(tmp))
            source_log.write_text(
                "checkpoint_logs appears in the workload output\n",
                encoding="utf-8",
            )
            audit = scoring.path_redaction_audit(
                source_log,
                [{"role": "user", "content": "observed token checkpoint_logs"}],
            )

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["source_path_tokens_found"], [])
        self.assertEqual(audit["source_content_overlap_tokens"], ["checkpoint_logs"])

    def test_short_parent_component_is_not_distinctive_enough_to_be_a_leak(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_log = self._source_log(Path(tmp))
            audit = scoring.path_redaction_audit(
                source_log,
                [{"role": "assistant", "content": "The token elk is incidental."}],
            )

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["source_path_tokens_found"], [])

    def test_provider_opaque_metadata_is_excluded_from_redaction_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_log = self._source_log(Path(tmp))
            audit = scoring.path_redaction_audit(
                source_log,
                [
                    {
                        "role": "assistant",
                        "content": "No source path is visible here.",
                        "provider_specific_fields": {
                            "thought_signature": "checkpoint_logs training.log"
                        },
                        "tool_call_id": "checkpoint_logs",
                    }
                ],
            )

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["source_path_tokens_found"], [])


if __name__ == "__main__":
    unittest.main()
