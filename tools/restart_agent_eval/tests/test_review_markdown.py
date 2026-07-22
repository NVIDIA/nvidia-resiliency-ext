# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from _panel_fixtures import rich_route_summary  # noqa: E402
from restart_agent_eval import review_markdown  # noqa: E402


class ReviewMarkdownTest(unittest.TestCase):
    def test_review_markdown_publishes_core_route_outcome(self) -> None:
        summary = {
            "run_label": "model.fast",
            "target": "fast",
            "model": "model/fast",
            "exit_code": 0,
            "l1_response_parsed": True,
            "l2_audit_status": "ok",
            "decision": "RESTART",
            "decision_basis": "general_retry_available",
            "primary_failure": {"fine_class": "checkpoint_read", "line": 12},
            "timing": {"total_wall_clock_s": 1.0},
            "token_usage": {"total_tokens": 100},
            "path_redaction_audit": {"passed": True},
            "model_calls": 1,
            "tool_calls": 0,
            "tool_names": [],
            "errors": [],
            "l1_model_output": {
                "schema_version": "restart_agent_evidence.v1",
                "primary_failure": {"fine_class": "checkpoint_read", "line": 12},
                "root_cause_assessment": {"status": "supported_but_unconfirmed"},
                "model_recovery_assessment": {
                    "failure_domain": {
                        "value": "unknown",
                        "status": "unknown",
                        "confidence": 50,
                    },
                    "retry_outlook_without_workload_change": {
                        "value": "unknown",
                        "status": "unknown",
                        "confidence": 50,
                    },
                    "rationale": "The current log is inconclusive.",
                },
                "related_failures": [],
                "evidence": [{"line": 12, "quote": "read failed", "supports": "primary"}],
                "justification": "The checkpoint read failed.",
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "route.review.md"

            review_markdown.write_review_markdown(path, summary)
            rendered = path.read_text(encoding="utf-8")

        self.assertIn("# Restart Agent Review: model.fast", rendered)
        self.assertIn("- decision: `RESTART`", rendered)
        self.assertIn("- primary: `checkpoint_read` line `12`", rendered)
        self.assertIn("## Artifact Guide", rendered)
        self.assertIn("## L1 Model Output", rendered)
        self.assertIn('"justification": "The checkpoint read failed."', rendered)

    def test_review_markdown_renders_stage_kpis_gold_and_diagnostics(self) -> None:
        summary = rich_route_summary()
        summary.update(
            {
                "run_label": "model.rich",
                "exit_code": 7,
                "errors": ["provider timeout"],
                "timing": {"total_wall_clock_s": 4.0, "l1_wall_clock_s": 2.0},
                "token_usage": {"total_tokens": 1234},
                "model_calls": 1,
                "tool_calls": 0,
                "tool_names": [],
                "effective_tool_profile": {"profile_id": "no-tools", "tools_enabled": False},
                "semantic_safety": {"semantic_safety": "ok"},
                "model_call_summary": {"calls": 1, "successful_calls": 1},
                "tool_efficiency": {"calls": 0},
                "gold_score": {"case_id": "case-a", "l1": {}, "l2": {}, "l4": {}},
                "path_redaction_audit": {
                    "passed": False,
                    "source_path_tokens_found": ["secret.log"],
                    "source_content_overlap_tokens": ["private-token"],
                },
            }
        )
        summary["primary_failure"]["failure_identity"] = {
            "policy_active": False,
            "family": {"label": "checkpoint_read", "fingerprint": "family-a"},
            "concrete": {"label": "decode_error", "fingerprint": "concrete-a"},
            "client_concrete": {"label": "utf8", "fingerprint": "client-a"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "route.review.md"
            review_markdown.write_review_markdown(path, summary)
            rendered = path.read_text(encoding="utf-8")

        for section in (
            "## Gold Comparison",
            "## Experimental Failure Identity",
            "## Model Route Qualification",
            "## L2 KPI Checklist",
            "## L3 KPI Checklist",
            "## L4 KPI Checklist",
            "## L0A Operations",
            "## Model Selection Signals",
            "## Semantic Safety",
            "## Model-Call Reliability",
            "## Tool-Use Efficiency",
        ):
            with self.subTest(section=section):
                self.assertIn(section, rendered)
        self.assertIn("path_redaction_tokens_found", rendered)
        self.assertIn("provider timeout", rendered)


if __name__ == "__main__":
    unittest.main()
