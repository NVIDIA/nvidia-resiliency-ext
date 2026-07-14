# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Malformed, empty, and boundary inputs for public scoring contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval import scoring  # noqa: E402


class GoldLabelBoundaryTest(unittest.TestCase):
    def test_reader_rejects_malformed_nonobject_and_wrong_schema_labels(self) -> None:
        values = ("{", "[]", '{"schema_version":"wrong"}')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gold.json"
            for value in values:
                with self.subTest(value=value):
                    path.write_text(value, encoding="utf-8")
                    with self.assertRaises(SystemExit):
                        scoring.read_gold_label(path)

    def test_reader_validates_source_digest_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.log"
            label = root / "gold.json"
            source.write_text("different", encoding="utf-8")
            label.write_text(
                json.dumps(
                    {
                        "schema_version": "restart_agent_eval.v1",
                        "case_id": "case-a",
                        "label_version": 1,
                        "review_status": "human_approved",
                        "source_sha256": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit):
                scoring.read_gold_label(label, source_log=source)


class StageKpiBoundaryTest(unittest.TestCase):
    def test_empty_l1_input_is_unusable(self) -> None:
        actual = scoring.l1_execution_status(l1_layer={}, model_call_summary={})

        self.assertEqual(actual, ("failed", ["unusable"]))

    def test_empty_l2_input_has_no_root_fingerprint(self) -> None:
        actual = scoring.l2_kpis(l2_audit={}, l0_primary={}, timing={})

        self.assertFalse(actual["root_fingerprint_available"])

    def test_empty_l3_input_has_no_history(self) -> None:
        actual = scoring.l3_kpis(analyzer_trace={}, timing={})

        self.assertIsNone(actual["history_available"])

    def test_empty_l4_input_uses_terminal_latency_mode(self) -> None:
        actual = scoring.l4_kpis(analysis={}, analyzer_trace={}, timing={})

        self.assertEqual(actual["latency_mode"], "terminal_request_to_result")

    def test_empty_l0_input_has_no_lines_or_failure_episodes(self) -> None:
        actual = scoring.l0_bundle_kpis(
            analysis={},
            l0_bundle={},
            l0_model_view={},
            timing={},
            tool_efficiency={},
        )

        self.assertIsNone(actual["line_count"])
        self.assertEqual(actual["failure_episode_count"], 0)

    def test_product_execution_status_takes_precedence_over_inferred_status(self) -> None:
        status = scoring.l1_execution_status(
            l1_layer={
                "execution_status": "cancelled",
                "execution_issues": ["route_cancelled"],
                "output_usable": True,
            },
            model_call_summary={"failed_calls": 1},
        )

        self.assertEqual(status, ("cancelled", ["route_cancelled"]))

    def test_malformed_product_execution_issues_are_ignored(self) -> None:
        status = scoring.l1_execution_status(
            l1_layer={"execution_status": "cancelled", "execution_issues": "invalid"},
            model_call_summary={},
        )

        self.assertEqual(status, ("cancelled", []))

    def test_l2_ignores_malformed_optional_collection_members(self) -> None:
        actual = scoring.l2_kpis(
            l2_audit={
                "field_findings": "not-a-map",
                "citation_audits": [None, {"status": "exact"}],
                "findings": ["not-a-map", {"severity": "advisory"}],
            },
            l0_primary={},
            timing={},
        )

        self.assertEqual(actual["finding_count"], 0)
        self.assertEqual(actual["citation_count"], 1)
        self.assertEqual(actual["finding_severity_counts"], {"advisory": 1})

    def test_l3_ignores_malformed_optional_mappings(self) -> None:
        actual = scoring.l3_kpis(
            analyzer_trace={
                "current_failure_facts": "not-a-map",
                "l3_history": "not-a-map",
            },
            timing={},
        )

        self.assertIsNone(actual["current_root_fingerprint"])
        self.assertIsNone(actual["history_available"])

    def test_l4_ignores_malformed_optional_mappings(self) -> None:
        actual = scoring.l4_kpis(
            analysis={"retry_policy": "not-a-map", "result_provenance": "not-a-map"},
            analyzer_trace={
                "l4_policy": "not-a-map",
                "latency_measurement": "not-a-map",
            },
            timing={},
        )

        self.assertIsNone(actual["rule"])


class ScoringUtilityBoundaryTest(unittest.TestCase):
    def test_distributed_incidents_ignore_nonobject_members(self) -> None:
        incidents = scoring.distributed_incident_summaries(
            {
                "distributed_failure_incidents": [
                    None,
                    "invalid",
                    {
                        "incident_id": "incident-a",
                        "incident_type": "collective_timeout",
                        "private_detail": "omitted",
                    },
                ]
            }
        )

        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents[0]["incident_id"], "incident-a")
        self.assertNotIn("private_detail", incidents[0])

    def test_path_effect_distinguishes_missing_improved_regressed_and_unchanged(self) -> None:
        scenarios = (
            (None, None, "not_available"),
            (None, {"metric": True}, "not_available"),
            ({}, {"metric": True}, "unscored"),
            ({"metric": False}, {"metric": True}, "improved"),
            ({"metric": True}, {"metric": False}, "regressed"),
            ({"metric": True}, {"metric": True}, "unchanged_correct"),
            ({"metric": False}, {"metric": False}, "unchanged_incorrect"),
        )
        for fallback, enriched, expected in scenarios:
            with self.subTest(expected=expected):
                actual = scoring.score_path_effect(fallback, enriched, "metric")

                self.assertEqual(actual, expected)

    def test_line_numbering_handles_empty_mixed_endings_and_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty.log"
            mixed = root / "mixed.log"
            empty.write_bytes(b"")
            mixed.write_bytes(b"one\r\ntwo\rthree\ninvalid:\xff")

            empty_summary = scoring.line_numbering_summary(empty)
            mixed_summary = scoring.line_numbering_summary(mixed)

        self.assertEqual(empty_summary["splitlines_count"], 0)
        self.assertEqual(mixed_summary["splitlines_count"], 4)
        self.assertEqual(mixed_summary["crlf_count"], 1)
        self.assertEqual(mixed_summary["cr_count"], 2)
        self.assertEqual(mixed_summary["lf_count"], 2)

    def test_model_call_summary_ignores_nonlist_and_nonobject_calls(self) -> None:
        invalid_collection = scoring.model_call_summary("invalid")
        mixed_collection = scoring.model_call_summary([None, "invalid", {"success": True}])

        self.assertEqual(invalid_collection["calls"], 0)
        self.assertEqual(mixed_collection["calls"], 1)
        self.assertEqual(mixed_collection["successful_calls"], 1)


if __name__ == "__main__":
    unittest.main()
