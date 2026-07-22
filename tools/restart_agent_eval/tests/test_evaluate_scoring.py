# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-case and corpus evaluation scoring tests across restart-agent stages."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from _builders import recovery_assessment, retry_policy  # noqa: E402
from _evaluation_fixtures import write_evaluation_case  # noqa: E402
from restart_agent_eval import evaluate as eval_harness  # noqa: E402
from restart_agent_eval.corpus import Case  # noqa: E402


def _case(
    *,
    label: dict | None = None,
    recovery_expectation: dict | None = None,
    retry_policy_expectation: dict | None = None,
) -> Case:
    return Case(
        case_id="case-a",
        label_path=Path("gold.json"),
        log_path=Path("case.log"),
        recovery_expectation=recovery_expectation or {},
        retry_policy_expectation=retry_policy_expectation or {},
        accepted_decisions=("RESTART",),
        label_version=1,
        label=label or {},
    )


class EvaluationScoringTest(unittest.TestCase):
    def test_empty_aggregate_has_no_rates(self) -> None:
        aggregate = eval_harness.aggregate_results("run-a", "gpt", [])

        self.assertEqual(aggregate["cases"], 0)
        self.assertEqual(aggregate["scored_cases"], 0)
        self.assertIsNone(aggregate["decision_accuracy"])
        self.assertIsNone(aggregate["l1_action_improvement_rate"])

    def test_score_checks_recovery_rule_decision_evidence_and_primary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_root, gold_root = write_evaluation_case(root)
            case = eval_harness.discover_cases(log_root, gold_root)[0]
            verdict = {
                "schema_version": "restart_agent_response.v1",
                "decision": "RESTART",
                "model_recovery_assessment": recovery_assessment(
                    failure_domain="infrastructure",
                    retry_outlook="may_recover",
                ),
                "retry_policy": retry_policy(
                    rule="bounded_retry",
                    allowed_retries=1,
                ),
                "primary_failure": {"line": 12},
                "evidence": [{"line": 12, "quote": "root"}],
            }
            result = eval_harness.score_case(
                run_id="run-a", target="deterministic", case=case, verdict=verdict
            )

        self.assertTrue(result.l1_recovery_correct)
        self.assertTrue(result.l4_retry_rule_correct)
        self.assertTrue(result.l4_allowed_retries_correct)
        self.assertTrue(result.l4_exhaustion_correct)
        self.assertTrue(result.evidence_line_hit)
        self.assertTrue(result.primary_anchor_hit)

    def test_should_score_retry_policy_dimensions_independently(self) -> None:
        result = eval_harness.score_case(
            run_id="run-a",
            target="deterministic",
            case=_case(
                retry_policy_expectation={
                    "accepted_rules": ["bounded_retry"],
                    "allowed_retries": 1,
                    "retry_budget_exhausted": False,
                }
            ),
            verdict={
                "decision": "RESTART",
                "retry_policy": {
                    "rule": "general_retry",
                    "allowed_retries": 1,
                    "retry_budget_exhausted": True,
                },
            },
        )
        self.assertFalse(result.l4_retry_rule_correct)
        self.assertTrue(result.l4_allowed_retries_correct)
        self.assertFalse(result.l4_exhaustion_correct)

    def test_corpus_score_separates_fallback_from_enriched_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_root, gold_root = write_evaluation_case(root)
            case = eval_harness.discover_cases(log_root, gold_root)[0]
            fallback = {
                "decision": "STOP",
                "retry_policy": retry_policy(
                    rule="workload_unrecoverable",
                    allowed_retries=0,
                    retry_budget_exhausted=True,
                ),
            }
            enriched = {
                "decision": "RESTART",
                "retry_policy": retry_policy(
                    rule="bounded_retry",
                    allowed_retries=1,
                ),
                "result_provenance": {"candidate_kind": "l1_enriched"},
            }
            trace_path = root / "route.trace.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "schema_version": "restart_agent_cli_trace.v1",
                        "request": {},
                        "analyzer_trace": {
                            "decision_candidates": {
                                "deterministic_fallback": {"result": fallback},
                                "l1_enriched": {"result": enriched},
                            }
                        },
                        "l0_bundle": {},
                        "analysis_result": enriched,
                    }
                ),
                encoding="utf-8",
            )

            result = eval_harness.score_case(
                run_id="run-a",
                target="gpt",
                case=case,
                verdict=enriched,
                trace_path=trace_path,
            )
            aggregate = eval_harness.aggregate_results("run-a", "gpt", [result])

        self.assertFalse(result.fallback_decision_correct)
        self.assertTrue(result.enriched_decision_correct)
        self.assertEqual(result.l1_action_effect, "improved")
        self.assertEqual(result.l1_policy_action_effect, "improved")
        self.assertEqual(aggregate["fallback_decision_accuracy"], 0.0)
        self.assertEqual(aggregate["enriched_decision_accuracy"], 1.0)
        self.assertEqual(aggregate["l1_action_improvement_rate"], 1.0)
        self.assertEqual(aggregate["l1_action_regression_rate"], 0.0)

    def test_should_score_l0_setup_expectation_from_product_trace(self) -> None:
        label = {
            "l0_expectation": {
                "required_setup_marker_types": ["checkpoint_load", "cuda_graph_build"],
                "minimum_setup_marker_count": 2,
                "required_coverage": {"setup_progress": "found"},
            }
        }
        bundle = {
            "progress": {
                "setup_markers": [
                    {"marker_type": "checkpoint_load"},
                    {"marker_type": "cuda_graph_build"},
                ]
            },
            "evidence_coverage": {"setup_progress": "found"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            verdict = {"decision": "RESTART"}
            trace_path.write_text(
                json.dumps(
                    {
                        "schema_version": "restart_agent_cli_trace.v1",
                        "request": {},
                        "analyzer_trace": {},
                        "l0_bundle": bundle,
                        "analysis_result": verdict,
                    }
                ),
                encoding="utf-8",
            )
            result = eval_harness.score_case(
                run_id="run-a",
                target="deterministic",
                case=_case(label=label),
                verdict=verdict,
                trace_path=trace_path,
            )

        self.assertTrue(result.l0a_quality_correct)

    def test_should_score_recovery_fields_independently(self) -> None:
        result = eval_harness.score_case(
            run_id="run-a",
            target="gpt",
            case=_case(
                recovery_expectation={
                    "failure_domain": ["workload"],
                    "failure_domain_status": ["supported_but_unconfirmed"],
                    "retry_outlook_without_workload_change": ["may_recover"],
                    "retry_outlook_status": ["supported_but_unconfirmed"],
                }
            ),
            verdict={
                "decision": "RESTART",
                "model_recovery_assessment": recovery_assessment(
                    failure_domain="workload",
                    retry_outlook="may_recover",
                    retry_outlook_status="established_by_current_log",
                ),
            },
        )
        self.assertFalse(result.l1_recovery_correct)
        self.assertTrue(result.l1_recovery_fields["failure_domain"])
        self.assertFalse(result.l1_recovery_fields["retry_outlook_status"])


if __name__ == "__main__":
    unittest.main()
