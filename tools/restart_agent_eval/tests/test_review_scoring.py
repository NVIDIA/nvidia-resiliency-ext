# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests from a layered product trace to a scored route review."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from _builders import retry_policy  # noqa: E402
from _mocks import process_result as _process_result  # noqa: E402
from restart_agent_eval import product_trace  # noqa: E402
from restart_agent_eval import review as review_log  # noqa: E402
from restart_agent_eval.profiles import RunTarget  # noqa: E402


def _layered_product_trace(source_log):
    result = {
        "decision": "RESTART",
        "decision_basis": "general_retry_available",
        "retry_policy": retry_policy(
            policy_version="retry_budget.v1",
            matching_prior_failures=0,
        ),
        "primary_failure": None,
        "result_provenance": {},
    }
    trace = {
        "schema_version": product_trace.SINGLE_TRACE_SCHEMA,
        "request": {"log_path": str(source_log)},
        "analysis_result": result,
        "analyzer_trace": {
            "decision_evidence": {
                "schema_version": "restart_agent_decision_evidence.v1",
                "deterministic_primary_candidate": {
                    "fine_class": "observed_exception",
                    "line": 1,
                },
            },
            "layers": {
                "L1": {
                    "output_status": "usable",
                    "output_usable": True,
                    "output_errors": [],
                }
            },
            "l1": {
                "enabled": True,
                "success": True,
                "parsed_evidence": {
                    "primary_failure": {
                        "fine_class": "checkpoint_decode_error",
                        "line": 2,
                    }
                },
                "model_calls": [],
                "tool_calls": [],
                "interaction_transcript": [],
            },
            "l2_grounded_semantics": {
                "primary_failure": {
                    "fine_class": "checkpoint_decode_error",
                    "line": 1,
                }
            },
            "current_failure_facts": {
                "source": "l2_grounded",
                "root_fingerprint": "observed:checkpoint:decode_error",
                "history_identity_ready": True,
            },
            "l2_audit": {
                "used": True,
                "audit_status": "resolved",
                "citation_audits": [
                    {
                        "original_line": 1,
                        "resolved_line": 1,
                        "status": "rendered_exact",
                    }
                ],
            },
            "l3_history": {"available": False},
            "l4_policy": {
                "retry_policy": retry_policy(
                    policy_version="retry_budget.v1",
                    matching_prior_failures=0,
                )
            },
        },
        "l0_bundle": {
            "deterministic_primary_candidate": {
                "fine_class": "observed_exception",
                "line": 1,
            },
            "failure_episodes": [
                {
                    "start_line": 1,
                    "end_line": 2,
                    "first_exception_line": 1,
                    "terminal_exception_line": 2,
                }
            ],
        },
    }
    return result, trace


class ReviewScoringIntegrationTest(unittest.TestCase):
    def test_write_review_summary_accepts_layered_product_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_log = root / "training.log"
            source_log.write_text("training\n", encoding="utf-8")
            paths = {
                "result_json": root / "result.json",
                "trace_json": root / "trace.json",
                "review_json": root / "review.json",
                "review_md": root / "review.md",
            }
            result, trace = _layered_product_trace(source_log)
            paths["result_json"].write_text(json.dumps(result), encoding="utf-8")
            paths["trace_json"].write_text(json.dumps(trace), encoding="utf-8")

            summary = review_log.write_review_summary(
                target=RunTarget(name="test"),
                completed=_process_result(),
                paths=paths,
                source_log=source_log,
            )

        self.assertEqual(summary["schema_version"], "restart_agent_review.v1")
        self.assertEqual(
            summary["l1_model_output"],
            trace["analyzer_trace"]["l1"]["parsed_evidence"],
        )
        self.assertEqual(summary["l1_kpis"]["output_status"], "usable")
        self.assertEqual(summary["tool_efficiency"]["calls"], 0)
        self.assertEqual(summary["l2_kpis"]["rendered_exact_citation_count"], 1)
        self.assertEqual(
            summary["primary_selection_by_stage"],
            {
                "l0_deterministic": {
                    "fine_class": "observed_exception",
                    "line": 1,
                    "policy_class": None,
                    "fault_outcome": None,
                    "causal_role": None,
                    "root_fingerprint": None,
                    "root_fingerprint_source": None,
                },
                "l1_semantic": {
                    "fine_class": "checkpoint_decode_error",
                    "line": 2,
                    "policy_class": None,
                    "fault_outcome": None,
                    "causal_role": None,
                    "root_fingerprint": None,
                    "root_fingerprint_source": None,
                },
                "l2_grounded": {
                    "fine_class": "checkpoint_decode_error",
                    "line": 1,
                    "policy_class": None,
                    "fault_outcome": None,
                    "causal_role": None,
                    "root_fingerprint": None,
                    "root_fingerprint_source": None,
                },
                "l1_relation_to_l0": "same_failure_episode",
                "l2_relation_to_l0": "same_line",
            },
        )
