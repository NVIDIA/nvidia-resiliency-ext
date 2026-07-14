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
    l1_assessment = {
        "primary_failure": {
            "failure_class": "checkpoint_decode_error",
            "line": 2,
        }
    }
    result = {
        "decision": "RESTART",
        "decision_basis": "general_retry_available",
        "retry_policy": retry_policy(
            policy_version="retry_budget.v1",
            matching_prior_attempts=0,
        ),
        "primary_failure": None,
        "result_provenance": {},
        "l1_assessment": l1_assessment,
        "l2_grounding": {
            "grounded_primary_failure": {
                "failure_class": "checkpoint_decode_error",
                "line": 1,
            },
            "track_grounding": {
                "primary": {"status": "grounded", "published": True},
                "observation": {"status": "unavailable", "published": False},
            },
            "enriched_failure_tracks": {
                "primary": {
                    "identity_kind": "root",
                    "root_fingerprint": "observed:checkpoint:decode_error",
                },
                "observation": None,
            },
        },
    }
    trace = {
        "schema_version": product_trace.SINGLE_TRACE_SCHEMA,
        "request": {"log_path": str(source_log)},
        "analysis_result": result,
        "analyzer_trace": {
            "decision_evidence": {
                "schema_version": "restart_agent_decision_evidence.v1",
                "deterministic_primary_candidate": {
                    "failure_class": "observed_exception",
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
                "semantic_payload": l1_assessment,
                "model_calls": [],
                "tool_calls": [],
                "interaction_transcript": [],
            },
            "l2_grounding": {
                "primary_failure": {
                    "failure_class": "checkpoint_decode_error",
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
            "l3_history": {
                "job_progress": {
                    "available": False,
                    "availability_reason": "no_prior_attempts",
                },
                "deterministic": {
                    "available": False,
                    "availability_reason": "no_prior_attempts",
                },
                "routes": {
                    "test": {
                        "primary": {
                            "available": True,
                            "availability_reason": "ready",
                            "matching_root_attempts": 2,
                        },
                        "observation": None,
                    }
                },
            },
            "l4_policy": {
                "path_selection": {
                    "path": "primary",
                    "route_id": "test",
                    "reason": "grounded_primary_available",
                },
                "retry_policy": retry_policy(
                    policy_version="retry_budget.v1",
                    matching_prior_attempts=0,
                ),
            },
        },
        "l0_bundle": {
            "deterministic_primary_candidate": {
                "failure_class": "observed_exception",
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
            persisted_summary = json.loads(paths["review_json"].read_text(encoding="utf-8"))
            review_markdown = paths["review_md"].read_text(encoding="utf-8")

        self.assertEqual(summary["schema_version"], "restart_agent_review.v1")
        self.assertEqual(persisted_summary, summary)
        for duplicated_field in (
            "l1_model_output",
            "root_cause_assessment",
            "model_recovery_assessment",
            "l2_grounding",
            "l2_audit",
            "retry_policy",
            "decision_evidence",
            "later_progress_after_fault_observations",
            "operation_artifact_comparisons",
            "distributed_failure_incidents",
        ):
            self.assertNotIn(duplicated_field, summary)
        self.assertIn("later_progress_after_fault_observations", summary["l0_kpis"])
        self.assertIn("operation_artifact_comparisons", summary["l0_kpis"])
        self.assertIn("distributed_failure_incidents", summary["l0_kpis"])
        self.assertIsNotNone(summary["decision_evidence_sha256"])
        self.assertIn("Complete L1 assessment, L2 grounding", review_markdown)
        self.assertNotIn('"schema_version": "restart_agent_evidence.v1"', review_markdown)
        self.assertEqual(summary["l1_kpis"]["output_status"], "usable")
        self.assertEqual(summary["failure_tracks"]["primary"]["status"], "grounded")
        self.assertEqual(
            summary["l3_history"]["routes"]["test"]["primary"]["availability_reason"],
            "ready",
        )
        self.assertTrue(summary["l3_kpis"]["history_available"])
        self.assertEqual(summary["l3_kpis"]["matching_root_attempts"], 2)
        self.assertEqual(summary["l4_path_selection"]["path"], "primary")
        self.assertEqual(summary["tool_efficiency"]["calls"], 0)
        self.assertEqual(summary["l2_kpis"]["rendered_exact_citation_count"], 1)
        self.assertEqual(
            summary["primary_selection_by_stage"],
            {
                "l0_deterministic": {
                    "failure_class": "observed_exception",
                    "line": 1,
                    "fault_outcome": None,
                    "causal_role": None,
                    "root_fingerprint": None,
                    "root_fingerprint_source": None,
                },
                "l1_semantic": {
                    "failure_class": "checkpoint_decode_error",
                    "line": 2,
                    "fault_outcome": None,
                    "causal_role": None,
                    "root_fingerprint": None,
                    "root_fingerprint_source": None,
                },
                "l2_grounded": {
                    "failure_class": "checkpoint_decode_error",
                    "line": 1,
                    "fault_outcome": None,
                    "causal_role": None,
                    "root_fingerprint": None,
                    "root_fingerprint_source": None,
                },
                "l1_relation_to_l0": "same_failure_episode",
                "l2_relation_to_l0": "same_line",
            },
        )
