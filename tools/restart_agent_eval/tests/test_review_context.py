# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalized review context across deterministic, enriched, and malformed payloads."""

from __future__ import annotations

import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval.review_context import ReviewContext  # noqa: E402


def _trace(analysis, analyzer_trace=None, collect_all_context=None):
    return {
        "schema_version": "restart_agent_cli_trace.v1",
        "request": {},
        "analysis_result": analysis,
        "analyzer_trace": analyzer_trace or {},
        "l0_bundle": {},
        "collect_all_context": collect_all_context,
    }


class ReviewContextTest(unittest.TestCase):
    def test_normalizes_all_stage_payloads_and_candidate_results(self) -> None:
        deterministic = {"decision": "RESTART"}
        enriched = {"decision": "STOP"}
        analyzer_trace = {
            "l0_model_view": {"schema_version": "l0b.v1"},
            "decision_evidence": {
                "deterministic_primary_candidate": {"line": 10},
            },
            "layers": {"L1": {"output_status": "usable"}},
            "l1": {
                "semantic_payload": {"primary_failure": {"line": 11}},
                "model_calls": [{"success": True}],
                "tool_calls": [{"name": "overview"}],
                "transcript_events": [{"event_type": "model_request"}],
            },
            "l2_audit": {"audit_status": "clean"},
            "l2_grounding": {"primary_failure": {"line": 12}},
            "current_failure_facts": {"observed": True},
            "timing": {"l1_wall_clock_s": 1.0},
            "latency_measurement": {"mode": "terminal"},
            "token_usage": {"total_tokens": 10},
            "token_limit": {"context_window_tokens": 100},
            "decision_candidates": {
                "deterministic": {"result": deterministic},
                "l1_enriched": {"result": enriched},
            },
        }
        analysis = {
            "decision": "STOP",
            "primary_failure": {"line": 12},
            "l1_assessment": {"primary_failure": {"line": 13}},
            "l2_grounding": {
                "grounded_primary_failure": {"line": 14},
                "grounding_status": "grounded",
            },
            "result_provenance": {"candidate_kind": "l1_enriched"},
        }

        context = ReviewContext.from_payloads(
            {"decision": "STOP"},
            _trace(
                analysis,
                analyzer_trace,
                {
                    "route_id": "gpt",
                    "execution_status": "completed",
                    "l1_execution_assessment": {
                        "execution_status": "completed",
                        "result_quality": "usable",
                    },
                },
            ),
        )

        self.assertEqual(context.route_execution_status, "completed")
        self.assertEqual(
            context.route_l1_execution_assessment,
            {"execution_status": "completed", "result_quality": "usable"},
        )
        self.assertEqual(context.l0_primary, {"line": 10})
        self.assertEqual(context.l1_primary, {"line": 13})
        self.assertEqual(context.l2_primary, {"line": 14})
        self.assertEqual(context.l2_grounding["grounding_status"], "grounded")
        self.assertEqual(context.deterministic_analysis, deterministic)
        self.assertEqual(context.enriched_analysis, enriched)
        self.assertEqual(context.interaction_transcript[0]["event_type"], "model_request")
        self.assertEqual(context.primary, {"line": 12})

    def test_selected_candidate_falls_back_to_analysis_when_envelope_is_absent(self) -> None:
        for candidate_kind, expected_field in (
            ("deterministic", "deterministic_analysis"),
            ("l1_enriched", "enriched_analysis"),
        ):
            analysis = {
                "decision": "RESTART",
                "result_provenance": {"candidate_kind": candidate_kind},
            }
            context = ReviewContext.from_payloads({}, _trace(analysis))

            with self.subTest(candidate_kind=candidate_kind):
                self.assertEqual(getattr(context, expected_field), analysis)

    def test_preserves_observation_only_outputs_by_stage(self) -> None:
        l1_observation = {"id": "o1", "line": 20, "causal_role": "cascade"}
        grounded_observation = {
            "failure_class": "tcpstore_connection_loss",
            "line": 20,
            "causal_role": "cascade",
            "observation_fingerprint": "transport:tcpstore_connection_loss",
        }
        analysis = {
            "decision": "RESTART",
            "primary_failure": None,
            "observed_failures": [grounded_observation],
            "selected_observed_failure": grounded_observation,
            "l1_assessment": {
                "primary_failure": None,
                "observed_failures": [l1_observation],
                "selected_observed_failure_id": "o1",
            },
            "l2_grounding": {
                "grounded_observed_failures": [grounded_observation],
                "grounded_selected_observation": grounded_observation,
            },
        }
        analyzer_trace = {
            "decision_evidence": {"selected_observed_failure": grounded_observation},
        }

        context = ReviewContext.from_payloads({}, _trace(analysis, analyzer_trace))

        self.assertEqual(context.l0_selected_observation, grounded_observation)
        self.assertEqual(context.l1_observed_failures, [l1_observation])
        self.assertEqual(context.l1_selected_observation, l1_observation)
        self.assertEqual(context.l2_selected_observation, grounded_observation)
        self.assertEqual(context.selected_observation, grounded_observation)
        self.assertEqual(context.primary, {})

    def test_projects_independent_failure_tracks_history_and_l4_selection(self) -> None:
        primary = {"failure_class": "checkpoint_decode_error", "line": 10}
        observation = {"failure_class": "process_exit", "line": 20}
        primary_facts = {"identity_kind": "root", "root_fingerprint": "root-a"}
        observation_facts = {
            "identity_kind": "observation_only",
            "observation_fingerprint": "observation-a",
        }
        analysis = {
            "primary_failure": primary,
            "observed_failures": [observation],
            "selected_observed_failure": None,
            "l2_grounding": {
                "grounded_primary_failure": primary,
                "grounded_selected_observation": observation,
                "track_grounding": {
                    "primary": {"status": "grounded"},
                    "observation": {"status": "grounded"},
                },
                "enriched_failure_tracks": {
                    "primary": primary_facts,
                    "observation": observation_facts,
                },
            },
        }
        analyzer_trace = {
            "decision_evidence": {"deterministic_primary_candidate": {"line": 9}},
            "l3_history": {
                "job_progress": {"available": True},
                "deterministic": {"available": False, "availability_reason": "no_match"},
                "routes": {
                    "gpt": {
                        "primary": {"available": True},
                        "observation": {
                            "available": False,
                            "availability_reason": "no_prior_attempts",
                        },
                    }
                },
            },
            "l4_policy": {
                "path_selection": {
                    "path": "primary",
                    "route_id": "gpt",
                    "reason": "grounded_primary_available",
                }
            },
        }

        context = ReviewContext.from_payloads({}, _trace(analysis, analyzer_trace))

        self.assertEqual(context.failure_tracks["primary"]["facts"], primary_facts)
        self.assertEqual(context.failure_tracks["observation"]["facts"], observation_facts)
        self.assertEqual(context.failure_tracks["deterministic"]["failure"], {"line": 9})
        self.assertTrue(context.cycle_history["routes"]["gpt"]["primary"]["available"])
        self.assertEqual(context.l4_path_selection["path"], "primary")

    def test_derives_observation_only_current_facts_when_internal_record_is_absent(self) -> None:
        observation = {
            "failure_class": "process_terminated",
            "line": 44,
            "causal_role": "unresolved",
            "fault_outcome": "terminal",
            "observation_fingerprint": "observation-a",
            "observation_fingerprint_source": "l0_registry_observation",
        }
        analysis = {
            "primary_failure": None,
            "observed_failures": [observation],
            "selected_observed_failure": observation,
            "result_provenance": {
                "candidate_kind": "deterministic",
                "evidence_source": "l0_observation_only",
            },
        }

        context = ReviewContext.from_payloads(
            {},
            _trace(
                analysis,
                {"layers": {"L3": {"selected_failure_facts_source": "l0_deterministic"}}},
            ),
        )

        self.assertEqual(context.current_failure_facts["identity_kind"], "observation_only")
        self.assertEqual(context.current_failure_facts["observation_fingerprint"], "observation-a")
        self.assertTrue(context.current_failure_facts["history_identity_ready"])
        self.assertEqual(context.current_failure_facts["source"], "l0_deterministic")

    def test_non_mapping_result_and_optional_stage_values_become_empty(self) -> None:
        context = ReviewContext.from_payloads(
            ["not-an-object"],
            _trace({"decision": "RESTART"}, {"l1": {"model_calls": None}}),
        )

        self.assertEqual(context.result, {})
        self.assertEqual(context.model_calls, [])
        self.assertEqual(context.tool_calls, [])
        self.assertEqual(context.interaction_transcript, [])
        self.assertIsNone(context.route_execution_status)

    def test_read_uses_injected_artifact_store(self) -> None:
        class _Store:
            def __init__(self) -> None:
                self.paths = []

            def read_json(self, path):
                self.paths.append(path)
                if path.name == "result.json":
                    return {"decision": "RESTART"}
                return _trace({"decision": "RESTART"})

        store = _Store()
        context = ReviewContext.read(
            {"result_json": Path("result.json"), "trace_json": Path("trace.json")},
            artifact_store=store,
        )

        self.assertEqual(context.analysis["decision"], "RESTART")
        self.assertEqual(store.paths, [Path("result.json"), Path("trace.json")])


if __name__ == "__main__":
    unittest.main()
