# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint reliability, context-budget, and model-call accounting."""

from __future__ import annotations

import unittest

from _bootstrap import configure_test_imports

configure_test_imports()

from _assertions import assert_mapping_fields  # noqa: E402
from restart_agent_eval import scoring  # noqa: E402


class EndpointScoringTest(unittest.TestCase):
    def test_l1_execution_status_is_degraded_after_recovered_provider_failure(
        self,
    ) -> None:
        status, issues = scoring.l1_execution_status(
            l1_layer={"output_usable": True, "output_status": "usable"},
            model_call_summary={
                "failed_calls": 1,
                "retried_calls": 1,
                "timeout_calls": 1,
                "provider_error_count": 1,
            },
        )

        self.assertEqual(status, "degraded")
        self.assertEqual(
            issues,
            [
                "model_call_failed",
                "retry_used",
                "provider_timeout",
                "provider_error",
            ],
        )

    def test_route_deadline_is_not_reported_as_endpoint_failure(self) -> None:
        assessment = {
            "execution_status": "failed",
            "result_quality": "unusable",
            "reason_codes": ["analysis_deadline_exceeded"],
            "unusable_reason": "analysis_deadline_exceeded",
        }
        summary = scoring.model_call_summary(
            [
                {
                    "model_turn": 1,
                    "attempt": 1,
                    "success": False,
                    "timeout": True,
                    "timeout_kind": "read",
                    "error": "analysis deadline exceeded during HTTP request",
                    "error_type": "analysis_deadline_exceeded",
                }
            ]
        )
        status, issues = scoring.l1_execution_status(
            l1_layer={},
            model_call_summary=summary,
            route_execution_status="failed",
            route_l1_execution_assessment=assessment,
        )
        signals = scoring.model_selection_signals(
            model_call_summary=summary,
            tool_efficiency={},
            semantic_safety={},
            route_execution_status="failed",
            route_l1_execution_assessment=assessment,
        )

        self.assertEqual(status, "failed")
        self.assertEqual(issues, ["analysis_deadline_exceeded"])
        self.assertEqual(summary["deadline_exceeded_calls"], 1)
        self.assertEqual(summary["timeout_calls"], 0)
        self.assertEqual(summary["timeout_kind_counts"], {"read": 1})
        self.assertEqual(summary["endpoint_failed_calls"], 0)
        self.assertEqual(signals["endpoint_reliability"], "not_observed")

    def test_model_call_summary_reports_context_budget_adjustment(self) -> None:
        summary = scoring.model_call_summary(
            [
                {
                    "success": True,
                    "latency_s": 2.5,
                    "finish_reason": "stop",
                    "context_budget": {
                        "context_window_tokens": 200000,
                        "estimated_input_tokens": 136001,
                        "configured_max_output_tokens": 64000,
                        "effective_max_output_tokens": 59803,
                        "adjusted": True,
                    },
                }
            ]
        )

        assert_mapping_fields(
            self,
            summary,
            {
                "context_budget_adjusted_calls": 1,
                "context_window_tokens": 200000,
                "max_estimated_input_tokens": 136001,
                "configured_max_output_tokens": 64000,
                "minimum_effective_max_output_tokens": 59803,
            },
        )

    def test_model_call_summary_aggregates_optional_provider_timing(self) -> None:
        summary = scoring.model_call_summary(
            [
                {
                    "success": True,
                    "provider_reported_timing": {
                        "source": "response_headers",
                        "downstream_llm_api_ms": 150.0,
                        "proxy_pre_processing_ms": 2.0,
                    },
                },
                {
                    "success": True,
                    "provider_reported_timing": {
                        "source": "response_headers",
                        "downstream_llm_api_ms": 250.0,
                        "proxy_post_processing_ms": 1.5,
                    },
                },
            ]
        )
        empty_summary = scoring.model_call_summary([])

        self.assertEqual(
            summary["provider_reported_timing"],
            {
                "source": "response_headers",
                "reported_call_count": 2,
                "components_ms_total": {
                    "downstream_llm_api_ms": 400.0,
                    "proxy_pre_processing_ms": 2.0,
                    "proxy_post_processing_ms": 1.5,
                },
            },
        )
        self.assertNotIn("provider_reported_timing", empty_summary)

    def test_model_call_summary_separates_turns_retries_and_endpoint_attempts(
        self,
    ) -> None:
        summary = scoring.model_call_summary(
            [
                {
                    "model_turn": 1,
                    "attempt": 1,
                    "success": False,
                    "timeout": True,
                    "retry_scheduled": True,
                    "error": "read timed out",
                },
                {
                    "model_turn": 1,
                    "attempt": 2,
                    "success": True,
                    "finish_reason": "tool_calls",
                },
                {"model_turn": 2, "attempt": 1, "success": True},
                {"model_turn": 3, "attempt": 1, "success": True},
            ]
        )

        assert_mapping_fields(
            self,
            summary,
            {
                "calls": 4,
                "model_turns": 3,
                "extra_model_turns_after_initial": 2,
                "failed_calls": 1,
                "retried_calls": 1,
                "timeout_calls": 1,
                "http_error_calls": 0,
            },
        )

    def test_endpoint_signals_do_not_double_count_timeout_failures(self) -> None:
        signals = scoring.model_selection_signals(
            model_call_summary={
                "model_turns": 1,
                "extra_model_turns_after_initial": 0,
                "failed_calls": 2,
                "retried_calls": 1,
                "timeout_calls": 2,
                "provider_error_count": 2,
                "http_error_calls": 1,
                "failed_latency_s_total": 125.0,
            },
            tool_efficiency={},
            semantic_safety={},
        )

        self.assertEqual(signals["endpoint_reliability"], "endpoint_issue")
        self.assertEqual(signals["failed_endpoint_attempts"], 2)
        self.assertEqual(signals["timeout_model_calls"], 2)
        self.assertEqual(signals["http_error_calls"], 1)
        self.assertNotIn("endpoint_issue_count", signals)

    def test_context_window_rejection_is_client_budget_not_endpoint_issue(self) -> None:
        for http_status, timeout in ((200, False), (504, True)):
            with self.subTest(http_status=http_status):
                summary = scoring.model_call_summary(
                    [
                        {
                            "model_turn": 5,
                            "attempt": 1,
                            "success": False,
                            "http_status": http_status,
                            "error": "maximum context length exceeded",
                            "error_type": "context_window_exceeded",
                            "timeout": timeout,
                            "timeout_kind": "read" if timeout else None,
                        }
                    ]
                )
                signals = scoring.model_selection_signals(
                    model_call_summary=summary,
                    tool_efficiency={},
                    semantic_safety={},
                )

                self.assertEqual(summary["context_window_error_calls"], 1)
                self.assertEqual(summary["timeout_calls"], 0)
                self.assertEqual(summary["timeout_kind_counts"], {})
                self.assertEqual(summary["endpoint_failed_calls"], 0)
                self.assertEqual(summary["provider_error_count"], 0)
                self.assertEqual(signals["client_request_health"], "context_window_exceeded")
                self.assertEqual(signals["endpoint_reliability"], "ok")

    def test_local_context_budget_failure_is_not_an_endpoint_issue(self) -> None:
        summary = scoring.model_call_summary(
            [
                {
                    "model_turn": 1,
                    "attempt": 1,
                    "success": False,
                    "error": "request exceeds configured context budget",
                    "error_type": "context_budget_exceeded",
                }
            ]
        )
        signals = scoring.model_selection_signals(
            model_call_summary=summary,
            tool_efficiency={},
            semantic_safety={},
        )

        self.assertEqual(summary["context_budget_error_calls"], 1)
        self.assertEqual(summary["endpoint_failed_calls"], 0)
        self.assertEqual(summary["provider_error_count"], 0)
        self.assertEqual(signals["client_request_health"], "context_budget_exceeded")
        self.assertEqual(signals["endpoint_reliability"], "ok")
