# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval.product_trace import (  # noqa: E402
    COLLECT_ALL_TRACE_SCHEMA,
    SINGLE_TRACE_SCHEMA,
    ProductTrace,
    decision_candidate_result,
)


class ProductTraceTest(unittest.TestCase):
    def test_parses_supported_single_model_trace(self) -> None:
        trace = ProductTrace.from_payload(
            {
                "schema_version": SINGLE_TRACE_SCHEMA,
                "request": {"log_path": "/tmp/job.log"},
                "analysis_result": {"decision": "RESTART"},
                "analyzer_trace": {"layers": {}},
                "l0_bundle": None,
                "collect_all_context": {
                    "route_id": "claude",
                    "execution_status": "deadline_exceeded",
                },
            }
        )

        self.assertFalse(trace.is_collect_all)
        self.assertEqual(trace.analysis_result, {"decision": "RESTART"})
        self.assertIsNone(trace.l0_bundle)
        self.assertEqual(
            trace.collect_all_context,
            {"route_id": "claude", "execution_status": "deadline_exceeded"},
        )

    def test_parses_supported_collect_all_trace(self) -> None:
        trace = ProductTrace.from_payload(
            {
                "schema_version": COLLECT_ALL_TRACE_SCHEMA,
                "request": {"log_path": "/tmp/job.log"},
                "collect_all_result": {"model_results": []},
                "analyzer_trace": {"shared": {}},
                "l0_bundle": {"schema_version": "restart_agent_l0_bundle.v1"},
            }
        )

        self.assertTrue(trace.is_collect_all)
        self.assertEqual(trace.collect_all_result, {"model_results": []})

    def test_rejects_unsupported_trace_schema(self) -> None:
        with self.assertRaises(ValueError):
            ProductTrace.from_payload(
                {
                    "schema_version": "restart_agent_cli_trace.v99",
                    "request": {},
                    "analysis_result": {},
                    "analyzer_trace": {},
                }
            )

    def test_rejects_missing_schema_specific_result(self) -> None:
        with self.assertRaises(ValueError):
            ProductTrace.from_payload(
                {
                    "schema_version": SINGLE_TRACE_SCHEMA,
                    "request": {},
                    "analyzer_trace": {},
                    "l0_bundle": {},
                }
            )

    def test_rejects_non_object_and_malformed_required_fields(self) -> None:
        invalid_payloads = (
            [],
            {
                "schema_version": SINGLE_TRACE_SCHEMA,
                "request": [],
                "analyzer_trace": {},
                "analysis_result": {},
            },
            {
                "schema_version": SINGLE_TRACE_SCHEMA,
                "request": {},
                "analyzer_trace": [],
                "analysis_result": {},
            },
            {
                "schema_version": SINGLE_TRACE_SCHEMA,
                "request": {},
                "analyzer_trace": {},
                "analysis_result": [],
            },
            {
                "schema_version": COLLECT_ALL_TRACE_SCHEMA,
                "request": {},
                "analyzer_trace": {},
            },
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises((TypeError, ValueError)):
                    ProductTrace.from_payload(payload)

    def test_read_loads_json_and_propagates_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": SINGLE_TRACE_SCHEMA,
                        "request": {},
                        "analyzer_trace": {},
                        "analysis_result": {"decision": "RESTART"},
                    }
                ),
                encoding="utf-8",
            )
            actual = ProductTrace.read(path)

            self.assertEqual(actual.analysis_result["decision"], "RESTART")

            path.write_text("{", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                ProductTrace.read(path)

    def test_decision_candidate_requires_mapping_result_envelope(self) -> None:
        missing = decision_candidate_result(None)
        malformed = decision_candidate_result({"result": []})
        valid = decision_candidate_result({"result": {"decision": "STOP"}})

        self.assertEqual(missing, {})
        self.assertEqual(malformed, {})
        self.assertEqual(
            valid,
            {"decision": "STOP"},
        )


if __name__ == "__main__":
    unittest.main()
