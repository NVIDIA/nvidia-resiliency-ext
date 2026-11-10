# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the closed L1 execution-assessment contract."""

from copy import deepcopy

import pytest

from nvidia_resiliency_ext.attribution.restart_agent.l1 import (
    L1EvidenceResult,
    assess_execution,
    model_evidence_contract_advisories,
    model_evidence_contract_errors,
    output_health,
)
from nvidia_resiliency_ext.attribution.restart_agent.l1.response_contract import (
    L1_RESPONSE_CONTRACT,
)
from nvidia_resiliency_ext.attribution.restart_agent.single_run import _run_l1_until_deadline


def _valid_no_failure_evidence():
    return {
        "schema_version": "restart_agent_evidence.v1",
        "analysis_status": "no_failure_observed",
        "primary_failure": None,
        "observed_failures": [],
        "selected_observed_failure_id": None,
        "root_cause_assessment": {
            "summary": L1_RESPONSE_CONTRACT.no_failure_summary,
            "status": "unknown",
            "plausible_causes": [],
            "missing_evidence": [],
        },
        "model_recovery_assessment": {
            "failure_domain": {
                "value": "unknown",
                "status": "unknown",
                "confidence": 1,
            },
            "retry_outlook_without_workload_change": {
                "value": "unknown",
                "status": "unknown",
                "confidence": 1,
            },
            "rationale": L1_RESPONSE_CONTRACT.no_failure_rationale,
        },
        "related_failures": [],
        "evidence": [],
        "category_selection": {
            "category_id": 0,
            "category_confidence": 0,
            "category_rationale": "no failure observed",
        },
    }


def _primary_evidence_without_recovery_support():
    return {
        "schema_version": "restart_agent_evidence.v1",
        "analysis_status": "primary_identified",
        "primary_failure": {
            "line": 2,
            "causal_role": "initiating",
            "failure_identity": {
                "operation": "checkpoint_load",
                "mechanism": "collective_operation_timeout",
                "component": "ProcessGroupNCCL",
                "direct_failure_object_path": None,
                "affected_artifact_path": "/checkpoints/model",
            },
        },
        "observed_failures": [],
        "selected_observed_failure_id": None,
        "root_cause_assessment": {
            "summary": "Checkpoint loading ended in an NCCL ALLGATHER timeout.",
            "status": "established_by_current_log",
            "plausible_causes": ["one or more ranks did not complete the collective"],
            "missing_evidence": ["node and network telemetry"],
        },
        "model_recovery_assessment": {
            "failure_domain": {
                "value": "unknown",
                "status": "unknown",
                "confidence": 1,
            },
            "retry_outlook_without_workload_change": {
                "value": "may_recover",
                "status": "supported_but_unconfirmed",
                "confidence": 40,
            },
            "rationale": (
                "The current log does not distinguish workload from infrastructure, "
                "but a transient collective stall may recover on retry."
            ),
        },
        "related_failures": [],
        "evidence": [
            {
                "id": "primary-timeout",
                "line": 2,
                "quote": "Watchdog caught collective operation timeout: ALLGATHER",
                "supports": ["primary_failure", "root_cause_assessment"],
            }
        ],
        "category_selection": {
            "category_id": 0,
            "category_confidence": 0,
            "category_rationale": "not applicable to this fixture",
        },
    }


def test_valid_response_has_usable_closed_execution_assessment():
    result = L1EvidenceResult(
        semantic_payload=_valid_no_failure_evidence(),
        model="test-model",
        success=True,
    )

    assessment = assess_execution(configured=True, result=result).to_payload()

    assert assessment == {
        "execution_status": "completed",
        "result_quality": "usable",
        "parse_status": "valid",
        "usable": True,
        "degraded": False,
        "evidence_present": True,
        "final_evidence_reason": None,
        "reason_codes": [],
        "unusable_reason": None,
        "errors": [],
    }


def test_missing_recovery_support_does_not_invalidate_structured_l1_result():
    evidence = _primary_evidence_without_recovery_support()
    result = L1EvidenceResult(
        semantic_payload=evidence,
        model="test-model",
        success=True,
    )

    assessment = assess_execution(configured=True, result=result).to_payload()

    assert model_evidence_contract_errors(evidence) == []
    assert assessment["execution_status"] == "completed"
    assert assessment["result_quality"] == "usable"
    assert assessment["parse_status"] == "valid"
    assert assessment["usable"] is True


def test_plausible_cause_count_is_advisory_and_result_remains_usable():
    evidence = _primary_evidence_without_recovery_support()
    evidence["root_cause_assessment"]["plausible_causes"] = [
        "persistent permission mismatch",
        "stale cache lock",
        "storage authorization failure",
        "incorrect cache namespace",
    ]
    result = L1EvidenceResult(
        semantic_payload=evidence,
        model="test-model",
        success=True,
    )

    assessment = assess_execution(configured=True, result=result).to_payload()

    assert model_evidence_contract_errors(evidence) == []
    assert model_evidence_contract_advisories(evidence) == [
        {
            "code": "plausible_causes_exceeds_recommended_limit",
            "field": "root_cause_assessment.plausible_causes",
            "message": "plausible_causes contains 4 items; the recommended maximum is 3",
            "observed_count": 4,
            "recommended_maximum": 3,
            "observational_only": True,
        }
    ]
    assert assessment["result_quality"] == "usable"
    assert assessment["parse_status"] == "valid"


def test_missing_primary_claim_support_is_advisory():
    evidence = _primary_evidence_without_recovery_support()
    evidence["evidence"][0]["supports"] = ["primary_failure"]

    assert model_evidence_contract_errors(evidence) == []
    assert model_evidence_contract_advisories(evidence) == [
        {
            "code": "root_cause_assessment_support_missing",
            "field": "evidence[].supports",
            "message": "no citation declares support for root_cause_assessment",
            "missing_support_tag": "root_cause_assessment",
            "observational_only": True,
        }
    ]


def test_unknown_evidence_support_tag_is_advisory_and_result_remains_usable():
    evidence = _primary_evidence_without_recovery_support()
    evidence["evidence"].append(
        {
            "id": "downstream-store-failure",
            "line": 3,
            "quote": "TCPStore connection closed after the primary failure",
            "supports": ["related_failures"],
        }
    )
    result = L1EvidenceResult(
        semantic_payload=evidence,
        model="test-model",
        success=True,
    )

    assessment = assess_execution(configured=True, result=result).to_payload()

    assert model_evidence_contract_errors(evidence) == []
    assert model_evidence_contract_advisories(evidence) == [
        {
            "code": "evidence_support_tag_unknown",
            "field": "evidence[].supports",
            "message": ("unknown evidence support tags are ignored for claim-support accounting"),
            "item_indexes": [1],
            "unknown_tags": ["related_failures"],
            "observational_only": True,
        }
    ]
    assert assessment["execution_status"] == "completed"
    assert assessment["result_quality"] == "usable"
    assert assessment["parse_status"] == "valid"
    assert assessment["usable"] is True


def test_non_string_evidence_support_tag_remains_invalid():
    evidence = _primary_evidence_without_recovery_support()
    evidence["evidence"][0]["supports"] = ["primary_failure", 7]

    assert model_evidence_contract_errors(evidence) == [
        "evidence[0].supports items must be non-empty strings"
    ]


def test_output_efficiency_limits_are_advisory():
    evidence = _primary_evidence_without_recovery_support()
    evidence["root_cause_assessment"]["missing_evidence"] = [
        f"missing evidence {index}" for index in range(6)
    ]
    evidence["observed_failures"] = [
        {
            "id": f"observed-{index}",
            "line": index + 10,
            "causal_role": "cascade",
            "failure_identity": {
                "operation": "checkpoint_load",
                "mechanism": "collective_operation_timeout",
                "component": "ProcessGroupNCCL",
                "direct_failure_object_path": None,
                "affected_artifact_path": "/checkpoints/model",
            },
            "rationale": "Repeated failure surface.",
            "evidence_ids": ["primary-timeout"],
        }
        for index in range(4)
    ]
    evidence["related_failures"] = [
        {
            "line": index + 20,
            "causal_role": "cascade",
            "rationale": "Repeated downstream failure.",
        }
        for index in range(4)
    ]
    evidence["evidence"] = [
        {
            "id": f"evidence-{index}",
            "line": index + 1,
            "quote": f"failure evidence {index}",
            "supports": (
                ["primary_failure", "root_cause_assessment"]
                if index == 0
                else ["root_cause_assessment"]
            ),
        }
        for index in range(13)
    ]

    assert model_evidence_contract_errors(evidence) == []
    assert [item["code"] for item in model_evidence_contract_advisories(evidence)] == [
        "missing_evidence_exceeds_recommended_limit",
        "observed_failures_exceeds_recommended_limit",
        "related_failures_exceeds_recommended_limit",
        "evidence_exceeds_recommended_limit",
    ]


def test_redundancy_and_identifier_length_are_advisory():
    evidence = _primary_evidence_without_recovery_support()
    evidence["observed_failures"] = [
        {
            "id": "o" * 65,
            "line": 2,
            "causal_role": "cascade",
            "failure_identity": {
                "operation": "checkpoint_load",
                "mechanism": "collective_operation_timeout",
                "component": "ProcessGroupNCCL",
                "direct_failure_object_path": None,
                "affected_artifact_path": "/checkpoints/model",
            },
            "rationale": "Repeated failure surface.",
            "evidence_ids": ["primary-timeout", "primary-timeout"],
        }
    ]
    evidence["evidence"][0]["id"] = "e" * 65
    evidence["evidence"][0]["supports"] = [
        "primary_failure",
        "primary_failure",
        "root_cause_assessment",
    ]

    assert model_evidence_contract_errors(evidence) == []
    assert [item["code"] for item in model_evidence_contract_advisories(evidence)] == [
        "observed_failure_id_exceeds_recommended_length",
        "observed_failure_evidence_ids_contain_duplicates",
        "evidence_id_exceeds_recommended_length",
        "evidence_supports_contain_duplicates",
    ]


def test_duplicate_object_ids_and_ambiguous_selection_are_advisory():
    evidence = _primary_evidence_without_recovery_support()
    observation = {
        "id": "terminal-surface",
        "line": 3,
        "causal_role": "cascade",
        "failure_identity": {
            "operation": "distributed_coordination",
            "mechanism": "tcpstore_connection_loss",
            "component": "c10d_tcpstore",
            "direct_failure_object_path": None,
            "affected_artifact_path": None,
        },
        "rationale": "The store connection failed after the primary.",
        "evidence_ids": ["primary-timeout"],
    }
    evidence["observed_failures"] = [observation, deepcopy(observation)]
    evidence["selected_observed_failure_id"] = "terminal-surface"
    evidence["evidence"].append(deepcopy(evidence["evidence"][0]))

    result = L1EvidenceResult(semantic_payload=evidence, model="test-model", success=True)

    assert model_evidence_contract_errors(evidence) == []
    assert [item["code"] for item in model_evidence_contract_advisories(evidence)] == [
        "duplicate_observed_failure_id",
        "selected_observation_id_ambiguous",
        "duplicate_evidence_id",
    ]
    assert assess_execution(configured=True, result=result).usable is True


def test_unresolved_selected_observation_id_is_advisory():
    evidence = _primary_evidence_without_recovery_support()
    evidence["selected_observed_failure_id"] = "missing-observation"

    assert model_evidence_contract_errors(evidence) == []
    assert [item["code"] for item in model_evidence_contract_advisories(evidence)] == [
        "selected_observation_id_unresolved"
    ]


def test_retry_diagnostics_degrade_without_erasing_valid_evidence():
    evidence = _valid_no_failure_evidence()
    result = L1EvidenceResult(
        semantic_payload=evidence,
        model="test-model",
        success=True,
        model_calls=(
            {
                "success": False,
                "retry_scheduled": True,
                "http_status": 503,
                "error_type": "http_error",
            },
            {"success": True},
        ),
    )

    assessment = assess_execution(configured=True, result=result)

    assert assessment.to_payload()["result_quality"] == "degraded"
    assert assessment.to_payload()["reason_codes"] == [
        "model_call_failed",
        "retry_used",
        "provider_http_error",
    ]
    assert assessment.usable is True
    assert result.semantic_payload == evidence


def test_deadline_timeout_diagnostic_is_not_classified_as_provider_timeout():
    result = L1EvidenceResult(
        semantic_payload=None,
        model="test-model",
        success=False,
        errors=("analysis deadline exceeded",),
        model_calls=(
            {
                "success": False,
                "error_type": "analysis_deadline_exceeded",
                "timeout": True,
                "timeout_kind": "read",
                "retry_scheduled": False,
            },
        ),
        anomalies={
            "provider_error": True,
            "provider_timeout": True,
            "deadline_exceeded": True,
        },
    )

    assessment = assess_execution(configured=True, result=result).to_payload()

    assert assessment["unusable_reason"] == "analysis_deadline_exceeded"
    assert assessment["reason_codes"] == [
        "analysis_deadline_exceeded",
        "model_call_failed",
    ]


@pytest.mark.parametrize(
    "reason",
    ["context_budget_exceeded", "context_window_exceeded"],
)
def test_context_limit_failure_has_exact_unusable_reason(reason):
    result = L1EvidenceResult(
        semantic_payload=None,
        model="test-model",
        success=False,
        errors=(reason,),
        anomalies={reason: True, "provider_error": True},
    )

    assessment = assess_execution(configured=True, result=result).to_payload()

    assert assessment["result_quality"] == "unusable"
    assert assessment["unusable_reason"] == reason
    assert assessment["reason_codes"] == [reason]


def test_http_context_rejection_is_not_counted_as_endpoint_degradation():
    result = L1EvidenceResult(
        semantic_payload=None,
        model="test-model",
        success=False,
        errors=("context_window_exceeded",),
        model_calls=(
            {
                "success": False,
                "http_status": 504,
                "error_type": "context_window_exceeded",
                "timeout": False,
                "retry_scheduled": False,
            },
        ),
        anomalies={
            "provider_error": True,
            "context_window_exceeded": True,
        },
    )

    assessment = assess_execution(configured=True, result=result).to_payload()

    assert assessment["unusable_reason"] == "context_window_exceeded"
    assert assessment["reason_codes"] == [
        "context_window_exceeded",
        "model_call_failed",
    ]


def test_late_single_route_result_preserves_selected_final_evidence_reason():
    late_result = L1EvidenceResult(
        semantic_payload=None,
        model="test-model",
        success=False,
        anomalies={
            "provider_error": True,
            "final_evidence_turn": True,
            "final_evidence_reason": "contract_repair",
        },
    )

    class _Future:
        def result(self, timeout):
            assert timeout == 5.0
            return late_result, 6.0, 6.0

        def cancel(self):
            raise AssertionError("completed future must not be cancelled")

    class _Executor:
        def submit(self, *args):
            return _Future()

        def shutdown(self, *, wait, cancel_futures):
            assert wait is False
            assert cancel_futures is True

    class _Clock:
        def monotonic(self):
            return 0.0

    result, wall_clock_s = _run_l1_until_deadline(
        object(),
        object(),
        object(),
        object(),
        deadline_monotonic=5.0,
        analysis_started=0.0,
        evidence_tools_factory=None,
        clock=_Clock(),
        executor_factory=lambda **_kwargs: _Executor(),
    )
    assessment = assess_execution(configured=True, result=result).to_payload()

    assert wall_clock_s == 6.0
    assert assessment["unusable_reason"] == "analysis_deadline_exceeded"
    assert assessment["final_evidence_reason"] == "contract_repair"


def test_contract_invalid_response_is_unusable_with_closed_reason():
    evidence = deepcopy(_valid_no_failure_evidence())
    evidence.pop("related_failures")
    result = L1EvidenceResult(semantic_payload=evidence, model="test-model", success=True)

    assessment = assess_execution(configured=True, result=result).to_payload()

    assert assessment["execution_status"] == "failed"
    assert assessment["result_quality"] == "unusable"
    assert assessment["parse_status"] == "contract_invalid"
    assert assessment["unusable_reason"] == "contract_invalid"
    assert assessment["evidence_present"] is True


def test_final_output_truncation_invalidates_an_apparently_parsed_payload():
    result = L1EvidenceResult(
        semantic_payload=_valid_no_failure_evidence(),
        model="test-model",
        success=True,
        anomalies={"model_output_truncated": True},
    )

    health = output_health(result)
    assessment = health["execution_assessment"]

    assert health["status"] == "truncated"
    assert health["usable"] is False
    assert assessment["result_quality"] == "unusable"
    assert assessment["unusable_reason"] == "output_truncated"


def test_prior_output_limit_with_valid_final_evidence_is_degraded_not_unusable():
    result = L1EvidenceResult(
        semantic_payload=_valid_no_failure_evidence(),
        model="test-model",
        success=True,
        anomalies={
            "final_evidence_turn": True,
            "final_evidence_reason": "forced_final_after_output_limit",
            "prior_output_truncated": True,
        },
    )

    assessment = assess_execution(configured=True, result=result).to_payload()

    assert assessment["result_quality"] == "degraded"
    assert assessment["final_evidence_reason"] == "forced_final_after_output_limit"
    assert assessment["reason_codes"] == ["output_truncated"]
    assert assessment["usable"] is True


def test_successful_contract_repair_is_usable_but_degraded():
    result = L1EvidenceResult(
        semantic_payload=_valid_no_failure_evidence(),
        model="test-model",
        success=True,
        anomalies={
            "final_evidence_turn": True,
            "final_evidence_reason": "contract_repair",
        },
    )

    assessment = assess_execution(configured=True, result=result).to_payload()

    assert assessment["result_quality"] == "degraded"
    assert assessment["usable"] is True
    assert assessment["final_evidence_reason"] == "contract_repair"
    assert assessment["reason_codes"] == ["contract_repair"]


def test_tool_round_exhaustion_with_valid_forced_response_is_degraded():
    result = L1EvidenceResult(
        semantic_payload=_valid_no_failure_evidence(),
        model="test-model",
        success=True,
        anomalies={"tool_round_exhausted": True},
    )

    assessment = assess_execution(configured=True, result=result).to_payload()

    assert assessment["result_quality"] == "degraded"
    assert assessment["usable"] is True
    assert assessment["reason_codes"] == ["tool_round_exhausted"]


def test_tool_round_exhaustion_without_valid_evidence_is_unusable():
    result = L1EvidenceResult(
        semantic_payload=None,
        model="test-model",
        success=False,
        anomalies={"tool_round_exhausted": True},
    )

    assessment = assess_execution(configured=True, result=result).to_payload()

    assert assessment["result_quality"] == "unusable"
    assert assessment["unusable_reason"] == "tool_round_exhausted"
    assert assessment["usable"] is False
