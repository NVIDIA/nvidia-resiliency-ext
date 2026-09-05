# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for observational L2 recovery-support auditing."""

from nvidia_resiliency_ext.attribution.restart_agent.infrastructure.log_source import LogSnapshot
from nvidia_resiliency_ext.attribution.restart_agent.l0 import (
    build_decision_evidence,
    build_l0_bundle,
    build_l0_model_facing_view,
)
from nvidia_resiliency_ext.attribution.restart_agent.l1 import L1EvidenceResult
from nvidia_resiliency_ext.attribution.restart_agent.l2 import (
    L2GroundingInput,
    ground_and_audit_model_evidence,
)
from nvidia_resiliency_ext.attribution.restart_agent.l4 import L4PolicyInput, evaluate_policy
from nvidia_resiliency_ext.attribution.restart_agent.models import (
    HistorySummary,
    ModelRecoveryAssessment,
)


def _checkpoint_timeout_evidence(failure_line):
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
                "value": "workload",
                "status": "established_by_current_log",
                "confidence": 90,
            },
            "retry_outlook_without_workload_change": {
                "value": "cannot_recover",
                "status": "established_by_current_log",
                "confidence": 85,
            },
            "rationale": "The model claims the unchanged workload cannot recover.",
        },
        "related_failures": [],
        "evidence": [
            {
                "id": "primary-timeout",
                "line": 2,
                "quote": failure_line,
                "supports": ["primary_failure", "root_cause_assessment"],
            }
        ],
        "category_selection": {
            "category_id": 0,
            "category_confidence": 0,
            "category_rationale": "not applicable",
        },
    }


def test_missing_recovery_support_is_an_l2_finding_without_losing_identity(tmp_path):
    checkpoint_line = "loading distributed checkpoint from /checkpoints/model at iteration 635000"
    failure_line = (
        "[rank482] Watchdog caught collective operation timeout: "
        "WorkNCCL(SeqNum=1, OpType=ALLGATHER)"
    )
    log_path = tmp_path / "job.log"
    log_path.write_text(f"{checkpoint_line}\n{failure_line}\n", encoding="utf-8")
    source_log = LogSnapshot.read(log_path)
    bundle = build_l0_bundle(log_path, source_log=source_log)
    model_view = build_l0_model_facing_view(bundle, build_decision_evidence(bundle))
    l1_result = L1EvidenceResult(
        semantic_payload=_checkpoint_timeout_evidence(failure_line),
        model="test-model",
        success=True,
        transcript_events=(
            {
                "event_type": "bundle_snapshot",
                "model_visible_payload": {
                    "evidence_bundle": {
                        "lines": [
                            {"line": 1, "text": checkpoint_line},
                            {"line": 2, "text": failure_line},
                        ]
                    }
                },
            },
        ),
    )

    result = ground_and_audit_model_evidence(
        L2GroundingInput(
            bundle=bundle,
            model_view=model_view,
            l1_result=l1_result,
            source_log=source_log,
        )
    )
    payload = result.to_payload()

    assert result.used is True
    assert result.primary is not None
    assert result.primary.line == 2
    assert result.primary_failure_facts is not None
    assert result.primary_failure_facts.affected_entity is not None
    assert result.primary_failure_facts.affected_entity.identity == "/checkpoints/model"
    assert payload["audit_influence"] == "observational_only"
    assert payload["failure_domain_support_expected"] is True
    assert payload["retry_outlook_support_expected"] is True
    assert payload["field_finding_codes"]["model_recovery_assessment"] == [
        "failure_domain_support_missing",
        "retry_outlook_support_missing",
    ]
    assert payload["audit_status"] == "findings"

    assessment_payload = l1_result.semantic_payload["model_recovery_assessment"]
    policy = evaluate_policy(
        L4PolicyInput(
            primary=result.primary,
            current_failure_facts=result.primary_failure_facts,
            current_affected_entity=result.primary_failure_facts.affected_entity,
            model_recovery_assessment=ModelRecoveryAssessment.from_mapping(assessment_payload),
            history=HistorySummary(available=False),
        )
    ).retry_policy

    assert policy.base_rule == "workload_unrecoverable"
    assert policy.decision == "STOP"


def test_l2_grounds_direct_object_and_enclosing_artifact_independently(tmp_path):
    dataset_path = "/datasets/train.json"
    lock_path = "/cache/datasets/train.lock"
    dataset_line = f"dataset_path={dataset_path}"
    failure_line = f"PermissionError: [Errno 13] Permission denied: '{lock_path}'"
    log_path = tmp_path / "job.log"
    log_path.write_text(f"{dataset_line}\n{failure_line}\n", encoding="utf-8")
    source_log = LogSnapshot.read(log_path)
    bundle = build_l0_bundle(log_path, source_log=source_log)
    model_view = build_l0_model_facing_view(bundle, build_decision_evidence(bundle))
    payload = _checkpoint_timeout_evidence(failure_line)
    payload["primary_failure"] = {
        "line": 2,
        "causal_role": "initiating",
        "failure_identity": {
            "operation": "load_dataset",
            "mechanism": "permission_denied",
            "component": "dataset_cache",
            "direct_failure_object_path": lock_path,
            "affected_artifact_path": dataset_path,
        },
    }
    payload["evidence"][0].update(
        {
            "id": "permission-denied",
            "quote": failure_line,
        }
    )
    l1_result = L1EvidenceResult(
        semantic_payload=payload,
        model="test-model",
        success=True,
        transcript_events=(
            {
                "event_type": "bundle_snapshot",
                "model_visible_payload": {
                    "evidence_bundle": {
                        "path_access_facts": [
                            {"line": 1, "path": dataset_path, "role": "configured_read"},
                            {"line": 2, "path": lock_path, "role": "failed_access"},
                        ],
                        "lines": [
                            {"line": 1, "text": dataset_line},
                            {"line": 2, "text": failure_line},
                        ],
                    }
                },
            },
        ),
    )

    result = ground_and_audit_model_evidence(
        L2GroundingInput(
            bundle=bundle,
            model_view=model_view,
            l1_result=l1_result,
            source_log=source_log,
        )
    )
    result_payload = result.to_payload()

    identity = result_payload["failure_identity_grounding"]
    assert identity["direct_failure_object_path"] == {
        "model_value": lock_path,
        "grounded_value": lock_path,
        "evidence_lines": [2],
        "status": "grounded",
    }
    assert identity["affected_artifact_path"] == {
        "model_value": dataset_path,
        "grounded_value": dataset_path,
        "evidence_lines": [1],
        "status": "grounded",
    }
    assert result.primary_failure_facts is not None
    assert result.primary_failure_facts.affected_entity is not None
    assert result.primary_failure_facts.affected_entity.identity == dataset_path
    assert result_payload["affected_entity_selection"] == {
        "source_field": "affected_artifact_path",
        "selection_reason": "grounded_affected_artifact_preferred",
        "evidence_lines": [1],
        "entity": result.primary_failure_facts.affected_entity.to_payload(),
    }
    assert "direct_failure_object_path" not in result.primary_failure_facts.to_payload()


def test_l2_uses_grounded_direct_object_when_enclosing_artifact_is_unavailable(tmp_path):
    affected_path = "/cache/datasets"
    lock_path = f"{affected_path}/train.lock"
    failure_line = f"PermissionError: [Errno 13] Permission denied: '{lock_path}'"
    log_path = tmp_path / "job.log"
    log_path.write_text(f"{failure_line}\n", encoding="utf-8")
    source_log = LogSnapshot.read(log_path)
    bundle = build_l0_bundle(log_path, source_log=source_log)
    model_view = build_l0_model_facing_view(bundle, build_decision_evidence(bundle))
    payload = _checkpoint_timeout_evidence(failure_line)
    payload["primary_failure"] = {
        "line": 1,
        "causal_role": "initiating",
        "failure_identity": {
            "operation": "acquire_lock",
            "mechanism": "permission_denied",
            "component": "dataset_cache",
            "direct_failure_object_path": lock_path,
            "affected_artifact_path": affected_path,
        },
    }
    payload["evidence"][0].update({"line": 1, "quote": failure_line})
    l1_result = L1EvidenceResult(
        semantic_payload=payload,
        model="test-model",
        success=True,
        transcript_events=(
            {
                "event_type": "bundle_snapshot",
                "model_visible_payload": {
                    "evidence_bundle": {
                        "lines": [{"line": 1, "text": failure_line}],
                    }
                },
            },
        ),
    )

    result = ground_and_audit_model_evidence(
        L2GroundingInput(
            bundle=bundle,
            model_view=model_view,
            l1_result=l1_result,
            source_log=source_log,
        )
    )
    identity = result.to_payload()["failure_identity_grounding"]

    assert identity["direct_failure_object_path"]["status"] == "grounded"
    assert identity["affected_artifact_path"]["status"] == "unavailable"
    assert result.primary_failure_facts is not None
    assert result.primary_failure_facts.affected_entity is not None
    assert result.primary_failure_facts.affected_entity.identity == lock_path
    assert result.to_payload()["affected_entity_selection"] == {
        "source_field": "direct_failure_object_path",
        "selection_reason": "grounded_direct_failure_object_fallback",
        "evidence_lines": [1],
        "entity": result.primary_failure_facts.affected_entity.to_payload(),
    }
