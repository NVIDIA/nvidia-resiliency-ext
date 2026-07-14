# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic rich route payloads shared by panel summary tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from _builders import recovery_assessment, retry_policy
from restart_agent_eval import panel


def published_panel(summaries):
    """Publish and read the public panel artifact contract."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        json_path, markdown_path = panel.write_panel_summary(run_dir, list(summaries))
        return (
            json.loads(json_path.read_text(encoding="utf-8")),
            markdown_path.read_text(encoding="utf-8"),
            (run_dir / "panel_diagnostics.md").read_text(encoding="utf-8"),
        )


def panel_summary(summaries):
    """Return the payload published by the public panel API."""
    payload, _, _ = published_panel(summaries)
    return payload


def rich_panel_artifacts():
    """Return two comparable routes and their published panel artifacts."""
    summary = rich_route_summary()
    second = {**summary, "target": "gpt", "model": "test-gpt"}
    payload, summary_markdown, diagnostics_markdown = published_panel([summary, second])
    return (
        summary,
        second,
        payload,
        payload["rows"][0],
        summary_markdown,
        diagnostics_markdown,
    )


def rich_route_summary() -> dict:
    summary = {
        "target": "qwen235b",
        "model": "test-model",
        "l1_response_parsed": True,
        "l2_audit_ran": True,
        "l2_audit_status": "clean",
        "decision": "RESTART",
        "decision_basis": "general_retry_available",
        "retry_policy": retry_policy(
            policy_version="retry_budget.v1",
            rule="bounded_retry",
            allowed_retries=1,
            matching_prior_failures=0,
        ),
        "primary_failure": {
            "fine_class": "checkpoint_decode_error",
            "policy_class": "ambiguous",
            "line": 12083,
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "root_fingerprint": "checkpoint_decode_error:utf8",
            "root_fingerprint_source": "observed_exception",
            "failure_identity": {
                "schema_version": "restart_agent_failure_identity.experimental.v1",
                "policy_active": False,
                "family": {
                    "operation": "checkpoint_load",
                    "mechanism": "metadata_deserialization",
                    "exception_type": "unicode_decode_error",
                    "label": "checkpoint_load|metadata_deserialization|unicode_decode_error",
                    "fingerprint": "family:sha256:1234567890abcdef",
                    "complete": True,
                },
                "concrete": {
                    "component": "torch.distributed.checkpoint",
                    "callsite": "tensor_to_object",
                    "artifact_path": "/lustre/checkpoints/iter_5000/metadata",
                    "failure_position": "8",
                    "stack_path": ["determine_global_metadata", "tensor_to_object"],
                    "label": "concrete identity",
                    "fingerprint": "concrete:sha256:abcdef1234567890",
                    "complete": True,
                },
                "client_concrete": {
                    "exception_type": "unicode_decode_error",
                    "message_signature": "unicode_decode_error_invalid_byte_position_n",
                    "source_file": "/workspace/checkpoint/utils.py",
                    "callsite": "tensor_to_object",
                    "artifact_path": "/lustre/checkpoints/iter_5000/metadata",
                    "failure_position": "8",
                    "phase": "checkpoint_load_start",
                    "checkpoint_iteration": 5000,
                    "operation_signature": "optype=allgather,seqnum=2,numelin=10,numelout=20",
                    "stack_path": [
                        "determine_global_metadata",
                        "tensor_to_object",
                    ],
                    "label": "client concrete identity",
                    "fingerprint": "client_concrete:sha256:fedcba0987654321",
                    "complete": True,
                },
            },
        },
        "primary_selection_by_stage": {
            "l0_deterministic": {
                "fine_class": "observed_exception",
                "line": 12080,
            },
            "l1_semantic": {
                "fine_class": "checkpoint_metadata_decode_error",
                "line": 12083,
            },
            "l2_grounded": {
                "fine_class": "checkpoint_decode_error",
                "line": 12083,
            },
            "l1_relation_to_l0": "same_failure_episode",
            "l2_relation_to_l0": "same_failure_episode",
        },
        "decision_evidence": {
            "schema_version": "restart_agent_decision_evidence.v1",
            "deterministic_primary_candidate": {
                "fine_class": "observed_exception",
                "line": 12080,
                "policy_class": "ambiguous",
                "fault_outcome": "terminal",
                "phase": "steady_mid",
                "causal_role": "unknown",
                "root_fingerprint": "observed:unicode_decode_error",
                "root_fingerprint_source": "observed_exception",
            },
            "canonical_observed_identity": {
                "available": True,
                "identity_anchor_line": 12083,
                "identity_anchor_reason": (
                    "deterministic_primary_is_episode_identity_anchor:terminal_exception"
                ),
                "root_fingerprint": "checkpoint_decode_error:utf8",
                "root_fingerprint_source": "checkpoint_decode_error",
                "registry_id": "observed_exception",
            },
            "selected_evidence_references": {
                "source_lines": [12080, 12083],
                "candidate_anchor_ids": ["ca-1"],
                "context_window_ids": ["w-1"],
                "failure_episode_ids": ["fe-1"],
                "distributed_incident_ids": [],
                "occurrence_group_ids": ["og-1"],
            },
            "progress_checkpoint_state": {
                "first_iteration": 609126,
                "last_iteration": 620340,
                "last_progress_line": 239900,
                "checkpoint_load_iteration": 620250,
                "last_checkpoint_iteration": 620250,
                "progress_after_failure_episode": False,
            },
        },
        "root_cause_assessment": {
            "summary": "Checkpoint metadata could not be decoded.",
            "status": "supported_but_unconfirmed",
            "plausible_causes": ["checkpoint corruption", "transient read"],
            "missing_evidence": ["a repeat read at the same position"],
        },
        "later_progress_after_fault_observations": [
            {
                "fine_class": "filesystem_access_error",
                "root_fingerprint": "observed:oserror:input_output_error",
                "event_count": 3,
                "sample_event_lines": [100, 200, 300],
                "sample_later_progress_lines": [110, 210, 310],
                "matches_terminal_fingerprint": False,
                "ordering_basis": "log_order",
                "interpretation": "job_progress_observed_after_event",
                "component_recovery_proven": False,
            }
        ],
        "model_recovery_assessment": recovery_assessment(
            failure_domain="workload",
            failure_domain_status="supported_but_unconfirmed",
            failure_domain_confidence=75,
            retry_outlook="may_recover",
            retry_outlook_status="supported_but_unconfirmed",
            retry_outlook_confidence=75,
            rationale="A checkpoint read may recover on retry.",
        ),
        "l1_kpis": {
            "response_parsed": True,
            "output_status": "usable",
            "output_usable": True,
            "contract_repair_requested": True,
            "model_calls": 3,
            "model_turns": 2,
            "extra_model_turns_after_initial": 1,
            "tool_driven_model_turns": 0,
            "contract_repair_turns": 1,
            "model_call_wall_clock_s": 2.4,
            "context_budget_adjusted_calls": 1,
            "context_window_tokens": 200000,
            "max_estimated_input_tokens": 136001,
            "configured_max_output_tokens": 64000,
            "minimum_effective_max_output_tokens": 59803,
            "provider_reported_timing": {
                "source": "response_headers",
                "reported_call_count": 2,
                "components_ms_total": {
                    "downstream_llm_api_ms": 1800.0,
                    "proxy_pre_processing_ms": 3.0,
                    "proxy_post_processing_ms": 1.0,
                },
            },
        },
        "l2_kpis": {
            "audit_status": "clean",
            "primary_available": True,
            "recovery_assessment_available": True,
            "finding_count": 0,
            "root_fingerprint_owner": "L2",
            "root_fingerprint": "checkpoint_decode_error:utf8",
            "root_fingerprint_source": "observed_exception",
            "root_fingerprint_available": True,
            "history_identity_ready": True,
            "matches_l0_root_fingerprint": False,
            "stable_identity_anchor_line": 12083,
            "stable_identity_anchor_reason": "model_primary_is_episode_terminal",
        },
        "l3_kpis": {
            "wall_clock_s": 0.001,
            "history_available": False,
            "same_job_attempts": 0,
            "matching_root_attempts": 0,
            "observed_advance_attempts": 0,
            "no_observed_advance_attempts": 0,
            "unknown_progress_attempts": 0,
            "exact_failure_position_attempts": 0,
            "same_data_position_attempts": 0,
            "same_artifact_attempts": 0,
            "consecutive_same_root_no_advance_attempts": 0,
            "advanced_beyond_all_comparable_attempts": False,
        },
        "l4_kpis": {
            "wall_clock_s": 0.001,
            "policy_version": "retry_budget.v1",
            "rule": "bounded_retry",
            "allowed_retries": 1,
            "matching_prior_failures": 0,
            "retry_budget_exhausted": False,
            "current_evidence_qualified": False,
            "observed_advance": False,
            "failure_domain": "workload",
            "failure_domain_status": "supported_but_unconfirmed",
            "failure_domain_confidence": 75,
            "retry_outlook_without_workload_change": "may_recover",
            "retry_outlook_status": "supported_but_unconfirmed",
            "retry_outlook_confidence": 75,
            "result_quality": "normal",
            "nvrx_use": "eligible",
        },
        "l0_bundle_kpis": {
            "l0_wall_clock_s": 2.5,
            "decision_evidence_wall_clock_s": 0.002,
            "line_count": 243184,
            "byte_size": 28038563,
            "context_window_count": 9,
            "candidate_anchor_count": 14,
            "occurrence_group_count": 7,
            "failure_episode_count": 3,
            "distributed_failure_incident_count": None,
            "root_fingerprint_owner": "L0",
            "root_fingerprint": "observed:unicode_decode_error",
            "root_fingerprint_source": "observed_exception",
            "root_fingerprint_available": True,
            "history_identity_ready": True,
            "restart_environment_context": {
                "workload_unchanged": True,
                "process_state_recreated": True,
                "normal_restart_delay_applies": True,
                "hardware_allocation_may_change": True,
                "external_service_state_may_change": True,
            },
            "candidate_anchors_without_excerpt": 5,
            "successful_runtime_seconds": 38496.7,
            "first_iteration": 609126,
            "last_iteration": 620340,
            "iteration_delta": 11214,
            "last_checkpoint_iteration": 620250,
            "iterations_since_checkpoint": 90,
            "checkpoint_load_iteration": 620250,
            "latest_observed_failure_iteration": 620340,
            "latest_observed_failure_iteration_line": 240000,
            "first_terminal_incident_line": 239950,
            "observed_iterations_after_checkpoint_load": 90,
            "observed_failure_phase": "steady_mid",
            "progress_after_fault": False,
            "later_progress_after_fault_observation_count": 1,
            "later_progress_after_fault_event_count": 3,
        },
        "model_selection_signals": {
            "context_efficiency": "good",
            "endpoint_reliability": "ok",
            "semantic_safety": "ok",
            "failed_endpoint_attempts": 0,
        },
        "path_redaction_audit": {"passed": True},
    }
    return summary
