# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observation-only evidence, history, and root-independent policy behavior."""

from __future__ import annotations

from copy import deepcopy

from nvidia_resiliency_ext.attribution.restart_agent.l0.assembly import build_l0_bundle
from nvidia_resiliency_ext.attribution.restart_agent.l0.codec import read_l0_bundle, write_l0_bundle
from nvidia_resiliency_ext.attribution.restart_agent.l1 import L1EvidenceResult
from nvidia_resiliency_ext.attribution.restart_agent.l1.advisories import (
    model_evidence_contract_advisories,
)
from nvidia_resiliency_ext.attribution.restart_agent.l1.validation import (
    model_evidence_contract_errors,
)
from nvidia_resiliency_ext.attribution.restart_agent.models import (
    PriorAttemptView,
    RestartAgentRequest,
)
from nvidia_resiliency_ext.attribution.restart_agent.pipeline import RestartAgent

_PROGRESS = "0: [2026-02-18 08:19:00.000000] iteration 10/ 100 | consumed samples: 100 |"
_TRANSPORT = "24: [rank24] TCPStore recvValue failed: Connection reset by peer"


class _ObservationExtractor:
    def __init__(self, payload=None):
        self.payload = payload

    def extract_evidence(self, context, *, deadline_monotonic=None):
        payload = self.payload or _observation_payload()
        return L1EvidenceResult(
            semantic_payload=payload,
            model="test-observation-model",
            success=True,
        )


def _observation_payload():
    return {
        "schema_version": "restart_agent_evidence.v1",
        "analysis_status": "insufficient_evidence",
        "primary_failure": None,
        "observed_failures": [
            {
                "id": "o1",
                "line": 2,
                "causal_role": "cascade",
                "failure_identity": {
                    "operation": "distributed_coordination",
                    "mechanism": "tcpstore_connection_loss",
                    "component": "c10d_tcpstore",
                    "direct_failure_object_path": None,
                    "affected_artifact_path": None,
                },
                "rationale": "The store connection failed after progress stopped.",
                "evidence_ids": ["e1"],
            }
        ],
        "selected_observed_failure_id": "o1",
        "root_cause_assessment": {
            "summary": "Insufficient evidence to identify a primary failure.",
            "status": "unknown",
            "plausible_causes": [],
            "missing_evidence": ["The store-owner termination is absent."],
        },
        "model_recovery_assessment": {
            "failure_domain": {"value": "unknown", "status": "unknown", "confidence": 1},
            "retry_outlook_without_workload_change": {
                "value": "unknown",
                "status": "unknown",
                "confidence": 1,
            },
            "rationale": "Recovery is not assessed without an identified primary failure.",
        },
        "related_failures": [],
        "evidence": [
            {
                "id": "e1",
                "line": 2,
                "quote": _TRANSPORT,
                "supports": ["root_cause_assessment"],
            }
        ],
        "category_selection": {
            "category_id": 0,
            "category_confidence": 0,
            "category_rationale": "not applicable",
        },
    }


def _primary_and_observation_payload(primary_line):
    payload = deepcopy(_observation_payload())
    payload["analysis_status"] = "primary_identified"
    payload["primary_failure"] = {
        "line": 1,
        "causal_role": "initiating",
        "failure_identity": {
            "operation": "checkpoint_load",
            "mechanism": "metadata_decode_error",
            "component": "checkpoint_loader",
            "direct_failure_object_path": None,
            "affected_artifact_path": "/checkpoints/model",
        },
    }
    payload["observed_failures"][0]["line"] = 2
    payload["evidence"] = [
        {
            "id": "primary-1",
            "line": 1,
            "quote": primary_line,
            "supports": ["primary_failure", "root_cause_assessment"],
        },
        {
            "id": "e1",
            "line": 2,
            "quote": _TRANSPORT,
            "supports": ["root_cause_assessment"],
        },
    ]
    return payload


def _write_log(tmp_path):
    path = tmp_path / "attempt.log"
    path.write_text(f"{_PROGRESS}\n{_TRANSPORT}\n", encoding="utf-8")
    return path


def test_l0_preserves_terminal_surface_without_promoting_a_primary(tmp_path):
    path = _write_log(tmp_path)

    bundle = build_l0_bundle(str(path))

    assert bundle.deterministic_primary_candidate is None
    assert bundle.selected_observed_failure is not None
    assert bundle.selected_observed_failure.line == 2
    assert bundle.selected_observed_failure.root_fingerprint is None
    assert bundle.selected_observed_failure.observation_fingerprint


def test_l0_bundle_codec_preserves_selected_observation(tmp_path):
    path = _write_log(tmp_path)
    record = tmp_path / "l0.json"
    bundle = build_l0_bundle(str(path))

    write_l0_bundle(record, bundle)
    replayed = read_l0_bundle(record, expected_log_path=str(path))

    assert replayed.selected_observed_failure == bundle.selected_observed_failure


def test_l1_contract_accepts_grounded_selected_observation():
    assert model_evidence_contract_errors(_observation_payload()) == []


def test_l1_contract_accepts_primary_with_independent_selected_observation():
    payload = _primary_and_observation_payload("RuntimeError: checkpoint metadata decode failed")

    assert model_evidence_contract_errors(payload) == []


def test_l2_publishes_primary_and_observation_as_independent_tracks(tmp_path):
    primary_line = "RuntimeError: checkpoint metadata decode failed"
    log_path = tmp_path / "both-tracks.log"
    log_path.write_text(f"{primary_line}\n{_TRANSPORT}\n", encoding="utf-8")
    payload = _primary_and_observation_payload(primary_line)

    run = RestartAgent(_ObservationExtractor(payload)).run(
        RestartAgentRequest(log_path=str(log_path), job_id="job-1", cycle_id=1)
    )

    assert run.attempt_record is not None
    route = run.attempt_record.enriched[0]
    assert route.primary is not None
    assert route.observation is not None
    assert run.result.result_provenance["selected_evidence_path"] == "primary"
    tracks = run.result.l2_grounding["enriched_failure_tracks"]
    assert tracks["primary"] is not None
    assert tracks["observation"] is not None


def test_l2_unresolved_selected_observation_does_not_erase_primary(tmp_path):
    primary_line = "RuntimeError: checkpoint metadata decode failed"
    log_path = tmp_path / "unresolved-observation.log"
    log_path.write_text(f"{primary_line}\n{_TRANSPORT}\n", encoding="utf-8")
    model_payload = _primary_and_observation_payload(primary_line)
    model_payload["selected_observed_failure_id"] = "missing-observation"

    run = RestartAgent(_ObservationExtractor(model_payload)).run(
        RestartAgentRequest(log_path=str(log_path), job_id="job-1", cycle_id=1)
    )
    payload = run.result.to_payload()

    assert model_evidence_contract_errors(model_payload) == []
    assert [item["code"] for item in model_evidence_contract_advisories(model_payload)] == [
        "selected_observation_id_unresolved"
    ]
    assert payload["l1_assessment"] == model_payload
    assert payload["l2_grounding"]["track_grounding"]["primary"]["published"] is True
    assert payload["l2_grounding"]["track_grounding"]["observation"]["published"] is False
    observation_findings = payload["l2_grounding"]["track_findings"]["observation"]
    assert any(
        item["code"] == "selected_observation_id_unresolved" for item in observation_findings
    )
    assert payload["result_provenance"]["model_contribution"] == "attempted_used"


def test_l2_ambiguous_selected_observation_does_not_erase_primary(tmp_path):
    primary_line = "RuntimeError: checkpoint metadata decode failed"
    log_path = tmp_path / "ambiguous-observation.log"
    log_path.write_text(f"{primary_line}\n{_TRANSPORT}\n", encoding="utf-8")
    model_payload = _primary_and_observation_payload(primary_line)
    model_payload["observed_failures"].append(deepcopy(model_payload["observed_failures"][0]))

    run = RestartAgent(_ObservationExtractor(model_payload)).run(
        RestartAgentRequest(log_path=str(log_path), job_id="job-1", cycle_id=1)
    )
    payload = run.result.to_payload()

    assert model_evidence_contract_errors(model_payload) == []
    assert payload["l2_grounding"]["track_grounding"]["primary"]["published"] is True
    assert payload["l2_grounding"]["track_grounding"]["observation"]["published"] is False
    observation_codes = {
        item["code"] for item in payload["l2_grounding"]["track_findings"]["observation"]
    }
    assert observation_codes == {"selected_observation_id_ambiguous"}


def test_l2_duplicate_evidence_ids_do_not_block_direct_observation_grounding(tmp_path):
    path = _write_log(tmp_path)
    model_payload = deepcopy(_observation_payload())
    model_payload["evidence"].append(deepcopy(model_payload["evidence"][0]))

    run = RestartAgent(_ObservationExtractor(model_payload)).run(
        RestartAgentRequest(log_path=str(path), job_id="job-1", cycle_id=1)
    )
    payload = run.result.to_payload()

    assert model_evidence_contract_errors(model_payload) == []
    assert [item["code"] for item in model_evidence_contract_advisories(model_payload)] == [
        "duplicate_evidence_id"
    ]
    assert payload["selected_observed_failure"]["line"] == 2
    assert payload["l2_grounding"]["track_grounding"]["observation"]["published"] is True


def test_l1_contract_defers_dangling_observation_reference_to_l2():
    payload = deepcopy(_observation_payload())
    payload["observed_failures"][0]["evidence_ids"] = ["line-2"]
    payload["evidence"] = []

    assert model_evidence_contract_errors(payload) == []


def test_l2_keeps_model_observation_separate_from_primary(tmp_path):
    path = _write_log(tmp_path)

    run = RestartAgent(_ObservationExtractor()).run(
        RestartAgentRequest(log_path=str(path), job_id="job-1", cycle_id=1)
    )
    payload = run.result.to_payload()

    assert payload["primary_failure"] is None
    assert payload["selected_observed_failure"]["line"] == 2
    assert payload["l2_grounding"]["history_identities"]["primary"] is None
    assert (
        payload["l2_grounding"]["history_identities"]["observation"]["identity_kind"]
        == "observation_only"
    )
    assert payload["result_provenance"]["result_quality"] == "normal"


def test_l2_audits_dangling_reference_and_grounds_direct_observation_line(tmp_path):
    path = _write_log(tmp_path)
    model_payload = deepcopy(_observation_payload())
    model_payload["observed_failures"][0]["evidence_ids"] = ["line-2"]
    model_payload["evidence"] = []

    run = RestartAgent(_ObservationExtractor(model_payload)).run(
        RestartAgentRequest(log_path=str(path), job_id="job-1", cycle_id=1)
    )
    payload = run.result.to_payload()

    assert payload["l1_assessment"] == model_payload
    assert payload["selected_observed_failure"]["line"] == 2
    assert payload["l2_grounding"]["grounding_status"] == "grounded"
    assert payload["l2_grounding"]["audit_status"] == "findings"
    dangling = next(
        finding
        for finding in payload["l2_grounding"]["findings"]
        if finding["code"] == "dangling_evidence_reference"
    )
    assert "line-2" in dangling["message"]
    assert payload["result_provenance"]["model_contribution"] == "attempted_used"


def test_l2_does_not_ground_dangling_reference_with_unseen_direct_line(tmp_path):
    path = _write_log(tmp_path)
    model_payload = deepcopy(_observation_payload())
    model_payload["observed_failures"][0]["line"] = 200
    model_payload["observed_failures"][0]["evidence_ids"] = ["line-200"]
    model_payload["evidence"] = []

    run = RestartAgent(_ObservationExtractor(model_payload)).run(
        RestartAgentRequest(log_path=str(path), job_id="job-1", cycle_id=1)
    )
    payload = run.result.to_payload()

    assert payload["l2_grounding"]["grounding_status"] == "unavailable"
    assert payload["result_provenance"]["model_contribution"] == ("attempted_not_used_ungrounded")


def test_l2_audits_unselected_observation_references_without_using_model(tmp_path):
    path = _write_log(tmp_path)
    model_payload = deepcopy(_observation_payload())
    model_payload["selected_observed_failure_id"] = None
    model_payload["observed_failures"][0]["evidence_ids"] = ["e-1", "e-2"]
    model_payload["evidence"] = []

    run = RestartAgent(_ObservationExtractor(model_payload)).run(
        RestartAgentRequest(log_path=str(path), job_id="job-1", cycle_id=1)
    )
    payload = run.result.to_payload()

    assert payload["l1_assessment"] == model_payload
    assert payload["l2_grounding"]["grounding_status"] == "unavailable"
    assert payload["l2_grounding"]["grounded_selected_observation"] is None
    dangling = next(
        finding
        for finding in payload["l2_grounding"]["findings"]
        if finding["code"] == "dangling_evidence_reference"
    )
    assert "e-1, e-2" in dangling["message"]
    assert payload["result_provenance"]["model_contribution"] == ("attempted_not_used_ungrounded")


def test_observation_only_policy_exhausts_same_job_no_progress_budget(tmp_path):
    path = _write_log(tmp_path)
    agent = RestartAgent()
    prior_records = []
    decisions = []

    for cycle_id in (1, 2, 3):
        prior = PriorAttemptView(
            records=tuple(prior_records),
            available=True,
            availability_reason="ready",
        )
        run = agent.run(
            RestartAgentRequest(log_path=str(path), job_id="job-1", cycle_id=cycle_id),
            prior_attempts=prior,
        )
        decisions.append(run.result.decision)
        assert run.attempt_record is not None
        prior_records.append(run.attempt_record)

    assert decisions == ["RESTART", "RESTART", "STOP"]
    retry_policy = run.result.retry_policy
    assert retry_policy["base_rule"] == "general_retry"
    assert retry_policy["selected_policy_ledger"]["history_match_scope"] == ("same_job_no_progress")
    assert retry_policy["selected_policy_ledger"]["matching_prior_attempts"] == 2
