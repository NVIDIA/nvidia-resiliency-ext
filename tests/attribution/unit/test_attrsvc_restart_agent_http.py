# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public HTTP lifecycle tests for attrsvc's direct Restart Agent path."""

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("slowapi")
from fastapi.testclient import TestClient  # noqa: E402

from nvidia_resiliency_ext.attribution.restart_agent import (  # noqa: E402
    AnalysisResult,
    DecisionCandidate,
    ModelAnalysisResult,
)
from nvidia_resiliency_ext.services.attrsvc.app import create_app  # noqa: E402
from nvidia_resiliency_ext.services.attrsvc.config import Settings  # noqa: E402


class _AttemptRecords:
    def records(self):
        return ()


class _Runtime:
    def __init__(self):
        self.attempt_record_control = _AttemptRecords()
        self.requests = []

    def analyze(
        self,
        request,
        *,
        on_deterministic_ready,
        on_route_complete,
        retain_detailed_artifacts=True,
    ):
        assert retain_detailed_artifacts is False
        self.requests.append(request)
        result = AnalysisResult(
            decision="STOP",
            decision_basis="test",
            result_provenance={"nvrx_use": "eligible"},
            justification="test stop result",
        )
        on_deterministic_ready(
            DecisionCandidate(
                candidate_kind="deterministic",
                result=result,
                ready_wall_clock_s=0.01,
                l1_execution_status="in_flight",
            )
        )
        model_result = ModelAnalysisResult(
            route_id="nvrx-default",
            model="test-model",
            endpoint="https://llm.example.test/v1",
            credential_ref="LLM_API_KEY_FILE",
            execution_status="completed",
            l1_usable=True,
            analysis_result=result,
        )
        on_route_complete(model_result, {})
        return SimpleNamespace(result=SimpleNamespace(model_results=(model_result,)))


def test_public_progressive_terminal_and_get_lifecycle(tmp_path, monkeypatch):
    key_file = tmp_path / "key"
    key_file.write_text("secret", encoding="utf-8")
    monkeypatch.setenv("LLM_API_KEY_FILE", str(key_file))
    runtime = _Runtime()
    monkeypatch.setattr(
        "nvidia_resiliency_ext.services.attrsvc.service.build_restart_agent_runtime",
        lambda config: runtime,
    )
    cfg = Settings(
        ALLOWED_ROOT=str(tmp_path),
        LLM_MODEL="test-model",
        LLM_BASE_URL="https://llm.example.test/v1",
        RESTART_AGENT_LOG_MAX_WAIT_SECONDS=0,
        _env_file=None,
    )
    log_path = tmp_path / "train_cycle0.log"

    with TestClient(create_app(cfg)) as client:
        start = client.post(
            "/logs",
            json={
                "log_path": str(log_path),
                "user": "alice",
                "job_id": "job-7",
                "cycle_id": 8,
                "analysis_intent": "progressive",
            },
        )
        pending = client.get("/logs", params={"log_path": str(log_path), "wait": False})
        log_path.write_text("failure\n", encoding="utf-8")
        terminal = client.post(
            "/logs",
            json={
                "log_path": str(log_path),
                "user": "alice",
                "job_id": "job-7",
                "cycle_id": 8,
                "analysis_intent": "terminal",
            },
        )
        completed = client.get("/logs", params={"log_path": str(log_path), "wait": True})

    assert start.status_code == 200
    assert terminal.status_code == 200
    assert pending.json()["status"] == "pending"
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
    assert completed.json()["recommendation"]["action"] == "STOP"
    assert completed.json()["candidate_recommendation"]["action"] == "STOP"
    assert len(runtime.requests) == 1
    assert runtime.requests[0].job_id == "job-7"
    assert runtime.requests[0].cycle_id == 8


def test_public_submit_rejects_boolean_cycle_id(tmp_path, monkeypatch):
    key_file = tmp_path / "key"
    key_file.write_text("secret", encoding="utf-8")
    monkeypatch.setenv("LLM_API_KEY_FILE", str(key_file))
    monkeypatch.setattr(
        "nvidia_resiliency_ext.services.attrsvc.service.build_restart_agent_runtime",
        lambda config: _Runtime(),
    )
    cfg = Settings(
        ALLOWED_ROOT=str(tmp_path),
        LLM_MODEL="test-model",
        LLM_BASE_URL="https://llm.example.test/v1",
        RESTART_AGENT_LOG_MAX_WAIT_SECONDS=0,
        _env_file=None,
    )

    with TestClient(create_app(cfg)) as client:
        response = client.post(
            "/logs",
            json={
                "log_path": str(tmp_path / "train.log"),
                "job_id": "job-7",
                "cycle_id": True,
                "analysis_intent": "progressive",
            },
        )

    assert response.status_code == 422
