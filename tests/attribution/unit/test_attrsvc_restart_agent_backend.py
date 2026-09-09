# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for attrsvc's direct Restart Agent backend."""

import asyncio
import logging
import threading
import time
from types import SimpleNamespace

import pytest

from nvidia_resiliency_ext.attribution.orchestration.types import LogAnalyzerError
from nvidia_resiliency_ext.attribution.restart_agent import (
    AnalysisResult,
    DecisionCandidate,
    DecisionEvidence,
    L0Artifacts,
    L0Bundle,
    L0ModelFacingView,
    ModelAnalysisResult,
    ProgressiveL0Accumulator,
    build_restart_agent_runtime,
    parse_restart_agent_config,
)
from nvidia_resiliency_ext.attribution.restart_agent.models import FailureEvidence
from nvidia_resiliency_ext.services.attrsvc import (
    restart_agent_backend as restart_agent_backend_module,
)
from nvidia_resiliency_ext.services.attrsvc.restart_agent_backend import (
    LogConvergencePolicy,
    ProgressiveAnalysisPolicy,
    RestartAgentServiceBackend,
)
from nvidia_resiliency_ext.services.attrsvc.restart_agent_logging import (
    RestartAgentLogContext,
    RestartAgentOperationalLogger,
)


def _config(*, max_total_records: int = 8):
    return parse_restart_agent_config(
        {
            "schema_version": "restart_agent_config.v1",
            "config_id": "attrsvc-test",
            "config_version": 1,
            "enrichment": {"enabled": True},
            "routing": {"mode": "collect_all", "max_parallel_models": 1},
            "runtime": {
                "history": {
                    "enabled": True,
                    "max_attempts_per_job": 4,
                    "max_total_records": max_total_records,
                }
            },
            "model_routes": [
                {
                    "route_id": "test-route",
                    "model": "test-model",
                    "base_url": "https://llm.example.test/v1",
                    "credential_ref": "TEST_LLM_KEY_FILE",
                }
            ],
        },
        environ={"TEST_LLM_KEY_FILE": "/unused/test-key"},
    )


def _analysis_result(
    decision: str,
    *,
    eligible: bool = True,
    candidate_kind: str | None = None,
) -> AnalysisResult:
    provenance = {"nvrx_use": "eligible" if eligible else "fallback_to_nvrx_default"}
    if candidate_kind is not None:
        provenance["candidate_kind"] = candidate_kind
    return AnalysisResult(
        decision=decision,
        decision_basis="test",
        result_provenance=provenance,
        justification=f"test {decision.lower()} result",
    )


class _FakeAttemptRecordControl:
    def records(self):
        return ()


class _FakeRuntime:
    def __init__(self, *, final_decision: str = "RESTART", block_after_deterministic: bool = False):
        self.attempt_record_control = _FakeAttemptRecordControl()
        self.final_decision = final_decision
        self.block_after_deterministic = block_after_deterministic
        self.deterministic_published = threading.Event()
        self.release = threading.Event()
        self.requests = []
        self.finalized_l0a = []
        self.route_trace = {
            "layers": {
                "L1": {
                    "wall_clock_s": 0.4,
                    "model_call_wall_clock_s": 0.3,
                    "tool_wall_clock_s": 0.1,
                    "model_calls": 2,
                    "failed_model_calls": 1,
                    "retried_model_calls": 1,
                    "tool_calls": 1,
                    "total_tokens": 120,
                },
                "L2": {
                    "wall_clock_s": 0.02,
                    "grounding_status": "grounded",
                    "history_identity_ready": True,
                    "root_fingerprint_source": "l2_grounded",
                    "affected_entity_available": True,
                    "audit_status": "credible",
                    "observational_finding_count": 0,
                },
                "L3": {
                    "wall_clock_s": 0.01,
                    "history_available": True,
                    "selected_failure_facts_source": "l2_grounded",
                    "same_job_attempts": 1,
                    "matching_root_attempts": 1,
                    "same_entity_attempts": 1,
                    "no_observed_advance_attempts": 1,
                    "consecutive_same_root_no_advance_attempts": 1,
                    "consecutive_same_root_and_entity_no_advance_attempts": 1,
                },
                "L4": {
                    "wall_clock_s": 0.01,
                    "base_rule": "concrete_confirmation_retry",
                    "effective_rule": "concrete_confirmation_retry",
                    "decision": "RESTART",
                    "decision_basis": "concrete_confirmation_retry_available",
                    "retry_budget_exhausted": False,
                    "general_root_ceiling": {
                        "matching_prior_attempts": 1,
                        "allowed_retries": 2,
                    },
                    "selected_policy_ledger": {
                        "matching_prior_attempts": 1,
                        "allowed_retries": 1,
                    },
                    "job_no_progress_guard": {
                        "matching_prior_attempts": 1,
                        "allowed_retries": 3,
                    },
                    "job_unknown_progress_guard": {
                        "matching_prior_attempts": 0,
                        "allowed_retries": 3,
                    },
                    "exhausted_by": [],
                },
            },
            "l1": {
                "model_calls": [
                    {
                        "model_turn": 1,
                        "attempt": 1,
                        "success": False,
                        "retry_scheduled": True,
                        "latency_s": 0.1,
                        "error_type": "http_error",
                        "http_status": 502,
                    },
                    {
                        "model_turn": 1,
                        "attempt": 2,
                        "success": True,
                        "latency_s": 0.2,
                        "finish_reason": "tool_calls",
                        "usage": {"prompt_tokens": 50, "completion_tokens": 10},
                    },
                ],
                "tool_calls": [
                    {
                        "model_turn": 1,
                        "name": "read_window",
                        "latency_ms": 100,
                        "result_lines": 20,
                    }
                ],
            },
            "l4_policy": {
                "retry_policy": {
                    "match_requirements": {"root": "required", "entity": "required"},
                }
            },
            "decision_candidates": {"selected": "l1_enriched"},
            "anomalies": {"provider_retries": 1},
        }

    def analyze(
        self,
        request,
        *,
        on_deterministic_ready,
        on_route_complete,
        on_l0_ready=None,
        retain_detailed_artifacts=True,
    ):
        assert retain_detailed_artifacts is False
        self.requests.append(request)
        if on_l0_ready is not None:
            on_l0_ready(_operational_l0_artifacts(request.log_path))
        return self._complete(on_deterministic_ready, on_route_complete)

    def analyze_prepared(
        self,
        request,
        finalized_l0a,
        *,
        on_deterministic_ready,
        on_route_complete,
        on_l0_ready=None,
        retain_detailed_artifacts=True,
    ):
        assert retain_detailed_artifacts is False
        self.requests.append(request)
        self.finalized_l0a.append(finalized_l0a)
        if on_l0_ready is not None:
            on_l0_ready(_operational_l0_artifacts(request.log_path))
        return self._complete(on_deterministic_ready, on_route_complete)

    def _complete(self, on_deterministic_ready, on_route_complete):
        deterministic = _analysis_result("STOP")
        on_deterministic_ready(
            DecisionCandidate(
                candidate_kind="deterministic",
                result=deterministic,
                ready_wall_clock_s=0.01,
                l1_execution_status="in_flight",
            )
        )
        self.deterministic_published.set()
        if self.block_after_deterministic:
            self.release.wait(timeout=2.0)
        final = _analysis_result(self.final_decision, candidate_kind="l1_enriched")
        model_result = ModelAnalysisResult(
            route_id="test-route",
            model="test-model",
            endpoint="https://llm.example.test/v1",
            credential_ref="TEST_LLM_KEY_FILE",
            execution_status="completed",
            l1_usable=True,
            analysis_result=final,
        )
        on_route_complete(model_result, self.route_trace)
        return SimpleNamespace(result=SimpleNamespace(model_results=(model_result,)))


def _backend(tmp_path, runtime, *, max_total_records: int = 8):
    return RestartAgentServiceBackend(
        allowed_root=str(tmp_path),
        runtime=runtime,
        config=_config(max_total_records=max_total_records),
        convergence=LogConvergencePolicy(
            minimum_wait_seconds=0,
            quiet_seconds=0,
            max_wait_seconds=0,
            poll_seconds=0,
        ),
        progressive=ProgressiveAnalysisPolicy(
            enabled=True,
            pre_end_poll_seconds=180,
            active_idle_seconds=900,
            max_active_states=64,
            max_completed_results=max_total_records,
        ),
    )


def _operational_l0_artifacts(log_path: str) -> L0Artifacts:
    primary = FailureEvidence(
        failure_class="cuda_runtime_failure",
        signature="not-logged-signature",
        root_fingerprint="not-logged-fingerprint",
        fault_outcome="terminal",
        line=12,
        quote="not-logged-source-text",
    )
    bundle = L0Bundle(
        log_path=log_path,
        byte_size=2048,
        line_count=20,
        deterministic_primary_candidate=primary,
    )
    decision_evidence = DecisionEvidence(
        deterministic_primary_candidate=primary,
        selected_observed_failure=None,
        canonical_observed_identity={"root_fingerprint": "not-logged-fingerprint"},
        selected_evidence_references={"source_lines": [12]},
        failure_position={"line": 12},
        progress_checkpoint_state={},
        later_progress_recovery={},
        locality={},
        coverage_lossiness={},
        provenance={},
    )
    model_view = L0ModelFacingView(
        decision_evidence=decision_evidence,
        failure_narrative={
            "status": "available",
            "identity_kind": "primary",
            "events": [],
            "known_unknowns": [],
        },
        decision_evidence_view={
            "canonical_observed_identity": {"root_fingerprint": "not-logged-fingerprint"},
            "failure_position": {"line": 12},
        },
        evidence_bundle={"context_windows": []},
        attempt_execution_context={},
        projection_metrics={
            "view_size": {"compact_json_characters": 1024, "estimated_tokens": 342},
            "selection_counts": {"context_windows": {"available": 2, "selected": 1, "omitted": 1}},
            "compaction_counts": {"truncated_context_windows": 1},
            "projection_integrity": {
                "status": "ok",
                "deterministic_payload_sha256": "sha256:test",
            },
        },
    )
    return L0Artifacts(
        bundle=bundle,
        decision_evidence=decision_evidence,
        model_view=model_view,
        l0a_wall_clock_s=0.2,
        decision_evidence_wall_clock_s=0.01,
        l0b_wall_clock_s=0.02,
        l0_wall_clock_s=0.23,
        l0_reused=False,
    )


def test_live_log_convergence_policy_defaults():
    policy = LogConvergencePolicy()

    assert policy.minimum_wait_seconds == 10.0
    assert policy.quiet_seconds == 5.0
    assert policy.max_wait_seconds == 40.0
    assert policy.poll_seconds == 0.25


def test_terminal_service_logs_lifecycle_and_completed_stage_events(tmp_path, caplog):
    # Arrange
    runtime = _FakeRuntime()
    backend = _backend(tmp_path, runtime)
    log_path = tmp_path / "train_cycle2.log"
    log_path.write_text("RuntimeError: failure\n", encoding="utf-8")
    caplog.set_level(logging.DEBUG)

    async def run():
        await backend.submit_log(
            str(log_path),
            job_id="job-7",
            analysis_intent="terminal",
        )
        return await backend.analyze_log(str(log_path), wait=True)

    try:
        # Act
        result = asyncio.run(run())

        # Assert
        assert result.status == "completed"
        messages = "\n".join(record.getMessage() for record in caplog.records)
        for event in (
            "restart_agent.request.accepted",
            "restart_agent.terminal.started",
            "restart_agent.terminal.drain_completed",
            "restart_agent.l0a.completed",
            "restart_agent.decision_evidence.completed",
            "restart_agent.l0b.completed",
            "restart_agent.candidate.ready",
            "restart_agent.l1.completed",
            "restart_agent.l1.model_call.completed",
            "restart_agent.l1.tool_call.completed",
            "restart_agent.l2.completed",
            "restart_agent.l3.completed",
            "restart_agent.l4.completed",
            "restart_agent.analysis.completed",
        ):
            assert f"event={event}" in messages
        assert "job_id=job-7" in messages
        assert "cycle_id=2" in messages
        assert "model_wall_clock_s=0.300000" in messages
        assert "tool_wall_clock_s=0.100000" in messages
        assert "total_tokens=120" in messages
        assert "endpoint_issues=1" in messages
        assert any(
            "event=restart_agent.candidate.ready" in record.getMessage()
            and "candidate_kind=l1_enriched" in record.getMessage()
            and "route_id=test-route" in record.getMessage()
            for record in caplog.records
        )
        assert "credential_ref" not in messages
    finally:
        backend.shutdown()


def test_l0_operational_log_uses_stage_artifacts_without_evidence_content(caplog):
    # Arrange
    artifacts = _operational_l0_artifacts("/logs/train_cycle2.log")
    target = logging.getLogger("test.restart_agent.operational")
    caplog.set_level(logging.DEBUG, logger=target.name)
    operational_log = RestartAgentOperationalLogger(target)

    # Act
    operational_log.l0_completed(
        RestartAgentLogContext("job-7", 2, "/logs/train_cycle2.log"),
        artifacts,
        progressive_metrics={
            "source_decode_wall_clock_s": 0.04,
            "source_index_classify_wall_clock_s": 0.05,
            "source_ingest_wall_clock_s": 0.09,
            "l0a_bundle_wall_clock_s": 0.11,
            "decision_evidence_wall_clock_s": 0.01,
            "l0a_reduction_wall_clock_s": 0.12,
        },
    )

    # Assert
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=restart_agent.l0a.completed" in messages
    assert "event=restart_agent.decision_evidence.completed" in messages
    assert "event=restart_agent.l0a.detail" in messages
    assert "event=restart_agent.l0b.completed" in messages
    assert "event=restart_agent.l0b.detail" in messages
    assert "source_ingest_s=0.090000" in messages
    assert "evidence_assembly_s=0.110000" in messages
    assert "decision_evidence_s=0.010000" in messages
    assert "cumulative_compute_s=0.210000" in messages
    assert "cumulative_selection_s=0.010000" in messages
    assert "compact_json_chars=1024" in messages
    assert "projection_integrity=ok" in messages
    assert "not-logged-signature" not in messages
    assert "not-logged-fingerprint" not in messages
    assert "not-logged-source-text" not in messages


def test_route_log_uses_selected_candidate_and_preserves_l2_not_run(caplog):
    # Arrange
    target = logging.getLogger("test.restart_agent.route-selection")
    caplog.set_level(logging.INFO, logger=target.name)
    operational_log = RestartAgentOperationalLogger(target)
    result = ModelAnalysisResult(
        route_id="test-route",
        model="test-model",
        endpoint="https://llm.example.test/v1",
        credential_ref="TEST_LLM_KEY_FILE",
        execution_status="completed",
        l1_usable=True,
        analysis_result=_analysis_result("RESTART"),
    )
    trace = {
        "layers": {
            "L1": {"wall_clock_s": 0.1},
            "L2": {
                "wall_clock_s": 0.0,
                "grounding_status": "not_run",
                "audit_status": "not_run",
            },
            "L3": {},
            "L4": {},
        },
        "decision_candidates": {"selected": "deterministic"},
    }

    # Act
    operational_log.route_completed(
        RestartAgentLogContext("job-7", 2, "/logs/train_cycle2.log"),
        result,
        trace,
        terminal_to_ready_s=0.2,
    )

    # Assert
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "event=restart_agent.l2.completed" in message and "status=not_run" in message
        for message in messages
    )
    for event in (
        "restart_agent.l3.completed",
        "restart_agent.l4.completed",
        "restart_agent.candidate.ready",
    ):
        assert any(
            f"event={event}" in message and "candidate_kind=deterministic" in message
            for message in messages
        )


@pytest.mark.parametrize(
    ("reason", "model_call"),
    [
        (
            "context_budget_exceeded",
            {
                "success": False,
                "error_type": "context_budget_exceeded",
                "error": "request exceeds configured context budget",
            },
        ),
        (
            "analysis_deadline_exceeded",
            {
                "success": False,
                "error_type": "analysis_deadline_exceeded",
                "error": "analysis deadline exceeded during HTTP request",
                "timeout": True,
            },
        ),
    ],
)
def test_route_log_separates_non_endpoint_l1_failure_from_endpoint_issue(
    caplog,
    reason,
    model_call,
):
    target = logging.getLogger(f"test.restart_agent.{reason}")
    caplog.set_level(logging.INFO, logger=target.name)
    operational_log = RestartAgentOperationalLogger(target)
    assessment = {
        "execution_status": "failed",
        "result_quality": "unusable",
        "reason_codes": [reason],
        "unusable_reason": reason,
    }
    result = ModelAnalysisResult(
        route_id="oversized-route",
        model="test-model",
        endpoint="https://llm.example.test/v1",
        credential_ref="TEST_LLM_KEY_FILE",
        execution_status="failed",
        l1_usable=False,
        analysis_result=_analysis_result("RESTART"),
        l1_execution_assessment=assessment,
    )
    trace = {
        "layers": {
            "L1": {"wall_clock_s": 0.01, "execution_assessment": assessment},
            "L2": {"grounding_status": "not_run", "audit_status": "not_run"},
            "L3": {},
            "L4": {},
        },
        "l1": {"model_calls": [model_call]},
        "decision_candidates": {"selected": "deterministic"},
    }

    operational_log.route_completed(
        RestartAgentLogContext("job-7", 2, "/logs/train_cycle2.log"),
        result,
        trace,
        terminal_to_ready_s=0.02,
    )

    l1_message = next(
        record.getMessage()
        for record in caplog.records
        if "event=restart_agent.l1.completed" in record.getMessage()
    )
    assert "execution_status=failed" in l1_message
    assert f"unusable_reason={reason}" in l1_message
    assert "endpoint_issues=0" in l1_message
    assert f"error_classification={reason}" in l1_message


def test_operational_handler_failure_does_not_strand_terminal_attempt(
    tmp_path,
    monkeypatch,
):
    # Arrange
    runtime = _FakeRuntime(final_decision="RESTART")
    backend = _backend(tmp_path, runtime)
    log_path = tmp_path / "train_cycle3.log"
    log_path.write_text("RuntimeError: failure\n", encoding="utf-8")

    def fail_logging(*args, **kwargs):
        raise RuntimeError("test log handler failure")

    for method in ("info", "debug", "warning", "error"):
        monkeypatch.setattr(backend._operational_log, method, fail_logging)

    async def run():
        await backend.submit_log(
            str(log_path),
            job_id="job-7",
            analysis_intent="terminal",
        )
        return await backend.analyze_log(str(log_path), wait=True)

    try:
        # Act
        result = asyncio.run(run())

        # Assert
        assert result.status == "completed"
        assert result.recommendation["action"] == "RESTART"
        assert result.candidate_recommendation["action"] == "RESTART"
        assert runtime.deterministic_published.is_set()
    finally:
        backend.shutdown()


def test_analysis_failure_emits_bounded_error_event(tmp_path, monkeypatch, caplog):
    # Arrange
    runtime = _FakeRuntime()
    backend = _backend(tmp_path, runtime)
    log_path = tmp_path / "train_cycle4.log"
    log_path.write_text("RuntimeError: failure\n", encoding="utf-8")
    caplog.set_level(logging.ERROR)

    def fail_analysis(*args, **kwargs):
        raise RuntimeError("sensitive provider response")

    monkeypatch.setattr(runtime, "analyze", fail_analysis)
    monkeypatch.setattr(runtime, "analyze_prepared", fail_analysis)

    async def run():
        await backend.submit_log(
            str(log_path),
            job_id="job-7",
            analysis_intent="terminal",
        )
        return await backend.analyze_log(str(log_path), wait=True)

    try:
        # Act
        result = asyncio.run(run())

        # Assert
        assert result.status == "completed"
        assert result.result == {
            "analysis_outcome": "failed",
            "error": "RuntimeError: sensitive provider response",
        }
        assert result.recommendation == {
            "action": "UNKNOWN",
            "reason": "analysis_failed",
            "source": "restart_agent",
        }
        messages = "\n".join(record.getMessage() for record in caplog.records)
        assert "event=restart_agent.analysis.failed" in messages
        assert "error_classification=RuntimeError" in messages
        assert "sensitive provider response" not in messages
    finally:
        backend.shutdown()


def test_progressive_start_accepts_expected_file_before_creation_and_preserves_cycle_zero(
    tmp_path,
):
    runtime = _FakeRuntime()
    backend = _backend(tmp_path, runtime)
    log_path = str(tmp_path / "train_cycle0.log")

    async def run():
        submitted = await backend.submit_log(
            log_path,
            user="alice",
            job_id="job-7",
            analysis_intent="progressive",
        )
        probe = await backend.analyze_log(log_path, wait=False)
        return submitted, probe

    try:
        submitted, probe = asyncio.run(run())
        jobs = backend.get_all_jobs()

        assert not isinstance(submitted, LogAnalyzerError)
        assert submitted.submitted is True
        assert probe.status == "pending"
        assert jobs["job-7"][0]["cycle_id"] == 0
        assert runtime.requests == []
    finally:
        backend.shutdown()


def test_default_terminal_first_policy_registers_then_analyzes_at_terminal(tmp_path):
    runtime = _FakeRuntime()
    backend = RestartAgentServiceBackend(
        allowed_root=str(tmp_path),
        runtime=runtime,
        config=_config(),
        convergence=LogConvergencePolicy(
            minimum_wait_seconds=0,
            quiet_seconds=0,
            max_wait_seconds=0,
            poll_seconds=0,
        ),
        progressive=ProgressiveAnalysisPolicy(),
    )
    log_path = tmp_path / "train_cycle1.log"
    log_path.write_text("RuntimeError: failure\n", encoding="utf-8")

    async def run():
        await backend.submit_log(
            str(log_path),
            job_id="job-7",
            analysis_intent="progressive",
        )
        before_terminal = await backend.get_stats()
        await backend.submit_log(
            str(log_path),
            analysis_intent="terminal",
        )
        result = await backend.analyze_log(str(log_path), wait=True)
        return before_terminal, result

    try:
        stats, result = asyncio.run(run())

        progressive = stats["restart_agent"]["progressive"]
        assert progressive["l0a_build_count"] == 0
        assert progressive["states"] == {"disabled": 1}
        assert result.status == "completed"
        assert len(runtime.finalized_l0a) == 1
    finally:
        backend.shutdown()


def test_progressive_start_precomputes_and_terminal_reuses_finalized_l0a(tmp_path, caplog):
    runtime = _FakeRuntime()
    backend = RestartAgentServiceBackend(
        allowed_root=str(tmp_path),
        runtime=runtime,
        config=_config(),
        convergence=LogConvergencePolicy(
            minimum_wait_seconds=0,
            quiet_seconds=0,
            max_wait_seconds=0,
            poll_seconds=0,
        ),
        progressive=ProgressiveAnalysisPolicy(
            enabled=True,
            pre_end_poll_seconds=60,
            active_idle_seconds=600,
            max_active_states=4,
            max_completed_results=8,
        ),
    )
    log_path = tmp_path / "train_cycle1.log"
    log_path.write_text(
        "[2026-01-01 00:00:00] iteration 7 / 100 | consumed samples: 64 |\n"
        "RuntimeError: CUDA out of memory\n",
        encoding="utf-8",
    )
    caplog.set_level(logging.DEBUG)

    async def run():
        await backend.submit_log(
            str(log_path),
            job_id="job-7",
            analysis_intent="progressive",
        )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            stats = await backend.get_stats()
            if stats["restart_agent"]["progressive"]["l0a_build_count"] == 1:
                break
            await asyncio.sleep(0.01)
        await backend.submit_log(
            str(log_path),
            analysis_intent="terminal",
        )
        return await backend.analyze_log(str(log_path), wait=True)

    try:
        result = asyncio.run(run())

        assert result.status == "completed"
        assert len(runtime.finalized_l0a) == 1
        finalized = runtime.finalized_l0a[0]
        assert finalized.precomputed is True
        assert finalized.source_log.lines == (
            "[2026-01-01 00:00:00] iteration 7 / 100 | consumed samples: 64 |",
            "RuntimeError: CUDA out of memory",
        )
        assert finalized.progressive_metrics["l0a_build_count"] == 1
        stats = asyncio.run(backend.get_stats())
        assert stats["restart_agent"]["progressive"]["active_state_count"] == 0
        messages = "\n".join(record.getMessage() for record in caplog.records)
        assert "event=restart_agent.progressive.registered" in messages
        assert "event=restart_agent.progressive.refresh.completed" in messages
        assert "source_ingest_s=" in messages
        assert "l0a_reduction_s=" in messages
    finally:
        backend.shutdown()


def test_terminal_drain_ingests_late_rank_output_and_precomputes_final_boundary(tmp_path):
    # Arrange
    runtime = _FakeRuntime()
    first_precompute_completed = threading.Event()

    class _ObservedAccumulator(ProgressiveL0Accumulator):
        def refresh(self, *, precompute=True):
            changed = super().refresh(precompute=precompute)
            if precompute:
                first_precompute_completed.set()
            return changed

    backend = RestartAgentServiceBackend(
        allowed_root=str(tmp_path),
        runtime=runtime,
        config=_config(),
        convergence=LogConvergencePolicy(
            minimum_wait_seconds=0,
            quiet_seconds=0.04,
            max_wait_seconds=0.3,
            poll_seconds=0.01,
        ),
        progressive=ProgressiveAnalysisPolicy(
            pre_end_poll_seconds=180,
            active_idle_seconds=900,
            max_active_states=4,
            max_completed_results=8,
        ),
        accumulator_factory=_ObservedAccumulator,
    )
    log_path = tmp_path / "train_cycle1.log"
    log_path.write_text("iteration 7 completed\n", encoding="utf-8")

    async def run():
        await backend.submit_log(
            str(log_path),
            job_id="job-7",
            analysis_intent="terminal",
        )
        assert await asyncio.to_thread(first_precompute_completed.wait, 0.3)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("RuntimeError: CUDA out of memory\n")
        return await backend.analyze_log(str(log_path), wait=True)

    try:
        # Act
        result = asyncio.run(run())

        # Assert
        assert result.status == "completed"
        finalized = runtime.finalized_l0a[0]
        assert finalized.source_log.lines == (
            "iteration 7 completed",
            "RuntimeError: CUDA out of memory",
        )
        assert finalized.progressive_metrics["poll_count"] >= 3
        assert finalized.progressive_metrics["growth_count"] >= 2
        assert finalized.precomputed is True
        assert finalized.progressive_metrics["l0a_build_count"] == 2
        assert (
            finalized.canonical_hash
            == ProgressiveL0Accumulator(str(log_path)).finalize().canonical_hash
        )
        assert finalized.progressive_metrics["source_decode_wall_clock_s"] >= 0
        assert finalized.progressive_metrics["source_index_classify_wall_clock_s"] >= 0
        assert finalized.progressive_metrics["source_ingest_wall_clock_s"] >= 0
        assert finalized.progressive_metrics["l0a_bundle_wall_clock_s"] >= 0
        assert finalized.progressive_metrics["decision_evidence_wall_clock_s"] >= 0
        assert finalized.progressive_metrics["l0a_reduction_wall_clock_s"] >= 0
        drain = finalized.progressive_metrics["terminal_drain"]
        assert drain["converged"] is True
        assert drain["max_wait_expired"] is False
        assert drain["poll_count"] >= 2

        stats = asyncio.run(backend.get_stats())
        terminal_timing = stats["restart_agent"]["terminal_timing"]
        assert terminal_timing["drain_wall_clock_s"] >= 0
        assert terminal_timing["l0a_ready_wall_clock_s"] >= terminal_timing["drain_wall_clock_s"]
        assert (
            terminal_timing["deterministic_ready_wall_clock_s"]
            >= terminal_timing["l0a_ready_wall_clock_s"]
        )
        assert (
            terminal_timing["first_route_ready_wall_clock_s"]
            >= terminal_timing["deterministic_ready_wall_clock_s"]
        )
        assert (
            terminal_timing["analysis_completed_wall_clock_s"]
            >= terminal_timing["first_route_ready_wall_clock_s"]
        )
    finally:
        backend.shutdown()


def test_terminal_drain_requires_minimum_observation_before_quiet_convergence(tmp_path):
    runtime = _FakeRuntime()
    backend = RestartAgentServiceBackend(
        allowed_root=str(tmp_path),
        runtime=runtime,
        config=_config(),
        convergence=LogConvergencePolicy(
            minimum_wait_seconds=0.04,
            quiet_seconds=0.02,
            max_wait_seconds=0.2,
            poll_seconds=0.005,
        ),
    )
    log_path = tmp_path / "train_cycle1.log"
    log_path.write_text("RuntimeError: CUDA out of memory\n", encoding="utf-8")

    try:
        outcome = backend._wait_for_log_convergence(str(log_path), accumulator=None)

        assert outcome.converged is True
        assert outcome.max_wait_expired is False
        assert outcome.wall_clock_s >= 0.05
        assert outcome.completion_reason == "quiet_after_minimum_wait"
        assert outcome.minimum_wait_seconds == 0.04
        assert outcome.quiet_seconds == 0.02
        assert outcome.max_wait_seconds == 0.2
    finally:
        backend.shutdown()


def test_terminal_drain_observes_source_while_initial_ingest_is_running(tmp_path, monkeypatch):
    # Arrange
    runtime = _FakeRuntime()
    backend = RestartAgentServiceBackend(
        allowed_root=str(tmp_path),
        runtime=runtime,
        config=_config(),
        convergence=LogConvergencePolicy(
            minimum_wait_seconds=0,
            quiet_seconds=0.02,
            max_wait_seconds=0.5,
            poll_seconds=0.01,
        ),
    )
    log_path = tmp_path / "train_cycle1.log"
    log_path.write_text("iteration 7 completed\n", encoding="utf-8")
    source_observed_twice = threading.Event()
    precompute_requested = threading.Event()
    real_stat = restart_agent_backend_module.os.stat
    stat_calls = 0

    def observed_stat(path):
        nonlocal stat_calls
        result = real_stat(path)
        if str(path) == str(log_path):
            stat_calls += 1
            if stat_calls >= 2:
                source_observed_twice.set()
        return result

    class _BlockedAccumulator:
        def refresh(self, *, precompute):
            if precompute:
                precompute_requested.set()
            else:
                assert source_observed_twice.wait(timeout=0.3)

    monkeypatch.setattr(restart_agent_backend_module.os, "stat", observed_stat)

    try:
        # Act
        outcome = backend._wait_for_log_convergence(str(log_path), _BlockedAccumulator())

        # Assert
        assert outcome.converged is True
        assert outcome.max_wait_expired is False
        assert outcome.poll_count >= 3
        assert precompute_requested.is_set()
    finally:
        backend.shutdown()


def test_terminal_drain_completion_wakes_reader_without_poll_delay(tmp_path, monkeypatch):
    # Arrange
    runtime = _FakeRuntime()
    backend = RestartAgentServiceBackend(
        allowed_root=str(tmp_path),
        runtime=runtime,
        config=_config(),
        convergence=LogConvergencePolicy(
            minimum_wait_seconds=0,
            quiet_seconds=0.01,
            max_wait_seconds=2,
            poll_seconds=1,
        ),
    )

    def finish_after_reader_is_waiting(
        path,
        *,
        reader_ready,
        notifications,
        stop_observer,
    ):
        del path, notifications, stop_observer
        assert reader_ready.wait(timeout=0.3)
        time.sleep(0.05)
        return restart_agent_backend_module._DrainOutcome(
            converged=True,
            max_wait_expired=False,
        )

    monkeypatch.setattr(
        backend,
        "_observe_log_convergence",
        finish_after_reader_is_waiting,
    )

    try:
        # Act
        started = time.monotonic()
        outcome = backend._wait_for_log_convergence(str(tmp_path / "job.log"), None)
        elapsed = time.monotonic() - started

        # Assert
        assert outcome.converged is True
        assert elapsed < 0.5
    finally:
        backend.shutdown()


def test_path_without_cycle_suffix_remains_analyzable_without_inventing_cycle_zero(tmp_path):
    runtime = _FakeRuntime()
    backend = _backend(tmp_path, runtime)
    log_path = tmp_path / "train.log"
    log_path.write_text("failure\n", encoding="utf-8")

    async def run():
        return await backend.submit_log(
            str(log_path),
            user="alice",
            job_id="job-7",
            analysis_intent="progressive",
        )

    try:
        submitted = asyncio.run(run())
        jobs = backend.get_all_jobs()

        assert not isinstance(submitted, LogAnalyzerError)
        assert jobs["job-7"][0]["cycle_id"] is None
    finally:
        backend.shutdown()


def test_explicit_cycle_id_takes_precedence_over_path_inference(tmp_path):
    runtime = _FakeRuntime()
    backend = _backend(tmp_path, runtime)
    log_path = tmp_path / "train_cycle2.log"
    log_path.write_text("failure\n", encoding="utf-8")

    async def run():
        started = await backend.submit_log(
            str(log_path),
            user="alice",
            job_id="job-7",
            cycle_id=9,
            analysis_intent="progressive",
        )
        terminal = await backend.submit_log(
            str(log_path),
            user="alice",
            analysis_intent="terminal",
        )
        result = await backend.analyze_log(str(log_path), wait=True)
        return started, terminal, result

    try:
        started, terminal, result = asyncio.run(run())

        assert not isinstance(started, LogAnalyzerError)
        assert not isinstance(terminal, LogAnalyzerError)
        assert result.status == "completed"
        assert runtime.requests[0].cycle_id == 9
        assert backend.get_all_jobs()["job-7"][0]["cycle_id"] == 9
    finally:
        backend.shutdown()


def test_explicit_cycle_id_conflict_is_rejected_after_registration(tmp_path):
    runtime = _FakeRuntime()
    backend = _backend(tmp_path, runtime)
    log_path = tmp_path / "train_cycle2.log"

    async def run():
        first = await backend.submit_log(
            str(log_path),
            job_id="job-7",
            cycle_id=9,
            analysis_intent="progressive",
        )
        conflicting = await backend.submit_log(
            str(log_path),
            job_id="job-7",
            cycle_id=10,
            analysis_intent="terminal",
        )
        return first, conflicting

    try:
        first, conflicting = asyncio.run(run())

        assert not isinstance(first, LogAnalyzerError)
        assert isinstance(conflicting, LogAnalyzerError)
        assert conflicting.error_code.value == "invalid_parameter"
    finally:
        backend.shutdown()


def test_direct_backend_rejects_non_integer_explicit_cycle_id(tmp_path):
    runtime = _FakeRuntime()
    backend = _backend(tmp_path, runtime)

    async def run():
        return await backend.submit_log(
            str(tmp_path / "train.log"),
            job_id="job-7",
            cycle_id=True,
            analysis_intent="progressive",
        )

    try:
        result = asyncio.run(run())

        assert isinstance(result, LogAnalyzerError)
        assert result.error_code.value == "invalid_parameter"
    finally:
        backend.shutdown()


def test_terminal_missing_log_returns_explicit_restart_agent_unavailable_result(tmp_path):
    config = _config()
    backend = RestartAgentServiceBackend(
        allowed_root=str(tmp_path),
        runtime=build_restart_agent_runtime(config),
        config=config,
        convergence=LogConvergencePolicy(
            minimum_wait_seconds=0,
            quiet_seconds=0,
            max_wait_seconds=0,
            poll_seconds=0,
        ),
    )
    log_path = tmp_path / "never_created_cycle1.log"

    async def run():
        submitted = await backend.submit_log(
            str(log_path), job_id="job-7", analysis_intent="terminal"
        )
        result = await backend.analyze_log(str(log_path), wait=True)
        return submitted, result

    try:
        submitted, result = asyncio.run(run())

        assert not isinstance(submitted, LogAnalyzerError)
        assert result.status == "completed"
        assert result.result["decision"] == "RESTART"
        assert result.result["decision_basis"] == "log_unavailable"
        assert result.recommendation["action"] == "UNKNOWN"
    finally:
        backend.shutdown()


def test_terminal_submission_is_idempotent_and_get_returns_completed_result(tmp_path):
    runtime = _FakeRuntime(final_decision="RESTART")
    backend = _backend(tmp_path, runtime)
    log_path = tmp_path / "train_cycle2.log"
    log_path.write_text("failure\n", encoding="utf-8")

    async def run():
        first = await backend.submit_log(str(log_path), job_id="job-7", analysis_intent="terminal")
        second = await backend.submit_log(str(log_path), job_id="job-7", analysis_intent="terminal")
        result = await backend.analyze_log(str(log_path), wait=True)
        return first, second, result

    try:
        first, second, result = asyncio.run(run())

        assert not isinstance(first, LogAnalyzerError)
        assert not isinstance(second, LogAnalyzerError)
        assert len(runtime.requests) == 1
        assert runtime.requests[0].job_id == "job-7"
        assert runtime.requests[0].cycle_id == 2
        assert result.status == "completed"
        assert result.recommendation["action"] == "RESTART"
    finally:
        backend.shutdown()


def test_same_path_cannot_be_registered_with_conflicting_job_identity(tmp_path):
    runtime = _FakeRuntime()
    backend = _backend(tmp_path, runtime)
    log_path = tmp_path / "train_cycle2.log"
    log_path.write_text("failure\n", encoding="utf-8")

    async def run():
        first = await backend.submit_log(
            str(log_path), job_id="job-7", analysis_intent="progressive"
        )
        conflicting = await backend.submit_log(
            str(log_path), job_id="job-8", analysis_intent="progressive"
        )
        return first, conflicting

    try:
        first, conflicting = asyncio.run(run())

        assert not isinstance(first, LogAnalyzerError)
        assert isinstance(conflicting, LogAnalyzerError)
        assert conflicting.error_code.value == "invalid_parameter"
    finally:
        backend.shutdown()


def test_nonblocking_get_publishes_deterministic_result_while_model_is_running(tmp_path):
    runtime = _FakeRuntime(final_decision="RESTART", block_after_deterministic=True)
    backend = _backend(tmp_path, runtime)
    log_path = tmp_path / "train_cycle3.log"
    log_path.write_text("failure\n", encoding="utf-8")

    async def run():
        await backend.submit_log(str(log_path), job_id="job-7", analysis_intent="terminal")
        assert runtime.deterministic_published.wait(timeout=1.0)
        deterministic = await backend.analyze_log(str(log_path), wait=False)
        runtime.release.set()
        completed = await backend.analyze_log(str(log_path), wait=True)
        return deterministic, completed

    try:
        deterministic, completed = asyncio.run(run())

        assert deterministic.status == "in_flight"
        assert deterministic.recommendation["action"] == "STOP"
        assert deterministic.recommendation["source"] == "deterministic"
        assert deterministic.candidate_recommendation["action"] == "STOP"
        assert deterministic.candidate_recommendation["source"] == "deterministic"
        assert completed.status == "completed"
        assert completed.recommendation["action"] == "RESTART"
        assert completed.candidate_recommendation["action"] == "RESTART"
        assert completed.candidate_recommendation["source"] == "l1_enriched:test-route"
    finally:
        runtime.release.set()
        backend.shutdown()


def test_registry_evicts_completed_attempt_before_rejecting_new_work(tmp_path):
    runtime = _FakeRuntime()
    backend = _backend(tmp_path, runtime, max_total_records=1)
    first = tmp_path / "train_cycle1.log"
    second = tmp_path / "train_cycle2.log"
    first.write_text("failure\n", encoding="utf-8")
    second.write_text("failure\n", encoding="utf-8")

    async def run():
        await backend.submit_log(str(first), job_id="job-7", analysis_intent="terminal")
        await backend.analyze_log(str(first), wait=True)
        submitted = await backend.submit_log(
            str(second), job_id="job-7", analysis_intent="progressive"
        )
        stats = await backend.get_stats()
        return submitted, stats

    try:
        submitted, stats = asyncio.run(run())

        assert not isinstance(submitted, LogAnalyzerError)
        assert stats["restart_agent"]["registry_evictions"] == 1
        assert list(backend.get_all_jobs()["job-7"])[0]["cycle_id"] == 2
    finally:
        backend.shutdown()


def test_outside_root_and_missing_parent_are_rejected(tmp_path):
    runtime = _FakeRuntime()
    backend = _backend(tmp_path, runtime)

    async def run():
        outside = await backend.submit_log(
            "/tmp/outside_cycle1.log", job_id="job-7", analysis_intent="progressive"
        )
        missing_parent = await backend.submit_log(
            str(tmp_path / "missing" / "train_cycle1.log"),
            job_id="job-7",
            analysis_intent="progressive",
        )
        return outside, missing_parent

    try:
        outside, missing_parent = asyncio.run(run())

        assert isinstance(outside, LogAnalyzerError)
        assert outside.error_code.value == "outside_root"
        assert isinstance(missing_parent, LogAnalyzerError)
        assert missing_parent.error_code.value == "not_found"
    finally:
        backend.shutdown()
