# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for attrsvc's direct Restart Agent backend."""

import asyncio
import threading
import time
from types import SimpleNamespace

from nvidia_resiliency_ext.attribution.orchestration.types import LogAnalyzerError
from nvidia_resiliency_ext.attribution.restart_agent import (
    AnalysisResult,
    DecisionCandidate,
    ModelAnalysisResult,
    ProgressiveL0Accumulator,
    build_restart_agent_runtime,
    parse_restart_agent_config,
)
from nvidia_resiliency_ext.services.attrsvc import (
    restart_agent_backend as restart_agent_backend_module,
)
from nvidia_resiliency_ext.services.attrsvc.restart_agent_backend import (
    LogConvergencePolicy,
    ProgressiveAnalysisPolicy,
    RestartAgentServiceBackend,
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


def _analysis_result(decision: str, *, eligible: bool = True) -> AnalysisResult:
    return AnalysisResult(
        decision=decision,
        decision_basis="test",
        result_provenance={"nvrx_use": "eligible" if eligible else "fallback_to_nvrx_default"},
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
        return self._complete(on_deterministic_ready, on_route_complete)

    def analyze_prepared(
        self,
        request,
        finalized_l0a,
        *,
        on_deterministic_ready,
        on_route_complete,
        retain_detailed_artifacts=True,
    ):
        assert retain_detailed_artifacts is False
        self.requests.append(request)
        self.finalized_l0a.append(finalized_l0a)
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
        final = _analysis_result(self.final_decision)
        model_result = ModelAnalysisResult(
            route_id="test-route",
            model="test-model",
            endpoint="https://llm.example.test/v1",
            credential_ref="TEST_LLM_KEY_FILE",
            execution_status="completed",
            l1_usable=True,
            analysis_result=final,
        )
        on_route_complete(model_result, {})
        return SimpleNamespace(result=SimpleNamespace(model_results=(model_result,)))


def _backend(tmp_path, runtime, *, max_total_records: int = 8):
    return RestartAgentServiceBackend(
        allowed_root=str(tmp_path),
        runtime=runtime,
        config=_config(max_total_records=max_total_records),
        convergence=LogConvergencePolicy(quiet_seconds=0, max_wait_seconds=0, poll_seconds=0),
        progressive=ProgressiveAnalysisPolicy(
            enabled=True,
            pre_end_poll_seconds=180,
            active_idle_seconds=900,
            max_active_states=64,
            max_completed_results=max_total_records,
        ),
    )


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


def test_progressive_start_precomputes_and_terminal_reuses_finalized_l0a(tmp_path):
    runtime = _FakeRuntime()
    backend = RestartAgentServiceBackend(
        allowed_root=str(tmp_path),
        runtime=runtime,
        config=_config(),
        convergence=LogConvergencePolicy(
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


def test_terminal_drain_observes_source_while_initial_ingest_is_running(tmp_path, monkeypatch):
    # Arrange
    runtime = _FakeRuntime()
    backend = RestartAgentServiceBackend(
        allowed_root=str(tmp_path),
        runtime=runtime,
        config=_config(),
        convergence=LogConvergencePolicy(
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
        convergence=LogConvergencePolicy(quiet_seconds=0, max_wait_seconds=0, poll_seconds=0),
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
