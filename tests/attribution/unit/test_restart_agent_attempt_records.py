# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for attempt-record assembly, history storage, and runtime state."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from nvidia_resiliency_ext.attribution.restart_agent.agent_runtime import RestartAgentRuntime
from nvidia_resiliency_ext.attribution.restart_agent.attempt_records import (
    AttemptRecordAssembler,
    AttemptRecordControl,
    InMemoryAttemptRecordStore,
    NullAttemptRecordStore,
)
from nvidia_resiliency_ext.attribution.restart_agent.cli import main as cli_main
from nvidia_resiliency_ext.attribution.restart_agent.config import parse_restart_agent_config
from nvidia_resiliency_ext.attribution.restart_agent.l1 import L1EvidenceResult, ModelRoute
from nvidia_resiliency_ext.attribution.restart_agent.l3 import (
    DETERMINISTIC_FACT_SELECTOR,
    HistoryEvaluationInput,
    evaluate_history,
)
from nvidia_resiliency_ext.attribution.restart_agent.models import (
    AttemptFailureFacts,
    AttemptFailureFactsSource,
    AttemptProgressSummary,
    AttemptRecord,
    PriorAttemptView,
    RestartAgentRequest,
    normalize_attempt_records,
)
from nvidia_resiliency_ext.attribution.restart_agent.pipeline import RestartAgent


class _EvidenceExtractor:
    def __init__(self, line: str) -> None:
        self._line = line

    def extract_evidence(self, context, *, deadline_monotonic=None):
        evidence = {
            "schema_version": "restart_agent_evidence.v1",
            "analysis_status": "primary_identified",
            "primary_failure": {
                "causal_role": "initiating",
                "line": 1,
                "failure_identity": {
                    "operation": "cuda_initialization",
                    "mechanism": "runtime_error",
                    "component": "cuda",
                    "artifact_path": None,
                },
            },
            "root_cause_assessment": {
                "summary": "CUDA initialization failed.",
                "status": "supported_but_unconfirmed",
                "plausible_causes": ["runtime initialization failure"],
                "missing_evidence": ["device diagnostics"],
            },
            "model_recovery_assessment": {
                "failure_domain": {
                    "value": "unknown",
                    "status": "unknown",
                    "confidence": 50,
                },
                "retry_outlook_without_workload_change": {
                    "value": "unknown",
                    "status": "unknown",
                    "confidence": 50,
                },
                "rationale": "The current log does not establish recovery behavior.",
            },
            "related_failures": [],
            "evidence": [
                {
                    "id": "e1",
                    "line": 1,
                    "quote": self._line,
                    "supports": [
                        "primary_failure",
                        "root_cause_assessment",
                        "failure_domain",
                        "retry_outlook_without_workload_change",
                    ],
                }
            ],
        }
        return L1EvidenceResult(
            evidence=evidence,
            model="test-model",
            raw_model_output=json.dumps(evidence),
            success=True,
        )


class _SlowEvidenceExtractor(_EvidenceExtractor):
    def extract_evidence(self, context, *, deadline_monotonic=None):
        time.sleep(0.1)
        return super().extract_evidence(context, deadline_monotonic=deadline_monotonic)


class _BlockingFirstEvidenceExtractor(_EvidenceExtractor):
    def __init__(self, line: str) -> None:
        super().__init__(line)
        self._lock = threading.Lock()
        self._calls = 0
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.second_started = threading.Event()

    def extract_evidence(self, context, *, deadline_monotonic=None):
        with self._lock:
            self._calls += 1
            call = self._calls
        if call == 1:
            self.first_started.set()
            self.release_first.wait(timeout=2)
        else:
            self.second_started.set()
        return super().extract_evidence(context, deadline_monotonic=deadline_monotonic)


def _facts(
    root: str = "observed:runtimeerror:test",
    *,
    outcome: str = "terminal",
    failure_iteration: int | None = None,
    source: AttemptFailureFactsSource = AttemptFailureFactsSource.L0_DETERMINISTIC,
) -> AttemptFailureFacts:
    return AttemptFailureFacts(
        source=source,
        fine_class="observed_exception",
        root_fingerprint=root,
        root_fingerprint_source="test_fixture",
        fault_outcome=outcome,
        failure_iteration=failure_iteration,
    )


def _progress(
    completed: int | None = None,
    checkpoint: int | None = None,
) -> AttemptProgressSummary:
    return AttemptProgressSummary(
        training_progress="observed" if completed is not None else "not_observed",
        first_completed_step=completed,
        last_completed_step=completed,
        completed_step_delta=0 if completed is not None else None,
        progress_marker_count=1 if completed is not None else 0,
        checkpoint_progress="observed" if checkpoint is not None else "not_observed",
        first_checkpoint_step=checkpoint,
        last_checkpoint_step=checkpoint,
        checkpoint_step_delta=0 if checkpoint is not None else None,
        checkpoint_marker_count=1 if checkpoint is not None else 0,
    )


def _record(
    job_id: str,
    cycle_id: int,
    *,
    root: str = "observed:runtimeerror:test",
    completed: int | None = None,
    checkpoint: int | None = None,
    failure_iteration: int | None = None,
) -> AttemptRecord:
    return AttemptRecord(
        job_id=job_id,
        cycle_id=cycle_id,
        progress=_progress(completed, checkpoint),
        deterministic=_facts(root, failure_iteration=failure_iteration),
    )


def test_store_replaces_same_attempt_without_recurrence_inflation():
    store = InMemoryAttemptRecordStore()

    store.upsert_attempt(_record("job-1", 1, completed=10))
    store.upsert_attempt(_record("job-1", 1, completed=20))

    records = store.records("job-1")
    assert len(records) == 1
    assert records[0].progress.last_completed_step == 20


def test_store_applies_per_job_then_total_retention():
    store = InMemoryAttemptRecordStore(max_attempts_per_job=2, max_total_records=3)
    store.upsert_attempt(_record("job-a", 1))
    store.upsert_attempt(_record("job-a", 2))
    store.upsert_attempt(_record("job-a", 3))
    assert [item.cycle_id for item in store.records("job-a")] == [2, 3]

    store.upsert_attempt(_record("job-b", 1))
    store.upsert_attempt(_record("job-c", 1))

    assert [(item.job_id, item.cycle_id) for item in store.records()] == [
        ("job-a", 3),
        ("job-b", 1),
        ("job-c", 1),
    ]
    assert store.metrics()["eviction_count"] == 2


def test_prior_view_is_exact_job_ordered_and_excludes_current_cycle():
    store = InMemoryAttemptRecordStore()
    for record in (
        _record("job-1", 3),
        _record("job-2", 1),
        _record("job-1", 1),
        _record("job-1", 2),
    ):
        store.upsert_attempt(record)

    view = store.get_prior_attempts("job-1", 3)

    assert view.available is True
    assert view.availability_reason == "ready"
    assert [item.cycle_id for item in view.records] == [1, 2]


def test_disabled_store_rejects_explicit_seed():
    control = AttemptRecordControl(NullAttemptRecordStore())

    with pytest.raises(ValueError, match="history_disabled"):
        control.seed([_record("job-1", 1)])


def test_attempt_record_control_supports_replace_merge_inspect_and_clear():
    control = AttemptRecordControl(InMemoryAttemptRecordStore())

    control.seed([_record("job-a", 1), _record("job-b", 1)])
    control.seed([_record("job-a", 2)], mode="merge")
    assert [(item.job_id, item.cycle_id) for item in control.records()] == [
        ("job-a", 1),
        ("job-a", 2),
        ("job-b", 1),
    ]

    control.clear("job-a")
    assert [(item.job_id, item.cycle_id) for item in control.records()] == [("job-b", 1)]
    control.clear()
    assert control.records() == ()


def test_prior_view_remains_immutable_after_store_replacement():
    store = InMemoryAttemptRecordStore()
    store.upsert_attempt(_record("job-1", 1, completed=10))
    view = store.get_prior_attempts("job-1", 2)

    store.upsert_attempt(_record("job-1", 1, completed=20))

    assert view.records[0].progress.last_completed_step == 10
    assert store.records("job-1")[0].progress.last_completed_step == 20


def test_fixture_normalization_revalidates_typed_records_and_unknown_fields():
    payload = _record("job-1", 1).to_payload()
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="unsupported fields"):
        normalize_attempt_records([payload])


def test_l3_reports_conflicting_positive_progress_dimensions_as_unknown():
    prior = _record("job-1", 1, completed=10, checkpoint=20)
    current = _record("job-1", 2, completed=11, checkpoint=19)

    summary = evaluate_history(
        HistoryEvaluationInput(
            current_record=current,
            fact_selector=DETERMINISTIC_FACT_SELECTOR,
            prior_attempts=PriorAttemptView(
                records=(prior,),
                available=True,
                availability_reason="ready",
            ),
        )
    )

    comparison = summary.comparisons[0]
    assert comparison.selected_basis == "completed_step_and_checkpoint_step"
    assert comparison.positive_progress_conflict is True
    assert comparison.relation == "unknown"
    assert summary.no_observed_advance_attempts == 0


def test_l3_uses_deterministic_identity_for_prior_records_in_mvp():
    assembler = AttemptRecordAssembler()
    prior = assembler.with_enriched(
        _record("job-1", 1, root="deterministic:prior", completed=10),
        route_id="gpt",
        facts=_facts("enriched:shared", source=AttemptFailureFactsSource.L2_GROUNDED),
    )
    current = assembler.with_enriched(
        _record("job-1", 2, root="deterministic:current", completed=10),
        route_id="gpt",
        facts=_facts("enriched:shared", source=AttemptFailureFactsSource.L2_GROUNDED),
    )

    summary = evaluate_history(
        HistoryEvaluationInput(
            current_record=current,
            fact_selector="gpt",
            prior_attempts=PriorAttemptView(
                records=(prior,),
                available=True,
                availability_reason="ready",
            ),
        )
    )

    assert summary.same_job_attempts == 1
    assert summary.matching_root_attempts == 0


def test_attempt_record_assembler_replaces_one_enriched_route():
    record = _record("job-1", 1)
    assembler = AttemptRecordAssembler()
    enriched = AttemptFailureFacts(
        source=AttemptFailureFactsSource.L2_GROUNDED,
        fine_class="checkpoint_read_failure",
        root_fingerprint="observed:checkpoint:read",
        root_fingerprint_source="l2_grounded_primary",
        fault_outcome="terminal",
    )

    once = assembler.with_enriched(record, route_id="gpt", facts=enriched)
    twice = assembler.with_enriched(once, route_id="gpt", facts=enriched)

    assert len(twice.enriched) == 1
    assert twice.enriched[0].route_id == "gpt"


def test_runtime_publishes_l0_record_and_reuses_it_as_prior_history(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: CUDA initialization failed\n", encoding="utf-8")
    store = InMemoryAttemptRecordStore()
    runtime = RestartAgentRuntime(RestartAgent(), attempt_record_store=store)

    first = runtime.analyze_one(
        RestartAgentRequest(log_path=str(log_path), job_id="job-1", cycle_id=1)
    )
    second = runtime.analyze_one(
        RestartAgentRequest(log_path=str(log_path), job_id="job-1", cycle_id=2)
    )

    assert first.trace["runtime_history"]["initial_upserted"] is True
    assert second.trace["runtime_history"]["prior_attempt_count"] == 1
    assert second.result.retry_policy["matching_prior_failures"] == 1
    assert [item.cycle_id for item in store.records("job-1")] == [1, 2]


def test_runtime_reanalysis_replaces_same_cycle_and_clears_old_enrichment(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: CUDA initialization failed\n", encoding="utf-8")
    store = InMemoryAttemptRecordStore()
    base = _record("job-1", 1)
    enriched = AttemptRecordAssembler().with_enriched(
        base,
        route_id="old-route",
        facts=AttemptFailureFacts(
            source=AttemptFailureFactsSource.L2_GROUNDED,
            fine_class="old",
            root_fingerprint=base.deterministic.root_fingerprint,
            root_fingerprint_source="old",
            fault_outcome="terminal",
        ),
    )
    store.upsert_attempt(enriched)
    runtime = RestartAgentRuntime(RestartAgent(), attempt_record_store=store)

    runtime.analyze_one(RestartAgentRequest(log_path=str(log_path), job_id="job-1", cycle_id=1))

    records = store.records("job-1")
    assert len(records) == 1
    assert records[0].enriched == ()


def test_runtime_aggregates_completed_routes_into_one_attempt_record(tmp_path):
    line = "RuntimeError: CUDA initialization failed"
    log_path = tmp_path / "job.log"
    log_path.write_text(line + "\n", encoding="utf-8")
    store = InMemoryAttemptRecordStore()
    runtime = RestartAgentRuntime(RestartAgent(), attempt_record_store=store)
    routes = (
        ModelRoute(route_id="fast", evidence_extractor=_EvidenceExtractor(line)),
        ModelRoute(route_id="deep", evidence_extractor=_EvidenceExtractor(line)),
    )

    run = runtime.analyze_many(
        RestartAgentRequest(log_path=str(log_path), job_id="job-1", cycle_id=1),
        routes,
    )

    record = store.records("job-1")[0]
    assert [entry.route_id for entry in record.enriched] == ["deep", "fast"]
    assert set(run.trace["runtime_history"]["enriched_route_updates"]) == {"fast", "deep"}
    assert run.trace["runtime_history"]["store_before"]["record_count"] == 0
    assert run.trace["runtime_history"]["store_after"]["record_count"] == 1


def test_runtime_disabled_history_assembles_but_does_not_persist_record(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: CUDA initialization failed\n", encoding="utf-8")
    runtime = RestartAgentRuntime(
        RestartAgent(),
        attempt_record_store=NullAttemptRecordStore(),
    )

    run = runtime.analyze_one(
        RestartAgentRequest(log_path=str(log_path), job_id="job-1", cycle_id=1)
    )

    runtime_history = run.trace["runtime_history"]
    assert runtime_history["upsert_reason"] == "history_disabled"
    assert runtime_history["initial_upserted"] is False
    assert runtime_history["current_attempt_record"]["cycle_id"] == 1
    assert runtime.attempt_record_control.records() == ()


@pytest.mark.parametrize(
    ("request_kwargs", "reason"),
    (
        ({"cycle_id": 1}, "missing_job_id"),
        ({"job_id": "job-1"}, "missing_cycle_id"),
    ),
)
def test_runtime_missing_attempt_identity_is_explicit_and_not_stored(
    tmp_path,
    request_kwargs,
    reason,
):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: CUDA initialization failed\n", encoding="utf-8")
    store = InMemoryAttemptRecordStore()
    runtime = RestartAgentRuntime(RestartAgent(), attempt_record_store=store)

    run = runtime.analyze_one(RestartAgentRequest(log_path=str(log_path), **request_kwargs))

    runtime_history = run.trace["runtime_history"]
    assert runtime_history["availability_reason"] == reason
    assert runtime_history["initial_upserted"] is False
    assert runtime_history["current_attempt_record"] is None
    assert store.records() == ()


def test_runtime_does_not_store_attempt_without_deterministic_fingerprint(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("training process exited normally\n", encoding="utf-8")
    store = InMemoryAttemptRecordStore()
    runtime = RestartAgentRuntime(RestartAgent(), attempt_record_store=store)

    run = runtime.analyze_one(
        RestartAgentRequest(log_path=str(log_path), job_id="job-1", cycle_id=1)
    )

    runtime_history = run.trace["runtime_history"]
    assert runtime_history["upsert_reason"] == "missing_root_fingerprint"
    assert runtime_history["initial_upserted"] is False
    assert runtime_history["current_attempt_record"] is not None
    assert store.records() == ()


def test_runtime_reports_l0_not_ready_when_log_is_unavailable(tmp_path):
    log_path = tmp_path / "missing.log"
    runtime = RestartAgentRuntime(
        RestartAgent(),
        attempt_record_store=InMemoryAttemptRecordStore(),
    )

    run = runtime.analyze_one(
        RestartAgentRequest(log_path=str(log_path), job_id="job-1", cycle_id=1)
    )

    runtime_history = run.trace["runtime_history"]
    assert runtime_history["availability_reason"] == "ready"
    assert runtime_history["upsert_reason"] == "l0_not_ready"
    assert runtime_history["initial_upserted"] is False


def test_runtime_does_not_accept_route_enrichment_after_deadline(tmp_path):
    line = "RuntimeError: CUDA initialization failed"
    log_path = tmp_path / "job.log"
    log_path.write_text(line + "\n", encoding="utf-8")
    store = InMemoryAttemptRecordStore()
    runtime = RestartAgentRuntime(RestartAgent(), attempt_record_store=store)

    run = runtime.analyze_many(
        RestartAgentRequest(log_path=str(log_path), job_id="job-1", cycle_id=1),
        (ModelRoute(route_id="slow", evidence_extractor=_SlowEvidenceExtractor(line)),),
        timeout_seconds=0.01,
    )
    time.sleep(0.15)

    assert run.result.model_results[0].execution_status == "deadline_exceeded"
    assert store.records("job-1")[0].enriched == ()
    assert run.trace["runtime_history"]["enriched_updates"] == 0


def test_runtime_serializes_same_attempt_invocations(tmp_path):
    line = "RuntimeError: CUDA initialization failed"
    log_path = tmp_path / "job.log"
    log_path.write_text(line + "\n", encoding="utf-8")
    extractor = _BlockingFirstEvidenceExtractor(line)
    runtime = RestartAgentRuntime(
        RestartAgent(evidence_extractor=extractor),
        attempt_record_store=InMemoryAttemptRecordStore(),
    )
    request = RestartAgentRequest(log_path=str(log_path), job_id="job-1", cycle_id=1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(runtime.analyze_one, request)
        assert extractor.first_started.wait(timeout=1)
        second = pool.submit(runtime.analyze_one, request)
        assert not extractor.second_started.wait(timeout=0.05)
        extractor.release_first.set()
        first.result(timeout=2)
        assert extractor.second_started.wait(timeout=1)
        second.result(timeout=2)

    assert len(runtime.attempt_record_control.records("job-1")) == 1


def test_config_history_defaults_and_overrides_are_closed():
    base = {
        "schema_version": "restart_agent_config.v1",
        "config_id": "test",
        "config_version": 1,
        "model_routes": [
            {
                "route_id": "one",
                "model": "test/model",
                "base_url": "https://example.invalid/v1",
                "credential_ref": "TEST_KEY_FILE",
            }
        ],
    }
    environment = {"TEST_KEY_FILE": "/private/tmp/test-key"}

    defaults = parse_restart_agent_config(base, environ=environment)
    overridden = parse_restart_agent_config(
        {
            **base,
            "runtime": {
                "history": {
                    "enabled": False,
                    "max_attempts_per_job": 4,
                    "max_total_records": 20,
                }
            },
        },
        environ=environment,
    )

    assert defaults.history.enabled is True
    assert defaults.history.max_attempts_per_job == 10
    assert defaults.history.max_total_records == 3000
    assert overridden.history.enabled is False
    assert overridden.history.max_attempts_per_job == 4
    assert overridden.history.max_total_records == 20
    with pytest.raises(ValueError, match="runtime.history has unsupported fields"):
        parse_restart_agent_config(
            {**base, "runtime": {"history": {"ttl_seconds": 10}}},
            environ=environment,
        )


def test_cli_round_trips_plain_attempt_record_fixture(tmp_path, capsys):
    log_path = tmp_path / "job.log"
    output_path = tmp_path / "attempts.json"
    log_path.write_text("RuntimeError: CUDA initialization failed\n", encoding="utf-8")

    assert (
        cli_main(
            [
                str(log_path),
                "--job-id",
                "job-1",
                "--cycle-id",
                "1",
                "--attempt-records-json-out",
                str(output_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    first_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(first_payload, list)
    assert first_payload[0]["cycle_id"] == 1

    assert (
        cli_main(
            [
                str(log_path),
                "--job-id",
                "job-1",
                "--cycle-id",
                "2",
                "--attempt-records-json-in",
                str(output_path),
                "--attempt-records-json-out",
                str(output_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert [item["cycle_id"] for item in json.loads(output_path.read_text())] == [1, 2]
