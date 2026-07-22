# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

import pytest

from nvidia_resiliency_ext.attribution.restart_agent import (
    Decision,
    DecisionBasis,
    ModelRoute,
    RestartAgentRequest,
)
from nvidia_resiliency_ext.attribution.restart_agent.agent_runtime import RestartAgentRuntime
from nvidia_resiliency_ext.attribution.restart_agent.attempt_records import (
    InMemoryAttemptRecordStore,
)
from nvidia_resiliency_ext.attribution.restart_agent.causality import build_result_cascades
from nvidia_resiliency_ext.attribution.restart_agent.cli import main as cli_main
from nvidia_resiliency_ext.attribution.restart_agent.config import (
    build_model_routes,
    load_restart_agent_config,
    parse_restart_agent_config,
)
from nvidia_resiliency_ext.attribution.restart_agent.identity import (
    build_experimental_failure_identity,
    canonical_observed_fingerprint,
)
from nvidia_resiliency_ext.attribution.restart_agent.infrastructure.l0_publisher import (
    L0ArtifactPublisher,
)
from nvidia_resiliency_ext.attribution.restart_agent.infrastructure.live_artifacts import (
    LiveArtifactWriter,
)
from nvidia_resiliency_ext.attribution.restart_agent.infrastructure.log_source import LogSnapshot
from nvidia_resiliency_ext.attribution.restart_agent.infrastructure.route_publisher import (
    load_route_artifact_manifest,
)
from nvidia_resiliency_ext.attribution.restart_agent.l0 import build_l0_bundle
from nvidia_resiliency_ext.attribution.restart_agent.l0.codec import read_l0_bundle, write_l0_bundle
from nvidia_resiliency_ext.attribution.restart_agent.l0.decision import (
    build_decision_evidence,
    canonical_identity_anchor_line,
)
from nvidia_resiliency_ext.attribution.restart_agent.l0.projection import _window_preview_lines
from nvidia_resiliency_ext.attribution.restart_agent.l0.projection import (
    build_l0_model_facing_view as _build_l0_model_facing_view,
)
from nvidia_resiliency_ext.attribution.restart_agent.l1.openai_compatible import (
    L1EvidenceResult,
    LlmCallError,
    LlmConfig,
    LlmEvidenceExtractor,
    OpenAICompatibleTransport,
    RetryingChatTransport,
    _execute_tool_call,
    _initial_user_message,
    _is_context_window_exceeded_error,
    _is_http_timeout_status,
    _provider_reported_timing,
    _request_context_budget,
    _tool_loop_profile,
    _tool_schemas,
)
from nvidia_resiliency_ext.attribution.restart_agent.l1.prompts import SYSTEM_PROMPT
from nvidia_resiliency_ext.attribution.restart_agent.l1.response_contract import (
    L1_RESPONSE_CONTRACT,
    model_response_schema,
)
from nvidia_resiliency_ext.attribution.restart_agent.l1.tools import (
    LogTools,
    build_l1_evidence_context,
)
from nvidia_resiliency_ext.attribution.restart_agent.l1.validation import (
    model_evidence_contract_errors,
)
from nvidia_resiliency_ext.attribution.restart_agent.l2.grounding import (
    model_visible_line_numbers,
    model_visible_line_texts,
)
from nvidia_resiliency_ext.attribution.restart_agent.l3.history import DETERMINISTIC_FACT_SELECTOR
from nvidia_resiliency_ext.attribution.restart_agent.l3.history import (
    HistoryEvaluationInput as CoreHistoryEvaluationInput,
)
from nvidia_resiliency_ext.attribution.restart_agent.l3.history import evaluate_history
from nvidia_resiliency_ext.attribution.restart_agent.l4.policy import L4PolicyInput, evaluate_policy
from nvidia_resiliency_ext.attribution.restart_agent.models import (
    AssessmentStatus,
    AttemptFailureFacts,
    AttemptFailureFactsSource,
    AttemptProgressSummary,
    AttemptRecord,
    ContextWindow,
    FailureDomain,
    FailureDomainAssessment,
    FailureEvidence,
    HistorySummary,
    L0Bundle,
    LogLine,
    ModelRecoveryAssessment,
    NormalizedOccurrenceGroup,
    PriorAttemptView,
    ProgressFacts,
    RetryOutlookAssessment,
    RetryOutlookWithoutWorkloadChange,
    RetryPolicyConfig,
    normalize_attempt_records,
)
from nvidia_resiliency_ext.attribution.restart_agent.observability.trace_builder import (
    l1_token_limit_summary,
)
from nvidia_resiliency_ext.attribution.restart_agent.pipeline import (
    RestartAgent as CoreRestartAgent,
)


class RestartAgent(CoreRestartAgent):
    """Test helper that records each thread's invocation-owned run artifacts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_state = threading.local()

    def analyze(self, *args, **kwargs):
        if args and isinstance(args[0], dict) and "schema_version" not in args[0]:
            args = ({"schema_version": "restart_agent_request.v1", **args[0]}, *args[1:])
        run = self.run(*args, **kwargs)
        self._test_state.run = run
        return run.result

    def analyze_many(self, *args, **kwargs):
        if args and isinstance(args[0], dict) and "schema_version" not in args[0]:
            args = ({"schema_version": "restart_agent_request.v1", **args[0]}, *args[1:])
        return self.run_many(*args, **kwargs).result

    @property
    def last_trace(self):
        return self._test_state.run.trace

    @property
    def last_bundle(self):
        return self._test_state.run.bundle

    @property
    def last_decision_evidence(self):
        return self._test_state.run.decision_evidence

    @property
    def last_model_view(self):
        return self._test_state.run.model_view


def build_l0_model_facing_view(bundle):
    return _build_l0_model_facing_view(bundle, build_decision_evidence(bundle))


def _canonical_identity_anchor_line(bundle, line):
    return canonical_identity_anchor_line(bundle, line, selection_label="model_primary")


def _attempt_records(*records):
    return normalize_attempt_records(tuple(_attempt_record_fixture(item) for item in records))


def _attempt_record_fixture(value):
    completed = value.get("last_completed_step")
    checkpoint = value.get("last_checkpoint_step")
    return AttemptRecord(
        job_id=value["job_id"],
        cycle_id=value["cycle_id"],
        progress=AttemptProgressSummary(
            training_progress="observed" if completed is not None else "unknown",
            first_completed_step=completed,
            last_completed_step=completed,
            completed_step_delta=0 if completed is not None else None,
            progress_marker_count=1 if completed is not None else 0,
            checkpoint_progress="observed" if checkpoint is not None else "unknown",
            first_checkpoint_step=checkpoint,
            last_checkpoint_step=checkpoint,
            checkpoint_step_delta=0 if checkpoint is not None else None,
            checkpoint_marker_count=1 if checkpoint is not None else 0,
        ),
        deterministic=AttemptFailureFacts(
            source=AttemptFailureFactsSource.L0_DETERMINISTIC,
            fine_class=value.get("fine_class"),
            root_fingerprint=value.get("root_fingerprint"),
            root_fingerprint_source=value.get("root_fingerprint_source", "test_fixture"),
            fault_outcome=value.get("primary_fault_outcome"),
            failure_iteration=value.get("failure_iteration"),
            data_position_fingerprint=value.get("data_position_fingerprint"),
            artifact_path=value.get("artifact_path"),
            faulting_rank=value.get("faulting_rank"),
            faulting_node=value.get("faulting_node"),
            faulting_gpu=value.get("faulting_gpu"),
            rank_to_gpu_map=value.get("rank_to_gpu_map", {}),
        ),
    )


def _recovery_assessment(
    *,
    domain: str,
    outlook: str,
    domain_status: Optional[str] = None,
    outlook_status: Optional[str] = None,
) -> ModelRecoveryAssessment:
    domain_status = domain_status or (
        AssessmentStatus.UNKNOWN.value
        if domain == FailureDomain.UNKNOWN.value
        else AssessmentStatus.SUPPORTED_BUT_UNCONFIRMED.value
    )
    outlook_status = outlook_status or (
        AssessmentStatus.UNKNOWN.value
        if outlook == RetryOutlookWithoutWorkloadChange.UNKNOWN.value
        else AssessmentStatus.SUPPORTED_BUT_UNCONFIRMED.value
    )
    return ModelRecoveryAssessment(
        failure_domain=FailureDomainAssessment(
            value=FailureDomain(domain),
            status=AssessmentStatus(domain_status),
            confidence=50,
        ),
        retry_outlook_without_workload_change=RetryOutlookAssessment(
            value=RetryOutlookWithoutWorkloadChange(outlook),
            status=AssessmentStatus(outlook_status),
            confidence=50,
        ),
        rationale="typed policy test input",
    )


def _attempt_facts_and_progress(
    primary: FailureEvidence,
    progress: ProgressFacts,
) -> tuple[AttemptFailureFacts, AttemptProgressSummary]:
    facts = AttemptFailureFacts(
        source=AttemptFailureFactsSource.L0_DETERMINISTIC,
        fine_class=primary.fine_class,
        root_fingerprint=primary.root_fingerprint,
        root_fingerprint_source=primary.root_fingerprint_source,
        fault_outcome=primary.fault_outcome,
        primary_line=primary.line,
        failure_iteration=primary.failure_iteration,
        data_position_fingerprint=primary.data_position_fingerprint,
        faulting_rank=primary.rank,
        faulting_node=primary.node,
        faulting_gpu=primary.gpu,
    )
    completed = progress.highest_completed_step
    checkpoint = progress.last_checkpoint_step
    return facts, AttemptProgressSummary(
        training_progress="observed" if completed is not None else "unknown",
        first_completed_step=completed,
        last_completed_step=completed,
        completed_step_delta=0 if completed is not None else None,
        progress_marker_count=1 if completed is not None else 0,
        checkpoint_progress="observed" if checkpoint is not None else "unknown",
        first_checkpoint_step=checkpoint,
        last_checkpoint_step=checkpoint,
        checkpoint_step_delta=0 if checkpoint is not None else None,
        checkpoint_marker_count=1 if checkpoint is not None else 0,
    )


def _history_evaluation_input(*, current_attempt, job_id, cycle_id, prior_records):
    facts, progress = current_attempt
    current_record = AttemptRecord(
        job_id=job_id,
        cycle_id=cycle_id,
        progress=progress,
        deterministic=facts,
    )
    prior = tuple(
        record for record in prior_records if record.job_id == job_id and record.cycle_id < cycle_id
    )
    return CoreHistoryEvaluationInput(
        current_record=current_record,
        fact_selector=DETERMINISTIC_FACT_SELECTOR,
        prior_attempts=PriorAttemptView(
            records=prior,
            available=True,
            availability_reason="ready",
        ),
    )


def _prior_attempt_view(*records):
    return PriorAttemptView(
        records=_attempt_records(*records),
        available=True,
        availability_reason="ready",
    )


def _current_evidence(value):
    fixture = dict(value)
    old_primary = dict(fixture.get("primary_failure") or {})
    existing_root = fixture.get("root_cause_assessment")
    existing_related = fixture.get("related_failures")
    existing_assessment = dict(
        fixture.get("model_recovery_assessment") or fixture.get("model_policy_assessment") or {}
    )
    policy_class = existing_assessment.get(
        "policy_class",
        old_primary.get("policy_class", "ambiguous"),
    )
    primary_line = old_primary.get("line")
    primary = None
    if primary_line is not None:
        old_identity = old_primary.get("failure_identity")
        identity = dict(old_identity) if isinstance(old_identity, dict) else {}
        primary = {
            "line": primary_line,
            "causal_role": old_primary.get("causal_role") or "initiating",
            "failure_identity": {
                "operation": identity.get("operation"),
                "mechanism": identity.get("mechanism") or old_primary.get("fine_class"),
                "component": identity.get("component"),
                "artifact_path": identity.get("artifact_path"),
            },
        }
    related = []
    for item in [
        *(fixture.get("secondary_failures", []) or []),
        *(fixture.get("cascades", []) or []),
    ]:
        if not isinstance(item, dict) or item.get("line") is None:
            continue
        related.append(
            {
                "line": item["line"],
                "causal_role": item.get("causal_role") or "unknown",
                "rationale": "Related failure from the test fixture.",
            }
        )
    root_cause = (
        dict(existing_root)
        if isinstance(existing_root, dict)
        else {
            "summary": fixture.get("justification") or "Test root cause.",
            "plausible_causes": ["test cause"],
        }
    )
    root_cause = {
        "summary": root_cause.get("summary") or "Test root cause.",
        "status": root_cause.get("status") or "established_by_current_log",
        "plausible_causes": list(root_cause.get("plausible_causes") or []),
        "missing_evidence": list(root_cause.get("missing_evidence") or []),
    }
    analysis_status = fixture.get("analysis_status") or (
        "primary_identified" if primary is not None else "insufficient_evidence"
    )
    if primary is None:
        missing_evidence = root_cause["missing_evidence"]
        if analysis_status == "insufficient_evidence" and not missing_evidence:
            missing_evidence = ["test fixture did not identify a primary failure"]
        if analysis_status == "no_failure_observed":
            summary = L1_RESPONSE_CONTRACT.no_failure_summary
            rationale = L1_RESPONSE_CONTRACT.no_failure_rationale
        else:
            summary = L1_RESPONSE_CONTRACT.insufficient_summary
            rationale = L1_RESPONSE_CONTRACT.insufficient_rationale
        return {
            "schema_version": "restart_agent_evidence.v1",
            "analysis_status": analysis_status,
            "primary_failure": None,
            "root_cause_assessment": {
                "summary": summary,
                "status": "unknown",
                "plausible_causes": [],
                "missing_evidence": (
                    [] if analysis_status == "no_failure_observed" else missing_evidence
                ),
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
                "rationale": rationale,
            },
            "related_failures": [],
            "evidence": [],
        }
    persistence = existing_assessment.get(
        "current_attempt_persistence_evidence",
        "affirmative" if policy_class == "user_failure" else "none",
    )
    old_recovery_path = existing_assessment.get("retry_recovery_path", "unknown")
    retry_outlook = existing_assessment.get("retry_outlook_without_workload_change")
    retry_outlook_mapping = retry_outlook if isinstance(retry_outlook, dict) else None
    if retry_outlook is None:
        retry_outlook = (
            "cannot_recover"
            if old_recovery_path in {"workload_change", "external_intervention"}
            else (
                "may_recover"
                if old_recovery_path
                in {"ordinary_retry", "wait_or_teardown", "workload_managed_retry_grace"}
                else (
                    "cannot_recover"
                    if policy_class == "user_failure"
                    else "may_recover" if policy_class == "not_user_failure" else "unknown"
                )
            )
        )
    domain = existing_assessment.get("failure_domain")
    domain_mapping = domain if isinstance(domain, dict) else None
    if domain_mapping is None:
        domain = domain or (
            "workload"
            if policy_class == "user_failure"
            else "infrastructure" if policy_class == "not_user_failure" else "unknown"
        )
    claim_status = (
        "established_by_current_log"
        if persistence == "affirmative"
        else "unknown" if domain == "unknown" else "supported_but_unconfirmed"
    )
    raw_evidence = list(fixture.get("evidence") or [])
    evidence = []
    for index, item in enumerate(raw_evidence, start=1):
        if not isinstance(item, dict):
            continue
        supports = item.get("supports")
        if not isinstance(supports, list):
            supports = [supports] if isinstance(supports, str) else []
        if primary is not None and item.get("line") == primary_line:
            supports = [
                "primary_failure",
                "root_cause_assessment",
                "failure_domain",
                "retry_outlook_without_workload_change",
            ]
        else:
            supports = [
                tag
                for tag in supports
                if tag
                in {
                    "primary_failure",
                    "root_cause_assessment",
                    "failure_domain",
                    "retry_outlook_without_workload_change",
                }
            ]
        evidence.append(
            {
                "id": item.get("id") or f"e{index}",
                "line": item.get("line"),
                "quote": item.get("quote") or "test evidence",
                "supports": supports or ["root_cause_assessment"],
            }
        )
    return {
        "schema_version": "restart_agent_evidence.v1",
        "analysis_status": analysis_status,
        "primary_failure": primary,
        "root_cause_assessment": root_cause,
        "model_recovery_assessment": {
            "failure_domain": domain_mapping
            or {
                "value": domain,
                "status": claim_status,
                "confidence": existing_assessment.get("confidence", 90),
            },
            "retry_outlook_without_workload_change": retry_outlook_mapping
            or {
                "value": retry_outlook,
                "status": ("unknown" if retry_outlook == "unknown" else claim_status),
                "confidence": existing_assessment.get("confidence", 90),
            },
            "rationale": existing_assessment.get("rationale")
            or fixture.get("justification")
            or "Test recovery assessment.",
        },
        "related_failures": (existing_related if isinstance(existing_related, list) else related),
        "evidence": evidence,
    }


class _FakeEvidenceExtractor:
    def __init__(
        self,
        evidence,
        *,
        success=True,
        malformed=False,
        errors=(),
        anomalies=None,
        finish_reason="stop",
        transcript_events=(),
    ):
        self.evidence = _current_evidence(evidence)
        self.success = success
        self.malformed = malformed
        self.errors = tuple(errors)
        self.anomalies = dict(anomalies or {})
        self.finish_reason = finish_reason
        self.transcript_events = tuple(transcript_events)
        self.model_view = None

    def extract_evidence(self, context, *, deadline_monotonic=None):
        self.model_view = context.model_view
        return L1EvidenceResult(
            evidence=self.evidence,
            model="fake-model",
            raw_model_output=json.dumps(self.evidence),
            success=self.success,
            malformed=self.malformed,
            errors=self.errors,
            model_calls=(
                {
                    "layer": "L1",
                    "model": "fake-model",
                    "success": self.success,
                    "latency_s": 0.001,
                    "finish_reason": self.finish_reason,
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "prompt_tokens_details": {"cached_tokens": 3},
                        "completion_tokens_details": {"reasoning_tokens": 2},
                    },
                },
            ),
            anomalies=self.anomalies,
            transcript_events=self.transcript_events,
        )


class _BlockingEvidenceExtractor(_FakeEvidenceExtractor):
    def __init__(self, evidence):
        super().__init__(evidence)
        self.started = threading.Event()
        self.release = threading.Event()
        self.completed = threading.Event()

    def extract_evidence(self, context, *, deadline_monotonic=None):
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release the L1 extractor")
        result = super().extract_evidence(
            context,
            deadline_monotonic=deadline_monotonic,
        )
        self.completed.set()
        return result


class _RetryOnceEvidenceExtractor(LlmEvidenceExtractor):
    def __init__(self, evidence):
        super().__init__(
            LlmConfig(
                api_key="test-key",
                max_retries=1,
                retry_backoff_seconds=0.0,
                tools_enabled=False,
            )
        )
        self.evidence = _current_evidence(evidence)
        self.calls = 0

    def _call_model(
        self,
        *,
        api_key,
        messages,
        include_tools,
        model_turn,
        attempt=1,
        max_retries=None,
        deadline_monotonic=None,
    ):
        self.calls += 1
        if self.calls == 1:
            raise LlmCallError(
                "HTTP 502: bad gateway",
                {
                    "layer": "L1",
                    "model": "nvidia/qwen/qwen3.5-35b-a3b",
                    "model_turn": model_turn,
                    "attempt": attempt,
                    "max_retries": max_retries,
                    "success": False,
                    "latency_s": 0.01,
                    "finish_reason": None,
                    "usage": None,
                    "tools_advertised": include_tools,
                    "error_type": "http_error",
                    "error": "HTTP 502",
                    "http_status": 502,
                    "response_body": "bad gateway",
                    "retryable": True,
                    "retry_scheduled": False,
                    "timeout": False,
                },
            )
        return (
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(self.evidence)},
                    }
                ],
                "usage": {"total_tokens": 100},
            },
            {
                "layer": "L1",
                "model": "nvidia/qwen/qwen3.5-35b-a3b",
                "model_turn": model_turn,
                "attempt": attempt,
                "max_retries": max_retries,
                "success": True,
                "latency_s": 0.02,
                "finish_reason": "stop",
                "usage": {"total_tokens": 100},
                "tools_advertised": include_tools,
            },
        )


class _ContractRepairEvidenceExtractor(LlmEvidenceExtractor):
    def __init__(self, complete_evidence):
        super().__init__(
            LlmConfig(
                api_key="test-key",
                max_retries=0,
                tools_enabled=False,
            )
        )
        self.complete_evidence = _current_evidence(complete_evidence)
        self.calls = 0

    def _call_model(
        self,
        *,
        api_key,
        messages,
        include_tools,
        model_turn,
        attempt=1,
        max_retries=None,
        deadline_monotonic=None,
    ):
        self.calls += 1
        content = (
            json.dumps(
                {
                    "primary_failure": self.complete_evidence["primary_failure"],
                    "related_failures": [],
                }
            )
            if self.calls == 1
            else json.dumps(self.complete_evidence)
        )
        return (
            {
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "usage": {"total_tokens": 100},
            },
            {
                "layer": "L1",
                "model": "nvidia/qwen/qwen3.5-35b-a3b",
                "model_turn": model_turn,
                "attempt": attempt,
                "max_retries": max_retries,
                "success": True,
                "latency_s": 0.01,
                "finish_reason": "stop",
                "usage": {"total_tokens": 100},
                "tools_advertised": include_tools,
            },
        )


class _SingleResponseEvidenceExtractor(LlmEvidenceExtractor):
    def __init__(self, evidence):
        super().__init__(
            LlmConfig(
                api_key="test-key",
                max_retries=0,
                tools_enabled=False,
            )
        )
        self.evidence = _current_evidence(evidence)
        self.calls = 0

    def _call_model(
        self,
        *,
        api_key,
        messages,
        include_tools,
        model_turn,
        attempt=1,
        max_retries=None,
        deadline_monotonic=None,
    ):
        self.calls += 1
        return (
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(self.evidence)},
                    }
                ],
                "usage": {"total_tokens": 100},
            },
            {
                "layer": "L1",
                "model": "nvidia/qwen/qwen3.5-35b-a3b",
                "model_turn": model_turn,
                "attempt": attempt,
                "max_retries": max_retries,
                "success": True,
                "latency_s": 0.01,
                "finish_reason": "stop",
                "usage": {"total_tokens": 100},
                "tools_advertised": include_tools,
            },
        )


def test_l0_terminal_episode_does_not_claim_unobserved_prior_progress(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "Traceback (most recent call last):\nUnicodeDecodeError: invalid byte\n",
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))

    assert bundle.failure_episodes
    assert bundle.failure_episodes[0].last_progress_before is None
    assert "no prior progress marker was observed" in bundle.failure_episodes[0].reason
    assert bundle.post_fault_summaries[0].progress_after_observed is False


def test_l0_coverage_separates_failure_candidate_from_taxonomy_primary(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "ERROR: checkpoint writer unexpected position 704 vs 598\n",
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))

    assert bundle.candidate_anchors
    assert bundle.deterministic_primary_candidate is None
    assert bundle.evidence_coverage["first_failure_candidate"] == "found"
    assert bundle.evidence_coverage["deterministic_taxonomy_primary"] == "not_found"
    assert "first_fault" not in bundle.evidence_coverage


def test_canonical_observed_fingerprint_ignores_runtime_locality_and_iteration():
    first = canonical_observed_fingerprint(
        "180: [rank180]: RuntimeError: Rank 180, node nvl72005-T14, device 0, "
        "iteration 670314: Unexpected result inf in bucket #0"
    )
    second = canonical_observed_fingerprint(
        "22: [rank22]: RuntimeError: Rank 22, node nvl99999-T01, device 7, "
        "iteration 670315: Unexpected result inf in bucket #9"
    )

    assert first == second


def test_observed_fingerprint_excludes_conditional_diagnostic_hypotheses():
    shared_memory = canonical_observed_fingerprint(
        "RuntimeError: DataLoader worker is killed by signal: Bus error. "
        "It is possible that workers are out of shared memory. "
        "Please try to raise your shared memory limit."
    )
    storage = canonical_observed_fingerprint(
        "RuntimeError: DataLoader worker is killed by signal: Bus error. "
        "This may be caused by an intermittent storage read."
    )

    assert shared_memory == storage
    assert "shared_memory" not in str(shared_memory)
    assert "storage" not in str(storage)


def test_llm_default_output_ceiling_is_64k():
    assert LlmConfig().max_output_tokens == 64_000


def test_prompt_exposes_exactly_two_l1_recovery_concepts(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("OSError: address already in use\n", encoding="utf-8")
    bundle = build_l0_bundle(str(log_path))
    model_view = build_l0_model_facing_view(bundle)
    user_payload = json.loads(_initial_user_message(model_view))
    normalized_system_prompt = " ".join(SYSTEM_PROMPT.split())

    assert "Return exactly these two current-attempt recovery claims" in SYSTEM_PROMPT
    assert "retry_outlook_without_workload_change" in SYSTEM_PROMPT
    assert "current_attempt_persistence_evidence" not in SYSTEM_PROMPT
    assert "recovery_requirement" not in SYSTEM_PROMPT
    assert "fixed request" in SYSTEM_PROMPT
    assert "missing cleanup message" in SYSTEM_PROMPT
    assert "Confidence is a 1..99 calibration signal" in SYSTEM_PROMPT
    assert "multi-rank fanout" in SYSTEM_PROMPT
    assert "Durable remediation" in SYSTEM_PROMPT
    assert "workload-selected framework or library behavior" in SYSTEM_PROMPT
    assert "Megatron" not in SYSTEM_PROMPT
    assert "CUDA_LAUNCH_BLOCKING" not in SYSTEM_PROMPT
    assert "NVLink" not in SYSTEM_PROMPT
    assert "namespace evidence" not in SYSTEM_PROMPT.lower()
    assert "Do not decide STOP or RESTART" not in SYSTEM_PROMPT
    assert "Do not use attempt history" not in SYSTEM_PROMPT
    assert "L0 evidence" not in SYSTEM_PROMPT
    assert "registry matches" not in SYSTEM_PROMPT

    assert "not a STOP/RESTART decision or policy input" not in SYSTEM_PROMPT
    assert "Prior progress proves runnability, not transience" in normalized_system_prompt
    assert "Replay distance and failure position do not establish" in normalized_system_prompt
    assert "Later aggregate progress" in normalized_system_prompt
    assert "Inspect supplied context before calling tools" in normalized_system_prompt
    assert set(user_payload) == {
        "attempt_execution_context",
        "decision_evidence",
        "evidence_bundle",
        "response_schema",
        "restart_environment_context",
    }
    assert user_payload["evidence_bundle"] == model_view.evidence_bundle
    assert "l0_bundle" not in user_payload
    assert "interpretation_constraints" not in user_payload["attempt_execution_context"]
    assert "task" not in user_payload
    assert "instructions" not in user_payload
    assert "available_tools" not in user_payload
    assert "tool_loop_profile" not in user_payload
    assert user_payload["restart_environment_context"] == {
        "workload_unchanged": True,
        "process_state_recreated": True,
        "normal_restart_delay_applies": True,
        "hardware_allocation_may_change": True,
        "external_service_state_may_change": True,
    }
    response_schema = user_payload["response_schema"]
    assessment_schema = response_schema["properties"]["model_recovery_assessment"]["properties"]
    assert response_schema["additionalProperties"] is False
    assert response_schema["properties"]["evidence"]["maxItems"] == 12
    assert response_schema["properties"]["related_failures"]["maxItems"] == 3
    assert (
        response_schema["semanticConstraints"]["no_failure_observed"]["recovery_claims"]
        == "unknown value, unknown status, confidence 1"
    )
    assert set(assessment_schema) == {
        "failure_domain",
        "retry_outlook_without_workload_change",
        "rationale",
    }
    assert set(assessment_schema["failure_domain"]["properties"]) == {
        "value",
        "status",
        "confidence",
    }
    assert set(assessment_schema["retry_outlook_without_workload_change"]["properties"]) == {
        "value",
        "status",
        "confidence",
    }


def test_l1_response_contract_single_sources_advertised_validation_constraints():
    schema = model_response_schema()
    properties = schema["properties"]
    evidence = properties["evidence"]
    claim = properties["model_recovery_assessment"]["properties"]["failure_domain"]

    assert set(schema["required"]) == L1_RESPONSE_CONTRACT.top_level_fields
    assert evidence["maxItems"] == L1_RESPONSE_CONTRACT.max_evidence_items
    assert set(evidence["items"]["properties"]["supports"]["items"]["enum"]) == (
        L1_RESPONSE_CONTRACT.evidence_support_tags
    )
    assert claim["properties"]["confidence"] == {
        "type": "integer",
        "minimum": L1_RESPONSE_CONTRACT.min_confidence,
        "maximum": L1_RESPONSE_CONTRACT.max_confidence,
        "description": "Calibration-only confidence in this claim.",
    }
    assert claim["semanticConstraint"] == (
        "value=unknown if and only if status=unknown; otherwise neither is unknown"
    )
    assert schema["semanticConstraints"]["evidence_ids"] == "unique non-empty strings"


@pytest.mark.parametrize("analysis_status", ["no_failure_observed", "insufficient_evidence"])
def test_l1_non_primary_contract_has_canonical_unknown_semantics(analysis_status):
    payload = _current_evidence(
        {
            "analysis_status": analysis_status,
            "justification": "The current log does not establish a primary failure.",
        }
    )

    assert model_evidence_contract_errors(payload) == []
    assert payload["primary_failure"] is None
    expected_summary = (
        L1_RESPONSE_CONTRACT.no_failure_summary
        if analysis_status == "no_failure_observed"
        else L1_RESPONSE_CONTRACT.insufficient_summary
    )
    assert payload["root_cause_assessment"]["summary"] == expected_summary
    assert payload["model_recovery_assessment"]["failure_domain"] == {
        "value": "unknown",
        "status": "unknown",
        "confidence": 1,
    }
    assert payload["related_failures"] == []
    assert payload["evidence"] == []


def test_l1_non_primary_contract_rejects_noncanonical_recovery_claim():
    payload = _current_evidence({"analysis_status": "no_failure_observed"})
    payload["model_recovery_assessment"]["failure_domain"]["confidence"] = 80

    assert model_evidence_contract_errors(payload) == [
        "non-primary failure_domain.confidence must be 1"
    ]


def test_l1_non_primary_contract_rejects_free_form_summary_and_rationale():
    payload = _current_evidence({"analysis_status": "no_failure_observed"})
    payload["root_cause_assessment"]["summary"] = "This is probably a workload bug."
    payload["model_recovery_assessment"]["rationale"] = "Stop the workload."

    assert model_evidence_contract_errors(payload) == [
        "non-primary root_cause_assessment.summary must be "
        f"{L1_RESPONSE_CONTRACT.no_failure_summary!r}",
        "non-primary model_recovery_assessment.rationale must be "
        f"{L1_RESPONSE_CONTRACT.no_failure_rationale!r}",
    ]


def test_l2_visibility_uses_exact_full_model_visible_payload(tmp_path):
    log_path = tmp_path / "job.log"
    failure_line = "RuntimeError: invalid configuration"
    log_path.write_text(failure_line + "\n", encoding="utf-8")
    model_view = build_l0_model_facing_view(build_l0_bundle(str(log_path)))
    result = L1EvidenceResult(
        evidence=None,
        model="test-model",
        transcript_events=(
            {
                "event_type": "bundle_snapshot",
                "model_visible_payload": {
                    "decision_evidence": {
                        "deterministic_primary_candidate": {
                            "line": 1,
                            "quote": failure_line,
                        }
                    },
                    "distributed_failure_incidents": [
                        {
                            "primary_observed_line": 2,
                            "primary_observed_quote": "Watchdog observed timeout",
                        }
                    ],
                    "post_fault_summaries": [
                        {
                            "last_high_signal_line": 3,
                            "last_high_signal_quote": "scheduler cancelled step",
                        }
                    ],
                    "evidence_bundle": {},
                },
            },
        ),
    )

    assert model_visible_line_numbers(model_view, result) == {1, 2, 3}
    assert model_visible_line_texts(model_view, result) == {
        1: {failure_line},
        2: {"Watchdog observed timeout"},
        3: {"scheduler cancelled step"},
    }


def test_decision_evidence_references_are_explicitly_provenance_only(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: invalid configuration\n", encoding="utf-8")

    model_view = build_l0_model_facing_view(build_l0_bundle(str(log_path)))
    references = model_view.decision_evidence.selected_evidence_references

    assert references["semantics"] == "provenance_only"
    assert references["resolution"] == "get_evidence_objects_when_advertised"
    assert set(model_view.attempt_execution_context) == {"scope", "terminal_timing"}


def test_l1_contract_rejects_retired_recovery_fields():
    payload = _current_evidence(
        {
            "primary_failure": {
                "line": 1,
                "fine_class": "observed_exception",
                "causal_role": "initiating",
            },
            "model_recovery_assessment": {
                "failure_domain": "workload",
                "retry_outlook_without_workload_change": "cannot_recover",
            },
            "evidence": [
                {
                    "line": 1,
                    "quote": "RuntimeError: invalid configuration",
                    "supports": "primary_failure",
                }
            ],
        }
    )
    payload["model_recovery_assessment"]["current_attempt_persistence_evidence"] = "affirmative"

    assert model_evidence_contract_errors(payload) == [
        "model_recovery_assessment has unsupported fields: " "current_attempt_persistence_evidence"
    ]


def test_restart_environment_context_is_closed_and_boolean(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: failed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown restart_environment_context fields"):
        RestartAgent(restart_environment_context={"allocation_always_changes": True}).analyze(
            {"log_path": str(log_path)}
        )

    with pytest.raises(
        TypeError,
        match="restart_environment_context.hardware_allocation_may_change must be a boolean",
    ):
        RestartAgent(
            restart_environment_context={"hardware_allocation_may_change": "sometimes"}
        ).analyze({"log_path": str(log_path)})


def test_decision_evidence_is_shared_with_l0b_and_trace(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "7: iteration 418 completed",
                "7: RuntimeError: CUDA out of memory",
                "7: destroy_process_group() called during shutdown",
            ]
        ),
        encoding="utf-8",
    )
    bundle = build_l0_bundle(str(log_path))
    decision_evidence = build_decision_evidence(bundle)
    model_view = _build_l0_model_facing_view(bundle, decision_evidence)

    assert decision_evidence.schema_version == "restart_agent_decision_evidence.v1"
    assert decision_evidence.deterministic_primary_candidate is not None
    assert decision_evidence.deterministic_primary_candidate.line == 2
    assert decision_evidence.provenance == {
        "source": "l0a_deterministic_selection",
        "log_line_count": 3,
        "log_byte_size": log_path.stat().st_size,
        "log_rescanned": False,
        "model_used": False,
    }
    assert model_view.prompt_payload()["decision_evidence"] == decision_evidence.to_payload()
    assert model_view.prompt_payload()["evidence_bundle"] == model_view.evidence_bundle
    assert "l0_bundle" not in model_view.prompt_payload()

    analyzer = RestartAgent(
        evidence_extractor=_FakeEvidenceExtractor({}, success=False, malformed=True)
    )
    analyzer.analyze({"log_path": str(log_path)}, l0_bundle=bundle)

    assert analyzer.last_decision_evidence == decision_evidence
    assert analyzer.last_model_view is not None
    assert analyzer.last_model_view.decision_evidence is analyzer.last_decision_evidence
    assert analyzer.last_trace["decision_evidence"] == decision_evidence.to_payload()
    assert (
        analyzer.last_trace["l0_model_view"]["decision_evidence"]
        == analyzer.last_trace["decision_evidence"]
    )
    assert (
        analyzer.last_trace["layers"]["L0"]["sub_stages"]["DecisionEvidence"]["status"]
        == "completed"
    )


def test_l0_records_path_namespace_mismatch_and_distributed_exception_fanout(tmp_path):
    log_path = tmp_path / "job.log"
    lock_path = "/lustre/fsw/users/wdai/hf_home/datasets/cache.lock"
    log_path.write_text(
        "\n".join(
            [
                "0:   data_cache_path ........................ /lustre/fsw/users/rwaleffe/cache",
                "0:   load ................................... /lustre/fsw/users/wdai/checkpoint",
                "0: successfully loaded checkpoint at iteration 0",
                "80: [rank80]: Traceback (most recent call last):",
                f"80: [rank80]: PermissionError: [Errno 13] Permission denied: '{lock_path}'",
                f"96: [rank96]: PermissionError: [Errno 13] Permission denied: '{lock_path}'",
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    prompt_bundle = build_l0_model_facing_view(bundle).evidence_bundle

    assert bundle.path_namespace_summary["failed_vs_configured_write_mismatch"] is True
    assert bundle.path_namespace_summary["ownership_verified"] is False
    assert bundle.path_namespace_summary["namespaces_by_role"] == {
        "configured_write": ["rwaleffe"],
        "configured_read": ["wdai"],
        "failed_access": ["wdai"],
    }
    failed = next(item for item in bundle.path_access_facts if item["role"] == "failed_access")
    assert failed["access_intent"] == "write_or_create"
    assert failed["path"] == lock_path
    assert prompt_bundle["path_namespace_summary"] == bundle.path_namespace_summary
    assert prompt_bundle["path_access_facts"] == list(bundle.path_access_facts)

    permission_matches = [item for item in bundle.registry_matches if item.line in {5, 6}]
    assert any(item.registry_id == "filesystem_permission_denied" for item in permission_matches)
    assert all(item.registry_id != "user_config_error" for item in permission_matches)
    assert len(bundle.distributed_failure_incidents) == 1
    incident = bundle.distributed_failure_incidents[0]
    assert incident.incident_kind == "distributed_fanout"
    assert incident.incident_type == "distributed_exception_fanout"
    assert incident.member_event_lines == (5, 6)
    assert incident.event_count == 2
    assert incident.observed_rank_count == 2
    assert incident.interpretation == "same_attempt_rank_fanout_not_cross_cycle_recurrence"
    prompt_incident = prompt_bundle["distributed_failure_incidents"][0]
    assert prompt_incident["sample_lines"] == [5, 6]
    assert "member_event_lines" not in prompt_incident


def test_experimental_failure_identity_separates_family_and_concrete_fields():
    artifact = "/lustre/checkpoints/job-1/iter_5000/metadata"
    identity = build_experimental_failure_identity(
        "4175: [rank4175]: UnicodeDecodeError: 'utf-8' codec can't decode byte "
        "0xde in position 8: invalid continuation byte",
        (
            "4175: [rank4175]: Traceback (most recent call last):",
            f"INFO loading checkpoint {artifact}",
            '  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/'
            'checkpoint/metadata.py", line 42, in determine_global_metadata',
            '4175: [rank4175]:   File "/usr/local/lib/python3.12/dist-packages/'
            'torch/distributed/checkpoint/utils.py", line 9, in tensor_to_object',
            '5124: [rank5124]:   File "/workspace/cleanup.py", line 7, in cleanup',
        ),
        model_identity={
            "operation": "checkpoint_load",
            "mechanism": "metadata_deserialization",
            "component": None,
            "artifact_path": artifact,
        },
    )

    assert identity["policy_active"] is False
    assert identity["family"] == {
        "operation": "checkpoint_load",
        "mechanism": "metadata_deserialization",
        "exception_type": "unicode_decode_error",
        "label": "checkpoint_load|metadata_deserialization|unicode_decode_error",
        "fingerprint": identity["family"]["fingerprint"],
        "complete": True,
    }
    assert identity["concrete"]["component"] == "torch.distributed.checkpoint"
    assert identity["concrete"]["callsite"] == "tensor_to_object"
    assert identity["concrete"]["artifact_path"] == artifact
    assert identity["concrete"]["failure_position"] == "8"
    assert identity["concrete"]["complete"] is True
    assert identity["client_concrete"]["exception_type"] == "unicode_decode_error"
    assert identity["client_concrete"]["source_file"].endswith(
        "/torch/distributed/checkpoint/utils.py"
    )
    assert identity["client_concrete"]["callsite"] == "tensor_to_object"
    assert identity["client_concrete"]["artifact_path"] == artifact
    assert identity["client_concrete"]["failure_position"] == "8"
    assert identity["client_concrete"]["stack_path"] == [
        "determine_global_metadata",
        "tensor_to_object",
    ]
    assert identity["client_concrete"]["complete"] is True


def test_client_concrete_identity_captures_checkpoint_collective_context():
    artifact = "/checkpoints/phase1/"
    identity = build_experimental_failure_identity(
        (
            "594: [rank594]: Watchdog caught collective operation timeout: "
            "WorkNCCL(SeqNum=2, OpType=ALLGATHER, NumelIn=278465, "
            "NumelOut=106930560, Timeout(ms)=600000)"
        ),
        (
            f"0: loading distributed checkpoint from {artifact} at iteration 635000",
            (
                "0: /usr/local/lib/python3.12/dist-packages/torch/checkpoint.py:964: "
                "FutureWarning: load_state_dict is deprecated"
            ),
        ),
        observed_phase="checkpoint_load_start",
    )

    concrete = identity["client_concrete"]
    assert concrete["artifact_path"] == artifact
    assert concrete["checkpoint_iteration"] == 635000
    assert concrete["phase"] == "checkpoint_load_start"
    assert concrete["operation_signature"] == (
        "optype=allgather,seqnum=2,numelin=278465,numelout=106930560"
    )
    assert "checkpoint.py" not in concrete["label"]


def test_client_concrete_identity_preserves_observed_failure_iteration():
    identity = build_experimental_failure_identity(
        (
            "404: [rank404]: RuntimeError: Rank 404, device 0, iteration 337071: "
            "Unexpected result inf"
        ),
        (
            "404: [rank404]: Traceback (most recent call last):",
            '404: [rank404]:   File "/workspace/training.py", line 42, in validate_result',
        ),
        model_identity={
            "operation": "gradient_validation",
            "mechanism": "non_finite_gradient",
        },
        observed_phase="steady_mid",
    )

    assert identity["schema_version"] == "restart_agent_failure_identity.experimental.v1"
    assert identity["concrete"]["failure_iteration"] == 337071
    assert identity["client_concrete"]["failure_iteration"] == 337071
    assert "failure_iteration=337071" in identity["client_concrete"]["label"]


def test_observed_collective_identity_ignores_c10d_severity_date_prefix():
    first = (
        "115: [rank115]:[E207 19:04:38.123186550 ProcessGroupNCCL.cpp:697] "
        "[Rank 14] Watchdog caught collective operation timeout: "
        "WorkNCCL(SeqNum=2, OpType=ALLGATHER, NumelIn=306689, "
        "NumelOut=29442144, Timeout(ms)=600000)"
    )
    second = first.replace("E207 19:04:38.123186550", "E208 03:17:02.987654321")

    first_fingerprint = canonical_observed_fingerprint(first)
    second_fingerprint = canonical_observed_fingerprint(second)
    first_identity = build_experimental_failure_identity(first)
    second_identity = build_experimental_failure_identity(second)

    assert first_fingerprint == second_fingerprint
    assert "e207" not in (first_fingerprint or "")
    assert "e208" not in (second_fingerprint or "")
    assert (
        first_identity["client_concrete"]["fingerprint"]
        == second_identity["client_concrete"]["fingerprint"]
    )


def test_observed_transport_identity_ignores_timestamp_node_and_process_routing():
    line = (
        "1000: [2026-02-02 12:15:56] nvl72026-T11:393849:395082 [0] "
        "transport/net_ib.cc:253 NCCL WARN NET/IB : mlx5_3:1 "
        "Got non-fatal async event: port error(10)"
    )

    fingerprint = canonical_observed_fingerprint(line)

    assert fingerprint is not None
    assert "nvl72026" not in fingerprint
    assert "393849" not in fingerprint
    assert "transport_net_ib" in fingerprint


def test_client_identity_excludes_stale_checkpoint_context_for_steady_failure():
    identity = build_experimental_failure_identity(
        (
            "5995: [rank5995]: Watchdog caught collective operation timeout: "
            "WorkNCCL(SeqNum=126639, OpType=COALESCED, NumelIn=10, "
            "NumelOut=20, Timeout(ms)=600000)"
        ),
        (
            "0: successfully saved checkpoint from iteration 589250 to " "/checkpoints/phase1/",
            "6143: [2026-02-02 12:15:29] iteration 589290/993410 | " "consumed samples: 1000 |",
        ),
        observed_phase="steady_mid",
    )

    concrete = identity["client_concrete"]
    assert concrete["phase"] == "steady_mid"
    assert concrete["artifact_path"] is None
    assert concrete["checkpoint_iteration"] is None


def test_l0_links_timeout_aligned_precursor_to_terminal_episode(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "0: [2026-02-02 12:15:29] iteration 589290/993410 | " "consumed samples: 1000 |",
                (
                    "1000: [2026-02-02 12:15:56] host-a:393849:395082 [0] "
                    "transport/net_ib.cc:253 NCCL WARN NET/IB : mlx5_3:1 "
                    "Got non-fatal async event: port error(10)"
                ),
                (
                    "5995: [rank5995]:[E202 12:25:57.436 ProcessGroupNCCL.cpp:697] "
                    "Watchdog caught collective operation timeout: "
                    "WorkNCCL(SeqNum=126639, OpType=COALESCED, NumelIn=10, "
                    "NumelOut=20, Timeout(ms)=600000) ran for 600000 milliseconds "
                    "before timing out."
                ),
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    episode = bundle.failure_episodes[0]
    prompt_episode = build_l0_model_facing_view(bundle).evidence_bundle["failure_episodes"][0]

    assert episode.precursor_lines == (2,)
    assert episode.identity_anchor_line == 2
    assert episode.identity_anchor_reason == ("observed_precursor_aligned_with_terminal_timeout")
    assert prompt_episode["identity_anchor_line"] == 2
    assert prompt_episode["precursor_lines"] == [2]

    def evidence(line, quote, fine_class):
        return {
            "schema_version": "restart_agent_evidence.v1",
            "primary_failure": {
                "fine_class": fine_class,
                "signature": fine_class,
                "proposed_root_fingerprint": None,
                "fault_outcome": "terminal",
                "causal_role": "initiating",
                "data_position_fingerprint": None,
                "line": line,
                "rank": None,
                "phase": "steady_mid",
            },
            "root_cause_assessment": {
                "summary": "The communication path stopped making progress.",
                "plausible_causes": ["transport disruption"],
                "persistence_evidence": [],
                "transient_alternatives": ["temporary link disruption"],
            },
            "model_recovery_assessment": {
                "failure_domain": "infrastructure",
                "next_attempt_same_failure_likelihood": "plausible",
                "current_attempt_persistence_evidence": "none",
                "retry_recovery_path": "unknown",
                "confidence": 80,
                "rationale": "A retry may recover from a transient transport event.",
                "supporting_evidence_lines": [line],
            },
            "related_failures": [],
            "evidence": [{"line": line, "quote": quote, "supports": "primary_failure"}],
            "justification": "The transport event precedes the collective timeout.",
        }

    precursor_analyzer = RestartAgent(
        evidence_extractor=_FakeEvidenceExtractor(
            evidence(2, log_path.read_text().splitlines()[1], "ib_port_error")
        )
    )
    timeout_analyzer = RestartAgent(
        evidence_extractor=_FakeEvidenceExtractor(
            evidence(3, log_path.read_text().splitlines()[2], "nccl_watchdog_timeout")
        )
    )
    precursor_result = precursor_analyzer.analyze(
        {"log_path": str(log_path)}, l0_bundle=bundle
    ).to_payload()
    timeout_result = timeout_analyzer.analyze(
        {"log_path": str(log_path)}, l0_bundle=bundle
    ).to_payload()

    assert precursor_result["primary_failure"]["root_fingerprint"] == (
        timeout_result["primary_failure"]["root_fingerprint"]
    )
    assert "transport_net_ib" in precursor_result["primary_failure"]["root_fingerprint"]
    assert precursor_analyzer.last_trace["l2_audit"]["stable_identity_anchor_line"] == 2
    assert timeout_analyzer.last_trace["l2_audit"]["stable_identity_anchor_line"] == 2
    timeout_identity = timeout_result["primary_failure"]["failure_identity"]["client_concrete"]
    assert timeout_identity["phase"] == "steady_mid"
    assert timeout_identity["artifact_path"] is None
    assert timeout_identity["checkpoint_iteration"] is None


def test_l2_accepts_abbreviated_quote_without_treating_same_event_fanout_as_persistence(
    tmp_path,
):
    log_path = tmp_path / "job.log"
    timeout = (
        "Watchdog caught collective operation timeout: "
        "WorkNCCL(SeqNum=7, OpType=ALLGATHER, NumelIn=10, NumelOut=20, "
        "Timeout(ms)=600000) ran for 600000 milliseconds before timing out."
    )
    lines = [
        "0: [2026-02-02 12:00:00] iteration 10/100 | consumed samples: 100 |",
        f"0: [rank0]: {timeout}",
        f"1: [rank1]: {timeout}",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "nccl_watchdog_timeout",
            "signature": "collective timeout",
            "proposed_root_fingerprint": None,
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 2,
            "rank": "0",
            "phase": "steady_mid",
            "failure_identity": {
                "operation": "allgather",
                "mechanism": "collective_timeout",
                "component": "nccl",
                "artifact_path": None,
            },
        },
        "root_cause_assessment": {
            "summary": "The collective timed out.",
            "status": "supported_but_unconfirmed",
            "plausible_causes": ["communication failure"],
            "missing_evidence": ["cross-attempt recurrence"],
        },
        "model_recovery_assessment": {
            "failure_domain": {
                "value": "infrastructure",
                "status": "supported_but_unconfirmed",
                "confidence": 70,
            },
            "retry_outlook_without_workload_change": {
                "value": "may_recover",
                "status": "supported_but_unconfirmed",
                "confidence": 65,
            },
            "rationale": "The fanout is one collective incident; a restart may recover.",
        },
        "related_failures": [],
        "evidence": [
            {
                "line": 2,
                "quote": (
                    "Watchdog caught collective operation timeout: "
                    "WorkNCCL(SeqNum=7, OpType=ALLGATHER, ... "
                    "Timeout(ms)=600000)"
                ),
                "supports": "primary_failure",
            },
            {"line": 3, "quote": lines[2], "supports": "root_cause_assessment"},
        ],
        "justification": "The timeout stopped progress.",
    }
    extractor = _FakeEvidenceExtractor(evidence)
    analyzer = RestartAgent(evidence_extractor=extractor)

    result = analyzer.analyze({"log_path": str(log_path)}).to_payload()
    audit = analyzer.last_trace["l2_audit"]

    assert result["decision"] == "RESTART"
    assert extractor.model_view is analyzer.last_model_view
    assert analyzer.last_trace["l0_model_view"] == analyzer.last_model_view.to_payload()
    l0_trace = analyzer.last_trace["layers"]["L0"]
    assert l0_trace["sub_stages"]["L0A"]["status"] == "completed"
    assert l0_trace["sub_stages"]["L0B"]["status"] == "completed"
    assert l0_trace["sub_stages"]["L0B"]["compaction_counts"]["model_facing_context_lines"] > 0
    l2_trace = analyzer.last_trace["layers"]["L2"]
    assert l2_trace["name"] == "evidence_grounding_and_identity"
    assert l2_trace["grounding_status"] == "grounded"
    assert l2_trace["grounding_method"] == "exact_source_line"
    assert l2_trace["root_fingerprint_owner"] == "L2"
    assert l2_trace["root_fingerprint_available"] is True
    assert l2_trace["history_identity_ready"] is True
    assert l2_trace["root_fingerprint"] == result["primary_failure"]["root_fingerprint"]
    assert (
        l2_trace["root_fingerprint_source"] == result["primary_failure"]["root_fingerprint_source"]
    )
    assert l2_trace["matches_l0_root_fingerprint"] is True
    assert analyzer.last_trace["selected_failure_facts"]["source"] == "l2_grounded"
    assert analyzer.last_trace["selected_failure_facts"]["history_identity_ready"] is True
    assert analyzer.last_trace["layers"]["L3"]["selected_failure_facts_source"] == ("l2_grounded")
    assert audit["citation_audits"][0]["status"] == "abbreviated_exact"
    assert audit["model_recovery_assessment"]["failure_domain"]["value"] == ("infrastructure")
    assert (
        audit["model_recovery_assessment"]["retry_outlook_without_workload_change"]["value"]
        == "may_recover"
    )
    assert audit["recovery_field_audits"] == []
    assert audit.get("grounding_adjustments", []) == []
    assert all("persistence" not in item["code"] for item in audit["findings"])


def test_l2_does_not_create_history_identity_from_an_ungrounded_model_line(tmp_path):
    log_path = tmp_path / "job.log"
    failure_line = "RuntimeError: checkpoint metadata decode failed"
    log_path.write_text(f"iteration 10 completed\n{failure_line}\n", encoding="utf-8")
    bundle = build_l0_bundle(str(log_path))
    l0_identity = build_decision_evidence(bundle).canonical_observed_identity
    evidence = {
        "primary_failure": {
            "fine_class": "checkpoint_metadata_decode_error",
            "signature": failure_line,
            "proposed_root_fingerprint": "model:checkpoint_metadata_decode_error",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 999,
            "rank": None,
            "phase": "steady_mid",
        },
        "model_recovery_assessment": {
            "policy_class": "ambiguous",
            "supporting_evidence_lines": [999],
        },
        "evidence": [{"line": 999, "quote": failure_line, "supports": "primary_failure"}],
        "justification": "The model cited a source line that does not exist.",
    }
    history = [
        {
            "job_id": "job-1",
            "cycle_id": 1,
            "root_fingerprint": l0_identity["root_fingerprint"],
            "primary_fault_outcome": "terminal",
            "last_completed_step": 10,
        }
    ]
    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))

    analyzer.analyze(
        {
            "log_path": str(log_path),
            "job_id": "job-1",
            "cycle_id": 2,
        },
        l0_bundle=bundle,
        prior_attempts=_prior_attempt_view(*history),
    )

    l2 = analyzer.last_trace["layers"]["L2"]
    current = analyzer.last_trace["selected_failure_facts"]
    assert l2["grounding_status"] == "unavailable"
    assert l2["root_fingerprint_available"] is False
    assert current["source"] == "l2_grounded"
    assert current["root_fingerprint"] is None
    assert current["history_identity_ready"] is False
    assert analyzer.last_trace["l3_history"]["matching_root_attempts"] == 0


def test_qwen_context_budget_reduces_only_the_effective_output_cap():
    config = LlmConfig(model="nvidia/qwen/eccn-qwen-235b", max_output_tokens=64_000)
    messages = [{"role": "user", "content": "x" * 450_000}]

    budget = _request_context_budget(config, messages, include_tools=False)

    assert budget["context_window_tokens"] == 200_000
    assert budget["estimated_input_tokens"] > budget["raw_estimated_input_tokens"]
    assert budget["estimation_multiplier"] == 1.15
    assert budget["configured_max_output_tokens"] == 64_000
    assert budget["effective_max_output_tokens"] < 64_000
    assert budget["adjusted"] is True
    assert (
        budget["estimated_input_tokens"]
        + budget["effective_max_output_tokens"]
        + budget["safety_tokens"]
        <= budget["context_window_tokens"]
    )


def test_qwen397b_context_window_is_known():
    config = LlmConfig(model="nvidia/qwen/eccn-qwen3-5-397b-a17b")

    assert config.resolved_context_window_tokens() == 262_144


def test_context_window_rejection_is_reported_as_token_limit():
    detail = (
        "ContextWindowExceededError: This model's maximum context length is "
        "200000 tokens; prompt contains 136001 input tokens"
    )

    assert _is_context_window_exceeded_error(detail) is True
    summary = l1_token_limit_summary(
        (
            {
                "model_turn": 5,
                "attempt": 1,
                "finish_reason": None,
                "error_type": "context_window_exceeded",
                "usage": None,
                "max_retries": 1,
            },
        )
    )

    assert summary["hit"] is True
    assert summary["hit_count"] == 1
    assert summary["hit_calls"][0]["limit_kind"] == "context_window"


def test_one_tool_round_profile_allows_two_model_turns():
    profile = _tool_loop_profile(
        LlmConfig(
            model="nvidia/qwen/eccn-qwen-235b",
            tools_enabled=True,
            max_tool_rounds=1,
        )
    )

    assert profile["max_tool_rounds"] == 1
    assert profile["max_model_turns"] == 2
    assert "tools-disabled final turn" in profile["meaning"]


def test_default_tool_profile_does_not_advertise_evidence_object_lookup():
    config = LlmConfig()

    assert config.resolved_advertised_tools() == ("overview", "grep_log", "read_window")
    assert [schema["function"]["name"] for schema in _tool_schemas(config)] == [
        "overview",
        "grep_log",
        "read_window",
    ]


def test_evidence_object_lookup_requires_explicit_tool_profile_opt_in():
    config = LlmConfig(advertised_tools=("get_evidence_objects",))

    assert [schema["function"]["name"] for schema in _tool_schemas(config)] == [
        "get_evidence_objects"
    ]
    assert _tool_loop_profile(config)["advertised_tools"] == ["get_evidence_objects"]


def test_unknown_advertised_tool_is_rejected_by_configuration():
    with pytest.raises(ValueError, match="unknown advertised tools: made_up_tool"):
        LlmConfig(advertised_tools=("made_up_tool",))


def test_get_evidence_objects_resolves_l0a_refs_without_log_read():
    bundle = L0Bundle(
        log_path="/path/that/must/not/be/read.log",
        byte_size=120,
        line_count=3,
        occurrence_groups=(
            NormalizedOccurrenceGroup(
                occurrence_group_id="og-1",
                normalized_shape="RuntimeError: <value>",
                first_line=2,
                count=1,
                sample_lines=(2,),
            ),
        ),
        context_windows=(
            ContextWindow(
                window_id="w-1",
                selected_by="failure_episode",
                start_line=1,
                end_line=3,
                seed_lines=(2,),
                occurrence_group_ids=("og-1",),
                lines=(
                    LogLine(line=1, text="iteration 10 completed"),
                    LogLine(line=2, text="RuntimeError: observed failure"),
                    LogLine(line=3, text="shutdown"),
                ),
            ),
        ),
    )

    result = LogTools(
        bundle,
        LogSnapshot(path=bundle.log_path, lines=(), byte_size=bundle.byte_size),
    ).get_evidence_objects(["w-1", "og-1", "missing"])

    assert result["schema_version"] == "restart_agent_evidence_objects.v1"
    assert [item["object_type"] for item in result["objects"]] == [
        "context_window",
        "occurrence_group",
    ]
    assert result["objects"][0]["payload"]["lines"][1] == {
        "line": 2,
        "text": "RuntimeError: observed failure",
    }
    assert result["missing_refs"] == ["missing"]
    assert result["truncated"] is False


def test_get_evidence_objects_bounds_large_object_payloads():
    bundle = L0Bundle(
        log_path="/not/read.log",
        byte_size=100_000,
        line_count=1,
        context_windows=(
            ContextWindow(
                window_id="w-large",
                selected_by="test",
                start_line=1,
                end_line=1,
                lines=(LogLine(line=1, text="x" * 100_000),),
            ),
        ),
    )

    result = LogTools(
        bundle,
        LogSnapshot(path=bundle.log_path, lines=(), byte_size=bundle.byte_size),
    ).get_evidence_objects(["w-large"])

    assert len(json.dumps(result, sort_keys=True)) <= 50_000
    assert result["truncated"] is True
    assert result["objects"][0]["truncated"] is True
    assert result["objects"][0]["payload"]["lines"][0]["text"]


def test_unadvertised_evidence_object_call_is_rejected_before_dispatch():
    bundle = L0Bundle(
        log_path="/not/read.log",
        byte_size=0,
        line_count=1,
        context_windows=(
            ContextWindow(
                window_id="w-1",
                selected_by="test",
                start_line=1,
                end_line=1,
                lines=(LogLine(line=1, text="observed failure"),),
            ),
        ),
    )
    tool_call = {
        "id": "call-1",
        "name": "get_evidence_objects",
        "arguments": json.dumps({"refs": ["w-1"]}),
    }

    result, record, unsupported = _execute_tool_call(
        LogTools(bundle, LogSnapshot(path=bundle.log_path, lines=(), byte_size=0)),
        tool_call,
        model_turn=1,
    )

    assert result["error"] == "tool_not_advertised"
    assert record["error"] == "tool_not_advertised"
    assert unsupported["rejection_reason"] == "tool_not_advertised"

    result, record, unsupported = _execute_tool_call(
        LogTools(bundle, LogSnapshot(path=bundle.log_path, lines=(), byte_size=0)),
        tool_call,
        model_turn=1,
        advertised_tools=("get_evidence_objects",),
    )

    assert result["objects"][0]["ref"] == "w-1"
    assert record["result_lines"] == 1
    assert unsupported is None


def test_client_concrete_identity_ignores_model_fields_and_model_context():
    terminal = "9: [rank9]: RuntimeError: Unexpected result inf in local grad norm"
    source_context = (
        '9: [rank9]: File "/workspace/train.py", line 10, in check_grads',
        '9: [rank9]: File "/workspace/validate.py", line 20, in validate_result',
    )
    first = build_experimental_failure_identity(
        terminal,
        source_context,
        model_identity={
            "operation": "backward_pass",
            "mechanism": "gradient_overflow",
            "component": "first_component",
            "artifact_path": None,
        },
        model_context=('9: [rank9]: File "/workspace/model_selected.py", line 30, in model_frame',),
    )
    second = build_experimental_failure_identity(
        terminal,
        source_context,
        model_identity={
            "operation": "gradient_validation",
            "mechanism": "inf_grad_norm",
            "component": "second_component",
            "artifact_path": None,
        },
    )

    assert first["family"]["fingerprint"] != second["family"]["fingerprint"]
    assert first["concrete"]["fingerprint"] != second["concrete"]["fingerprint"]
    assert first["client_concrete"]["fingerprint"] == second["client_concrete"]["fingerprint"]


def test_model_primary_lines_in_same_episode_share_history_and_client_identity(tmp_path):
    log_path = tmp_path / "job.log"
    summary = (
        "0: ERROR:megatron.core.utils:Exception in async function: "
        "CUDA out of memory. Tried to allocate 12.12 GiB."
    )
    terminal = (
        "0: torch.OutOfMemoryError: iteration 71: CUDA out of memory. "
        "Tried to allocate 12.12 GiB."
    )
    log_path.write_text(
        "\n".join(
            [
                "0: INFO: server ready",
                summary,
                "0: Traceback (most recent call last):",
                '0:   File "/workspace/experts.py", line 751, in bias_act_func',
                terminal,
                "1: ProcessGroupNCCL watchdog caught collective operation timeout",
            ]
        ),
        encoding="utf-8",
    )
    bundle = build_l0_bundle(str(log_path))

    def evidence(line, quote):
        return {
            "schema_version": "restart_agent_evidence.v1",
            "primary_failure": {
                "fine_class": "cuda_oom",
                "signature": "CUDA out of memory",
                "proposed_root_fingerprint": None,
                "fault_outcome": "terminal",
                "causal_role": "initiating",
                "data_position_fingerprint": None,
                "line": line,
                "rank": "0",
                "phase": "setup",
            },
            "root_cause_assessment": {
                "summary": "The workload exhausted accelerator memory.",
                "plausible_causes": ["insufficient memory headroom"],
                "persistence_evidence": [],
                "transient_alternatives": ["fresh allocator state"],
            },
            "model_recovery_assessment": {
                "failure_domain": "workload",
                "next_attempt_same_failure_likelihood": "likely",
                "recurrence_confidence": 70,
                "current_attempt_persistence_evidence": "none",
                "retry_recovery_path": "workload_change",
                "confidence": 70,
                "rationale": "A fresh process may recover once.",
                "supporting_evidence_lines": [line],
            },
            "related_failures": [],
            "evidence": [{"line": line, "quote": quote, "supports": "primary_failure"}],
            "justification": "The OOM is the initiating failure.",
        }

    summary_analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence(2, summary)))
    summary_result = summary_analyzer.analyze(
        {"log_path": str(log_path)},
        l0_bundle=bundle,
    ).to_payload()
    terminal_analyzer = RestartAgent(
        evidence_extractor=_FakeEvidenceExtractor(evidence(5, terminal))
    )
    terminal_result = terminal_analyzer.analyze(
        {"log_path": str(log_path)},
        l0_bundle=bundle,
    ).to_payload()

    assert summary_result["primary_failure"]["line"] == 2
    assert terminal_result["primary_failure"]["line"] == 5
    assert summary_result["primary_failure"]["root_fingerprint"] == (
        terminal_result["primary_failure"]["root_fingerprint"]
    )
    assert summary_result["primary_failure"]["failure_iteration"] == 71
    assert terminal_result["primary_failure"]["failure_iteration"] == 71
    assert summary_result["primary_failure"]["root_fingerprint"] == ("cuda_oom:allocation_failure")
    assert summary_result["primary_failure"]["failure_identity"]["client_concrete"] == (
        terminal_result["primary_failure"]["failure_identity"]["client_concrete"]
    )
    assert summary_analyzer.last_trace["l2_audit"]["stable_identity_anchor_line"] == 2
    assert summary_analyzer.last_trace["l2_audit"]["stable_identity_anchor_reason"] == (
        "model_primary_is_episode_identity_anchor:"
        "nearby_high_signal_error_precedes_failure_episode"
    )
    assert terminal_analyzer.last_trace["l2_audit"]["stable_identity_anchor_reason"] == (
        "failure_episode_identity_anchor:nearby_high_signal_error_precedes_failure_episode"
    )


def test_logger_error_name_does_not_create_failure_episode(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "INFO:hypercorn.error:Running on http://0.0.0.0:51470 (CTRL + C to quit)\n",
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))

    assert bundle.failure_episodes == ()


def test_info_logger_error_name_is_not_failure_episode_precursor(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "0: INFO:hypercorn.error:Running on http://0.0.0.0:51470",
                (
                    "0: ERROR:workload.runtime:Exception in async function: "
                    "CUDA error: out of memory"
                ),
                "0: Traceback (most recent call last):",
                '0:   File "/workspace/client.py", line 10, in receive',
                "0: torch.AcceleratorError: CUDA error: out of memory",
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))

    assert bundle.deterministic_primary_candidate is not None
    assert bundle.deterministic_primary_candidate.line == 2
    assert bundle.failure_episodes[0].precursor_lines == (2,)
    assert bundle.failure_episodes[0].identity_anchor_line == 2
    assert bundle.failure_episodes[0].identity_anchor_reason == (
        "nearby_high_signal_error_precedes_failure_episode"
    )


def test_l0_bundle_json_round_trip_is_replayable(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "Traceback (most recent call last):\nValueError: invalid option\n",
        encoding="utf-8",
    )
    bundle_path = tmp_path / "bundle.json"
    bundle = build_l0_bundle(str(log_path))

    write_l0_bundle(bundle_path, bundle)
    replayed = read_l0_bundle(bundle_path, expected_log_path=str(log_path))
    analyzer = RestartAgent()
    original = analyzer.analyze({"log_path": str(log_path)}, l0_bundle=bundle)
    replayed_result = analyzer.analyze({"log_path": str(log_path)}, l0_bundle=replayed)

    assert replayed == bundle
    assert replayed_result.to_payload() == original.to_payload()
    assert analyzer.last_trace["timing"]["l0_reused"] is True


def test_l1_overlay_rebuilds_related_failures_and_cascades_from_validated_primary(
    tmp_path,
):
    log_path = tmp_path / "job.log"
    unicode_line = "7: [rank7]: UnicodeDecodeError: invalid continuation byte"
    cleanup_line = "8: FileNotFoundError: semaphore already removed"
    log_path.write_text(
        "\n".join(
            [
                "7: [rank7]: Traceback (most recent call last):",
                unicode_line,
                "8: NCCL watchdog caught collective operation timeout",
                "8: File multiprocessing/synchronize.py, line 87, in _cleanup",
                cleanup_line,
            ]
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "checkpoint_metadata_decode_error",
            "signature": "UnicodeDecodeError while decoding metadata",
            "proposed_root_fingerprint": "model:free_form",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 2,
            "rank": "7",
            "phase": "setup",
        },
        "root_cause_assessment": {
            "summary": "Checkpoint metadata could not be decoded.",
            "plausible_causes": ["persistent corruption", "transient exchange failure"],
            "persistence_evidence": [],
            "transient_alternatives": ["transient exchange failure"],
        },
        "model_recovery_assessment": {
            "policy_class": "ambiguous",
            "confidence": 85,
            "rationale": "One load attempt cannot establish persistence.",
            "supporting_evidence_lines": [2],
        },
        "related_failures": [
            {
                "line": 5,
                "causal_role": "teardown",
                "rationale": "Multiprocessing finalizer cleanup after termination.",
            }
        ],
        "evidence": [{"line": 2, "quote": unicode_line, "supports": "primary_failure"}],
        "justification": "The decode failure precedes NCCL and cleanup symptoms.",
    }

    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    payload = analyzer.analyze({"log_path": str(log_path)}).to_payload()

    assert payload["primary_failure"]["line"] == 2
    assert payload["primary_failure"]["root_fingerprint"] != "model:free_form"
    assert payload["root_cause_assessment"]["plausible_causes"]
    assert payload["model_recovery_assessment"]["failure_domain"]["value"] == "unknown"
    assert payload["model_recovery_assessment"]["retry_outlook_without_workload_change"][
        "value"
    ] == ("unknown")
    assert payload["model_recovery_assessment"]["failure_domain"]["confidence"] == 85
    assert payload["secondary_failures"] == []
    cascades_by_line = {item["first_line"]: item for item in payload["cascades"]}
    assert cascades_by_line[3]["causal_role"] == "cascade"
    assert cascades_by_line[3]["relationship_rationales"] == []
    assert cascades_by_line[5]["causal_role"] == "teardown"
    assert cascades_by_line[5]["relationship_rationales"] == [
        "Multiprocessing finalizer cleanup after termination."
    ]


def test_l1_only_cascade_is_retained_in_public_result(tmp_path):
    log_path = tmp_path / "job.log"
    primary_line = "ValueError: invalid workload configuration"
    cascade_line = "opaque downstream worker failure report"
    log_path.write_text(f"{primary_line}\n{cascade_line}\n", encoding="utf-8")
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "workload_configuration_error",
            "signature": primary_line,
            "proposed_root_fingerprint": None,
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 1,
            "rank": None,
            "phase": "setup",
        },
        "root_cause_assessment": {
            "summary": "The workload configuration was invalid.",
            "plausible_causes": ["invalid configuration"],
            "persistence_evidence": [],
            "transient_alternatives": [],
        },
        "model_recovery_assessment": {
            "policy_class": "ambiguous",
            "confidence": 80,
            "rationale": "The first line is the initiating failure.",
            "supporting_evidence_lines": [1],
        },
        "related_failures": [
            {
                "line": 2,
                "causal_role": "cascade",
                "rationale": "The worker report followed the configuration failure.",
            }
        ],
        "evidence": [{"line": 1, "quote": primary_line, "supports": "primary_failure"}],
        "justification": "The configuration error preceded the worker report.",
    }

    payload = (
        RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
        .analyze({"log_path": str(log_path)})
        .to_payload()
    )

    assert payload["secondary_failures"] == []
    assert len(payload["cascades"]) == 1
    cascade = payload["cascades"][0]
    assert cascade["fine_class"] == "related_failure"
    assert cascade["causal_role"] == "cascade"
    assert cascade["first_line"] == 2
    assert cascade["last_line"] == 2
    assert cascade["count"] == 1
    assert cascade["sample_lines"] == [2]
    assert cascade["relationship_rationales"] == [
        "The worker report followed the configuration failure."
    ]


def test_l2_uses_canonical_evidence_tags_for_recovery_support(tmp_path):
    log_path = tmp_path / "job.log"
    load_line = "loading distributed checkpoint at iteration 622125"
    failure_line = "7: [rank7]: UnicodeDecodeError: invalid continuation byte"
    log_path.write_text(
        f"{load_line}\nTraceback (most recent call last):\n{failure_line}\n",
        encoding="utf-8",
    )
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "checkpoint_metadata_decode_error",
            "signature": "UnicodeDecodeError while decoding metadata",
            "proposed_root_fingerprint": "model:checkpoint_decode",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 3,
            "rank": "7",
            "phase": "setup",
        },
        "root_cause_assessment": {
            "summary": "Checkpoint metadata could not be decoded.",
            "plausible_causes": ["checkpoint corruption", "transient read failure"],
            "persistence_evidence": [],
            "transient_alternatives": ["transient read failure"],
        },
        "model_recovery_assessment": {
            "policy_class": "ambiguous",
            "confidence": 75,
            "rationale": "The load context and decode failure do not prove persistence.",
            "supporting_evidence_lines": [1, 3],
        },
        "related_failures": [],
        "evidence": [{"line": 3, "quote": failure_line, "supports": "primary_failure"}],
        "justification": "The decode failure occurred while loading a checkpoint.",
    }

    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    payload = analyzer.analyze({"log_path": str(log_path)}).to_payload()
    validation = analyzer.last_trace["l2_audit"]

    assert validation["audit_status"] == "clean"
    assert validation["recovery_assessment_used"] is True
    assert validation["failure_domain_supporting_lines"] == [3]
    assert validation["retry_outlook_supporting_lines"] == [3]
    assert validation["field_audits"]["model_recovery_assessment"]["status"] == "available"
    assert analyzer.last_trace["l1"]["parsed_evidence"] == _current_evidence(evidence)
    assert payload["primary_failure"]["line"] == 3
    assert payload["model_recovery_assessment"]["failure_domain"]["value"] == "unknown"


def test_l2_resolves_unique_nearby_citation_without_discarding_l1(tmp_path):
    log_path = tmp_path / "job.log"
    lines = [
        "loading distributed checkpoint at iteration 622125",
        'File "validation.py", line 558, in determine_global_metadata',
        "torch.distributed.all_gather_object(global_metadata, local_metadata)",
        "4175: [rank4175]: UnicodeDecodeError: invalid continuation byte",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "checkpoint_metadata_corruption",
            "signature": "UnicodeDecodeError: invalid continuation byte",
            "proposed_root_fingerprint": "checkpoint_metadata_corruption:unpickle_failure",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 4,
            "rank": "4175",
            "phase": "setup",
        },
        "root_cause_assessment": {
            "summary": "Checkpoint metadata could not be decoded.",
            "plausible_causes": ["persistent corruption", "transient read failure"],
            "persistence_evidence": [],
            "transient_alternatives": ["transient read failure"],
        },
        "model_recovery_assessment": {
            "policy_class": "ambiguous",
            "confidence": 85,
            "rationale": "One load failure does not prove persistence.",
            "supporting_evidence_lines": [1, 2, 4],
        },
        "related_failures": [],
        "evidence": [
            {"line": 1, "quote": lines[0], "supports": "checkpoint load context"},
            {"line": 2, "quote": lines[2], "supports": "distributed metadata gather"},
            {"line": 4, "quote": lines[3], "supports": "primary failure"},
        ],
        "justification": "Checkpoint metadata deserialization failed during load.",
    }

    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    payload = analyzer.analyze({"log_path": str(log_path)}).to_payload()
    audit = analyzer.last_trace["l2_audit"]

    assert audit["used"] is True
    assert audit["audit_status"] == "resolved", audit
    assert audit["failure_domain_supporting_lines"] == [4]
    assert audit["retry_outlook_supporting_lines"] == [4]
    assert audit["citation_audits"][1]["original_line"] == 2
    assert audit["citation_audits"][1]["resolved_line"] == 3
    assert audit["citation_audits"][1]["status"] == "nearby_resolved"
    assert payload["primary_failure"]["line"] == 4
    assert payload["primary_failure"]["policy_class"] == "ambiguous"
    assert payload["decision"] == Decision.RESTART.value
    assert payload["result_provenance"]["model_contribution"] == (
        "attempted_used_with_resolved_findings"
    )
    assert payload["result_provenance"]["result_quality"] == "normal"


def test_l2_accepts_exact_model_visible_truncated_rendering(tmp_path):
    log_path = tmp_path / "job.log"
    progress_line = "iteration 634870 | " + ("metric=1.0 | " * 80)
    failure_line = "RuntimeError: checkpoint write failed"
    log_path.write_text(f"{progress_line}\n{failure_line}\n", encoding="utf-8")
    rendered_progress = progress_line[:120] + "...[truncated]"
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "checkpoint_write_failure",
            "signature": failure_line,
            "proposed_root_fingerprint": "checkpoint:write",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 2,
            "rank": None,
            "phase": "checkpointing",
        },
        "root_cause_assessment": {
            "summary": "Checkpoint writing failed after training progress.",
            "plausible_causes": ["storage or writer failure"],
            "persistence_evidence": [],
            "transient_alternatives": ["transient storage failure"],
        },
        "model_recovery_assessment": {
            "policy_class": "ambiguous",
            "confidence": 60,
            "rationale": "One write failure does not prove persistence.",
            "supporting_evidence_lines": [1, 2],
        },
        "related_failures": [],
        "evidence": [
            {"line": 1, "quote": rendered_progress, "supports": "prior progress"},
            {"line": 2, "quote": failure_line, "supports": "primary failure"},
        ],
        "justification": "The checkpoint write failed after prior progress.",
    }
    extractor = _FakeEvidenceExtractor(
        evidence,
        transcript_events=(
            {
                "event_type": "bundle_snapshot",
                "model_visible_payload": {
                    "context": [
                        {"line": 1, "text": rendered_progress},
                        {"line": 2, "text": failure_line},
                    ]
                },
            },
        ),
    )

    analyzer = RestartAgent(evidence_extractor=extractor)
    payload = analyzer.analyze({"log_path": str(log_path)}).to_payload()
    audit = analyzer.last_trace["l2_audit"]

    assert [item["status"] for item in audit["citation_audits"]] == [
        "rendered_exact",
        "exact",
    ]
    assert audit["field_findings"] == {}
    assert audit["audit_status"] == "resolved"
    assert payload["result_provenance"]["model_contribution"] == (
        "attempted_used_with_resolved_findings"
    )


def test_l2_does_not_ground_exact_source_quote_that_was_not_model_visible(tmp_path):
    log_path = tmp_path / "job.log"
    failure_line = "RuntimeError: checkpoint write failed"
    log_path.write_text(f"iteration 10 completed\n{failure_line}\n", encoding="utf-8")
    evidence = {
        "primary_failure": {
            "line": 2,
            "fine_class": "checkpoint_write_failure",
            "causal_role": "initiating",
        },
        "evidence": [{"line": 2, "quote": failure_line}],
        "justification": "The checkpoint write failed.",
    }
    extractor = _FakeEvidenceExtractor(
        evidence,
        transcript_events=(
            {
                "event_type": "bundle_snapshot",
                "model_visible_payload": {
                    "context": [{"line": 1, "text": "iteration 10 completed"}]
                },
            },
        ),
    )

    analyzer = RestartAgent(evidence_extractor=extractor)
    analyzer.analyze({"log_path": str(log_path)})
    audit = analyzer.last_trace["l2_audit"]

    assert audit["citation_audits"][0]["status"] == "not_model_visible"
    assert "evidence_not_model_visible" in audit["field_finding_codes"]["evidence"]
    assert audit["recovery_assessment_policy_grounded"] is False


def test_invalid_related_role_is_l1_contract_failure(tmp_path):
    log_path = tmp_path / "job.log"
    initiating_line = "ERROR: checkpoint writer unexpected position"
    terminal_line = "CheckpointException: write failed"
    log_path.write_text(f"{initiating_line}\n{terminal_line}\n", encoding="utf-8")
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "checkpoint_write_exception",
            "signature": terminal_line,
            "proposed_root_fingerprint": "checkpoint:write",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 2,
            "rank": None,
            "phase": "checkpointing",
        },
        "root_cause_assessment": {
            "summary": "Checkpoint writing failed.",
            "plausible_causes": ["storage or serialization failure"],
            "persistence_evidence": [],
            "transient_alternatives": ["transient storage failure"],
        },
        "model_recovery_assessment": {
            "policy_class": "ambiguous",
            "confidence": 50,
            "rationale": "Persistence is unknown.",
            "supporting_evidence_lines": [1, 2],
        },
        "related_failures": [
            {
                "line": 1,
                "causal_role": "initiating",
                "rationale": "Earlier initiating writer error.",
            }
        ],
        "evidence": [
            {"line": 1, "quote": initiating_line, "supports": "earlier mechanism"},
            {"line": 2, "quote": terminal_line, "supports": "terminal exception"},
        ],
        "justification": "The terminal exception followed the writer error.",
    }

    assert model_evidence_contract_errors(_current_evidence(evidence)) == [
        "related_failures[0].causal_role is invalid"
    ]


def test_l2_rejects_related_failure_line_not_visible_to_model(tmp_path):
    log_path = tmp_path / "job.log"
    primary_line = "RuntimeError: invalid configuration"
    teardown_line = "FileNotFoundError: cleanup marker missing"
    log_path.write_text(f"{primary_line}\n{teardown_line}\n", encoding="utf-8")
    evidence = {
        "primary_failure": {
            "line": 1,
            "causal_role": "initiating",
            "fine_class": "invalid_configuration",
        },
        "root_cause_assessment": {
            "summary": "The configuration is invalid.",
            "status": "established_by_current_log",
            "plausible_causes": ["invalid workload configuration"],
            "missing_evidence": [],
        },
        "model_recovery_assessment": {
            "failure_domain": {
                "value": "workload",
                "status": "established_by_current_log",
                "confidence": 95,
            },
            "retry_outlook_without_workload_change": {
                "value": "cannot_recover",
                "status": "established_by_current_log",
                "confidence": 95,
            },
            "rationale": "The unchanged configuration remains invalid.",
        },
        "related_failures": [
            {
                "line": 2,
                "causal_role": "teardown",
                "rationale": "Cleanup failed after the initiating error.",
            }
        ],
        "evidence": [{"line": 1, "quote": primary_line, "supports": "primary_failure"}],
    }
    extractor = _FakeEvidenceExtractor(
        evidence,
        transcript_events=(
            {
                "event_type": "bundle_snapshot",
                "model_visible_payload": {
                    "evidence_bundle": {"lines": [{"line": 1, "text": primary_line}]}
                },
            },
        ),
    )

    analyzer = RestartAgent(evidence_extractor=extractor)
    analyzer.analyze({"log_path": str(log_path)})
    audit = analyzer.last_trace["l2_audit"]

    assert audit["field_finding_codes"]["related_failures"] == [
        "related_failure_line_not_model_visible"
    ]
    assert audit["audited_related_failures"] == []


def test_l2_uses_canonical_evidence_tags_instead_of_removed_supporting_line_field(
    tmp_path,
):
    log_path = tmp_path / "job.log"
    failure_line = "7: [rank7]: UnicodeDecodeError: invalid continuation byte"
    log_path.write_text(f"{failure_line}\n", encoding="utf-8")
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "checkpoint_metadata_decode_error",
            "signature": "UnicodeDecodeError while decoding metadata",
            "proposed_root_fingerprint": "model:checkpoint_decode",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 1,
            "rank": "7",
            "phase": "setup",
        },
        "root_cause_assessment": {
            "summary": "Checkpoint metadata could not be decoded.",
            "plausible_causes": ["checkpoint corruption"],
            "persistence_evidence": [],
            "transient_alternatives": [],
        },
        "model_recovery_assessment": {
            "policy_class": "user_failure",
            "confidence": 95,
            "rationale": "An unseen line allegedly proves persistence.",
            "supporting_evidence_lines": [999],
        },
        "related_failures": [],
        "evidence": [{"line": 1, "quote": failure_line, "supports": "primary_failure"}],
        "justification": "The decode failure is the initiating exception.",
    }

    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    payload = analyzer.analyze({"log_path": str(log_path)}).to_payload()
    validation = analyzer.last_trace["l2_audit"]

    assert validation["audit_status"] == "clean"
    assert validation["primary_used"] is True
    assert validation["recovery_assessment_used"] is True
    assert validation["failure_domain_supporting_lines"] == [1]
    assert validation["retry_outlook_supporting_lines"] == [1]
    assert payload["primary_failure"]["line"] == 1
    assert payload["primary_failure"]["policy_class"] == "ambiguous"
    assert payload["model_recovery_assessment"]["failure_domain"]["value"] == "workload"
    assert "recovery_requirement" not in payload["model_recovery_assessment"]
    assert validation["recovery_assessment_policy_grounded"] is True
    assert payload["retry_policy"]["recovery_assessment_policy_grounded"] is True
    assert payload["decision"] == Decision.STOP.value
    assert payload["result_provenance"]["model_contribution"] == "attempted_used"
    assert payload["result_provenance"]["result_quality"] == "normal"


def test_trace_exposes_distinct_l0_through_l4_layers(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: CUDA initialization failed\n", encoding="utf-8")

    analyzer = RestartAgent()
    analyzer.analyze({"log_path": str(log_path)})

    layers = analyzer.last_trace["layers"]
    assert list(layers) == ["L0", "L1", "L2", "L3", "L4"]
    assert layers["L0"]["name"] == "evidence_assembly_and_projection"
    assert layers["L0"]["sub_stages"]["L0A"]["name"] == "complete_evidence_assembly"
    assert layers["L0"]["sub_stages"]["L0A"]["status"] == "completed"
    assert layers["L0"]["root_fingerprint_owner"] == "L0"
    assert layers["L0"]["root_fingerprint_available"] is True
    assert layers["L0"]["history_identity_ready"] is True
    assert layers["L0"]["root_fingerprint"]
    assert layers["L0"]["sub_stages"]["DecisionEvidence"]["root_fingerprint"] == (
        layers["L0"]["root_fingerprint"]
    )
    assert layers["L0"]["sub_stages"]["L0B"] == {
        "name": "initial_model_evidence_view",
        "wall_clock_s": 0.0,
        "status": "not_run",
    }
    assert analyzer.last_trace["l0_model_view"] is None
    assert layers["L1"]["name"] == "semantic_analysis"
    assert layers["L2"]["name"] == "evidence_grounding_and_identity"
    assert layers["L2"]["grounding_status"] == "not_run"
    assert layers["L2"]["root_fingerprint_owner"] == "L2"
    assert layers["L2"]["root_fingerprint_available"] is False
    assert layers["L2"]["history_identity_ready"] is False
    assert layers["L2"]["matches_l0_root_fingerprint"] is None
    assert layers["L3"]["name"] == "history_enrichment"
    assert layers["L3"]["selected_failure_facts_source"] == "l0_deterministic"
    assert layers["L3"]["history_identity_ready"] is True
    assert layers["L4"]["name"] == "policy_decision"
    assert "operational_policy_mapping" not in analyzer.last_trace["l3_history"]
    assert analyzer.last_trace["l4_policy"]["retry_policy"]["rule"] == ("general_retry")
    assert analyzer.last_trace["latency_measurement"] == {
        "mode": "terminal_request_to_result",
        "terminal_total_wall_clock_s": analyzer.last_trace["timing"]["total_wall_clock_s"],
        "post_progressive_end_wall_clock_s": None,
        "progressive_decision_window_hit": None,
        "production_gate_measured": False,
    }


def test_log_unavailable_returns_restart(tmp_path):
    result = RestartAgent().analyze({"log_path": str(tmp_path / "missing.log")})

    payload = result.to_payload()
    assert payload["decision"] == Decision.RESTART.value
    assert payload["decision_basis"] == DecisionBasis.LOG_UNAVAILABLE.value
    assert payload["primary_failure"] is None
    assert payload["evidence"] == []
    assert payload["result_provenance"]["evidence_source"] == "fallback_log_unavailable"
    assert payload["result_provenance"]["result_quality"] == "fallback_only"
    assert payload["result_provenance"]["nvrx_use"] == "fallback_to_nvrx_default"


def test_missing_log_path_is_rejected():
    with pytest.raises(TypeError, match="log_path is required"):
        RestartAgent().analyze({})


def test_relative_log_path_is_rejected():
    with pytest.raises(ValueError, match="log_path must be absolute"):
        RestartAgent().analyze({"log_path": "relative.log"})


def test_non_integer_cycle_id_is_rejected(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: failed", encoding="utf-8")

    with pytest.raises(TypeError, match="cycle_id must be an integer"):
        RestartAgent().analyze({"log_path": str(log_path), "cycle_id": "1"})


def test_training_runtime_gpu_error_uses_generic_cuda_assert(tmp_path):
    log_path = tmp_path / "service_input.sanitized.log"
    log_path.write_text(
        "\n".join(
            [
                (
                    "7:  [2026-05-27 07:58:10.261551] iteration        1/       4 | "
                    "consumed samples:           64 | lm loss: 1.339357E+01 | "
                    "number of skipped iterations:   0 | number of nan iterations:   0 |"
                ),
                "INFO number of nan iterations: 0",
                "CRITICAL:training.runtime:raising GPU error on cuda:0",
                "RuntimeError: CUDA error: device-side assert triggered",
            ]
        ),
        encoding="utf-8",
    )

    result = RestartAgent().analyze({"log_path": str(log_path), "job_id": "job-1"})
    payload = result.to_payload()

    assert payload["decision"] == Decision.RESTART.value
    assert payload["decision_basis"] == DecisionBasis.GENERAL_RETRY_AVAILABLE.value
    assert payload["primary_failure"]["fine_class"] == "observed_exception"
    assert payload["primary_failure"]["policy_class"] == "ambiguous"


def test_ambiguous_cuda_assert_defaults_to_restart(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "INFO training iteration 120",
                "RuntimeError: CUDA error: device-side assert triggered",
            ]
        ),
        encoding="utf-8",
    )

    result = RestartAgent().analyze({"log_path": str(log_path), "job_id": "job-1"})
    payload = result.to_payload()

    assert payload["decision"] == Decision.RESTART.value
    assert payload["decision_basis"] == DecisionBasis.GENERAL_RETRY_AVAILABLE.value
    assert payload["primary_failure"]["fine_class"] == "observed_exception"


def test_cuda_debugging_advice_is_context_not_failure_candidate(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing",
                "CUDA kernel errors might be asynchronously reported at some other API call.",
                "For debugging consider passing CUDA_LAUNCH_BLOCKING=1.",
                "Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.",
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    prompt_bundle = build_l0_model_facing_view(bundle).evidence_bundle

    assert bundle.deterministic_primary_candidate is not None
    assert bundle.deterministic_primary_candidate.line == 1
    assert bundle.deterministic_primary_candidate.fine_class == "observed_exception"
    assert {match.line for match in bundle.registry_matches} == {1}
    assert {anchor.line for anchor in bundle.candidate_anchors}.isdisjoint({2, 3, 4})
    prompt_lines = {
        item["line"]: item
        for window in prompt_bundle["context_windows"]
        for item in window["lines"]
    }
    assert prompt_lines[1]["line_role"] == "observed_log"
    assert prompt_lines[2]["line_role"] == "diagnostic_context"
    assert prompt_lines[3]["line_role"] == "diagnostic_context"
    assert prompt_lines[4]["line_role"] == "diagnostic_context"


def test_peer_gpu_memory_access_is_ambiguous_and_forms_failure_episode(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                (
                    "7: [2026-02-21 03:30:33.910316] iteration 686650/794728 | "
                    "consumed samples: 2109388800 | lm loss: 9.602139E-01 |"
                ),
                (
                    "171: [rank171]: Process group watchdog thread terminated with "
                    "exception: CUDA error: Invalid access of peer GPU memory over "
                    "nvlink or a hardware error"
                ),
                (
                    "171: what(): Process group watchdog thread terminated with "
                    "exception: CUDA error: Invalid access of peer GPU memory over "
                    "nvlink or a hardware error"
                ),
                "171: Fatal Python error: Aborted",
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))

    assert bundle.deterministic_primary_candidate is not None
    assert bundle.deterministic_primary_candidate.line == 2
    assert bundle.deterministic_primary_candidate.fine_class == "peer_gpu_memory_access_failure"
    assert bundle.deterministic_primary_candidate.policy_class == "ambiguous"
    assert bundle.deterministic_primary_candidate.root_fingerprint == (
        "peer_gpu_memory_access_failure:peer_gpu_memory_access"
    )
    assert "gpu_hardware_fault" not in {match.registry_id for match in bundle.registry_matches}
    assert len(bundle.failure_episodes) == 1
    assert bundle.failure_episodes[0].start_line == 2
    assert bundle.failure_episodes[0].terminal_exception_line == 2
    assert bundle.failure_episodes[0].duplicate_rendering_lines == (3,)
    assert bundle.failure_episodes[0].status == "terminal"


def test_direct_nvlink_link_failure_remains_hardware_evidence(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "NVRM: Xid 74: NVLink fatal link down on GPU 3",
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))

    assert bundle.deterministic_primary_candidate is not None
    assert bundle.deterministic_primary_candidate.fine_class == "gpu_hardware_fault"
    assert bundle.deterministic_primary_candidate.policy_class == "not_user_failure"


def test_concatenated_cuda_peer_error_is_not_hardware_confirmation(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                (
                    "7: [2026-02-24 23:01:00.201888] iteration 745580/794728 | "
                    "consumed samples: 2290421760 | lm loss: 9.240151E-01 |"
                ),
                (
                    "5250: CUDA error encountered at: file=/home/DeepEP/csrc/"
                    "hybrid_ep/extension/permute.cu, line=535, "
                    "call='cudaGetLastError()', Reason=cudaErrorContained:Invalid "
                    "access of peer GPU memory over nvlink or a hardware errorFatal "
                    "Python error: Aborted"
                ),
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))

    assert bundle.deterministic_primary_candidate is not None
    assert bundle.deterministic_primary_candidate.line == 2
    assert bundle.deterministic_primary_candidate.fine_class == "peer_gpu_memory_access_failure"
    assert bundle.deterministic_primary_candidate.policy_class == "ambiguous"
    assert "gpu_hardware_fault" not in {match.registry_id for match in bundle.registry_matches}
    assert len(bundle.failure_episodes) == 1
    assert bundle.failure_episodes[0].terminal_exception_line == 2
    assert bundle.failure_episodes[0].first_process_termination_line == 2
    assert bundle.failure_episodes[0].status == "terminal"


def test_conditional_cause_language_remains_observed_but_is_marked_hypothetical(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "RuntimeError: worker was killed. It is possible that memory is exhausted.\n",
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    prompt_bundle = build_l0_model_facing_view(bundle).evidence_bundle
    prompt_line = next(
        item
        for window in prompt_bundle["context_windows"]
        for item in window["lines"]
        if item["line"] == 1
    )

    assert bundle.deterministic_primary_candidate is not None
    assert prompt_line["line_role"] == "observed_log_with_causal_hypothesis"
    assert prompt_line["diagnostic_uncertainty_kind"] == "conditional_cause_language"


def test_attempt_execution_context_highlights_runtime_and_checkpoint_replay_distance(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "0: [2026-02-18 08:00:00] iteration 100/1000 | consumed samples: 1000 |",
                "0: successfully saved checkpoint from iteration 100 to /checkpoints/",
                "0: [2026-02-18 08:01:00] iteration 110/1000 | consumed samples: 1100 |",
                "0: RuntimeError: terminal failure",
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    model_view = build_l0_model_facing_view(bundle)
    context = model_view.attempt_execution_context
    progress = model_view.decision_evidence.progress_checkpoint_state

    assert progress["iteration_delta"] == 10
    assert progress["successful_runtime_seconds"] == 60.0
    assert progress["checkpoint_marker_count"] == 1
    assert progress["iterations_since_checkpoint"] == 10
    assert progress["progress_after_failure_episode"] is False
    assert set(context) == {"scope", "terminal_timing"}


def test_later_progress_after_fault_does_not_claim_component_recovery(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "0: [2026-02-18 08:00:00] iteration 100/1000 | consumed samples: 1000 |",
                "1: RuntimeError: temporary network disturbance",
                "0: [2026-02-18 08:01:00] iteration 110/1000 | consumed samples: 1100 |",
                "2: RuntimeError: terminal failure",
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    continuation = next(
        item
        for item in bundle.later_progress_after_fault_observations
        if item.sample_event_lines == (2,)
    )

    assert continuation.sample_later_progress_lines == (3,)
    assert continuation.interpretation == "job_progress_observed_after_event"
    assert continuation.component_recovery_proven is False


def test_l0_consolidates_serialized_inner_and_outer_exception_renderings(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "0: [2026-02-18 08:00:00] iteration 100/1000 | consumed samples: 1000 |",
                "7: ['Traceback (most recent call last):\\n', 'RuntimeError: worker SIGBUS\\n']",
                "7: [rank7]: Traceback (most recent call last):",
                "7: [rank7]: RuntimeError: worker SIGBUS",
                "7: [rank7]: The above exception was the direct cause of the following exception:",
                "7: [rank7]: Traceback (most recent call last):",
                "7: [rank7]: RuntimeError: worker exited unexpectedly",
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))

    assert len(bundle.failure_episodes) == 1
    episode = bundle.failure_episodes[0]
    assert episode.terminal_exception_line == 4
    assert episode.exception_chain_lines == (2, 4, 7)
    assert episode.duplicate_rendering_lines == (2,)
    assert episode.wrapper_exception_lines == (7,)
    assert bundle.deterministic_primary_candidate is not None
    assert bundle.deterministic_primary_candidate.line == 4

    payload = RestartAgent().analyze({"log_path": str(log_path)}).to_payload()
    assert payload["primary_failure"]["line"] == 4
    assert {item["line"] for item in payload["secondary_failures"]}.isdisjoint(
        episode.exception_chain_lines
    )


def test_l0_user_failure_candidate_does_not_stop_without_l1(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "INFO loading checkpoint",
                "FileNotFoundError: [Errno 2] No such file or directory",
            ]
        ),
        encoding="utf-8",
    )

    analyzer = RestartAgent()
    result = analyzer.analyze({"log_path": str(log_path), "job_id": "job-1"})
    payload = result.to_payload()

    assert payload["decision"] == Decision.RESTART.value
    assert payload["decision_basis"] == DecisionBasis.GENERAL_RETRY_AVAILABLE.value
    assert payload["primary_failure"]["fine_class"] == "observed_exception"
    assert payload["primary_failure"]["policy_class"] == "ambiguous"
    assert all(item["policy_class"] != "user_failure" for item in payload["secondary_failures"])
    assert payload["result_provenance"]["evidence_source"] == "l0_deterministic"
    assert payload["result_provenance"]["model_contribution"] == "not_enabled"
    assert payload["result_provenance"]["notes"] == []
    assert (
        analyzer.last_trace["l0_summary"]["deterministic_primary_candidate"]["policy_class"]
        == "ambiguous"
    )
    assert analyzer.last_trace["l2_grounded_semantics"] is None
    assert analyzer.last_trace["external_output"]["primary_failure"]["policy_class"] == (
        "ambiguous"
    )
    assert analyzer.last_trace["l2_audit"].get("l0_policy_downgraded") is None


def test_l1_runs_for_l0_user_failure_candidate(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "INFO loading checkpoint",
                "ValueError: invalid config option tensor_model_parallel_size",
            ]
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "python_user_exception",
            "policy_class": "user_failure",
            "signature": "ValueError:",
            "root_fingerprint": "python_user_exception:valueerror",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 2,
            "rank": None,
            "phase": "setup",
        },
        "secondary_failures": [],
        "cascades": [],
        "evidence": [
            {
                "line": 2,
                "quote": "ValueError: invalid config option tensor_model_parallel_size",
                "supports": "primary_failure",
            }
        ],
        "justification": "The cited current-log error was selected as user failure.",
    }

    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    result = analyzer.analyze({"log_path": str(log_path), "job_id": "job-1"})
    payload = result.to_payload()

    assert payload["decision"] == Decision.STOP.value
    assert payload["decision_basis"] == DecisionBasis.WORKLOAD_UNRECOVERABLE.value
    assert payload["primary_failure"]["policy_class"] == "ambiguous"
    assert payload["result_provenance"]["evidence_source"] == "l1_model_audited"
    assert payload["result_provenance"]["model_contribution"] == "attempted_used"
    assert analyzer.last_trace["timing"]["l1_model_calls"] == 1
    assert analyzer.last_trace["l2_audit"]["used"] is True
    assert not analyzer.last_trace["l2_audit"].get("l0_policy_downgraded")
    assert payload["evidence"] == _current_evidence(evidence)["evidence"]
    assert payload["justification"] == evidence["justification"]


def test_incomplete_l1_contract_is_rejected_without_manufactured_evidence(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "ValueError: invalid config option tensor_model_parallel_size\n",
        encoding="utf-8",
    )
    incomplete = {
        "primary_failure": {
            "fine_class": "python_user_exception",
            "policy_class": "user_failure",
            "signature": "ValueError:",
            "root_fingerprint": "python_user_exception:valueerror",
            "fault_outcome": "terminal",
            "line": 1,
        },
        "secondary_failures": [],
        "cascades": [],
    }

    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(incomplete))
    payload = analyzer.analyze({"log_path": str(log_path)}).to_payload()

    assert payload["decision"] == Decision.RESTART.value
    assert payload["result_provenance"]["model_contribution"] == "attempted_not_used_malformed"
    assert analyzer.last_trace["l2_audit"]["used"] is False
    assert analyzer.last_trace["l2_audit"]["audit_status"] == "not_run"
    assert analyzer.last_trace["layers"]["L1"]["output_status"] == "contract_invalid"
    assert any(
        "evidence must support primary_failure" in error
        for error in analyzer.last_trace["layers"]["L1"]["output_errors"]
    )
    assert payload["justification"] != ""


def test_l2_reports_suspect_primary_role_without_discarding_l1(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "FileNotFoundError: [Errno 2] No such file or directory\n",
        encoding="utf-8",
    )
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "multiprocessing_cleanup",
            "policy_class": "user_failure",
            "signature": "FileNotFoundError:",
            "root_fingerprint": "multiprocessing_cleanup:filenotfounderror",
            "fault_outcome": "terminal",
            "causal_role": "teardown",
            "data_position_fingerprint": None,
            "line": 1,
            "rank": None,
            "phase": "teardown",
        },
        "secondary_failures": [],
        "cascades": [],
        "evidence": [
            {
                "line": 1,
                "quote": "FileNotFoundError: [Errno 2] No such file or directory",
                "supports": "primary_failure",
            }
        ],
        "justification": "This exception occurred during teardown.",
    }

    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    analyzer.analyze({"log_path": str(log_path)})

    validation = analyzer.last_trace["l2_audit"]
    assert validation["used"] is True
    assert validation["audit_status"] == "findings"
    assert validation["field_audits"]["primary_failure"]["finding_classes"] == [
        "primary_causal_role_suspect"
    ]


def test_non_utf8_log_uses_latin1_fallback(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_bytes(
        b"INFO training iteration 120\n"
        b"RuntimeError: CUDA error: device-side assert triggered \xff\n"
    )

    bundle = build_l0_bundle(str(log_path))
    window = LogTools(bundle, LogSnapshot.read(log_path)).read_window(2, before=0, after=0)
    payload = RestartAgent().analyze({"log_path": str(log_path), "job_id": "job-1"})

    assert bundle.line_count == 2
    assert bundle.deterministic_primary_candidate is not None
    assert chr(255) in (bundle.deterministic_primary_candidate.quote or "")
    assert "\ufffd" not in (bundle.deterministic_primary_candidate.quote or "")
    assert chr(255) in window["lines"][0]["text"]
    assert payload.to_payload()["primary_failure"]["fine_class"] == "observed_exception"


def test_llm_prompt_and_tools_do_not_expose_path_labels(tmp_path):
    log_dir = tmp_path / "cuda_oom_logs" / "observed_label_user_failure"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "misleading_user_failure_name.log"
    log_path.write_text(
        "\n".join(
            [
                "1: iteration 1/4 | consumed samples: 64 |",
                "RuntimeError: CUDA error: device-side assert triggered",
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    prompt_payload = build_l0_model_facing_view(bundle).evidence_bundle
    overview_payload = LogTools(bundle, LogSnapshot.read(log_path)).overview()

    assert "cuda_oom_logs" in bundle.path_hints
    for payload in (prompt_payload, overview_payload):
        serialized = json.dumps(payload, sort_keys=True)
        assert "cuda_oom_logs" not in serialized
        assert "observed_label_user_failure" not in serialized
        assert "misleading_user_failure_name.log" not in serialized
        assert str(log_path) not in serialized


def test_context_preview_preserves_high_signal_lines_between_head_seed_and_tail():
    lines = [LogLine(line=line, text=f"INFO ordinary line {line}") for line in range(1152, 1253)]
    lines[1159 - 1152] = LogLine(
        line=1159,
        text="1: CRITICAL:training.runtime:raising GPU error on cuda:0",
    )

    preview = _window_preview_lines(lines, seed_lines=(1172,))
    preview_lines = [item["line"] for item in preview]

    assert 1159 in preview_lines
    assert 1172 in preview_lines
    assert len(preview_lines) <= 8
    assert preview_lines == sorted(preview_lines)


def test_bundle_prompt_uses_bounded_excerpts_for_top_context_windows():
    setup_window = ContextWindow(
        window_id="w-setup",
        selected_by="registry_match",
        start_line=800,
        end_line=840,
        seed_lines=(820,),
        lines=tuple(LogLine(line=line, text=f"INFO setup line {line}") for line in range(800, 841)),
    )
    candidate_lines = []
    for line in range(1119, 1280):
        text = f"INFO ordinary line {line}"
        if line == 1151:
            text = "7: iteration        1/       4 | consumed samples: 64 |"
        elif line == 1159:
            text = "1: CRITICAL:training.runtime:raising GPU error on cuda:0"
        elif line == 1162:
            text = "7: iteration        2/       4 | consumed samples: 128 |"
        elif line == 1172:
            text = "0: ProcessGroupNCCL watchdog caught collective operation timeout"
        elif line == 1272:
            text = "1: IndexKernel.cu:111: Assertion `index < sizes[i]` failed."
        candidate_lines.append(LogLine(line=line, text=text))
    early_fault_window = ContextWindow(
        window_id="w-early",
        selected_by="registry_match",
        start_line=1119,
        end_line=1252,
        seed_lines=(1159,),
        occurrence_group_ids=("og-critical",),
        lines=tuple(item for item in candidate_lines if item.line <= 1252),
    )
    primary_window = ContextWindow(
        window_id="w-primary",
        selected_by="registry_match",
        start_line=1202,
        end_line=1279,
        seed_lines=(1272,),
        occurrence_group_ids=("og-assert",),
        lines=tuple(item for item in candidate_lines if item.line >= 1202),
    )
    bundle = L0Bundle(
        log_path="/tmp/job.log",
        byte_size=1,
        line_count=1280,
        context_windows=(setup_window, early_fault_window, primary_window),
        deterministic_primary_candidate=FailureEvidence(
            fine_class="cuda_device_assert",
            policy_class="ambiguous",
            signature="bounds assertion",
            root_fingerprint="cuda_device_assert:bounds",
            fault_outcome="terminal",
            line=1272,
        ),
    )

    model_view = build_l0_model_facing_view(bundle)
    prompt_bundle = model_view.evidence_bundle
    candidate = next(
        window
        for window in prompt_bundle["context_windows"]
        if window["window_id"] == "w-early+w-primary"
    )
    candidate_prompt_lines = [item["line"] for item in candidate["lines"]]

    assert candidate["prompt_view"] == "bounded_excerpt"
    assert candidate["occurrence_group_ids"] == ["og-assert", "og-critical"]
    assert "pattern_ids" not in candidate
    assert candidate["source_line_count"] > 8
    assert candidate["lines_in_prompt"] > 8
    assert {1151, 1159, 1162, 1172, 1272}.issubset(candidate_prompt_lines)
    metrics = model_view.projection_metrics
    assert model_view.schema_version == "restart_agent_l0_model_view.v1"
    assert metrics["view_size"]["compact_json_characters"] > 0
    assert metrics["view_size"]["estimated_tokens"] > 0
    assert metrics["budget_utilization"]["context_window_slots"] == {
        "used": 3,
        "limit": 4,
        "utilization_pct": 75.0,
    }
    assert metrics["selection_counts"]["context_windows"] == {
        "available": 3,
        "selected": 3,
        "omitted": 0,
        "limit": 4,
        "projected_after_merge": 2,
    }
    assert metrics["compaction_counts"]["context_windows_merged"] == 1
    assert metrics["compaction_counts"]["context_windows_omitted"] == 0
    assert metrics["projection_integrity"]["status"] == "ok"
    assert metrics["projection_integrity"]["checks"]["decision_evidence_references_resolve"]


def test_candidate_anchors_include_high_signal_line_without_registry_match(tmp_path):
    log_path = tmp_path / "service_input.sanitized.log"
    log_path.write_text(
        "\n".join(
            [
                (
                    "7:  [2026-05-27 07:58:10.261551] iteration        1/       4 | "
                    "consumed samples:           64 | lm loss: 1.339357E+01 |"
                ),
                "INFO warmup continues",
                "1: CRITICAL:training.runtime:raising GPU error on cuda:0",
                (
                    "7:  [2026-05-27 07:58:11.261551] iteration        2/       4 | "
                    "consumed samples:          128 | lm loss: 1.239357E+01 |"
                ),
                "0: ProcessGroupNCCL watchdog caught collective operation timeout",
                "1: RuntimeError: CUDA error: device-side assert triggered",
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    anchor = next(item for item in bundle.candidate_anchors if item.line == 3)
    prompt_bundle = build_l0_model_facing_view(bundle).evidence_bundle
    prompt_anchor = next(item for item in prompt_bundle["candidate_anchors"] if item["line"] == 3)

    assert "high_signal" in anchor.sources
    assert anchor.taxonomy_match is None
    assert anchor.anchor_rank == "1"
    assert anchor.prior_observed_progress_line == 1
    assert anchor.later_observed_progress_line == 4
    assert anchor.prior_progress_rank == "7"
    assert anchor.later_progress_rank == "7"
    assert anchor.later_progress_rank_relation == "different_rank"
    assert anchor.later_observation_proves_recovery is False
    assert anchor.first_downstream_registry_match is not None
    assert anchor.first_downstream_registry_match.line == 5
    assert anchor.first_downstream_cascade is not None
    assert anchor.first_downstream_cascade.line == 5
    assert anchor.context_window_ids
    assert prompt_anchor["taxonomy_hint"] is None
    assert prompt_anchor["nearby_progress_observations"] == {
        "later_observation_proves_recovery": False,
        "later_observed_progress_line": 4,
        "later_progress_rank": "7",
        "later_progress_rank_relation": "different_rank",
        "prior_observed_progress_line": 1,
        "prior_progress_rank": "7",
    }
    assert prompt_anchor["first_downstream_registry_hint"]["line"] == 5
    assert prompt_anchor["first_downstream_cascade"]["line"] == 5
    assert prompt_anchor["covered_by_excerpt"] is True


def test_l4_maps_established_unrecoverable_workload_failure_to_stop(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "INFO training iteration 120",
                "RuntimeError: CUDA error: device-side assert triggered",
            ]
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "runtime_assertion",
            "policy_class": "user_failure",
            "signature": "device-side assert triggered",
            "root_fingerprint": "runtime_assertion:device_side_assert_terminal",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 2,
            "rank": None,
            "phase": "steady_mid",
        },
        "secondary_failures": [],
        "cascades": [],
        "evidence": [
            {
                "line": 2,
                "quote": "RuntimeError: CUDA error: device-side assert triggered",
                "supports": "primary_failure",
            }
        ],
        "justification": "The cited current-log error was selected as user failure.",
    }

    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    result = analyzer.analyze(
        {
            "log_path": str(log_path),
            "job_id": "job-1",
        }
    )
    payload = result.to_payload()

    assert payload["decision"] == Decision.STOP.value
    assert payload["decision_basis"] == DecisionBasis.WORKLOAD_UNRECOVERABLE.value
    assert payload["primary_failure"]["fine_class"] == "observed_exception"
    assert payload["primary_failure"]["policy_class"] == "ambiguous"
    assert payload["primary_failure"]["root_fingerprint"].startswith("observed:runtimeerror:")
    assert payload["result_provenance"]["evidence_source"] == "l1_model_audited"
    assert payload["result_provenance"]["model_contribution"] == "attempted_used"
    assert payload["result_provenance"]["result_quality"] == "normal"
    assert payload["result_provenance"]["nvrx_use"] == "eligible"
    assert analyzer.last_trace["context_mode"] == "tool_loop"
    assert analyzer.last_trace["result_provenance"] == payload["result_provenance"]
    assert analyzer.last_trace["l2_audit"]["used"] is True
    assert analyzer.last_trace["l2_audit"]["model_failure_domain"] == "workload"
    assert (
        analyzer.last_trace["l2_audit"]["model_retry_outlook_without_workload_change"]
        == "cannot_recover"
    )
    assert analyzer.last_trace["l4_policy"]["retry_policy"]["rule"] == ("workload_unrecoverable")
    assert analyzer.last_trace["l4_policy"]["retry_policy"]["allowed_retries"] == 0
    assert analyzer.last_trace["token_usage"]["prompt_tokens"] == 10
    assert analyzer.last_trace["token_usage"]["completion_tokens"] == 5
    assert analyzer.last_trace["token_usage"]["reasoning_tokens"] == 2
    assert analyzer.last_trace["token_usage"]["cached_prompt_tokens"] == 3
    assert analyzer.last_trace["token_usage"]["model_calls_with_usage"] == 1
    assert analyzer.last_trace["token_limit"]["hit"] is False


def test_deterministic_fallback_is_ready_while_l1_is_pending(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "INFO training iteration 120\nRuntimeError: CUDA error: device-side assert triggered\n",
        encoding="utf-8",
    )
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "runtime_assertion",
            "policy_class": "user_failure",
            "signature": "device-side assert triggered",
            "root_fingerprint": "runtime_assertion:device_side_assert_terminal",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 2,
            "rank": None,
            "phase": "steady_mid",
        },
        "secondary_failures": [],
        "cascades": [],
        "evidence": [
            {
                "line": 2,
                "quote": "RuntimeError: CUDA error: device-side assert triggered",
                "supports": "primary_failure",
            }
        ],
        "justification": "The current-log assertion is likely to repeat.",
    }
    extractor = _BlockingEvidenceExtractor(evidence)
    observed = []

    def on_fallback_ready(candidate):
        assert not extractor.started.is_set()
        assert not extractor.completed.is_set()
        observed.append(candidate)
        extractor.release.set()

    analyzer = RestartAgent(evidence_extractor=extractor)
    result = analyzer.analyze(
        {"log_path": str(log_path), "job_id": "job-1"},
        on_fallback_ready=on_fallback_ready,
    )

    assert len(observed) == 1
    fallback = observed[0]
    assert fallback.candidate_kind == "deterministic_fallback"
    assert fallback.l1_execution_status == "in_flight"
    assert fallback.result.decision == Decision.RESTART.value
    assert fallback.result.result_provenance["model_contribution"] == "pending_not_used"
    assert fallback.result.result_provenance["result_quality"] == "degraded"
    assert result.decision == Decision.STOP.value
    assert result.result_provenance["candidate_kind"] == "l1_enriched"
    candidates = analyzer.last_trace["decision_candidates"]
    assert candidates["fallback_ready"] is True
    assert candidates["enriched_ready"] is True
    assert candidates["selected"] == "l1_enriched"
    assert candidates["best_available_kind"] == "l1_enriched"
    assert candidates["best_available"]["result"]["decision"] == "STOP"
    assert candidates["deterministic_fallback"]["result"]["decision"] == "RESTART"
    assert candidates["l1_enriched"]["result"]["decision"] == "STOP"

    failing_callback_extractor = _BlockingEvidenceExtractor(evidence)

    def failing_callback(candidate):
        failing_callback_extractor.release.set()
        raise RuntimeError("candidate sink is unavailable")

    failing_callback_analyzer = RestartAgent(evidence_extractor=failing_callback_extractor)
    callback_result = failing_callback_analyzer.analyze(
        {"log_path": str(log_path), "job_id": "job-1"},
        on_fallback_ready=failing_callback,
    )

    assert callback_result.decision == Decision.STOP.value
    assert (
        failing_callback_analyzer.last_trace["anomalies"]["fallback_callback_error"]
        == "RuntimeError: candidate sink is unavailable"
    )


def test_deterministic_fallback_can_apply_history_before_l1_finishes(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        (
            "7: [2026-05-27 07:58:10.261551] iteration 120/400 | "
            "consumed samples: 7680 | lm loss: 1.3E+01 |\n"
            "RuntimeError: CUDA error: device-side assert triggered\n"
        ),
        encoding="utf-8",
    )
    bundle = build_l0_bundle(str(log_path))
    current = bundle.deterministic_primary_candidate
    assert current is not None
    history = [
        {
            "job_id": "job-1",
            "cycle_id": 1,
            "root_fingerprint": current.root_fingerprint,
            "primary_fault_outcome": "terminal",
            "last_completed_step": 120,
        },
        {
            "job_id": "job-1",
            "cycle_id": 2,
            "root_fingerprint": current.root_fingerprint,
            "primary_fault_outcome": "terminal",
            "last_completed_step": 120,
        },
    ]
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "runtime_assertion",
            "policy_class": "ambiguous",
            "signature": "device-side assert triggered",
            "root_fingerprint": current.root_fingerprint,
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "line": 2,
            "phase": "steady_mid",
        },
        "evidence": [
            {
                "line": 2,
                "quote": "RuntimeError: CUDA error: device-side assert triggered",
                "supports": "primary_failure",
            }
        ],
        "justification": "The assertion is ambiguous in one cycle.",
    }
    extractor = _BlockingEvidenceExtractor(evidence)
    observed = []
    prior_attempts = _prior_attempt_view(*history)

    def on_fallback_ready(candidate):
        observed.append(candidate)
        for record in history:
            record["root_fingerprint"] = "caller-mutated-after-publication"
        extractor.release.set()

    result = RestartAgent(evidence_extractor=extractor).analyze(
        {
            "log_path": str(log_path),
            "job_id": "job-1",
            "cycle_id": 3,
        },
        l0_bundle=bundle,
        prior_attempts=prior_attempts,
        on_fallback_ready=on_fallback_ready,
    )

    assert observed[0].result.decision == Decision.RESTART.value
    assert observed[0].result.decision_basis == DecisionBasis.GENERAL_RETRY_AVAILABLE.value
    assert observed[0].result.result_provenance["history_contribution"] == ("checked_no_effect")
    assert observed[0].result.result_provenance["result_quality"] == "degraded"
    assert result.decision == Decision.RESTART.value
    assert result.decision_basis == DecisionBasis.GENERAL_RETRY_AVAILABLE.value


def test_l4_keeps_retryable_port_with_wait_precondition_ambiguous(tmp_path):
    log_path = tmp_path / "job.log"
    failure_line = "OSError: [Errno 98] Address already in use"
    log_path.write_text(f"{failure_line}\n", encoding="utf-8")
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "socket_bind_address_in_use",
            "signature": failure_line,
            "proposed_root_fingerprint": None,
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 1,
            "rank": None,
            "phase": "setup",
        },
        "root_cause_assessment": {
            "summary": "A workload server found its selected port occupied.",
            "plausible_causes": ["mutable external socket state"],
            "persistence_evidence": [],
            "transient_alternatives": ["normal teardown or delay may release the port"],
        },
        "model_recovery_assessment": {
            "failure_domain": "workload",
            "next_attempt_same_failure_likelihood": "likely",
            "current_attempt_persistence_evidence": "none",
            "retry_recovery_path": "wait_or_teardown",
            "confidence": 60,
            "rationale": "The same port will be requested, but its future occupancy is unknown.",
            "supporting_evidence_lines": [1],
        },
        "related_failures": [],
        "evidence": [{"line": 1, "quote": failure_line, "supports": "primary_failure"}],
        "justification": "The socket bind failure is the initiating failure.",
    }

    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    payload = analyzer.analyze({"log_path": str(log_path)}).to_payload()

    assert payload["decision"] == Decision.RESTART.value
    assert payload["decision_basis"] == DecisionBasis.RETRY_RECOVERY_AVAILABLE.value
    assert payload["primary_failure"]["policy_class"] == "ambiguous"
    assert "recovery_requirement" not in payload["model_recovery_assessment"]
    assert "model_recovery_requirement" not in analyzer.last_trace["l2_audit"]
    assert analyzer.last_trace["l4_policy"]["retry_policy"]["rule"] == "bounded_retry"


def test_model_recovery_confidence_is_preserved_without_recurrence_score(tmp_path):
    log_path = tmp_path / "job.log"
    failure_line = "OSError: [Errno 98] Address already in use"
    log_path.write_text(f"{failure_line}\n", encoding="utf-8")
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "socket_bind_address_in_use",
            "signature": failure_line,
            "proposed_root_fingerprint": None,
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 1,
            "rank": None,
            "phase": "setup",
        },
        "root_cause_assessment": {
            "summary": "The selected port was occupied.",
            "plausible_causes": ["mutable socket state"],
            "persistence_evidence": [],
            "transient_alternatives": ["normal teardown may release the port"],
        },
        "model_recovery_assessment": {
            "failure_domain": "workload",
            "next_attempt_same_failure_likelihood": "unknown",
            "current_attempt_persistence_evidence": "none",
            "retry_recovery_path": "wait_or_teardown",
            "confidence": 80,
            "rationale": "A restart delay may satisfy the retry precondition.",
            "supporting_evidence_lines": [1],
        },
        "related_failures": [],
        "evidence": [{"line": 1, "quote": failure_line, "supports": "primary_failure"}],
        "justification": "The bind failure ended setup.",
    }
    extractor = _FakeEvidenceExtractor(evidence)

    analyzer = RestartAgent(evidence_extractor=extractor)
    payload = analyzer.analyze({"log_path": str(log_path)}).to_payload()

    assert payload["decision"] == Decision.RESTART.value
    assert payload["result_provenance"]["model_contribution"] == "attempted_used"
    assert payload["result_provenance"]["result_quality"] == "normal"
    assert payload["result_provenance"]["nvrx_use"] == "eligible"
    assert payload["model_recovery_assessment"]["failure_domain"]["confidence"] == 80
    assert "recurrence_confidence" not in payload["model_recovery_assessment"]
    audit = analyzer.last_trace["l2_audit"]
    assert audit["field_findings"] == {}
    assert analyzer.last_trace["layers"]["L2"]["material_finding_count"] == 0


@pytest.mark.parametrize(
    ("persistence_strength", "expected_decision", "expected_rule"),
    [
        (
            "affirmative",
            Decision.STOP.value,
            "workload_unrecoverable",
        ),
        ("circumstantial", Decision.RESTART.value, "general_retry"),
    ],
)
def test_immediate_stop_requires_affirmative_grounded_persistence_evidence(
    tmp_path,
    persistence_strength,
    expected_decision,
    expected_rule,
):
    log_path = tmp_path / "job.log"
    failure_line = "PermissionError: [Errno 13] Permission denied: '/shared/cache.lock'"
    log_path.write_text(f"{failure_line}\n", encoding="utf-8")
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "filesystem_permission_denied",
            "signature": failure_line,
            "proposed_root_fingerprint": None,
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 1,
            "rank": None,
            "phase": "setup",
        },
        "root_cause_assessment": {
            "summary": "The process could not create a cache lock.",
            "plausible_causes": ["filesystem permission or ACL state"],
            "persistence_evidence": ["the denied path is unchanged"],
            "transient_alternatives": ["the access state may be repaired externally"],
        },
        "model_recovery_assessment": {
            "failure_domain": "workload",
            "next_attempt_same_failure_likelihood": "likely",
            "recurrence_confidence": 80,
            "current_attempt_persistence_evidence": persistence_strength,
            "retry_recovery_path": "external_intervention",
            "confidence": 80,
            "rationale": "The access condition must change before retry.",
            "supporting_evidence_lines": [1],
        },
        "related_failures": [],
        "evidence": [{"line": 1, "quote": failure_line, "supports": "primary_failure"}],
        "justification": "The permission error ended setup.",
    }

    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    payload = analyzer.analyze({"log_path": str(log_path)}).to_payload()

    assert payload["decision"] == expected_decision
    assert payload["retry_policy"]["rule"] == expected_rule
    assert payload["retry_policy"]["recovery_assessment_policy_grounded"] is True


def test_l2_flags_unverified_path_owner_without_treating_rank_fanout_as_persistence(
    tmp_path,
):
    log_path = tmp_path / "job.log"
    lock_path = "/lustre/fsw/users/wdai/hf_home/datasets/cache.lock"
    failure_line = f"PermissionError: [Errno 13] Permission denied: '{lock_path}'"
    log_path.write_text(
        "\n".join(
            [
                "data_cache_path ........ /lustre/fsw/users/rwaleffe/cache",
                "load ................... /lustre/fsw/users/wdai/checkpoint",
                f"[rank80]: {failure_line}",
                f"[rank96]: {failure_line}",
            ]
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "filesystem_permission_denied",
            "signature": failure_line,
            "proposed_root_fingerprint": None,
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 3,
            "rank": "80",
            "phase": "setup",
        },
        "root_cause_assessment": {
            "summary": "The lock belongs to wdai while the job runs as a different user.",
            "status": "supported_but_unconfirmed",
            "plausible_causes": ["cross-user cache permissions"],
            "persistence_evidence": ["multiple ranks prove persistence"],
            "transient_alternatives": ["external ACL repair"],
            "missing_evidence": ["effective UID and ownership metadata"],
        },
        "model_recovery_assessment": {
            "failure_domain": "workload",
            "next_attempt_same_failure_likelihood": "likely",
            "recurrence_confidence": 90,
            "current_attempt_persistence_evidence": "affirmative",
            "retry_recovery_path": "external_intervention",
            "confidence": 90,
            "rationale": "The same permission error occurred across multiple ranks.",
            "supporting_evidence_lines": [3, 4],
        },
        "related_failures": [],
        "evidence": [
            {"line": 3, "quote": f"[rank80]: {failure_line}", "supports": "primary_failure"},
            {"line": 4, "quote": f"[rank96]: {failure_line}", "supports": "persistence"},
        ],
        "justification": "The denied cache lock ended setup on the ranks.",
    }

    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    payload = analyzer.analyze({"log_path": str(log_path)}).to_payload()
    audit = analyzer.last_trace["l2_audit"]

    assert payload["decision"] == Decision.STOP.value
    assert payload["model_recovery_assessment"]["failure_domain"]["status"] == (
        "established_by_current_log"
    )
    assert payload["retry_policy"]["rule"] == "workload_unrecoverable"
    assert "model_recovery_assessment" not in audit["field_finding_codes"]
    assert audit["field_finding_codes"]["root_cause_assessment"] == [
        "path_namespace_identity_unverified"
    ]
    assert audit.get("grounding_adjustments", []) == []
    assert all("persistence" not in item["code"] for item in audit["findings"])


def test_l4_stops_for_established_unrecoverable_workload_claims(tmp_path):
    log_path = tmp_path / "job.log"
    failure_line = "torch.AcceleratorError: CUDA error: out of memory"
    log_path.write_text(f"{failure_line}\n", encoding="utf-8")
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "cuda_out_of_memory",
            "signature": failure_line,
            "proposed_root_fingerprint": None,
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 1,
            "rank": "0",
            "phase": "setup",
        },
        "root_cause_assessment": {
            "summary": "The workload exhausted its allocated accelerator memory.",
            "plausible_causes": ["insufficient workload memory headroom"],
            "persistence_evidence": ["the workload crossed its memory budget"],
            "transient_alternatives": ["process teardown may clear allocator state"],
        },
        "model_recovery_assessment": {
            "failure_domain": "workload",
            "next_attempt_same_failure_likelihood": "likely",
            "current_attempt_persistence_evidence": "affirmative",
            "retry_recovery_path": "workload_change",
            "confidence": 85,
            "rationale": (
                "A restart may clear allocator state, but the workload must restore "
                "reliable memory headroom."
            ),
            "supporting_evidence_lines": [1],
        },
        "related_failures": [],
        "evidence": [{"line": 1, "quote": failure_line, "supports": "primary_failure"}],
        "justification": "The workload OOM is the initiating failure.",
    }

    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    payload = analyzer.analyze({"log_path": str(log_path)}).to_payload()

    assert payload["decision"] == Decision.STOP.value
    assessment = payload["model_recovery_assessment"]
    assert assessment["retry_outlook_without_workload_change"]["value"] == "cannot_recover"
    assert assessment["retry_outlook_without_workload_change"]["status"] == (
        "established_by_current_log"
    )
    assert "recovery_requirement" not in assessment
    assert payload["retry_policy"]["rule"] == "workload_unrecoverable"
    assert payload["retry_policy"]["allowed_retries"] == 0


def test_l4_does_not_stop_on_circumstantial_persistence_hypothesis(tmp_path):
    log_path = tmp_path / "job.log"
    failure_line = (
        "RuntimeError: worker was killed. It is possible that shared memory is exhausted."
    )
    log_path.write_text(failure_line + "\n", encoding="utf-8")
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "worker_failure",
            "signature": "worker was killed",
            "proposed_root_fingerprint": None,
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 1,
            "rank": "0",
            "phase": "steady_mid",
        },
        "root_cause_assessment": {
            "summary": "Resource exhaustion is only a diagnostic hypothesis.",
            "status": "hypothesis_only",
            "plausible_causes": ["resource exhaustion"],
            "persistence_evidence": [],
            "transient_alternatives": ["transient worker or external-resource fault"],
            "missing_evidence": ["direct resource-capacity evidence"],
        },
        "model_recovery_assessment": {
            "failure_domain": "workload",
            "next_attempt_same_failure_likelihood": "likely",
            "recurrence_confidence": 70,
            "current_attempt_persistence_evidence": "circumstantial",
            "retry_recovery_path": "workload_change",
            "confidence": 70,
            "rationale": "The diagnostic suggests a workload resource issue.",
            "supporting_evidence_lines": [1],
        },
        "related_failures": [],
        "evidence": [{"line": 1, "quote": failure_line, "supports": "primary_failure"}],
        "justification": "The worker failure terminated the attempt.",
    }

    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    payload = analyzer.analyze({"log_path": str(log_path)}).to_payload()

    assert payload["decision"] == Decision.RESTART.value
    assert payload["primary_failure"]["policy_class"] == "ambiguous"
    assert payload["retry_policy"]["rule"] == "general_retry"


def test_l4_preserves_retry_recovery_for_ordinary_retry_recovery_path(tmp_path):
    log_path = tmp_path / "job.log"
    failure_line = "torch.OutOfMemoryError: CUDA out of memory"
    log_path.write_text(f"{failure_line}\n", encoding="utf-8")
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "cuda_oom",
            "signature": failure_line,
            "proposed_root_fingerprint": None,
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 1,
            "rank": "0",
            "phase": "setup",
        },
        "root_cause_assessment": {
            "summary": "The workload exhausted accelerator memory.",
            "plausible_causes": ["insufficient workload memory headroom"],
            "persistence_evidence": ["the workload crossed its memory budget"],
            "transient_alternatives": ["process teardown may clear allocator state"],
        },
        "model_recovery_assessment": {
            "failure_domain": "workload",
            "next_attempt_same_failure_likelihood": "likely",
            "recurrence_confidence": 75,
            "current_attempt_persistence_evidence": "none",
            "retry_recovery_path": "ordinary_retry",
            "confidence": 75,
            "rationale": (
                "A fresh process may recover once, while durable workload change "
                "is still required."
            ),
            "supporting_evidence_lines": [1],
        },
        "related_failures": [],
        "evidence": [{"line": 1, "quote": failure_line, "supports": "primary_failure"}],
        "justification": "The workload OOM is the initiating failure.",
    }

    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    payload = analyzer.analyze({"log_path": str(log_path)}).to_payload()

    assert payload["decision"] == Decision.RESTART.value
    assert payload["decision_basis"] == DecisionBasis.RETRY_RECOVERY_AVAILABLE.value
    assert payload["primary_failure"]["policy_class"] == "ambiguous"
    assert "recovery_requirement" not in payload["model_recovery_assessment"]
    assert payload["retry_policy"]["rule"] == "bounded_retry"


def test_l4_keeps_retry_recoverable_workload_failure_restartable(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "INFO training iteration 120",
                "RuntimeError: CUDA error: device-side assert triggered",
            ]
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "runtime_assertion",
            "signature": "device-side assert triggered",
            "proposed_root_fingerprint": "runtime_assertion:device_side_assert_terminal",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 2,
            "rank": None,
            "phase": "steady_mid",
        },
        "root_cause_assessment": {
            "summary": "The runtime assertion may be handled by workload retry logic.",
            "plausible_causes": ["retryable workload data"],
            "persistence_evidence": [],
            "transient_alternatives": ["workload retry recovery"],
        },
        "model_recovery_assessment": {
            "failure_domain": "workload",
            "next_attempt_same_failure_likelihood": "likely",
            "current_attempt_persistence_evidence": "none",
            "retry_recovery_path": "workload_managed_retry_grace",
            "confidence": 80,
            "rationale": "The workload is expected to skip the item after retries.",
            "supporting_evidence_lines": [2],
        },
        "related_failures": [],
        "evidence": [
            {
                "line": 2,
                "quote": "RuntimeError: CUDA error: device-side assert triggered",
                "supports": "primary_failure",
            }
        ],
        "justification": "The cited current-log error was selected as user failure.",
    }

    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    result = analyzer.analyze(
        {
            "log_path": str(log_path),
            "job_id": "job-1",
        }
    )
    payload = result.to_payload()

    assert payload["decision"] == Decision.RESTART.value
    assert payload["decision_basis"] == DecisionBasis.RETRY_RECOVERY_AVAILABLE.value
    assert payload["primary_failure"]["fine_class"] == "observed_exception"
    assert payload["primary_failure"]["policy_class"] == "ambiguous"
    assert analyzer.last_trace["l2_audit"]["used"] is True
    assert payload["model_recovery_assessment"]["retry_outlook_without_workload_change"][
        "value"
    ] == ("may_recover")
    assert "recovery_requirement" not in payload["model_recovery_assessment"]
    assert payload["retry_policy"]["rule"] == "bounded_retry"
    assert "workload_managed_retry_grace" not in json.dumps(payload, sort_keys=True)


def test_l4_maps_infrastructure_domain_to_restart(tmp_path):
    log_path = tmp_path / "job.log"
    failure_line = "RuntimeError: GPU has fallen off the bus"
    log_path.write_text(f"{failure_line}\n", encoding="utf-8")
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "gpu_device_loss",
            "signature": "GPU has fallen off the bus",
            "proposed_root_fingerprint": None,
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 1,
            "rank": None,
            "phase": "steady_mid",
        },
        "root_cause_assessment": {
            "summary": "The accelerator became unavailable.",
            "plausible_causes": ["GPU or PCIe infrastructure fault"],
            "persistence_evidence": [],
            "transient_alternatives": ["replacement placement"],
        },
        "model_recovery_assessment": {
            "failure_domain": "infrastructure",
            "next_attempt_same_failure_likelihood": "plausible",
            "current_attempt_persistence_evidence": "none",
            "retry_recovery_path": "ordinary_retry",
            "confidence": 95,
            "rationale": "A restart can move the workload to healthy infrastructure.",
            "supporting_evidence_lines": [1],
        },
        "related_failures": [],
        "evidence": [{"line": 1, "quote": failure_line, "supports": "primary_failure"}],
        "justification": "The device-loss exception is the initiating failure.",
    }

    payload = (
        RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
        .analyze({"log_path": str(log_path)})
        .to_payload()
    )

    assert payload["decision"] == Decision.RESTART.value
    assert payload["primary_failure"]["policy_class"] == "ambiguous"
    assert payload["retry_policy"]["rule"] == "bounded_retry"


def test_l4_does_not_immediately_stop_established_infrastructure_failure(
    tmp_path,
):
    log_path = tmp_path / "job.log"
    failure_line = "RuntimeError: GPU has fallen off the bus"
    log_path.write_text(f"{failure_line}\n", encoding="utf-8")
    evidence = {
        "primary_failure": {
            "fine_class": "gpu_device_loss",
            "signature": failure_line,
            "proposed_root_fingerprint": None,
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 1,
            "rank": None,
            "phase": "steady_mid",
        },
        "root_cause_assessment": {
            "summary": "The current accelerator became unavailable.",
            "plausible_causes": ["GPU or PCIe infrastructure fault"],
            "persistence_evidence": ["the device-loss event terminated this attempt"],
            "transient_alternatives": ["replacement placement or recovered hardware"],
        },
        "model_recovery_assessment": {
            "failure_domain": "infrastructure",
            "next_attempt_same_failure_likelihood": "likely",
            "current_attempt_persistence_evidence": "affirmative",
            "retry_recovery_path": "external_intervention",
            "confidence": 90,
            "rationale": "The device fault may require intervention.",
            "supporting_evidence_lines": [1],
        },
        "evidence": [{"line": 1, "quote": failure_line, "supports": "primary_failure"}],
        "justification": "The device-loss exception is the initiating failure.",
    }

    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    payload = analyzer.analyze({"log_path": str(log_path)}).to_payload()

    assert payload["decision"] == Decision.RESTART.value
    assert payload["primary_failure"]["policy_class"] == "ambiguous"
    assert payload["model_recovery_assessment"]["failure_domain"] == {
        "value": "infrastructure",
        "status": "established_by_current_log",
        "confidence": 90,
    }
    assert payload["retry_policy"]["rule"] == "general_retry"
    assert analyzer.last_trace["l2_audit"]["recovery_field_audits"] == []


def test_l4_uses_general_retry_for_single_immutable_infrastructure_failure(tmp_path):
    log_path = tmp_path / "job.log"
    failure_line = "Xid 79: GPU reset required before reuse"
    log_path.write_text(f"{failure_line}\n", encoding="utf-8")
    evidence = {
        "primary_failure": {
            "fine_class": "gpu_reset_required",
            "signature": failure_line,
            "proposed_root_fingerprint": None,
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 1,
            "rank": None,
            "phase": "steady_mid",
        },
        "root_cause_assessment": {
            "summary": "The assigned GPU requires an external reset.",
            "plausible_causes": ["latched accelerator fault"],
            "persistence_evidence": ["the driver explicitly requires a reset"],
            "transient_alternatives": [],
        },
        "model_recovery_assessment": {
            "failure_domain": "infrastructure",
            "next_attempt_same_failure_likelihood": "likely",
            "current_attempt_persistence_evidence": "affirmative",
            "retry_recovery_path": "external_intervention",
            "confidence": 95,
            "rationale": "The immutable allocation requires a GPU reset before retry.",
            "supporting_evidence_lines": [1],
        },
        "evidence": [{"line": 1, "quote": failure_line, "supports": "primary_failure"}],
        "justification": "The reset-required Xid is the initiating failure.",
    }

    analyzer = RestartAgent(
        evidence_extractor=_FakeEvidenceExtractor(evidence),
        restart_environment_context={
            "hardware_allocation_may_change": False,
            "external_service_state_may_change": False,
        },
    )
    payload = analyzer.analyze({"log_path": str(log_path)}).to_payload()

    assert payload["decision"] == Decision.RESTART.value
    assert payload["decision_basis"] == DecisionBasis.GENERAL_RETRY_AVAILABLE.value
    assert payload["primary_failure"]["policy_class"] == "ambiguous"
    assert payload["retry_policy"]["rule"] == "general_retry"


def test_history_can_stop_repeated_infrastructure_failure_without_progress(tmp_path):
    log_path = tmp_path / "job.log"
    failure_line = "RuntimeError: GPU has fallen off the bus"
    log_path.write_text(
        (
            "7: [2026-05-27 07:58:10] iteration 120/400 | "
            f"consumed samples: 7680 |\n{failure_line}\n"
        ),
        encoding="utf-8",
    )
    bundle = build_l0_bundle(str(log_path))
    current = bundle.deterministic_primary_candidate
    assert current is not None
    evidence = {
        "primary_failure": {
            "fine_class": "gpu_device_loss",
            "signature": failure_line,
            "proposed_root_fingerprint": current.root_fingerprint,
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 2,
            "rank": None,
            "phase": "steady_mid",
        },
        "root_cause_assessment": {
            "summary": "The accelerator became unavailable.",
            "plausible_causes": ["GPU or PCIe infrastructure fault"],
            "persistence_evidence": [],
            "transient_alternatives": ["replacement placement"],
        },
        "model_recovery_assessment": {
            "failure_domain": "infrastructure",
            "next_attempt_same_failure_likelihood": "plausible",
            "current_attempt_persistence_evidence": "none",
            "retry_recovery_path": "ordinary_retry",
            "confidence": 80,
            "rationale": "A fresh allocation may recover.",
            "supporting_evidence_lines": [2],
        },
        "evidence": [{"line": 2, "quote": failure_line, "supports": "primary_failure"}],
        "justification": "The device-loss exception is the initiating failure.",
    }
    history = [
        {
            "job_id": "job-1",
            "cycle_id": 1,
            "root_fingerprint": current.root_fingerprint,
            "primary_fault_outcome": "terminal",
            "last_completed_step": 120,
        },
        {
            "job_id": "job-1",
            "cycle_id": 2,
            "root_fingerprint": current.root_fingerprint,
            "primary_fault_outcome": "terminal",
            "last_completed_step": 120,
        },
    ]

    payload = (
        RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
        .analyze(
            {
                "log_path": str(log_path),
                "job_id": "job-1",
                "cycle_id": 3,
            },
            l0_bundle=bundle,
            prior_attempts=_prior_attempt_view(*history),
        )
        .to_payload()
    )

    assert payload["decision"] == Decision.STOP.value
    assert payload["decision_basis"] == DecisionBasis.RETRY_BUDGET_EXHAUSTED.value
    assert payload["primary_failure"]["policy_class"] == "ambiguous"
    assert payload["retry_policy"]["rule"] == "bounded_retry"
    assert payload["result_provenance"]["history_contribution"] == ("retry_budget_exhausted")


def test_l1_recovery_assessment_drives_policy_without_model_policy_class(tmp_path):
    log_path = tmp_path / "job.log"
    failure_line = (
        "180: [rank180]: RuntimeError: Rank 180, node nvl72005-T14, "
        "device 0, iteration 670314: Unexpected result inf "
        "(message='found Inf in local grad norm for bucket #0 in "
        "backward pass before data-parallel communication collective')"
    )
    log_path.write_text(
        "\n".join(
            [
                "6143: iteration   670310/  794728 | consumed samples: 2059192320 |",
                failure_line,
            ]
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "numeric_instability",
            "policy_class": "user_failure",
            "signature": "inf_in_grad_norm",
            "root_fingerprint": "numeric_instability:inf_in_grad_norm",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 2,
            "rank": "180",
            "phase": "steady_mid",
        },
        "secondary_failures": [],
        "cascades": [],
        "evidence": [
            {
                "line": 2,
                "quote": failure_line,
                "supports": "primary_failure",
            }
        ],
        "justification": "The model selected an unmapped L1-only numeric instability.",
    }

    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    result = analyzer.analyze({"log_path": str(log_path), "job_id": "job-1"})
    payload = result.to_payload()

    assert payload["decision"] == Decision.STOP.value
    assert payload["decision_basis"] == DecisionBasis.WORKLOAD_UNRECOVERABLE.value
    assert payload["primary_failure"]["fine_class"] == "observed_exception"
    assert payload["primary_failure"]["policy_class"] == "ambiguous"
    assert payload["primary_failure"]["root_fingerprint"].startswith("observed:runtimeerror:")
    assert payload["primary_failure"]["root_fingerprint_source"] == ("observed_exception")
    assert (
        analyzer.last_trace["l2_audit"]["stable_root_fingerprint"]
        == payload["primary_failure"]["root_fingerprint"]
    )
    assert analyzer.last_trace["l2_audit"]["used"] is True
    assert analyzer.last_trace["l2_audit"]["model_failure_domain"] == "workload"
    assert analyzer.last_trace["l4_policy"]["retry_policy"]["rule"] == ("workload_unrecoverable")


def test_l1_retries_retryable_provider_failure_then_uses_valid_evidence(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "INFO training iteration 120",
                "ValueError: invalid config option tensor_model_parallel_size",
            ]
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "python_user_exception",
            "policy_class": "user_failure",
            "signature": "ValueError:",
            "root_fingerprint": "python_user_exception:ValueError:",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 2,
            "rank": None,
            "phase": "setup",
        },
        "secondary_failures": [],
        "cascades": [],
        "evidence": [
            {
                "line": 2,
                "quote": "ValueError: invalid config option tensor_model_parallel_size",
                "supports": "primary_failure",
            }
        ],
        "justification": "The cited current-log error was selected as user failure.",
    }

    analyzer = RestartAgent(evidence_extractor=_RetryOnceEvidenceExtractor(evidence))
    result = analyzer.analyze(
        {
            "log_path": str(log_path),
            "job_id": "job-1",
        }
    )
    payload = result.to_payload()
    l1_trace = analyzer.last_trace["l1"]

    assert payload["decision"] == Decision.STOP.value
    assert payload["decision_basis"] == DecisionBasis.WORKLOAD_UNRECOVERABLE.value
    assert analyzer.last_trace["l2_audit"]["used"] is True
    assert len(l1_trace["model_calls"]) == 2
    assert l1_trace["model_calls"][0]["success"] is False
    assert l1_trace["model_calls"][0]["retry_scheduled"] is True
    assert l1_trace["model_calls"][1]["success"] is True
    assert analyzer.last_trace["timing"]["l1_failed_model_calls"] == 1
    assert analyzer.last_trace["timing"]["l1_retried_model_calls"] == 1
    assert analyzer.last_trace["anomalies"]["provider_retries"] == 1
    assert payload["result_provenance"]["l1_execution_status"] == "degraded"
    assert payload["result_provenance"]["l1_execution_issues"] == [
        "model_call_failed",
        "retry_used",
        "provider_http_error",
    ]
    assert analyzer.last_trace["layers"]["L1"]["execution_status"] == "degraded"
    request_events = [
        event
        for event in l1_trace["interaction_transcript"]
        if event.get("event_type") == "model_request"
    ]
    assert len(request_events) == 2
    assert request_events[0]["request_body"]["messages"][0]["role"] == "system"
    assert request_events[0]["advertised_tool_schemas"] == []
    assert request_events[0]["payload_sha256"].startswith("sha256:")
    assert request_events[0]["payload_bytes"] > 0
    assert request_events[0]["truncated"] is False
    assert "test-key" not in json.dumps(request_events, sort_keys=True)


def test_http_504_is_classified_as_endpoint_timeout():
    assert _is_http_timeout_status(504) is True
    assert _is_http_timeout_status(500) is False


def test_provider_reported_timing_extracts_only_valid_proxy_headers():
    timing = _provider_reported_timing(
        {
            "X-LiteLLM-Timing-LLM-API-MS": "156.64",
            "x-litellm-timing-pre-processing-ms": "1.839",
            "x-litellm-timing-post-processing-ms": "0.69",
            "x-litellm-timing-message-copy-ms": "0.0017",
            "x-litellm-timing-ignored-ms": "22",
        }
    )

    assert timing == {
        "source": "response_headers",
        "downstream_llm_api_ms": 156.64,
        "proxy_pre_processing_ms": 1.839,
        "proxy_post_processing_ms": 0.69,
        "proxy_message_copy_ms": 0.0017,
    }
    assert (
        _provider_reported_timing(
            {
                "x-litellm-timing-llm-api-ms": "not-a-number",
                "x-litellm-timing-pre-processing-ms": "-1",
            }
        )
        == {}
    )


def test_l1_http_timeout_is_clamped_to_remaining_analysis_budget(monkeypatch):
    observed = {}

    class _Response:
        headers = {"x-litellm-timing-llm-api-ms": "156.64"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        observed["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(
        "nvidia_resiliency_ext.attribution.restart_agent.l1.openai_compatible.urllib.request.urlopen",
        fake_urlopen,
    )
    extractor = LlmEvidenceExtractor(
        LlmConfig(api_key="test-key", timeout_seconds=120, tools_enabled=False)
    )
    deadline = time.monotonic() + 0.25

    _, call_record = extractor._call_model(
        api_key="test-key",
        messages=[{"role": "user", "content": "test"}],
        include_tools=False,
        model_turn=1,
        deadline_monotonic=deadline,
    )

    assert 0 < observed["timeout"] <= 0.25
    assert call_record["effective_request_timeout_seconds"] == observed["timeout"]
    assert call_record["configured_request_timeout_seconds"] == 120
    assert call_record["provider_reported_timing"] == {
        "source": "response_headers",
        "downstream_llm_api_ms": 156.64,
    }


def test_l1_does_not_start_provider_call_after_analysis_deadline(monkeypatch):
    called = False

    def fake_urlopen(request, timeout):
        nonlocal called
        called = True
        raise AssertionError("provider request must not start after deadline")

    monkeypatch.setattr(
        "nvidia_resiliency_ext.attribution.restart_agent.l1.openai_compatible.urllib.request.urlopen",
        fake_urlopen,
    )
    extractor = LlmEvidenceExtractor(LlmConfig(api_key="test-key"))

    with pytest.raises(LlmCallError) as error:
        extractor._call_model(
            api_key="test-key",
            messages=[{"role": "user", "content": "test"}],
            include_tools=False,
            model_turn=1,
            deadline_monotonic=time.monotonic() - 1,
        )

    assert called is False
    assert error.value.call_record["error_type"] == "analysis_deadline_exceeded"
    assert error.value.call_record["retryable"] is False


def test_l1_deadline_prevents_retry_after_slow_provider_failure(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: failure\n", encoding="utf-8")

    class _SlowRetryExtractor(LlmEvidenceExtractor):
        def __init__(self):
            super().__init__(
                LlmConfig(
                    api_key="test-key",
                    tools_enabled=False,
                    max_retries=5,
                    retry_backoff_seconds=0.05,
                )
            )
            self.calls = 0

        def _call_model(self, **kwargs):
            self.calls += 1
            time.sleep(0.02)
            raise LlmCallError(
                "HTTP 502: bad gateway",
                {
                    "layer": "L1",
                    "model": self._config.model,
                    "model_turn": kwargs["model_turn"],
                    "attempt": kwargs["attempt"],
                    "max_retries": kwargs["max_retries"],
                    "success": False,
                    "latency_s": 0.02,
                    "retryable": True,
                    "retry_scheduled": False,
                    "timeout": False,
                    "error_type": "http_error",
                    "error": "HTTP 502",
                },
            )

    extractor = _SlowRetryExtractor()
    bundle = build_l0_bundle(str(log_path))
    result = extractor.extract_evidence(
        build_l1_evidence_context(
            bundle,
            build_l0_model_facing_view(bundle),
            LogSnapshot.read(log_path),
        ),
        deadline_monotonic=time.monotonic() + 0.01,
    )

    assert extractor.calls == 1
    assert result.success is False
    assert result.anomalies["deadline_exceeded"] is True
    assert result.model_calls[0]["retry_blocked_by_deadline"] is True


def test_l1_tool_loop_stops_at_configured_round_limit(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: failure\n", encoding="utf-8")

    class _AlwaysToolExtractor(LlmEvidenceExtractor):
        def __init__(self):
            super().__init__(
                LlmConfig(
                    api_key="test-key",
                    max_retries=0,
                    max_tool_rounds=2,
                )
            )
            self.calls = 0

        def _call_model(self, **kwargs):
            self.calls += 1
            return (
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": f"tool-{self.calls}",
                                        "type": "function",
                                        "function": {
                                            "name": "overview",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                },
                {
                    "layer": "L1",
                    "model": self._config.model,
                    "model_turn": kwargs["model_turn"],
                    "attempt": kwargs["attempt"],
                    "max_retries": kwargs["max_retries"],
                    "success": True,
                    "latency_s": 0.001,
                    "finish_reason": "tool_calls",
                    "usage": {"total_tokens": 10},
                    "tools_advertised": kwargs["include_tools"],
                },
            )

    extractor = _AlwaysToolExtractor()
    bundle = build_l0_bundle(str(log_path))
    result = extractor.extract_evidence(
        build_l1_evidence_context(
            bundle,
            build_l0_model_facing_view(bundle),
            LogSnapshot.read(log_path),
        ),
        deadline_monotonic=time.monotonic() + 5,
    )

    assert extractor.calls == 3
    assert len(result.tool_calls) == 2
    assert result.success is False
    assert result.anomalies["forced_final_evidence_call"] is True


def test_l1_repairs_incomplete_contract_once_without_tools(tmp_path):
    log_path = tmp_path / "job.log"
    log_line = "ValueError: invalid config option tensor_model_parallel_size"
    log_path.write_text(log_line + "\n", encoding="utf-8")
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "python_user_exception",
            "policy_class": "user_failure",
            "signature": "ValueError:",
            "root_fingerprint": "python_user_exception:valueerror",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 1,
            "rank": None,
            "phase": "setup",
        },
        "secondary_failures": [],
        "cascades": [],
        "evidence": [{"line": 1, "quote": log_line, "supports": "primary_failure"}],
        "justification": "The configuration error is the initiating failure.",
    }
    extractor = _ContractRepairEvidenceExtractor(evidence)
    bundle = build_l0_bundle(str(log_path))

    result = extractor.extract_evidence(
        build_l1_evidence_context(
            bundle,
            build_l0_model_facing_view(bundle),
            LogSnapshot.read(log_path),
        )
    )

    assert result.success is True
    assert extractor.calls == 2
    assert result.anomalies["contract_repair_requested"] is True
    assert len(result.model_calls) == 2
    request_events = [
        event for event in result.transcript_events if event.get("event_type") == "model_request"
    ]
    assert len(request_events) == 2
    repair_messages = request_events[1]["request_body"]["messages"]
    assert repair_messages[-2]["role"] == "assistant"
    assert repair_messages[-1]["role"] == "user"
    assert "could not be accepted" in repair_messages[-1]["content"]
    assert request_events[1]["advertised_tool_schemas"] == []


def test_l1_prohibited_action_field_is_rejected_by_closed_contract(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "INFO training iteration 120",
                "RuntimeError: CUDA error: device-side assert triggered",
            ]
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "decision": "STOP",
        "primary_failure": {
            "fine_class": "cuda_device_assert",
            "policy_class": "user_failure",
            "signature": "device-side assert triggered",
            "root_fingerprint": "cuda_device_assert:device_side_assert",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 2,
            "rank": None,
            "phase": "steady_mid",
        },
        "secondary_failures": [],
        "cascades": [],
        "evidence": [
            {
                "line": 2,
                "quote": "RuntimeError: CUDA error: device-side assert triggered",
                "supports": "primary_failure",
            }
        ],
        "justification": "Bad model output.",
    }

    current = _current_evidence(evidence)
    current["decision"] = "STOP"

    assert model_evidence_contract_errors(current) == ["top-level has unsupported fields: decision"]


def test_l1_provider_timeout_is_not_trusted_even_with_evidence(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "INFO training iteration 120",
                "RuntimeError: CUDA error: device-side assert triggered",
            ]
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "cuda_device_assert",
            "policy_class": "user_failure",
            "signature": "device-side assert triggered",
            "root_fingerprint": "cuda_device_assert:device_side_assert",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 2,
            "rank": None,
            "phase": "steady_mid",
        },
        "secondary_failures": [],
        "cascades": [],
        "evidence": [
            {
                "line": 2,
                "quote": "RuntimeError: CUDA error: device-side assert triggered",
                "supports": "primary_failure",
            }
        ],
        "justification": "Should not be trusted because provider timed out.",
    }

    analyzer = RestartAgent(
        evidence_extractor=_FakeEvidenceExtractor(
            evidence,
            success=False,
            errors=("LLM request timed out",),
            anomalies={"provider_error": True, "provider_timeout": True},
        )
    )
    result = analyzer.analyze({"log_path": str(log_path), "job_id": "job-1"})
    payload = result.to_payload()

    assert payload["decision"] == Decision.RESTART.value
    assert payload["decision_basis"] == DecisionBasis.GENERAL_RETRY_AVAILABLE.value
    assert payload["primary_failure"]["policy_class"] == "ambiguous"
    assert payload["result_provenance"]["model_contribution"] == "attempted_not_used_timeout"
    assert payload["result_provenance"]["candidate_kind"] == "deterministic_fallback"
    assert payload["result_provenance"]["result_quality"] == "degraded"
    assert payload["result_provenance"]["nvrx_use"] == "eligible_degraded"
    assert analyzer.last_trace["l2_audit"]["used"] is False
    assert analyzer.last_trace["l2_audit"]["audit_status"] == "not_run"
    assert analyzer.last_trace["layers"]["L1"]["output_status"] == "provider_timeout"
    assert "LLM request timed out" in analyzer.last_trace["layers"]["L1"]["output_errors"]
    candidates = analyzer.last_trace["decision_candidates"]
    assert candidates["selected"] == "deterministic_fallback"
    assert candidates["deterministic_fallback"]["l1_execution_status"] == "in_flight"
    assert candidates["best_available"]["l1_execution_status"] == "failed"
    assert candidates["best_available"]["result"] == payload


def test_l1_truncated_output_is_not_trusted_even_with_evidence(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "INFO training iteration 120",
                "RuntimeError: CUDA error: device-side assert triggered",
            ]
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "cuda_device_assert",
            "policy_class": "user_failure",
            "signature": "device-side assert triggered",
            "root_fingerprint": "cuda_device_assert:device_side_assert",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 2,
            "rank": None,
            "phase": "steady_mid",
        },
        "secondary_failures": [],
        "cascades": [],
        "evidence": [
            {
                "line": 2,
                "quote": "RuntimeError: CUDA error: device-side assert triggered",
                "supports": "primary_failure",
            }
        ],
        "justification": "Should not be trusted because output was truncated.",
    }

    analyzer = RestartAgent(
        evidence_extractor=_FakeEvidenceExtractor(
            evidence,
            success=True,
            anomalies={"model_output_truncated": True},
        )
    )
    result = analyzer.analyze({"log_path": str(log_path), "job_id": "job-1"})
    payload = result.to_payload()

    assert payload["decision"] == Decision.RESTART.value
    assert payload["decision_basis"] == DecisionBasis.GENERAL_RETRY_AVAILABLE.value
    assert payload["primary_failure"]["policy_class"] == "ambiguous"
    assert payload["result_provenance"]["model_contribution"] == "attempted_not_used_truncated"
    assert payload["result_provenance"]["result_quality"] == "degraded"
    assert payload["result_provenance"]["nvrx_use"] == "eligible_degraded"
    assert analyzer.last_trace["l2_audit"]["used"] is False
    assert analyzer.last_trace["l2_audit"]["audit_status"] == "not_run"
    assert analyzer.last_trace["layers"]["L1"]["output_status"] == "truncated"
    assert "model output was truncated" in analyzer.last_trace["layers"]["L1"]["output_errors"]


def test_l1_token_limit_hit_is_reported_from_finish_reason_length(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "INFO training iteration 120",
                "RuntimeError: CUDA error: device-side assert triggered",
            ]
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "cuda_device_assert",
            "policy_class": "user_failure",
            "signature": "device-side assert triggered",
            "root_fingerprint": "cuda_device_assert:device_side_assert",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 2,
            "rank": None,
            "phase": "steady_mid",
        },
        "secondary_failures": [],
        "cascades": [],
        "evidence": [
            {
                "line": 2,
                "quote": "RuntimeError: CUDA error: device-side assert triggered",
                "supports": "primary_failure",
            }
        ],
        "justification": "Token limit hit should be visible in trace.",
    }

    analyzer = RestartAgent(
        evidence_extractor=_FakeEvidenceExtractor(evidence, finish_reason="length")
    )
    analyzer.analyze({"log_path": str(log_path), "job_id": "job-1"})

    assert analyzer.last_trace["token_limit"]["hit"] is True
    assert analyzer.last_trace["token_limit"]["hit_count"] == 1
    assert analyzer.last_trace["token_limit"]["hit_calls"][0]["model_turn"] is None
    assert analyzer.last_trace["token_limit"]["hit_calls"][0]["completion_tokens"] == 5
    assert analyzer.last_trace["token_limit"]["hit_calls"][0]["reasoning_tokens"] == 2
    assert analyzer.last_trace["anomalies"]["token_limit_hit"] is True


def test_same_root_no_progress_history_waits_for_general_retry_budget(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                (
                    "7:  [2026-05-27 07:58:10.261551] iteration      120/     400 | "
                    "consumed samples:         7680 | lm loss: 1.339357E+01 |"
                ),
                "RuntimeError: CUDA error: device-side assert triggered",
            ]
        ),
        encoding="utf-8",
    )
    current = build_l0_bundle(str(log_path)).deterministic_primary_candidate
    assert current is not None
    history = [
        {
            "job_id": "job-1",
            "cycle_id": 1,
            "root_fingerprint": current.root_fingerprint,
            "primary_fault_outcome": "terminal",
            "last_completed_step": 120,
        },
        {
            "job_id": "job-1",
            "cycle_id": 2,
            "root_fingerprint": current.root_fingerprint,
            "primary_fault_outcome": "terminal",
            "last_completed_step": 120,
        },
    ]

    result = RestartAgent().analyze(
        {
            "log_path": str(log_path),
            "job_id": "job-1",
            "cycle_id": 3,
        },
        prior_attempts=_prior_attempt_view(*history),
    )
    payload = result.to_payload()

    assert payload["decision"] == Decision.RESTART.value
    assert payload["decision_basis"] == DecisionBasis.GENERAL_RETRY_AVAILABLE.value
    assert payload["result_provenance"]["evidence_source"] == "l0_deterministic"
    assert payload["result_provenance"]["history_contribution"] == "checked_no_effect"
    assert payload["result_provenance"]["nvrx_use"] == "eligible"


def test_history_requires_exact_observed_failure_iteration_when_available():
    primary = FailureEvidence(
        fine_class="numeric_instability",
        policy_class="ambiguous",
        signature="Unexpected result inf",
        root_fingerprint="observed:runtimeerror:validate_result",
        fault_outcome="terminal",
        failure_iteration=337071,
    )
    progress = ProgressFacts(
        latest_observed_failure_iteration=337071,
        latest_observed_failure_iteration_line=200,
    )
    base_record = {
        "job_id": "job-1",
        "cycle_id": 1,
        "root_fingerprint": primary.root_fingerprint,
        "primary_fault_outcome": "terminal",
    }

    mismatch = evaluate_history(
        _history_evaluation_input(
            current_attempt=_attempt_facts_and_progress(primary, progress),
            job_id="job-1",
            cycle_id=2,
            prior_records=_attempt_records({**base_record, "failure_iteration": 337072}),
        )
    )
    assert mismatch.matching_root_attempts == 1
    assert mismatch.no_observed_advance_attempts == 1
    assert mismatch.exact_failure_position_attempts == 0
    assert mismatch.comparisons[0].relation == "regressed"
    assert mismatch.comparisons[0].same_failure_iteration is False

    exact = evaluate_history(
        _history_evaluation_input(
            current_attempt=_attempt_facts_and_progress(primary, progress),
            job_id="job-1",
            cycle_id=2,
            prior_records=_attempt_records({**base_record, "failure_iteration": 337071}),
        )
    )
    assert exact.no_observed_advance_attempts == 1
    assert exact.exact_failure_position_attempts == 1
    assert exact.comparisons[0].relation == "same"
    assert exact.comparisons[0].same_failure_iteration is True


def test_history_reports_one_step_observed_advance_without_policy_decision():
    primary = FailureEvidence(
        fine_class="numeric_instability",
        policy_class="ambiguous",
        signature="Unexpected result inf",
        root_fingerprint="observed:runtimeerror:validate_result",
        fault_outcome="terminal",
    )
    summary = evaluate_history(
        _history_evaluation_input(
            current_attempt=_attempt_facts_and_progress(
                primary,
                ProgressFacts(highest_completed_step=121),
            ),
            job_id="job-1",
            cycle_id=2,
            prior_records=_attempt_records(
                {
                    "job_id": "job-1",
                    "cycle_id": 1,
                    "root_fingerprint": primary.root_fingerprint,
                    "primary_fault_outcome": "terminal",
                    "last_completed_step": 120,
                },
            ),
        )
    )

    assert summary.observed_advance_attempts == 1
    assert summary.no_observed_advance_attempts == 0
    assert summary.advanced_beyond_all_comparable_attempts is True
    assert summary.comparisons[0].selected_basis == "completed_step"
    assert summary.comparisons[0].dimension_comparisons[0].delta == 1
    assert summary.comparisons[0].relation == "advanced"


def test_history_marks_different_progress_marker_types_unknown():
    primary = FailureEvidence(
        fine_class="numeric_instability",
        policy_class="ambiguous",
        signature="Unexpected result inf",
        root_fingerprint="observed:runtimeerror:validate_result",
        fault_outcome="terminal",
    )
    summary = evaluate_history(
        _history_evaluation_input(
            current_attempt=_attempt_facts_and_progress(
                primary,
                ProgressFacts(highest_completed_step=121),
            ),
            job_id="job-1",
            cycle_id=2,
            prior_records=_attempt_records(
                {
                    "job_id": "job-1",
                    "cycle_id": 1,
                    "root_fingerprint": primary.root_fingerprint,
                    "primary_fault_outcome": "terminal",
                    "last_checkpoint_step": 120,
                },
            ),
        )
    )

    assert summary.unknown_progress_attempts == 1
    assert summary.no_observed_advance_attempts == 0
    assert summary.comparisons[0].relation == "unknown"


def test_single_weak_no_advance_history_does_not_stop(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "INFO training iteration 120\nRuntimeError: CUDA error: device-side assert triggered\n",
        encoding="utf-8",
    )
    current = build_l0_bundle(str(log_path)).deterministic_primary_candidate
    assert current is not None
    history = [
        {
            "job_id": "job-1",
            "cycle_id": 1,
            "root_fingerprint": current.root_fingerprint,
            "primary_fault_outcome": "terminal",
            "last_completed_step": 120,
        }
    ]
    result = RestartAgent().analyze(
        {
            "log_path": str(log_path),
            "job_id": "job-1",
            "cycle_id": 2,
        },
        prior_attempts=_prior_attempt_view(*history),
    )

    assert result.decision == Decision.RESTART.value
    assert result.result_provenance["history_contribution"] == "checked_no_effect"


def test_l4_history_thresholds_are_not_owned_by_l3():
    primary = FailureEvidence(
        fine_class="numeric_instability",
        policy_class="ambiguous",
        signature="Unexpected result inf",
        root_fingerprint="observed:runtimeerror:validate_result",
        fault_outcome="terminal",
    )

    bounded_once = evaluate_policy(
        L4PolicyInput(
            primary=primary,
            history=HistorySummary(
                available=True,
                matching_root_attempts=1,
                consecutive_same_root_no_advance_attempts=1,
            ),
            model_recovery_assessment=_recovery_assessment(
                domain="unknown",
                outlook="may_recover",
            ),
            assessment_grounded=True,
        )
    ).retry_policy
    assert bounded_once.decision == Decision.STOP.value
    assert bounded_once.rule == "bounded_retry"
    assert bounded_once.matching_prior_failures == 1
    assert bounded_once.allowed_retries == 1
    assert bounded_once.retry_budget_exhausted is True

    general_twice = evaluate_policy(
        L4PolicyInput(
            primary=primary,
            history=HistorySummary(
                available=True,
                matching_root_attempts=2,
                consecutive_same_root_no_advance_attempts=2,
            ),
            model_recovery_assessment=_recovery_assessment(
                domain="unknown",
                outlook="unknown",
            ),
            assessment_grounded=True,
        )
    ).retry_policy
    assert general_twice.decision == Decision.RESTART.value
    assert general_twice.allowed_retries == 3
    assert general_twice.retry_budget_exhausted is False

    general_exhausted = evaluate_policy(
        L4PolicyInput(
            primary=primary,
            history=HistorySummary(
                available=True,
                matching_root_attempts=3,
                consecutive_same_root_no_advance_attempts=3,
            ),
            model_recovery_assessment=_recovery_assessment(
                domain="unknown",
                outlook="unknown",
            ),
            assessment_grounded=True,
        )
    ).retry_policy
    assert general_exhausted.decision == Decision.STOP.value
    assert general_exhausted.retry_budget_exhausted is True


def test_l4_stops_for_grounded_unrecoverable_workload_evidence():
    primary = FailureEvidence(
        fine_class="code_failure",
        policy_class="ambiguous",
        signature="deterministic code failure",
        root_fingerprint="observed:code:failure",
        fault_outcome="terminal",
    )
    assessment = _recovery_assessment(
        domain="workload",
        outlook="cannot_recover",
        domain_status="established_by_current_log",
        outlook_status="established_by_current_log",
    )

    grounded = evaluate_policy(
        L4PolicyInput(
            primary=primary,
            history=HistorySummary(available=False),
            model_recovery_assessment=assessment,
            assessment_grounded=True,
        )
    ).retry_policy
    assert grounded.decision == Decision.STOP.value
    assert grounded.rule == "workload_unrecoverable"
    assert grounded.allowed_retries == 0

    ungrounded = evaluate_policy(
        L4PolicyInput(
            primary=primary,
            history=HistorySummary(available=False),
            model_recovery_assessment=assessment,
            assessment_grounded=False,
        )
    ).retry_policy
    assert ungrounded.decision == Decision.RESTART.value
    assert ungrounded.rule == "general_retry"


@pytest.mark.parametrize(
    ("domain", "outlook", "domain_status", "outlook_status"),
    [
        (
            "infrastructure",
            "cannot_recover",
            "established_by_current_log",
            "established_by_current_log",
        ),
        (
            "workload",
            "may_recover",
            "established_by_current_log",
            "established_by_current_log",
        ),
        (
            "workload",
            "cannot_recover",
            "supported_but_unconfirmed",
            "established_by_current_log",
        ),
        (
            "workload",
            "cannot_recover",
            "established_by_current_log",
            "supported_but_unconfirmed",
        ),
    ],
)
def test_l4_does_not_stop_for_incomplete_current_evidence_predicate(
    domain,
    outlook,
    domain_status,
    outlook_status,
):
    primary = FailureEvidence(
        fine_class="observed_failure",
        policy_class="ambiguous",
        signature="observed failure",
        root_fingerprint="observed:failure",
        fault_outcome="terminal",
    )

    policy = evaluate_policy(
        L4PolicyInput(
            primary=primary,
            history=HistorySummary(available=False),
            model_recovery_assessment=_recovery_assessment(
                domain=domain,
                outlook=outlook,
                domain_status=domain_status,
                outlook_status=outlook_status,
            ),
            assessment_grounded=True,
        )
    ).retry_policy

    assert policy.decision == Decision.RESTART.value
    assert policy.current_evidence_qualified is False


def test_l4_uses_general_retry_when_recovery_assessment_is_not_grounded():
    primary = FailureEvidence(
        fine_class="observed_failure",
        policy_class="ambiguous",
        signature="observed failure",
        root_fingerprint="observed:failure",
        fault_outcome="terminal",
    )

    policy = evaluate_policy(
        L4PolicyInput(
            primary=primary,
            history=HistorySummary(available=False),
            model_recovery_assessment=_recovery_assessment(
                domain="unknown",
                outlook="may_recover",
            ),
            assessment_grounded=False,
        )
    ).retry_policy

    assert policy.rule == "general_retry"
    assert policy.allowed_retries == 3


def test_l4_observed_advance_does_not_exhaust_retry_budget():
    primary = FailureEvidence(
        fine_class="numeric_instability",
        policy_class="ambiguous",
        signature="Unexpected result inf",
        root_fingerprint="observed:runtimeerror:validate_result",
        fault_outcome="terminal",
    )
    policy = evaluate_policy(
        L4PolicyInput(
            primary=primary,
            history=HistorySummary(
                available=True,
                matching_root_attempts=5,
                consecutive_same_root_no_advance_attempts=3,
                advanced_beyond_all_comparable_attempts=True,
            ),
            retry_policy=RetryPolicyConfig(general_retry_allowed_retries=3),
        )
    ).retry_policy
    assert policy.decision == Decision.RESTART.value
    assert policy.decision_basis == DecisionBasis.OBSERVED_ADVANCE.value
    assert policy.retry_budget_exhausted is False


def test_history_distinguishes_iteration_from_same_rank_iteration_replay():
    primary = FailureEvidence(
        fine_class="bad_token_or_window",
        policy_class="ambiguous",
        signature="Unexpected result inf",
        root_fingerprint="observed:data:non_finite",
        fault_outcome="terminal",
        rank="7",
        failure_iteration=418,
    )
    summary = evaluate_history(
        _history_evaluation_input(
            current_attempt=_attempt_facts_and_progress(
                primary,
                ProgressFacts(latest_observed_failure_iteration=418),
            ),
            job_id="job-1",
            cycle_id=3,
            prior_records=_attempt_records(
                {
                    "job_id": "job-1",
                    "cycle_id": 1,
                    "root_fingerprint": primary.root_fingerprint,
                    "primary_fault_outcome": "terminal",
                    "failure_iteration": 418,
                    "faulting_rank": "7",
                },
                {
                    "job_id": "job-1",
                    "cycle_id": 2,
                    "root_fingerprint": primary.root_fingerprint,
                    "primary_fault_outcome": "terminal",
                    "failure_iteration": 418,
                    "faulting_rank": "9",
                },
            ),
        )
    )

    assert summary.exact_failure_position_attempts == 2
    assert summary.same_rank_iteration_attempts == 1
    assert [item.same_rank for item in summary.comparisons] == [True, False]


def test_history_rejects_duplicate_job_cycle_records():
    primary = FailureEvidence(
        fine_class="numeric_instability",
        policy_class="ambiguous",
        signature="Unexpected result inf",
        root_fingerprint="observed:runtimeerror:validate_result",
        fault_outcome="terminal",
    )
    record = {
        "job_id": "job-1",
        "cycle_id": 1,
        "root_fingerprint": primary.root_fingerprint,
        "primary_fault_outcome": "terminal",
        "last_completed_step": 120,
    }
    with pytest.raises(ValueError, match="duplicate job_id/cycle_id"):
        _attempt_records(record, dict(record))


@pytest.mark.parametrize("cycle_id", [None, True, "1"])
def test_history_requires_integer_cycle_id(cycle_id):
    record = {
        "job_id": "job-1",
        "cycle_id": cycle_id,
        "root_fingerprint": "observed:runtimeerror:validate_result",
    }

    expected_error = TypeError if cycle_id is not None else ValueError
    with pytest.raises(expected_error, match=r"attempt records\[0\]\.cycle_id"):
        _attempt_records(record)


def test_history_normalization_orders_records_by_cycle_id():
    records = _attempt_records(
        {
            "job_id": "job-1",
            "cycle_id": 3,
            "root_fingerprint": "observed:runtimeerror:validate_result",
        },
        {
            "job_id": "job-1",
            "cycle_id": 1,
            "root_fingerprint": "observed:runtimeerror:validate_result",
        },
    )

    assert [record.cycle_id for record in records] == [1, 3]


def test_history_revalidates_typed_records_at_runtime_boundary():
    with pytest.raises(TypeError, match=r"attempt records\[0\]\.cycle_id"):
        normalize_attempt_records(
            (
                AttemptRecord(
                    job_id="job-1",
                    cycle_id=True,
                    progress=AttemptProgressSummary(),
                    deterministic=AttemptFailureFacts(
                        source=AttemptFailureFactsSource.L0_DETERMINISTIC,
                        fine_class=None,
                        root_fingerprint="observed:runtimeerror:validate_result",
                        root_fingerprint_source="test_fixture",
                        fault_outcome="terminal",
                    ),
                ),
            )
        )


def test_history_locality_uses_only_qualifying_terminal_records():
    primary = FailureEvidence(
        fine_class="numeric_instability",
        policy_class="ambiguous",
        signature="Unexpected result inf",
        root_fingerprint="observed:runtimeerror:validate_result",
        fault_outcome="terminal",
        node="node-b",
    )
    summary = evaluate_history(
        _history_evaluation_input(
            current_attempt=_attempt_facts_and_progress(
                primary,
                ProgressFacts(highest_completed_step=120),
            ),
            job_id="job-1",
            cycle_id=3,
            prior_records=_attempt_records(
                {
                    "job_id": "job-1",
                    "cycle_id": 1,
                    "root_fingerprint": primary.root_fingerprint,
                    "primary_fault_outcome": "recovered",
                    "last_completed_step": 120,
                    "faulting_node": "node-b",
                },
                {
                    "job_id": "job-1",
                    "cycle_id": 2,
                    "root_fingerprint": primary.root_fingerprint,
                    "primary_fault_outcome": "terminal",
                    "last_completed_step": 120,
                    "faulting_node": "node-a",
                },
            ),
        )
    )

    assert summary.same_node_recurrence is False
    assert summary.cross_node_recurrence is True


def test_megatron_progress_pattern_ignores_nan_iteration_metric(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "INFO number of nan iterations:   0 |",
                "INFO training iteration 120",
                (
                    "31:  [2026-05-26 19:52:15.571763] iteration        5/       8 | "
                    "consumed samples:          160 | lm loss: 1.214552E+01 | "
                    "number of nan iterations:   0 |"
                ),
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))

    assert bundle.progress.highest_completed_step == 5
    assert bundle.progress.progress_lines == (3,)
    assert any(group.classification == "progress" for group in bundle.occurrence_groups)


def test_l0_records_setup_progress_without_promoting_it_to_forward_progress(tmp_path):
    log_path = tmp_path / "setup.log"
    log_path.write_text(
        "\n".join(
            [
                (
                    "0: successfully loaded checkpoint from /checkpoints/680000 "
                    "at iteration 661000"
                ),
                "7: INFO:root:> built cuda graph(s) in 46.77 sec",
                "8: INFO:root:> built cuda graph(s) in 46.78 sec",
                "0: OSError: [Errno 98] Address already in use",
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    prompt_bundle = build_l0_model_facing_view(bundle).evidence_bundle

    assert [marker.marker_type for marker in bundle.progress.setup_markers] == [
        "checkpoint_load",
        "cuda_graph_build",
    ]
    assert bundle.progress.setup_markers[0].value == 661000
    assert bundle.progress.setup_lines == (1, 2)
    assert bundle.progress.progress_markers == ()
    assert bundle.progress.checkpoint_markers == ()
    assert bundle.progress.highest_completed_step is None
    assert bundle.progress.last_checkpoint_step is None
    assert bundle.run_progress_summary.setup_marker_count == 2
    assert bundle.run_progress_summary.last_setup_marker_type == "cuda_graph_build"
    assert bundle.run_progress_summary.last_setup_line == 2
    assert bundle.evidence_coverage["setup_progress"] == "found"
    assert bundle.evidence_coverage["application_progress"] == "not_found"
    assert bundle.evidence_coverage["checkpoint_progress"] == "not_found"
    assert [
        marker["marker_type"] for marker in prompt_bundle["progress"]["recent_setup_markers"]
    ] == ["checkpoint_load", "cuda_graph_build"]


def test_l0_uses_observed_failure_iteration_without_claiming_completed_progress(tmp_path):
    log_path = tmp_path / "iteration_failure.log"
    log_path.write_text(
        "\n".join(
            [
                "0: successfully loaded checkpoint at iteration 337000",
                (
                    "404: [rank404]: RuntimeError: Rank 404, device 0, iteration 337071: "
                    "Unexpected result inf"
                ),
                "910: [rank910]: Traceback (most recent call last):",
                (
                    '910: [rank910]:   File "/usr/lib/python3.12/multiprocessing/util.py", '
                    "line 303, in _run_finalizers"
                ),
                "910: [rank910]:     finalizer()",
                (
                    '910: [rank910]:   File "/usr/lib/python3.12/multiprocessing/'
                    'connection.py", line 399, in _finalize_listener'
                ),
                (
                    "910: [rank910]: FileNotFoundError: [Errno 2] No such file or "
                    "directory: '/tmp/pymp-a/listener-b'"
                ),
                (
                    "910: [rank910]: FileNotFoundError: [Errno 2] No such file or "
                    "directory: '/tmp/pymp-a/listener-b'"
                ),
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    prompt_bundle = build_l0_model_facing_view(bundle).evidence_bundle

    assert bundle.progress.highest_completed_step is None
    assert bundle.progress.latest_observed_failure_iteration == 337071
    assert bundle.progress.latest_observed_failure_iteration_line == 2
    assert bundle.run_progress_summary.checkpoint_load_iteration == 337000
    assert bundle.run_progress_summary.latest_observed_failure_iteration == 337071
    assert bundle.run_progress_summary.observed_iterations_after_checkpoint_load == 71
    assert bundle.deterministic_primary_candidate is not None
    assert bundle.deterministic_primary_candidate.failure_iteration == 337071
    assert bundle.deterministic_primary_candidate.phase == "steady_mid"
    assert bundle.evidence_coverage["application_progress"] == "not_found"
    assert bundle.evidence_coverage["observed_failure_iteration"] == "found"
    assert prompt_bundle["run_progress_summary"]["observed_iterations_after_checkpoint_load"] == 71

    cleanup_matches = [match for match in bundle.registry_matches if match.line in {7, 8}]
    assert len(cleanup_matches) == 2
    assert all(match.policy_class == "cascade" for match in cleanup_matches)
    assert all(match.causal_role == "teardown" for match in cleanup_matches)
    assert all(match.phase == "teardown" for match in cleanup_matches)
    assert all(
        match.root_fingerprint_source == "l0_teardown_structure" for match in cleanup_matches
    )
    assert len({match.root_fingerprint for match in cleanup_matches}) == 1
    assert len(bundle.cascades) == 1
    assert bundle.cascades[0].count == 2


def test_job_metadata_uses_rank_lower_bound_without_explicit_world_size(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "0: INFO rank zero started",
                ("7: [2026-02-20 09:00:00.000000] iteration 10/100 | " "consumed samples: 100 |"),
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    prompt_bundle = build_l0_model_facing_view(bundle).evidence_bundle

    assert bundle.job_metadata.explicit_world_size is None
    assert bundle.job_metadata.observed_rank_min == 0
    assert bundle.job_metadata.observed_rank_max == 7
    assert bundle.job_metadata.observed_rank_count == 2
    assert bundle.job_metadata.inferred_world_size_lower_bound == 8
    assert bundle.job_metadata.world_size_source == "observed_rank_lower_bound"
    assert bundle.job_metadata.world_size_confidence == "observed_lower_bound"
    assert prompt_bundle["job_metadata"]["inferred_world_size_lower_bound"] == 8


def test_late_terminal_exception_gets_context_from_generic_structural_match(tmp_path):
    log_path = tmp_path / "job.log"
    lines = [
        "0: NCCL WARN NET/IB : Got non-fatal async event: port error",
        "1: NCCL WARN NET/IB : Got non-fatal async event: port error",
        "2: NCCL WARN NET/IB : Got non-fatal async event: port error",
        (
            "6143: [2026-02-20 03:44:26] iteration   670310/  1000000 | "
            "consumed samples:     123456 | lm loss: 1.234E+00 | "
            "number of nan iterations:   0 |"
        ),
    ]
    lines.extend(f"INFO ordinary training log line {line}" for line in range(5, 52))
    lines.append(
        "180: [rank180]: RuntimeError: Rank 180, node nvl72005-T14, "
        "device 0, iteration 670314: Unexpected result inf "
        "(message='found Inf in local grad norm for bucket #0 in "
        "backward pass before data-parallel communication collective')"
    )
    log_path.write_text("\n".join(lines), encoding="utf-8")

    bundle = build_l0_bundle(str(log_path))
    prompt_bundle = build_l0_model_facing_view(bundle).evidence_bundle
    late_line = len(lines)

    assert bundle.deterministic_primary_candidate is not None
    assert bundle.deterministic_primary_candidate.fine_class == "observed_exception"
    anchor = next(item for item in bundle.candidate_anchors if item.line == late_line)
    prompt_anchor = next(
        item for item in prompt_bundle["candidate_anchors"] if item["line"] == late_line
    )

    assert "high_signal" in anchor.sources
    assert anchor.taxonomy_match is not None
    assert anchor.taxonomy_match.registry_id == "observed_exception"
    assert anchor.prior_observed_progress_line == 4
    assert anchor.context_window_ids
    assert prompt_anchor["taxonomy_hint"]["registry_id"] == "observed_exception"
    assert prompt_anchor["covered_by_excerpt"] is True
    assert any(late_line in window["seed_lines"] for window in prompt_bundle["context_windows"])


def test_failure_episode_uses_progress_then_terminal_exception(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "INFO world_size ...................................... 6144",
                (
                    "6143:  [2026-02-20 09:05:52.888966] iteration   670250/  794728 | "
                    "consumed samples:   2059008000 | lm loss: 9.709713E-01 | "
                    "grad norm: 0.153 | number of skipped iterations:   0 | "
                    "number of nan iterations:   0 |"
                ),
                (
                    "0:   [2026-02-20 09:06:27.646389] successfully saved checkpoint "
                    "from iteration  670250 to /checkpoints/phase2/"
                ),
                (
                    "6143:  [2026-02-20 09:09:24.491434] iteration   670310/  794728 | "
                    "consumed samples:   2059192320 | lm loss: 9.726213E-01 | "
                    "grad norm: 0.098 | number of skipped iterations:   0 | "
                    "number of nan iterations:   0 |"
                ),
                "180: [rank180]: Traceback (most recent call last):",
                "180: [rank180]:   File \"/train.py\", line 1, in <module>",
                "180: [rank180]:     raise RuntimeError(full_message)",
                (
                    "180: [rank180]: RuntimeError: Rank 180, node nvl72005-T14, "
                    "device 0, iteration 670314: Unexpected result inf "
                    "(message='found Inf in local grad norm')"
                ),
                "0: error: *** STEP 1745223.0 ON nvl72003-T01 CANCELLED ***",
                (
                    "4717: [rank4717]: TCPStore recvValue failed: Failed to recv, "
                    "got 0 bytes. Connection was likely closed."
                ),
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    prompt_bundle = build_l0_model_facing_view(bundle).evidence_bundle
    episode = bundle.failure_episodes[0]
    prompt_episode = prompt_bundle["failure_episodes"][0]

    assert bundle.progress.highest_completed_step == 670310
    assert bundle.progress.last_checkpoint_step == 670250
    assert bundle.progress.progress_markers[-1].value == 670310
    assert bundle.progress.checkpoint_markers[-1].value == 670250
    assert bundle.run_progress_summary.first_iteration == 670250
    assert bundle.run_progress_summary.last_iteration == 670310
    assert bundle.run_progress_summary.iteration_delta == 60
    assert bundle.run_progress_summary.total_iterations == 794728
    assert bundle.run_progress_summary.last_checkpoint_iteration == 670250
    assert bundle.run_progress_summary.progress_after_failure_episode is False
    assert bundle.job_metadata.explicit_world_size == 6144
    assert bundle.job_metadata.explicit_world_size_line == 1
    assert bundle.job_metadata.observed_rank_min == 0
    assert bundle.job_metadata.observed_rank_max == 6143
    assert bundle.job_metadata.observed_rank_count == 4
    assert bundle.job_metadata.inferred_world_size_lower_bound == 6144
    assert bundle.job_metadata.world_size_source == "explicit"
    assert episode.status == "terminal"
    assert episode.start_line == 5
    assert episode.terminal_exception_line == 8
    assert episode.terminal_exception_iteration == 670314
    assert episode.exception_rank == "180"
    assert episode.exception_node == "nvl72005-T14"
    assert episode.exception_gpu == "0"
    assert episode.last_progress_before is not None
    assert episode.last_progress_before.value == 670310
    assert episode.first_progress_after is None
    assert episode.first_teardown_line is None
    assert episode.first_process_termination_line is None
    assert episode.first_scheduler_cancel_line == 9
    assert episode.context_window_ids
    assert prompt_bundle["job_metadata"]["explicit_world_size"] == 6144
    assert prompt_bundle["run_progress_summary"]["last_iteration"] == 670310
    assert prompt_episode["last_progress_before"]["value"] == 670310
    assert prompt_episode["first_progress_after"] is None
    assert prompt_episode["first_scheduler_cancel_line"] == 9
    assert "first_cancellation_line" not in prompt_episode


def test_l0_promotes_nearby_specific_error_ahead_of_distributed_wrapper_fanout(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "0: iteration 42/100 | consumed samples: 1024 |",
                (
                    "7: ERROR:checkpoint.writer:Local process encountered an error: "
                    "unexpected pos 704 vs 598"
                ),
                "8: [rank8]: Traceback (most recent call last):",
                '8: [rank8]:   File "/workspace/checkpoint_writer.py", line 10, in save',
                (
                    "8: [rank8]: torch.distributed.checkpoint.api.CheckpointException: "
                    "CheckpointException ranks:dict_keys([])"
                ),
                "9: [rank9]: Traceback (most recent call last):",
                '9: [rank9]:   File "/workspace/checkpoint_writer.py", line 10, in save',
                (
                    "9: [rank9]: torch.distributed.checkpoint.api.CheckpointException: "
                    "CheckpointException ranks:dict_keys([])"
                ),
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))

    assert bundle.deterministic_primary_candidate is not None
    assert bundle.deterministic_primary_candidate.line == 2
    assert bundle.deterministic_primary_candidate.fine_class == "observed_failure"
    assert bundle.deterministic_primary_candidate.phase == "checkpoint"
    assert bundle.failure_episodes[0].precursor_lines == (2,)
    assert bundle.failure_episodes[0].identity_anchor_line == 2
    assert bundle.failure_episodes[0].identity_anchor_reason == (
        "nearby_high_signal_error_precedes_failure_episode"
    )
    assert _canonical_identity_anchor_line(bundle, 5)[0] == 2
    assert len(bundle.cascades) == 1
    assert bundle.cascades[0].first_line == 5
    assert bundle.cascades[0].count == 2
    assert bundle.cascades[0].sample_lines == (5, 8)

    alternate_primary = FailureEvidence(
        fine_class="observed_exception",
        policy_class="ambiguous",
        signature="different selected episode",
        root_fingerprint="observed:different_episode",
        fault_outcome="terminal",
        causal_role="initiating",
        line=8,
        phase="checkpoint",
    )
    assert build_result_cascades(bundle, alternate_primary, {}) == ()


def test_different_checkpoint_success_is_structured_without_overriding_l1(
    tmp_path,
):
    log_path = tmp_path / "job.log"
    checkpoint_path = "/checkpoints/phase2/"
    lines = [
        "0: world_size ...................................... 6144",
        (
            "0: [2026-02-16 10:20:00.000000] iteration 656120/ 993410 | "
            "consumed samples: 2015000000 |"
        ),
        f"0: saving checkpoint at iteration 656125 to {checkpoint_path} in torch_dist format",
        ("0: successfully saved checkpoint from iteration 656125 to " f"{checkpoint_path}"),
        (
            "0: [2026-02-16 10:38:00.000000] iteration 656250/ 993410 | "
            "consumed samples: 2016000000 |"
        ),
        f"0: saving checkpoint at iteration 656250 to {checkpoint_path} in torch_dist format",
        ("0: successfully saved checkpoint from iteration 656250 to " f"{checkpoint_path}"),
        (
            "0: [2026-02-16 10:46:16.000000] iteration 656370/ 993410 | "
            "consumed samples: 2016368640 |"
        ),
        f"0: saving checkpoint at iteration 656375 to {checkpoint_path} in torch_dist format",
        "261: [rank261]: Traceback (most recent call last):",
        (
            "261: [rank261]:   File \"/megatron/training/training.py\", line 569, "
            "in get_start_time_from_progress_log"
        ),
        (
            "261: [rank261]: AssertionError: Should have seen at least one "
            "'Starting job' entry with same world_size"
        ),
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    bundle = build_l0_bundle(str(log_path))
    assert len(bundle.operation_artifact_comparisons) == 1
    comparison = bundle.operation_artifact_comparisons[0]
    assert comparison.operation == "checkpoint_save"
    assert comparison.artifact_path == checkpoint_path.rstrip("/")
    assert comparison.success_count == 2
    assert comparison.success_logical_artifact_ids == (
        "/checkpoints/phase2#checkpoint_iteration=656125",
        "/checkpoints/phase2#checkpoint_iteration=656250",
    )
    assert comparison.logical_artifact_id == ("/checkpoints/phase2#checkpoint_iteration=656375")
    assert comparison.comparison_level == "same_operation_different_artifact"
    assert comparison.current_start_line == 9
    assert comparison.current_outcome == "started_not_completed"
    assert comparison.failure_line == 12
    assert bundle.evidence_coverage["operation_artifact_comparisons"] == "found"

    prompt_evidence = build_l0_model_facing_view(bundle).decision_evidence
    prompt_comparison = prompt_evidence.operation_artifact_facts[0]
    assert prompt_comparison["comparison_level"] == "same_operation_different_artifact"
    assert prompt_comparison["interpretation"] == (
        "operation_pipeline_was_runnable; current_artifact_health_not_established"
    )

    bundle_path = tmp_path / "l0.json"
    write_l0_bundle(bundle_path, bundle)
    replay = read_l0_bundle(bundle_path, expected_log_path=str(log_path))
    assert replay.operation_artifact_comparisons == bundle.operation_artifact_comparisons

    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "progress_log_assertion",
            "signature": lines[11],
            "proposed_root_fingerprint": None,
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 12,
            "rank": "261",
            "phase": "checkpoint",
        },
        "root_cause_assessment": {
            "summary": "The progress-log assertion failed during checkpoint save.",
            "status": "established_by_current_log",
            "plausible_causes": ["missing matching progress-log entry"],
            "persistence_evidence": ["the assertion is deterministic"],
            "transient_alternatives": [],
            "missing_evidence": [],
        },
        "model_recovery_assessment": {
            "failure_domain": "workload",
            "next_attempt_same_failure_likelihood": "likely",
            "recurrence_confidence": 95,
            "current_attempt_persistence_evidence": "affirmative",
            "retry_recovery_path": "workload_change",
            "confidence": 95,
            "rationale": "The assertion will repeat on restart.",
            "supporting_evidence_lines": [10, 11, 12],
        },
        "related_failures": [],
        "evidence": [{"line": 12, "quote": lines[11], "supports": "primary_failure"}],
        "justification": "The progress-log assertion terminated checkpoint save.",
    }
    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    payload = analyzer.analyze({"log_path": str(log_path)}).to_payload()
    audit = analyzer.last_trace["l2_audit"]

    assert payload["decision"] == Decision.STOP.value
    assert (
        payload["model_recovery_assessment"]["retry_outlook_without_workload_change"]["status"]
        == "established_by_current_log"
    )
    assert payload["retry_policy"]["rule"] == "workload_unrecoverable"
    assert "model_recovery_assessment" not in audit["field_finding_codes"]
    assert audit["grounding_adjustments"] == []
    assert audit["recovery_field_audits"] == []


def test_checkpoint_load_fanout_preserves_shard_uncertainty(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                (
                    "0: [rank0]: loading distributed checkpoint from "
                    "/checkpoints/model/ at iteration 42"
                ),
                (
                    "0: [rank0]: successfully loaded checkpoint from "
                    "/checkpoints/model/ at iteration 42"
                ),
                (
                    "7: [rank7]: RuntimeError: checkpoint metadata "
                    "UnicodeDecodeError: invalid continuation byte"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    comparison = next(
        item
        for item in bundle.operation_artifact_comparisons
        if item.operation == "checkpoint_load"
    )

    assert comparison.logical_artifact_id == ("/checkpoints/model#checkpoint_iteration=42")
    assert comparison.physical_unit_id is None
    assert comparison.comparison_level == ("same_logical_artifact_other_or_unknown_unit")
    assert comparison.observation_kind == "distributed_fanout"
    assert comparison.current_outcome == "mixed_success_and_failure"
    assert comparison.successful_observer_ranks == ("0",)
    assert comparison.failed_observer_ranks == ("7",)
    assert comparison.interpretation == (
        "same_logical_artifact_success_observed_on_another_or_unknown_shard"
    )


def test_dataloader_comparison_retains_file_region_and_observer_identity(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                (
                    "0: [rank0]: DataLoader successfully read file "
                    "/data/train/shard-0001.bin offset=100"
                ),
                (
                    "1: [rank1]: DataLoader failed to read file "
                    "/data/train/shard-0001.bin offset=200 "
                    "checksum mismatch expected=abc actual=def"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    comparison = next(
        item
        for item in bundle.operation_artifact_comparisons
        if item.operation == "dataloader_read"
    )

    assert comparison.physical_unit_id == "/data/train/shard-0001.bin"
    assert comparison.data_region == "offset=200"
    assert comparison.success_data_regions == ("offset=100",)
    assert comparison.integrity_marker == "checksum mismatch expected=abc actual=def"
    assert comparison.comparison_level == "exact_physical_unit"
    assert comparison.observation_kind == "distributed_fanout"
    assert comparison.successful_observer_ranks == ("0",)
    assert comparison.failed_observer_ranks == ("1",)


def test_dataloader_success_on_different_file_is_pipeline_evidence_only(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "0: [rank0]: DataLoader successfully read file /data/train/shard-0001.bin",
                "1: [rank1]: DataLoader failed to read file /data/train/shard-0002.bin",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    comparison = next(
        item
        for item in bundle.operation_artifact_comparisons
        if item.physical_unit_id == "/data/train/shard-0002.bin"
    )

    assert comparison.comparison_level == "same_operation_different_artifact"
    assert comparison.current_outcome == "started_not_completed"
    assert comparison.observation_kind == "current_log_comparison"
    assert comparison.interpretation == (
        "operation_pipeline_was_runnable; current_artifact_health_not_established"
    )


def test_failure_episode_separates_teardown_process_and_scheduler_lines(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                (
                    "0: [2026-02-20 09:00:00.000000] iteration   10/  100 | "
                    "consumed samples:  100 | lm loss: 1.0 |"
                ),
                "180: [rank180]: Traceback (most recent call last):",
                "180: [rank180]: RuntimeError: found Inf in local grad norm",
                (
                    "180: [rank180]: Warning: destroy_process_group() was not called "
                    "before program exit"
                ),
                (
                    "180: [rank180]: Producer process has been terminated before all "
                    "shared CUDA tensors released"
                ),
                "0: slurmstepd: error: *** STEP 1745223.0 CANCELLED AT 2026-02-20T09:10:13 ***",
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    prompt_bundle = build_l0_model_facing_view(bundle).evidence_bundle
    episode = bundle.failure_episodes[0]
    prompt_episode = prompt_bundle["failure_episodes"][0]

    assert episode.status == "terminal"
    assert episode.first_teardown_line == 4
    assert episode.first_process_termination_line == 5
    assert episode.first_scheduler_cancel_line == 6
    assert prompt_episode["first_teardown_line"] == 4
    assert prompt_episode["first_process_termination_line"] == 5
    assert prompt_episode["first_scheduler_cancel_line"] == 6
    assert "first_cancellation_line" not in prompt_episode


def test_l0_links_late_explicit_oom_confirmation_to_abrupt_kill_episode(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                (
                    "0: [2026-02-16 11:35:17.842705] iteration 656001/993410 | "
                    "consumed samples: 100 | lm loss: 1.0 |"
                ),
                "48: Killed",
                "72: TCPStore recvValue failed: Connection reset by peer",
                "72: Failed to check should dump on TCPStore: Broken pipe",
                (
                    "48: Feb 16 11:46:12.517314 slurmstepd: error: Detected 6 "
                    "oom_kill events in StepId=1742530.0. Some of the step tasks "
                    "have been OOM Killed."
                ),
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    prompt_bundle = build_l0_model_facing_view(bundle).evidence_bundle
    episode = bundle.failure_episodes[0]

    assert bundle.deterministic_primary_candidate is None
    assert len(bundle.cause_confirmations) == 1
    assert bundle.cause_confirmations[0].registry_id == "linux_oom_kill_confirmation"
    assert bundle.cause_confirmations[0].line == 5
    assert episode.terminal_exception_line == 2
    assert episode.terminal_exception_quote == "48: Killed"
    assert episode.first_process_termination_line == 2
    assert episode.identity_anchor_line == 5
    assert episode.identity_anchor_reason == "explicit_cause_confirmation"
    assert [item.line for item in episode.cause_confirmations] == [5]
    assert _canonical_identity_anchor_line(bundle, 2) == (
        5,
        "failure_episode_identity_anchor:explicit_cause_confirmation",
    )
    assert _canonical_identity_anchor_line(bundle, 5) == (
        5,
        "model_primary_is_episode_identity_anchor:explicit_cause_confirmation",
    )
    assert any(
        5 in window.seed_lines and window.selected_by == "cause_confirmation"
        for window in bundle.context_windows
    )
    assert any(
        anchor.line == 5 and "cause_confirmation" in anchor.sources
        for anchor in bundle.candidate_anchors
    )
    assert prompt_bundle["failure_episodes"][0]["cause_confirmations"][0]["line"] == 5
    assert prompt_bundle["cause_confirmations"][0]["line"] == 5

    bundle_path = tmp_path / "bundle.json"
    write_l0_bundle(bundle_path, bundle)
    replayed = read_l0_bundle(bundle_path, expected_log_path=str(log_path))
    assert replayed.failure_episodes[0].cause_confirmations[0].line == 5


def test_bare_killed_does_not_infer_oom_without_confirmation(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                (
                    "0: [2026-02-16 11:35:17.842705] iteration 656001/993410 | "
                    "consumed samples: 100 | lm loss: 1.0 |"
                ),
                "48: Killed",
                "72: TCPStore recvValue failed: Connection reset by peer",
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))

    assert bundle.cause_confirmations == ()
    assert bundle.failure_episodes[0].terminal_exception_quote == "48: Killed"
    assert bundle.failure_episodes[0].cause_confirmations == ()
    assert "oom" not in bundle.failure_episodes[0].terminal_exception_quote.lower()


def test_checkpoint_traceback_anchors_terminal_exception_and_marks_cleanup(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "0: loading distributed checkpoint at iteration 622125",
                "4175: [rank4175]: Traceback (most recent call last):",
                "4175: [rank4175]:   File \"checkpointing.py\", line 1710, in load_checkpoint",
                "4175: [rank4175]:     torch.distributed.all_gather_object(metadata)",
                "4175: [rank4175]:     return _unpickler(buf).load()",
                (
                    "4175: [rank4175]: UnicodeDecodeError: 'utf-8' codec can't "
                    "decode byte 0xde in position 8: invalid continuation byte"
                ),
                "0: slurmstepd: error: *** STEP 1.0 CANCELLED AT 2026-02-13T23:23:41 ***",
                "5124: Traceback (most recent call last):",
                "5124:   File \"multiprocessing/util.py\", line 303, in _run_finalizers",
                "5124:   File \"multiprocessing/synchronize.py\", line 87, in _cleanup",
                "5124:     sem_unlink(name)",
                (
                    "5124: FileNotFoundError: [Errno 2] No such file or directory: "
                    "'/tmp/pymp-a/listener-a'"
                ),
                "5125: Traceback (most recent call last):",
                "5125:   File \"multiprocessing/util.py\", line 303, in _run_finalizers",
                "5125:   File \"multiprocessing/synchronize.py\", line 87, in _cleanup",
                "5125:     sem_unlink(name)",
                (
                    "5125: FileNotFoundError: [Errno 2] No such file or directory: "
                    "'/tmp/pymp-b/listener-b'"
                ),
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    prompt_bundle = build_l0_model_facing_view(bundle).evidence_bundle

    assert bundle.failure_episodes[0].terminal_exception_line == 6
    assert bundle.failure_episodes[0].terminal_exception_causal_role_hint == "unknown"
    assert len(bundle.failure_episodes) == 1
    assert bundle.failure_episodes[0].wrapper_exception_lines == (12, 17)

    terminal_anchor = next(anchor for anchor in bundle.candidate_anchors if anchor.line == 6)
    cleanup_anchor = next(anchor for anchor in bundle.candidate_anchors if anchor.line == 12)
    assert "terminal_exception" in terminal_anchor.sources
    assert cleanup_anchor.causal_role_hint == "teardown"

    assert "deterministic_primary_candidate" not in prompt_bundle
    assert "registry_matches" not in prompt_bundle
    assert "pattern_groups" not in prompt_bundle
    assert prompt_bundle["occurrence_groups"]
    occurrence_group = prompt_bundle["occurrence_groups"][0]
    assert occurrence_group["occurrence_group_id"].startswith("og-")
    assert occurrence_group["normalized_shape"]
    assert "pattern_id" not in occurrence_group
    assert "normalized_pattern" not in occurrence_group
    assert prompt_bundle["registry_candidate"]["provisional"] is True
    assert prompt_bundle["registry_candidate"]["line"] == 6
    assert prompt_bundle["registry_candidate"]["causal_role_hint"] == "unknown"
    file_not_found_groups = [
        group
        for group in prompt_bundle["registry_candidate_groups"]
        if group["signature_hint"] == "FileNotFoundError:"
    ]
    assert len(file_not_found_groups) == 1
    assert file_not_found_groups[0]["count"] == 2
    assert file_not_found_groups[0]["causal_role_hint"] == "teardown"
    assert "policy_class" not in file_not_found_groups[0]
    cleanup_occurrence_groups = [
        group
        for group in bundle.occurrence_groups
        if group.classification == "cascade"
        and "filenotfounderror" in group.normalized_shape.casefold()
    ]
    assert len(cleanup_occurrence_groups) == 1
    assert cleanup_occurrence_groups[0].count == 2


def test_l0_consolidates_rank_fanout_and_finalizer_cleanup_into_one_episode(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "0: loading checkpoint at iteration 337000",
                "404: [rank404]: Traceback (most recent call last):",
                "404: [rank404]:   File \"rerun_state_machine.py\", line 88, in validate_result",
                "404: [rank404]: RuntimeError: raising error at iteration 337071: Inf grad norm",
                "405: [rank405]: Traceback (most recent call last):",
                "405: [rank405]:   File \"rerun_state_machine.py\", line 88, in validate_result",
                "405: [rank405]: RuntimeError: raising error at iteration 337071: Inf grad norm",
                "99: Traceback (most recent call last):",
                "99:   File \"multiprocessing/util.py\", line 303, in _run_finalizers",
                "99:   File \"multiprocessing/synchronize.py\", line 87, in _cleanup",
                "99:     sem_unlink(name)",
                (
                    "99: FileNotFoundError: [Errno 2] No such file or directory: "
                    "'/tmp/pymp-random/listener-random'"
                ),
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))

    assert len(bundle.distributed_failure_incidents) == 1
    assert bundle.distributed_failure_incidents[0].member_event_lines == (4, 7)
    assert len(bundle.failure_episodes) == 1
    episode = bundle.failure_episodes[0]
    assert episode.terminal_exception_line == 4
    assert episode.duplicate_rendering_lines == (7,)
    assert episode.wrapper_exception_lines == (12,)


def test_l2_does_not_treat_current_attempt_fanout_as_cross_attempt_persistence(tmp_path):
    log_path = tmp_path / "job.log"
    lines = [
        "0: successfully loaded checkpoint at iteration 337000",
        "404: [rank404]: Traceback (most recent call last):",
        "404: [rank404]: RuntimeError: raising error at iteration 337071: Inf grad norm",
        "405: [rank405]: Traceback (most recent call last):",
        "405: [rank405]: RuntimeError: raising error at iteration 337071: Inf grad norm",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "numeric_instability",
            "signature": lines[2],
            "proposed_root_fingerprint": None,
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 3,
            "rank": "404",
            "phase": "steady_mid",
        },
        "root_cause_assessment": {
            "summary": "The observed Inf mechanism is established; its cause is unconfirmed.",
            "status": "supported_but_unconfirmed",
            "plausible_causes": ["data or model numerical instability"],
            "missing_evidence": ["whether iteration 337071 failed in a prior attempt"],
        },
        "model_recovery_assessment": {
            "failure_domain": {
                "value": "workload",
                "status": "supported_but_unconfirmed",
                "confidence": 88,
            },
            "retry_outlook_without_workload_change": {
                "value": "cannot_recover",
                "status": "supported_but_unconfirmed",
                "confidence": 70,
            },
            "rationale": "The unchanged attempt will likely reach the same failure quickly.",
        },
        "related_failures": [],
        "evidence": [
            {"line": 3, "quote": lines[2], "supports": "primary_failure"},
            {"line": 5, "quote": lines[4], "supports": "same_attempt_fanout"},
            {"line": 1, "quote": lines[0], "supports": "execution_position"},
        ],
        "justification": "The mechanism is observed, while recurrence remains unproven.",
    }

    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    payload = analyzer.analyze({"log_path": str(log_path)}).to_payload()
    audit = analyzer.last_trace["l2_audit"]

    assert payload["decision"] == Decision.RESTART.value
    assert (
        payload["model_recovery_assessment"]["retry_outlook_without_workload_change"]["status"]
        == "supported_but_unconfirmed"
    )
    assert "model_recovery_assessment" not in audit["field_finding_codes"]
    assert analyzer.last_trace["layers"]["L2"]["material_finding_count"] == 0
    assert audit["grounding_adjustments"] == []
    assert audit["recovery_field_audits"] == []


def test_l0_compacts_rank_fanout_and_preserves_timeout_episode_context(tmp_path):
    log_path = tmp_path / "job.log"
    timeout_lines = [
        (
            f"{rank}: [rank{rank}]: Watchdog caught collective operation timeout: "
            "WorkNCCL(SeqNum=2, OpType=ALLGATHER, NumelIn=10, NumelOut=20, "
            "Timeout(ms)=600000) ran for 600000 milliseconds before timing out."
        )
        for rank in range(20)
    ]
    log_path.write_text(
        "\n".join(
            [
                "0: INFO:trainer:Setting up optimizer with config OptimizerConfig()",
                "0: sharded_state_dict metadata loaded from the checkpoint: {}",
                "0: Job sharding has changed: Rerun state will be ignored",
                "0: loading distributed checkpoint from /checkpoints/a/ at iteration 635000",
                *timeout_lines,
                "0: slurmstepd: error: *** STEP 1.0 CANCELLED AT 2026-02-13T23:23:41 ***",
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))

    assert bundle.selection_summary["candidate_lines_after_filters"] == 20
    assert bundle.selection_summary["retained_registry_matches"] == 7
    assert bundle.selection_summary["dropped_duplicate_registry_matches"] == 13
    assert len(bundle.registry_matches) == 7
    timeout_group = next(
        group
        for group in bundle.occurrence_groups
        if group.registry_id == "observed_distributed_operation_timeout"
    )
    assert timeout_group.count == 20
    assert len(bundle.failure_episodes) == 1
    assert bundle.failure_episodes[0].terminal_exception_line == 5
    assert len(bundle.distributed_failure_incidents) == 1
    incident = bundle.distributed_failure_incidents[0]
    assert incident.incident_kind == "distributed_mechanism"
    assert incident.event_count == 20
    assert incident.unique_operation_count == 1
    assert incident.operation_types == ("allgather",)
    assert incident.observed_rank_count == 20
    assert incident.history_fingerprint == (
        "distributed_incident:collective_operation_timeout:checkpoint_load_start"
    )
    assert bundle.deterministic_primary_candidate is not None
    assert bundle.deterministic_primary_candidate.root_fingerprint == incident.history_fingerprint
    assert (
        bundle.deterministic_primary_candidate.root_fingerprint_source == "l0_distributed_incident"
    )
    assert bundle.run_progress_summary.checkpoint_load_iteration is None
    assert bundle.run_progress_summary.last_setup_marker_type == "checkpoint_load_start"


def test_single_rank_ordinary_failure_is_episode_only(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "7: iteration 418 completed",
                "7: [rank7]: Traceback (most recent call last):",
                "7: [rank7]: RuntimeError: failed to decode input record",
                "7: destroy_process_group() called during shutdown",
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))

    assert len(bundle.failure_episodes) == 1
    assert bundle.failure_episodes[0].exception_rank == "7"
    assert bundle.distributed_failure_incidents == ()


def test_single_observer_collective_timeout_is_mechanism_incident_and_episode(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                "7: iteration 418 completed",
                (
                    "7: [rank7]: Watchdog caught collective operation timeout: "
                    "WorkNCCL(SeqNum=2, OpType=ALLGATHER, Timeout(ms)=600000) "
                    "ran for 600000 milliseconds before timing out."
                ),
                "7: destroy_process_group() called during shutdown",
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))

    assert len(bundle.failure_episodes) == 1
    assert len(bundle.distributed_failure_incidents) == 1
    incident = bundle.distributed_failure_incidents[0]
    assert incident.incident_kind == "distributed_mechanism"
    assert incident.event_count == 1
    assert incident.observed_rank_count == 1
    prompt_incident = build_l0_model_facing_view(bundle).evidence_bundle[
        "distributed_failure_incidents"
    ][0]
    assert prompt_incident["incident_kind"] == "distributed_mechanism"

    bundle_path = tmp_path / "bundle.json"
    write_l0_bundle(bundle_path, bundle)
    replayed = read_l0_bundle(bundle_path, expected_log_path=str(log_path))
    assert replayed.distributed_failure_incidents[0].incident_kind == ("distributed_mechanism")


def test_distributed_timeout_incident_groups_operations_and_has_order_invariant_history_key(
    tmp_path,
):
    progress = (
        "0: [2026-02-24 18:52:02.194471] iteration 743490/900000 | " "consumed samples: 1000 |"
    )

    def timeout(rank, timestamp, sequence, operation):
        return (
            f"{rank}: [rank{rank}]:[E224 {timestamp} ProcessGroupNCCL.cpp:697] "
            "Watchdog caught collective operation timeout: "
            f"WorkNCCL(SeqNum={sequence}, OpType={operation}, NumelIn=10, "
            "NumelOut=20, Timeout(ms)=600000) ran for 600001 milliseconds "
            "before timing out."
        )

    first_path = tmp_path / "first.log"
    first_lines = [
        progress,
        timeout(42, "19:02:08.080684864", 100, "COALESCED"),
        (
            "42: [rank42]:[E224 19:02:08.100000000 ProcessGroupNCCL.cpp:2403] "
            "[PG ID 5 PG GUID 7563(TENSOR_MODEL_PARALLEL_GROUP) Rank 0] "
            "failure detected by watchdog at work sequence id: 100"
        ),
        timeout(7, "19:02:08.900000000", 200, "_ALLGATHER_BASE"),
    ]
    first_path.write_text("\n".join(first_lines), encoding="utf-8")

    second_path = tmp_path / "second.log"
    second_lines = [
        progress,
        timeout(999, "19:02:08.200000000", 901, "_ALLGATHER_BASE"),
        timeout(3, "19:02:08.700000000", 902, "COALESCED"),
    ]
    second_path.write_text("\n".join(second_lines), encoding="utf-8")

    first_bundle = build_l0_bundle(str(first_path))
    second_bundle = build_l0_bundle(str(second_path))

    first_incident = first_bundle.distributed_failure_incidents[0]
    second_incident = second_bundle.distributed_failure_incidents[0]
    assert len(first_bundle.failure_episodes) == 1
    assert first_bundle.failure_episodes[0].exception_chain_lines == (2, 4)
    assert first_incident.event_count == 2
    assert first_incident.unique_operation_count == 2
    assert first_incident.operation_types == ("allgather_base", "coalesced")
    assert first_incident.process_group_types == ("tensor_model_parallel_group",)
    assert first_incident.last_progress_line == 1
    assert first_incident.last_progress_timestamp == "2026-02-24 18:52:02.194471"
    assert first_incident.first_detection_timestamp == "19:02:08.080684864"
    assert first_incident.seconds_since_last_progress == pytest.approx(605.886, abs=0.001)
    assert first_incident.detection_lag_seconds == pytest.approx(5.886, abs=0.001)
    assert first_incident.root_cause_status == "unknown"
    assert first_incident.history_fingerprint == second_incident.history_fingerprint
    assert first_incident.history_fingerprint == (
        "distributed_incident:collective_operation_timeout:steady_mid"
    )
    assert first_bundle.deterministic_primary_candidate is not None
    assert second_bundle.deterministic_primary_candidate is not None
    assert all(
        cascade.fine_class != "observed_distributed_operation_timeout"
        for cascade in first_bundle.cascades
    )
    assert (
        first_bundle.deterministic_primary_candidate.root_fingerprint
        == second_bundle.deterministic_primary_candidate.root_fingerprint
    )
    assert first_bundle.run_progress_summary.first_terminal_incident_line == 2
    assert first_bundle.run_progress_summary.terminal_detection_lag_seconds == pytest.approx(
        5.886,
        abs=0.001,
    )

    evidence = {
        "schema_version": "restart_agent_evidence.v1",
        "primary_failure": {
            "fine_class": "collective_operation_timeout",
            "signature": "collective operation timeout",
            "proposed_root_fingerprint": "model:operation_specific",
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 4,
            "rank": "7",
            "phase": "steady_mid",
        },
        "root_cause_assessment": {
            "summary": "The collective watchdog timed out; the initiating cause is unknown.",
            "plausible_causes": ["communication progress stopped"],
            "persistence_evidence": [],
            "transient_alternatives": ["transient communication disruption"],
        },
        "model_recovery_assessment": {
            "failure_domain": "unknown",
            "next_attempt_same_failure_likelihood": "unknown",
            "current_attempt_persistence_evidence": "none",
            "retry_recovery_path": "unknown",
            "confidence": 70,
            "rationale": "One attempt does not establish persistence.",
            "supporting_evidence_lines": [4],
        },
        "related_failures": [],
        "evidence": [{"line": 4, "quote": first_lines[3], "supports": "primary_failure"}],
        "justification": "The later operation belongs to the same timeout wave.",
    }
    analyzer = RestartAgent(evidence_extractor=_FakeEvidenceExtractor(evidence))
    payload = analyzer.analyze(
        {"log_path": str(first_path)},
        l0_bundle=first_bundle,
    ).to_payload()
    assert payload["primary_failure"]["root_fingerprint"] == first_incident.history_fingerprint
    assert payload["primary_failure"]["root_fingerprint_source"] == ("l0_distributed_incident")
    assert analyzer.last_trace["l2_audit"]["distributed_incident_id"] == "di-1"


def test_failure_episode_records_progress_after_exception(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                (
                    "0: [2026-02-20 09:00:00.000000] iteration   10/  100 | "
                    "consumed samples:  100 | lm loss: 1.0 |"
                ),
                "0: RuntimeError: retryable data loader hiccup",
                (
                    "0: [2026-02-20 09:01:00.000000] iteration   20/  100 | "
                    "consumed samples:  200 | lm loss: 0.9 |"
                ),
            ]
        ),
        encoding="utf-8",
    )

    bundle = build_l0_bundle(str(log_path))
    episode = bundle.failure_episodes[0]

    assert episode.status == "progressed_after"
    assert episode.last_progress_before is not None
    assert episode.last_progress_before.value == 10
    assert episode.first_progress_after is not None
    assert episode.first_progress_after.value == 20


def test_cli_trace_json_writes_l0_bundle_and_optional_summary(tmp_path, capsys):
    log_path = tmp_path / "service_input.sanitized.log"
    trace_path = tmp_path / "trace.json"
    log_path.write_text(
        "\n".join(
            [
                (
                    "7:  [2026-05-27 07:58:10.261551] iteration        1/       4 | "
                    "consumed samples:           64 | lm loss: 1.339357E+01 |"
                ),
                "CRITICAL:training.runtime:raising GPU error on cuda:0",
                "RuntimeError: CUDA error: device-side assert triggered",
            ]
        ),
        encoding="utf-8",
    )

    rc = cli_main(
        [
            str(log_path),
            "--job-id",
            "job-1",
            "--trace-json",
            str(trace_path),
            "--summary",
        ]
    )

    captured = capsys.readouterr()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))

    assert rc == 0
    assert "restart_agent summary:" in captured.err
    assert trace["analyzer_trace"]["context_mode"] == "fallback_precurated"
    assert trace["analysis_result"]["result_provenance"]["evidence_source"] == "l0_deterministic"
    assert trace["analyzer_trace"]["result_provenance"]["evidence_source"] == "l0_deterministic"
    assert trace["analysis_result"]["primary_failure"]["fine_class"] == "observed_exception"
    assert trace["l0_bundle"]["progress"]["highest_completed_step"] == 1
    assert (
        trace["l0_bundle"]["deterministic_primary_candidate"]["fine_class"] == "observed_exception"
    )


def test_analyze_many_runs_routes_in_parallel_over_shared_l0(tmp_path):
    log_path = tmp_path / "job.log"
    failure_line = "RuntimeError: CUDA error: device-side assert triggered"
    log_path.write_text(f"{failure_line}\n", encoding="utf-8")
    evidence = {
        "primary_failure": {
            "fine_class": "runtime_assertion",
            "signature": failure_line,
            "fault_outcome": "terminal",
            "causal_role": "initiating",
            "data_position_fingerprint": None,
            "line": 1,
            "rank": None,
            "phase": "setup",
        },
        "root_cause_assessment": {
            "summary": "A runtime assertion terminated setup.",
            "plausible_causes": ["workload code or data"],
            "persistence_evidence": [],
            "transient_alternatives": ["retry may recover"],
        },
        "model_recovery_assessment": {
            "failure_domain": "workload",
            "next_attempt_same_failure_likelihood": "plausible",
            "current_attempt_persistence_evidence": "none",
            "retry_recovery_path": "ordinary_retry",
            "confidence": 70,
            "rationale": "One attempt does not prove persistence.",
            "supporting_evidence_lines": [1],
        },
        "evidence": [{"line": 1, "quote": failure_line, "supports": "primary_failure"}],
        "justification": "The assertion is the initiating observed failure.",
    }
    barrier = threading.Barrier(2)
    observed_model_views: list[int] = []

    class _BarrierExtractor(_FakeEvidenceExtractor):
        def extract_evidence(self, context, *, deadline_monotonic=None):
            observed_model_views.append(id(context.model_view))
            barrier.wait(timeout=2)
            return super().extract_evidence(
                context,
                deadline_monotonic=deadline_monotonic,
            )

    result = RestartAgent().analyze_many(
        RestartAgentRequest(log_path=str(log_path)),
        (
            ModelRoute(
                route_id="first",
                evidence_extractor=_BarrierExtractor(evidence),
                model="model-a",
                endpoint="https://endpoint-a/v1",
                credential_ref="KEY_A",
            ),
            ModelRoute(
                route_id="second",
                evidence_extractor=_BarrierExtractor(evidence),
                model="model-b",
                endpoint="https://endpoint-b/v1",
                credential_ref="KEY_B",
            ),
        ),
        max_parallel_models=2,
    )

    assert [item.route_id for item in result.model_results] == ["first", "second"]
    assert all(item.execution_status == "completed" for item in result.model_results)
    assert len(set(observed_model_views)) == 1
    assert result.shared_analysis["route_count"] == 2
    assert result.shared_analysis["max_parallel_models"] == 2
    assert result.shared_analysis["l0_bundle_hash"].startswith("sha256:")
    assert result.shared_analysis["l0_model_view_hash"].startswith("sha256:")


def test_analyze_many_publishes_fallback_before_starting_model_routes(tmp_path):
    log_path = tmp_path / "job.log"
    failure_line = "RuntimeError: failure"
    log_path.write_text(failure_line + "\n", encoding="utf-8")
    fallback_published = threading.Event()
    observed_candidates = []

    class _PublicationAwareExtractor(_FakeEvidenceExtractor):
        def extract_evidence(self, context, *, deadline_monotonic=None):
            assert fallback_published.is_set()
            return super().extract_evidence(
                context,
                deadline_monotonic=deadline_monotonic,
            )

    def on_fallback_ready(candidate):
        observed_candidates.append(candidate)
        fallback_published.set()

    result = RestartAgent().analyze_many(
        RestartAgentRequest(log_path=str(log_path)),
        (
            ModelRoute(
                route_id="first",
                evidence_extractor=_PublicationAwareExtractor(
                    {
                        "primary_failure": {
                            "fine_class": "runtime_error",
                            "signature": failure_line,
                            "fault_outcome": "terminal",
                            "causal_role": "initiating",
                            "line": 1,
                            "rank": None,
                            "phase": "setup",
                        },
                        "evidence": [
                            {"line": 1, "quote": failure_line, "supports": "primary_failure"}
                        ],
                        "justification": "The runtime error ended setup.",
                    }
                ),
            ),
        ),
        on_fallback_ready=on_fallback_ready,
    )

    assert fallback_published.is_set()
    assert len(observed_candidates) == 1
    assert observed_candidates[0].l1_execution_status == "in_flight"
    assert observed_candidates[0].result == result.deterministic_result
    assert result.deterministic_result.result_provenance["model_contribution"] == (
        "pending_not_used"
    )
    assert result.deterministic_result.result_provenance["l1_execution_status"] == "in_flight"
    assert result.shared_analysis["fallback_ready_wall_clock_s"] is not None


def test_analyze_many_publishes_canonical_l0_artifacts_while_route_is_running(tmp_path):
    log_path = tmp_path / "job.log"
    failure_line = "RuntimeError: failure"
    log_path.write_text(failure_line + "\n", encoding="utf-8")
    bundle_path = tmp_path / "l0_bundle.json"
    decision_evidence_path = tmp_path / "decision_evidence.json"
    model_view_path = tmp_path / "l0_model_view.json"
    published = threading.Event()
    slow_extractor = _BlockingEvidenceExtractor({"justification": "slow route"})
    publisher = L0ArtifactPublisher(
        bundle_path=bundle_path,
        decision_evidence_path=decision_evidence_path,
        model_view_path=model_view_path,
        on_published=lambda artifacts, paths: published.set(),
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            RestartAgent().analyze_many,
            RestartAgentRequest(log_path=str(log_path)),
            (ModelRoute(route_id="slow", evidence_extractor=slow_extractor),),
            on_l0_ready=publisher.publish,
            timeout_seconds=2,
        )
        assert slow_extractor.started.wait(timeout=1)
        assert published.wait(timeout=1)
        assert not slow_extractor.completed.is_set()
        assert read_l0_bundle(bundle_path, expected_log_path=str(log_path)).line_count == 1
        decision_payload = json.loads(decision_evidence_path.read_text(encoding="utf-8"))
        model_view_payload = json.loads(model_view_path.read_text(encoding="utf-8"))
        assert decision_payload["schema_version"].startswith("restart_agent_decision_evidence")
        assert model_view_payload["schema_version"].startswith("restart_agent_l0_model_view")
        slow_extractor.release.set()
        result = future.result(timeout=2)

    assert result.model_results[0].execution_status == "completed"
    assert publisher.wait()["l0_bundle"] == str(bundle_path)
    publisher.close()


def test_l0_artifact_publisher_uses_injected_executor_factory():
    calls = []

    def executor_factory(*, max_workers, thread_name_prefix):
        calls.append((max_workers, thread_name_prefix))
        return ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )

    publisher = L0ArtifactPublisher(executor_factory=executor_factory)
    publisher.close()

    assert calls == [(1, "restart-agent-l0-artifacts")]


def test_analyze_many_publishes_each_route_without_waiting_for_slowest(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: failure\n", encoding="utf-8")
    slow_extractor = _BlockingEvidenceExtractor({"justification": "slow route"})
    observed_routes = []

    def on_route_complete(route_result, route_trace):
        observed_routes.append(route_result.route_id)
        assert route_trace
        if route_result.route_id == "fast":
            assert not slow_extractor.completed.is_set()
            slow_extractor.release.set()

    result = RestartAgent().analyze_many(
        RestartAgentRequest(log_path=str(log_path)),
        (
            ModelRoute(
                route_id="fast",
                evidence_extractor=_FakeEvidenceExtractor({"justification": "fast route"}),
            ),
            ModelRoute(
                route_id="slow",
                evidence_extractor=slow_extractor,
            ),
        ),
        max_parallel_models=2,
        on_route_complete=on_route_complete,
        timeout_seconds=2,
    )

    assert observed_routes == ["fast", "slow"]
    assert [route.route_id for route in result.model_results] == ["fast", "slow"]
    assert result.shared_analysis["route_completion_callback_errors"] == {}


def test_analyze_many_isolates_route_completion_callback_failure(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: failure\n", encoding="utf-8")

    def failing_callback(route_result, route_trace):
        raise RuntimeError(f"cannot publish {route_result.route_id}")

    result = RestartAgent().analyze_many(
        RestartAgentRequest(log_path=str(log_path)),
        (
            ModelRoute(
                route_id="route-a",
                evidence_extractor=_FakeEvidenceExtractor({"justification": "route"}),
            ),
        ),
        on_route_complete=failing_callback,
    )

    assert result.model_results[0].route_id == "route-a"
    assert result.shared_analysis["route_completion_callback_errors"] == {
        "route-a": "RuntimeError: cannot publish route-a"
    }


def test_live_artifact_writer_uses_injected_clock(tmp_path):
    class _Clock:
        def __init__(self):
            self.monotonic_values = iter((10.0, 12.5))

        def monotonic(self):
            return next(self.monotonic_values)

        def now_utc(self):
            return datetime(2026, 7, 19, 1, 2, 3, tzinfo=timezone.utc)

    live_dir = tmp_path / "live"
    writer = LiveArtifactWriter(live_dir, clock=_Clock())
    writer.start(routes=[], config_metadata={})

    status = json.loads((live_dir / "run_status.json").read_text(encoding="utf-8"))
    event = json.loads((live_dir / "events.jsonl").read_text(encoding="utf-8"))
    assert status["started_at_utc"] == "2026-07-19T01:02:03+00:00"
    assert event["timestamp_utc"] == "2026-07-19T01:02:03+00:00"
    assert event["elapsed_s"] == 2.5


def test_live_artifacts_publish_fallback_and_routes_before_final_result(tmp_path):
    live_dir = tmp_path / "live"
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: failure\n", encoding="utf-8")
    l0_artifacts = []
    RestartAgent().run(
        RestartAgentRequest(log_path=str(log_path)),
        on_l0_ready=l0_artifacts.append,
    )
    writer = LiveArtifactWriter(live_dir)
    writer.start(
        routes=[
            {
                "route_id": "route-a",
                "model": "test-model",
                "endpoint": "https://example.invalid/v1",
                "credential_ref": "TEST_KEY_FILE",
            }
        ],
        config_metadata={"config_id": "test"},
    )
    writer.publish_l0_artifacts(
        l0_artifacts[0],
        {
            "l0_bundle": str(tmp_path / "l0_bundle.json"),
            "decision_evidence": str(tmp_path / "decision_evidence.json"),
        },
    )

    fallback_path = tmp_path / "deterministic_fallback.json"
    fallback_path.write_text("{}\n", encoding="utf-8")
    writer.publish_fallback(
        {
            "ready_wall_clock_s": 0.25,
            "result": {
                "decision": "RESTART",
                "decision_basis": "general_retry",
            },
        },
        artifact_path=str(fallback_path),
    )
    assert fallback_path.is_file()
    assert not (live_dir / "deterministic_fallback.json").exists()
    assert not (live_dir / "collect_all.result.json").exists()

    route_result_path = tmp_path / "model.test-model.result.json"
    route_trace_path = tmp_path / "model.test-model.trace.json"
    route_trace_path.write_text("{}\n", encoding="utf-8")
    route_result_path.write_text("{}\n", encoding="utf-8")
    writer.publish_route(
        {
            "route_id": "route-a",
            "model": "test-model",
            "execution_status": "ok",
            "l1_usable": True,
            "analysis_result": {"decision": "STOP"},
        },
        artifact_paths={
            "result_json": str(route_result_path),
            "trace_json": str(route_trace_path),
        },
    )
    status = json.loads((live_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "running"
    assert status["fallback"]["status"] == "ready"
    assert status["routes"]["route-a"]["status"] == "ok"
    assert status["routes"]["route-a"]["result_artifact"] == route_result_path.name
    assert status["routes"]["route-a"]["trace_artifact"] == route_trace_path.name
    assert not (live_dir / "routes").exists()

    writer.complete(
        final_artifacts={"trace_json": "batch.trace.json"},
    )
    final_status = json.loads((live_dir / "run_status.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (live_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert final_status["status"] == "completed"
    assert final_status["l0"]["status"] == "ready"
    assert [event["event"] for event in events] == [
        "run_started",
        "l0_artifacts_ready",
        "deterministic_fallback_ready",
        "route_completed",
        "run_completed",
    ]


def test_route_artifact_manifest_resolves_relative_paths_and_rejects_aliases(tmp_path):
    manifest_path = tmp_path / "route-artifacts.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "restart_agent_route_artifacts.v1",
                "routes": {
                    "fast": {
                        "result_json": "model.fast.result.json",
                        "trace_json": "model.fast.trace.json",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = load_route_artifact_manifest(
        manifest_path,
        expected_route_ids=["fast"],
    )

    assert manifest["fast"].result_json == (tmp_path / "model.fast.result.json")
    assert manifest["fast"].trace_json == (tmp_path / "model.fast.trace.json")

    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "restart_agent_route_artifacts.v1",
                "routes": {
                    "fast": {
                        "result_json": "shared.json",
                        "trace_json": "shared.json",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artifact path .* is shared"):
        load_route_artifact_manifest(
            manifest_path,
            expected_route_ids=["fast"],
        )


def test_analyze_many_returns_fallback_when_route_exceeds_deadline(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: failure\n", encoding="utf-8")
    extractor = _BlockingEvidenceExtractor({"justification": "unused"})

    started = time.monotonic()
    result = RestartAgent().analyze_many(
        RestartAgentRequest(log_path=str(log_path)),
        (ModelRoute(route_id="slow", evidence_extractor=extractor),),
        timeout_seconds=0.05,
    )
    elapsed = time.monotonic() - started
    extractor.release.set()
    extractor.completed.wait(timeout=2)

    route_result = result.model_results[0]
    assert elapsed < 0.5
    assert route_result.execution_status == "deadline_exceeded"
    assert route_result.l1_usable is False
    assert route_result.analysis_result == result.deterministic_result
    assert result.shared_analysis["analysis_timeout_seconds"] == 0.05
    assert result.shared_analysis["deadline_exceeded_route_count"] == 1


def test_analyze_many_isolates_route_orchestration_failure(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: failure\n", encoding="utf-8")

    class _ExplodingExtractor:
        def extract_evidence(self, context, *, deadline_monotonic=None):
            raise RuntimeError("route failed")

    result = RestartAgent().analyze_many(
        RestartAgentRequest(log_path=str(log_path)),
        (
            ModelRoute(
                route_id="broken",
                evidence_extractor=_ExplodingExtractor(),
                model="broken-model",
            ),
        ),
    )

    route_result = result.model_results[0]
    assert route_result.execution_status == "provider_error"
    assert route_result.l1_usable is False
    assert route_result.error == "RuntimeError: route failed"
    assert route_result.analysis_result.decision == result.deterministic_result.decision
    assert route_result.analysis_result.result_provenance["model_contribution"] == (
        "attempted_not_used_provider_error"
    )


def test_restart_agent_config_resolves_defaults_overrides_and_credentials(tmp_path, monkeypatch):
    primary_key = tmp_path / "primary.key"
    secondary_key = tmp_path / "secondary.key"
    primary_key.write_text("primary", encoding="utf-8")
    secondary_key.write_text("secondary", encoding="utf-8")
    monkeypatch.setenv("PRIMARY_ROUTE_KEY", str(primary_key))
    monkeypatch.setenv("SECONDARY_ROUTE_KEY", str(secondary_key))
    monkeypatch.setenv("LLM_API_KEY", "ambient-secret-must-not-win")
    config_path = tmp_path / "restart_agent.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "restart_agent_config.v1",
                "config_id": "qwen-fast-and-enriched",
                "config_version": 1,
                "routing": {"mode": "collect_all", "max_parallel_models": 2},
                "retry_policy": {
                    "bounded_retry_allowed_retries": 2,
                    "general_retry_allowed_retries": 4,
                },
                "model_defaults": {
                    "base_url": "https://shared.example/v1",
                    "credential_ref": "PRIMARY_ROUTE_KEY",
                    "request": {"temperature": 0, "top_p": 1},
                    "tools": {"enabled": False, "max_rounds": 0},
                    "reasoning": {"thinking_mode": "disable"},
                },
                "model_routes": [
                    {
                        "route_id": "fast",
                        "model": "provider/fast",
                    },
                    {
                        "route_id": "enriched",
                        "model": "provider/fast",
                        "base_url": "https://accurate.example/v1",
                        "credential_ref": "SECONDARY_ROUTE_KEY",
                        "tools": {
                            "enabled": True,
                            "advertisement": {
                                "overview": True,
                                "grep_log": False,
                                "read_window": True,
                                "get_evidence_objects": True,
                            },
                            "max_rounds": 2,
                        },
                        "reasoning": {"thinking_mode": "allow"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = load_restart_agent_config(config_path)
    routes = build_model_routes(config, LlmEvidenceExtractor)

    assert [(route.route_id, route.model, route.credential_ref) for route in routes] == [
        ("fast", "provider/fast", "PRIMARY_ROUTE_KEY"),
        ("enriched", "provider/fast", "SECONDARY_ROUTE_KEY"),
    ]
    assert config.max_parallel_models == 2
    assert config.timeout_seconds == 600.0
    assert config.retry_policy["bounded_retry_allowed_retries"] == 2
    assert config.retry_policy["general_retry_allowed_retries"] == 4
    assert "policy_version" not in config.retry_policy
    assert "policy_version" not in config.effective_config["retry_policy"]
    assert config.config_fingerprint.startswith("sha256:")
    effective_routes = config.effective_config["model_routes"]
    assert effective_routes[0]["request"]["temperature"] == 0.0
    assert effective_routes[0]["tools"] == {
        "enabled": False,
        "advertisement": {
            "overview": True,
            "grep_log": True,
            "read_window": True,
            "get_evidence_objects": False,
        },
        "effective_advertised": [],
        "max_rounds": 0,
    }
    assert effective_routes[1]["tools"] == {
        "enabled": True,
        "advertisement": {
            "overview": True,
            "grep_log": False,
            "read_window": True,
            "get_evidence_objects": True,
        },
        "effective_advertised": [
            "overview",
            "read_window",
            "get_evidence_objects",
        ],
        "max_rounds": 2,
    }
    assert routes[0].evidence_extractor._config.api_key is None
    assert routes[0].evidence_extractor._config.api_key_file == str(primary_key)
    assert "ambient-secret-must-not-win" not in json.dumps(config.metadata())
    assert "primary" not in json.dumps(config.metadata())
    assert "secondary" not in json.dumps(config.metadata())


def test_restart_agent_config_rejects_unimplemented_top_level_sections(tmp_path):
    config_path = tmp_path / "restart_agent.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "restart_agent_config.v1",
                "config_id": "unsupported-section",
                "config_version": 1,
                "routing": {"mode": "collect_all"},
                "model_routes": [{"route_id": "model-a", "model": "provider/model-a"}],
                "history": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported fields: history"):
        load_restart_agent_config(config_path, environ={"LLM_API_KEY_FILE": "/tmp/key"})


def test_cli_collect_all_writes_batch_result_and_trace_for_unavailable_log(
    tmp_path,
    monkeypatch,
    capsys,
):
    key_path = tmp_path / "route.key"
    key_path.write_text("secret", encoding="utf-8")
    monkeypatch.setenv("ROUTE_KEY", str(key_path))
    config_path = tmp_path / "restart_agent.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "restart_agent_config.v1",
                "config_id": "unavailable-log-test",
                "config_version": 1,
                "routing": {
                    "mode": "collect_all",
                    "max_parallel_models": 1,
                    "timeout_seconds": 37,
                },
                "model_routes": [
                    {
                        "route_id": "model-a",
                        "model": "provider/model-a",
                        "credential_ref": "ROUTE_KEY",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    trace_path = tmp_path / "batch.trace.json"
    result_path = tmp_path / "batch.result.json"
    route_result_path = tmp_path / "model-a.result.json"
    route_trace_path = tmp_path / "model-a.trace.json"
    route_manifest_path = tmp_path / "route-artifacts.json"
    route_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "restart_agent_route_artifacts.v1",
                "routes": {
                    "model-a": {
                        "result_json": str(route_result_path),
                        "trace_json": str(route_trace_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    live_dir = tmp_path / "live"
    missing_log = tmp_path / "missing.log"

    rc = cli_main(
        [
            str(missing_log),
            "--config",
            str(config_path),
            "--trace-json",
            str(trace_path),
            "--result-json",
            str(result_path),
            "--route-artifact-manifest",
            str(route_manifest_path),
            "--incremental-artifact-dir",
            str(live_dir),
            "--summary",
        ]
    )

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert result["schema_version"] == "restart_agent_collect_all.v1"
    assert result["model_results"][0]["execution_status"] == ("not_run_log_unavailable")
    assert trace["schema_version"] == "restart_agent_cli_collect_all_trace.v1"
    assert json.loads(result_path.read_text(encoding="utf-8")) == result
    assert trace["analyzer_trace"]["routing_mode"] == "collect_all"
    config_metadata = result["shared_analysis"]["restart_agent_config"]
    assert config_metadata["config_id"] == "unavailable-log-test"
    assert config_metadata["config_fingerprint"].startswith("sha256:")
    assert result["shared_analysis"]["analysis_timeout_seconds"] == 37.0
    assert config_metadata["timeout_seconds"] == 37.0
    assert "restart_agent collect_all:" in captured.err
    live_status = json.loads((live_dir / "run_status.json").read_text(encoding="utf-8"))
    live_events = [
        json.loads(line)
        for line in (live_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert live_status["status"] == "completed"
    assert live_status["fallback"]["status"] == "not_published"
    assert live_status["routes"]["model-a"]["status"] == "not_run_log_unavailable"
    assert route_result_path.is_file()
    assert route_trace_path.is_file()
    assert not (live_dir / "routes").exists()
    assert [event["event"] for event in live_events] == [
        "run_started",
        "route_completed",
        "run_completed",
    ]


def test_run_owns_serializable_immutable_execution_artifacts(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: failure\n", encoding="utf-8")

    run = RestartAgent().run(RestartAgentRequest(log_path=str(log_path)))

    assert run.result.decision == Decision.RESTART.value
    assert run.bundle is not None
    assert run.decision_evidence is not None
    assert run.trace["log_path"] == str(log_path)
    json.dumps(run.trace, sort_keys=True)
    with pytest.raises(TypeError, match="frozen JSON payload"):
        run.trace["log_path"] = "other.log"


def test_core_restart_agent_exposes_only_invocation_owned_artifacts(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: failure\n", encoding="utf-8")

    analyzer = CoreRestartAgent()
    run = analyzer.run(RestartAgentRequest(log_path=str(log_path)))

    assert run.bundle is not None
    assert not hasattr(analyzer, "last_trace")
    assert not hasattr(analyzer, "analyze")


def test_public_request_requires_exact_schema_and_rejects_internal_fields(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: failure\n", encoding="utf-8")
    analyzer = CoreRestartAgent()

    with pytest.raises(ValueError, match="schema_version"):
        analyzer.run({"log_path": str(log_path)})
    with pytest.raises(ValueError, match="attempt_history"):
        analyzer.run(
            {
                "schema_version": "restart_agent_request.v1",
                "log_path": str(log_path),
                "attempt_history": [],
            }
        )
    with pytest.raises(TypeError, match="job_id must be a string"):
        analyzer.run(
            {
                "schema_version": "restart_agent_request.v1",
                "log_path": str(log_path),
                "job_id": 123,
            }
        )
    with pytest.raises(ValueError, match="unsupported analysis_mode"):
        analyzer.run(
            {
                "schema_version": "restart_agent_request.v1",
                "log_path": str(log_path),
                "analysis_mode": "eventually",
            }
        )


def test_runtime_selects_history_behind_public_request_boundary(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: failure\n", encoding="utf-8")
    request = RestartAgentRequest(log_path=str(log_path), job_id="job-1", cycle_id=2)
    runtime = RestartAgentRuntime(
        CoreRestartAgent(),
        attempt_record_store=InMemoryAttemptRecordStore(),
    )
    run = runtime.analyze_one(request)

    assert run.trace["request"] == request.to_payload()
    assert "attempt_history" not in run.trace["request"]
    assert run.trace["runtime_history"]["availability_reason"] == "ready"
    assert run.result.schema_version == "restart_agent_response.v1"


def test_l0b_shared_payload_is_deeply_immutable_and_json_compatible(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: failure\n", encoding="utf-8")
    model_view = build_l0_model_facing_view(build_l0_bundle(str(log_path)))

    json.dumps(model_view.evidence_bundle, sort_keys=True)
    with pytest.raises(TypeError, match="frozen JSON payload"):
        model_view.evidence_bundle["line_count"] = 2
    with pytest.raises(TypeError, match="frozen JSON payload"):
        model_view.evidence_bundle["context_windows"].append({})


def test_same_agent_concurrent_calls_do_not_overwrite_run_artifacts(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f"job-{index}.log"
        path.write_text(f"RuntimeError: failure {index}\n", encoding="utf-8")
        paths.append(path)
    analyzer = RestartAgent()

    def analyze(path):
        result = analyzer.analyze(RestartAgentRequest(log_path=str(path)))
        return result, analyzer.last_trace["log_path"], analyzer.last_bundle.log_path

    with ThreadPoolExecutor(max_workers=2) as pool:
        observed = list(pool.map(analyze, paths))

    assert [item[1] for item in observed] == [str(path) for path in paths]
    assert [item[2] for item in observed] == [str(path) for path in paths]


def test_l0_contract_contains_no_history_input_or_coverage(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: failure\n", encoding="utf-8")

    bundle = build_l0_bundle(str(log_path))

    assert "history" not in bundle.evidence_coverage


def test_l4_rejects_untyped_stage_inputs():
    with pytest.raises(TypeError, match="model_recovery_assessment must be typed"):
        L4PolicyInput(
            primary=None,
            history=HistorySummary(),
            model_recovery_assessment={"failure_domain": "unknown"},
        )
    with pytest.raises(TypeError, match="retry_policy must be typed"):
        L4PolicyInput(
            primary=None,
            history=HistorySummary(),
            retry_policy={"general_retry_allowed_retries": 3},
        )


def test_restart_agent_reuses_injected_log_snapshot_across_stages(tmp_path):
    log_path = tmp_path / "job.log"
    text = "iteration 1 completed\nRuntimeError: observed failure\n"
    log_path.write_text(text, encoding="utf-8")
    snapshot = LogSnapshot(
        path=str(log_path),
        lines=tuple(text.splitlines()),
        byte_size=len(text.encode("utf-8")),
    )

    class _CountingLogSource:
        def __init__(self, path):
            assert path == str(log_path)
            self.path = path
            self.snapshot_calls = 0

        def unavailable_reason(self):
            return None

        def snapshot(self):
            self.snapshot_calls += 1
            return snapshot

    sources = []

    def source_factory(path):
        source = _CountingLogSource(path)
        sources.append(source)
        return source

    run = CoreRestartAgent(log_source_factory=source_factory).run(
        RestartAgentRequest(log_path=str(log_path))
    )

    assert run.bundle is not None
    assert run.bundle.line_count == 2
    assert len(sources) == 1
    assert sources[0].snapshot_calls == 1


def test_llm_evidence_extractor_accepts_injected_transport():
    class _Transport:
        def __init__(self):
            self.calls = []

        def call(self, **kwargs):
            self.calls.append(kwargs)
            return {"choices": []}, {"success": True}

    transport = _Transport()
    extractor = LlmEvidenceExtractor(
        LlmConfig(api_key="test-key"),
        transport=transport,
    )

    response, record = extractor._call_model(model_turn=1)

    assert response == {"choices": []}
    assert record == {"success": True}
    assert transport.calls == [{"model_turn": 1}]


def test_llm_config_reads_only_supplied_environment():
    config = LlmConfig.from_env(
        environ={
            "NVRX_LLM_MODEL": "test/model",
            "NVRX_LLM_TIMEOUT_SECONDS": "17",
            "LLM_API_KEY": "supplied-key",
        }
    )

    assert config.model == "test/model"
    assert config.timeout_seconds == 17.0
    assert config.api_key == "supplied-key"


def test_restart_agent_config_parser_has_no_file_dependency(tmp_path):
    key_path = tmp_path / "key"
    key_path.write_text("secret", encoding="utf-8")
    config = parse_restart_agent_config(
        {
            "schema_version": "restart_agent_config.v1",
            "config_id": "pure-parser",
            "config_version": 1,
            "model_routes": [{"route_id": "route-a"}],
        },
        environ={"LLM_API_KEY_FILE": str(key_path)},
    )

    assert config.config_id == "pure-parser"
    assert config.model_route_specs[0].route_id == "route-a"


def test_llm_extractor_uses_injected_credential_provider(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: failed\n", encoding="utf-8")
    snapshot = LogSnapshot.read(log_path)
    bundle = build_l0_bundle(str(log_path), source_log=snapshot)
    decision_evidence = build_decision_evidence(bundle)
    model_view = _build_l0_model_facing_view(bundle, decision_evidence)
    context = build_l1_evidence_context(bundle, model_view, snapshot)

    class _CredentialProvider:
        def __init__(self):
            self.configs = []

        def load(self, config):
            self.configs.append(config)
            raise RuntimeError("credential sentinel")

    provider = _CredentialProvider()
    extractor = LlmEvidenceExtractor(
        LlmConfig(api_key_file="must-not-be-read"),
        credential_provider=provider,
    )

    result = extractor.extract_evidence(context)

    assert result.success is False
    assert result.errors == ("credential sentinel",)
    assert provider.configs == [extractor._config]


def test_openai_transport_uses_injected_http_client_and_clock():
    class _Clock:
        def __init__(self):
            self.values = iter((10.0, 12.0))

        def monotonic(self):
            return next(self.values)

    class _Response:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}]}'

    class _HttpClient:
        def __init__(self):
            self.calls = []

        def open(self, request, *, timeout):
            self.calls.append((request.full_url, timeout))
            return _Response()

    client = _HttpClient()
    transport = OpenAICompatibleTransport(
        LlmConfig(base_url="https://provider.example/v1", timeout_seconds=9),
        http_client=client,
        clock=_Clock(),
    )

    _payload, record = transport.call(
        api_key="secret",
        messages=[],
        include_tools=False,
        model_turn=1,
    )

    assert client.calls == [("https://provider.example/v1/chat/completions", 9)]
    assert record["latency_s"] == 2.0


def test_retry_transport_uses_injected_sleeper():
    calls = []

    def request_call(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise LlmCallError(
                "retry",
                {"retryable": True, "latency_s": 0.1, "error_type": "http_error"},
            )
        return {"choices": []}, {"success": True}

    class _Sleeper:
        def __init__(self):
            self.delays = []

        def sleep(self, seconds):
            self.delays.append(seconds)

    sleeper = _Sleeper()
    transport = RetryingChatTransport(
        LlmConfig(max_retries=1, retry_backoff_seconds=0.75),
        request_call,
        sleeper=sleeper,
    )

    outcome = transport.call(
        api_key="secret",
        messages=[],
        include_tools=False,
        model_turn=1,
        deadline_monotonic=None,
    )

    assert outcome.call_record == {"success": True}
    assert sleeper.delays == [0.75]


def test_model_route_specs_are_composed_by_injected_factory(tmp_path):
    key_path = tmp_path / "key"
    key_path.write_text("secret", encoding="utf-8")
    config_path = tmp_path / "restart_agent.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "restart_agent_config.v1",
                "config_id": "injected-factory",
                "config_version": 1,
                "model_routes": [
                    {
                        "route_id": "route-a",
                        "model": "provider/model-a",
                        "base_url": "https://provider.example/v1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = load_restart_agent_config(
        config_path,
        environ={"LLM_API_KEY_FILE": str(key_path)},
    )
    captured = []

    class _Extractor:
        def extract_evidence(self, context, *, deadline_monotonic=None):
            raise AssertionError("not invoked")

    def extractor_factory(llm_config):
        captured.append(llm_config)
        return _Extractor()

    routes = build_model_routes(config, extractor_factory)

    assert [spec.route_id for spec in config.model_route_specs] == ["route-a"]
    assert [route.route_id for route in routes] == ["route-a"]
    assert captured == [config.model_route_specs[0].llm_config]
