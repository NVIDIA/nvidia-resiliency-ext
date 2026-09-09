# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded operational logging for attrsvc's Restart Agent adapter."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nvidia_resiliency_ext.attribution.restart_agent import (
    DecisionCandidate,
    L0Artifacts,
    ModelAnalysisResult,
)


@dataclass(frozen=True)
class RestartAgentLogContext:
    """Stable service identity attached to every attempt log event."""

    job_id: str | None
    cycle_id: int | None
    log_path: str


class RestartAgentOperationalLogger:
    """Translate typed stage outputs and traces into bounded service events."""

    def __init__(self, target: logging.Logger) -> None:
        self._target = target

    def info(self, event: str, context: RestartAgentLogContext, **fields: Any) -> None:
        self._target.info(_event_message(event, context, fields))

    def debug(self, event: str, context: RestartAgentLogContext, **fields: Any) -> None:
        if self._target.isEnabledFor(logging.DEBUG):
            self._target.debug(_event_message(event, context, fields))

    def warning(self, event: str, context: RestartAgentLogContext, **fields: Any) -> None:
        self._target.warning(_event_message(event, context, fields))

    def error(self, event: str, context: RestartAgentLogContext, **fields: Any) -> None:
        self._target.error(_event_message(event, context, fields))

    def l0_completed(
        self,
        context: RestartAgentLogContext,
        artifacts: L0Artifacts,
        *,
        progressive_metrics: Mapping[str, Any] | None = None,
    ) -> None:
        bundle = artifacts.bundle
        primary = bundle.deterministic_primary_candidate
        metrics = dict(progressive_metrics or {})
        source_ingest_s = _metric_float(metrics, "source_ingest_wall_clock_s", 0.0)
        evidence_assembly_s = _metric_float(
            metrics,
            "l0a_bundle_wall_clock_s",
            artifacts.l0a_wall_clock_s,
        )
        decision_evidence_s = _metric_float(
            metrics,
            "decision_evidence_wall_clock_s",
            artifacts.decision_evidence_wall_clock_s,
        )
        self.info(
            "restart_agent.l0a.completed",
            context,
            status="completed",
            wall_clock_s=artifacts.l0a_wall_clock_s,
            source_ingest_s=source_ingest_s,
            evidence_assembly_s=evidence_assembly_s,
            decision_evidence_s=decision_evidence_s,
            cumulative_compute_s=(source_ingest_s + evidence_assembly_s + decision_evidence_s),
            reused=artifacts.l0_reused,
            source_bytes=bundle.byte_size,
            source_lines=bundle.line_count,
            occurrence_groups=len(bundle.occurrence_groups),
            candidate_anchors=len(bundle.candidate_anchors),
            failure_episodes=len(bundle.failure_episodes),
            distributed_incidents=len(bundle.distributed_failure_incidents),
            primary_line=primary.line if primary is not None else None,
            primary_class=primary.failure_class if primary is not None else None,
            root_fingerprint_ready=bool(primary and primary.root_fingerprint),
        )
        self.info(
            "restart_agent.decision_evidence.completed",
            context,
            wall_clock_s=artifacts.decision_evidence_wall_clock_s,
            cumulative_selection_s=decision_evidence_s,
            primary_line=(
                artifacts.decision_evidence.deterministic_primary_candidate.line
                if artifacts.decision_evidence.deterministic_primary_candidate is not None
                else None
            ),
            selected_reference_groups=sum(
                bool(value)
                for value in artifacts.decision_evidence.selected_evidence_references.values()
            ),
            root_fingerprint_ready=bool(
                artifacts.decision_evidence.canonical_observed_identity.get("root_fingerprint")
            ),
        )
        self.debug(
            "restart_agent.l0a.detail",
            context,
            source_decode_s=metrics.get("source_decode_wall_clock_s"),
            source_index_classify_s=metrics.get("source_index_classify_wall_clock_s"),
            source_ingest_s=metrics.get("source_ingest_wall_clock_s"),
            evidence_assembly_s=metrics.get("l0a_bundle_wall_clock_s"),
            decision_evidence_s=metrics.get("decision_evidence_wall_clock_s"),
            l0a_reduction_s=metrics.get("l0a_reduction_wall_clock_s"),
            read_mode=metrics.get("read_mode"),
            chunks=metrics.get("chunk_count"),
            bytes_ingested=metrics.get("bytes_ingested"),
            bytes_reread=metrics.get("bytes_reread"),
            decode_replacements=metrics.get("decode_replacement_count"),
            decode_replacement_lines=metrics.get("decode_replacement_line_count"),
            resets=metrics.get("reset_count"),
            pending_line_bytes=metrics.get("pending_line_bytes"),
            discarded_incomplete_tail_bytes=metrics.get("discarded_incomplete_tail_bytes"),
        )
        if artifacts.model_view is not None:
            self._l0b_completed(context, artifacts)

    def _l0b_completed(
        self,
        context: RestartAgentLogContext,
        artifacts: L0Artifacts,
    ) -> None:
        model_view = artifacts.model_view
        if model_view is None:
            return
        metrics = _mapping(model_view.projection_metrics)
        view_size = _mapping(metrics.get("view_size"))
        selection = _mapping(metrics.get("selection_counts"))
        compaction = _mapping(metrics.get("compaction_counts"))
        integrity = _mapping(metrics.get("projection_integrity"))
        selected_windows = len(model_view.evidence_bundle.get("context_windows") or ())
        omitted_objects = sum(
            _as_int(_mapping(value).get("omitted")) for value in selection.values()
        )
        truncated_objects = sum(
            _as_int(value) for key, value in compaction.items() if str(key).startswith("truncated_")
        )
        self.info(
            "restart_agent.l0b.completed",
            context,
            status="completed",
            wall_clock_s=artifacts.l0b_wall_clock_s,
            compact_json_chars=view_size.get("compact_json_characters"),
            estimated_tokens=view_size.get("estimated_tokens"),
            selected_windows=selected_windows,
            omitted_objects=omitted_objects,
            truncated_objects=truncated_objects,
            projection_integrity=integrity.get("status"),
        )
        self.debug(
            "restart_agent.l0b.detail",
            context,
            budget_utilization=_compact_mapping(metrics.get("budget_utilization")),
            selection_counts=_compact_mapping(selection),
            compaction_counts=_compact_mapping(compaction),
            payload_hash=integrity.get("deterministic_payload_sha256"),
        )

    def deterministic_candidate_ready(
        self,
        context: RestartAgentLogContext,
        candidate: DecisionCandidate,
        *,
        terminal_to_ready_s: float | None,
    ) -> None:
        self._candidate_ready(
            context,
            candidate_kind=candidate.candidate_kind,
            route_id=None,
            decision=candidate.result.decision,
            decision_basis=candidate.result.decision_basis,
            terminal_to_ready_s=terminal_to_ready_s,
        )
        self._l3_from_candidate(context, candidate)
        self._l4_from_candidate(context, candidate)

    def route_completed(
        self,
        context: RestartAgentLogContext,
        result: ModelAnalysisResult,
        trace: Mapping[str, Any],
        *,
        terminal_to_ready_s: float | None,
    ) -> None:
        layers = _mapping(trace.get("layers"))
        l1 = _mapping(layers.get("L1"))
        l2 = _mapping(layers.get("L2"))
        l3 = _mapping(layers.get("L3"))
        l4_policy = _mapping(_mapping(trace.get("l4_policy")).get("retry_policy"))
        l4 = {**l4_policy, **_mapping(layers.get("L4"))}
        candidate_kind = result.selected_candidate_kind
        l1_raw = _mapping(trace.get("l1"))
        model_calls = _sequence_of_mappings(l1_raw.get("model_calls"))
        tool_calls = _sequence_of_mappings(l1_raw.get("tool_calls"))
        endpoint_issues = sum(_is_endpoint_issue(item) for item in model_calls)
        execution_assessment = _mapping(result.l1_execution_assessment)
        l1_fields = {
            "route_id": result.route_id,
            "model": result.model,
            "execution_status": result.execution_status,
            "unusable_reason": execution_assessment.get("unusable_reason"),
            "reason_codes": execution_assessment.get("reason_codes") or [],
            "usable": result.l1_usable,
            "wall_clock_s": l1.get("wall_clock_s"),
            "model_wall_clock_s": l1.get("model_call_wall_clock_s"),
            "tool_wall_clock_s": l1.get("tool_wall_clock_s"),
            "model_calls": l1.get("model_calls", len(model_calls)),
            "tool_calls": l1.get("tool_calls", len(tool_calls)),
            "failed_calls": l1.get("failed_model_calls", endpoint_issues),
            "retries": l1.get("retried_model_calls", _retry_count(model_calls)),
            "endpoint_issues": endpoint_issues,
            "total_tokens": l1.get("total_tokens"),
        }
        if result.l1_usable and result.execution_status == "completed":
            self.info("restart_agent.l1.completed", context, **l1_fields)
        else:
            self.warning(
                "restart_agent.l1.completed",
                context,
                **l1_fields,
                error_classification=_l1_error_classification(trace, result),
            )
        self._l1_call_details(context, result.route_id, model_calls, tool_calls)
        self._l2_completed(context, result.route_id, l2)
        self._l3_completed(context, candidate_kind, result.route_id, l3)
        self._l4_completed(context, candidate_kind, result.route_id, l4)
        self._candidate_ready(
            context,
            candidate_kind=candidate_kind,
            route_id=result.route_id,
            decision=result.analysis_result.decision,
            decision_basis=result.analysis_result.decision_basis,
            terminal_to_ready_s=terminal_to_ready_s,
        )

    def _l1_call_details(
        self,
        context: RestartAgentLogContext,
        route_id: str,
        model_calls: tuple[Mapping[str, Any], ...],
        tool_calls: tuple[Mapping[str, Any], ...],
    ) -> None:
        for index, call in enumerate(model_calls, start=1):
            usage = _mapping(call.get("usage"))
            self.debug(
                "restart_agent.l1.model_call.completed",
                context,
                route_id=route_id,
                call_index=index,
                model_turn=call.get("model_turn"),
                attempt=call.get("attempt"),
                latency_s=call.get("latency_s"),
                success=call.get("success"),
                retry_scheduled=call.get("retry_scheduled", False),
                finish_reason=call.get("finish_reason"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                timeout=call.get("timeout", False),
                timeout_kind=call.get("timeout_kind"),
                error_type=call.get("error_type"),
            )
        for index, call in enumerate(tool_calls, start=1):
            self.debug(
                "restart_agent.l1.tool_call.completed",
                context,
                route_id=route_id,
                call_index=index,
                model_turn=call.get("model_turn"),
                tool=call.get("name"),
                latency_s=(
                    _as_float(call.get("latency_ms")) / 1000
                    if call.get("latency_ms") is not None
                    else None
                ),
                success=not bool(call.get("error")),
                returned_lines=call.get("result_lines"),
                returned_context=(
                    _as_int(call.get("result_lines")) > 0
                    if call.get("result_lines") is not None
                    else None
                ),
                new_context="unknown",
                truncated=call.get("truncated", False),
            )

    def _l2_completed(
        self,
        context: RestartAgentLogContext,
        route_id: str,
        layer: Mapping[str, Any],
    ) -> None:
        status = _l2_execution_status(layer)
        self.info(
            "restart_agent.l2.completed",
            context,
            route_id=route_id,
            status=status,
            wall_clock_s=layer.get("wall_clock_s"),
            grounding_status=layer.get("grounding_status"),
            history_identity_ready=layer.get("history_identity_ready"),
            root_fingerprint_source=layer.get("root_fingerprint_source"),
            affected_entity_ready=layer.get("affected_entity_available"),
            audit_status=layer.get("audit_status"),
            observational_findings=layer.get("observational_finding_count"),
        )
        self.debug(
            "restart_agent.l2.detail",
            context,
            route_id=route_id,
            grounding_method=layer.get("grounding_method"),
            citation_count=layer.get("citation_count"),
            nearby_resolved=layer.get("nearby_resolved_count"),
            rendered_exact=layer.get("rendered_exact_count"),
            abbreviated_exact=layer.get("abbreviated_exact_count"),
            identity_lineage=layer.get("identity_lineage"),
            finding_count=layer.get("finding_count"),
            grounding_adjustments=layer.get("grounding_adjustment_count"),
        )

    def _l3_from_candidate(
        self,
        context: RestartAgentLogContext,
        candidate: DecisionCandidate,
    ) -> None:
        layer = dict(candidate.history_summary)
        layer["wall_clock_s"] = candidate.stage_timings.get("l3_wall_clock_s")
        self._l3_completed(context, candidate.candidate_kind, None, layer)

    def _l3_completed(
        self,
        context: RestartAgentLogContext,
        candidate_kind: str,
        route_id: str | None,
        layer: Mapping[str, Any],
    ) -> None:
        self.info(
            "restart_agent.l3.completed",
            context,
            candidate_kind=candidate_kind,
            route_id=route_id,
            wall_clock_s=layer.get("wall_clock_s"),
            history_available=layer.get("history_available", layer.get("available")),
            selected_failure_facts_source=layer.get("selected_failure_facts_source"),
            same_job_attempts=layer.get("same_job_attempts"),
            matching_root_attempts=layer.get("matching_root_attempts"),
            same_entity_attempts=layer.get("same_entity_attempts"),
            no_observed_advance_attempts=layer.get("no_observed_advance_attempts"),
            consecutive_root_no_advance=layer.get("consecutive_same_root_no_advance_attempts"),
            consecutive_root_entity_no_advance=layer.get(
                "consecutive_same_root_and_entity_no_advance_attempts"
            ),
        )
        self.debug(
            "restart_agent.l3.detail",
            context,
            candidate_kind=candidate_kind,
            route_id=route_id,
            observed_advance_attempts=layer.get("observed_advance_attempts"),
            unknown_progress_attempts=layer.get("unknown_progress_attempts"),
            exact_failure_position_attempts=layer.get("exact_failure_position_attempts"),
            different_entity_attempts=layer.get("different_entity_attempts"),
            unknown_entity_attempts=layer.get("unknown_entity_attempts"),
        )

    def _l4_from_candidate(
        self,
        context: RestartAgentLogContext,
        candidate: DecisionCandidate,
    ) -> None:
        policy = dict(candidate.result.retry_policy)
        policy["wall_clock_s"] = candidate.stage_timings.get("l4_wall_clock_s")
        policy.setdefault("decision", candidate.result.decision)
        policy.setdefault("decision_basis", candidate.result.decision_basis)
        self._l4_completed(context, candidate.candidate_kind, None, policy)

    def _l4_completed(
        self,
        context: RestartAgentLogContext,
        candidate_kind: str,
        route_id: str | None,
        layer: Mapping[str, Any],
    ) -> None:
        general = _mapping(layer.get("general_root_ceiling"))
        selected = _mapping(layer.get("selected_policy_ledger"))
        effective = _mapping(layer.get("effective_policy"))
        job_no_progress = _mapping(layer.get("job_no_progress_guard"))
        job_unknown = _mapping(layer.get("job_unknown_progress_guard"))
        self.info(
            "restart_agent.l4.completed",
            context,
            candidate_kind=candidate_kind,
            route_id=route_id,
            wall_clock_s=layer.get("wall_clock_s"),
            base_rule=layer.get("base_rule"),
            effective_rule=effective.get("rule"),
            decision=layer.get("decision"),
            decision_basis=layer.get("decision_basis"),
            retry_budget_exhausted=layer.get("retry_budget_exhausted"),
            general_matching_attempts=general.get("matching_prior_attempts"),
            selected_policy_matching_attempts=selected.get("matching_prior_attempts"),
            allowed_retries=effective.get("allowed_retries"),
            job_no_progress_attempts=job_no_progress.get("matching_prior_attempts"),
            job_unknown_progress_attempts=job_unknown.get("matching_prior_attempts"),
            exhausted_by=layer.get("exhausted_by"),
        )
        self.debug(
            "restart_agent.l4.detail",
            context,
            candidate_kind=candidate_kind,
            route_id=route_id,
            general_ledger=_compact_mapping(general),
            selected_ledger=_compact_mapping(selected),
            current_evidence_qualified=layer.get("current_evidence_qualified"),
            failure_domain_grounded=layer.get("failure_domain_grounded"),
            retry_outlook_grounded=layer.get("retry_outlook_grounded"),
            match_requirements=_compact_mapping(layer.get("match_requirements")),
        )

    def _candidate_ready(
        self,
        context: RestartAgentLogContext,
        *,
        candidate_kind: str,
        route_id: str | None,
        decision: str,
        decision_basis: str,
        terminal_to_ready_s: float | None,
    ) -> None:
        self.info(
            "restart_agent.candidate.ready",
            context,
            candidate_kind=candidate_kind,
            route_id=route_id,
            decision=decision,
            decision_basis=decision_basis,
            terminal_to_ready_s=terminal_to_ready_s,
        )


def _event_message(
    event: str,
    context: RestartAgentLogContext,
    fields: Mapping[str, Any],
) -> str:
    ordered = {
        "event": event,
        "job_id": context.job_id,
        "cycle_id": context.cycle_id,
        "log_path": context.log_path,
        **fields,
    }
    return " ".join(f"{key}={_log_value(value)}" for key, value in ordered.items())


def _log_value(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (list, tuple, set, frozenset)):
        return ",".join(str(item) for item in value) or "none"
    text = str(value).replace("\n", "\\n").replace("\r", "\\r")
    return text if not any(character.isspace() for character in text) else repr(text)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence_of_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _metric_float(metrics: Mapping[str, Any], key: str, fallback: float) -> float:
    if key not in metrics or metrics.get(key) is None:
        return float(fallback)
    return _as_float(metrics.get(key))


def _l2_execution_status(layer: Mapping[str, Any]) -> str:
    if not layer:
        return "not_available"
    grounding_status = layer.get("grounding_status")
    audit_status = layer.get("audit_status")
    if grounding_status == "not_run" and audit_status == "not_run":
        return "not_run"
    return "completed"


def _retry_count(model_calls: tuple[Mapping[str, Any], ...]) -> int:
    return sum(bool(call.get("retry_scheduled")) for call in model_calls)


def _is_endpoint_issue(call: Mapping[str, Any]) -> bool:
    if call.get("success"):
        return False
    error_type = str(call.get("error_type") or "")
    if error_type in {
        "analysis_deadline_exceeded",
        "context_budget_exceeded",
        "context_window_exceeded",
    }:
        return False
    if call.get("timeout") or call.get("http_status") is not None:
        return True
    return error_type in {
        "close_error",
        "connect_error",
        "http_error",
        "proxy_error",
        "provider_response_decode_error",
        "read_error",
        "remote_protocol_error",
        "request_error",
        "response_decoding_error",
        "timeout",
        "write_error",
    }


def _compact_mapping(value: Any) -> str:
    mapping = _mapping(value)
    if not mapping:
        return "none"
    return ",".join(f"{key}:{_log_value(mapping[key])}" for key in sorted(mapping))


def _l1_error_classification(
    trace: Mapping[str, Any],
    result: ModelAnalysisResult,
) -> str:
    unusable_reason = result.l1_execution_assessment.get("unusable_reason")
    if unusable_reason:
        return str(unusable_reason)
    anomalies = _mapping(trace.get("anomalies"))
    if anomalies.get("provider_timeout"):
        return "provider_timeout"
    if anomalies.get("context_window_exceeded"):
        return "context_window_exceeded"
    if anomalies.get("token_limit_hit"):
        return "token_limit"
    if anomalies.get("provider_error"):
        return str(anomalies.get("provider_error_type") or "provider_error")
    if result.error:
        return "route_error"
    return "unusable_output"
