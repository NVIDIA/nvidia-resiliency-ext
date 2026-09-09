# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L1 invocation boundary, output health, and deadline degradation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from ..infrastructure.log_source import LogSnapshot
from ..models import L0Bundle, L0ModelFacingView
from ..runtime import SYSTEM_CLOCK, Clock
from .advisories import model_evidence_contract_advisories
from .contracts import (
    EvidenceExtractor,
    L1EvidenceResult,
    L1ExecutionAssessment,
    L1ExecutionReason,
    L1ExecutionStatus,
    L1FinalEvidenceReason,
    L1ParseStatus,
    L1ResultQuality,
)
from .normalization import normalize_model_evidence_payload
from .tools import EvidenceToolsFactory, build_l1_evidence_context
from .validation import model_evidence_contract_errors


def extract_timed(
    extractor: EvidenceExtractor,
    bundle: L0Bundle,
    model_view: L0ModelFacingView,
    source_log: LogSnapshot,
    deadline_monotonic: float | None,
    tools_factory: EvidenceToolsFactory | None = None,
    clock: Clock = SYSTEM_CLOCK,
) -> tuple[L1EvidenceResult, float, float]:
    """Invoke one extractor and contain provider-specific exceptions."""

    started = clock.monotonic()
    try:
        result = extractor.extract_evidence(
            build_l1_evidence_context(
                bundle,
                model_view,
                source_log,
                tools_factory=tools_factory,
            ),
            deadline_monotonic=deadline_monotonic,
        )
    except Exception as exc:  # defensive boundary for custom extractors
        result = L1EvidenceResult(
            semantic_payload=None,
            model=type(extractor).__name__,
            success=False,
            errors=(f"{type(exc).__name__}: {exc}",),
            anomalies={
                "l1_enabled": True,
                "provider_error": True,
                "provider_error_type": "extractor_exception",
            },
        )
    result = _normalize_result(result)
    completed = clock.monotonic()
    return result, round(completed - started, 3), completed


def deadline_exceeded_result(
    *,
    model: str,
    final_evidence_reason: str | None = None,
) -> L1EvidenceResult:
    """Represent an L1 route that did not complete inside the analysis budget."""

    error = "analysis deadline exceeded before L1 completed"
    anomalies: dict[str, Any] = {
        "l1_enabled": True,
        "provider_error": True,
        "provider_error_type": "analysis_deadline_exceeded",
        "provider_timeout": True,
        "deadline_exceeded": True,
    }
    if final_evidence_reason is not None:
        anomalies["final_evidence_turn"] = True
        anomalies["final_evidence_reason"] = final_evidence_reason
    return L1EvidenceResult(
        semantic_payload=None,
        model=model,
        success=False,
        errors=(error,),
        anomalies=anomalies,
    )


def assess_execution(
    *,
    configured: bool,
    result: L1EvidenceResult,
) -> L1ExecutionAssessment:
    """Build the single closed execution assessment for one L1 result."""

    if not configured:
        return L1ExecutionAssessment(
            execution_status=L1ExecutionStatus.NOT_RUN,
            result_quality=L1ResultQuality.NOT_APPLICABLE,
            parse_status=L1ParseStatus.NOT_RUN,
            evidence_present=False,
        )

    degradation_reasons: list[L1ExecutionReason] = []
    final_evidence_reason = _final_evidence_reason(result)
    if result.anomalies.get("final_evidence_reason") and final_evidence_reason is None:
        degradation_reasons.append(L1ExecutionReason.ORCHESTRATION_ERROR)
    if final_evidence_reason is L1FinalEvidenceReason.CONTRACT_REPAIR:
        degradation_reasons.append(L1ExecutionReason.CONTRACT_REPAIR)
    failed_calls = [call for call in result.model_calls if not call.get("success")]
    if failed_calls:
        degradation_reasons.append(L1ExecutionReason.MODEL_CALL_FAILED)
    if any(call.get("retry_scheduled") for call in failed_calls):
        degradation_reasons.append(L1ExecutionReason.RETRY_USED)
    if any(
        call.get("timeout")
        and call.get("error_type")
        not in {
            "analysis_deadline_exceeded",
            "context_budget_exceeded",
            "context_window_exceeded",
        }
        for call in failed_calls
    ):
        degradation_reasons.append(L1ExecutionReason.PROVIDER_TIMEOUT)
    if any(call.get("error_type") == "context_budget_exceeded" for call in failed_calls):
        degradation_reasons.append(L1ExecutionReason.CONTEXT_BUDGET_EXCEEDED)
    if any(call.get("error_type") == "context_window_exceeded" for call in failed_calls):
        degradation_reasons.append(L1ExecutionReason.CONTEXT_WINDOW_EXCEEDED)
    if any(
        call.get("http_status") and call.get("error_type") != "context_window_exceeded"
        for call in failed_calls
    ):
        degradation_reasons.append(L1ExecutionReason.PROVIDER_HTTP_ERROR)
    if result.unsupported_tool_requests:
        degradation_reasons.append(L1ExecutionReason.UNSUPPORTED_TOOL_REQUEST)
    if result.anomalies.get("tool_round_exhausted"):
        degradation_reasons.append(L1ExecutionReason.TOOL_ROUND_EXHAUSTED)
    if result.anomalies.get("prior_output_truncated"):
        degradation_reasons.append(L1ExecutionReason.OUTPUT_TRUNCATED)

    fatal_reason = _fatal_execution_reason(result)
    contract_errors = (
        model_evidence_contract_errors(result.semantic_payload)
        if result.semantic_payload is not None
        else []
    )
    if (
        fatal_reason is None
        and result.success
        and result.semantic_payload is not None
        and not contract_errors
    ):
        reasons = _unique_reasons(degradation_reasons)
        return L1ExecutionAssessment(
            execution_status=L1ExecutionStatus.COMPLETED,
            result_quality=(L1ResultQuality.DEGRADED if reasons else L1ResultQuality.USABLE),
            parse_status=L1ParseStatus.VALID,
            evidence_present=True,
            final_evidence_reason=final_evidence_reason,
            reason_codes=reasons,
            errors=tuple(result.errors),
        )

    if fatal_reason is not None:
        reason = fatal_reason
        parse_status = (
            L1ParseStatus.MALFORMED
            if reason is L1ExecutionReason.OUTPUT_TRUNCATED
            else L1ParseStatus.NOT_AVAILABLE
        )
        errors = tuple(result.errors) or _default_errors(reason)
    elif result.anomalies.get("contract_invalid_model_evidence") or contract_errors:
        reason = L1ExecutionReason.CONTRACT_INVALID
        parse_status = L1ParseStatus.CONTRACT_INVALID
        errors = tuple(result.errors) or tuple(contract_errors)
    else:
        reason = _primary_unusable_reason(result)
        parse_status = (
            L1ParseStatus.MALFORMED
            if reason in (L1ExecutionReason.MALFORMED_OUTPUT, L1ExecutionReason.OUTPUT_TRUNCATED)
            else L1ParseStatus.NOT_AVAILABLE
        )
        errors = tuple(result.errors) or _default_errors(reason)

    return L1ExecutionAssessment(
        execution_status=L1ExecutionStatus.FAILED,
        result_quality=L1ResultQuality.UNUSABLE,
        parse_status=parse_status,
        evidence_present=result.semantic_payload is not None,
        final_evidence_reason=final_evidence_reason,
        reason_codes=_unique_reasons([reason, *degradation_reasons]),
        errors=errors,
    )


def output_health(l1_result: L1EvidenceResult) -> dict[str, Any]:
    """Return the compatibility health view from the canonical assessment."""

    configured = bool(l1_result.model or l1_result.anomalies.get("l1_enabled"))
    assessment = assess_execution(configured=configured, result=l1_result)
    status_by_reason = {
        L1ExecutionReason.PROVIDER_TIMEOUT: "provider_timeout",
        L1ExecutionReason.OUTPUT_TRUNCATED: "truncated",
        L1ExecutionReason.CONTRACT_INVALID: "contract_invalid",
        L1ExecutionReason.MALFORMED_OUTPUT: "malformed",
    }
    status = "usable" if assessment.usable else assessment.execution_status.value
    if assessment.result_quality is L1ResultQuality.NOT_APPLICABLE:
        status = "not_run"
    elif assessment.unusable_reason is not None:
        status = status_by_reason.get(assessment.unusable_reason, "provider_error")
    return {
        "status": status,
        "usable": assessment.usable,
        "errors": list(assessment.errors),
        "execution_assessment": assessment.to_payload(),
    }


def l1_contract_advisories(l1_result: L1EvidenceResult) -> list[dict[str, Any]]:
    """Return canonical advisories for built-in or injected L1 extractors."""

    if l1_result.semantic_payload is None:
        return []
    advisories = [
        dict(item)
        for item in l1_result.anomalies.get("contract_advisories", ())
        if isinstance(item, Mapping)
    ]
    advisories.extend(model_evidence_contract_advisories(l1_result.semantic_payload))
    return _unique_advisories(advisories)


def _normalize_result(result: L1EvidenceResult) -> L1EvidenceResult:
    if result.semantic_payload is None or not result.success:
        return result
    if model_evidence_contract_errors(result.semantic_payload):
        return result
    normalized_payload, _ = normalize_model_evidence_payload(result.semantic_payload)
    advisories = model_evidence_contract_advisories(result.semantic_payload)
    existing = [
        dict(item)
        for item in result.anomalies.get("contract_advisories", ())
        if isinstance(item, Mapping)
    ]
    anomalies = dict(result.anomalies)
    anomalies["contract_advisories"] = _unique_advisories([*existing, *advisories])
    return replace(result, semantic_payload=normalized_payload, anomalies=anomalies)


def _unique_advisories(advisories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for advisory in advisories:
        key = repr(sorted(advisory.items()))
        if key in seen:
            continue
        seen.add(key)
        result.append(advisory)
    return result


def _final_evidence_reason(result: L1EvidenceResult) -> L1FinalEvidenceReason | None:
    value = result.anomalies.get("final_evidence_reason")
    if not value:
        return None
    try:
        return L1FinalEvidenceReason(str(value))
    except ValueError:
        return None


def pending_output_health() -> dict[str, Any]:
    """Represent a configured L1 route before its result is available."""

    assessment = L1ExecutionAssessment(
        execution_status=L1ExecutionStatus.IN_FLIGHT,
        result_quality=L1ResultQuality.NOT_APPLICABLE,
        parse_status=L1ParseStatus.NOT_RUN,
        evidence_present=False,
    )
    return {
        "status": "pending",
        "usable": False,
        "errors": [],
        "execution_assessment": assessment.to_payload(),
    }


def _primary_unusable_reason(result: L1EvidenceResult) -> L1ExecutionReason:
    fatal_reason = _fatal_execution_reason(result)
    if fatal_reason is not None:
        return fatal_reason
    anomalies = result.anomalies
    if anomalies.get("contract_invalid_model_evidence"):
        return L1ExecutionReason.CONTRACT_INVALID
    if result.malformed or anomalies.get("malformed_model_evidence"):
        return L1ExecutionReason.MALFORMED_OUTPUT
    if anomalies.get("tool_round_exhausted"):
        return L1ExecutionReason.TOOL_ROUND_EXHAUSTED
    if anomalies.get("provider_error"):
        return L1ExecutionReason.PROVIDER_ERROR
    return L1ExecutionReason.NO_VALID_EVIDENCE


def _fatal_execution_reason(result: L1EvidenceResult) -> L1ExecutionReason | None:
    anomalies = result.anomalies
    if anomalies.get("deadline_exceeded"):
        return L1ExecutionReason.ANALYSIS_DEADLINE_EXCEEDED
    if anomalies.get("context_budget_exceeded"):
        return L1ExecutionReason.CONTEXT_BUDGET_EXCEEDED
    if anomalies.get("context_window_exceeded"):
        return L1ExecutionReason.CONTEXT_WINDOW_EXCEEDED
    if anomalies.get("provider_timeout"):
        return L1ExecutionReason.PROVIDER_TIMEOUT
    if anomalies.get("model_output_truncated"):
        return L1ExecutionReason.OUTPUT_TRUNCATED
    return None


def _default_errors(reason: L1ExecutionReason) -> tuple[str, ...]:
    defaults = {
        L1ExecutionReason.PROVIDER_TIMEOUT: "LLM provider timed out",
        L1ExecutionReason.OUTPUT_TRUNCATED: "model output was truncated",
        L1ExecutionReason.TOOL_ROUND_EXHAUSTED: "tool-round limit exhausted without valid evidence",
        L1ExecutionReason.NO_VALID_EVIDENCE: "L1 produced no valid evidence",
    }
    message = defaults.get(reason)
    return (message,) if message else ()


def _unique_reasons(
    reasons: list[L1ExecutionReason],
) -> tuple[L1ExecutionReason, ...]:
    return tuple(dict.fromkeys(reasons))
