# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L1 invocation boundary, output health, and deadline degradation."""

from __future__ import annotations

from typing import Any, Mapping

from ..infrastructure.log_source import LogSnapshot
from ..models import L0Bundle, L0ModelFacingView
from ..runtime import SYSTEM_CLOCK, Clock
from .contracts import EvidenceExtractor, L1EvidenceResult
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
            evidence=None,
            model=type(extractor).__name__,
            success=False,
            errors=(f"{type(exc).__name__}: {exc}",),
            anomalies={
                "l1_enabled": True,
                "provider_error": True,
                "provider_error_type": "extractor_exception",
            },
        )
    completed = clock.monotonic()
    return result, round(completed - started, 3), completed


def deadline_exceeded_result(*, model: str) -> L1EvidenceResult:
    """Represent an L1 route that did not complete inside the analysis budget."""

    error = "analysis deadline exceeded before L1 completed"
    return L1EvidenceResult(
        evidence=None,
        model=model,
        success=False,
        errors=(error,),
        anomalies={
            "l1_enabled": True,
            "provider_error": True,
            "provider_error_type": "analysis_deadline_exceeded",
            "provider_timeout": True,
            "deadline_exceeded": True,
        },
    )


def output_health(l1_result: L1EvidenceResult) -> dict[str, Any]:
    """Classify whether an L1 response is usable by L2."""

    if not l1_result.model and not l1_result.anomalies.get("l1_enabled"):
        return {"status": "not_run", "usable": False, "errors": []}
    if l1_result.anomalies.get("provider_timeout"):
        return {
            "status": "provider_timeout",
            "usable": False,
            "errors": list(l1_result.errors) or ["LLM provider timed out"],
        }
    if l1_result.anomalies.get("model_output_truncated"):
        return {
            "status": "truncated",
            "usable": False,
            "errors": list(l1_result.errors) or ["model output was truncated"],
        }
    if not l1_result.success or l1_result.evidence is None:
        status = "provider_error" if l1_result.anomalies.get("provider_error") else "malformed"
        return {"status": status, "usable": False, "errors": list(l1_result.errors)}
    contract_errors = model_evidence_contract_errors(l1_result.evidence)
    if contract_errors:
        return {
            "status": "contract_invalid",
            "usable": False,
            "errors": contract_errors,
        }
    return {"status": "usable", "usable": True, "errors": []}


def execution_status(
    *,
    configured: bool,
    result: L1EvidenceResult,
    health: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize model/endpoint execution independently of semantic quality."""

    if not configured:
        return {"status": "not_run", "issues": []}

    issues: list[str] = []
    failed_calls = [call for call in result.model_calls if not call.get("success")]
    if failed_calls:
        issues.append("model_call_failed")
    if any(call.get("retry_scheduled") for call in failed_calls):
        issues.append("retry_used")
    if any(call.get("timeout") for call in failed_calls):
        issues.append("provider_timeout")
    if any(call.get("error_type") == "context_window_exceeded" for call in failed_calls):
        issues.append("client_context_budget_exceeded")
    if any(
        call.get("http_status") and call.get("error_type") != "context_window_exceeded"
        for call in failed_calls
    ):
        issues.append("provider_http_error")
    if result.unsupported_tool_requests:
        issues.append("unsupported_tool_request")

    if not health.get("usable"):
        issues.append(str(health.get("status") or "unusable"))
        return {"status": "failed", "issues": list(dict.fromkeys(issues))}
    if issues:
        return {"status": "degraded", "issues": list(dict.fromkeys(issues))}
    return {"status": "ok", "issues": []}
