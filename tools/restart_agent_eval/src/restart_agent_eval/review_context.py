"""Normalized product data consumed by one-model review generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifact_io import LOCAL_ARTIFACT_STORE, ArtifactStore
from .product_trace import ProductTrace, decision_candidate_result


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class ReviewContext:
    """Validated, normalized stage payloads for a completed product route."""

    result: dict[str, Any]
    trace: ProductTrace
    analysis: dict[str, Any]
    analyzer_trace: dict[str, Any]
    collect_all_context: dict[str, Any]
    route_execution_status: str | None
    l0_bundle: dict[str, Any]
    l0_model_view: dict[str, Any]
    decision_evidence: dict[str, Any]
    l0_primary: dict[str, Any]
    l1: dict[str, Any]
    l1_model_output: dict[str, Any]
    l1_layer: dict[str, Any]
    l1_primary: dict[str, Any]
    l2_audit: dict[str, Any]
    l2_primary: dict[str, Any]
    current_failure_facts: dict[str, Any]
    timing: dict[str, Any]
    latency_measurement: dict[str, Any]
    token_usage: dict[str, Any]
    token_limit: dict[str, Any]
    primary: dict[str, Any]
    provenance: dict[str, Any]
    fallback_candidate: dict[str, Any]
    enriched_candidate: dict[str, Any]
    fallback_analysis: dict[str, Any]
    enriched_analysis: dict[str, Any]
    model_calls: list[Any]
    tool_calls: list[Any]
    interaction_transcript: list[Any]

    @classmethod
    def read(
        cls,
        paths: Mapping[str, Path],
        *,
        artifact_store: ArtifactStore = LOCAL_ARTIFACT_STORE,
    ) -> "ReviewContext":
        return cls.from_payloads(
            artifact_store.read_json(paths["result_json"]),
            artifact_store.read_json(paths["trace_json"]),
        )

    @classmethod
    def from_payloads(
        cls,
        result_payload: Any,
        trace_payload: Any,
    ) -> "ReviewContext":
        """Normalize already-loaded route artifacts without filesystem access."""

        result = _mapping(result_payload)
        trace = ProductTrace.from_payload(trace_payload)
        analysis = _mapping(trace.analysis_result) or result
        analyzer_trace = _mapping(trace.analyzer_trace)
        collect_all_context = _mapping(trace.collect_all_context)
        route_execution_status = str(collect_all_context.get("execution_status") or "") or None

        l0_bundle = _mapping(trace.l0_bundle)
        l0_model_view = _mapping(analyzer_trace.get("l0_model_view"))
        decision_evidence = _mapping(analyzer_trace.get("decision_evidence"))
        l0_primary = _mapping(
            decision_evidence.get("deterministic_primary_candidate")
            or l0_bundle.get("deterministic_primary_candidate")
        )

        l1 = _mapping(analyzer_trace.get("l1"))
        l1_model_output = _mapping(l1.get("parsed_evidence"))
        layers = _mapping(analyzer_trace.get("layers"))
        l1_layer = _mapping(layers.get("L1"))
        l1_primary = _mapping(_mapping(l1.get("parsed_evidence")).get("primary_failure"))
        l2_audit = _mapping(analyzer_trace.get("l2_audit"))
        l2_primary = _mapping(
            _mapping(analyzer_trace.get("l2_grounded_semantics")).get("primary_failure")
        )

        provenance = _mapping(analysis.get("result_provenance"))
        decision_candidates = _mapping(analyzer_trace.get("decision_candidates"))
        fallback_candidate = _mapping(decision_candidates.get("deterministic_fallback"))
        enriched_candidate = _mapping(decision_candidates.get("l1_enriched"))
        fallback_analysis = decision_candidate_result(fallback_candidate)
        enriched_analysis = decision_candidate_result(enriched_candidate)
        candidate_kind = str(provenance.get("candidate_kind") or "")
        if not fallback_analysis and candidate_kind == "deterministic_fallback":
            fallback_analysis = analysis
        if not enriched_analysis and candidate_kind == "l1_enriched":
            enriched_analysis = analysis

        model_calls = list(l1.get("model_calls") or [])
        tool_calls = list(l1.get("tool_calls") or [])
        interaction_transcript = list(
            l1.get("interaction_transcript") or l1.get("transcript_events") or []
        )
        return cls(
            result=result,
            trace=trace,
            analysis=analysis,
            analyzer_trace=analyzer_trace,
            collect_all_context=collect_all_context,
            route_execution_status=route_execution_status,
            l0_bundle=l0_bundle,
            l0_model_view=l0_model_view,
            decision_evidence=decision_evidence,
            l0_primary=l0_primary,
            l1=l1,
            l1_model_output=l1_model_output,
            l1_layer=l1_layer,
            l1_primary=l1_primary,
            l2_audit=l2_audit,
            l2_primary=l2_primary,
            current_failure_facts=_mapping(analyzer_trace.get("current_failure_facts")),
            timing=_mapping(analyzer_trace.get("timing")),
            latency_measurement=_mapping(analyzer_trace.get("latency_measurement")),
            token_usage=_mapping(analyzer_trace.get("token_usage")),
            token_limit=_mapping(analyzer_trace.get("token_limit")),
            primary=_mapping(analysis.get("primary_failure")),
            provenance=provenance,
            fallback_candidate=fallback_candidate,
            enriched_candidate=enriched_candidate,
            fallback_analysis=fallback_analysis,
            enriched_analysis=enriched_analysis,
            model_calls=model_calls,
            tool_calls=tool_calls,
            interaction_transcript=interaction_transcript,
        )
