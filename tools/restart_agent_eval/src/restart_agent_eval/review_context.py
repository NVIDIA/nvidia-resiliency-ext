"""Normalized product data consumed by one-model review generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifact_io import LOCAL_ARTIFACT_STORE, ArtifactStore
from .product_trace import ProductTrace, decision_candidate_result


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _l1_assessment(
    analysis: Mapping[str, Any],
    l1_trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Read the stage-owned L1 block, with trace fallback for older runs."""

    assessment = _mapping(analysis.get("l1_assessment"))
    if assessment:
        return assessment
    assessment = _mapping(l1_trace.get("semantic_payload"))
    if assessment:
        return assessment
    legacy_root = _mapping(analysis.get("root_cause_assessment"))
    legacy_recovery = _mapping(analysis.get("model_recovery_assessment"))
    if not legacy_root and not legacy_recovery:
        return {}
    return {
        "root_cause_assessment": legacy_root,
        "model_recovery_assessment": legacy_recovery,
    }


@dataclass(frozen=True)
class ReviewContext:
    """Validated, normalized stage payloads for a completed product route."""

    result: dict[str, Any]
    trace: ProductTrace
    analysis: dict[str, Any]
    analyzer_trace: dict[str, Any]
    collect_all_context: dict[str, Any]
    route_execution_status: str | None
    route_l1_execution_assessment: dict[str, Any]
    l0_bundle: dict[str, Any]
    l0_model_view: dict[str, Any]
    decision_evidence: dict[str, Any]
    l0_primary: dict[str, Any]
    l0_selected_observation: dict[str, Any]
    l1: dict[str, Any]
    l1_model_output: dict[str, Any]
    l1_layer: dict[str, Any]
    l1_primary: dict[str, Any]
    l1_observed_failures: list[dict[str, Any]]
    l1_selected_observation: dict[str, Any]
    l2_grounding: dict[str, Any]
    l2_audit: dict[str, Any]
    l2_primary: dict[str, Any]
    l2_observed_failures: list[dict[str, Any]]
    l2_selected_observation: dict[str, Any]
    failure_tracks: dict[str, Any]
    cycle_history: dict[str, Any]
    l4_path_selection: dict[str, Any]
    current_failure_facts: dict[str, Any]
    timing: dict[str, Any]
    latency_measurement: dict[str, Any]
    token_usage: dict[str, Any]
    token_limit: dict[str, Any]
    primary: dict[str, Any]
    observed_failures: list[dict[str, Any]]
    selected_observation: dict[str, Any]
    provenance: dict[str, Any]
    deterministic_candidate: dict[str, Any]
    enriched_candidate: dict[str, Any]
    deterministic_analysis: dict[str, Any]
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
        route_l1_execution_assessment = _mapping(collect_all_context.get("l1_execution_assessment"))

        l0_bundle = _mapping(trace.l0_bundle)
        l0_model_view = _mapping(analyzer_trace.get("l0_model_view"))
        decision_evidence = _mapping(analyzer_trace.get("decision_evidence"))
        l0_primary = _mapping(
            decision_evidence.get("deterministic_primary_candidate")
            or l0_bundle.get("deterministic_primary_candidate")
        )
        l0_selected_observation = _mapping(
            decision_evidence.get("selected_observed_failure")
            or l0_bundle.get("selected_observed_failure")
        )

        l1 = _mapping(analyzer_trace.get("l1"))
        l1_model_output = _l1_assessment(analysis, l1)
        layers = _mapping(analyzer_trace.get("layers"))
        l1_layer = _mapping(layers.get("L1"))
        l1_primary = _mapping(l1_model_output.get("primary_failure"))
        l1_observed_failures = _mapping_list(l1_model_output.get("observed_failures"))
        l1_selected_observation = _selected_observation(
            l1_observed_failures,
            l1_model_output.get("selected_observed_failure_id"),
        )
        l2_grounding = _mapping(analysis.get("l2_grounding"))
        l2_audit = _mapping(analyzer_trace.get("l2_audit"))
        l2_primary = _mapping(l2_grounding.get("grounded_primary_failure")) or _mapping(
            _mapping(analyzer_trace.get("l2_grounding")).get("primary_failure")
        )
        l2_observed_failures = _mapping_list(l2_grounding.get("grounded_observed_failures"))
        l2_selected_observation = _mapping(l2_grounding.get("grounded_selected_observation"))
        failure_tracks = _failure_tracks(
            l0_primary=l0_primary,
            l0_observation=l0_selected_observation,
            l2_grounding=l2_grounding,
            l2_primary=l2_primary,
            l2_observation=l2_selected_observation,
        )
        cycle_history = _mapping(analyzer_trace.get("l3_history"))
        l4_path_selection = _mapping(
            _mapping(analyzer_trace.get("l4_policy")).get("path_selection")
        )
        if not l4_path_selection:
            result_provenance = _mapping(analysis.get("result_provenance"))
            selected_path = result_provenance.get("selected_evidence_path")
            if selected_path:
                l4_path_selection = {
                    "path": selected_path,
                    "route_id": result_provenance.get("selected_route_id"),
                    "reason": result_provenance.get("path_selection_reason"),
                }

        provenance = _mapping(analysis.get("result_provenance"))
        decision_candidates = _mapping(analyzer_trace.get("decision_candidates"))
        deterministic_candidate = _mapping(decision_candidates.get("deterministic"))
        enriched_candidate = _mapping(decision_candidates.get("l1_enriched"))
        deterministic_analysis = decision_candidate_result(deterministic_candidate)
        enriched_analysis = decision_candidate_result(enriched_candidate)
        candidate_kind = str(provenance.get("candidate_kind") or "")
        if not deterministic_analysis and candidate_kind == "deterministic":
            deterministic_analysis = analysis
        if not enriched_analysis and candidate_kind == "l1_enriched":
            enriched_analysis = analysis

        primary = _mapping(analysis.get("primary_failure"))
        observed_failures = _mapping_list(analysis.get("observed_failures"))
        selected_observation = _mapping(analysis.get("selected_observed_failure"))
        current_failure_facts = _current_failure_facts(
            _mapping(
                analyzer_trace.get("selected_failure_facts")
                or analyzer_trace.get("current_failure_facts")
            ),
            primary=primary,
            selected_observation=selected_observation,
            l3_layer=_mapping(layers.get("L3")),
            provenance=provenance,
        )

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
            route_l1_execution_assessment=route_l1_execution_assessment,
            l0_bundle=l0_bundle,
            l0_model_view=l0_model_view,
            decision_evidence=decision_evidence,
            l0_primary=l0_primary,
            l0_selected_observation=l0_selected_observation,
            l1=l1,
            l1_model_output=l1_model_output,
            l1_layer=l1_layer,
            l1_primary=l1_primary,
            l1_observed_failures=l1_observed_failures,
            l1_selected_observation=l1_selected_observation,
            l2_grounding=l2_grounding,
            l2_audit=l2_audit,
            l2_primary=l2_primary,
            l2_observed_failures=l2_observed_failures,
            l2_selected_observation=l2_selected_observation,
            failure_tracks=failure_tracks,
            cycle_history=cycle_history,
            l4_path_selection=l4_path_selection,
            current_failure_facts=current_failure_facts,
            timing=_mapping(analyzer_trace.get("timing")),
            latency_measurement=_mapping(analyzer_trace.get("latency_measurement")),
            token_usage=_mapping(analyzer_trace.get("token_usage")),
            token_limit=_mapping(analyzer_trace.get("token_limit")),
            primary=primary,
            observed_failures=observed_failures,
            selected_observation=selected_observation,
            provenance=provenance,
            deterministic_candidate=deterministic_candidate,
            enriched_candidate=enriched_candidate,
            deterministic_analysis=deterministic_analysis,
            enriched_analysis=enriched_analysis,
            model_calls=model_calls,
            tool_calls=tool_calls,
            interaction_transcript=interaction_transcript,
        )


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _selected_observation(
    observations: list[dict[str, Any]],
    selected_id: Any,
) -> dict[str, Any]:
    if not isinstance(selected_id, str):
        return {}
    return next(
        (dict(item) for item in observations if item.get("id") == selected_id),
        {},
    )


def _failure_tracks(
    *,
    l0_primary: Mapping[str, Any],
    l0_observation: Mapping[str, Any],
    l2_grounding: Mapping[str, Any],
    l2_primary: Mapping[str, Any],
    l2_observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Project stage-owned failure tracks without deriving one from another."""

    enriched = _mapping(l2_grounding.get("enriched_failure_tracks"))
    grounding = _mapping(l2_grounding.get("track_grounding"))
    deterministic_kind = "root" if l0_primary else "observation_only" if l0_observation else "none"
    return {
        "deterministic": {
            "status": "available" if deterministic_kind != "none" else "not_available",
            "identity_kind": deterministic_kind,
            "failure": dict(l0_primary or l0_observation) if deterministic_kind != "none" else None,
        },
        "primary": {
            "status": _track_status(grounding.get("primary"), l2_primary),
            "failure": dict(l2_primary) if l2_primary else None,
            "facts": _mapping(enriched.get("primary")) or None,
        },
        "observation": {
            "status": _track_status(grounding.get("observation"), l2_observation),
            "failure": dict(l2_observation) if l2_observation else None,
            "facts": _mapping(enriched.get("observation")) or None,
        },
    }


def _track_status(grounding: Any, failure: Mapping[str, Any]) -> str:
    detail = _mapping(grounding)
    if failure:
        return "grounded"
    return str(detail.get("status") or detail.get("grounding_status") or "not_available")


def _current_failure_facts(
    traced: dict[str, Any],
    *,
    primary: dict[str, Any],
    selected_observation: dict[str, Any],
    l3_layer: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Normalize current identity when the product omits its internal record."""

    if traced:
        return traced
    source = l3_layer.get("selected_failure_facts_source") or provenance.get("evidence_source")
    root_fingerprint = primary.get("root_fingerprint")
    observation_fingerprint = selected_observation.get("observation_fingerprint")
    if root_fingerprint:
        identity_kind = "root"
    elif observation_fingerprint:
        identity_kind = "observation_only"
    else:
        identity_kind = "none"
    return {
        "source": source,
        "identity_kind": identity_kind,
        "history_identity_ready": bool(root_fingerprint or observation_fingerprint),
        "root_fingerprint": root_fingerprint,
        "root_fingerprint_source": primary.get("root_fingerprint_source"),
        "observation_fingerprint": observation_fingerprint,
        "observation_fingerprint_source": selected_observation.get(
            "observation_fingerprint_source"
        ),
        "selected_observation_line": selected_observation.get("line"),
        "selected_observation_causal_role": selected_observation.get("causal_role"),
        "affected_entity": primary.get("affected_entity"),
        "fault_outcome": primary.get("fault_outcome") or selected_observation.get("fault_outcome"),
    }
