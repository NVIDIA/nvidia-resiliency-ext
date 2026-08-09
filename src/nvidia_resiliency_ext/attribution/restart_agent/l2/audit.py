# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L2 source grounding, history identity, and credibility diagnostics."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..current_failure_facts import build_attempt_failure_facts
from ..identity import (
    canonical_observed_fingerprint,
    extract_failure_iteration,
    extract_gpu,
    extract_node,
    extract_rank,
)
from ..infrastructure.log_source import LogSnapshot
from ..l0.decision import canonical_identity_anchor_line, distributed_incident_for_line
from ..l1.contracts import L1EvidenceResult
from ..l1.response_contract import (
    AFFECTED_ARTIFACT_PATH_FIELD,
    DIRECT_FAILURE_OBJECT_PATH_FIELD,
    FAILURE_DOMAIN_SUPPORT_TAG,
    RETRY_OUTLOOK_SUPPORT_TAG,
)
from ..models import (
    AffectedEntity,
    AssessmentStatus,
    AttemptFailureFacts,
    AttemptFailureFactsSource,
    CausalRole,
    FailureEvidence,
    FaultOutcome,
    L0Bundle,
    L0ModelFacingView,
)
from .failure_facts import select_grounded_affected_entity
from .grounding import (
    model_visible_line_texts,
    model_visible_value_line_numbers,
    text_contains_exact_value,
)

NEARBY_EVIDENCE_LINE_RADIUS = 5
MIN_ABBREVIATED_FRAGMENT_ALNUM_CHARS = 8
_ELLIPSIS_RE = re.compile(r"(?:\.\.\.\[truncated\]|\.\.\.|…)")


@dataclass(frozen=True)
class L2GroundingInput:
    """Complete typed input required to ground and audit one L1 result."""

    bundle: L0Bundle
    model_view: L0ModelFacingView
    l1_result: L1EvidenceResult
    source_log: LogSnapshot


@dataclass(frozen=True)
class L2Result:
    """Independently grounded L1 primary and selected-observation tracks."""

    primary: FailureEvidence | None
    selected_observed_failure: FailureEvidence | None
    primary_failure_facts: AttemptFailureFacts | None
    observation_failure_facts: AttemptFailureFacts | None
    grounding_status: str
    audit_status: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def used(self) -> bool:
        return self.primary_failure_facts is not None or self.observation_failure_facts is not None

    def to_payload(self) -> dict[str, Any]:
        payload = dict(self.diagnostics)
        observation_grounding = payload.get("observation_grounding")
        observation_status = (
            str(observation_grounding.get("grounding_status") or "unavailable")
            if isinstance(observation_grounding, Mapping)
            else (
                self.grounding_status
                if self.observation_failure_facts is not None
                else "unavailable"
            )
        )
        primary_status = str(payload.get("primary_grounding_status") or self.grounding_status)
        payload.update(
            {
                "used": self.used,
                "grounding_status": self.grounding_status,
                "audit_status": self.audit_status,
                "primary_used": self.primary is not None and self.used,
                "selected_observation_used": (
                    self.selected_observed_failure is not None and self.used
                ),
                "grounded_selected_observation": (
                    self.selected_observed_failure.to_failure_payload()
                    if self.selected_observed_failure is not None
                    else None
                ),
                "track_grounding": {
                    "primary": {
                        "status": (
                            primary_status
                            if self.primary_failure_facts is not None
                            else "unavailable"
                        ),
                        "published": self.primary_failure_facts is not None,
                    },
                    "observation": {
                        "status": observation_status,
                        "published": self.observation_failure_facts is not None,
                    },
                },
                "enriched_failure_tracks": {
                    "primary": (
                        self.primary_failure_facts.to_payload()
                        if self.primary_failure_facts is not None
                        else None
                    ),
                    "observation": (
                        self.observation_failure_facts.to_payload()
                        if self.observation_failure_facts is not None
                        else None
                    ),
                },
                "audit_influence": "observational_only",
            }
        )
        return payload

    @classmethod
    def not_run(cls, reason: str) -> "L2Result":
        return cls(
            primary=None,
            selected_observed_failure=None,
            primary_failure_facts=None,
            observation_failure_facts=None,
            grounding_status="not_run",
            audit_status="not_run",
            diagnostics={
                "audit_influence": "observational_only",
                "not_run_reason": reason,
                "field_findings": {},
                "field_finding_codes": {},
                "findings": [],
                "citation_audits": [],
                "grounding_adjustments": [],
                "recovery_field_audits": [],
            },
        )


@dataclass(frozen=True)
class PrimaryGrounding:
    """Source-grounded primary selection and its citation resolution."""

    model_primary: Mapping[str, Any]
    line: int
    log_line: str
    source_line_available: bool
    model_visible_support: bool
    primary_grounded: bool
    grounding_method: str
    grounded_evidence: tuple[Mapping[str, Any], ...]
    resolved_lines: Mapping[int, int]
    cited_lines: frozenset[int]
    text_visible_lines: frozenset[int]
    failure_identity_grounding: Mapping[str, Any]


@dataclass(frozen=True)
class HistoryIdentity:
    """Source-grounded identity used for current-failure history comparison."""

    anchor_line: int
    anchor_reason: str
    log_line: str
    l0_match: FailureEvidence | None
    root_fingerprint: str | None
    root_fingerprint_source: str
    affected_entity: AffectedEntity | None


@dataclass(frozen=True)
class CitationGrounding:
    """Pure resolution result for one model citation."""

    original_line: int
    resolved_line: int | None
    quote: str | None
    supports: tuple[str, ...]
    status: str
    candidate_lines: tuple[int, ...] = ()


def result_from_payload(
    primary: FailureEvidence | None,
    selected_observed_failure: FailureEvidence | None,
    payload: Mapping[str, Any],
    model_view: L0ModelFacingView,
) -> L2Result:
    details = dict(payload)
    details.pop("used", None)
    grounding_status = str(details.pop("grounding_status", "unavailable"))
    audit_status = str(details.pop("audit_status", "findings"))
    details["primary_grounding_status"] = grounding_status if primary is not None else "unavailable"
    details.pop("primary_used", None)
    identity_lineage = details.get("identity_lineage")
    same_l0_incident = bool(
        isinstance(identity_lineage, Mapping)
        and identity_lineage.get("relationship_to_l0") == "same_canonical_incident"
    )
    primary_failure_facts = (
        build_attempt_failure_facts(
            primary,
            model_view.decision_evidence,
            source=AttemptFailureFactsSource.L2_GROUNDED,
            identity_anchor_line=_optional_int(details.get("stable_identity_anchor_line")),
            identity_anchor_reason=_optional_str(details.get("stable_identity_anchor_reason")),
            root_locality=(model_view.decision_evidence.locality if same_l0_incident else {}),
            selected_observation=None,
        )
        if primary is not None
        else None
    )
    observation_failure_facts = (
        build_attempt_failure_facts(
            None,
            model_view.decision_evidence,
            source=AttemptFailureFactsSource.L2_GROUNDED,
            identity_anchor_line=selected_observed_failure.line,
            identity_anchor_reason="model_selected_observation",
            selected_observation=selected_observed_failure,
        )
        if selected_observed_failure is not None
        else None
    )
    observation_grounding = details.get("observation_grounding")
    observation_status = (
        str(observation_grounding.get("grounding_status") or "unavailable")
        if isinstance(observation_grounding, Mapping)
        else (grounding_status if primary is None else "unavailable")
    )
    aggregate_grounding_status = (
        "grounded"
        if primary_failure_facts is not None or observation_failure_facts is not None
        else grounding_status
    )
    if (
        isinstance(observation_grounding, Mapping)
        and observation_grounding.get("audit_status") == "findings"
    ):
        audit_status = "findings"
    return L2Result(
        primary=primary,
        selected_observed_failure=selected_observed_failure,
        primary_failure_facts=primary_failure_facts,
        observation_failure_facts=observation_failure_facts,
        grounding_status=aggregate_grounding_status,
        audit_status=audit_status,
        diagnostics=details,
    )


def ground_and_audit_model_evidence(
    grounding_input: L2GroundingInput,
) -> L2Result:
    primary, selected_observed_failure, payload = _audit_model_evidence_payload(
        grounding_input.bundle,
        grounding_input.model_view,
        grounding_input.l1_result,
        grounding_input.source_log,
    )
    return result_from_payload(
        primary,
        selected_observed_failure,
        payload,
        grounding_input.model_view,
    )


def _new_audit(l1_result: L1EvidenceResult) -> dict[str, Any]:
    return {
        "used": False,
        "audit_status": "findings",
        "audit_influence": "observational_only",
        "primary_used": False,
        "recovery_assessment_audited": False,
        "field_findings": {},
        "field_finding_codes": {},
        "findings": [],
        "citation_audits": [],
        "grounding_adjustments": [],
        "recovery_field_audits": [],
        "model": l1_result.model or None,
    }


def _normalized_l1_evidence(
    l1_result: L1EvidenceResult,
    audit: dict[str, Any],
) -> dict[str, Any]:
    if not l1_result.success or l1_result.semantic_payload is None:
        raise ValueError("L2 grounding requires a structurally usable L1 response")

    return dict(l1_result.semantic_payload)


def _ground_primary_selection(
    *,
    bundle: L0Bundle,
    model_view: L0ModelFacingView,
    l1_result: L1EvidenceResult,
    source_log: LogSnapshot,
    evidence: Mapping[str, Any],
    audit: dict[str, Any],
) -> PrimaryGrounding | None:
    primary = evidence.get("primary_failure")
    if primary is None:
        return None
    assert isinstance(primary, Mapping)

    causal_role = str(primary.get("causal_role") or "")
    primary_role_eligible = causal_role in {
        CausalRole.INITIATING.value,
        CausalRole.UNKNOWN.value,
    }
    if not primary_role_eligible:
        _record_field_finding(
            audit,
            "primary_failure",
            "primary_failure causal_role cannot represent an initiating primary",
            code="primary_causal_role_ineligible",
        )

    line = _optional_int(primary.get("line"))
    assert line is not None
    log_line = _line_text(source_log, line)
    source_line_available = log_line is not None
    if log_line is None:
        _record_field_finding(
            audit,
            "primary_failure",
            f"primary_failure line {line} is outside the source log",
            code="primary_line_outside_log",
        )
        log_line = "unverified model-selected failure"

    visible_texts = model_visible_line_texts(model_view, l1_result)
    failure_identity = primary.get("failure_identity")
    failure_identity_grounding = _ground_failure_identity_paths(
        failure_identity if isinstance(failure_identity, Mapping) else {},
        model_view=model_view,
        l1_result=l1_result,
        source_log=source_log,
    )
    audit["failure_identity_grounding"] = failure_identity_grounding
    grounded_evidence, resolved_lines = _audited_model_evidence(
        bundle,
        evidence,
        audit,
        source_log=source_log,
        model_visible_texts=visible_texts,
    )
    resolved_primary_line = resolved_lines.get(line, line)
    grounding_method = "unavailable"
    if resolved_primary_line != line:
        audit.setdefault("grounding_adjustments", []).append(
            {
                "field": "primary_failure.line",
                "from": line,
                "to": resolved_primary_line,
                "reason": "nearby_unique_quote_match",
            }
        )
        line = resolved_primary_line
        log_line = _line_text(source_log, line)
        if log_line is None:
            raise AssertionError("resolved evidence line must exist in source log")
        source_line_available = True
    model_visible_support = bool(visible_texts.get(line))
    primary_grounded = source_line_available and model_visible_support and primary_role_eligible
    if primary_grounded:
        grounding_method = (
            "nearby_unique_quote_match"
            if resolved_primary_line != int(primary["line"])
            else "exact_source_line"
        )
    elif source_line_available and not model_visible_support:
        _record_field_finding(
            audit,
            "primary_failure",
            "primary_failure line exists in the source log but its text was not visible "
            "to the model",
            code="primary_evidence_not_model_visible",
        )

    audit["source_line_available"] = source_line_available
    audit["primary_model_visible_support"] = model_visible_support
    audit["primary_causal_role_eligible"] = primary_role_eligible
    audit["grounding_status"] = "grounded" if primary_grounded else "unavailable"
    audit["grounding_method"] = grounding_method
    return PrimaryGrounding(
        model_primary=primary,
        line=line,
        log_line=log_line,
        source_line_available=source_line_available,
        model_visible_support=model_visible_support,
        primary_grounded=primary_grounded,
        grounding_method=grounding_method,
        grounded_evidence=grounded_evidence,
        resolved_lines=resolved_lines,
        cited_lines=frozenset(int(item["line"]) for item in grounded_evidence),
        text_visible_lines=frozenset(visible_texts),
        failure_identity_grounding=failure_identity_grounding,
    )


def _audit_recovery_assessment(
    *,
    bundle: L0Bundle,
    grounding: PrimaryGrounding,
    root_cause_assessment: Mapping[str, Any],
    recovery_assessment: Mapping[str, Any],
    audit: dict[str, Any],
) -> None:
    failure_domain = recovery_assessment.get("failure_domain")
    retry_outlook = recovery_assessment.get("retry_outlook_without_workload_change")
    assert isinstance(failure_domain, Mapping)
    assert isinstance(retry_outlook, Mapping)
    failure_domain_value = str(failure_domain.get("value") or "")
    failure_domain_status = str(failure_domain.get("status") or "")
    retry_outlook_value = str(retry_outlook.get("value") or "")
    retry_outlook_status = str(retry_outlook.get("status") or "")
    root_cause_status = str(root_cause_assessment.get("status") or "")
    audit.update(
        {
            "model_failure_domain": failure_domain_value,
            "model_failure_domain_status": failure_domain_status,
            "model_failure_domain_confidence": int(failure_domain.get("confidence") or 0),
            "model_retry_outlook_without_workload_change": retry_outlook_value,
            "model_retry_outlook_status": retry_outlook_status,
            "model_retry_outlook_confidence": int(retry_outlook.get("confidence") or 0),
            "model_root_cause_status": root_cause_status,
            "path_namespace_summary": dict(bundle.path_namespace_summary),
        }
    )
    _audit_unverified_path_identity_claims(
        bundle,
        root_cause_assessment,
        recovery_assessment,
        audit,
    )
    _ground_recovery_support(
        grounding=grounding,
        audit=audit,
        failure_domain_status=failure_domain_status,
        retry_outlook_status=retry_outlook_status,
    )
    _audit_progress_claim(
        bundle,
        root_cause_assessment,
        recovery_assessment,
        audit,
    )


def _ground_recovery_support(
    *,
    grounding: PrimaryGrounding,
    audit: dict[str, Any],
    failure_domain_status: str,
    retry_outlook_status: str,
) -> None:
    domain_lines = frozenset(
        int(item["line"])
        for item in grounding.grounded_evidence
        if FAILURE_DOMAIN_SUPPORT_TAG in item.get("supports", [])
    )
    retry_lines = frozenset(
        int(item["line"])
        for item in grounding.grounded_evidence
        if RETRY_OUTLOOK_SUPPORT_TAG in item.get("supports", [])
    )
    domain_unresolved = tuple(
        sorted(
            int(item["original_line"])
            for item in audit.get("citation_audits") or []
            if item.get("resolved_line") is None
            and FAILURE_DOMAIN_SUPPORT_TAG in set(item.get("supports") or [])
            and isinstance(item.get("original_line"), int)
        )
    )
    retry_unresolved = tuple(
        sorted(
            int(item["original_line"])
            for item in audit.get("citation_audits") or []
            if item.get("resolved_line") is None
            and RETRY_OUTLOOK_SUPPORT_TAG in set(item.get("supports") or [])
            and isinstance(item.get("original_line"), int)
        )
    )
    audit.update(
        {
            "failure_domain_support_expected": _support_audit_expected(failure_domain_status),
            "retry_outlook_support_expected": _support_audit_expected(retry_outlook_status),
            "failure_domain_supporting_lines": sorted(domain_lines),
            "retry_outlook_supporting_lines": sorted(retry_lines),
            "failure_domain_unresolved_supporting_lines": list(domain_unresolved),
            "retry_outlook_unresolved_supporting_lines": list(retry_unresolved),
        }
    )
    if _support_audit_expected(failure_domain_status) and not domain_lines:
        _record_field_finding(
            audit,
            "model_recovery_assessment",
            "failure_domain has no grounded evidence",
            code="failure_domain_support_missing",
        )
    if _support_audit_expected(retry_outlook_status) and not retry_lines:
        _record_field_finding(
            audit,
            "model_recovery_assessment",
            "retry_outlook has no grounded evidence",
            code="retry_outlook_support_missing",
        )
    unresolved = tuple(sorted(set(domain_unresolved).union(retry_unresolved)))
    if unresolved:
        _record_field_finding(
            audit,
            "model_recovery_assessment",
            "recovery evidence could not be grounded at lines: "
            + ", ".join(str(value) for value in unresolved),
            code="recovery_support_ungrounded",
        )


def _support_audit_expected(status: str) -> bool:
    """Return whether a substantive recovery claim is expected to cite support."""

    return status in {
        AssessmentStatus.ESTABLISHED_BY_CURRENT_LOG.value,
        AssessmentStatus.SUPPORTED_BUT_UNCONFIRMED.value,
    }


def _audit_progress_claim(
    bundle: L0Bundle,
    root_cause_assessment: Mapping[str, Any],
    recovery_assessment: Mapping[str, Any],
    audit: dict[str, Any],
) -> None:
    if not _observed_failure_position_overclaimed_as_completed_progress(
        bundle,
        root_cause_assessment,
        recovery_assessment,
    ):
        return
    _record_field_finding(
        audit,
        "model_recovery_assessment",
        "the model describes the observed checkpoint-to-failure position as "
        "completed progress, but L0 observed no completed progress marker for "
        "that interval",
        code="observed_failure_position_treated_as_completed_progress",
        severity="credibility",
    )


def _audit_model_evidence_payload(
    bundle: L0Bundle,
    model_view: L0ModelFacingView,
    l1_result: L1EvidenceResult,
    source_log: LogSnapshot,
) -> tuple[FailureEvidence | None, FailureEvidence | None, dict[str, Any]]:
    audit = _new_audit(l1_result)
    evidence = _normalized_l1_evidence(l1_result, audit)
    observation_audit = _new_audit(l1_result)
    selected_observation = _ground_selected_observation(
        bundle=bundle,
        model_view=model_view,
        l1_result=l1_result,
        source_log=source_log,
        evidence=evidence,
        audit=observation_audit,
    )
    grounding = _ground_primary_selection(
        bundle=bundle,
        model_view=model_view,
        l1_result=l1_result,
        source_log=source_log,
        evidence=evidence,
        audit=audit,
    )
    if grounding is None:
        return None, selected_observation, observation_audit

    audit["observation_grounding"] = {
        key: value
        for key, value in observation_audit.items()
        if key not in {"model", "audit_influence"}
    }

    primary = grounding.model_primary
    root_cause_assessment = evidence.get("root_cause_assessment")
    recovery_assessment = evidence.get("model_recovery_assessment")
    assert isinstance(root_cause_assessment, Mapping)
    assert isinstance(recovery_assessment, Mapping)
    _audit_recovery_assessment(
        bundle=bundle,
        grounding=grounding,
        root_cause_assessment=dict(root_cause_assessment),
        recovery_assessment=dict(recovery_assessment),
        audit=audit,
    )
    quote = _model_quote_for_line(grounding.grounded_evidence, grounding.line)
    if quote is None:
        _record_field_finding(
            audit,
            "primary_failure",
            "primary_failure line lacks grounded model evidence",
            code="primary_evidence_ungrounded",
        )
        quote = grounding.log_line
    audit["grounded_evidence"] = grounding.grounded_evidence
    related_failures = _audited_related_failures(
        bundle,
        evidence,
        audit,
        source_log=source_log,
        text_visible_lines=grounding.text_visible_lines,
    )
    audit["audited_related_failures"] = related_failures

    _finalize_model_audit(grounding, audit)
    if not grounding.primary_grounded:
        audit.update(
            {
                "stable_identity_anchor_line": None,
                "stable_identity_anchor_reason": None,
                "stable_root_fingerprint": None,
                "root_fingerprint_source": "unavailable",
                "history_identity_ready": False,
                "affected_entity": None,
            }
        )
        return None, selected_observation, audit

    history_identity = _build_history_identity(
        bundle=bundle,
        source_log=source_log,
        grounding=grounding,
        audit=audit,
    )
    l0_match = _l0_match_for_line(bundle, grounding.line)
    canonical_l0_match = history_identity.l0_match or l0_match
    failure_class = canonical_l0_match.failure_class if canonical_l0_match else "observed_failure"
    signature = canonical_l0_match.signature if canonical_l0_match else quote[:120]
    if l0_match is not None:
        audit["same_line_l0_registry_id"] = l0_match.registry_id
    audit["identity_lineage"] = _identity_lineage(
        bundle,
        model_selected_line=grounding.line,
        identity_anchor_line=history_identity.anchor_line,
    )

    fault_outcome = (
        canonical_l0_match.fault_outcome
        if canonical_l0_match is not None
        else _client_fault_outcome(bundle, grounding.line)
    )
    return (
        _build_grounded_failure_evidence(
            model_primary=primary,
            grounding=grounding,
            l0_match=l0_match,
            history_identity=history_identity,
            failure_class=failure_class,
            signature=signature,
            fault_outcome=fault_outcome,
            quote=quote,
        ),
        selected_observation,
        audit,
    )


def _audit_observation_evidence_references(
    *,
    evidence: Mapping[str, Any],
    audit: dict[str, Any],
) -> None:
    canonical_evidence_ids = {
        item.get("id")
        for item in evidence.get("evidence") or ()
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    unresolved: set[str] = set()
    for index, observation in enumerate(evidence.get("observed_failures") or ()):
        if not isinstance(observation, Mapping):
            continue
        cited_ids = {
            item for item in observation.get("evidence_ids") or () if isinstance(item, str)
        }
        dangling_ids = sorted(cited_ids.difference(canonical_evidence_ids))
        if not dangling_ids:
            continue
        unresolved.update(dangling_ids)
        _record_field_finding(
            audit,
            f"observed_failures[{index}].evidence_ids",
            "observation references undefined evidence ids: " + ", ".join(dangling_ids),
            code="dangling_evidence_reference",
        )
    if unresolved:
        audit["unresolved_evidence_references"] = sorted(unresolved)


def _ground_selected_observation(
    *,
    bundle: L0Bundle,
    model_view: L0ModelFacingView,
    l1_result: L1EvidenceResult,
    source_log: LogSnapshot,
    evidence: Mapping[str, Any],
    audit: dict[str, Any],
) -> FailureEvidence | None:
    selected_id = evidence.get("selected_observed_failure_id")
    observations = evidence.get("observed_failures") or []
    _audit_observation_evidence_references(evidence=evidence, audit=audit)
    if not isinstance(selected_id, str):
        _mark_observation_track_unavailable(audit)
        return None
    selected_matches = [
        item for item in observations if isinstance(item, Mapping) and item.get("id") == selected_id
    ]
    if len(selected_matches) != 1:
        if not selected_matches:
            code = "selected_observation_id_unresolved"
            message = (
                f"selected_observed_failure_id {selected_id!r} does not resolve to an "
                "observed failure"
            )
        else:
            code = "selected_observation_id_ambiguous"
            message = (
                f"selected_observed_failure_id {selected_id!r} resolves to "
                f"{len(selected_matches)} observed failures"
            )
        _record_field_finding(
            audit,
            "selected_observed_failure_id",
            message,
            code=code,
        )
        _mark_observation_track_unavailable(audit)
        return None
    selected = selected_matches[0]
    visible_texts = model_visible_line_texts(model_view, l1_result)
    grounded_evidence, resolved_lines = _audited_model_evidence(
        bundle,
        evidence,
        audit,
        source_log=source_log,
        model_visible_texts=visible_texts,
    )
    cited_ids = set(selected.get("evidence_ids") or ())
    selected_evidence = tuple(item for item in grounded_evidence if item.get("id") in cited_ids)
    original_line = _optional_int(selected.get("line"))
    if original_line is None:
        return None
    line = resolved_lines.get(original_line, original_line)
    log_line = _line_text(source_log, line)
    grounded = bool(log_line is not None and visible_texts.get(line))
    audit.update(
        {
            "used": grounded,
            "grounding_status": "grounded" if grounded else "unavailable",
            "selected_observation_model_visible_support": bool(visible_texts.get(line)),
            "grounded_evidence": selected_evidence,
            "grounded_observed_failures": [dict(selected)] if grounded else [],
            "selected_observation_used": grounded,
            "primary_used": False,
            "recovery_assessment_audited": grounded,
        }
    )
    if not grounded or log_line is None:
        return None
    l0_match = next((match for match in bundle.registry_matches if match.line == line), None)
    l0_selected = bundle.selected_observed_failure
    canonical = l0_selected if l0_selected is not None and l0_selected.line == line else l0_match
    observation_fingerprint = (
        canonical.observation_fingerprint
        if canonical is not None and canonical.observation_fingerprint
        else canonical_observed_fingerprint(log_line, _identity_source_context(source_log, line))
    )
    observation_source = (
        canonical.observation_fingerprint_source
        if canonical is not None and canonical.observation_fingerprint_source
        else "l2_grounded_observation"
    )
    audit.update(
        {
            "stable_identity_anchor_line": line,
            "stable_identity_anchor_reason": "model_selected_observation",
            "stable_root_fingerprint": None,
            "root_fingerprint_source": None,
            "stable_observation_fingerprint": observation_fingerprint,
            "observation_fingerprint_source": observation_source,
            "history_identity_ready": True,
            "identity_kind": "observation_only",
            "affected_entity": None,
        }
    )
    if audit["field_findings"]:
        audit["audit_status"] = "findings"
    else:
        audit["audit_status"] = "clean"
    identity = selected.get("failure_identity")
    failure_identity_grounding = _ground_failure_identity_paths(
        identity if isinstance(identity, Mapping) else {},
        model_view=model_view,
        l1_result=l1_result,
        source_log=source_log,
    )
    audit["failure_identity_grounding"] = failure_identity_grounding
    mechanism = identity.get("mechanism") if isinstance(identity, Mapping) else None
    return FailureEvidence(
        failure_class=str(
            mechanism or (canonical.failure_class if canonical else "observed_failure")
        ),
        signature=(canonical.signature if canonical is not None else log_line[:120]),
        root_fingerprint=None,
        root_fingerprint_source=None,
        observation_fingerprint=observation_fingerprint,
        observation_fingerprint_source=observation_source,
        fault_outcome=(
            canonical.fault_outcome
            if canonical is not None
            else _client_fault_outcome(bundle, line)
        ),
        causal_role=str(selected.get("causal_role") or CausalRole.UNKNOWN.value),
        line=line,
        quote=next(
            (str(item.get("quote")) for item in selected_evidence if item.get("quote")),
            log_line,
        ),
        rank=extract_rank(log_line),
        node=extract_node(log_line),
        gpu=extract_gpu(log_line),
        registry_id=(canonical.registry_id if canonical is not None else "model_observation"),
        role=(canonical.role if canonical is not None else None),
        retry_lifecycle=(canonical.retry_lifecycle if canonical is not None else None),
    )


def _mark_observation_track_unavailable(audit: dict[str, Any]) -> None:
    audit.update(
        {
            "used": False,
            "grounding_status": "unavailable",
            "audit_status": "findings" if audit["field_findings"] else "clean",
            "selected_observation_model_visible_support": None,
            "grounded_observed_failures": [],
            "selected_observation_used": False,
            "recovery_assessment_audited": False,
        }
    )


def _build_history_identity(
    *,
    bundle: L0Bundle,
    source_log: LogSnapshot,
    grounding: PrimaryGrounding,
    audit: dict[str, Any],
) -> HistoryIdentity:
    anchor_line, anchor_reason = canonical_identity_anchor_line(
        bundle,
        grounding.line,
        selection_label="model_primary",
    )
    log_line = _line_text(source_log, anchor_line) or grounding.log_line
    l0_match = _l0_match_for_line(bundle, anchor_line)
    source_context = _identity_source_context(source_log, anchor_line)
    root_fingerprint_source = "unavailable"
    root_fingerprint = None
    distributed_incident = distributed_incident_for_line(bundle, anchor_line)
    if grounding.source_line_available and distributed_incident is not None:
        root_fingerprint = distributed_incident.history_fingerprint
        root_fingerprint_source = distributed_incident.history_fingerprint_source
        audit.update(
            {
                "distributed_incident_id": distributed_incident.incident_id,
                "distributed_incident_kind": distributed_incident.incident_kind,
                "distributed_incident_type": distributed_incident.incident_type,
            }
        )
    elif grounding.source_line_available and l0_match is not None:
        root_fingerprint = l0_match.root_fingerprint
        root_fingerprint_source = l0_match.root_fingerprint_source
    if grounding.source_line_available and not root_fingerprint:
        root_fingerprint_source = "observed_exception"
        root_fingerprint = canonical_observed_fingerprint(log_line, source_context)

    affected_entity_selection = select_grounded_affected_entity(
        grounding.failure_identity_grounding
    )
    affected_entity = (
        affected_entity_selection.entity if affected_entity_selection is not None else None
    )
    audit.update(
        {
            "stable_identity_anchor_line": anchor_line,
            "stable_identity_anchor_reason": anchor_reason,
            "stable_root_fingerprint": root_fingerprint,
            "root_fingerprint_source": root_fingerprint_source,
            "history_identity_ready": bool(root_fingerprint),
            "affected_entity": (
                affected_entity.to_payload() if affected_entity is not None else None
            ),
            "affected_entity_selection": (
                affected_entity_selection.to_payload()
                if affected_entity_selection is not None
                else None
            ),
        }
    )
    return HistoryIdentity(
        anchor_line=anchor_line,
        anchor_reason=anchor_reason,
        log_line=log_line,
        l0_match=l0_match,
        root_fingerprint=root_fingerprint,
        root_fingerprint_source=root_fingerprint_source,
        affected_entity=affected_entity,
    )


def _ground_failure_identity_paths(
    identity: Mapping[str, Any],
    *,
    model_view: L0ModelFacingView,
    l1_result: L1EvidenceResult,
    source_log: LogSnapshot,
) -> dict[str, Any]:
    return {
        field: _ground_model_path(
            identity.get(field),
            model_view=model_view,
            l1_result=l1_result,
            source_log=source_log,
        )
        for field in (DIRECT_FAILURE_OBJECT_PATH_FIELD, AFFECTED_ARTIFACT_PATH_FIELD)
    }


def _ground_model_path(
    value: Any,
    *,
    model_view: L0ModelFacingView,
    l1_result: L1EvidenceResult,
    source_log: LogSnapshot,
) -> dict[str, Any]:
    model_value = _optional_str(value)
    if model_value is None:
        return {
            "model_value": None,
            "grounded_value": None,
            "evidence_lines": [],
            "status": "not_provided",
        }
    visible_lines = model_visible_value_line_numbers(model_view, l1_result, model_value)
    source_lines = sorted(
        line
        for line in visible_lines
        if (source_text := _line_text(source_log, line)) is not None
        and text_contains_exact_value(source_text, model_value)
    )
    grounded = bool(source_lines)
    return {
        "model_value": model_value,
        "grounded_value": model_value if grounded else None,
        "evidence_lines": source_lines,
        "status": "grounded" if grounded else "unavailable",
    }


def _identity_lineage(
    bundle: L0Bundle,
    *,
    model_selected_line: int,
    identity_anchor_line: int,
) -> Mapping[str, Any]:
    l0_primary = bundle.deterministic_primary_candidate
    l0_primary_line = l0_primary.line if l0_primary is not None else None
    if l0_primary_line is None:
        relationship = "l0_identity_unavailable"
    else:
        l0_anchor_line, _ = canonical_identity_anchor_line(
            bundle,
            l0_primary_line,
            selection_label="l0_primary",
        )
        relationship = (
            "same_canonical_incident"
            if l0_anchor_line == identity_anchor_line
            else "different_grounded_incident"
        )
    return {
        "model_selected_line": model_selected_line,
        "l0_primary_line": l0_primary_line,
        "canonical_identity_anchor_line": identity_anchor_line,
        "relationship_to_l0": relationship,
        "client_identity_source": (
            "l0_canonical_identity"
            if relationship == "same_canonical_incident"
            else "l2_source_grounding"
        ),
    }


def _finalize_model_audit(
    grounding: PrimaryGrounding,
    audit: dict[str, Any],
) -> None:
    audit["recovery_assessment_audited"] = grounding.primary_grounded
    audit["used"] = grounding.primary_grounded
    audit["primary_used"] = grounding.primary_grounded
    if audit["field_findings"]:
        audit["audit_status"] = "findings"
    elif any(item.get("status") != "exact" for item in audit["citation_audits"]):
        audit["audit_status"] = "resolved"
    else:
        audit["audit_status"] = "clean"


def _build_grounded_failure_evidence(
    *,
    model_primary: Mapping[str, Any],
    grounding: PrimaryGrounding,
    l0_match: FailureEvidence | None,
    history_identity: HistoryIdentity,
    failure_class: str,
    signature: str,
    fault_outcome: str | None,
    quote: str,
) -> FailureEvidence:
    identity_l0_match = history_identity.l0_match
    lifecycle_match = identity_l0_match or l0_match
    log_line = grounding.log_line
    return FailureEvidence(
        failure_class=failure_class,
        signature=signature,
        root_fingerprint=history_identity.root_fingerprint,
        fault_outcome=fault_outcome,
        causal_role=str(model_primary.get("causal_role") or ""),
        failure_iteration=(
            identity_l0_match.failure_iteration
            if identity_l0_match
            else extract_failure_iteration(history_identity.log_line)
        ),
        line=grounding.line,
        quote=quote,
        rank=(
            l0_match.rank
            if l0_match
            else (extract_rank(log_line) if grounding.source_line_available else None)
        ),
        phase=l0_match.phase if l0_match else None,
        node=(
            l0_match.node
            if l0_match
            else (extract_node(log_line) if grounding.source_line_available else None)
        ),
        gpu=(
            l0_match.gpu
            if l0_match
            else (extract_gpu(log_line) if grounding.source_line_available else None)
        ),
        registry_id=l0_match.registry_id if l0_match else "model_selected",
        role=l0_match.role if l0_match else None,
        root_fingerprint_source=history_identity.root_fingerprint_source,
        affected_entity=history_identity.affected_entity,
        retry_lifecycle=(lifecycle_match.retry_lifecycle if lifecycle_match is not None else None),
    )


def _client_fault_outcome(bundle: L0Bundle, line: int) -> str:
    for episode in bundle.failure_episodes:
        episode_lines = {
            episode.first_exception_line,
            episode.terminal_exception_line,
            episode.identity_anchor_line,
            *episode.precursor_lines,
            *episode.exception_chain_lines,
        }
        if line in episode_lines or episode.start_line <= line <= episode.end_line:
            return episode.status
    return FaultOutcome.UNRESOLVED.value


def _l0_match_for_line(bundle: L0Bundle, line: int) -> FailureEvidence | None:
    primary = bundle.deterministic_primary_candidate
    if primary is not None and primary.line == line:
        return primary
    for match in bundle.registry_matches:
        if match.line == line and match.causal_role not in {
            CausalRole.CASCADE.value,
            CausalRole.TEARDOWN.value,
        }:
            return match
    return None


def _model_quote_for_line(evidence: tuple[Mapping[str, Any], ...], line: int) -> str | None:
    for item in evidence:
        if _optional_int(item.get("line")) == line:
            return _optional_str(item.get("quote"))
    return None


def _audited_model_evidence(
    bundle: L0Bundle,
    evidence: Mapping[str, Any],
    audit: dict[str, Any],
    *,
    source_log: LogSnapshot,
    model_visible_texts: Mapping[int, set[str]],
) -> tuple[tuple[Mapping[str, Any], ...], dict[int, int]]:
    result: list[Mapping[str, Any]] = []
    resolved_lines: dict[int, int] = {}
    for index, item in enumerate(evidence.get("evidence") or []):
        if not isinstance(item, Mapping):
            _record_field_finding(
                audit,
                "evidence",
                f"evidence[{index}] is not an object",
                code="evidence_not_object",
            )
            continue
        line = _optional_int(item.get("line"))
        quote = _optional_str(item.get("quote"))
        supports = tuple(str(value) for value in item.get("supports") or [])
        log_line = _line_text(source_log, line) if line is not None else None
        if line is None or log_line is None:
            _record_field_finding(
                audit,
                "evidence",
                f"evidence[{index}] line is outside the log",
                code="evidence_line_outside_log",
            )
            audit["citation_audits"].append(
                {
                    "index": index,
                    "original_line": line,
                    "resolved_line": None,
                    "supports": list(supports),
                    "status": "ungrounded",
                }
            )
            continue
        if not supports:
            _record_field_finding(
                audit,
                "evidence",
                f"evidence[{index}] supports is missing",
                code="evidence_supports_missing",
            )
            supports = ("unspecified",)
        resolution = _resolve_citation(
            source_log=source_log,
            line=line,
            log_line=log_line,
            quote=quote,
            supports=supports,
            model_visible_texts=model_visible_texts,
        )
        citation_audit = {
            "index": index,
            "original_line": line,
            "resolved_line": resolution.resolved_line,
            "supports": list(resolution.supports),
            "status": resolution.status,
        }
        if resolution.status in {
            "ambiguous_nearby_match",
            "not_model_visible",
            "ungrounded",
        }:
            citation_audit["candidate_lines"] = list(resolution.candidate_lines)
        audit["citation_audits"].append(citation_audit)
        if resolution.resolved_line is not None:
            result.append(
                {
                    "id": item.get("id"),
                    "line": resolution.resolved_line,
                    "quote": resolution.quote,
                    "supports": list(resolution.supports),
                }
            )
        if resolution.status == "nearby_resolved":
            resolved_lines[line] = resolution.resolved_line
            audit.setdefault("grounding_adjustments", []).append(
                {
                    "field": f"evidence[{index}].line",
                    "from": line,
                    "to": resolution.resolved_line,
                    "reason": "nearby_unique_quote_match",
                }
            )
            continue
        if resolution.resolved_line is not None:
            continue
        message = (
            f"evidence[{index}] line and quote were not visible to the model"
            if resolution.status == "not_model_visible"
            else f"evidence[{index}] quote could not be uniquely grounded near line {line}"
        )
        _record_field_finding(
            audit,
            "evidence",
            message,
            code=f"evidence_{resolution.status}",
        )
    return tuple(result), resolved_lines


def _resolve_citation(
    *,
    source_log: LogSnapshot,
    line: int,
    log_line: str,
    quote: str | None,
    supports: tuple[str, ...],
    model_visible_texts: Mapping[int, set[str]],
) -> CitationGrounding:
    visible_matches = _visible_quote_matches(model_visible_texts, line, quote)
    if line not in visible_matches:
        if len(visible_matches) == 1:
            resolved_line = visible_matches[0]
            resolved_text = _line_text(source_log, resolved_line)
            if resolved_text is not None and quote and _quote_matches(resolved_text, quote):
                return CitationGrounding(
                    line,
                    resolved_line,
                    quote,
                    supports,
                    "nearby_resolved",
                )
        return CitationGrounding(
            line,
            None,
            quote,
            supports,
            "ambiguous_nearby_match" if visible_matches else "not_model_visible",
            visible_matches,
        )
    visible_texts = model_visible_texts[line]
    if quote and quote in log_line and any(quote in text for text in visible_texts):
        return CitationGrounding(line, line, quote, supports, "exact")
    if quote and any(quote in visible_text for visible_text in visible_texts):
        return CitationGrounding(line, line, quote, supports, "rendered_exact")
    if (
        quote
        and _abbreviated_quote_matches(log_line, quote)
        and any(_abbreviated_quote_matches(visible_text, quote) for visible_text in visible_texts)
    ):
        return CitationGrounding(line, line, quote, supports, "abbreviated_exact")
    nearby = tuple(
        candidate
        for candidate in _nearby_quote_matches(source_log, line, quote)
        if candidate in visible_matches
    )
    if len(nearby) == 1:
        return CitationGrounding(line, nearby[0], quote, supports, "nearby_resolved")
    return CitationGrounding(
        line,
        None,
        quote,
        supports,
        "ambiguous_nearby_match" if nearby else "ungrounded",
        nearby,
    )


def _visible_quote_matches(
    model_visible_texts: Mapping[int, set[str]],
    line: int,
    quote: str | None,
) -> tuple[int, ...]:
    if not quote:
        return ()
    start = max(1, line - NEARBY_EVIDENCE_LINE_RADIUS)
    stop = line + NEARBY_EVIDENCE_LINE_RADIUS
    return tuple(
        candidate
        for candidate, texts in model_visible_texts.items()
        if start <= candidate <= stop
        and any(
            quote in text or (candidate == line and _abbreviated_quote_matches(text, quote))
            for text in texts
        )
    )


def _nearby_quote_matches(
    source_log: LogSnapshot,
    line: int,
    quote: str | None,
) -> list[int]:
    if not quote:
        return []
    matches: list[int] = []
    start = max(1, line - NEARBY_EVIDENCE_LINE_RADIUS)
    for candidate in range(start, line + NEARBY_EVIDENCE_LINE_RADIUS + 1):
        if candidate == line:
            continue
        text = _line_text(source_log, candidate)
        if text is not None and _quote_matches(text, quote):
            matches.append(candidate)
    return matches


def _quote_matches(source: str, quote: str) -> bool:
    if quote in source:
        return True
    return _abbreviated_quote_matches(source, quote)


def _abbreviated_quote_matches(source: str, quote: str) -> bool:
    if _ELLIPSIS_RE.search(quote) is None:
        return False
    fragments = [fragment.strip() for fragment in _ELLIPSIS_RE.split(quote) if fragment.strip()]
    if len(fragments) < 2 or any(
        sum(character.isalnum() for character in fragment) < MIN_ABBREVIATED_FRAGMENT_ALNUM_CHARS
        for fragment in fragments
    ):
        return False
    offset = 0
    for fragment in fragments:
        index = source.find(fragment, offset)
        if index < 0:
            return False
        offset = index + len(fragment)
    return True


def _audit_unverified_path_identity_claims(
    bundle: L0Bundle,
    root_cause_assessment: Mapping[str, Any],
    recovery_assessment: Mapping[str, Any],
    audit: dict[str, Any],
) -> None:
    summary = bundle.path_namespace_summary
    if not summary.get("cross_namespace_paths_observed") or summary.get("ownership_verified"):
        return
    assessment_text = " ".join(
        str(value)
        for value in (
            root_cause_assessment.get("summary"),
            root_cause_assessment.get("plausible_causes"),
            recovery_assessment.get("rationale"),
        )
        if value is not None
    )
    if not re.search(
        r"\b(?:owned by|belongs to|different user|another user|effective user|running as)\b",
        assessment_text,
        re.I,
    ):
        return
    _record_field_finding(
        audit,
        "root_cause_assessment",
        "path namespaces do not prove the effective process user, file owner, mode, or ACL",
        code="path_namespace_identity_unverified",
    )


def _audited_related_failures(
    bundle: L0Bundle,
    evidence: Mapping[str, Any],
    audit: dict[str, Any],
    *,
    source_log: LogSnapshot,
    text_visible_lines: frozenset[int],
) -> tuple[Mapping[str, Any], ...]:
    primary = evidence.get("primary_failure")
    primary_line = _optional_int(primary.get("line")) if isinstance(primary, Mapping) else None
    result: list[Mapping[str, Any]] = []
    audited_roles: list[dict[str, Any]] = []
    for index, item in enumerate(evidence.get("related_failures") or []):
        if not isinstance(item, Mapping):
            _record_field_finding(
                audit,
                "related_failures",
                f"related_failures[{index}] must be an object",
                code="related_failure_not_object",
            )
            continue
        line = _optional_int(item.get("line"))
        role = _optional_str(item.get("causal_role"))
        rationale = _optional_str(item.get("rationale"))
        log_line = _line_text(source_log, line) if line is not None else None
        if line is None or log_line is None or role is None or rationale is None:
            _record_field_finding(
                audit,
                "related_failures",
                f"related_failures[{index}] must cite a valid line, role, and rationale",
                code="related_failure_invalid_reference",
            )
            continue
        if line not in text_visible_lines:
            _record_field_finding(
                audit,
                "related_failures",
                f"related_failures[{index}] line {line} was not visible to the model",
                code="related_failure_line_not_model_visible",
            )
            continue
        if role not in {
            CausalRole.CASCADE.value,
            CausalRole.TEARDOWN.value,
            CausalRole.UNKNOWN.value,
        }:
            _record_field_finding(
                audit,
                "related_failures",
                f"related_failures[{index}] has invalid related causal role {role}",
                code="related_failure_causal_role_invalid",
            )
            continue
        if line == primary_line:
            _record_field_finding(
                audit,
                "related_failures",
                f"related_failures[{index}] duplicates primary_failure.line",
                code="related_failure_duplicates_primary",
            )
            continue
        if role in {CausalRole.CASCADE.value, CausalRole.TEARDOWN.value} and (
            primary_line is not None and line < primary_line
        ):
            _record_field_finding(
                audit,
                "related_failures",
                f"related_failures[{index}] cannot precede the primary with role {role}",
                code="related_failure_impossible_chronology",
            )
        audited_roles.append({"line": line, "causal_role": role, "rationale": rationale})
        l0_match = _l0_match_for_line(bundle, line)
        failure_class = l0_match.failure_class if l0_match else "related_failure"
        signature = l0_match.signature if l0_match else log_line[:120]
        root_fingerprint = (
            l0_match.root_fingerprint
            if l0_match and l0_match.root_fingerprint
            else canonical_observed_fingerprint(
                log_line,
                _context_before_line(bundle, line),
            )
        )
        payload = FailureEvidence(
            failure_class=failure_class,
            signature=signature,
            root_fingerprint=root_fingerprint,
            fault_outcome=l0_match.fault_outcome if l0_match else "unresolved",
            causal_role=role,
            line=line,
            quote=log_line,
            rank=l0_match.rank if l0_match else extract_rank(log_line),
            phase=l0_match.phase if l0_match else None,
            node=l0_match.node if l0_match else extract_node(log_line),
            gpu=l0_match.gpu if l0_match else extract_gpu(log_line),
            root_fingerprint_source=(
                l0_match.root_fingerprint_source
                if l0_match and l0_match.root_fingerprint
                else "observed_exception"
            ),
            retry_lifecycle=(l0_match.retry_lifecycle if l0_match is not None else None),
        ).to_failure_payload()
        payload["relationship_rationale"] = rationale
        result.append(payload)
    audit["audited_related_failure_roles"] = audited_roles
    return tuple(result)


def _record_field_finding(
    audit: dict[str, Any],
    field: str,
    message: str,
    *,
    code: str,
    severity: str = "credibility",
) -> None:
    field_findings = audit.setdefault("field_findings", {})
    field_findings.setdefault(field, []).append(message)
    field_finding_codes = audit.setdefault("field_finding_codes", {})
    field_finding_codes.setdefault(field, []).append(code)
    audit.setdefault("findings", []).append(
        {
            "field": field,
            "code": code,
            "message": message,
            "severity": severity,
            "observational_only": True,
        }
    )


def _observed_failure_position_overclaimed_as_completed_progress(
    bundle: L0Bundle,
    root_cause_assessment: Mapping[str, Any],
    recovery_assessment: Mapping[str, Any],
) -> bool:
    summary = bundle.run_progress_summary
    distance = summary.observed_iterations_after_checkpoint_load
    failure_iteration = summary.latest_observed_failure_iteration
    if not distance or failure_iteration is None:
        return False
    if summary.last_iteration is not None and summary.last_iteration >= failure_iteration:
        return False

    text = " ".join(
        str(value)
        for value in (
            root_cause_assessment.get("summary"),
            root_cause_assessment.get("plausible_causes"),
            root_cause_assessment.get("missing_evidence"),
            recovery_assessment.get("rationale"),
        )
        if value is not None
    )
    distance_pattern = re.escape(str(distance))
    claims_completion = any(
        re.search(pattern, text, re.I)
        for pattern in (
            rf"\b(?:progressed|advanced)\s+(?:by\s+)?{distance_pattern}\s+"
            r"(?:iterations?|steps?)\b",
            rf"\b{distance_pattern}\s+(?:successful\s+)?(?:iterations?|steps?)\s+"
            r"(?:of\s+)?successful\s+(?:execution|progress)\b",
            rf"\b{distance_pattern}\s+successful\s+(?:iterations?|steps?)\b",
            r"\bmodel state (?:has )?evolved deterministically\b",
        )
    )
    explicitly_limits_claim = bool(
        re.search(
            r"\b(?:does not|doesn't|cannot|can't)\s+(?:itself\s+)?"
            r"(?:establish|prove|confirm).{0,80}\b(?:recovery|persistence|success)",
            text,
            re.I,
        )
    )
    return claims_completion and not explicitly_limits_claim


def _context_before_line(bundle: L0Bundle, line: int) -> tuple[str, ...]:
    by_line: dict[int, str] = {}
    for window in bundle.context_windows:
        for item in window.lines:
            if item.line < line:
                by_line[item.line] = item.text
    return tuple(by_line[key] for key in sorted(by_line)[-80:])


def _identity_source_context(source_log: LogSnapshot, line: int) -> tuple[str, ...]:
    return source_log.context_before(line, limit=80)


def _line_text(source_log: LogSnapshot, line: int) -> str | None:
    return source_log.line(line)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
