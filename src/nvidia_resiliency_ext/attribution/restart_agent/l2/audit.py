# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L2 source grounding, history identity, and credibility diagnostics."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..identity import (
    build_experimental_failure_identity,
    canonical_observed_fingerprint,
    extract_data_position_fingerprint,
    extract_failure_iteration,
    extract_gpu,
    extract_node,
    extract_rank,
)
from ..infrastructure.log_source import LogSnapshot
from ..l0.decision import canonical_identity_anchor_line, distributed_incident_for_line
from ..l1.contracts import L1EvidenceResult
from ..l1.response_contract import FAILURE_DOMAIN_SUPPORT_TAG, RETRY_OUTLOOK_SUPPORT_TAG
from ..models import (
    AssessmentStatus,
    AttemptFailureFacts,
    AttemptFailureFactsSource,
    CausalRole,
    FailureEvidence,
    FaultOutcome,
    L0Bundle,
    L0ModelFacingView,
    ModelRecoveryAssessment,
    PolicyClass,
)
from .failure_facts import build_attempt_failure_facts
from .grounding import model_visible_line_numbers, model_visible_line_texts

PROHIBITED_MODEL_FIELDS = {
    "user_failure",
    "not_user_failure",
    "decision",
    "decision_basis",
    "evidence_coverage",
}
NEARBY_EVIDENCE_LINE_RADIUS = 5


@dataclass(frozen=True)
class L2GroundingInput:
    """Complete typed input required to ground and audit one L1 result."""

    bundle: L0Bundle
    model_view: L0ModelFacingView
    l1_result: L1EvidenceResult
    source_log: LogSnapshot


@dataclass(frozen=True)
class L2Result:
    """Grounded L1 semantics plus optional audit diagnostics."""

    primary: FailureEvidence | None
    enriched_failure_facts: AttemptFailureFacts | None
    used: bool
    grounding_status: str
    audit_status: str
    root_cause_assessment: Mapping[str, Any] | None = None
    model_recovery_assessment: ModelRecoveryAssessment | None = None
    recovery_assessment_used: bool = False
    recovery_assessment_policy_grounded: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload = dict(self.diagnostics)
        payload.update(
            {
                "used": self.used,
                "grounding_status": self.grounding_status,
                "audit_status": self.audit_status,
                "primary_used": self.primary is not None and self.used,
                "enriched_failure_facts": (
                    self.enriched_failure_facts.to_payload()
                    if self.enriched_failure_facts is not None
                    else None
                ),
                "recovery_assessment_used": self.recovery_assessment_used,
                "recovery_assessment_policy_grounded": (self.recovery_assessment_policy_grounded),
                "root_cause_assessment": (
                    dict(self.root_cause_assessment)
                    if self.root_cause_assessment is not None
                    else None
                ),
                "model_recovery_assessment": (
                    self.model_recovery_assessment.to_payload()
                    if self.model_recovery_assessment is not None
                    else None
                ),
            }
        )
        return payload

    @classmethod
    def not_run(cls, reason: str) -> "L2Result":
        return cls(
            primary=None,
            enriched_failure_facts=None,
            used=False,
            grounding_status="not_run",
            audit_status="not_run",
            recovery_assessment_policy_grounded=False,
            diagnostics={
                "not_run_reason": reason,
                "field_findings": {},
                "field_finding_codes": {},
                "findings": [],
                "ignored_prohibited_fields": [],
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
    grounding_method: str
    grounded_evidence: tuple[Mapping[str, Any], ...]
    resolved_lines: Mapping[int, int]
    cited_lines: frozenset[int]
    visible_lines: frozenset[int]


@dataclass(frozen=True)
class RecoveryAudit:
    """Grounded support and credibility signals for the model recovery assessment."""

    root_cause_assessment: Mapping[str, Any]
    recovery_assessment: Mapping[str, Any]
    failure_domain_value: str
    failure_domain_status: str
    failure_domain_confidence: int
    retry_outlook_value: str
    retry_outlook_status: str
    retry_outlook_confidence: int
    root_cause_status: str
    failure_domain_supporting_lines: frozenset[int]
    retry_outlook_supporting_lines: frozenset[int]
    unresolved_supporting_lines: tuple[int, ...]


@dataclass(frozen=True)
class RecoverySupport:
    """Resolved source support attached to the model recovery assessment."""

    failure_domain_lines: frozenset[int]
    retry_outlook_lines: frozenset[int]
    unresolved_lines: tuple[int, ...]


@dataclass(frozen=True)
class HistoryIdentity:
    """Source-grounded identity used for current-failure history comparison."""

    anchor_line: int
    anchor_reason: str
    log_line: str
    l0_match: FailureEvidence | None
    root_fingerprint: str | None
    root_fingerprint_source: str
    failure_identity: Mapping[str, Any]


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
    payload: Mapping[str, Any],
    model_view: L0ModelFacingView,
) -> L2Result:
    details = dict(payload)
    root_cause = details.pop("root_cause_assessment", None)
    assessment = details.pop("model_recovery_assessment", None)
    used = bool(details.pop("used", False))
    grounding_status = str(details.pop("grounding_status", "unavailable"))
    audit_status = str(details.pop("audit_status", "findings"))
    recovery_assessment_used = bool(details.pop("recovery_assessment_used", False))
    recovery_assessment_policy_grounded = bool(
        details.pop("recovery_assessment_policy_grounded", False)
    )
    details.pop("primary_used", None)
    enriched_failure_facts = (
        build_attempt_failure_facts(
            primary,
            model_view.decision_evidence,
            source=AttemptFailureFactsSource.L2_GROUNDED,
            identity_anchor_line=_optional_int(details.get("stable_identity_anchor_line")),
            identity_anchor_reason=_optional_str(details.get("stable_identity_anchor_reason")),
        )
        if primary is not None
        else None
    )
    return L2Result(
        primary=primary,
        enriched_failure_facts=enriched_failure_facts,
        used=used,
        grounding_status=grounding_status,
        audit_status=audit_status,
        root_cause_assessment=(dict(root_cause) if isinstance(root_cause, Mapping) else None),
        model_recovery_assessment=(
            ModelRecoveryAssessment.from_mapping(assessment)
            if isinstance(assessment, Mapping)
            else None
        ),
        recovery_assessment_used=recovery_assessment_used,
        recovery_assessment_policy_grounded=recovery_assessment_policy_grounded,
        diagnostics=details,
    )


def ground_and_audit_model_evidence(
    grounding_input: L2GroundingInput,
) -> L2Result:
    primary, payload = _audit_model_evidence_payload(
        grounding_input.bundle,
        grounding_input.model_view,
        grounding_input.l1_result,
        grounding_input.source_log,
    )
    return result_from_payload(primary, payload, grounding_input.model_view)


def _new_audit(l1_result: L1EvidenceResult) -> dict[str, Any]:
    return {
        "used": False,
        "audit_status": "findings",
        "primary_used": False,
        "recovery_assessment_used": False,
        "recovery_assessment_policy_grounded": False,
        "field_findings": {},
        "field_finding_codes": {},
        "findings": [],
        "ignored_prohibited_fields": [],
        "citation_audits": [],
        "grounding_adjustments": [],
        "recovery_field_audits": [],
        "model": l1_result.model or None,
    }


def _normalized_l1_evidence(
    l1_result: L1EvidenceResult,
    audit: dict[str, Any],
) -> dict[str, Any]:
    if not l1_result.success or l1_result.evidence is None:
        raise ValueError("L2 grounding requires a structurally usable L1 response")

    evidence = dict(l1_result.evidence)
    prohibited = sorted(PROHIBITED_MODEL_FIELDS.intersection(evidence))
    if not prohibited:
        return evidence

    audit["ignored_prohibited_fields"] = prohibited
    _record_field_finding(
        audit,
        "top_level",
        "ignored prohibited model fields: " + ", ".join(prohibited),
        code="prohibited_fields_ignored",
    )
    return {key: value for key, value in evidence.items() if key not in prohibited}


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
    if causal_role not in {CausalRole.INITIATING.value, CausalRole.UNKNOWN.value}:
        _record_field_finding(
            audit,
            "primary_failure",
            "primary_failure causal_role is not initiating or unknown",
            code="primary_causal_role_suspect",
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

    visible_lines = model_visible_line_numbers(model_view, l1_result)
    grounded_evidence, resolved_lines = _audited_model_evidence(
        bundle,
        evidence,
        audit,
        source_log=source_log,
        model_visible_texts=model_visible_line_texts(model_view, l1_result),
    )
    resolved_primary_line = resolved_lines.get(line, line)
    grounding_method = "exact_source_line" if source_line_available else "unavailable"
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
        grounding_method = "nearby_unique_quote_match"

    audit["grounding_status"] = "grounded" if source_line_available else "unavailable"
    audit["grounding_method"] = grounding_method
    return PrimaryGrounding(
        model_primary=primary,
        line=line,
        log_line=log_line,
        source_line_available=source_line_available,
        grounding_method=grounding_method,
        grounded_evidence=grounded_evidence,
        resolved_lines=resolved_lines,
        cited_lines=frozenset(int(item["line"]) for item in grounded_evidence),
        visible_lines=frozenset(visible_lines),
    )


def _audit_recovery_assessment(
    *,
    bundle: L0Bundle,
    grounding: PrimaryGrounding,
    root_cause_assessment: Mapping[str, Any],
    recovery_assessment: Mapping[str, Any],
    audit: dict[str, Any],
) -> RecoveryAudit:
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
            "root_cause_assessment": dict(root_cause_assessment),
            "model_recovery_assessment": dict(recovery_assessment),
            "path_namespace_summary": dict(bundle.path_namespace_summary),
        }
    )
    _audit_unverified_path_identity_claims(
        bundle,
        root_cause_assessment,
        recovery_assessment,
        audit,
    )
    support = _ground_recovery_support(grounding=grounding, audit=audit)
    _audit_progress_claim(
        bundle,
        root_cause_assessment,
        recovery_assessment,
        audit,
    )

    return RecoveryAudit(
        root_cause_assessment=root_cause_assessment,
        recovery_assessment=recovery_assessment,
        failure_domain_value=failure_domain_value,
        failure_domain_status=failure_domain_status,
        failure_domain_confidence=int(failure_domain.get("confidence") or 0),
        retry_outlook_value=retry_outlook_value,
        retry_outlook_status=retry_outlook_status,
        retry_outlook_confidence=int(retry_outlook.get("confidence") or 0),
        root_cause_status=root_cause_status,
        failure_domain_supporting_lines=support.failure_domain_lines,
        retry_outlook_supporting_lines=support.retry_outlook_lines,
        unresolved_supporting_lines=support.unresolved_lines,
    )


def _ground_recovery_support(
    *, grounding: PrimaryGrounding, audit: dict[str, Any]
) -> RecoverySupport:
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
    unresolved = tuple(
        sorted(
            int(item["original_line"])
            for item in audit.get("citation_audits") or []
            if item.get("resolved_line") is None
            and set(item.get("supports") or []).intersection(
                {FAILURE_DOMAIN_SUPPORT_TAG, RETRY_OUTLOOK_SUPPORT_TAG}
            )
            and isinstance(item.get("original_line"), int)
        )
    )
    audit.update(
        {
            "failure_domain_supporting_lines": sorted(domain_lines),
            "retry_outlook_supporting_lines": sorted(retry_lines),
            "recovery_unresolved_supporting_lines": list(unresolved),
        }
    )
    if not domain_lines:
        _record_field_finding(
            audit,
            "model_recovery_assessment",
            "failure_domain has no grounded evidence",
            code="failure_domain_support_missing",
        )
    if not retry_lines:
        _record_field_finding(
            audit,
            "model_recovery_assessment",
            "retry_outlook has no grounded evidence",
            code="retry_outlook_support_missing",
        )
    if unresolved:
        _record_field_finding(
            audit,
            "model_recovery_assessment",
            "recovery evidence could not be grounded at lines: "
            + ", ".join(str(value) for value in unresolved),
            code="recovery_support_ungrounded",
        )
    return RecoverySupport(domain_lines, retry_lines, unresolved)


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
        policy_material=False,
    )


def _audit_model_evidence_payload(
    bundle: L0Bundle,
    model_view: L0ModelFacingView,
    l1_result: L1EvidenceResult,
    source_log: LogSnapshot,
) -> tuple[FailureEvidence | None, dict[str, Any]]:
    audit = _new_audit(l1_result)
    evidence = _normalized_l1_evidence(l1_result, audit)
    grounding = _ground_primary_selection(
        bundle=bundle,
        model_view=model_view,
        l1_result=l1_result,
        source_log=source_log,
        evidence=evidence,
        audit=audit,
    )
    if grounding is None:
        return None, audit

    primary = grounding.model_primary
    root_cause_assessment = evidence.get("root_cause_assessment")
    recovery_assessment = evidence.get("model_recovery_assessment")
    assert isinstance(root_cause_assessment, Mapping)
    assert isinstance(recovery_assessment, Mapping)
    recovery = _audit_recovery_assessment(
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
        visible_lines=grounding.visible_lines,
    )
    audit["audited_related_failures"] = related_failures

    l0_match = _l0_match_for_line(bundle, grounding.line)
    model_identity = primary.get("failure_identity")
    model_mechanism = (
        _optional_str(model_identity.get("mechanism"))
        if isinstance(model_identity, Mapping)
        else None
    )
    fine_class = l0_match.fine_class if l0_match else model_mechanism or "model_selected_failure"
    signature = l0_match.signature if l0_match else quote[:120]
    if l0_match is not None:
        audit["same_line_l0_registry_id"] = l0_match.registry_id

    fault_outcome = (
        l0_match.fault_outcome
        if l0_match is not None
        else _client_fault_outcome(bundle, grounding.line)
    )

    history_identity = _build_history_identity(
        bundle=bundle,
        source_log=source_log,
        grounding=grounding,
        model_primary=primary,
        audit=audit,
    )
    _finalize_model_audit(grounding, recovery, audit)
    return (
        _build_grounded_failure_evidence(
            model_primary=primary,
            grounding=grounding,
            l0_match=l0_match,
            history_identity=history_identity,
            fine_class=fine_class,
            signature=signature,
            fault_outcome=fault_outcome,
            quote=quote,
        ),
        audit,
    )


def _build_history_identity(
    *,
    bundle: L0Bundle,
    source_log: LogSnapshot,
    grounding: PrimaryGrounding,
    model_primary: Mapping[str, Any],
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

    model_identity = model_primary.get("failure_identity")
    failure_identity = build_experimental_failure_identity(
        log_line,
        source_context,
        model_identity=model_identity if isinstance(model_identity, Mapping) else None,
        model_context=tuple(
            str(item.get("quote")) for item in grounding.grounded_evidence if item.get("quote")
        ),
        observed_phase=_identity_source_phase(bundle, anchor_line),
    )
    audit.update(
        {
            "stable_identity_anchor_line": anchor_line,
            "stable_identity_anchor_reason": anchor_reason,
            "stable_root_fingerprint": root_fingerprint,
            "root_fingerprint_source": root_fingerprint_source,
            "history_identity_ready": bool(root_fingerprint),
            "experimental_failure_identity": failure_identity,
        }
    )
    return HistoryIdentity(
        anchor_line=anchor_line,
        anchor_reason=anchor_reason,
        log_line=log_line,
        l0_match=l0_match,
        root_fingerprint=root_fingerprint,
        root_fingerprint_source=root_fingerprint_source,
        failure_identity=failure_identity,
    )


def _finalize_model_audit(
    grounding: PrimaryGrounding,
    recovery: RecoveryAudit,
    audit: dict[str, Any],
) -> None:
    audit["recovery_assessment_used"] = True
    audit["recovery_assessment_policy_grounded"] = bool(
        grounding.source_line_available
        and recovery.root_cause_status
        in {
            AssessmentStatus.ESTABLISHED_BY_CURRENT_LOG.value,
            AssessmentStatus.SUPPORTED_BUT_UNCONFIRMED.value,
        }
        and recovery.failure_domain_supporting_lines
        and recovery.retry_outlook_supporting_lines
        and not recovery.unresolved_supporting_lines
    )
    audit["used"] = True
    audit["primary_used"] = True
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
    fine_class: str,
    signature: str,
    fault_outcome: str | None,
    quote: str,
) -> FailureEvidence:
    identity_l0_match = history_identity.l0_match
    log_line = grounding.log_line
    return FailureEvidence(
        fine_class=fine_class,
        policy_class=PolicyClass.AMBIGUOUS.value,
        signature=signature,
        root_fingerprint=history_identity.root_fingerprint,
        fault_outcome=fault_outcome,
        causal_role=str(model_primary.get("causal_role") or ""),
        failure_iteration=(
            identity_l0_match.failure_iteration
            if identity_l0_match
            else extract_failure_iteration(history_identity.log_line)
        ),
        data_position_fingerprint=(
            l0_match.data_position_fingerprint
            if l0_match
            else (
                extract_data_position_fingerprint(log_line)
                if grounding.source_line_available
                else None
            )
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
        recovery_behavior=l0_match.recovery_behavior if l0_match else "none",
        root_fingerprint_source=history_identity.root_fingerprint_source,
        failure_identity=history_identity.failure_identity,
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
        if match.line == line and match.policy_class != PolicyClass.CASCADE.value:
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
    if quote and (
        _quote_matches(log_line, quote)
        or any(_quote_matches(visible_text, quote) for visible_text in visible_texts)
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
        and any(quote in text or _quote_matches(text, quote) for text in texts)
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
    fragments = [
        fragment.strip()
        for fragment in re.split(r"(?:\.\.\.\[truncated\]|\.\.\.|…)", quote)
        if fragment.strip()
    ]
    if len(fragments) == 1 and not ("..." in quote or "…" in quote):
        return False
    offset = 0
    for fragment in fragments:
        index = source.find(fragment, offset)
        if index < 0:
            return False
        offset = index + len(fragment)
    return bool(fragments)


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
    visible_lines: frozenset[int],
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
        if line not in visible_lines:
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
            continue
        audited_roles.append({"line": line, "causal_role": role, "rationale": rationale})
        l0_match = _l0_match_for_line(bundle, line)
        fine_class = l0_match.fine_class if l0_match else "related_failure"
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
            fine_class=fine_class,
            policy_class=PolicyClass.AMBIGUOUS.value,
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
    policy_material: bool = True,
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
            "policy_material": policy_material,
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


def _identity_source_phase(bundle: L0Bundle, line: int) -> str | None:
    match = _l0_match_for_line(bundle, line)
    if match is not None and match.phase:
        return match.phase
    primary = bundle.deterministic_primary_candidate
    if primary is not None and primary.line == line and primary.phase:
        return primary.phase
    for episode in bundle.failure_episodes:
        observed_lines = {
            *episode.precursor_lines,
            *episode.exception_chain_lines,
            episode.terminal_exception_line,
        }
        if line not in observed_lines:
            continue
        terminal_line = episode.terminal_exception_line
        terminal_match = (
            _l0_match_for_line(bundle, terminal_line) if terminal_line is not None else None
        )
        if terminal_match is not None and terminal_match.phase:
            return terminal_match.phase
    prior_progress = [
        marker
        for marker in (
            *bundle.progress.progress_markers,
            *bundle.progress.checkpoint_markers,
        )
        if marker.line <= line
    ]
    if prior_progress:
        first_progress_line = min(marker.line for marker in prior_progress)
        return "first_iter" if line <= first_progress_line else "steady_mid"
    prior_setup = [marker for marker in bundle.progress.setup_markers if marker.line <= line]
    if prior_setup:
        return max(prior_setup, key=lambda marker: marker.line).marker_type
    return None


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
