# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral validation for the closed L1 semantic evidence contract."""

from __future__ import annotations

from typing import Any, Mapping

from ..models import L1_EVIDENCE_SCHEMA_VERSION, AssessmentStatus, L1AnalysisStatus
from .response_contract import L1_RESPONSE_CONTRACT


def model_evidence_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Return structural errors without applying semantic policy judgment."""

    errors = _object_shape_errors(
        payload,
        L1_RESPONSE_CONTRACT.top_level_fields,
        "top-level",
        optional=L1_RESPONSE_CONTRACT.optional_top_level_fields,
    )
    errors.extend(_category_selection_errors(payload.get("category_selection")))
    if payload.get("schema_version") != L1_EVIDENCE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {L1_EVIDENCE_SCHEMA_VERSION}")

    analysis_status = payload.get("analysis_status")
    if analysis_status not in L1_RESPONSE_CONTRACT.analysis_statuses:
        errors.append("analysis_status is invalid")

    primary = payload.get("primary_failure")
    if primary is not None and not isinstance(primary, Mapping):
        errors.append("primary_failure must be an object or null")
    elif isinstance(primary, Mapping):
        errors.extend(_primary_failure_errors(primary))

    errors.extend(_observed_failure_errors(payload.get("observed_failures")))
    selected_observation_id = payload.get("selected_observed_failure_id")
    if selected_observation_id is not None and not _nonempty_string(selected_observation_id):
        errors.append("selected_observed_failure_id must be a non-empty string or null")

    if analysis_status == L1AnalysisStatus.PRIMARY_IDENTIFIED.value and not isinstance(
        primary, Mapping
    ):
        errors.append(
            "primary_failure must be an object when analysis_status is primary_identified"
        )
    if (
        analysis_status
        in {
            L1AnalysisStatus.NO_FAILURE_OBSERVED.value,
            L1AnalysisStatus.INSUFFICIENT_EVIDENCE.value,
        }
        and primary is not None
    ):
        errors.append("primary_failure must be null when no primary was identified")

    errors.extend(_root_cause_errors(payload.get("root_cause_assessment")))
    errors.extend(_recovery_assessment_errors(payload.get("model_recovery_assessment")))
    errors.extend(_related_failure_errors(payload.get("related_failures")))
    evidence_errors, _support_tags = _evidence_errors(payload.get("evidence"))
    errors.extend(evidence_errors)
    if analysis_status in {
        L1AnalysisStatus.NO_FAILURE_OBSERVED.value,
        L1AnalysisStatus.INSUFFICIENT_EVIDENCE.value,
    }:
        errors.extend(_non_primary_semantic_errors(payload, analysis_status))
    return errors


def _object_shape_errors(
    value: Mapping[str, Any],
    expected: frozenset[str],
    field: str,
    *,
    optional: frozenset[str] = frozenset(),
) -> list[str]:
    errors: list[str] = []
    missing = sorted(expected.difference(value))
    if missing:
        errors.append(f"{field} missing fields: " + ", ".join(missing))
    return errors


def _primary_failure_errors(primary: Mapping[str, Any]) -> list[str]:
    errors = _object_shape_errors(
        primary,
        L1_RESPONSE_CONTRACT.primary_failure_fields,
        "primary_failure",
    )
    errors.extend(_positive_line_errors(primary.get("line"), "primary_failure.line"))
    if primary.get("causal_role") not in L1_RESPONSE_CONTRACT.primary_causal_roles:
        errors.append("primary_failure.causal_role is invalid")
    identity = primary.get("failure_identity")
    if not isinstance(identity, Mapping):
        errors.append("primary_failure.failure_identity must be an object")
        return errors
    errors.extend(
        _object_shape_errors(
            identity,
            L1_RESPONSE_CONTRACT.failure_identity_fields,
            "primary_failure.failure_identity",
        )
    )
    for name in L1_RESPONSE_CONTRACT.failure_identity_fields:
        value = identity.get(name)
        if value is not None and not isinstance(value, str):
            errors.append(f"primary_failure.failure_identity.{name} must be a string or null")
    return errors


def _root_cause_errors(root_cause: Any) -> list[str]:
    if not isinstance(root_cause, Mapping):
        return ["root_cause_assessment must be an object"]
    errors = _object_shape_errors(
        root_cause,
        L1_RESPONSE_CONTRACT.root_cause_fields,
        "root_cause_assessment",
    )
    if not _nonempty_string(root_cause.get("summary")):
        errors.append("root_cause_assessment.summary must be a non-empty string")
    if root_cause.get("status") not in L1_RESPONSE_CONTRACT.assessment_statuses:
        errors.append("root_cause_assessment.status is invalid")
    for field in ("plausible_causes", "missing_evidence"):
        value = root_cause.get(field)
        if not isinstance(value, list):
            errors.append(f"root_cause_assessment.{field} must be an array")
        elif not all(_nonempty_string(item) for item in value):
            errors.append(f"root_cause_assessment.{field} items must be non-empty strings")
    return errors


def _recovery_assessment_errors(assessment: Any) -> list[str]:
    if not isinstance(assessment, Mapping):
        return ["model_recovery_assessment must be an object"]
    errors = _object_shape_errors(
        assessment,
        L1_RESPONSE_CONTRACT.recovery_assessment_fields,
        "model_recovery_assessment",
    )
    errors.extend(
        _claim_errors(
            assessment.get("failure_domain"),
            field="model_recovery_assessment.failure_domain",
            valid_values=L1_RESPONSE_CONTRACT.failure_domains,
        )
    )
    errors.extend(
        _claim_errors(
            assessment.get("retry_outlook_without_workload_change"),
            field="model_recovery_assessment.retry_outlook_without_workload_change",
            valid_values=L1_RESPONSE_CONTRACT.retry_outlooks,
        )
    )
    if not _nonempty_string(assessment.get("rationale")):
        errors.append("model_recovery_assessment.rationale must be a non-empty string")
    return errors


def _claim_errors(value: Any, *, field: str, valid_values: frozenset[str]) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{field} must be an object"]
    errors = _object_shape_errors(value, L1_RESPONSE_CONTRACT.claim_fields, field)
    claim_value = value.get("value")
    claim_status = value.get("status")
    if claim_value not in valid_values:
        errors.append(f"{field}.value is invalid")
    if claim_status not in L1_RESPONSE_CONTRACT.assessment_statuses:
        errors.append(f"{field}.status is invalid")
    confidence = value.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int)
        or not L1_RESPONSE_CONTRACT.min_confidence
        <= confidence
        <= L1_RESPONSE_CONTRACT.max_confidence
    ):
        errors.append(
            f"{field}.confidence must be an integer from "
            f"{L1_RESPONSE_CONTRACT.min_confidence} to "
            f"{L1_RESPONSE_CONTRACT.max_confidence}"
        )
    if (claim_value == "unknown") != (claim_status == AssessmentStatus.UNKNOWN.value):
        errors.append(f"{field} must use value=unknown and status=unknown together")
    return errors


def _related_failure_errors(related: Any) -> list[str]:
    if not isinstance(related, list):
        return ["related_failures must be an array"]
    errors: list[str] = []
    for index, item in enumerate(related):
        field = f"related_failures[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{field} must be an object")
            continue
        errors.extend(
            _object_shape_errors(item, L1_RESPONSE_CONTRACT.related_failure_fields, field)
        )
        errors.extend(_positive_line_errors(item.get("line"), f"{field}.line"))
        if item.get("causal_role") not in L1_RESPONSE_CONTRACT.related_causal_roles:
            errors.append(f"{field}.causal_role is invalid")
        if not _nonempty_string(item.get("rationale")):
            errors.append(f"{field}.rationale must be a non-empty string")
    return errors


def _observed_failure_errors(value: Any) -> list[str]:
    if not isinstance(value, list):
        return ["observed_failures must be an array"]
    errors: list[str] = []
    for index, item in enumerate(value):
        field = f"observed_failures[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{field} must be an object")
            continue
        errors.extend(
            _object_shape_errors(item, L1_RESPONSE_CONTRACT.observed_failure_fields, field)
        )
        item_id = item.get("id")
        if not _nonempty_string(item_id):
            errors.append(f"{field}.id must be a non-empty string")
        errors.extend(_positive_line_errors(item.get("line"), f"{field}.line"))
        if item.get("causal_role") not in L1_RESPONSE_CONTRACT.related_causal_roles:
            errors.append(f"{field}.causal_role is invalid")
        if not _nonempty_string(item.get("rationale")):
            errors.append(f"{field}.rationale must be a non-empty string")
        identity = item.get("failure_identity")
        if not isinstance(identity, Mapping):
            errors.append(f"{field}.failure_identity must be an object")
        else:
            errors.extend(
                _object_shape_errors(
                    identity,
                    L1_RESPONSE_CONTRACT.failure_identity_fields,
                    f"{field}.failure_identity",
                )
            )
        refs = item.get("evidence_ids")
        if not isinstance(refs, list) or not refs or not all(_nonempty_string(ref) for ref in refs):
            errors.append(f"{field}.evidence_ids must be a non-empty string array")
    return errors


def _evidence_errors(evidence: Any) -> tuple[list[str], set[str]]:
    if not isinstance(evidence, list):
        return ["evidence must be an array"], set()
    errors: list[str] = []
    support_tags: set[str] = set()
    for index, item in enumerate(evidence):
        field = f"evidence[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{field} must be an object")
            continue
        errors.extend(_object_shape_errors(item, L1_RESPONSE_CONTRACT.evidence_fields, field))
        evidence_id = item.get("id")
        if not _nonempty_string(evidence_id):
            errors.append(f"{field}.id must be a non-empty string")
        errors.extend(_positive_line_errors(item.get("line"), f"{field}.line"))
        if not _nonempty_string(item.get("quote")):
            errors.append(f"{field}.quote must be a non-empty string")
        supports = item.get("supports")
        if not isinstance(supports, list) or not supports:
            errors.append(f"{field}.supports must be a non-empty array")
        elif not all(_nonempty_string(tag) for tag in supports):
            errors.append(f"{field}.supports items must be non-empty strings")
        else:
            support_tags.update(
                tag for tag in supports if tag in L1_RESPONSE_CONTRACT.evidence_support_tags
            )
    return errors, support_tags


def _non_primary_semantic_errors(
    payload: Mapping[str, Any],
    analysis_status: str,
) -> list[str]:
    errors: list[str] = []
    root_cause = payload.get("root_cause_assessment")
    assessment = payload.get("model_recovery_assessment")
    related = payload.get("related_failures")
    evidence = payload.get("evidence")
    observed = payload.get("observed_failures")
    selected_observation_id = payload.get("selected_observed_failure_id")
    if isinstance(root_cause, Mapping):
        if root_cause.get("status") != AssessmentStatus.UNKNOWN.value:
            errors.append("non-primary root_cause_assessment.status must be unknown")
        if root_cause.get("plausible_causes") != []:
            errors.append("non-primary root_cause_assessment.plausible_causes must be empty")
        missing = root_cause.get("missing_evidence")
        if analysis_status == L1AnalysisStatus.NO_FAILURE_OBSERVED.value and missing != []:
            errors.append("no_failure_observed missing_evidence must be empty")
        if (
            analysis_status == L1AnalysisStatus.INSUFFICIENT_EVIDENCE.value
            and isinstance(missing, list)
            and not missing
        ):
            errors.append("insufficient_evidence missing_evidence must not be empty")
    require_placeholder_recovery = not (
        analysis_status == L1AnalysisStatus.INSUFFICIENT_EVIDENCE.value
        and isinstance(selected_observation_id, str)
    )
    if isinstance(assessment, Mapping) and require_placeholder_recovery:
        for name in ("failure_domain", "retry_outlook_without_workload_change"):
            claim = assessment.get(name)
            if not isinstance(claim, Mapping):
                continue
            if claim.get("value") != "unknown" or claim.get("status") != "unknown":
                errors.append(f"non-primary {name} must use value=unknown and status=unknown")
    if related != []:
        errors.append("non-primary related_failures must be empty")
    if analysis_status == L1AnalysisStatus.NO_FAILURE_OBSERVED.value:
        if observed != [] or selected_observation_id is not None:
            errors.append("no_failure_observed forbids observed failures")
        if evidence != []:
            errors.append("no_failure_observed evidence must be empty")
    return errors


def _positive_line_errors(value: Any, field: str) -> list[str]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return [f"{field} must be a positive integer"]
    return []


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _category_selection_errors(value: Any) -> list[str]:
    """Validate the optional L1 category_selection block.

    The block is optional. When present, it must have all three curated fields
    (category_id, category_confidence, category_rationale). category_id must be
    an integer in the range [0, 38] (0 means "no listed category matches").
    category_confidence must be an integer in [0, 100]. category_rationale must
    be a non-empty string of at most max_category_rationale_chars characters.
    """

    if value is None:
        return []
    if not isinstance(value, Mapping):
        return ["category_selection must be an object"]
    errors = _object_shape_errors(
        value, L1_RESPONSE_CONTRACT.category_selection_fields, "category_selection"
    )
    cid = value.get("category_id")
    if isinstance(cid, bool) or not isinstance(cid, int) or cid < 0 or cid > 38:
        errors.append("category_selection.category_id must be an integer in [0, 38]")
    conf = value.get("category_confidence")
    if isinstance(conf, bool) or not isinstance(conf, int) or conf < 0 or conf > 100:
        errors.append("category_selection.category_confidence must be an integer in [0, 100]")
    rationale = value.get("category_rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        errors.append("category_selection.category_rationale must be a non-empty string")
    elif len(rationale) > L1_RESPONSE_CONTRACT.max_category_rationale_chars:
        errors.append(
            f"category_selection.category_rationale must be at most "
            f"{L1_RESPONSE_CONTRACT.max_category_rationale_chars} characters"
        )
    return errors
def repair_schema_version(payload: Any) -> list[str]:
    """In-place fix: normalize a similar-but-wrong schema_version string.

    Some models (observed: gemini) hallucinate a schema version like
    "restart_agent_decision_evidence.v1" instead of the expected
    "restart_agent_evidence.v1" - the "decision" substring is bleed-through
    from another schema name used elsewhere in the analyzer artifacts.

    Only rewrites when the payload is otherwise shaped like this schema
    (has an analysis_status field) and the current schema_version either
    is missing or starts with "restart_agent". This prevents silently
    accepting a truly foreign payload while forgiving cosmetic model errors.

    Returns a list of repair notes for observability. Empty list means no
    repair was applied.
    """

    notes: list[str] = []
    if not isinstance(payload, dict):
        return notes
    if payload.get("analysis_status") is None:
        # Not obviously our schema; do not touch.
        return notes
    actual = payload.get("schema_version")
    if actual == L1_EVIDENCE_SCHEMA_VERSION:
        return notes
    if actual is None:
        payload["schema_version"] = L1_EVIDENCE_SCHEMA_VERSION
        notes.append(f"schema_version: added missing field, set to {L1_EVIDENCE_SCHEMA_VERSION!r}")
    elif isinstance(actual, str) and actual.startswith("restart_agent"):
        payload["schema_version"] = L1_EVIDENCE_SCHEMA_VERSION
        notes.append(
            f"schema_version: normalized {actual!r} -> {L1_EVIDENCE_SCHEMA_VERSION!r} "
            f"(model hallucinated similar version string)"
        )
    return notes



def repair_biconditional_unknowns(payload: Any) -> list[str]:
    """In-place fix: normalize model_recovery_assessment claim (value, status) pairs.

    Contract rule: value == "unknown" iff status == "unknown". Some models
    (observed: nemotron) mis-read `unknown` as "not fully committed" and pair
    value=unknown with status=supported_but_unconfirmed / hypothesis_only. This
    repair silently normalizes such pairs to value=unknown / status=unknown
    (dropping any confidence hint on the claim to reflect the demotion), rather
    than rejecting the entire L1 response.

    Returns a list of repair notes for observability. Empty list means no
    repair was applied.
    """

    notes: list[str] = []
    if not isinstance(payload, dict):
        return notes
    assessment = payload.get("model_recovery_assessment")
    if not isinstance(assessment, dict):
        return notes
    for field in ("failure_domain", "retry_outlook_without_workload_change"):
        claim = assessment.get(field)
        if not isinstance(claim, dict):
            continue
        claim_value = claim.get("value")
        claim_status = claim.get("status")
        if (claim_value == "unknown") == (claim_status == AssessmentStatus.UNKNOWN.value):
            continue
        # Biconditional violated. Normalize both sides to unknown/unknown and
        # keep the claim structurally valid so downstream validation passes.
        claim["value"] = "unknown"
        claim["status"] = AssessmentStatus.UNKNOWN.value
        if "confidence" in claim and not (
            isinstance(claim["confidence"], int) and not isinstance(claim["confidence"], bool)
        ):
            # keep confidence field valid; the model's number stays if it was a legal int
            claim["confidence"] = L1_RESPONSE_CONTRACT.min_confidence
        notes.append(
            f"model_recovery_assessment.{field}: normalized "
            f"(value={claim_value!r}, status={claim_status!r}) -> unknown/unknown"
        )
    return notes



def repair_overlong_lists(payload: Any) -> list[str]:
    """No-op under PR #400: over-long lists are advisory, not rejected.

    PR #390 rejected responses whose plausible_causes / missing_evidence /
    related_failures exceeded fixed caps. PR #400 downgraded these to
    "recommended" limits (L1 contract advisories) and does NOT reject the
    response for exceeding them. This repair therefore has nothing to do
    under PR #400 and returns an empty note list.
    """

    return []



def repair_invalid_evidence_supports(payload: Any) -> list[str]:
    """In-place fix: drop unknown/invalid tags from evidence[N].supports arrays.

    The contract restricts evidence.supports to four tags: primary_failure,
    root_cause_assessment, failure_domain, retry_outlook_without_workload_change.
    Some models (observed: nemotron) mistakenly use a section-name like
    "related_failures" as a support tag. Rather than reject the whole response,
    drop the invalid tags and keep the valid ones. If an item ends up with an
    empty supports array after filtering, drop the item entirely.

    Returns a list of repair notes for observability. Empty list means no
    repair was applied.
    """

    notes: list[str] = []
    if not isinstance(payload, dict):
        return notes
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        return notes
    valid_tags = L1_RESPONSE_CONTRACT.evidence_support_tags
    kept: list[Any] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            kept.append(item)
            continue
        supports = item.get("supports")
        if not isinstance(supports, list):
            kept.append(item)
            continue
        cleaned: list[str] = []
        seen: set[str] = set()
        for tag in supports:
            if isinstance(tag, str) and tag in valid_tags and tag not in seen:
                cleaned.append(tag)
                seen.add(tag)
        if cleaned == list(supports):
            kept.append(item)
            continue
        dropped = [t for t in supports if t not in cleaned]
        if not cleaned:
            notes.append(
                f"evidence[{index}]: dropped entirely (no valid supports left; "
                f"originally {supports!r})"
            )
            continue
        item["supports"] = cleaned
        notes.append(f"evidence[{index}]: dropped invalid supports {dropped!r}, kept {cleaned!r}")
        kept.append(item)
    if len(kept) != len(evidence):
        payload["evidence"] = kept
    return notes



def repair_overlong_category_rationale(payload: Any) -> list[str]:
    """In-place fix: truncate category_selection.category_rationale to the contract cap.

    Some models produce a rationale longer than max_category_rationale_chars (400).
    Truncating preserves the leading, most-relevant content rather than rejecting
    the whole response.

    Returns a list of repair notes for observability. Empty list means no
    repair was applied.
    """

    notes: list[str] = []
    if not isinstance(payload, dict):
        return notes
    selection = payload.get("category_selection")
    if not isinstance(selection, dict):
        return notes
    rationale = selection.get("category_rationale")
    max_chars = L1_RESPONSE_CONTRACT.max_category_rationale_chars
    if isinstance(rationale, str) and len(rationale) > max_chars:
        selection["category_rationale"] = rationale[:max_chars]
        notes.append(
            f"category_selection.category_rationale: truncated "
            f"{len(rationale)} -> {max_chars} chars"
        )
    return notes



def repair_model_evidence(payload: Any) -> list[str]:
    """Aggregate all in-place repairs. Returns the concatenated repair notes.

    Note: repair_invalid_evidence_supports is intentionally NOT called here.
    PR #400 already surfaces unknown/duplicate support tags as advisories
    (evidence_support_tag_unknown, evidence_supports_contain_duplicates in
    l1/advisories.py) without rejecting the response. Pre-emptively dropping
    those tags in-place would suppress those advisories. The repair function
    is kept in this module for reference and possible future use.
    """

    notes: list[str] = []
    notes.extend(repair_schema_version(payload))
    notes.extend(repair_biconditional_unknowns(payload))
    notes.extend(repair_overlong_lists(payload))
    notes.extend(repair_overlong_category_rationale(payload))
    return notes


