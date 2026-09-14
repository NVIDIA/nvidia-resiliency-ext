# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-blocking quality findings for structurally usable L1 responses."""

from __future__ import annotations

from typing import Any, Mapping

from ..models import L1AnalysisStatus
from .normalization import normalize_model_evidence_payload
from .response_contract import L1_RESPONSE_CONTRACT


def model_evidence_contract_advisories(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return non-blocking response-contract quality findings."""

    advisories: list[dict[str, Any]] = []
    _, ignored_fields = normalize_model_evidence_payload(payload)
    if ignored_fields:
        advisories.append(
            _advisory(
                code="unknown_response_fields_ignored",
                field="model_response",
                message="unknown response fields are ignored by the L1 semantic contract",
                field_paths=list(ignored_fields),
            )
        )
    root_cause = payload.get("root_cause_assessment")
    if isinstance(root_cause, Mapping):
        _append_collection_limit_advisory(
            advisories,
            root_cause.get("plausible_causes"),
            code="plausible_causes_exceeds_recommended_limit",
            field="root_cause_assessment.plausible_causes",
            recommended_maximum=L1_RESPONSE_CONTRACT.recommended_max_plausible_causes,
        )
        _append_collection_limit_advisory(
            advisories,
            root_cause.get("missing_evidence"),
            code="missing_evidence_exceeds_recommended_limit",
            field="root_cause_assessment.missing_evidence",
            recommended_maximum=L1_RESPONSE_CONTRACT.recommended_max_missing_evidence,
        )

    observed_failures = payload.get("observed_failures")
    _append_collection_limit_advisory(
        advisories,
        observed_failures,
        code="observed_failures_exceeds_recommended_limit",
        field="observed_failures",
        recommended_maximum=L1_RESPONSE_CONTRACT.recommended_max_observed_failures,
    )
    _append_long_id_advisory(
        advisories,
        observed_failures,
        code="observed_failure_id_exceeds_recommended_length",
        field="observed_failures[].id",
    )
    _append_duplicate_array_value_advisory(
        advisories,
        observed_failures,
        item_field="evidence_ids",
        code="observed_failure_evidence_ids_contain_duplicates",
        field="observed_failures[].evidence_ids",
        message="observation citation identifiers should be unique",
    )
    observed_id_counts = _identifier_counts(observed_failures)
    _append_duplicate_object_id_advisory(
        advisories,
        observed_id_counts,
        code="duplicate_observed_failure_id",
        field="observed_failures[].id",
        message="observed-failure identifiers should be unique",
    )
    selected_observation_id = payload.get("selected_observed_failure_id")
    if isinstance(selected_observation_id, str):
        match_count = observed_id_counts.get(selected_observation_id, 0)
        if match_count == 0:
            advisories.append(
                _advisory(
                    code="selected_observation_id_unresolved",
                    field="selected_observed_failure_id",
                    message="selected observation id does not resolve to an observed failure",
                    selected_id=selected_observation_id,
                    match_count=0,
                )
            )
        elif match_count > 1:
            advisories.append(
                _advisory(
                    code="selected_observation_id_ambiguous",
                    field="selected_observed_failure_id",
                    message="selected observation id resolves to multiple observed failures",
                    selected_id=selected_observation_id,
                    match_count=match_count,
                )
            )

    _append_collection_limit_advisory(
        advisories,
        payload.get("related_failures"),
        code="related_failures_exceeds_recommended_limit",
        field="related_failures",
        recommended_maximum=L1_RESPONSE_CONTRACT.recommended_max_related_failures,
    )

    evidence = payload.get("evidence")
    _append_collection_limit_advisory(
        advisories,
        evidence,
        code="evidence_exceeds_recommended_limit",
        field="evidence",
        recommended_maximum=L1_RESPONSE_CONTRACT.recommended_max_evidence_items,
    )
    _append_long_id_advisory(
        advisories,
        evidence,
        code="evidence_id_exceeds_recommended_length",
        field="evidence[].id",
    )
    _append_duplicate_array_value_advisory(
        advisories,
        evidence,
        item_field="supports",
        code="evidence_supports_contain_duplicates",
        field="evidence[].supports",
        message="evidence support tags should be unique within each citation",
    )
    _append_unknown_evidence_support_tag_advisory(advisories, evidence)
    _append_duplicate_object_id_advisory(
        advisories,
        _identifier_counts(evidence),
        code="duplicate_evidence_id",
        field="evidence[].id",
        message="evidence identifiers should be unique",
    )
    support_tags = _evidence_support_tags(evidence)
    if payload.get("analysis_status") == L1AnalysisStatus.PRIMARY_IDENTIFIED.value:
        for required_support in sorted(L1_RESPONSE_CONTRACT.required_primary_evidence_support_tags):
            if required_support not in support_tags:
                advisories.append(
                    _advisory(
                        code=f"{required_support}_support_missing",
                        field="evidence[].supports",
                        message=f"no citation declares support for {required_support}",
                        missing_support_tag=required_support,
                    )
                )

    _append_non_primary_presentation_advisories(advisories, payload)
    return advisories


def _identifier_counts(value: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(value, list):
        return counts
    for item in value:
        if not isinstance(item, Mapping):
            continue
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id.strip():
            counts[item_id] = counts.get(item_id, 0) + 1
    return counts


def _append_duplicate_object_id_advisory(
    advisories: list[dict[str, Any]],
    counts: Mapping[str, int],
    *,
    code: str,
    field: str,
    message: str,
) -> None:
    duplicate_ids = sorted(item_id for item_id, count in counts.items() if count > 1)
    if not duplicate_ids:
        return
    advisories.append(
        _advisory(
            code=code,
            field=field,
            message=message,
            duplicate_ids=duplicate_ids,
        )
    )


def _evidence_support_tags(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {
        tag
        for item in value
        if isinstance(item, Mapping) and isinstance(item.get("supports"), list)
        for tag in item["supports"]
        if isinstance(tag, str) and tag in L1_RESPONSE_CONTRACT.evidence_support_tags
    }


def _append_unknown_evidence_support_tag_advisory(
    advisories: list[dict[str, Any]],
    evidence: Any,
) -> None:
    if not isinstance(evidence, list):
        return
    unknown_tags: set[str] = set()
    item_indexes: list[int] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            continue
        supports = item.get("supports")
        if not isinstance(supports, list):
            continue
        item_unknown_tags = {
            tag
            for tag in supports
            if isinstance(tag, str)
            and tag.strip()
            and tag not in L1_RESPONSE_CONTRACT.evidence_support_tags
        }
        if item_unknown_tags:
            item_indexes.append(index)
            unknown_tags.update(item_unknown_tags)
    if not unknown_tags:
        return
    advisories.append(
        _advisory(
            code="evidence_support_tag_unknown",
            field="evidence[].supports",
            message=("unknown evidence support tags are ignored for claim-support accounting"),
            item_indexes=item_indexes,
            unknown_tags=sorted(unknown_tags),
        )
    )


def _append_collection_limit_advisory(
    advisories: list[dict[str, Any]],
    value: Any,
    *,
    code: str,
    field: str,
    recommended_maximum: int,
) -> None:
    if not isinstance(value, list) or len(value) <= recommended_maximum:
        return
    label = field.rsplit(".", 1)[-1]
    advisories.append(
        _advisory(
            code=code,
            field=field,
            message=(
                f"{label} contains {len(value)} items; "
                f"the recommended maximum is {recommended_maximum}"
            ),
            observed_count=len(value),
            recommended_maximum=recommended_maximum,
        )
    )


def _append_duplicate_array_value_advisory(
    advisories: list[dict[str, Any]],
    value: Any,
    *,
    item_field: str,
    code: str,
    field: str,
    message: str,
) -> None:
    if not isinstance(value, list):
        return
    item_indexes = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            continue
        items = item.get(item_field)
        if not isinstance(items, list) or not all(isinstance(entry, str) for entry in items):
            continue
        if len(set(items)) != len(items):
            item_indexes.append(index)
    if item_indexes:
        advisories.append(
            _advisory(
                code=code,
                field=field,
                message=message,
                item_indexes=item_indexes,
            )
        )


def _append_long_id_advisory(
    advisories: list[dict[str, Any]],
    value: Any,
    *,
    code: str,
    field: str,
) -> None:
    if not isinstance(value, list):
        return
    recommended_maximum = L1_RESPONSE_CONTRACT.recommended_max_evidence_id_chars
    item_indexes = [
        index
        for index, item in enumerate(value)
        if isinstance(item, Mapping)
        and isinstance(item.get("id"), str)
        and len(item["id"]) > recommended_maximum
    ]
    if not item_indexes:
        return
    advisories.append(
        _advisory(
            code=code,
            field=field,
            message=f"identifier length exceeds the recommended maximum of {recommended_maximum}",
            item_indexes=item_indexes,
            recommended_maximum=recommended_maximum,
        )
    )


def _append_non_primary_presentation_advisories(
    advisories: list[dict[str, Any]],
    payload: Mapping[str, Any],
) -> None:
    analysis_status = payload.get("analysis_status")
    if analysis_status not in {
        L1AnalysisStatus.NO_FAILURE_OBSERVED.value,
        L1AnalysisStatus.INSUFFICIENT_EVIDENCE.value,
    }:
        return
    if analysis_status == L1AnalysisStatus.NO_FAILURE_OBSERVED.value:
        expected_summary = L1_RESPONSE_CONTRACT.no_failure_summary
        expected_rationale = L1_RESPONSE_CONTRACT.no_failure_rationale
    else:
        expected_summary = L1_RESPONSE_CONTRACT.insufficient_summary
        expected_rationale = L1_RESPONSE_CONTRACT.insufficient_rationale
    root_cause = payload.get("root_cause_assessment")
    if isinstance(root_cause, Mapping) and root_cause.get("summary") != expected_summary:
        advisories.append(
            _advisory(
                code="non_primary_summary_not_canonical",
                field="root_cause_assessment.summary",
                message="non-primary summary differs from the canonical placeholder",
            )
        )
    selected_observation_id = payload.get("selected_observed_failure_id")
    require_placeholder_recovery = not (
        analysis_status == L1AnalysisStatus.INSUFFICIENT_EVIDENCE.value
        and isinstance(selected_observation_id, str)
    )
    assessment = payload.get("model_recovery_assessment")
    if not isinstance(assessment, Mapping) or not require_placeholder_recovery:
        return
    if assessment.get("rationale") != expected_rationale:
        advisories.append(
            _advisory(
                code="non_primary_recovery_rationale_not_canonical",
                field="model_recovery_assessment.rationale",
                message="non-primary rationale differs from the canonical placeholder",
            )
        )
    for name in ("failure_domain", "retry_outlook_without_workload_change"):
        claim = assessment.get(name)
        if (
            isinstance(claim, Mapping)
            and claim.get("confidence") != L1_RESPONSE_CONTRACT.non_primary_confidence
        ):
            advisories.append(
                _advisory(
                    code="non_primary_confidence_not_canonical",
                    field=f"model_recovery_assessment.{name}.confidence",
                    message=(
                        "unknown non-primary confidence differs from the canonical "
                        f"value {L1_RESPONSE_CONTRACT.non_primary_confidence}"
                    ),
                    observed_value=claim.get("confidence"),
                    recommended_value=L1_RESPONSE_CONTRACT.non_primary_confidence,
                )
            )


def _advisory(*, code: str, field: str, message: str, **details: Any) -> dict[str, Any]:
    return {
        "code": code,
        "field": field,
        "message": message,
        **details,
        "observational_only": True,
    }
