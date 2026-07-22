"""Validation for human-approved restart-agent eval labels."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any


class GoldSchemaError(ValueError):
    """Raised when a gold label contains unsupported schema vocabulary."""


HUMAN_REVIEW_STATUSES = frozenset(
    {
        "human_approved",
        "human_reviewed_ambiguous_rca",
        "human_reviewed_supported_but_unconfirmed_rca",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "case_id",
        "label_version",
        "review_status",
        "source",
        "source_sha256",
        "decision",
        "l0_expectation",
        "l0b_expectation",
        "primary_anchor_expectation",
        "root_cause_expectation",
        "recovery_assessment_expectation",
        "retry_policy_expectation",
        "action_expectation",
        "causal_role_expectation",
        "cascade_expectation",
        "l2_audit_expectation",
        "history_identity_expectation",
        "unsupported_claims",
        "human_assessment",
        "notes",
    }
)

L0_EXPECTATION_FIELDS = frozenset(
    {
        "required_setup_marker_types",
        "minimum_setup_marker_count",
        "required_coverage",
        "accepted_primary_lines",
        "line_tolerance",
        "accepted_root_fingerprints",
        "required_progress_lines",
        "required_checkpoint_lines",
        "expected_primary_phase",
        "expected_checkpoint_load_iteration",
        "expected_progress_after_failure_episode",
        "required_cascade_lines",
        "required_operation_artifact_comparisons",
    }
)

L0_COVERAGE_FIELDS = frozenset(
    {
        "application_progress",
        "candidate_anchors",
        "cascade",
        "checkpoint_progress",
        "context_windows",
        "deterministic_taxonomy_primary",
        "distributed_failure_incidents",
        "first_failure_candidate",
        "job_metadata",
        "observed_failure_iteration",
        "occurrence_groups",
        "operation_artifact_comparisons",
        "path_access_facts",
        "path_hints",
        "progress_segments",
        "setup_progress",
    }
)

OPERATION_COMPARISON_FIELDS = frozenset(
    {
        "operation",
        "logical_artifact_id",
        "physical_unit_id",
        "minimum_success_count",
        "current_outcome",
        "comparison_level",
    }
)

OBJECT_FIELDS = MappingProxyType(
    {
        "l0b_expectation": frozenset(
            {
                "required_evidence_lines",
                "accepted_primary_lines",
                "line_tolerance",
                "required_reference_ids",
            }
        ),
        "primary_anchor_expectation": frozenset(
            {"accepted_lines", "rejected_downstream_lines", "tolerance_lines"}
        ),
        "root_cause_expectation": frozenset(
            {
                "observed_mechanism",
                "required_concept_groups",
                "require_uncertainty_preserved",
                "uncertainty_terms_any",
                "accepted_interpretation",
                "accepted_operations",
                "rejected_mechanism_terms",
            }
        ),
        "recovery_assessment_expectation": frozenset(
            {
                "failure_domain",
                "failure_domain_status",
                "retry_outlook_without_workload_change",
                "retry_outlook_status",
            }
        ),
        "retry_policy_expectation": frozenset(
            {"accepted_rules", "allowed_retries", "retry_budget_exhausted"}
        ),
        "action_expectation": frozenset({"accepted", "scope"}),
        "causal_role_expectation": frozenset({"accepted_primary_roles", "rejected_primary_roles"}),
        "cascade_expectation": frozenset(
            {"expected_lines", "expected_groups", "teardown_lines", "rejected_as_primary"}
        ),
        "history_identity_expectation": frozenset(
            {
                "operation",
                "mechanism",
                "canonical_anchor_line",
                "same_episode_lines",
                "expected_cross_route_identity_count",
            }
        ),
    }
)

L0B_REFERENCE_FIELDS = frozenset(
    {
        "context_window_ids",
        "candidate_anchor_ids",
        "occurrence_group_ids",
        "failure_episode_ids",
        "distributed_incident_ids",
    }
)
L2_AUDIT_FIELDS = frozenset({"field", "expected", "normalized_value", "reason_class"})
UNSUPPORTED_CLAIM_FIELDS = frozenset({"id", "text_patterns", "policy_classes"})
CASCADE_GROUP_FIELDS = frozenset(
    {"causal_role", "first_line", "minimum_count", "minimum_rank_count"}
)


def validate_gold_label(value: Mapping[str, Any]) -> None:
    """Reject unsupported scored-label fields instead of silently ignoring them."""

    _validate_keys(value, TOP_LEVEL_FIELDS, "gold")

    l0 = _optional_object(value, "l0_expectation", "gold")
    if l0 is not None:
        _validate_keys(l0, L0_EXPECTATION_FIELDS, "gold.l0_expectation")
        coverage = _optional_object(l0, "required_coverage", "gold.l0_expectation")
        if coverage is not None:
            _validate_keys(
                coverage,
                L0_COVERAGE_FIELDS,
                "gold.l0_expectation.required_coverage",
            )
        _validate_object_list(
            l0.get("required_operation_artifact_comparisons"),
            OPERATION_COMPARISON_FIELDS,
            "gold.l0_expectation.required_operation_artifact_comparisons",
        )

    for field, allowed in OBJECT_FIELDS.items():
        item = _optional_object(value, field, "gold")
        if item is None:
            continue
        _validate_keys(item, allowed, f"gold.{field}")
        if field == "l0b_expectation":
            references = _optional_object(item, "required_reference_ids", f"gold.{field}")
            if references is not None:
                _validate_keys(
                    references,
                    L0B_REFERENCE_FIELDS,
                    f"gold.{field}.required_reference_ids",
                )
        if field == "cascade_expectation":
            _validate_object_list(
                item.get("expected_groups"),
                CASCADE_GROUP_FIELDS,
                "gold.cascade_expectation.expected_groups",
            )

    _validate_object_list(
        value.get("l2_audit_expectation"),
        L2_AUDIT_FIELDS,
        "gold.l2_audit_expectation",
    )
    _validate_object_list(
        value.get("unsupported_claims"),
        UNSUPPORTED_CLAIM_FIELDS,
        "gold.unsupported_claims",
    )


def validate_scored_gold_label(value: Mapping[str, Any]) -> None:
    """Validate fields that make a label eligible for semantic scoring."""

    validate_gold_label(value)
    review_status = value.get("review_status")
    if review_status not in HUMAN_REVIEW_STATUSES:
        raise GoldSchemaError(
            "gold.review_status must be one of: " + ", ".join(sorted(HUMAN_REVIEW_STATUSES))
        )
    source_sha256 = value.get("source_sha256")
    if not isinstance(source_sha256, str) or not SHA256_RE.fullmatch(source_sha256):
        raise GoldSchemaError("gold.source_sha256 must be a lowercase SHA-256 digest")


def source_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a source log without loading the whole file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gold_source(value: Mapping[str, Any], source_log: Path) -> None:
    """Reject a reviewed label when its source bytes no longer match."""

    expected = str(value["source_sha256"])
    actual = source_sha256(source_log)
    if actual != expected:
        raise GoldSchemaError(
            f"gold.source_sha256 does not match {source_log}: expected {expected}, got {actual}"
        )


def _optional_object(
    value: Mapping[str, Any], field: str, parent_path: str
) -> Mapping[str, Any] | None:
    item = value.get(field)
    if item is None:
        return None
    if not isinstance(item, Mapping):
        raise GoldSchemaError(f"{parent_path}.{field} must be an object")
    return item


def _validate_object_list(value: Any, allowed: frozenset[str], path: str) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise GoldSchemaError(f"{path} must be a list")
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise GoldSchemaError(f"{path}[{index}] must be an object")
        _validate_keys(item, allowed, f"{path}[{index}]")


def _validate_keys(value: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        fields = ", ".join(unknown)
        raise GoldSchemaError(f"{path} contains unsupported field(s): {fields}")
