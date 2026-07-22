# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for the L1 model response contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models import (
    L1_EVIDENCE_SCHEMA_VERSION,
    AssessmentStatus,
    CausalRole,
    FailureDomain,
    L1AnalysisStatus,
    RetryOutlookWithoutWorkloadChange,
)

PRIMARY_FAILURE_SUPPORT_TAG = "primary_failure"
ROOT_CAUSE_SUPPORT_TAG = "root_cause_assessment"
FAILURE_DOMAIN_SUPPORT_TAG = "failure_domain"
RETRY_OUTLOOK_SUPPORT_TAG = "retry_outlook_without_workload_change"


@dataclass(frozen=True)
class L1ResponseContract:
    """Closed L1 shape, limits, enums, and model-visible description."""

    top_level_fields: frozenset[str] = frozenset(
        {
            "schema_version",
            "analysis_status",
            "primary_failure",
            "root_cause_assessment",
            "model_recovery_assessment",
            "related_failures",
            "evidence",
        }
    )
    primary_failure_fields: frozenset[str] = frozenset({"line", "causal_role", "failure_identity"})
    failure_identity_fields: frozenset[str] = frozenset(
        {"operation", "mechanism", "component", "artifact_path"}
    )
    root_cause_fields: frozenset[str] = frozenset(
        {"summary", "status", "plausible_causes", "missing_evidence"}
    )
    recovery_assessment_fields: frozenset[str] = frozenset(
        {"failure_domain", "retry_outlook_without_workload_change", "rationale"}
    )
    claim_fields: frozenset[str] = frozenset({"value", "status", "confidence"})
    related_failure_fields: frozenset[str] = frozenset({"line", "causal_role", "rationale"})
    evidence_fields: frozenset[str] = frozenset({"id", "line", "quote", "supports"})
    evidence_support_tags: frozenset[str] = frozenset(
        {
            PRIMARY_FAILURE_SUPPORT_TAG,
            ROOT_CAUSE_SUPPORT_TAG,
            FAILURE_DOMAIN_SUPPORT_TAG,
            RETRY_OUTLOOK_SUPPORT_TAG,
        }
    )
    max_plausible_causes: int = 3
    max_missing_evidence: int = 5
    max_related_failures: int = 3
    max_evidence_items: int = 12
    max_evidence_id_chars: int = 64
    min_confidence: int = 1
    max_confidence: int = 99
    non_primary_confidence: int = 1
    require_unique_evidence_ids: bool = True
    no_failure_summary: str = "No failure was observed in the supplied evidence."
    no_failure_rationale: str = "Recovery is not assessed because no failure was observed."
    insufficient_summary: str = "Insufficient evidence to identify a primary failure."
    insufficient_rationale: str = "Recovery is not assessed without an identified primary failure."

    @property
    def analysis_statuses(self) -> frozenset[str]:
        return frozenset(item.value for item in L1AnalysisStatus)

    @property
    def causal_roles(self) -> frozenset[str]:
        return frozenset(item.value for item in CausalRole)

    @property
    def related_causal_roles(self) -> frozenset[str]:
        return frozenset(
            {
                CausalRole.CASCADE.value,
                CausalRole.TEARDOWN.value,
                CausalRole.UNKNOWN.value,
            }
        )

    @property
    def failure_domains(self) -> frozenset[str]:
        return frozenset(item.value for item in FailureDomain)

    @property
    def retry_outlooks(self) -> frozenset[str]:
        return frozenset(item.value for item in RetryOutlookWithoutWorkloadChange)

    @property
    def assessment_statuses(self) -> frozenset[str]:
        return frozenset(item.value for item in AssessmentStatus)

    @property
    def required_primary_support_tags(self) -> frozenset[str]:
        return self.evidence_support_tags

    def model_schema(self) -> dict[str, Any]:
        """Return the complete contract advertised in the initial model request."""

        claim_statuses = sorted(self.assessment_statuses)

        def claim(values: frozenset[str]) -> dict[str, Any]:
            return {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(self.claim_fields),
                "properties": {
                    "value": {"type": "string", "enum": sorted(values)},
                    "status": {"type": "string", "enum": claim_statuses},
                    "confidence": {
                        "type": "integer",
                        "minimum": self.min_confidence,
                        "maximum": self.max_confidence,
                        "description": "Calibration-only confidence in this claim.",
                    },
                },
                "semanticConstraint": (
                    "value=unknown if and only if status=unknown; " "otherwise neither is unknown"
                ),
            }

        failure_identity = {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(self.failure_identity_fields),
            "properties": {
                name: {"type": ["string", "null"]} for name in sorted(self.failure_identity_fields)
            },
        }
        primary_failure = {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(self.primary_failure_fields),
            "properties": {
                "line": {"type": "integer", "minimum": 1},
                "causal_role": {"type": "string", "enum": sorted(self.causal_roles)},
                "failure_identity": failure_identity,
            },
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(self.top_level_fields),
            "properties": {
                "schema_version": {"type": "string", "const": L1_EVIDENCE_SCHEMA_VERSION},
                "analysis_status": {
                    "type": "string",
                    "enum": sorted(self.analysis_statuses),
                },
                "primary_failure": {"oneOf": [primary_failure, {"type": "null"}]},
                "root_cause_assessment": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(self.root_cause_fields),
                    "properties": {
                        "summary": {"type": "string", "minLength": 1},
                        "status": {"type": "string", "enum": claim_statuses},
                        "plausible_causes": {
                            "type": "array",
                            "maxItems": self.max_plausible_causes,
                            "items": {"type": "string", "minLength": 1},
                        },
                        "missing_evidence": {
                            "type": "array",
                            "maxItems": self.max_missing_evidence,
                            "items": {"type": "string", "minLength": 1},
                        },
                    },
                },
                "model_recovery_assessment": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(self.recovery_assessment_fields),
                    "properties": {
                        "failure_domain": claim(self.failure_domains),
                        "retry_outlook_without_workload_change": claim(self.retry_outlooks),
                        "rationale": {"type": "string", "minLength": 1},
                    },
                },
                "related_failures": {
                    "type": "array",
                    "maxItems": self.max_related_failures,
                    "description": (
                        "Grounded source-line references for diagnostic relationships; "
                        "they are not canonical policy-claim citations."
                    ),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": sorted(self.related_failure_fields),
                        "properties": {
                            "line": {"type": "integer", "minimum": 1},
                            "causal_role": {
                                "type": "string",
                                "enum": sorted(self.related_causal_roles),
                            },
                            "rationale": {"type": "string", "minLength": 1},
                        },
                    },
                },
                "evidence": {
                    "type": "array",
                    "maxItems": self.max_evidence_items,
                    "description": "Canonical citations for the four policy-relevant claims.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": sorted(self.evidence_fields),
                        "properties": {
                            "id": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": self.max_evidence_id_chars,
                            },
                            "line": {"type": "integer", "minimum": 1},
                            "quote": {"type": "string", "minLength": 1},
                            "supports": {
                                "type": "array",
                                "minItems": 1,
                                "uniqueItems": True,
                                "items": {
                                    "type": "string",
                                    "enum": sorted(self.evidence_support_tags),
                                },
                            },
                        },
                    },
                },
            },
            "semanticConstraints": {
                "evidence_ids": "unique non-empty strings",
                "primary_identified": {
                    "primary_failure": "required object",
                    "evidence": ("non-empty and collectively supports every evidence support tag"),
                },
                "no_failure_observed": {
                    "primary_failure": None,
                    "root_cause_summary": self.no_failure_summary,
                    "root_cause_status": "unknown",
                    "plausible_causes": [],
                    "missing_evidence": [],
                    "recovery_claims": "unknown value, unknown status, confidence 1",
                    "recovery_rationale": self.no_failure_rationale,
                    "related_failures": [],
                    "evidence": [],
                },
                "insufficient_evidence": {
                    "primary_failure": None,
                    "root_cause_summary": self.insufficient_summary,
                    "root_cause_status": "unknown",
                    "plausible_causes": [],
                    "missing_evidence": "one or more strings",
                    "recovery_claims": "unknown value, unknown status, confidence 1",
                    "recovery_rationale": self.insufficient_rationale,
                    "related_failures": [],
                    "evidence": [],
                },
            },
        }


L1_RESPONSE_CONTRACT = L1ResponseContract()


def model_response_schema() -> dict[str, Any]:
    """Return the model-visible response schema from the canonical contract."""

    return L1_RESPONSE_CONTRACT.model_schema()
