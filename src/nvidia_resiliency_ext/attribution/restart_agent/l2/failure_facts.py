# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build typed branch-specific failure facts for L3 history comparison."""

from __future__ import annotations

from typing import Any

from ..identity import build_affected_entity, extract_data_position_identity
from ..models import (
    AffectedEntity,
    AffectedEntityKind,
    AttemptFailureFacts,
    AttemptFailureFactsSource,
    DecisionEvidence,
    FailureEvidence,
)


def build_attempt_failure_facts(
    primary: FailureEvidence | None,
    decision_evidence: DecisionEvidence,
    *,
    source: AttemptFailureFactsSource,
    identity_anchor_line: int | None = None,
    identity_anchor_reason: str | None = None,
) -> AttemptFailureFacts:
    """Build the shared branch-specific failure-facts contract."""

    canonical_identity = decision_evidence.canonical_observed_identity
    if source == AttemptFailureFactsSource.L0_DETERMINISTIC:
        root_fingerprint = _optional_str(canonical_identity.get("root_fingerprint"))
        root_fingerprint_source = _optional_str(canonical_identity.get("root_fingerprint_source"))
        identity_anchor_line = _optional_int(canonical_identity.get("identity_anchor_line"))
        identity_anchor_reason = _optional_str(canonical_identity.get("identity_anchor_reason"))
    else:
        root_fingerprint = primary.root_fingerprint if primary is not None else None
        root_fingerprint_source = primary.root_fingerprint_source if primary is not None else None

    return AttemptFailureFacts(
        source=source,
        root_fingerprint=root_fingerprint,
        root_fingerprint_source=root_fingerprint_source,
        fault_outcome=primary.fault_outcome if primary is not None else None,
        primary_line=primary.line if primary is not None else None,
        identity_anchor_line=identity_anchor_line,
        identity_anchor_reason=identity_anchor_reason,
        failure_iteration=primary.failure_iteration if primary is not None else None,
        affected_entity=_affected_entity(primary),
        faulting_rank=primary.rank if primary is not None else None,
        faulting_node=primary.node if primary is not None else None,
        faulting_gpu=primary.gpu if primary is not None else None,
    )


def _affected_entity(primary: FailureEvidence | None) -> AffectedEntity | None:
    if primary is None:
        return None
    if primary.affected_entity is not None:
        return primary.affected_entity
    data_position = _data_position_identity(primary)
    if data_position is not None:
        return build_affected_entity(
            AffectedEntityKind.DATA_POSITION,
            data_position,
            evidence_line=primary.line,
        )
    return None


def build_grounded_affected_entity(
    *,
    quote: str | None,
    signature: str | None,
    data_position_fingerprint: str | None,
    artifact_path: str | None,
    evidence_line: int | None,
) -> AffectedEntity | None:
    """Build an exact entity from source-grounded current-failure evidence."""

    data_position = None
    for text in (quote, signature):
        if text:
            data_position = extract_data_position_identity(text)
            if data_position is not None:
                break
    if data_position is None:
        data_position = data_position_fingerprint
    if data_position is not None:
        return build_affected_entity(
            AffectedEntityKind.DATA_POSITION,
            data_position,
            evidence_line=evidence_line,
        )
    if artifact_path is None:
        return None
    normalized_path = artifact_path.rstrip("/") or "/"
    return build_affected_entity(
        AffectedEntityKind.ARTIFACT,
        normalized_path,
        evidence_line=evidence_line,
    )


def _data_position_identity(primary: FailureEvidence) -> str | None:
    for text in (primary.quote, primary.signature):
        if text:
            identity = extract_data_position_identity(text)
            if identity is not None:
                return identity
    return primary.data_position_fingerprint


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
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
