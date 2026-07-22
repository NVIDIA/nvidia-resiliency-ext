# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build typed branch-specific failure facts for L3 history comparison."""

from __future__ import annotations

from typing import Any, Mapping

from ..models import (
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
        fine_class=primary.fine_class if primary is not None else None,
        root_fingerprint=root_fingerprint,
        root_fingerprint_source=root_fingerprint_source,
        fault_outcome=primary.fault_outcome if primary is not None else None,
        primary_line=primary.line if primary is not None else None,
        identity_anchor_line=identity_anchor_line,
        identity_anchor_reason=identity_anchor_reason,
        failure_iteration=primary.failure_iteration if primary is not None else None,
        data_position_fingerprint=(
            primary.data_position_fingerprint if primary is not None else None
        ),
        artifact_path=_artifact_path(primary),
        faulting_rank=primary.rank if primary is not None else None,
        faulting_node=primary.node if primary is not None else None,
        faulting_gpu=primary.gpu if primary is not None else None,
    )


def _artifact_path(primary: FailureEvidence | None) -> str | None:
    if primary is None or not isinstance(primary.failure_identity, Mapping):
        return None
    for section_name in ("client_concrete", "concrete"):
        section = primary.failure_identity.get(section_name)
        if isinstance(section, Mapping):
            artifact = _optional_str(section.get("artifact_path"))
            if artifact is not None:
                return artifact
    return None


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
