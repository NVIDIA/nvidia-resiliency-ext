# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build stage-neutral current-attempt failure facts for history comparison."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .models import (
    AffectedEntity,
    AttemptFailureFacts,
    AttemptFailureFactsSource,
    DecisionEvidence,
    FailureEvidence,
    HistoryIdentityKind,
)


def build_attempt_failure_facts(
    primary: FailureEvidence | None,
    decision_evidence: DecisionEvidence,
    *,
    source: AttemptFailureFactsSource,
    identity_anchor_line: int | None = None,
    identity_anchor_reason: str | None = None,
    root_locality: Mapping[str, Any] | None = None,
    selected_observation: FailureEvidence | None = None,
) -> AttemptFailureFacts:
    """Build the shared branch-specific failure-facts contract."""

    canonical_identity = decision_evidence.canonical_observed_identity
    if primary is not None and selected_observation is not None:
        raise ValueError("primary and selected observation are mutually exclusive")
    selected = primary or selected_observation
    if source == AttemptFailureFactsSource.L0_DETERMINISTIC:
        root_fingerprint = _optional_str(canonical_identity.get("root_fingerprint"))
        root_fingerprint_source = _optional_str(canonical_identity.get("root_fingerprint_source"))
        observation_fingerprint = _optional_str(canonical_identity.get("observation_fingerprint"))
        observation_fingerprint_source = _optional_str(
            canonical_identity.get("observation_fingerprint_source")
        )
        identity_anchor_line = _optional_int(canonical_identity.get("identity_anchor_line"))
        identity_anchor_reason = _optional_str(canonical_identity.get("identity_anchor_reason"))
    else:
        root_fingerprint = primary.root_fingerprint if primary is not None else None
        root_fingerprint_source = primary.root_fingerprint_source if primary is not None else None
        observation_fingerprint = (
            selected_observation.observation_fingerprint
            if selected_observation is not None
            else None
        )
        observation_fingerprint_source = (
            selected_observation.observation_fingerprint_source
            if selected_observation is not None
            else None
        )

    identity_kind = (
        HistoryIdentityKind.ROOT.value
        if root_fingerprint is not None and primary is not None
        else (
            HistoryIdentityKind.OBSERVATION_ONLY.value
            if observation_fingerprint is not None and selected_observation is not None
            else HistoryIdentityKind.NONE.value
        )
    )

    locality = (
        decision_evidence.locality
        if source == AttemptFailureFactsSource.L0_DETERMINISTIC
        else dict(root_locality or {})
    )

    return AttemptFailureFacts(
        source=source,
        identity_kind=identity_kind,
        root_fingerprint=root_fingerprint,
        root_fingerprint_source=root_fingerprint_source,
        observation_fingerprint=observation_fingerprint,
        observation_fingerprint_source=observation_fingerprint_source,
        fault_outcome=selected.fault_outcome if selected is not None else None,
        primary_line=primary.line if primary is not None else None,
        selected_observation_line=(
            selected_observation.line if selected_observation is not None else None
        ),
        selected_observation_causal_role=(
            selected_observation.causal_role if selected_observation is not None else None
        ),
        identity_anchor_line=identity_anchor_line,
        identity_anchor_reason=identity_anchor_reason,
        failure_iteration=selected.failure_iteration if selected is not None else None,
        classifiers=_classifiers(selected, decision_evidence),
        affected_entity=_affected_entity(primary),
        faulting_rank=selected.rank if selected is not None else None,
        faulting_node=selected.node if selected is not None else None,
        faulting_gpu=selected.gpu if selected is not None else None,
        root_observer_ranks=(
            _optional_string_sequence(locality.get("root_observer_ranks"))
            if identity_kind == HistoryIdentityKind.ROOT.value
            else None
        ),
        unattributed_root_occurrence_count=(
            _optional_int(locality.get("unattributed_root_occurrence_count"))
            if identity_kind == HistoryIdentityKind.ROOT.value
            else None
        ),
    )


def _classifiers(
    primary: FailureEvidence | None,
    decision_evidence: DecisionEvidence,
) -> tuple[str, ...]:
    if primary is None:
        return ()
    canonical_identity = decision_evidence.canonical_observed_identity
    deterministic = decision_evidence.deterministic_primary_candidate
    canonical_line = _optional_int(canonical_identity.get("identity_anchor_line"))
    deterministic_line = deterministic.line if deterministic is not None else None
    if primary.line not in {canonical_line, deterministic_line}:
        return ()
    value = canonical_identity.get("classifiers", ())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("canonical observed identity classifiers must be an array")
    if any(not isinstance(item, str) or not item for item in value):
        raise TypeError("canonical observed identity classifiers items must be non-empty strings")
    return tuple(value)


def _affected_entity(primary: FailureEvidence | None) -> AffectedEntity | None:
    if primary is None:
        return None
    return primary.affected_entity


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


def _optional_string_sequence(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("root_observer_ranks must be an array or null")
    if any(not isinstance(item, str) or not item for item in value):
        raise TypeError("root_observer_ranks items must be non-empty strings")
    return tuple(value)
