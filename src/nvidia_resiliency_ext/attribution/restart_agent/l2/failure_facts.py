# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build exact entities from L2-grounded failure evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..identity import build_affected_entity
from ..l1.response_contract import AFFECTED_ARTIFACT_PATH_FIELD, DIRECT_FAILURE_OBJECT_PATH_FIELD
from ..models import AffectedEntity, AffectedEntityKind


@dataclass(frozen=True)
class GroundedAffectedEntitySelection:
    """Exact grounded entity selected for L3 history comparison."""

    entity: AffectedEntity
    source_field: str
    selection_reason: str
    evidence_lines: tuple[int, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_field": self.source_field,
            "selection_reason": self.selection_reason,
            "evidence_lines": list(self.evidence_lines),
            "entity": self.entity.to_payload(),
        }


def select_grounded_affected_entity(
    failure_identity_grounding: Mapping[str, Any],
) -> GroundedAffectedEntitySelection | None:
    """Prefer an enclosing artifact, then use the exact failed object."""

    candidates = (
        (
            AFFECTED_ARTIFACT_PATH_FIELD,
            "grounded_affected_artifact_preferred",
        ),
        (
            DIRECT_FAILURE_OBJECT_PATH_FIELD,
            "grounded_direct_failure_object_fallback",
        ),
    )
    for source_field, selection_reason in candidates:
        grounding = failure_identity_grounding.get(source_field)
        if not isinstance(grounding, Mapping):
            continue
        grounded_value = grounding.get("grounded_value")
        if not isinstance(grounded_value, str) or not grounded_value.strip():
            continue
        evidence_lines = tuple(
            line
            for line in grounding.get("evidence_lines") or ()
            if isinstance(line, int) and not isinstance(line, bool) and line > 0
        )
        normalized_path = grounded_value.rstrip("/") or "/"
        entity = build_affected_entity(
            AffectedEntityKind.ARTIFACT,
            normalized_path,
            evidence_line=evidence_lines[0] if evidence_lines else None,
        )
        return GroundedAffectedEntitySelection(
            entity=entity,
            source_field=source_field,
            selection_reason=selection_reason,
            evidence_lines=evidence_lines,
        )

    return None
