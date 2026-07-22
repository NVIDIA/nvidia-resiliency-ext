# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble causal downstream evidence for the public result."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .l0 import build_cascades_for_primary
from .models import CascadeEvidence, CausalRole, FailureEvidence, L0Bundle, PolicyClass


def build_result_cascades(
    bundle: L0Bundle,
    primary: FailureEvidence | None,
    l2_audit: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Merge deterministic downstream groups with grounded L1 relationships."""

    l0_primary = bundle.deterministic_primary_candidate
    primary_matches_l0 = (
        primary is not None
        and l0_primary is not None
        and primary.line is not None
        and primary.line == l0_primary.line
    )
    deterministic = list(bundle.cascades) if primary_matches_l0 else []
    if not deterministic:
        deterministic = build_cascades_for_primary(
            bundle.registry_matches,
            primary,
            bundle.distributed_failure_incidents,
        )
    audited_roles = tuple(
        item
        for item in l2_audit.get("audited_related_failure_roles") or ()
        if isinstance(item, Mapping)
        and item.get("causal_role") in {CausalRole.CASCADE.value, CausalRole.TEARDOWN.value}
    )
    audited_failures_by_line = {
        item.get("line"): item
        for item in l2_audit.get("audited_related_failures") or ()
        if isinstance(item, Mapping) and isinstance(item.get("line"), int)
    }
    covered_audited_lines: set[int] = set()
    result: list[Mapping[str, Any]] = []

    for cascade in deterministic:
        member_lines = _cascade_member_lines(bundle, primary, cascade)
        rationales = tuple(
            str(item.get("rationale"))
            for item in audited_roles
            if item.get("line") in member_lines and item.get("rationale")
        )
        covered_audited_lines.update(
            int(item["line"]) for item in audited_roles if item.get("line") in member_lines
        )
        result.append(_cascade_payload(replace(cascade, relationship_rationales=rationales)))

    for item in audited_roles:
        line = item.get("line")
        if not isinstance(line, int) or line in covered_audited_lines:
            continue
        failure = audited_failures_by_line.get(line, {})
        role = str(item.get("causal_role"))
        result.append(
            {
                "fine_class": failure.get("fine_class") or "related_failure",
                "policy_class": PolicyClass.CASCADE.value,
                "cascade_fingerprint": failure.get("root_fingerprint"),
                "causal_role": role,
                "first_line": line,
                "last_line": line,
                "count": 1,
                "sample_lines": [line],
                "rank_spread": _optional_singleton(failure.get("rank")),
                "node_spread": _optional_singleton(failure.get("node")),
                "gpu_spread": _optional_singleton(failure.get("gpu")),
                "reason": f"L2-grounded {role} related to primary failure",
                "relationship_rationales": [str(item.get("rationale"))],
            }
        )

    return tuple(sorted(result, key=lambda item: int(item.get("first_line") or 0)))


def _cascade_member_lines(
    bundle: L0Bundle,
    primary: FailureEvidence | None,
    cascade: CascadeEvidence,
) -> set[int]:
    primary_line = primary.line if primary is not None else None
    if primary_line is None:
        return set()
    return {
        int(match.line)
        for match in bundle.registry_matches
        if match.line is not None
        and match.line > primary_line
        and match.fine_class == cascade.fine_class
        and match.root_fingerprint == cascade.cascade_fingerprint
        and _downstream_role(match) == cascade.causal_role
    }


def _downstream_role(failure: FailureEvidence) -> str:
    if failure.causal_role in {CausalRole.CASCADE.value, CausalRole.TEARDOWN.value}:
        return failure.causal_role
    return CausalRole.CASCADE.value


def _cascade_payload(cascade: CascadeEvidence) -> dict[str, Any]:
    return {
        "fine_class": cascade.fine_class,
        "policy_class": cascade.policy_class,
        "cascade_fingerprint": cascade.cascade_fingerprint,
        "causal_role": cascade.causal_role,
        "first_line": cascade.first_line,
        "last_line": cascade.last_line,
        "count": cascade.count,
        "sample_lines": list(cascade.sample_lines),
        "rank_spread": list(cascade.rank_spread),
        "node_spread": list(cascade.node_spread),
        "gpu_spread": list(cascade.gpu_spread),
        "reason": cascade.reason,
        "relationship_rationales": list(cascade.relationship_rationales),
    }


def _optional_singleton(value: Any) -> list[str]:
    return [] if value is None or value == "" else [str(value)]
