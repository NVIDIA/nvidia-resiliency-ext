"""Mirrored source-log and human-gold corpus discovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .gold import validate_gold_source, validate_scored_gold_label

SCHEMA_VERSION = "restart_agent_eval.v1"


@dataclass(frozen=True)
class Case:
    """One source log paired with its independent human-approved label."""

    case_id: str
    label_path: Path
    log_path: Path
    recovery_expectation: Mapping[str, Any]
    retry_policy_expectation: Mapping[str, Any]
    accepted_decisions: tuple[str, ...]
    label_version: int
    label: Mapping[str, Any]

    @property
    def available(self) -> bool:
        return self.log_path.is_file()


def discover_cases(log_root: Path, gold_root: Path) -> list[Case]:
    """Discover ``gold.json`` files whose parent mirrors a source-log path."""

    cases: list[Case] = []
    for label_path in sorted(gold_root.glob("**/gold.json")):
        relative_log_path = label_path.parent.relative_to(gold_root)
        if any(part.startswith("_") for part in relative_log_path.parts):
            continue
        try:
            label = json.loads(label_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label_path}: invalid JSON: {exc}") from exc
        cases.append(
            case_from_label(
                log_path=log_root / relative_log_path,
                label_path=label_path,
                label=label,
                default_case_id=relative_log_path.as_posix(),
            )
        )
    return cases


def case_from_label(
    *,
    log_path: Path,
    label_path: Path,
    label: Any,
    default_case_id: str,
) -> Case:
    """Validate one label and bind it to its mirrored source log."""

    if not isinstance(label, Mapping):
        raise ValueError(f"{label_path}: label must be an object")
    validate_scored_gold_label(label)
    if label.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{label_path}: schema_version must be {SCHEMA_VERSION!r}")
    action_expectation = label.get("action_expectation") or {}
    accepted_decisions = tuple(
        str(value)
        for value in action_expectation.get("accepted")
        or ([label.get("decision")] if label.get("decision") else [])
    )
    if not accepted_decisions or any(
        value not in {"STOP", "RESTART"} for value in accepted_decisions
    ):
        raise ValueError(f"{label_path}: action_expectation.accepted must use STOP/RESTART")
    recovery_expectation = label.get("recovery_assessment_expectation") or {}
    retry_expectation = label.get("retry_policy_expectation") or {}
    if not isinstance(recovery_expectation, Mapping):
        raise ValueError(f"{label_path}: recovery_assessment_expectation must be an object")
    if not isinstance(retry_expectation, Mapping):
        raise ValueError(f"{label_path}: retry_policy_expectation must be an object")
    if log_path.is_file():
        validate_gold_source(label, log_path)
    return Case(
        case_id=str(label.get("case_id") or default_case_id),
        label_path=label_path,
        log_path=log_path.resolve(),
        recovery_expectation=dict(recovery_expectation),
        retry_policy_expectation=dict(retry_expectation),
        accepted_decisions=accepted_decisions,
        label_version=int(label.get("label_version") or 1),
        label=dict(label),
    )
