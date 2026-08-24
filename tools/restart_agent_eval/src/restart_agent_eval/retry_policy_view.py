# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only helpers for the Restart Agent concurrent retry-ledger contract."""

from __future__ import annotations

from typing import Any


def general_root_ceiling(retry_policy: dict[str, Any]) -> dict[str, Any]:
    value = retry_policy.get("general_root_ceiling")
    return value if isinstance(value, dict) else {}


def selected_policy_ledger(retry_policy: dict[str, Any]) -> dict[str, Any]:
    value = retry_policy.get("selected_policy_ledger")
    return value if isinstance(value, dict) else {}


def effective_policy(retry_policy: dict[str, Any]) -> dict[str, Any]:
    value = retry_policy.get("effective_policy")
    return value if isinstance(value, dict) else {}


def base_rule(retry_policy: dict[str, Any]) -> str | None:
    value = retry_policy.get("base_rule")
    return str(value) if value is not None else None


def effective_rule(retry_policy: dict[str, Any]) -> str | None:
    value = effective_policy(retry_policy).get("rule")
    return str(value) if value is not None else None


def rule_expectation_correct(
    retry_policy: dict[str, Any],
    *,
    accepted_base_rules: set[str],
    accepted_effective_rules: set[str],
) -> bool | None:
    """Score independently declared base and effective rule expectations."""

    checks: list[bool] = []
    if accepted_base_rules:
        checks.append(str(base_rule(retry_policy)) in accepted_base_rules)
    if accepted_effective_rules:
        checks.append(str(effective_rule(retry_policy)) in accepted_effective_rules)
    return all(checks) if checks else None


def job_no_progress_guard(retry_policy: dict[str, Any]) -> dict[str, Any]:
    value = retry_policy.get("job_no_progress_guard")
    return value if isinstance(value, dict) else {}


def job_unknown_progress_guard(retry_policy: dict[str, Any]) -> dict[str, Any]:
    value = retry_policy.get("job_unknown_progress_guard")
    return value if isinstance(value, dict) else {}


def effective_retry_ledger(retry_policy: dict[str, Any]) -> dict[str, Any]:
    """Return the selected ledger when present, otherwise the root ceiling."""

    return selected_policy_ledger(retry_policy) or general_root_ceiling(retry_policy)


def effective_allowed_retries(retry_policy: dict[str, Any]) -> int | None:
    """Return the effective allowance represented by the selected policy rule."""

    value = effective_policy(retry_policy).get("allowed_retries")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def any_observed_advance(retry_policy: dict[str, Any]) -> bool:
    return any(
        bool(ledger.get("applicable")) and bool(ledger.get("observed_advance"))
        for ledger in (
            general_root_ceiling(retry_policy),
            selected_policy_ledger(retry_policy),
            job_no_progress_guard(retry_policy),
            job_unknown_progress_guard(retry_policy),
        )
        if ledger
    )


def ledger_ratio(ledger: dict[str, Any]) -> str:
    if not ledger:
        return "-"
    if ledger.get("applicable") is False:
        return str(ledger.get("inapplicable_reason") or "not_applicable")
    return f"{ledger.get('matching_prior_attempts')}/" f"{ledger.get('allowed_retries')}"
