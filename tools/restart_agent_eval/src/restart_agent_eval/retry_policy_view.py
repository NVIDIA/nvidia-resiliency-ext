# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only helpers for the Restart Agent concurrent retry-ledger contract."""

from __future__ import annotations

from typing import Any


def general_root_ceiling(retry_policy: dict[str, Any]) -> dict[str, Any]:
    value = retry_policy.get("general_root_ceiling")
    return value if isinstance(value, dict) else {}


def selected_rule_budget(retry_policy: dict[str, Any]) -> dict[str, Any]:
    value = retry_policy.get("selected_rule_budget")
    return value if isinstance(value, dict) else {}


def effective_rule_budget(retry_policy: dict[str, Any]) -> dict[str, Any]:
    """Return the narrower rule budget when present, otherwise the root ceiling."""

    return selected_rule_budget(retry_policy) or general_root_ceiling(retry_policy)


def effective_allowed_retries(retry_policy: dict[str, Any]) -> int | None:
    """Return the effective allowance represented by the selected policy rule."""

    if retry_policy.get("rule") == "workload_unrecoverable":
        return 0
    value = effective_rule_budget(retry_policy).get("allowed_retries")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def any_observed_advance(retry_policy: dict[str, Any]) -> bool:
    return any(
        bool(ledger.get("applicable")) and bool(ledger.get("observed_advance"))
        for ledger in (
            general_root_ceiling(retry_policy),
            selected_rule_budget(retry_policy),
        )
        if ledger
    )


def ledger_ratio(ledger: dict[str, Any]) -> str:
    if not ledger:
        return "-"
    if ledger.get("applicable") is False:
        return str(ledger.get("inapplicable_reason") or "not_applicable")
    return f"{ledger.get('matching_prior_failures')}/" f"{ledger.get('allowed_retries')}"
