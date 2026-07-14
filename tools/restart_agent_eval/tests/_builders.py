# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract-shaped test-data builders shared across harness test modules."""

from __future__ import annotations

from typing import Any


def retry_policy(
    *,
    rule: str | None = "general_retry",
    allowed_retries: int = 3,
    retry_budget_exhausted: bool = False,
    policy_version: str | None = None,
    matching_prior_failures: int | None = None,
    history_match_scope: str | None = None,
) -> dict[str, Any]:
    """Build an L4 retry-policy payload, including optional product metadata."""
    matching = 0 if matching_prior_failures is None else matching_prior_failures
    selected_rule = rule in {
        "confirmation_retry",
        "bounded_retry",
        "workload_managed_recovery",
    }
    immediate_stop = rule == "workload_unrecoverable"
    general_applicable = rule is not None and not immediate_stop
    general_allowed_retries = allowed_retries if rule == "general_retry" else 3
    general_exhausted = retry_budget_exhausted and general_applicable and not selected_rule
    selected_exhausted = retry_budget_exhausted and selected_rule
    selected_scope = history_match_scope or (
        "root_and_entity" if rule == "workload_managed_recovery" else "root_only"
    )
    payload: dict[str, Any] = {
        "rule": rule,
        "retry_budget_exhausted": (retry_budget_exhausted if not immediate_stop else False),
        "exhausted_by": (
            ["selected_rule_budget"]
            if selected_exhausted
            else ["general_root_ceiling"] if general_exhausted else []
        ),
        "general_root_ceiling": {
            "ledger_id": "general_root_ceiling",
            "applicable": general_applicable,
            "rule": "general_retry",
            "history_match_scope": "root_only",
            "allowed_retries": (general_allowed_retries if general_applicable else None),
            "matching_prior_failures": matching,
            "observed_advance": False,
            "exhausted": general_exhausted,
            "inapplicable_reason": (
                "immediate_unrecoverable"
                if immediate_stop
                else "missing_primary" if rule is None else None
            ),
        },
        "selected_rule_budget": (
            {
                "ledger_id": "selected_rule_budget",
                "applicable": True,
                "rule": rule,
                "history_match_scope": selected_scope,
                "allowed_retries": allowed_retries,
                "matching_prior_failures": matching,
                "observed_advance": False,
                "exhausted": selected_exhausted,
                "inapplicable_reason": None,
            }
            if selected_rule
            else None
        ),
    }
    if policy_version is not None:
        payload = {"policy_version": policy_version, **payload}
    return payload


def recovery_assessment(
    *,
    failure_domain: str = "unknown",
    failure_domain_status: str | None = None,
    failure_domain_confidence: int = 50,
    retry_outlook: str = "unknown",
    retry_outlook_status: str | None = None,
    retry_outlook_confidence: int = 50,
    rationale: str = "test recovery assessment",
) -> dict[str, Any]:
    """Build the L1 recovery contract with optional supporting fields."""
    failure_domain_status = failure_domain_status or (
        "unknown" if failure_domain == "unknown" else "supported_but_unconfirmed"
    )
    retry_outlook_status = retry_outlook_status or (
        "unknown" if retry_outlook == "unknown" else "supported_but_unconfirmed"
    )
    return {
        "failure_domain": {
            "value": failure_domain,
            "status": failure_domain_status,
            "confidence": failure_domain_confidence,
        },
        "retry_outlook_without_workload_change": {
            "value": retry_outlook,
            "status": retry_outlook_status,
            "confidence": retry_outlook_confidence,
        },
        "rationale": rationale,
    }
