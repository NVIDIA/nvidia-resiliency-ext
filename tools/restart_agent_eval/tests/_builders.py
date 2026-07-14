# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract-shaped test-data builders shared across harness test modules."""

from __future__ import annotations

from typing import Any


def retry_policy(
    *,
    base_rule: str | None = "general_retry",
    allowed_retries: int = 2,
    retry_budget_exhausted: bool = False,
    policy_version: str | None = None,
    matching_prior_attempts: int | None = None,
    history_match_scope: str | None = None,
) -> dict[str, Any]:
    """Build an L4 retry-policy payload, including optional product metadata."""
    matching = 0 if matching_prior_attempts is None else matching_prior_attempts
    selected_rule = base_rule in {
        "concrete_confirmation_retry",
        "workload_confirmation_retry",
    }
    immediate_stop = base_rule == "workload_unrecoverable"
    general_applicable = base_rule is not None and not immediate_stop
    general_allowed_retries = allowed_retries if base_rule == "general_retry" else 2
    general_exhausted = retry_budget_exhausted and general_applicable and not selected_rule
    selected_exhausted = retry_budget_exhausted and selected_rule
    selected_scope = (
        None
        if immediate_stop
        else history_match_scope
        or ("root_and_entity" if base_rule == "concrete_confirmation_retry" else "root_only")
    )
    payload: dict[str, Any] = {
        "base_rule": base_rule,
        "effective_policy": (
            {
                "source": "base_rule",
                "rule": base_rule,
                "history_match_scope": selected_scope,
                "allowed_retries": allowed_retries,
                "policy_context_id": None,
            }
            if base_rule is not None
            else None
        ),
        "applied_policy_context": None,
        "retry_budget_exhausted": (retry_budget_exhausted if not immediate_stop else False),
        "exhausted_by": (
            ["selected_policy_ledger"]
            if selected_exhausted
            else ["general_root_ceiling"] if general_exhausted else []
        ),
        "general_root_ceiling": {
            "ledger_id": "general_root_ceiling",
            "applicable": general_applicable,
            "rule": "general_retry",
            "history_match_scope": "root_only",
            "allowed_retries": (general_allowed_retries if general_applicable else None),
            "matching_prior_attempts": matching,
            "observed_advance": False,
            "exhausted": general_exhausted,
            "inapplicable_reason": (
                "immediate_unrecoverable"
                if immediate_stop
                else "missing_primary" if base_rule is None else None
            ),
        },
        "selected_policy_ledger": (
            {
                "ledger_id": "selected_policy_ledger",
                "applicable": True,
                "rule": base_rule,
                "history_match_scope": selected_scope,
                "allowed_retries": allowed_retries,
                "matching_prior_attempts": matching,
                "observed_advance": False,
                "exhausted": selected_exhausted,
                "inapplicable_reason": None,
            }
            if selected_rule
            else None
        ),
        "job_no_progress_guard": {
            "ledger_id": "job_no_progress_guard",
            "applicable": True,
            "rule": "job_no_progress_guard",
            "history_match_scope": "same_job_no_progress",
            "allowed_retries": 3,
            "matching_prior_attempts": 0,
            "observed_advance": False,
            "exhausted": False,
            "inapplicable_reason": None,
        },
        "job_unknown_progress_guard": {
            "ledger_id": "job_unknown_progress_guard",
            "applicable": True,
            "rule": "job_unknown_progress_guard",
            "history_match_scope": "same_job_unknown_progress",
            "allowed_retries": 3,
            "matching_prior_attempts": 0,
            "observed_advance": False,
            "exhausted": False,
            "inapplicable_reason": None,
        },
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
