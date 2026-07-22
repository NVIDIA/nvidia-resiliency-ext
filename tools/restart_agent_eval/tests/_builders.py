# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract-shaped test-data builders shared across harness test modules."""

from __future__ import annotations

from typing import Any


def retry_policy(
    *,
    rule: str = "general_retry",
    allowed_retries: int = 3,
    retry_budget_exhausted: bool = False,
    policy_version: str | None = None,
    matching_prior_failures: int | None = None,
) -> dict[str, Any]:
    """Build an L4 retry-policy payload, including optional product metadata."""
    payload: dict[str, Any] = {
        "rule": rule,
        "allowed_retries": allowed_retries,
        "retry_budget_exhausted": retry_budget_exhausted,
    }
    if policy_version is not None:
        payload = {"policy_version": policy_version, **payload}
    if matching_prior_failures is not None:
        payload["matching_prior_failures"] = matching_prior_failures
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
