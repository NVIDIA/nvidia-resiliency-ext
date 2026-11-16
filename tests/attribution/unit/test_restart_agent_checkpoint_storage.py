# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cat 44: a checkpoint save that fails because the filesystem is full.

A production run saved checkpoint N, then hit '[Errno 122] Disk quota exceeded'
from the async writers 42 lines later. Cat 19 took it as transient async-write
I/O and the job restarted. The three cycles that followed never reached a single
iteration, the whole scheduler job then ended, and the run only continued the
next day under a fresh job id once space had been reclaimed by hand.

The symptom that won the classification was a PyTorch inline_container position
mismatch, which outnumbered the capacity errors two to one. It is the wake of a
truncated write rather than an independent fault, so the category has to claim
precedence over it explicitly or the same misread recurs.

Capacity is a property of the shared filesystem, so no node can be drained to
clear it and a retry writes into the same full target. That is what makes it a
STOP rather than a tight retry budget.
"""

from __future__ import annotations

import pytest

from nvidia_resiliency_ext.attribution.restart_agent.l1.categories import category_by_id
from nvidia_resiliency_ext.attribution.restart_agent.l4 import L4PolicyInput, evaluate_policy
from nvidia_resiliency_ext.attribution.restart_agent.models import (
    AssessmentStatus,
    AttemptFailureFacts,
    AttemptFailureFactsSource,
    Decision,
    FailureDomain,
    FailureDomainAssessment,
    FailureEvidence,
    FaultOutcome,
    HistorySummary,
    L1CategoryConfirmedRestartConfig,
    ModelRecoveryAssessment,
    PolicyContextConfig,
    RetryOutlookAssessment,
    RetryOutlookWithoutWorkloadChange,
)

STORAGE_EXHAUSTED_CATEGORY_ID = 44
TRANSIENT_CHECKPOINT_IO_CATEGORY_ID = 19


# ---------------------------------------------------------------------------
# Taxonomy entry
# ---------------------------------------------------------------------------


def test_the_category_stops():
    """Restarting writes into the same full filesystem. The production run spent
    three cycles proving it."""

    category = category_by_id(STORAGE_EXHAUSTED_CATEGORY_ID)
    assert category is not None
    assert category.decision == "STOP"


@pytest.mark.parametrize(
    "token",
    [
        "Errno 122",
        "Disk quota exceeded",
        "Errno 28",
        "No space left on device",
        "filesystem_async",
    ],
    ids=["edquot_errno", "edquot_text", "enospc_errno", "enospc_text", "writer"],
)
def test_the_category_names_the_capacity_surfaces(token):
    """Both capacity errnos, since a full filesystem and an exceeded quota are
    the same failure to a job and only differ by which limit was hit first."""

    assert token in category_by_id(STORAGE_EXHAUSTED_CATEGORY_ID).description


def test_the_category_claims_precedence_over_the_truncated_write():
    """The mismatch outnumbered the capacity errors in production and is what
    pulled the classification to cat 19. Without an explicit claim, the more
    numerous surface wins again."""

    description = category_by_id(STORAGE_EXHAUSTED_CATEGORY_ID).description
    assert "inline_container" in description
    assert "truncated write" in description


def test_the_category_excludes_its_neighbours():
    description = category_by_id(STORAGE_EXHAUSTED_CATEGORY_ID).description
    for excluded in ("Not cat 19", "Not cat 16", "Not cat 22"):
        assert excluded in description


def test_a_bare_mismatch_without_a_capacity_errno_stays_transient():
    """The boundary must cut both ways, or cat 19 loses cases that are genuinely
    transient and restartable."""

    description = category_by_id(STORAGE_EXHAUSTED_CATEGORY_ID).description
    assert "A bare inline_container mismatch with no capacity errno" in description


# ---------------------------------------------------------------------------
# The carve-out on the category it is taken from
# ---------------------------------------------------------------------------


def test_cat_19_now_carves_out_exhausted_storage():
    """Cat 19 already excluded memory exhaustion on the checkpoint path. Storage
    exhaustion is the same shape and had no such exclusion, which is why it
    absorbed this failure."""

    description = category_by_id(TRANSIENT_CHECKPOINT_IO_CATEGORY_ID).description
    assert "cat 44" in description
    assert "does not cover failures whose mechanism is exhausted storage" in description
    # The pre-existing memory carve-out must survive the edit.
    assert "memory exhaustion" in description


# ---------------------------------------------------------------------------
# L4: a STOP label stops on first sight
# ---------------------------------------------------------------------------


def _evaluate(category_id: int):
    primary = FailureEvidence(
        failure_class="os_error",
        signature="[Errno 122] Disk quota exceeded",
        root_fingerprint="observed:oserror:disk_quota_exceeded",
        root_fingerprint_source="observed",
        fault_outcome=FaultOutcome.TERMINAL.value,
        line=129021,
        registry_id="observed",
    )
    return evaluate_policy(
        L4PolicyInput(
            primary=primary,
            current_failure_facts=AttemptFailureFacts(
                source=AttemptFailureFactsSource.L2_GROUNDED,
                root_fingerprint=primary.root_fingerprint,
                root_fingerprint_source="observed",
                fault_outcome=FaultOutcome.TERMINAL.value,
                primary_line=129021,
                classifiers=(),
            ),
            history=HistorySummary(available=True, matching_root_attempts=0),
            model_recovery_assessment=ModelRecoveryAssessment(
                failure_domain=FailureDomainAssessment(
                    value=FailureDomain.WORKLOAD,
                    status=AssessmentStatus.SUPPORTED_BUT_UNCONFIRMED,
                    confidence=85,
                ),
                retry_outlook_without_workload_change=RetryOutlookAssessment(
                    value=RetryOutlookWithoutWorkloadChange.MAY_RECOVER,
                    status=AssessmentStatus.SUPPORTED_BUT_UNCONFIRMED,
                    confidence=61,
                ),
                rationale="checkpoint write path failed",
            ),
            policy_contexts=PolicyContextConfig(
                l1_category_confirmed_restart=L1CategoryConfirmedRestartConfig(
                    enabled=True, allowed_retries=1
                ),
            ),
            l1_category_selection={
                "category_id": category_id,
                "category_confidence": 86,
                "category_rationale": "test",
            },
        )
    ).retry_policy


def test_the_stop_label_forces_zero_retries_on_first_sight():
    """l1_category_confirmed_stop overrides whatever budget the base rule chose."""

    outcome = _evaluate(STORAGE_EXHAUSTED_CATEGORY_ID)

    assert outcome.decision == Decision.STOP.value
    assert outcome.effective_policy.policy_context_id == "l1_category_confirmed_stop"
    assert outcome.effective_policy.allowed_retries == 0


def test_the_old_category_would_have_restarted_instead():
    """What production did: cat 19 is RESTART-labelled, so the base rule's retry
    budget survived and the job went round again."""

    outcome = _evaluate(TRANSIENT_CHECKPOINT_IO_CATEGORY_ID)

    assert outcome.decision == Decision.RESTART.value
    assert outcome.effective_policy.allowed_retries != 0
