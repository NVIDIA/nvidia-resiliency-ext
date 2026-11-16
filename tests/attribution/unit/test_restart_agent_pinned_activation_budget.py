# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cat 43: an activation-offload buffer that refuses an allocation on its own budget.

Two production jobs raised the same RuntimeError with opposite shapes. In one it
recurred every cycle on a tight iteration clock, on all data-parallel replicas of
a single expert rank. In the other it appeared once, on one rank, in one of eight
cycles, and the next cycle ran clean. Cat 28 called both a deterministic launch
or workload-code error and stopped the job, which was wrong for the second and
wrong about the mechanism for the first.

Labelling the category RESTART hands it to ``l1_category_confirmed_restart``,
which overrides the workload_unrecoverable STOP on the *first* occurrence only
and scopes the budget to the root fingerprint. A second occurrence falls through
to the base rule and stops. These tests pin both halves, and the normalization
the recurrence half silently depends on.
"""

from nvidia_resiliency_ext.attribution.restart_agent.identity import canonical_observed_fingerprint
from nvidia_resiliency_ext.attribution.restart_agent.l1.categories import category_by_id
from nvidia_resiliency_ext.attribution.restart_agent.l4 import L4PolicyInput, evaluate_policy
from nvidia_resiliency_ext.attribution.restart_agent.models import (
    AssessmentStatus,
    AttemptFailureFacts,
    AttemptFailureFactsSource,
    Decision,
    DecisionBasis,
    FailureDomain,
    FailureDomainAssessment,
    FailureEvidence,
    FaultOutcome,
    HistoryMatchScope,
    HistorySummary,
    L1CategoryConfirmedRestartConfig,
    ModelRecoveryAssessment,
    PolicyContextConfig,
    RetryOutlookAssessment,
    RetryOutlookWithoutWorkloadChange,
    RetryPolicyRule,
)

PINNED_BUDGET_CATEGORY_ID = 43

# The two production occurrences, verbatim apart from the byte counts.
OCCURRENCE_A = (
    "RuntimeError: Pinned activation buffer budget exceeded: request=4965007360 bytes, "
    "capacity=137438953472 bytes, live=134504464384 bytes, unused_tail=2934489088 bytes. "
    "Increase fine_grained_offloading_buffer_size_gib or reduce offloaded activations."
)
OCCURRENCE_B = (
    "RuntimeError: Pinned activation buffer budget exceeded: request=1011351040 bytes, "
    "capacity=137438953472 bytes, live=136427602432 bytes, unused_tail=1011351040 bytes. "
    "Increase fine_grained_offloading_buffer_size_gib or reduce offloaded activations."
)


# ---------------------------------------------------------------------------
# Taxonomy entry
# ---------------------------------------------------------------------------


def test_category_is_restart_labelled():
    """The label is the whole mechanism: ``l1_category_confirmed_restart`` returns
    early unless the chosen category's decision is RESTART, so a STOP label would
    silently leave the retry unreachable."""

    category = category_by_id(PINNED_BUDGET_CATEGORY_ID)
    assert category is not None
    assert category.decision == "RESTART"


def test_category_names_the_raising_frame_and_the_saturation_shape():
    """A picker choosing between this and cat 28 needs the discriminator, not just
    the message: the pool is full rather than the request oversized."""

    description = category_by_id(PINNED_BUDGET_CATEGORY_ID).description
    for token in (
        "Pinned activation buffer budget exceeded",
        "pinned_activation_buffer.py",
        "fine_grained_activation_offload.py",
        "unused_tail",
    ):
        assert token in description, f"{token!r} missing from the category description"


def test_category_excludes_the_three_it_is_confusable_with():
    """Cat 28 took both production cases. Cat 31 and cat 32 are the other two
    readings of 'ran out of memory', and both are STOP, so a wrong turn there
    costs the job its restart."""

    description = category_by_id(PINNED_BUDGET_CATEGORY_ID).description
    assert "Not cat 28" in description
    assert "Not cat 31" in description
    assert "Not cat 32" in description


def test_category_description_states_no_verdict():
    """Descriptions say what the failure is; the decision field says what to do."""

    description = category_by_id(PINNED_BUDGET_CATEGORY_ID).description
    lowered = description.lower()
    for forbidden in ("stop the job", "product policy", "verdict", "restart the job"):
        assert forbidden not in lowered


# ---------------------------------------------------------------------------
# The normalization the ROOT_ONLY budget depends on
# ---------------------------------------------------------------------------


def test_two_occurrences_share_one_root_fingerprint():
    """``workload_confirmation_retry`` scopes its budget to the root fingerprint.

    The byte counts differ on every occurrence - different request, different
    live, different tail - so if they survived normalization each recurrence
    would look like a new failure, the first-occurrence guard would never fire,
    and the job would restart forever on a budget that never exhausts.
    """

    assert canonical_observed_fingerprint(OCCURRENCE_A) == canonical_observed_fingerprint(
        OCCURRENCE_B
    )


def test_the_fingerprint_still_names_the_mechanism():
    """Normalizing must not flatten it into a bare 'runtimeerror' shared with
    every other RuntimeError in the corpus."""

    fingerprint = canonical_observed_fingerprint(OCCURRENCE_A)
    assert "pinned_activation_buffer_budget_exceeded" in fingerprint


# ---------------------------------------------------------------------------
# L4: one restart, then stop
# ---------------------------------------------------------------------------


def _primary() -> FailureEvidence:
    return FailureEvidence(
        failure_class="runtime_error",
        signature=OCCURRENCE_A,
        root_fingerprint=canonical_observed_fingerprint(OCCURRENCE_A),
        root_fingerprint_source="observed",
        fault_outcome=FaultOutcome.TERMINAL.value,
        line=128010,
        registry_id="observed",
    )


def _facts() -> AttemptFailureFacts:
    return AttemptFailureFacts(
        source=AttemptFailureFactsSource.L2_GROUNDED,
        root_fingerprint=canonical_observed_fingerprint(OCCURRENCE_A),
        root_fingerprint_source="observed",
        fault_outcome=FaultOutcome.TERMINAL.value,
        primary_line=128010,
        classifiers=(),
    )


def _workload_unrecoverable() -> ModelRecoveryAssessment:
    return ModelRecoveryAssessment(
        failure_domain=FailureDomainAssessment(
            value=FailureDomain.WORKLOAD,
            status=AssessmentStatus.ESTABLISHED_BY_CURRENT_LOG,
            confidence=90,
        ),
        retry_outlook_without_workload_change=RetryOutlookAssessment(
            value=RetryOutlookWithoutWorkloadChange.CANNOT_RECOVER,
            status=AssessmentStatus.ESTABLISHED_BY_CURRENT_LOG,
            confidence=90,
        ),
        rationale="activation offload buffer budget exhausted",
    )


def _evaluate(matching_root_attempts: int):
    return evaluate_policy(
        L4PolicyInput(
            primary=_primary(),
            current_failure_facts=_facts(),
            history=HistorySummary(available=True, matching_root_attempts=matching_root_attempts),
            model_recovery_assessment=_workload_unrecoverable(),
            policy_contexts=PolicyContextConfig(
                l1_category_confirmed_restart=L1CategoryConfirmedRestartConfig(
                    enabled=True, allowed_retries=1
                ),
            ),
            l1_category_selection={
                "category_id": PINNED_BUDGET_CATEGORY_ID,
                "category_confidence": 90,
                "category_rationale": "test",
            },
        )
    ).retry_policy


def test_first_occurrence_restarts_once():
    """The 1-of-8-cycles case: cat 28 stopped a job whose next cycle ran clean."""

    outcome = _evaluate(matching_root_attempts=0)

    assert outcome.decision == Decision.RESTART.value
    assert outcome.effective_policy.rule == RetryPolicyRule.WORKLOAD_CONFIRMATION_RETRY.value
    assert outcome.effective_policy.history_match_scope == HistoryMatchScope.ROOT_ONLY.value
    assert outcome.effective_policy.allowed_retries == 1


def test_second_consecutive_occurrence_stops():
    """The every-cycle case: one restart proves it recurs, and the job stops
    rather than paying the same tax indefinitely."""

    outcome = _evaluate(matching_root_attempts=1)

    assert outcome.decision == Decision.STOP.value
    assert outcome.decision_basis == DecisionBasis.WORKLOAD_UNRECOVERABLE.value
    assert outcome.effective_policy.policy_context_id is None
