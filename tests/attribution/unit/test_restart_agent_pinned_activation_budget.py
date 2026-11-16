# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cat 43: an activation-offload buffer that refuses an allocation on its own budget.

Two production jobs raised the same RuntimeError with opposite shapes. In one it
recurred every cycle on a tight iteration clock, on all data-parallel replicas of
a single expert rank. In the other it appeared once, on one rank, in one of eight
cycles, and the next cycle ran clean. Cat 28 called both a deterministic launch
or workload-code error and stopped the job, which was wrong for the second and
wrong about the mechanism for the first.

What the RESTART label buys is the *absence* of an override. A STOP-labelled
category fires ``l1_category_confirmed_stop``, which forces allowed_retries=0
regardless of what the base rule concluded; a RESTART-labelled one falls through
to the base rule, which already models retries. Running both production logs
through the real pipeline confirms it end to end: first occurrence RESTART under
concrete_confirmation_retry with allowed_retries=1, second occurrence of the same
root STOP on retry_budget_exhausted.

``l1_category_confirmed_restart`` is a separate path that only engages when the
base rule reached workload_unrecoverable, which needs an *established* failure
domain. Both production logs assess the domain as supported_but_unconfirmed, so
they never take it. It is covered here too, because a future log with a firmer
assessment would.

These tests pin the label, both override paths, and the normalization the
recurrence behaviour silently depends on.
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


def test_first_occurrence_restarts_once_when_the_domain_is_established():
    """The override path. Not what the production logs take - they assess the
    domain as supported_but_unconfirmed - but a firmer assessment would."""

    outcome = _evaluate(matching_root_attempts=0)

    assert outcome.decision == Decision.RESTART.value
    assert outcome.effective_policy.rule == RetryPolicyRule.WORKLOAD_CONFIRMATION_RETRY.value
    assert outcome.effective_policy.history_match_scope == HistoryMatchScope.ROOT_ONLY.value
    assert outcome.effective_policy.allowed_retries == 1


def test_second_consecutive_occurrence_stops_on_the_override_path():
    """Recurrence suppresses the first-occurrence override and the base rule stops."""

    outcome = _evaluate(matching_root_attempts=1)

    assert outcome.decision == Decision.STOP.value
    assert outcome.decision_basis == DecisionBasis.WORKLOAD_UNRECOVERABLE.value
    assert outcome.effective_policy.policy_context_id is None


# ---------------------------------------------------------------------------
# The path the production logs actually take
# ---------------------------------------------------------------------------


def _unconfirmed_workload() -> ModelRecoveryAssessment:
    """What both production logs assess: workload domain, but not established.

    This is why the base rule never reaches workload_unrecoverable, and so why
    the category-restart override never engages on these logs.
    """
    return ModelRecoveryAssessment(
        failure_domain=FailureDomainAssessment(
            value=FailureDomain.WORKLOAD,
            status=AssessmentStatus.SUPPORTED_BUT_UNCONFIRMED,
            confidence=92,
        ),
        retry_outlook_without_workload_change=RetryOutlookAssessment(
            value=RetryOutlookWithoutWorkloadChange.MAY_RECOVER,
            status=AssessmentStatus.SUPPORTED_BUT_UNCONFIRMED,
            confidence=80,
        ),
        rationale="allocator budget exhausted; domain supported but not established",
    )


def _evaluate_unconfirmed(category_id: int):
    return evaluate_policy(
        L4PolicyInput(
            primary=_primary(),
            current_failure_facts=_facts(),
            history=HistorySummary(available=True, matching_root_attempts=0),
            model_recovery_assessment=_unconfirmed_workload(),
            policy_contexts=PolicyContextConfig(
                l1_category_confirmed_restart=L1CategoryConfirmedRestartConfig(
                    enabled=True, allowed_retries=1
                ),
            ),
            l1_category_selection={
                "category_id": category_id,
                "category_confidence": 98,
                "category_rationale": "test",
            },
        )
    ).retry_policy


def test_a_restart_label_leaves_the_base_rule_in_charge():
    """No category context is stamped, so whatever retry budget the base rule
    chose survives. This is the whole effect of the label on these logs."""

    outcome = _evaluate_unconfirmed(PINNED_BUDGET_CATEGORY_ID)

    assert outcome.decision == Decision.RESTART.value
    assert outcome.effective_policy.policy_context_id is None
    assert outcome.effective_policy.allowed_retries != 0


def test_a_stop_label_would_force_zero_retries_instead():
    """The regression this category exists to undo.

    Cat 28 is STOP-labelled, so it fires l1_category_confirmed_stop and
    overrides the base rule with allowed_retries=0 - a stop on first sight,
    whatever the base rule had concluded.
    """

    outcome = _evaluate_unconfirmed(28)

    assert outcome.decision == Decision.STOP.value
    assert outcome.effective_policy.policy_context_id == "l1_category_confirmed_stop"
    assert outcome.effective_policy.allowed_retries == 0
