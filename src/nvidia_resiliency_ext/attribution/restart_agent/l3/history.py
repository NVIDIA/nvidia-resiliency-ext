# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic L3 comparison over immutable attempt records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..models import (
    AffectedEntityRelation,
    AttemptFailureFacts,
    AttemptProgressSummary,
    AttemptRecord,
    CycleHistoryComparison,
    FaultOutcome,
    HistoryDimensionComparison,
    HistoryIdentityKind,
    HistoryProgressComparison,
    HistoryProgressRelation,
    HistorySummary,
    JobProgressHistory,
    PriorAttemptView,
    RouteHistorySummary,
)

DETERMINISTIC_FACT_SELECTOR = "deterministic"
PRIMARY_FACT_SELECTOR_PREFIX = "primary:"
OBSERVATION_FACT_SELECTOR_PREFIX = "observation:"


def primary_fact_selector(route_id: str) -> str:
    return f"{PRIMARY_FACT_SELECTOR_PREFIX}{route_id}"


def observation_fact_selector(route_id: str) -> str:
    return f"{OBSERVATION_FACT_SELECTOR_PREFIX}{route_id}"


@dataclass(frozen=True)
class HistoryEvaluationInput:
    """Current record, selected facts, and immutable earlier-attempt view."""

    current_record: AttemptRecord
    fact_selector: str
    prior_attempts: PriorAttemptView
    job_progress: JobProgressHistory | None = None


def evaluate_history(history_input: HistoryEvaluationInput) -> HistorySummary:
    """Compare one explicitly selected failure track without cross-track fallback."""

    prior_view = history_input.prior_attempts
    current_facts = _selected_current_facts(
        history_input.current_record,
        history_input.fact_selector,
    )
    if not prior_view.available:
        return HistorySummary(
            available=False,
            availability_reason=prior_view.availability_reason,
            job_history_available=False,
            job_history_availability_reason=prior_view.availability_reason,
        )

    ordered = tuple(sorted(prior_view.records, key=lambda record: record.cycle_id))
    job_progress = history_input.job_progress or _evaluate_job_progress(
        history_input.current_record,
        prior_view,
    )
    job_fields = {
        "identity_kind": (
            current_facts.identity_kind
            if current_facts is not None
            else HistoryIdentityKind.NONE.value
        ),
        "same_job_attempts": job_progress.same_job_attempts,
        "job_history_available": job_progress.available,
        "job_history_availability_reason": job_progress.availability_reason,
        "job_comparisons": job_progress.comparisons,
        "consecutive_same_job_no_advance_attempts": (job_progress.consecutive_no_advance_attempts),
        "consecutive_same_job_unknown_progress_attempts": (
            job_progress.consecutive_unknown_progress_attempts
        ),
        "job_progress_advanced": job_progress.progress_advanced,
    }
    observation_fields = _observation_history_fields(
        history_input.current_record,
        current_facts,
        ordered,
        fact_selector=history_input.fact_selector,
    )
    if current_facts is None or not current_facts.root_fingerprint:
        return HistorySummary(
            available=False,
            availability_reason="missing_root_fingerprint",
            **observation_fields,
            **job_fields,
        )

    matching_pairs = tuple(
        (record, prior_facts)
        for record in ordered
        if (prior_facts := _selected_prior_facts(record, history_input.fact_selector)) is not None
        and prior_facts.root_fingerprint == current_facts.root_fingerprint
    )
    comparisons = tuple(
        _compare_progress(
            history_input.current_record,
            current_facts,
            record,
            prior_facts,
        )
        for record, prior_facts in matching_pairs
    )
    qualifying_pairs = tuple(
        (record, prior_facts, comparison)
        for (record, prior_facts), comparison in zip(matching_pairs, comparisons)
        if comparison.prior_fault_outcome
        in {FaultOutcome.TERMINAL.value, FaultOutcome.UNRESOLVED.value}
    )
    qualifying = tuple(comparison for _record, _facts, comparison in qualifying_pairs)
    qualifying_facts = tuple(facts for _record, facts, _comparison in qualifying_pairs)

    observed_advance = _relation_count(qualifying, HistoryProgressRelation.ADVANCED)
    same_progress = _relation_count(qualifying, HistoryProgressRelation.SAME)
    regressed = _relation_count(qualifying, HistoryProgressRelation.REGRESSED)
    unknown = _relation_count(qualifying, HistoryProgressRelation.UNKNOWN)
    no_observed_advance = same_progress + regressed
    exact_failure_position = sum(
        item.same_failure_iteration
        and item.relation
        in {HistoryProgressRelation.SAME.value, HistoryProgressRelation.REGRESSED.value}
        for item in qualifying
    )
    same_rank_iteration = sum(
        item.same_failure_iteration
        and item.same_rank
        and item.relation
        in {HistoryProgressRelation.SAME.value, HistoryProgressRelation.REGRESSED.value}
        for item in qualifying
    )
    comparable = tuple(
        item for item in qualifying if item.relation != HistoryProgressRelation.UNKNOWN.value
    )
    same_entity = tuple(
        item
        for item in qualifying
        if item.affected_entity_relation == AffectedEntityRelation.SAME.value
    )
    same_entity_comparable = tuple(
        item for item in same_entity if item.relation != HistoryProgressRelation.UNKNOWN.value
    )

    rank_matches = current_facts.faulting_rank is not None and any(
        facts.faulting_rank == current_facts.faulting_rank for facts in qualifying_facts
    )
    node_matches = current_facts.faulting_node is not None and any(
        facts.faulting_node == current_facts.faulting_node for facts in qualifying_facts
    )
    cross_node = current_facts.faulting_node is not None and any(
        facts.faulting_node is not None and facts.faulting_node != current_facts.faulting_node
        for facts in qualifying_facts
    )
    gpu_matches = current_facts.faulting_gpu is not None and any(
        facts.faulting_gpu == current_facts.faulting_gpu for facts in qualifying_facts
    )

    return HistorySummary(
        available=True,
        availability_reason="ready",
        matching_root_attempts=len(matching_pairs),
        comparisons=comparisons,
        observed_advance_attempts=observed_advance,
        same_progress_attempts=same_progress,
        regressed_progress_attempts=regressed,
        unknown_progress_attempts=unknown,
        no_observed_advance_attempts=no_observed_advance,
        matching_root_attempts_with_observed_training_progress=sum(
            record.progress.training_progress == "observed" for record, _facts in matching_pairs
        ),
        matching_root_attempts_before_observed_training_progress=sum(
            record.progress.failure_position == "before_observed_training_progress"
            for record, _facts in matching_pairs
        ),
        matching_root_attempts_with_unknown_training_progress=sum(
            record.progress.training_progress == "unknown" for record, _facts in matching_pairs
        ),
        exact_failure_position_attempts=exact_failure_position,
        same_rank_iteration_attempts=same_rank_iteration,
        same_entity_attempts=len(same_entity),
        different_entity_attempts=sum(
            item.affected_entity_relation == AffectedEntityRelation.DIFFERENT.value
            for item in qualifying
        ),
        unknown_entity_attempts=sum(
            item.affected_entity_relation == AffectedEntityRelation.UNKNOWN.value
            for item in qualifying
        ),
        consecutive_same_root_no_advance_attempts=_consecutive_same_root_no_advance(
            history_input.current_record,
            current_facts,
            ordered,
            fact_selector=history_input.fact_selector,
        ),
        consecutive_same_root_and_entity_no_advance_attempts=(
            _consecutive_same_root_no_advance(
                history_input.current_record,
                current_facts,
                ordered,
                fact_selector=history_input.fact_selector,
                require_same_entity=True,
            )
        ),
        advanced_beyond_all_comparable_attempts=bool(comparable)
        and all(item.relation == HistoryProgressRelation.ADVANCED.value for item in comparable),
        advanced_beyond_all_same_entity_comparable_attempts=bool(same_entity_comparable)
        and all(
            item.relation == HistoryProgressRelation.ADVANCED.value
            for item in same_entity_comparable
        ),
        cross_node_recurrence=cross_node,
        same_node_recurrence=node_matches,
        same_gpu_recurrence=gpu_matches,
        same_rank_only_recurrence=rank_matches and not node_matches and not gpu_matches,
        rank_to_gpu_mapping_available=any(facts.rank_to_gpu_map for facts in qualifying_facts),
        **observation_fields,
        **job_fields,
    )


def evaluate_cycle_history(
    *,
    current_record: AttemptRecord,
    prior_attempts: PriorAttemptView,
) -> CycleHistoryComparison:
    """Compare deterministic and route tracks independently for one cycle."""

    job_progress = _evaluate_job_progress(current_record, prior_attempts)
    deterministic = evaluate_history(
        HistoryEvaluationInput(
            current_record=current_record,
            fact_selector=DETERMINISTIC_FACT_SELECTOR,
            prior_attempts=prior_attempts,
            job_progress=job_progress,
        )
    )
    routes = tuple(
        RouteHistorySummary(
            route_id=entry.route_id,
            primary=(
                evaluate_history(
                    HistoryEvaluationInput(
                        current_record=current_record,
                        fact_selector=primary_fact_selector(entry.route_id),
                        prior_attempts=prior_attempts,
                        job_progress=job_progress,
                    )
                )
                if entry.primary is not None
                else None
            ),
            observation=(
                evaluate_history(
                    HistoryEvaluationInput(
                        current_record=current_record,
                        fact_selector=observation_fact_selector(entry.route_id),
                        prior_attempts=prior_attempts,
                        job_progress=job_progress,
                    )
                )
                if entry.observation is not None
                else None
            ),
        )
        for entry in current_record.enriched
    )
    return CycleHistoryComparison(
        job_progress=job_progress,
        deterministic=deterministic,
        routes=routes,
    )


def _evaluate_job_progress(
    current_record: AttemptRecord,
    prior_attempts: PriorAttemptView,
) -> JobProgressHistory:
    if not prior_attempts.available:
        return JobProgressHistory(
            available=False,
            availability_reason=prior_attempts.availability_reason,
        )
    ordered = tuple(sorted(prior_attempts.records, key=lambda record: record.cycle_id))
    comparisons = tuple(_compare_job_progress(current_record, record) for record in ordered)
    return JobProgressHistory(
        available=True,
        availability_reason="ready",
        same_job_attempts=len(ordered),
        comparisons=comparisons,
        consecutive_no_advance_attempts=_consecutive_job_relations(
            comparisons,
            {
                HistoryProgressRelation.SAME.value,
                HistoryProgressRelation.REGRESSED.value,
            },
        ),
        consecutive_unknown_progress_attempts=_consecutive_job_relations(
            comparisons,
            {HistoryProgressRelation.UNKNOWN.value},
        ),
        progress_advanced=bool(comparisons)
        and comparisons[-1].relation == HistoryProgressRelation.ADVANCED.value,
    )


def _observation_history_fields(
    current_record: AttemptRecord,
    current_facts: AttemptFailureFacts | None,
    ordered: Sequence[AttemptRecord],
    *,
    fact_selector: str,
) -> dict[str, object]:
    if (
        current_facts is None
        or current_facts.identity_kind != HistoryIdentityKind.OBSERVATION_ONLY.value
        or not current_facts.observation_fingerprint
    ):
        return {
            "observation_history_available": False,
            "observation_history_availability_reason": (
                "current_identity_is_root"
                if current_facts is not None
                and current_facts.identity_kind == HistoryIdentityKind.ROOT.value
                else "missing_observation_fingerprint"
            ),
        }
    matching_pairs = tuple(
        (record, prior_facts)
        for record in ordered
        if (prior_facts := _selected_prior_facts(record, fact_selector)) is not None
        and prior_facts.identity_kind == HistoryIdentityKind.OBSERVATION_ONLY.value
        and prior_facts.observation_fingerprint == current_facts.observation_fingerprint
    )
    comparisons = tuple(
        _compare_progress(current_record, current_facts, record, prior_facts)
        for record, prior_facts in matching_pairs
    )
    consecutive = 0
    for record in reversed(ordered):
        prior_facts = _selected_prior_facts(record, fact_selector)
        if (
            prior_facts is None
            or prior_facts.identity_kind != HistoryIdentityKind.OBSERVATION_ONLY.value
            or prior_facts.observation_fingerprint != current_facts.observation_fingerprint
            or prior_facts.fault_outcome
            not in {FaultOutcome.TERMINAL.value, FaultOutcome.UNRESOLVED.value}
        ):
            break
        comparison = _compare_progress(current_record, current_facts, record, prior_facts)
        if comparison.relation not in {
            HistoryProgressRelation.SAME.value,
            HistoryProgressRelation.REGRESSED.value,
        }:
            break
        consecutive += 1
    return {
        "observation_history_available": True,
        "observation_history_availability_reason": "ready",
        "matching_observation_attempts": len(matching_pairs),
        "observation_comparisons": comparisons,
        "consecutive_same_observation_no_advance_attempts": consecutive,
    }


def _selected_current_facts(
    record: AttemptRecord,
    selector: str,
) -> AttemptFailureFacts | None:
    if selector == DETERMINISTIC_FACT_SELECTOR:
        return record.deterministic
    track_name, route_id = _split_track_selector(selector)
    for entry in record.enriched:
        if entry.route_id == route_id:
            return entry.primary if track_name == "primary" else entry.observation
    return None


def _selected_prior_facts(
    record: AttemptRecord,
    selector: str,
) -> AttemptFailureFacts | None:
    if selector == DETERMINISTIC_FACT_SELECTOR:
        return record.deterministic
    track_name, route_id = _split_track_selector(selector)
    for entry in record.enriched:
        if entry.route_id == route_id:
            return entry.primary if track_name == "primary" else entry.observation
    return None


def _split_track_selector(selector: str) -> tuple[str, str]:
    for prefix, track_name in (
        (PRIMARY_FACT_SELECTOR_PREFIX, "primary"),
        (OBSERVATION_FACT_SELECTOR_PREFIX, "observation"),
    ):
        if selector.startswith(prefix) and selector[len(prefix) :]:
            return track_name, selector[len(prefix) :]
    raise ValueError(f"invalid enriched fact selector: {selector}")


def _compare_progress(
    current_record: AttemptRecord,
    current_facts: AttemptFailureFacts,
    prior_record: AttemptRecord,
    prior_facts: AttemptFailureFacts,
) -> HistoryProgressComparison:
    dimensions = _positive_progress_dimensions(current_record.progress, prior_record.progress)
    selected_basis = _selected_basis(dimensions)
    relation, conflict = _combine_positive_dimensions(dimensions)
    if not dimensions or all(
        item.relation == HistoryProgressRelation.UNKNOWN.value for item in dimensions
    ):
        fallback = _failure_iteration_dimension(
            current_facts.failure_iteration,
            prior_facts.failure_iteration,
        )
        dimensions = (*dimensions, fallback)
        selected_basis = "failure_iteration"
        relation = fallback.relation
        conflict = False

    return HistoryProgressComparison(
        prior_cycle_id=prior_record.cycle_id,
        selected_basis=selected_basis,
        dimension_comparisons=dimensions,
        positive_progress_conflict=conflict,
        relation=relation,
        prior_attempt_progress=prior_record.progress.to_payload(),
        prior_fault_outcome=prior_facts.fault_outcome,
        same_failure_iteration=(
            current_facts.failure_iteration is not None
            and prior_facts.failure_iteration == current_facts.failure_iteration
        ),
        same_rank=(
            current_facts.faulting_rank is not None
            and prior_facts.faulting_rank == current_facts.faulting_rank
        ),
        affected_entity_relation=_affected_entity_relation(current_facts, prior_facts),
        same_root_observer_count=_same_observer_count(current_facts, prior_facts),
        same_unattributed_root_occurrence_count=(
            _same_unattributed_count(current_facts, prior_facts)
        ),
    )


def _compare_job_progress(
    current_record: AttemptRecord,
    prior_record: AttemptRecord,
) -> HistoryProgressComparison:
    dimensions = _positive_progress_dimensions(current_record.progress, prior_record.progress)
    relation, conflict = _combine_positive_dimensions(dimensions)
    return HistoryProgressComparison(
        prior_cycle_id=prior_record.cycle_id,
        selected_basis=_selected_basis(dimensions),
        dimension_comparisons=dimensions,
        positive_progress_conflict=conflict,
        relation=relation,
        prior_attempt_progress=prior_record.progress.to_payload(),
        prior_fault_outcome=prior_record.deterministic.fault_outcome,
    )


def _positive_progress_dimensions(
    current: AttemptProgressSummary,
    prior: AttemptProgressSummary,
) -> tuple[HistoryDimensionComparison, ...]:
    candidates = (
        (
            "completed_step",
            prior.training_progress,
            current.training_progress,
            prior.last_completed_step,
            current.last_completed_step,
        ),
        (
            "checkpoint_step",
            prior.checkpoint_progress,
            current.checkpoint_progress,
            prior.last_checkpoint_step,
            current.last_checkpoint_step,
        ),
    )
    return tuple(
        _dimension_comparison(
            dimension,
            prior_status,
            current_status,
            prior_value,
            current_value,
        )
        for dimension, prior_status, current_status, prior_value, current_value in candidates
        if prior_status != "unknown" or current_status != "unknown"
    )


def _dimension_comparison(
    dimension: str,
    prior_status: str,
    current_status: str,
    prior_value: int | None,
    current_value: int | None,
) -> HistoryDimensionComparison:
    delta = None
    if prior_status == "unknown" or current_status == "unknown":
        relation = HistoryProgressRelation.UNKNOWN.value
    elif current_status == "observed" and prior_status == "not_observed":
        relation = HistoryProgressRelation.ADVANCED.value
    elif current_status == "not_observed" and prior_status == "observed":
        relation = HistoryProgressRelation.REGRESSED.value
    elif current_status == prior_status == "not_observed":
        relation = HistoryProgressRelation.SAME.value
    elif prior_value is None or current_value is None:
        relation = HistoryProgressRelation.UNKNOWN.value
    else:
        delta = current_value - prior_value
        relation = _relation_from_delta(delta)
    return HistoryDimensionComparison(
        dimension=dimension,
        prior_observation_status=prior_status,
        current_observation_status=current_status,
        prior_value=prior_value,
        current_value=current_value,
        delta=delta,
        relation=relation,
    )


def _failure_iteration_dimension(
    current_value: int | None,
    prior_value: int | None,
) -> HistoryDimensionComparison:
    delta = (
        current_value - prior_value
        if current_value is not None and prior_value is not None
        else None
    )
    return HistoryDimensionComparison(
        dimension="failure_iteration",
        prior_observation_status="observed" if prior_value is not None else "unknown",
        current_observation_status="observed" if current_value is not None else "unknown",
        prior_value=prior_value,
        current_value=current_value,
        delta=delta,
        relation=(
            _relation_from_delta(delta)
            if delta is not None
            else HistoryProgressRelation.UNKNOWN.value
        ),
    )


def _combine_positive_dimensions(
    dimensions: Sequence[HistoryDimensionComparison],
) -> tuple[str, bool]:
    relations = {item.relation for item in dimensions}
    advanced = HistoryProgressRelation.ADVANCED.value in relations
    regressed = HistoryProgressRelation.REGRESSED.value in relations
    if advanced and regressed:
        return HistoryProgressRelation.UNKNOWN.value, True
    if advanced:
        return HistoryProgressRelation.ADVANCED.value, False
    if regressed:
        return HistoryProgressRelation.REGRESSED.value, False
    if (
        relations
        and relations.issubset(
            {HistoryProgressRelation.SAME.value, HistoryProgressRelation.UNKNOWN.value}
        )
        and HistoryProgressRelation.SAME.value in relations
    ):
        return HistoryProgressRelation.SAME.value, False
    return HistoryProgressRelation.UNKNOWN.value, False


def _selected_basis(dimensions: Sequence[HistoryDimensionComparison]) -> str:
    names = {
        item.dimension
        for item in dimensions
        if item.relation != HistoryProgressRelation.UNKNOWN.value
    }
    if names == {"completed_step", "checkpoint_step"}:
        return "completed_step_and_checkpoint_step"
    if names == {"completed_step"}:
        return "completed_step"
    if names == {"checkpoint_step"}:
        return "checkpoint_step"
    return "none"


def _relation_from_delta(delta: int) -> str:
    if delta > 0:
        return HistoryProgressRelation.ADVANCED.value
    if delta < 0:
        return HistoryProgressRelation.REGRESSED.value
    return HistoryProgressRelation.SAME.value


def _relation_count(
    comparisons: Sequence[HistoryProgressComparison],
    relation: HistoryProgressRelation,
) -> int:
    return sum(item.relation == relation.value for item in comparisons)


def _consecutive_same_root_no_advance(
    current_record: AttemptRecord,
    current_facts: AttemptFailureFacts,
    ordered: Sequence[AttemptRecord],
    *,
    fact_selector: str,
    require_same_entity: bool = False,
) -> int:
    count = 0
    for prior_record in reversed(ordered):
        prior_facts = _selected_prior_facts(prior_record, fact_selector)
        if prior_facts is None or prior_facts.root_fingerprint != current_facts.root_fingerprint:
            break
        comparison = _compare_progress(
            current_record,
            current_facts,
            prior_record,
            prior_facts,
        )
        if (
            require_same_entity
            and comparison.affected_entity_relation != AffectedEntityRelation.SAME.value
        ):
            break
        if comparison.prior_fault_outcome not in {
            FaultOutcome.TERMINAL.value,
            FaultOutcome.UNRESOLVED.value,
        }:
            break
        if comparison.relation not in {
            HistoryProgressRelation.SAME.value,
            HistoryProgressRelation.REGRESSED.value,
        }:
            break
        count += 1
    return count


def _consecutive_job_relations(
    comparisons: Sequence[HistoryProgressComparison],
    qualifying_relations: set[str],
) -> int:
    count = 0
    for comparison in reversed(comparisons):
        if comparison.relation not in qualifying_relations:
            break
        count += 1
    return count


def _same_observer_count(
    current_facts: AttemptFailureFacts,
    prior_facts: AttemptFailureFacts,
) -> bool:
    current = current_facts.root_observer_ranks
    prior = prior_facts.root_observer_ranks
    return current is not None and prior is not None and len(current) == len(prior)


def _same_unattributed_count(
    current_facts: AttemptFailureFacts,
    prior_facts: AttemptFailureFacts,
) -> bool:
    current = current_facts.unattributed_root_occurrence_count
    prior = prior_facts.unattributed_root_occurrence_count
    return current is not None and prior is not None and current == prior


def _affected_entity_relation(
    current_facts: AttemptFailureFacts,
    prior_facts: AttemptFailureFacts,
) -> str:
    current = current_facts.affected_entity
    prior = prior_facts.affected_entity
    if current is None or prior is None:
        return AffectedEntityRelation.UNKNOWN.value
    if current.kind == prior.kind and current.fingerprint == prior.fingerprint:
        return AffectedEntityRelation.SAME.value
    return AffectedEntityRelation.DIFFERENT.value
