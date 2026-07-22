# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic L3 comparison over immutable attempt records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..models import (
    AttemptFailureFacts,
    AttemptProgressSummary,
    AttemptRecord,
    FaultOutcome,
    HistoryDimensionComparison,
    HistoryProgressComparison,
    HistoryProgressRelation,
    HistorySummary,
    PriorAttemptView,
)

DETERMINISTIC_FACT_SELECTOR = "deterministic"


@dataclass(frozen=True)
class HistoryEvaluationInput:
    """Current record, selected facts, and immutable earlier-attempt view."""

    current_record: AttemptRecord
    fact_selector: str
    prior_attempts: PriorAttemptView


def evaluate_history(history_input: HistoryEvaluationInput) -> HistorySummary:
    prior_view = history_input.prior_attempts
    current_facts = _selected_current_facts(
        history_input.current_record,
        history_input.fact_selector,
    )
    if not prior_view.available:
        return HistorySummary(
            available=False,
            availability_reason=prior_view.availability_reason,
        )
    if current_facts is None or not current_facts.root_fingerprint:
        return HistorySummary(
            available=False,
            availability_reason="missing_root_fingerprint",
            same_job_attempts=len(prior_view.records),
        )

    ordered = tuple(sorted(prior_view.records, key=lambda record: record.cycle_id))
    matching = tuple(
        record
        for record in ordered
        if record.deterministic.root_fingerprint == current_facts.root_fingerprint
    )
    comparisons = tuple(
        _compare_progress(history_input.current_record, current_facts, record)
        for record in matching
    )
    qualifying_pairs = tuple(
        (record, comparison)
        for record, comparison in zip(matching, comparisons)
        if comparison.prior_fault_outcome
        in {FaultOutcome.TERMINAL.value, FaultOutcome.UNRESOLVED.value}
    )
    qualifying = tuple(comparison for _record, comparison in qualifying_pairs)
    qualifying_records = tuple(record for record, _comparison in qualifying_pairs)

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
    same_data_position = sum(
        item.same_data_position
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

    rank_matches = current_facts.faulting_rank is not None and any(
        record.deterministic.faulting_rank == current_facts.faulting_rank
        for record in qualifying_records
    )
    node_matches = current_facts.faulting_node is not None and any(
        record.deterministic.faulting_node == current_facts.faulting_node
        for record in qualifying_records
    )
    cross_node = current_facts.faulting_node is not None and any(
        record.deterministic.faulting_node is not None
        and record.deterministic.faulting_node != current_facts.faulting_node
        for record in qualifying_records
    )
    gpu_matches = current_facts.faulting_gpu is not None and any(
        record.deterministic.faulting_gpu == current_facts.faulting_gpu
        for record in qualifying_records
    )

    return HistorySummary(
        available=True,
        availability_reason="ready",
        same_job_attempts=len(ordered),
        matching_root_attempts=len(matching),
        comparisons=comparisons,
        observed_advance_attempts=observed_advance,
        same_progress_attempts=same_progress,
        regressed_progress_attempts=regressed,
        unknown_progress_attempts=unknown,
        no_observed_advance_attempts=no_observed_advance,
        matching_root_attempts_with_observed_training_progress=sum(
            record.progress.training_progress == "observed" for record in matching
        ),
        matching_root_attempts_before_observed_training_progress=sum(
            record.progress.failure_position == "before_observed_training_progress"
            for record in matching
        ),
        matching_root_attempts_with_unknown_training_progress=sum(
            record.progress.training_progress == "unknown" for record in matching
        ),
        exact_failure_position_attempts=exact_failure_position,
        same_rank_iteration_attempts=same_rank_iteration,
        same_data_position_attempts=same_data_position,
        same_artifact_attempts=sum(item.same_artifact for item in qualifying),
        consecutive_same_root_no_advance_attempts=_consecutive_same_root_no_advance(
            history_input.current_record,
            current_facts,
            ordered,
        ),
        advanced_beyond_all_comparable_attempts=bool(comparable)
        and all(item.relation == HistoryProgressRelation.ADVANCED.value for item in comparable),
        cross_node_recurrence=cross_node,
        same_node_recurrence=node_matches,
        same_gpu_recurrence=gpu_matches,
        same_rank_only_recurrence=rank_matches and not node_matches and not gpu_matches,
        rank_to_gpu_mapping_available=any(
            record.deterministic.rank_to_gpu_map for record in qualifying_records
        ),
    )


def _selected_current_facts(
    record: AttemptRecord,
    selector: str,
) -> AttemptFailureFacts | None:
    if selector == DETERMINISTIC_FACT_SELECTOR:
        return record.deterministic
    for entry in record.enriched:
        if entry.route_id == selector:
            return entry.facts
    return None


def _compare_progress(
    current_record: AttemptRecord,
    current_facts: AttemptFailureFacts,
    prior_record: AttemptRecord,
) -> HistoryProgressComparison:
    prior_facts = prior_record.deterministic
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
        same_data_position=(
            bool(current_facts.data_position_fingerprint)
            and prior_facts.data_position_fingerprint == current_facts.data_position_fingerprint
        ),
        same_artifact=bool(
            current_facts.artifact_path and prior_facts.artifact_path == current_facts.artifact_path
        ),
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
) -> int:
    count = 0
    for prior_record in reversed(ordered):
        if prior_record.deterministic.root_fingerprint != current_facts.root_fingerprint:
            break
        comparison = _compare_progress(current_record, current_facts, prior_record)
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
