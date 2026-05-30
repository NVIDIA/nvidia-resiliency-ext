# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from nvidia_resiliency_ext.attribution.fact.models import (
    AvoidCandidate,
    AvoidDecision,
    FactHistoryRecord,
)


def compute_repeat_offender_decision(
    *,
    cycle_id: int,
    current_suspect_nodes: Iterable[str],
    history_records: Iterable[FactHistoryRecord],
    hot_cache_records: Iterable[FactHistoryRecord],
    max_candidate_nodes: int,
    min_repeat_count_for_avoid: int,
    max_avoids_per_cycle: int,
) -> AvoidDecision:
    suspects = sorted({str(node) for node in current_suspect_nodes if str(node)})
    if not suspects:
        return AvoidDecision(cycle_id=cycle_id, status="skipped")

    if len(suspects) > max_candidate_nodes:
        return AvoidDecision(cycle_id=cycle_id, status="skipped")

    prior_by_node: dict[str, dict[tuple[str, str, str], FactHistoryRecord]] = {
        node: {} for node in suspects
    }
    for record in list(history_records) + list(hot_cache_records):
        if record.node not in prior_by_node:
            continue
        prior_by_node[record.node][(record.cluster, record.node, record.episode_id)] = record

    candidates = []
    for node in suspects:
        prior_records = list(prior_by_node[node].values())
        repeat_count = 1 + len(prior_records)
        if repeat_count < min_repeat_count_for_avoid:
            continue
        prior_last_seen = _max_event_time(prior_records)
        candidates.append(
            AvoidCandidate(
                node=node,
                repeat_count=repeat_count,
                prior_last_seen=prior_last_seen,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate.repeat_count,
            -_timestamp(candidate.prior_last_seen),
            candidate.node,
        )
    )
    avoid_nodes = [candidate.node for candidate in candidates[: max(0, max_avoids_per_cycle)]]
    return AvoidDecision(
        cycle_id=cycle_id,
        status="ready",
        avoid_nodes=avoid_nodes,
        candidates=candidates,
    )


def _max_event_time(records: Iterable[FactHistoryRecord]) -> datetime | None:
    event_times = [record.event_time for record in records]
    if not event_times:
        return None
    return max(event_times)


def _timestamp(value: datetime | None) -> float:
    if value is None:
        return 0.0
    return value.timestamp()
