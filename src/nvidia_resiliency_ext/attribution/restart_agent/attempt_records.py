# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attempt-record assembly, bounded storage, and test control surfaces."""

from __future__ import annotations

from dataclasses import replace
from threading import RLock
from typing import Literal, Protocol, Sequence

from .l2.failure_facts import build_attempt_failure_facts
from .models import (
    AttemptFailureFacts,
    AttemptFailureFactsSource,
    AttemptProgressSummary,
    AttemptRecord,
    DecisionEvidence,
    EnrichedAttemptFacts,
    L0Bundle,
    PriorAttemptView,
    normalize_attempt_records,
)

DEFAULT_MAX_ATTEMPTS_PER_JOB = 10
DEFAULT_MAX_TOTAL_RECORDS = 3000


class AttemptRecordStore(Protocol):
    """Runtime-owned storage contract for current-process attempt history."""

    @property
    def enabled(self) -> bool: ...

    def get_prior_attempts(self, job_id: str, before_cycle_id: int) -> PriorAttemptView: ...

    def upsert_attempt(self, record: AttemptRecord) -> None: ...

    def records(self, job_id: str | None = None) -> tuple[AttemptRecord, ...]: ...

    def replace(self, records: Sequence[AttemptRecord]) -> None: ...

    def clear(self, job_id: str | None = None) -> None: ...

    def metrics(self) -> dict[str, int | bool]: ...


class InMemoryAttemptRecordStore:
    """Thread-safe bounded attempt records with idempotent key replacement."""

    def __init__(
        self,
        *,
        max_attempts_per_job: int = DEFAULT_MAX_ATTEMPTS_PER_JOB,
        max_total_records: int = DEFAULT_MAX_TOTAL_RECORDS,
    ) -> None:
        self._max_attempts_per_job = _positive_int(max_attempts_per_job, "max_attempts_per_job")
        self._max_total_records = _positive_int(max_total_records, "max_total_records")
        self._records: dict[tuple[str, int], AttemptRecord] = {}
        self._insertion_sequence: dict[tuple[str, int], int] = {}
        self._next_sequence = 0
        self._eviction_count = 0
        self._lock = RLock()

    @property
    def enabled(self) -> bool:
        return True

    def get_prior_attempts(self, job_id: str, before_cycle_id: int) -> PriorAttemptView:
        with self._lock:
            records = tuple(
                sorted(
                    (
                        record
                        for (stored_job, cycle_id), record in self._records.items()
                        if stored_job == job_id and cycle_id < before_cycle_id
                    ),
                    key=lambda item: item.cycle_id,
                )
            )
        return PriorAttemptView(
            records=records[-self._max_attempts_per_job :],
            available=True,
            availability_reason="ready",
        )

    def upsert_attempt(self, record: AttemptRecord) -> None:
        _validate_storable_record(record)
        key = (record.job_id, record.cycle_id)
        with self._lock:
            if key not in self._records:
                self._insertion_sequence[key] = self._next_sequence
                self._next_sequence += 1
            self._records[key] = record
            self._evict_per_job(record.job_id)
            self._evict_total()

    def records(self, job_id: str | None = None) -> tuple[AttemptRecord, ...]:
        with self._lock:
            selected = (
                record
                for (stored_job, _cycle_id), record in self._records.items()
                if job_id is None or stored_job == job_id
            )
            return tuple(sorted(selected, key=lambda item: (item.job_id, item.cycle_id)))

    def replace(self, records: Sequence[AttemptRecord]) -> None:
        normalized = normalize_attempt_records(records)
        for record in normalized:
            _validate_storable_record(record)
        with self._lock:
            self._records.clear()
            self._insertion_sequence.clear()
            self._next_sequence = 0
            for record in normalized:
                self._upsert_locked(record)

    def clear(self, job_id: str | None = None) -> None:
        with self._lock:
            if job_id is None:
                self._records.clear()
                self._insertion_sequence.clear()
                return
            keys = [key for key in self._records if key[0] == job_id]
            for key in keys:
                self._remove_locked(key)

    def metrics(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "enabled": True,
                "record_count": len(self._records),
                "job_count": len({job_id for job_id, _cycle_id in self._records}),
                "enriched_entry_count": sum(
                    len(record.enriched) for record in self._records.values()
                ),
                "eviction_count": self._eviction_count,
                "max_attempts_per_job": self._max_attempts_per_job,
                "max_total_records": self._max_total_records,
            }

    def _upsert_locked(self, record: AttemptRecord) -> None:
        key = (record.job_id, record.cycle_id)
        if key not in self._records:
            self._insertion_sequence[key] = self._next_sequence
            self._next_sequence += 1
        self._records[key] = record
        self._evict_per_job(record.job_id)
        self._evict_total()

    def _evict_per_job(self, job_id: str) -> None:
        job_keys = sorted(
            (key for key in self._records if key[0] == job_id),
            key=lambda key: key[1],
        )
        while len(job_keys) > self._max_attempts_per_job:
            self._remove_locked(job_keys.pop(0))

    def _evict_total(self) -> None:
        while len(self._records) > self._max_total_records:
            oldest = min(self._insertion_sequence, key=self._insertion_sequence.__getitem__)
            self._remove_locked(oldest)

    def _remove_locked(self, key: tuple[str, int]) -> None:
        if key in self._records:
            self._records.pop(key)
            self._insertion_sequence.pop(key, None)
            self._eviction_count += 1


class NullAttemptRecordStore:
    """Explicit disabled-history store used by the composition root."""

    @property
    def enabled(self) -> bool:
        return False

    def get_prior_attempts(self, job_id: str, before_cycle_id: int) -> PriorAttemptView:
        return PriorAttemptView(availability_reason="history_disabled")

    def upsert_attempt(self, record: AttemptRecord) -> None:
        return None

    def records(self, job_id: str | None = None) -> tuple[AttemptRecord, ...]:
        return ()

    def replace(self, records: Sequence[AttemptRecord]) -> None:
        raise ValueError("history_disabled")

    def clear(self, job_id: str | None = None) -> None:
        return None

    def metrics(self) -> dict[str, int | bool]:
        return {
            "enabled": False,
            "record_count": 0,
            "job_count": 0,
            "enriched_entry_count": 0,
            "eviction_count": 0,
        }


class AttemptRecordControl:
    """Transport-independent test seam for seeding and inspecting records."""

    def __init__(self, store: AttemptRecordStore) -> None:
        self._store = store

    def seed(
        self,
        records: Sequence[AttemptRecord | dict[str, object]],
        *,
        mode: Literal["replace", "merge"] = "replace",
    ) -> None:
        if not self._store.enabled:
            raise ValueError("history_disabled")
        normalized = normalize_attempt_records(records)
        if mode == "replace":
            self._store.replace(normalized)
        elif mode == "merge":
            for record in normalized:
                self._store.upsert_attempt(record)
        else:
            raise ValueError("attempt-record seed mode must be 'replace' or 'merge'")

    def records(self, job_id: str | None = None) -> tuple[AttemptRecord, ...]:
        return self._store.records(job_id)

    def clear(self, job_id: str | None = None) -> None:
        self._store.clear(job_id)


class AttemptRecordAssembler:
    """Build immutable initial records and route-keyed enriched replacements."""

    def initial_record(
        self,
        *,
        job_id: str,
        cycle_id: int,
        bundle: L0Bundle,
        decision_evidence: DecisionEvidence,
    ) -> AttemptRecord:
        deterministic = build_attempt_failure_facts(
            decision_evidence.deterministic_primary_candidate,
            decision_evidence,
            source=AttemptFailureFactsSource.L0_DETERMINISTIC,
        )
        return AttemptRecord(
            job_id=job_id,
            cycle_id=cycle_id,
            progress=build_attempt_progress_summary(bundle, decision_evidence),
            deterministic=deterministic,
            enriched=(),
        )

    def with_enriched(
        self,
        record: AttemptRecord,
        *,
        route_id: str,
        facts: AttemptFailureFacts,
    ) -> AttemptRecord:
        if not route_id:
            raise ValueError("route_id is required for enriched attempt facts")
        entries = {entry.route_id: entry for entry in record.enriched}
        entries[route_id] = EnrichedAttemptFacts(route_id=route_id, facts=facts)
        return replace(
            record,
            enriched=tuple(entries[key] for key in sorted(entries)),
        )


def build_attempt_progress_summary(
    bundle: L0Bundle,
    decision_evidence: DecisionEvidence,
) -> AttemptProgressSummary:
    """Derive route-independent attempt progress from fully scanned L0 facts."""

    completed = _logical_marker_values(bundle.progress.progress_markers, marker_type="iteration")
    checkpoints = _logical_marker_values(bundle.progress.checkpoint_markers)
    primary = decision_evidence.deterministic_primary_candidate
    primary_line = primary.line if primary is not None else None
    progress_lines = sorted(
        {
            marker.line
            for marker in bundle.progress.progress_markers
            if marker.marker_type == "iteration"
        }
    )
    if primary_line is None:
        failure_position = "unknown"
        progress_after_failure = "unknown"
    else:
        prior_progress = any(line <= primary_line for line in progress_lines)
        failure_position = (
            "after_observed_training_progress"
            if prior_progress
            else "before_observed_training_progress"
        )
        progress_after_failure = (
            "observed"
            if bundle.run_progress_summary.progress_after_failure_episode is True
            else (
                "not_observed"
                if bundle.run_progress_summary.progress_after_failure_episode is False
                else "unknown"
            )
        )
    return AttemptProgressSummary(
        training_progress="observed" if completed else "not_observed",
        first_completed_step=min(completed) if completed else None,
        last_completed_step=max(completed) if completed else None,
        completed_step_delta=(max(completed) - min(completed) if completed else None),
        progress_marker_count=len(completed),
        checkpoint_progress="observed" if checkpoints else "not_observed",
        checkpoint_load_step=bundle.run_progress_summary.checkpoint_load_iteration,
        first_checkpoint_step=min(checkpoints) if checkpoints else None,
        last_checkpoint_step=max(checkpoints) if checkpoints else None,
        checkpoint_step_delta=(max(checkpoints) - min(checkpoints) if checkpoints else None),
        checkpoint_marker_count=len(checkpoints),
        failure_position=failure_position,
        progress_after_failure=progress_after_failure,
    )


def _logical_marker_values(markers: Sequence[object], marker_type: str | None = None) -> list[int]:
    values = {
        int(marker.value)
        for marker in markers
        if isinstance(getattr(marker, "value", None), int)
        and (marker_type is None or getattr(marker, "marker_type", None) == marker_type)
    }
    return sorted(values)


def _validate_storable_record(record: AttemptRecord) -> None:
    if not record.job_id:
        raise ValueError("AttemptRecord job_id is required")
    if isinstance(record.cycle_id, bool) or not isinstance(record.cycle_id, int):
        raise TypeError("AttemptRecord cycle_id must be an integer")
    if not record.deterministic.root_fingerprint:
        raise ValueError("AttemptRecord deterministic root_fingerprint is required")
    if not record.deterministic.root_fingerprint_source:
        raise ValueError("AttemptRecord deterministic root_fingerprint_source is required")


def _positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be greater than zero")
    return value
