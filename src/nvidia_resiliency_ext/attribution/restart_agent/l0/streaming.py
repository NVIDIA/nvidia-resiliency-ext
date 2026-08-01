# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One resumable L0A byte-to-evidence path for terminal and progressive use."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any, Mapping

from ..infrastructure.log_source import (
    ChunkedLogReader,
    DecodedLine,
    IncrementalLineDecoder,
    IndexedFileLineStore,
    LogSnapshot,
    SourceBoundary,
)
from ..models import DecisionEvidence, L0Bundle
from ..runtime import SYSTEM_CLOCK, Clock
from .assembly import L0ObservationAccumulator, build_l0_bundle_from_observations
from .decision import build_decision_evidence


class ProgressiveSourceUnavailable(OSError):
    """Raised when finalization has no usable source bytes."""


@dataclass(frozen=True)
class ProgressiveL0State:
    """Observable state for one resumable L0A accumulator."""

    phase: str
    source_boundary: SourceBoundary | None
    read_mode: str
    chunk_bytes: int
    encoding: str
    line_count: int
    pending_line_bytes: int
    discarded_incomplete_tail_bytes: int
    read_offset: int
    poll_count: int
    chunk_count: int
    growth_count: int
    unchanged_poll_count: int
    missing_poll_count: int
    reset_count: int
    decode_replacement_count: int
    decode_replacement_line_count: int
    bytes_ingested: int
    bytes_reread: int
    l0a_build_count: int
    source_decode_wall_clock_s: float
    source_index_classify_wall_clock_s: float
    source_ingest_wall_clock_s: float
    l0a_reduction_wall_clock_s: float
    last_growth_monotonic: float | None
    last_poll_monotonic: float | None
    last_error: str | None

    def to_payload(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "source_boundary": (
                self.source_boundary.to_payload() if self.source_boundary is not None else None
            ),
        }


@dataclass(frozen=True)
class _L0Checkpoint:
    bundle: L0Bundle
    decision_evidence: DecisionEvidence
    source_boundary: SourceBoundary
    l0a_wall_clock_s: float
    decision_evidence_wall_clock_s: float


@dataclass(frozen=True)
class FinalizedL0A:
    """Canonical L0A evidence and immutable source boundary."""

    bundle: L0Bundle
    decision_evidence: DecisionEvidence
    source_log: LogSnapshot
    source_boundary: SourceBoundary
    l0a_wall_clock_s: float
    decision_evidence_wall_clock_s: float
    precomputed: bool
    progressive_metrics: Mapping[str, Any]
    canonical_hash: str


class ProgressiveL0Accumulator:
    """Incrementally ingest source bytes and reduce reusable L0A checkpoints."""

    def __init__(
        self,
        log_path: str,
        *,
        reader: ChunkedLogReader | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._log_path = log_path
        self._reader = reader or ChunkedLogReader()
        self._clock = clock
        self._lock = RLock()
        self._boundary: SourceBoundary | None = None
        self._read_offset = 0
        self._encoding = "utf-8"
        self._decoder = IncrementalLineDecoder(encoding=self._encoding)
        self._observations = L0ObservationAccumulator()
        self._source_store: IndexedFileLineStore | None = None
        self._checkpoint: _L0Checkpoint | None = None
        self._phase = "registered"
        self._poll_count = 0
        self._chunk_count = 0
        self._growth_count = 0
        self._unchanged_poll_count = 0
        self._missing_poll_count = 0
        self._reset_count = 0
        self._bytes_ingested = 0
        self._bytes_reread = 0
        self._l0a_build_count = 0
        self._source_decode_wall_clock_s = 0.0
        self._source_index_classify_wall_clock_s = 0.0
        self._l0a_reduction_wall_clock_s = 0.0
        self._last_growth_monotonic: float | None = None
        self._last_poll_monotonic: float | None = None
        self._last_error: str | None = None
        self._discarded_incomplete_tail_bytes = 0
        self._finalized = False

    @property
    def log_path(self) -> str:
        return self._log_path

    def state(self) -> ProgressiveL0State:
        with self._lock:
            return ProgressiveL0State(
                phase=self._phase,
                source_boundary=self._boundary,
                read_mode=self._reader.read_mode,
                chunk_bytes=self._reader.chunk_bytes,
                encoding=self._encoding,
                line_count=len(self._observations),
                pending_line_bytes=self._decoder.pending_bytes,
                discarded_incomplete_tail_bytes=self._discarded_incomplete_tail_bytes,
                read_offset=self._read_offset,
                poll_count=self._poll_count,
                chunk_count=self._chunk_count,
                growth_count=self._growth_count,
                unchanged_poll_count=self._unchanged_poll_count,
                missing_poll_count=self._missing_poll_count,
                reset_count=self._reset_count,
                decode_replacement_count=self._decoder.decode_replacement_count,
                decode_replacement_line_count=(self._decoder.decode_replacement_line_count),
                bytes_ingested=self._bytes_ingested,
                bytes_reread=self._bytes_reread,
                l0a_build_count=self._l0a_build_count,
                source_decode_wall_clock_s=round(
                    self._source_decode_wall_clock_s,
                    6,
                ),
                source_index_classify_wall_clock_s=round(
                    self._source_index_classify_wall_clock_s,
                    6,
                ),
                source_ingest_wall_clock_s=round(
                    self._source_decode_wall_clock_s + self._source_index_classify_wall_clock_s,
                    6,
                ),
                l0a_reduction_wall_clock_s=round(
                    self._l0a_reduction_wall_clock_s,
                    6,
                ),
                last_growth_monotonic=self._last_growth_monotonic,
                last_poll_monotonic=self._last_poll_monotonic,
                last_error=self._last_error,
            )

    def refresh(self, *, precompute: bool = True) -> bool:
        """Ingest new bytes, optionally reducing an updated L0A checkpoint."""

        with self._lock:
            if self._finalized:
                raise RuntimeError("cannot refresh finalized L0A state")
            self._phase = "precomputing"
            self._poll_count += 1
            self._last_poll_monotonic = self._clock.monotonic()
            try:
                boundary = self._reader.boundary(self._log_path)
            except FileNotFoundError:
                self._missing_poll_count += 1
                self._phase = "waiting"
                self._last_error = None
                return False
            except OSError as exc:
                self._phase = "waiting"
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise

            change_kind = _boundary_change(self._boundary, boundary)
            if change_kind == "unchanged":
                self._unchanged_poll_count += 1
                if (
                    precompute
                    and boundary.byte_size
                    and (self._checkpoint is None or self._checkpoint.source_boundary != boundary)
                ):
                    self._build_checkpoint()
                self._phase = "waiting"
                self._last_error = None
                return False

            previous = self._boundary
            if change_kind in {"initial", "growth"}:
                if previous is None:
                    self._bytes_ingested += boundary.byte_size
                else:
                    self._bytes_ingested += boundary.byte_size - previous.byte_size
            else:
                self._reset_count += 1
                self._bytes_reread += boundary.byte_size
                self._reset_analysis(encoding="utf-8")

            self._ingest_to_boundary(boundary)

            self._boundary = boundary
            self._growth_count += 1
            self._last_growth_monotonic = self._clock.monotonic()
            self._last_error = None
            if boundary.byte_size and precompute:
                self._build_checkpoint()
            elif not boundary.byte_size:
                self._checkpoint = None
            self._phase = "waiting"
            return True

    def finalize(self, *, include_incomplete_tail: bool = True) -> FinalizedL0A:
        """Finalize one boundary, optionally excluding an actively written tail."""

        with self._lock:
            if self._finalized:
                raise RuntimeError("L0A state is already finalized")
            self._phase = "finalizing"
            try:
                changed = self.refresh(precompute=False)
            except OSError:
                self._reset_analysis(encoding="utf-8")
                self._boundary = None
                self._reset_count += 1
                changed = self.refresh(precompute=False)

            boundary = self._boundary
            if boundary is None:
                raise ProgressiveSourceUnavailable(f"log path is missing: {self._log_path}")
            if boundary.byte_size == 0:
                raise ProgressiveSourceUnavailable(f"log path is empty: {self._log_path}")

            decode_started = self._clock.monotonic()
            final_records = (
                self._decoder.feed_records(b"", final=True) if include_incomplete_tail else ()
            )
            self._source_decode_wall_clock_s += self._clock.monotonic() - decode_started
            if not include_incomplete_tail:
                self._discarded_incomplete_tail_bytes = self._decoder.discard_pending()
            self._append_records(final_records)
            if final_records:
                changed = True
            reused_checkpoint = (
                not changed
                and self._checkpoint is not None
                and self._checkpoint.source_boundary == boundary
            )
            if not reused_checkpoint:
                self._build_checkpoint()

            checkpoint = self._checkpoint
            if checkpoint is None:
                raise AssertionError("L0A finalization did not produce a checkpoint")

            self._finalized = True
            self._phase = "completed"
            precomputed = reused_checkpoint
            source_store = self._source_store
            if source_store is None:
                raise AssertionError("L0A finalization did not create a source index")
            source_log = LogSnapshot.from_line_store(
                path=self._log_path,
                line_store=source_store,
                byte_size=boundary.byte_size,
                source_boundary=boundary,
                encoding=self._encoding,
                read_mode=self._reader.read_mode,
            )
            metrics = self.state().to_payload()
            return FinalizedL0A(
                bundle=checkpoint.bundle,
                decision_evidence=checkpoint.decision_evidence,
                source_log=source_log,
                source_boundary=boundary,
                l0a_wall_clock_s=(0.0 if precomputed else checkpoint.l0a_wall_clock_s),
                decision_evidence_wall_clock_s=(
                    0.0 if precomputed else checkpoint.decision_evidence_wall_clock_s
                ),
                precomputed=precomputed,
                progressive_metrics=metrics,
                canonical_hash=canonical_l0a_hash(
                    checkpoint.bundle,
                    checkpoint.decision_evidence,
                ),
            )

    def _ingest_to_boundary(self, boundary: SourceBoundary) -> None:
        if self._source_store is None:
            self._source_store = IndexedFileLineStore(
                self._log_path,
                boundary=boundary,
                encoding=self._encoding,
                read_chunk_bytes=self._reader.chunk_bytes,
            )
            self._observations = L0ObservationAccumulator(line_store=self._source_store)
        else:
            self._source_store.update_boundary(boundary)
        for chunk in self._reader.chunks(
            self._log_path,
            boundary=boundary,
            start_offset=self._read_offset,
        ):
            self._chunk_count += 1
            self._read_offset += len(chunk)
            decode_started = self._clock.monotonic()
            records = self._decoder.feed_records(chunk)
            self._source_decode_wall_clock_s += self._clock.monotonic() - decode_started
            self._append_records(records)

    def _append_records(self, records: tuple[DecodedLine, ...]) -> None:
        classify_started = self._clock.monotonic()
        for record in records:
            self._observations.append_record(record)
        self._source_index_classify_wall_clock_s += self._clock.monotonic() - classify_started

    def _reset_analysis(self, *, encoding: str) -> None:
        if self._source_store is not None:
            self._source_store.close()
        self._read_offset = 0
        self._encoding = encoding
        self._decoder = IncrementalLineDecoder(encoding=encoding)
        self._observations = L0ObservationAccumulator()
        self._source_store = None
        self._checkpoint = None

    def _build_checkpoint(self) -> None:
        boundary = self._boundary
        if boundary is None:
            raise AssertionError("cannot build an L0A checkpoint without a source boundary")
        l0a_started = self._clock.monotonic()
        bundle = build_l0_bundle_from_observations(
            self._log_path,
            byte_size=self._read_offset,
            observations=self._observations,
        )
        l0a_wall_clock_s = self._clock.monotonic() - l0a_started
        decision_started = self._clock.monotonic()
        decision_evidence = build_decision_evidence(bundle)
        decision_wall_clock_s = self._clock.monotonic() - decision_started
        self._checkpoint = _L0Checkpoint(
            bundle=bundle,
            decision_evidence=decision_evidence,
            source_boundary=boundary,
            l0a_wall_clock_s=round(l0a_wall_clock_s, 3),
            decision_evidence_wall_clock_s=round(decision_wall_clock_s, 3),
        )
        self._l0a_build_count += 1
        self._l0a_reduction_wall_clock_s += l0a_wall_clock_s + decision_wall_clock_s


def finalize_log_snapshot(
    source_log: LogSnapshot,
    *,
    l0_bundle: L0Bundle | None = None,
    clock: Clock = SYSTEM_CLOCK,
) -> FinalizedL0A:
    """Finalize an immutable snapshot through the same observation reducer."""

    l0a_started = clock.monotonic()
    if l0_bundle is None:
        observations = L0ObservationAccumulator(line_store=source_log)
        for item in source_log.log_lines():
            observations.observe_existing(item)
        bundle = build_l0_bundle_from_observations(
            source_log.path,
            byte_size=source_log.byte_size,
            observations=observations,
        )
    else:
        bundle = l0_bundle
    l0a_wall_clock_s = clock.monotonic() - l0a_started
    decision_started = clock.monotonic()
    decision_evidence = build_decision_evidence(bundle)
    decision_wall_clock_s = clock.monotonic() - decision_started
    boundary = source_log.source_boundary or SourceBoundary(
        device=0,
        inode=0,
        byte_size=source_log.byte_size,
        mtime_ns=0,
    )
    return FinalizedL0A(
        bundle=bundle,
        decision_evidence=decision_evidence,
        source_log=source_log,
        source_boundary=boundary,
        l0a_wall_clock_s=round(l0a_wall_clock_s, 3),
        decision_evidence_wall_clock_s=round(decision_wall_clock_s, 3),
        precomputed=l0_bundle is not None,
        progressive_metrics={},
        canonical_hash=canonical_l0a_hash(bundle, decision_evidence),
    )


def canonical_l0a_payload(
    bundle: L0Bundle,
    decision_evidence: DecisionEvidence,
) -> dict[str, Any]:
    """Return the policy-relevant L0A projection used by equivalence tests."""

    return {
        "l0_bundle": asdict(bundle),
        "decision_evidence": decision_evidence.to_payload(),
    }


def canonical_l0a_hash(
    bundle: L0Bundle,
    decision_evidence: DecisionEvidence,
) -> str:
    payload = canonical_l0a_payload(bundle, decision_evidence)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _boundary_change(
    previous: SourceBoundary | None,
    current: SourceBoundary,
) -> str:
    if previous is None:
        return "initial"
    if not previous.same_file(current):
        return "replaced"
    if current.byte_size < previous.byte_size:
        return "truncated"
    if current.byte_size == previous.byte_size and current.mtime_ns == previous.mtime_ns:
        return "unchanged"
    if current.byte_size > previous.byte_size:
        return "growth"
    return "modified_without_growth"
