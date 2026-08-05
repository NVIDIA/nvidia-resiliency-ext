# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Domain types for nvrx-watch: the immutable Snapshot the pipeline builds,
and the Finding/Action it emits.

Everything a detector is allowed to see lives in :class:`Snapshot`. Detectors are pure
functions of (snapshot, config), which is what makes them testable without a cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

# Severities, in the order a pager cares about them.
INFO = "info"
WARNING = "warning"
CRITICAL = "critical"

# Capabilities a snapshot may or may not carry. A source that failed marks its
# capability missing, and detectors requiring it are skipped rather than run against
# partial data.
CAP_PLATFORM = "platform"
CAP_CYCLES = "cycles"
CAP_CHECKPOINT = "checkpoint"

# Slurm states that mean "this task may still do something".
LIVE_STATES = frozenset(
    {"RUNNING", "PENDING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING", "CONFIGURING"}
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TaskInfo:
    """One array task (Slurm) or one worker replica (K8s)."""

    task: int
    state: str
    exit_code: int | None = None
    end_time: datetime | None = None

    @property
    def is_live(self) -> bool:
        return self.state.upper().split()[0] in LIVE_STATES if self.state else False


@dataclass(frozen=True)
class ChainGeneration:
    """One array in the singleton chain: the unit that is replaced wholesale."""

    gen_id: str
    tasks: tuple[TaskInfo, ...] = ()

    @property
    def pending(self) -> tuple[TaskInfo, ...]:
        return tuple(t for t in self.tasks if t.state.upper() == "PENDING")

    @property
    def task0(self) -> TaskInfo | None:
        return next((t for t in self.tasks if t.task == 0), None)

    @property
    def has_live_task(self) -> bool:
        return any(t.is_live for t in self.tasks)


@dataclass(frozen=True)
class CycleRecord:
    """One NVRx restart cycle, parsed from a cycle_info.<job>.<attempt>.<cycle> file.

    ``job_id`` is the array job id of the generation that produced it, so cycles from
    every generation ever run under the work dir coexist here and are grouped by it.
    """

    job_id: str
    attempt_index: int
    cycle_number: int
    start_time: datetime | None = None
    end_time: datetime | None = None
    active_nodes: str = ""
    standby_nodes: str = ""
    log_file: str = ""
    path: str = ""

    @property
    def key(self) -> str:
        return f"{self.job_id}.{self.attempt_index}.{self.cycle_number}"

    @property
    def is_open(self) -> bool:
        """True while the cycle is running: NVRx writes cycle_end_time when it ends."""
        return self.end_time is None

    def duration_seconds(self, now: datetime | None = None) -> float | None:
        if self.start_time is None:
            return None
        end = self.end_time or now
        if end is None:
            return None
        return (end - self.start_time).total_seconds()


@dataclass(frozen=True)
class CheckpointProgress:
    """Contents and mtime of the launcher's --ft-checkpoint-iteration-file."""

    value: int | None = None
    mtime: datetime | None = None


@dataclass(frozen=True)
class PriorState:
    """What the previous pass saw. Turns a point-in-time snapshot into a stall timer."""

    checkpoint_value: int | None = None
    checkpoint_first_seen: datetime | None = None
    latest_cycle_key: str | None = None
    latest_cycle_first_seen: datetime | None = None
    last_pass: datetime | None = None


@dataclass(frozen=True)
class Action:
    """A corrective step a finding carries. The runner applies it unless dry-run."""

    kind: str  # "cancel_pending"
    target: str
    description: str = ""


@dataclass(frozen=True)
class Finding:
    key: str  # stable dedup key; the pager collapses repeats on it
    detector: str
    severity: str
    summary: str
    detail: str = ""
    action: Action | None = None


@dataclass(frozen=True)
class Snapshot:
    """Everything one pass observed."""

    observed_at: datetime = field(default_factory=utcnow)
    job_name: str = ""
    capabilities: frozenset[str] = frozenset()
    generations: tuple[ChainGeneration, ...] = ()
    cycles: tuple[CycleRecord, ...] = ()
    checkpoint: CheckpointProgress = CheckpointProgress()
    prior: PriorState = PriorState()
    chain_expected: bool = False
    max_restarts: int | None = None
    # Filled by the platform source on demand: (gen_id, task) -> TaskInfo from sacct.
    terminal_info: dict[tuple[str, int], TaskInfo] = field(default_factory=dict)
    # (generation id, its task 0 record) for generations that ended inside the churn
    # window. Accounting-derived, so it covers generations already out of the queue.
    recent_endings: tuple[tuple[str, TaskInfo], ...] = ()

    def has(self, *caps: str) -> bool:
        return all(c in self.capabilities for c in caps)

    @property
    def current_job_id(self) -> str | None:
        """Job id of the generation that produced the most recent cycle."""
        latest = self.latest_cycle
        return latest.job_id if latest else None

    @property
    def latest_cycle(self) -> CycleRecord | None:
        dated = [c for c in self.cycles if c.start_time is not None]
        if not dated:
            return None
        return max(dated, key=lambda c: (c.start_time, c.attempt_index, c.cycle_number))

    def cycles_of(self, job_id: str | None) -> tuple[CycleRecord, ...]:
        if job_id is None:
            return ()
        selected = [c for c in self.cycles if c.job_id == job_id and c.start_time is not None]
        selected.sort(key=lambda c: (c.start_time, c.attempt_index, c.cycle_number))
        return tuple(selected)

    @property
    def current_cycles(self) -> tuple[CycleRecord, ...]:
        """Cycles of the current generation only; a new generation resets the history."""
        return self.cycles_of(self.current_job_id)


def sorted_findings(findings: Iterable[Finding]) -> list[Finding]:
    order = {CRITICAL: 0, WARNING: 1, INFO: 2}
    return sorted(findings, key=lambda f: (order.get(f.severity, 3), f.detector, f.key))


def summarize(findings: Sequence[Finding]) -> str:
    if not findings:
        return "no findings"
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    return ", ".join(f"{counts[s]} {s}" for s in (CRITICAL, WARNING, INFO) if s in counts)
