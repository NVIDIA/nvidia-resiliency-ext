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

"""Detectors: pure functions of (snapshot, config) returning findings.

Two families. Chain reconcilers need a scheduler and correct states the job itself
cannot -- because the thing that was supposed to act (task 0's EXIT trap) is gone.
Restart-anomaly detectors read only NVRx cycle-info files and the checkpoint iteration
file, and find failures that are a pattern rather than an event.

Only one detector carries an action. A watcher that cancels jobs on a heuristic is a
new failure mode; everything else reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Callable

from .config import Config
from .parsing import expand_nodelist
from .types import (
    CAP_CHECKPOINT,
    CAP_CYCLES,
    CAP_PLATFORM,
    CRITICAL,
    INFO,
    WARNING,
    Action,
    Finding,
    Snapshot,
)

DetectorFn = Callable[[Snapshot, Config], list[Finding]]


@dataclass(frozen=True)
class Detector:
    name: str
    requires: tuple[str, ...]
    fn: DetectorFn


def _fmt_age(seconds: float) -> str:
    if seconds < 90:
        return f"{int(seconds)}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


# ---------------------------------------------------------------------------------
# Chain reconciliation
# ---------------------------------------------------------------------------------
def orphaned_generation(snapshot: Snapshot, config: Config) -> list[Finding]:
    """Task 0 is terminal but cold spares are still queued.

    Either its node died so the EXIT trap never ran, or the trap's scancel failed
    silently. Same state either way, same fix. This is the one condition the sbatch
    cannot handle itself.
    """
    findings: list[Finding] = []
    for generation in snapshot.generations:
        task0 = generation.task0
        if task0 is not None and task0.is_live:
            continue  # healthy: task 0 is still in the queue
        pending = generation.pending
        if not pending:
            continue  # draining normally

        record = snapshot.terminal_info.get((generation.gen_id, 0))
        if record is None or not record.state:
            # sacct silent. Unknown is not gone; a watcher that cancels a live
            # generation because accounting lagged is worse than one that waits a pass.
            continue
        if record.is_live:
            continue

        age = None
        if record.end_time is not None:
            age = (snapshot.observed_at - record.end_time).total_seconds()
        if age is not None and age < config.grace:
            continue  # a trap that is mid-flight gets to finish first

        age_text = f"{_fmt_age(age)} ago" if age is not None else "at an unknown time"
        findings.append(
            Finding(
                key=f"nvrx-orphan-{generation.gen_id}",
                detector="orphaned_generation",
                severity=WARNING,
                summary=(
                    f"Generation {generation.gen_id}: task 0 {record.state} {age_text} "
                    f"but {len(pending)} cold spare(s) still queued."
                ),
                detail=(
                    "Task 0's EXIT trap did not release the pool (SIGKILL or node death), "
                    "or its scancel failed. Each queued spare will otherwise be allocated "
                    "a node, fail to reach the dead rendezvous host, and wait out "
                    "store_connect_wait_seconds before exiting."
                ),
                action=Action(
                    kind="cancel_pending",
                    target=generation.gen_id,
                    description=f"scancel --state=PENDING {generation.gen_id}",
                ),
            )
        )
    return findings


def chain_exhausted(snapshot: Snapshot, config: Config) -> list[Finding]:
    """No generation left, running or queued. Nothing else logs this; it is the most
    likely way to lose a night. Reported only when the chain marker exists, so an idle
    account stays quiet."""
    if not snapshot.chain_expected:
        return []
    if any(generation.tasks for generation in snapshot.generations):
        return []
    return [
        Finding(
            key=f"nvrx-chain-exhausted-{snapshot.job_name}",
            detector="chain_exhausted",
            severity=CRITICAL,
            summary=(
                f"Chain '{snapshot.job_name}' has no jobs left (running or queued). "
                "Training has stopped."
            ),
            detail="Resubmit with submit_chain.sh, or remove the chain marker file "
            f"({config.expect_file}) if the run is finished.",
        )
    ]


def chain_not_cancelled(snapshot: Snapshot, config: Config) -> list[Finding]:
    """A generation exited with the no-restart code but successors are still queued.

    task0_exit reads $? to decide whether to cancel the chain. If any hop in
    ft_launcher -> sh -c -> srun rewrote the code, cancel_chain never fires and the
    chain restarts a job NVRx said must not restart -- silently, because every broken
    hop yields 0, which routes to the ordinary branch.
    """
    queued = [g for g in snapshot.generations if g.tasks]
    if not queued:
        return []
    findings: list[Finding] = []
    for gen_id, record in snapshot.recent_endings:
        if record.exit_code != config.no_restart_exit_code:
            continue
        successors = [g for g in queued if g.gen_id != gen_id]
        if not successors:
            continue  # only the ended generation is still draining; nothing to stop
        # Note this is a "no successors left" test, not "the ended generation is absent".
        # cancel_chain scancels by job name, so one call covers both the successor
        # generations and the ended generation's own cold spares -- the failure this
        # detector exists to catch leaves the ended generation queued alongside them.
        # Suppressing on mere overlap would go silent in exactly that case, and stay
        # silent for the whole drain, which is when a human can still act: the successor
        # cannot start until the predecessor's last spare exits.
        findings.append(
            Finding(
                key=f"nvrx-no-restart-not-honoured-{gen_id}",
                detector="chain_not_cancelled",
                severity=CRITICAL,
                summary=(
                    f"Generation {gen_id} exited {config.no_restart_exit_code} (NVRx: do not "
                    f"restart) but {len(successors)} '{snapshot.job_name}' generation(s) are "
                    "still queued or running."
                ),
                detail=(
                    "The chain will reproduce the failure NVRx asked to stop. Cancel the "
                    f"queued generations: scancel --state=PENDING --name={snapshot.job_name}"
                ),
            )
        )
    return findings


def generation_churn(snapshot: Snapshot, config: Config) -> list[Finding]:
    """The chain is burning generations. Each one costs a full restart from checkpoint,
    so a high rate means the run is spending its wall clock on recovery."""
    cutoff = snapshot.observed_at - timedelta(seconds=config.churn_window)
    recent = [
        (gen_id, rec)
        for gen_id, rec in snapshot.recent_endings
        if rec.end_time is not None and rec.end_time >= cutoff
    ]
    if len(recent) <= config.max_generations_per_window:
        return []
    return [
        Finding(
            key=f"nvrx-generation-churn-{snapshot.job_name}",
            detector="generation_churn",
            severity=WARNING,
            summary=(
                f"{len(recent)} generations of '{snapshot.job_name}' ended in the last "
                f"{_fmt_age(config.churn_window)} (threshold "
                f"{config.max_generations_per_window})."
            ),
            detail="Generations ended: "
            + ", ".join(f"{gen_id}({rec.state})" for gen_id, rec in recent[-6:]),
        )
    ]


# ---------------------------------------------------------------------------------
# Restart anomalies -- platform independent
# ---------------------------------------------------------------------------------
def restart_storm(snapshot: Snapshot, config: Config) -> list[Finding]:
    """Too many NVRx restart cycles in a short window."""
    cutoff = snapshot.observed_at - timedelta(seconds=config.storm_window)
    recent = [c for c in snapshot.current_cycles if c.start_time and c.start_time >= cutoff]
    if len(recent) < config.storm_cycles:
        return []
    job_id = snapshot.current_job_id or "unknown"
    return [
        Finding(
            key=f"nvrx-restart-storm-{job_id}",
            detector="restart_storm",
            severity=WARNING,
            summary=(
                f"{len(recent)} NVRx cycles started in the last {_fmt_age(config.storm_window)} "
                f"on generation {job_id} (threshold {config.storm_cycles})."
            ),
            detail="Cycles: " + ", ".join(str(c.cycle_number) for c in recent[-10:]),
        )
    ]


def stalled_progress(snapshot: Snapshot, config: Config) -> list[Finding]:
    """Cycles keep completing but the checkpoint iteration does not move.

    The failure mode nothing else reports: NVRx restarts cleanly, the workload comes
    back, and every cycle dies before the next save interval. The job looks alive from
    every angle and burns nodes indefinitely without advancing training. This is the
    --ft-min-progress-iterations condition observed rather than enforced.
    """
    cycles = snapshot.current_cycles
    if not cycles:
        return []
    job_id = snapshot.current_job_id or "unknown"
    checkpoint = snapshot.checkpoint

    # Gate on the current generation being a live one, when the platform can tell us. The
    # cycle-info glob spans every generation ever run under the work dir, so when nothing is
    # currently running (e.g. all array tasks still PENDING) the most recent cycles belong
    # to a stale, already-dead generation whose files persist on disk. A dead generation
    # cannot make progress or reach its next save, and its terminal state is the chain
    # detectors' business -- alerting "no checkpoint"/"stalled" on it is a false positive.
    # Without platform data (--platform none) we cannot check liveness, so we do not gate.
    if snapshot.has(CAP_PLATFORM):
        live_gen_ids = {g.gen_id for g in snapshot.generations if g.has_live_task}
        if job_id not in live_gen_ids:
            return []

    if checkpoint.value is None:
        # Never checkpointed. Only meaningful once enough cycles have completed that a
        # save should have happened.
        completed = [c for c in cycles if not c.is_open]
        if len(completed) < config.stall_cycles:
            return []
        expected = config.resolved_checkpoint_file or "the checkpoint iteration file"
        return [
            Finding(
                key=f"nvrx-no-checkpoint-{job_id}",
                detector="stalled_progress",
                severity=CRITICAL,
                summary=(
                    f"{len(completed)} cycles completed on generation {job_id} with no "
                    "checkpoint iteration recorded at all."
                ),
                detail=(
                    f"Expected {expected} to exist. Either the workload never reaches its "
                    "first save, or --ft-checkpoint-iteration-file does not point where the "
                    "workload writes."
                ),
            )
        ]

    since = snapshot.prior.checkpoint_first_seen or checkpoint.mtime
    if since is None:
        return []
    unproductive = [c for c in cycles if not c.is_open and c.start_time and c.start_time >= since]
    if len(unproductive) < config.stall_cycles:
        return []
    return [
        Finding(
            key=f"nvrx-stalled-progress-{job_id}-{checkpoint.value}",
            detector="stalled_progress",
            severity=CRITICAL,
            summary=(
                f"{len(unproductive)} cycles completed since iteration {checkpoint.value} "
                f"({_fmt_age((snapshot.observed_at - since).total_seconds())} ago) without "
                "advancing it."
            ),
            detail=(
                "Restarts are succeeding but training is not progressing: each cycle dies "
                "before the next save interval. Consider --ft-min-progress-iterations to have "
                "NVRx stop the job itself."
            ),
        )
    ]


def cycle_stalled(snapshot: Snapshot, config: Config) -> list[Finding]:
    """The current cycle is open and nothing has moved for a long time.

    The complement of stalled_progress: nothing is restarting because nothing is
    happening. Section timeouts should catch this, so firing means either a section's
    timeout is too loose or the launcher itself is wedged.
    """
    latest = snapshot.latest_cycle
    if latest is None or not latest.is_open or latest.start_time is None:
        return []
    # A last cycle whose cycle_end_time was never written (task 0 SIGKILLed, node death,
    # or an unclean generation exit) stays is_open forever, so "open" alone does not mean
    # "running". If the platform can see the generation that owns it is no longer alive,
    # this is a stale record from an ended generation, not a hung cycle -- that case is
    # orphaned_generation's or chain_exhausted's, not ours. Only diagnose a stall when
    # the generation is, or may still be, running. Without a platform we cannot tell, so
    # the check is best-effort there (the operator opted into cycle-info-only mode).
    if snapshot.has(CAP_PLATFORM):
        owner = next((g for g in snapshot.generations if g.gen_id == latest.job_id), None)
        if owner is None or not owner.has_live_task:
            return []
    movements = [latest.start_time]
    if snapshot.checkpoint.mtime is not None:
        movements.append(snapshot.checkpoint.mtime)
    idle = (snapshot.observed_at - max(movements)).total_seconds()
    if idle < config.stall_seconds:
        return []
    return [
        Finding(
            key=f"nvrx-cycle-stalled-{latest.key}",
            detector="cycle_stalled",
            severity=CRITICAL,
            summary=(
                f"Cycle {latest.cycle_number} of generation {latest.job_id} has been open for "
                f"{_fmt_age(idle)} with no checkpoint or cycle activity."
            ),
            detail=f"Cycle log: {latest.log_file or 'unknown'}",
        )
    ]


def restart_budget_low(snapshot: Snapshot, config: Config) -> list[Finding]:
    """The generation is close to exhausting ft_launcher's --max-restarts.

    Past the budget the generation ends and the chain pays a full cross-job restart, so
    this is the last point at which a human can intervene cheaply.
    """
    if not config.max_restarts or config.max_restarts <= 0:
        return []
    latest = snapshot.latest_cycle
    if latest is None:
        return []
    threshold = config.max_restarts * config.budget_fraction
    if latest.cycle_number < threshold:
        return []
    return [
        Finding(
            key=f"nvrx-restart-budget-{latest.job_id}-{latest.cycle_number}",
            detector="restart_budget_low",
            severity=WARNING,
            summary=(
                f"Generation {latest.job_id} is on cycle {latest.cycle_number} of "
                f"{config.max_restarts} allowed restarts."
            ),
            detail="When the budget is spent the generation ends and the singleton chain "
            "starts the next one from the last checkpoint.",
        )
    ]


def spares_exhausted(snapshot: Snapshot, config: Config) -> list[Finding]:
    """No hot spare in the current cycle, and no cold spare queued behind it.

    Informational by design: the run is healthy, but the next node failure costs a
    generation instead of a restart cycle.
    """
    latest = snapshot.latest_cycle
    if latest is None or latest.standby_nodes.strip():
        return []
    if snapshot.has(CAP_PLATFORM):
        generation = next(
            (g for g in snapshot.generations if g.gen_id == latest.job_id),
            None,
        )
        if generation is None or generation.pending:
            return []
    return [
        Finding(
            key=f"nvrx-spares-exhausted-{latest.job_id}",
            detector="spares_exhausted",
            severity=INFO,
            summary=(
                f"Generation {latest.job_id} is running with no standby node and no queued "
                "spare; the next node failure ends the generation."
            ),
            detail=f"Active nodes: {latest.active_nodes or 'unknown'}",
        )
    ]


def suspect_node(snapshot: Snapshot, config: Config) -> list[Finding]:
    """A node present in every one of the last few short-lived cycles.

    Deliberately weak: cycle-info records which nodes were active, not why a cycle
    ended, so this is correlation only. It names a candidate for --exclude.
    """
    completed = [c for c in snapshot.current_cycles if not c.is_open]
    if len(completed) < config.suspect_cycles:
        return []
    trailing = []
    for cycle in reversed(completed):
        duration = cycle.duration_seconds()
        if duration is None or duration >= config.short_cycle_seconds:
            break
        trailing.append(cycle)
    if len(trailing) < config.suspect_cycles:
        return []

    common: set[str] | None = None
    for cycle in trailing:
        nodes = set(expand_nodelist(cycle.active_nodes))
        common = nodes if common is None else (common & nodes)
        if not common:
            return []
    if not common:
        return []

    job_id = snapshot.current_job_id or "unknown"
    listed = sorted(common)
    return [
        Finding(
            key=f"nvrx-suspect-node-{job_id}-{len(trailing)}",
            detector="suspect_node",
            severity=WARNING,
            summary=(
                f"{len(trailing)} consecutive cycles shorter than "
                f"{_fmt_age(config.short_cycle_seconds)} shared {len(listed)} node(s): "
                + ", ".join(listed[:5])
                + ("..." if len(listed) > 5 else "")
            ),
            detail=(
                "Correlation only -- cycle info does not record why a cycle ended. Check the "
                "per-cycle logs before excluding anything."
            ),
        )
    ]


ALL: tuple[Detector, ...] = (
    Detector("orphaned_generation", (CAP_PLATFORM,), orphaned_generation),
    Detector("chain_exhausted", (CAP_PLATFORM,), chain_exhausted),
    Detector("chain_not_cancelled", (CAP_PLATFORM,), chain_not_cancelled),
    Detector("generation_churn", (CAP_PLATFORM,), generation_churn),
    Detector("restart_storm", (CAP_CYCLES,), restart_storm),
    Detector("stalled_progress", (CAP_CYCLES, CAP_CHECKPOINT), stalled_progress),
    Detector("cycle_stalled", (CAP_CYCLES,), cycle_stalled),
    Detector("restart_budget_low", (CAP_CYCLES,), restart_budget_low),
    Detector("spares_exhausted", (CAP_CYCLES,), spares_exhausted),
    Detector("suspect_node", (CAP_CYCLES,), suspect_node),
)


def enabled(config: Config) -> tuple[Detector, ...]:
    return tuple(d for d in ALL if d.name not in config.disable)


def run(snapshot: Snapshot, config: Config) -> list[Finding]:
    """Run every enabled detector whose required capabilities the snapshot has.

    A detector whose source failed is skipped, not run against partial data: a missing
    squeue must never read as 'no generations are queued'.
    """
    findings: list[Finding] = []
    for detector in enabled(config):
        if not snapshot.has(*detector.requires):
            continue
        findings.extend(detector.fn(snapshot, config))
    return findings
