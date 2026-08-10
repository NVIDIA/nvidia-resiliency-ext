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

"""One pass: gather a snapshot, run detectors, apply actions, report, heartbeat."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from . import detectors, persistence, readers, sinks
from .config import Config
from .platform import Platform, PlatformError
from .types import (
    CAP_CHECKPOINT,
    CAP_CYCLES,
    CAP_PLATFORM,
    CRITICAL,
    WARNING,
    Finding,
    Snapshot,
    sorted_findings,
    summarize,
    utcnow,
)

logger = logging.getLogger("nvrx_watch")


@dataclass
class PassResult:
    findings: list[Finding] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    degraded: bool = False  # a source failed: no heartbeat, non-zero exit
    snapshot: Snapshot | None = None

    @property
    def exit_code(self) -> int:
        if self.degraded:
            return 1
        return 2 if any(f.severity == CRITICAL for f in self.findings) else 0


def gather(config: Config, platform: Platform) -> tuple[Snapshot, list[Finding]]:
    """Build a snapshot. A source that fails marks its capability missing rather than
    contributing empty data -- a failed squeue must never read as 'nothing is queued'."""
    now = utcnow()
    capabilities: set[str] = set()
    findings: list[Finding] = []
    generations: tuple = ()
    terminal: dict = {}
    endings: tuple = ()

    if platform.name != "none":
        try:
            generations = tuple(platform.list_generations(config.job_name))
            capabilities.add(CAP_PLATFORM)
        except PlatformError as exc:
            # Blind. Alert (the reporting path is independent of the scheduler) and
            # send no heartbeat: a blind watcher must not look healthy.
            findings.append(
                Finding(
                    key=f"nvrx-watch-blind-{os.uname().nodename}",
                    detector="observer",
                    severity=WARNING,
                    summary=f"nvrx-watch cannot observe {platform.name}: {exc}",
                    detail="No chain reconciliation this pass; the dead-man heartbeat was "
                    "deliberately not sent.",
                )
            )

    if CAP_PLATFORM in capabilities:
        # sacct feeds orphaned_generation (terminal_info, per generation) and
        # chain_not_cancelled / generation_churn (recent_endings, one call). Query each
        # independently: one generation's terminal_info failure must not skip the other
        # generations nor the separate recent_endings call. Any failure is collected and
        # reported as one observer finding -- which marks the pass degraded and withholds
        # the heartbeat -- while every query that did succeed still populates the snapshot,
        # so the reconcilers act on all the data we do have (not just none of it).
        accounting_errors: list[str] = []
        for generation in generations:
            task0 = generation.task0
            if (task0 is None or not task0.is_live) and generation.pending:
                try:
                    record = platform.terminal_info(generation.gen_id, 0)
                except PlatformError as exc:
                    accounting_errors.append(f"terminal_info({generation.gen_id}): {exc}")
                    continue
                if record is not None:
                    terminal[(generation.gen_id, 0)] = record
        try:
            endings = tuple(platform.recent_endings(config.job_name, config.churn_window))
        except PlatformError as exc:
            accounting_errors.append(f"recent_endings: {exc}")
        if accounting_errors:
            findings.append(
                Finding(
                    key=f"nvrx-watch-blind-accounting-{os.uname().nodename}",
                    detector="observer",
                    severity=WARNING,
                    summary="nvrx-watch could not read sacct: " + "; ".join(accounting_errors),
                    detail="Chain reconciliation ran on partial data this pass; the "
                    "dead-man heartbeat was deliberately not sent.",
                )
            )

    cycle_records = readers.read_cycles(config.resolved_cycle_info_glob)
    if cycle_records:
        capabilities.add(CAP_CYCLES)
    elif config.resolved_cycle_info_glob:
        logger.info("no cycle info files under %s", config.resolved_cycle_info_glob)

    # Unconditional: the file is absent until the first save, and "never checkpointed
    # after N cycles" is one of the things stalled_progress reports.
    checkpoint = readers.read_checkpoint_progress(config.resolved_checkpoint_file)
    capabilities.add(CAP_CHECKPOINT)

    prior, _ = persistence.load(config.state_file)

    snapshot = Snapshot(
        observed_at=now,
        job_name=config.job_name,
        capabilities=frozenset(capabilities),
        generations=generations,
        cycles=cycle_records,
        checkpoint=checkpoint,
        prior=prior,
        chain_expected=os.path.exists(config.expect_file),
        max_restarts=config.max_restarts,
        terminal_info=terminal,
        recent_endings=endings,
    )
    return snapshot, findings


def apply_actions(findings: list[Finding], config: Config, platform: Platform) -> list[str]:
    applied: list[str] = []
    for finding in findings:
        action = finding.action
        if action is None:
            continue
        if config.dry_run:
            logger.info("[dry-run] would run: %s", action.description or action.kind)
            applied.append(f"[dry-run] {action.description or action.kind}")
            continue
        if config.observe_only:
            # SRE mode: the finding is still reported (the owner is notified), but the
            # watcher does not touch jobs it may not own.
            logger.info(
                "[observe-only] not acting; owner should run: %s",
                action.description or action.kind,
            )
            applied.append(f"[observe-only] {action.description or action.kind}")
            continue
        if action.kind == "cancel_pending":
            if platform.cancel_pending(action.target):
                logger.info("released pending spares of generation %s", action.target)
                applied.append(action.description or action.kind)
            else:
                # Not fatal: the finding still pages, and the next pass retries.
                logger.warning("cancel_pending failed for %s; will retry next pass", action.target)
        else:
            logger.warning("unknown action %r on finding %s", action.kind, finding.key)
    return applied


def report(findings: list[Finding], config: Config, sink_list: list) -> None:
    """Emit findings, honouring the dedup cooldown so a persistent condition pages
    periodically rather than every pass -- and is never silently forgotten."""
    prior, alerts = persistence.load(config.state_file)
    now = utcnow()
    for finding in findings:
        for sink in sink_list:
            # The log sink is local visibility, not paging: it emits every pass, ungated
            # and unrecorded.
            if sink.name == "log":
                sink.emit(finding)
                continue
            # Cooldown is tracked per (finding, sink): a sink that accepted the alert enters
            # its own cooldown, while one that rejected or timed out retries next pass. A
            # shared per-finding flag would either re-page the sink that already accepted
            # (until all succeed at once) or suppress the one that never did.
            alert_key = f"{finding.key}\x00{sink.name}"
            last_sent = alerts.get(alert_key)
            due = last_sent is None or (now - last_sent).total_seconds() >= config.alert_cooldown
            if not due:
                continue
            if config.dry_run:
                logger.info("[dry-run] would notify %s: %s", sink.name, finding.key)
                continue
            if sink.emit(finding):
                alerts[alert_key] = now
    alerts = persistence.prune_alerts(alerts, max(config.alert_cooldown * 4, 86400.0), now)
    if not config.dry_run:
        persistence.save(config.state_file, prior, alerts)


def run_once(config: Config, platform: Platform, sink_list: list | None = None) -> PassResult:
    sink_list = sinks.build(config) if sink_list is None else sink_list
    snapshot, source_findings = gather(config, platform)

    findings = sorted_findings(source_findings + detectors.run(snapshot, config))
    result = PassResult(findings=findings, snapshot=snapshot)
    result.degraded = any(f.detector == "observer" for f in findings)

    result.actions_taken = apply_actions(findings, config, platform)
    report(findings, config, sink_list)

    # Persist observation state after reporting, so a crash mid-pass re-reports rather
    # than silently advancing the stall timers.
    if not config.dry_run:
        prior, alerts = persistence.load(config.state_file)
        advanced = persistence.advance(
            prior, snapshot.checkpoint, snapshot.latest_cycle, snapshot.observed_at
        )
        persistence.save(config.state_file, advanced, alerts)

    logger.info(
        "pass complete: %s%s",
        summarize(findings),
        " (degraded: no heartbeat sent)" if result.degraded else "",
    )
    if not result.degraded:
        sinks.heartbeat(config.heartbeat_url)
    return result
