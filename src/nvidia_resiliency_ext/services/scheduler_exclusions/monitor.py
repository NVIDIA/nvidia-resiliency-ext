# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler exclusion monitoring and cache publication."""

from __future__ import annotations

import logging
import os
import random
import re
import shutil
import subprocess  # nosec B404
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from .decision_file import (
    DecisionFileWriteError,
    DecisionFileWriter,
    DecisionObservation,
    build_decision_response,
    decision_file_path,
)

logger = logging.getLogger(__name__)

DEFAULT_REFRESH_INTERVAL_SECONDS = 600.0
DEFAULT_CACHE_TTL_SECONDS = 1800.0
DEFAULT_QUERY_TIMEOUT_SECONDS = 30.0
DEFAULT_JITTER_FRACTION = 0.1
_LARGE_EXCLUDED_NODE_COUNT = 16
_MAX_RECOVERY_VERIFICATION_CANDIDATES = 16
_ARRAY_SQUEUE_FORMAT = "--Format=ArrayTaskID:64|,RestartCnt:16|,Partition:128|,NodeList:1024"
_SINFO_EXCLUSION_FILTER = "drain,down,fail,no_respond"

SCHEDULER_EXCLUSION_STATES = frozenset(
    {
        "down",
        "drain",
        "drained",
        "draining",
        "fail",
        "failing",
        "no_respond",
    }
)


class SchedulerExclusionError(RuntimeError):
    """Raised when scheduler state cannot be queried safely."""


class MalformedSchedulerResponse(SchedulerExclusionError):
    """Raised when Slurm returns a response that is unsafe to consume."""


class CommandRunner(Protocol):
    """Runs one command in the scheduler environment."""

    def run(self, argv: Sequence[str]) -> str:
        """Return stdout or raise :class:`SchedulerExclusionError`."""


@dataclass(frozen=True)
class JobIdentity:
    """Slurm identity inherited by the Scheduler Exclusion Service."""

    job_id: str
    is_array: bool


@dataclass(frozen=True)
class SchedulerExclusionConfig:
    """Configuration for one Scheduler Exclusion monitor."""

    slurm_bin_dir: str = ""
    slurm_conf: str = ""
    scheduler_exclusion_dir: str = ""
    refresh_interval_seconds: float = DEFAULT_REFRESH_INTERVAL_SECONDS
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS
    query_timeout_seconds: float = DEFAULT_QUERY_TIMEOUT_SECONDS
    jitter_fraction: float = DEFAULT_JITTER_FRACTION

    @property
    def enabled(self) -> bool:
        return True


@dataclass(frozen=True)
class NodeStateRecord:
    """One scheduler response row."""

    node: str
    state: str
    reason: str
    excluded: bool


@dataclass(frozen=True)
class ExcludedNodeObservation:
    """One cached Scheduler Exclusion observation."""

    state: str
    reason: str
    observed_at: float
    array_tasks: tuple[ArrayTaskGeneration, ...]


@dataclass(frozen=True)
class ArrayTaskGeneration:
    """One running incarnation of a Slurm array task."""

    task_id: str
    restart_count: int


@dataclass(frozen=True)
class _PublishedSnapshot:
    """Immutable cache view captured by HTTP readers."""

    job_id: str | None
    last_complete_poll: float | None
    last_poll_attempt: float | None
    observations: tuple[tuple[str, ExcludedNodeObservation], ...]
    last_error: str | None
    last_decision_write: float | None
    last_decision_error: str | None
    polls_attempted: int
    polls_completed: int
    decision_write_failures: int
    current_nodes: int


@dataclass(frozen=True)
class AllocationSnapshot:
    """Current Slurm allocation grouped by partition."""

    nodes_by_partition: dict[str, tuple[str, ...]]
    array_task_generations_by_node: dict[str, tuple[ArrayTaskGeneration, ...]]

    @property
    def nodes(self) -> set[str]:
        return {
            node for partition_nodes in self.nodes_by_partition.values() for node in partition_nodes
        }


def job_identity_from_env(env: Mapping[str, str] | None = None) -> JobIdentity | None:
    """Resolve an array parent ID or regular job ID from the inherited environment."""
    values = os.environ if env is None else env
    array_job_id = str(values.get("SLURM_ARRAY_JOB_ID", "")).strip()
    if array_job_id:
        return JobIdentity(job_id=array_job_id, is_array=True)

    job_id = str(values.get("SLURM_JOB_ID", "")).strip()
    if job_id:
        return JobIdentity(job_id=job_id, is_array=False)
    return None


class LocalSlurmCommandRunner:
    """Execute fixed Slurm argv in the local batch-host environment."""

    def __init__(
        self,
        *,
        slurm_bin_dir: str = "",
        slurm_conf: str = "",
        timeout_seconds: float = DEFAULT_QUERY_TIMEOUT_SECONDS,
    ) -> None:
        self.slurm_bin_dir = slurm_bin_dir
        self.slurm_conf = slurm_conf
        self.timeout_seconds = float(timeout_seconds)

    def run(self, argv: Sequence[str]) -> str:
        if not argv:
            raise ValueError("command must not be empty")
        command = [str(arg) for arg in argv]
        if self.slurm_bin_dir:
            command[0] = os.path.join(self.slurm_bin_dir, command[0])
        else:
            resolved = shutil.which(command[0])
            if resolved is None:
                raise SchedulerExclusionError(f"{command[0]} command not found")
            command[0] = resolved
        env = os.environ.copy()
        if self.slurm_conf:
            env["SLURM_CONF"] = self.slurm_conf

        try:
            result = subprocess.run(  # nosec B603
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise SchedulerExclusionError(
                f"{argv[0]} timed out after {self.timeout_seconds:.1f}s"
            ) from exc
        except OSError as exc:
            raise SchedulerExclusionError(f"{argv[0]} could not be started: {exc}") from exc

        if result.returncode != 0:
            message = (result.stderr or result.stdout or f"{argv[0]} failed").strip()
            raise SchedulerExclusionError(message)
        return result.stdout


def discover_allocation(
    runner: CommandRunner,
    identity: JobIdentity,
    *,
    stop_requested: Callable[[], bool] | None = None,
) -> AllocationSnapshot:
    """Discover current running nodes for a regular job or array parent."""
    command = ["squeue", "--noheader"]
    if identity.is_array:
        command.append("--array")
        # Long-format fields are truncated to their requested width by some
        # supported Slurm releases. Leave enough room for compressed nodelists.
        output_option = _ARRAY_SQUEUE_FORMAT
    else:
        output_option = "--format=%P|%N"
    command.extend(
        [
            "--jobs",
            identity.job_id,
            "--states=RUNNING",
            output_option,
        ]
    )
    output = runner.run(command)
    if stop_requested is not None and stop_requested():
        raise SchedulerExclusionError("Scheduler Exclusion monitor is stopping")
    allocation_rows = _parse_squeue_output(output, is_array=identity.is_array)
    if not allocation_rows:
        raise SchedulerExclusionError(
            f"squeue returned no running allocations for job {identity.job_id}"
        )

    partition_nodes: dict[str, list[str]] = {}
    array_task_generations_by_node: dict[str, set[ArrayTaskGeneration]] = {}
    expanded_nodelists: dict[str, tuple[str, ...]] = {}
    seen_nodes: set[str] = set()
    # Expand each unique task nodelist separately to retain its node association.
    for task_generation, partition, nodelist in allocation_rows:
        nodes = expanded_nodelists.get(nodelist)
        if nodes is None:
            if stop_requested is not None and stop_requested():
                raise SchedulerExclusionError("Scheduler Exclusion monitor is stopping")
            expanded = runner.run(["scontrol", "show", "hostnames", nodelist])
            if stop_requested is not None and stop_requested():
                raise SchedulerExclusionError("Scheduler Exclusion monitor is stopping")
            nodes = tuple(line.strip() for line in expanded.splitlines() if line.strip())
            if not nodes:
                raise MalformedSchedulerResponse(
                    f"scontrol returned no nodes for nodelist {nodelist!r}"
                )
            expanded_nodelists[nodelist] = nodes
        for node in nodes:
            if task_generation is not None:
                array_task_generations_by_node.setdefault(node, set()).add(task_generation)
            if node not in seen_nodes:
                seen_nodes.add(node)
                partition_nodes.setdefault(partition, []).append(node)

    nodes_by_partition = {
        partition: tuple(nodes) for partition, nodes in partition_nodes.items() if nodes
    }

    if not nodes_by_partition:
        raise SchedulerExclusionError(
            f"allocation expansion returned no nodes for job {identity.job_id}"
        )
    return AllocationSnapshot(
        nodes_by_partition=nodes_by_partition,
        array_task_generations_by_node={
            node: tuple(sorted(generations, key=_array_task_generation_sort_key))
            for node, generations in array_task_generations_by_node.items()
        },
    )


def query_scheduler_exclusions(
    runner: CommandRunner,
    *,
    exclusion_states: frozenset[str] = SCHEDULER_EXCLUSION_STATES,
) -> dict[str, NodeStateRecord]:
    """Query scheduler-unavailable nodes across the cluster."""
    output = runner.run(
        [
            "sinfo",
            "--noheader",
            "--Node",
            f"--states={_SINFO_EXCLUSION_FILTER}",
            "--format=%N|%T|%E",
        ]
    )
    records = _parse_sinfo_output(output, exclusion_states)
    unexpected = sorted(node for node, record in records.items() if not record.excluded)
    if unexpected:
        raise MalformedSchedulerResponse(
            "filtered sinfo response contained allocatable nodes: " + ", ".join(unexpected)
        )
    return records


def _query_recovery_candidates(
    runner: CommandRunner,
    nodes: Sequence[str],
    *,
    exclusion_states: frozenset[str] = SCHEDULER_EXCLUSION_STATES,
) -> dict[str, NodeStateRecord]:
    """Query full scheduler state for a bounded set of prior exclusions."""
    requested_nodes = tuple(sorted(set(nodes)))
    if not requested_nodes:
        return {}
    output = runner.run(
        [
            "sinfo",
            "--noheader",
            "--Node",
            f"--nodes={','.join(requested_nodes)}",
            "--format=%N|%T|%E",
        ]
    )
    records = _parse_sinfo_output(output, exclusion_states)
    unexpected = sorted(set(records) - set(requested_nodes))
    if unexpected:
        raise MalformedSchedulerResponse(
            "recovery verification returned unrequested nodes: " + ", ".join(unexpected)
        )
    return records


class SchedulerExclusionMonitor:
    """Poll Slurm and publish copy-on-write snapshots of excluded nodes."""

    def __init__(
        self,
        config: SchedulerExclusionConfig,
        *,
        env: Mapping[str, str] | None = None,
        runner: CommandRunner | None = None,
        clock: Callable[[], float] = time.time,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if config.refresh_interval_seconds <= 0:
            raise ValueError("refresh_interval_seconds must be positive")
        if config.cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be positive")
        if config.query_timeout_seconds <= 0:
            raise ValueError("query_timeout_seconds must be positive")
        if config.scheduler_exclusion_dir and not os.path.isabs(config.scheduler_exclusion_dir):
            raise ValueError("scheduler_exclusion_dir must be an absolute path")
        if not 0 <= config.jitter_fraction < 1:
            raise ValueError("jitter_fraction must be in [0, 1)")

        self.config = config
        self.identity = job_identity_from_env(env)
        self._runner = runner
        if self._runner is None:
            self._runner = LocalSlurmCommandRunner(
                slurm_bin_dir=config.slurm_bin_dir,
                slurm_conf=config.slurm_conf,
                timeout_seconds=config.query_timeout_seconds,
            )
        self._clock = clock
        self._jitter = jitter
        self._decision_writer = None
        if config.scheduler_exclusion_dir and self.identity is not None:
            path = decision_file_path(config.scheduler_exclusion_dir, self.identity.job_id)
            self._decision_writer = DecisionFileWriter(path)
        self._lock = threading.Lock()
        self._snapshot_lock = threading.Lock()
        self._refresh_condition = threading.Condition()
        self._refresh_requested = False
        self._poll_in_progress = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._excluded_nodes: dict[str, ExcludedNodeObservation] = {}
        self._current_nodes: set[str] = set()
        self._last_complete_poll: float | None = None
        self._last_poll_attempt: float | None = None
        self._last_error: str | None = None
        self._last_decision_write: float | None = None
        self._last_decision_error: str | None = None
        self._polls_attempted = 0
        self._polls_completed = 0
        self._decision_write_failures = 0
        self._state_version = 0
        self._published_snapshot = _PublishedSnapshot(
            job_id=self.identity.job_id if self.identity is not None else None,
            last_complete_poll=None,
            last_poll_attempt=None,
            observations=(),
            last_error=None,
            last_decision_write=None,
            last_decision_error=None,
            polls_attempted=0,
            polls_completed=0,
            decision_write_failures=0,
            current_nodes=0,
        )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def start(self) -> None:
        """Start one immediate poll followed by periodic refreshes."""
        if not self.enabled:
            return
        if self.identity is None:
            logger.warning(
                "Scheduler Exclusion monitoring is enabled but SLURM_ARRAY_JOB_ID and "
                "SLURM_JOB_ID are unset"
            )
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        with self._refresh_condition:
            self._refresh_requested = False
            self._poll_in_progress = False
        self._thread = threading.Thread(
            target=self._run,
            name="nvrx-scheduler-exclusion",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the polling worker."""
        self._stop_event.set()
        with self._refresh_condition:
            self._refresh_condition.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self.config.query_timeout_seconds + 1.0)
            if thread.is_alive():
                logger.warning("Scheduler Exclusion polling worker is still stopping")
                return
        self._thread = None

    def request_refresh(self) -> bool:
        """Wake the existing worker without performing scheduler I/O."""
        if not self.enabled or self.identity is None or self._stop_event.is_set():
            return False
        with self._refresh_condition:
            if self._thread is None or not self._thread.is_alive():
                return False
            if self._poll_in_progress or self._refresh_requested:
                return False
            self._refresh_requested = True
            self._refresh_condition.notify()
        return True

    def poll_once(self) -> bool:
        """Run one allocation and node-state refresh."""
        if not self.enabled or self.identity is None or self._runner is None:
            return False

        attempted_at = self._clock()
        with self._lock:
            self._last_poll_attempt = attempted_at
            self._polls_attempted += 1
            self._state_version += 1
        self._publish_snapshot()

        try:
            allocation = discover_allocation(
                self._runner,
                self.identity,
                stop_requested=self._stop_event.is_set,
            )
        except SchedulerExclusionError as exc:
            if self._stop_event.is_set():
                return False
            self._record_error(f"allocation discovery failed: {exc}")
            return False

        if self._stop_event.is_set():
            return False
        try:
            cluster_exclusions = query_scheduler_exclusions(self._runner)
        except SchedulerExclusionError as exc:
            if self._stop_event.is_set():
                return False
            self._record_error(f"scheduler-state query failed: {exc}")
            return False

        completed_at = self._clock()
        current_nodes = allocation.nodes
        staged_records = {
            node: record for node, record in cluster_exclusions.items() if node in current_nodes
        }
        with self._lock:
            recovery_candidates = sorted(
                node
                for node, observation in self._excluded_nodes.items()
                if node in current_nodes
                and node not in staged_records
                and completed_at - observation.observed_at <= self.config.cache_ttl_seconds
            )

        retained_recovery_nodes = set(recovery_candidates)
        verified_allocatable_nodes: set[str] = set()
        if recovery_candidates:
            if len(recovery_candidates) > _MAX_RECOVERY_VERIFICATION_CANDIDATES:
                logger.warning(
                    "Scheduler Exclusion recovery verification skipped job_id=%s "
                    "candidates=%s limit=%s",
                    self.identity.job_id,
                    len(recovery_candidates),
                    _MAX_RECOVERY_VERIFICATION_CANDIDATES,
                )
            else:
                try:
                    recovery_records = _query_recovery_candidates(
                        self._runner,
                        recovery_candidates,
                    )
                except SchedulerExclusionError as exc:
                    logger.warning(
                        "Scheduler Exclusion recovery verification failed job_id=%s "
                        "candidates=%s: %s",
                        self.identity.job_id,
                        len(recovery_candidates),
                        exc,
                    )
                else:
                    for node, record in recovery_records.items():
                        retained_recovery_nodes.discard(node)
                        if record.excluded:
                            staged_records[node] = record
                        else:
                            verified_allocatable_nodes.add(node)

        if self._stop_event.is_set():
            return False
        completed_at = self._clock()
        with self._lock:
            previous_excluded = {
                node for node in self._excluded_nodes if node in self._current_nodes
            }
            self._current_nodes = current_nodes
            self._apply_records_locked(
                staged_records,
                current_nodes=current_nodes,
                observed_at=completed_at,
                array_task_generations_by_node=allocation.array_task_generations_by_node,
                retained_nodes=retained_recovery_nodes,
            )
            self._last_complete_poll = completed_at
            self._last_error = None
            self._polls_completed += 1
            self._evict_expired_locked(completed_at)
            current_excluded = {
                node for node in self._excluded_nodes if node in self._current_nodes
            }
            self._state_version += 1
        self._publish_snapshot()
        self._publish_decision_file(generated_at=completed_at)
        logger.info(
            "Scheduler Exclusion refresh completed job_id=%s nodes=%s "
            "cluster_exclusions=%s excluded=%s duration_seconds=%.3f",
            self.identity.job_id,
            len(current_nodes),
            len(cluster_exclusions),
            len(current_excluded),
            max(0.0, completed_at - attempted_at),
        )
        newly_excluded = sorted(current_excluded - previous_excluded)
        cleared_exclusions = previous_excluded - current_excluded
        became_allocatable_set = cleared_exclusions & verified_allocatable_nodes
        left_allocation_set = cleared_exclusions - current_nodes
        expired_exclusions = sorted(
            cleared_exclusions - became_allocatable_set - left_allocation_set
        )
        became_allocatable = sorted(became_allocatable_set)
        left_allocation = sorted(left_allocation_set)
        if newly_excluded or became_allocatable or left_allocation or expired_exclusions:
            logger.info(
                "Scheduler Exclusion transitions job_id=%s newly_excluded=%s "
                "became_allocatable=%s left_allocation=%s expired_exclusions=%s",
                self.identity.job_id,
                newly_excluded,
                became_allocatable,
                left_allocation,
                expired_exclusions,
            )
        if len(current_excluded) > _LARGE_EXCLUDED_NODE_COUNT:
            logger.warning(
                "Scheduler Exclusion monitor found %s excluded nodes for job %s",
                len(current_excluded),
                self.identity.job_id,
            )
        return True

    def snapshot(self) -> dict:
        """Materialize the current immutable cache view without scheduler I/O."""
        now = self._clock()
        with self._snapshot_lock:
            published = self._published_snapshot

        observations = {
            node: observation
            for node, observation in published.observations
            if now - observation.observed_at <= self.config.cache_ttl_seconds
        }
        last_error = published.last_error
        if published.job_id is None:
            last_error = "Slurm job identity is unavailable"

        complete_cache_is_fresh = (
            published.last_complete_poll is not None
            and now - published.last_complete_poll <= self.config.cache_ttl_seconds
        )
        if complete_cache_is_fresh:
            cache_quality = "complete"
        else:
            cache_quality = "unavailable"

        return {
            "job_id": published.job_id,
            "last_complete_poll": _format_timestamp(published.last_complete_poll),
            "last_poll_attempt": _format_timestamp(published.last_poll_attempt),
            "last_decision_write": _format_timestamp(published.last_decision_write),
            "excluded_nodes": sorted(observations),
            "observations": {
                node: {
                    "state": observation.state,
                    "reason": observation.reason,
                    "observed_at": _format_timestamp(observation.observed_at),
                    "array_tasks": [
                        {
                            "task_id": task.task_id,
                            "restart_count": task.restart_count,
                        }
                        for task in observation.array_tasks
                    ],
                }
                for node, observation in sorted(observations.items())
            },
            "last_error": last_error,
            "last_decision_error": published.last_decision_error,
            "stats": {
                "polls_attempted": published.polls_attempted,
                "polls_completed": published.polls_completed,
                "decision_write_failures": published.decision_write_failures,
                "current_nodes": published.current_nodes,
                "current_excluded_nodes": len(observations),
                "cache_quality": cache_quality,
            },
        }

    def scheduler_exclusions(self) -> dict[str, object] | None:
        """Return current task and node decisions without scheduler or file I/O."""
        now = self._clock()
        with self._snapshot_lock:
            published = self._published_snapshot

        if self.identity is None or published.last_complete_poll is None:
            return None

        observations = [
            DecisionObservation(
                node=node,
                state=observation.state,
                reason=observation.reason,
                observed_at=observation.observed_at,
                array_tasks=tuple(
                    (task.task_id, task.restart_count) for task in observation.array_tasks
                ),
            )
            for node, observation in published.observations
            if now - observation.observed_at <= self.config.cache_ttl_seconds
        ]
        return build_decision_response(
            job_id=self.identity.job_id,
            generated_at=published.last_complete_poll,
            cache_ttl_seconds=self.config.cache_ttl_seconds,
            observations=observations,
        )

    def _publish_snapshot(self) -> None:
        """Build outside locks and atomically publish the newest state version."""
        while True:
            with self._lock:
                version = self._state_version
                excluded_nodes = dict(self._excluded_nodes)
                current_nodes = frozenset(self._current_nodes)
                job_id = self.identity.job_id if self.identity is not None else None
                last_complete_poll = self._last_complete_poll
                last_poll_attempt = self._last_poll_attempt
                last_error = self._last_error
                last_decision_write = self._last_decision_write
                last_decision_error = self._last_decision_error
                polls_attempted = self._polls_attempted
                polls_completed = self._polls_completed
                decision_write_failures = self._decision_write_failures

            published = _PublishedSnapshot(
                job_id=job_id,
                last_complete_poll=last_complete_poll,
                last_poll_attempt=last_poll_attempt,
                observations=tuple(
                    (node, observation)
                    for node, observation in sorted(excluded_nodes.items())
                    if node in current_nodes
                ),
                last_error=last_error,
                last_decision_write=last_decision_write,
                last_decision_error=last_decision_error,
                polls_attempted=polls_attempted,
                polls_completed=polls_completed,
                decision_write_failures=decision_write_failures,
                current_nodes=len(current_nodes),
            )

            # Do not let an older build replace state changed while it was built.
            with self._lock:
                if version != self._state_version:
                    continue
                with self._snapshot_lock:
                    self._published_snapshot = published
                return

    def _publish_decision_file(self, *, generated_at: float) -> None:
        writer = self._decision_writer
        identity = self.identity
        if writer is None or identity is None:
            return

        with self._snapshot_lock:
            published = self._published_snapshot
        observations = [
            DecisionObservation(
                node=node,
                state=observation.state,
                reason=observation.reason,
                observed_at=observation.observed_at,
                array_tasks=tuple(
                    (task.task_id, task.restart_count) for task in observation.array_tasks
                ),
            )
            for node, observation in published.observations
        ]
        try:
            excluded_tasks, excluded_nodes = writer.publish(
                job_id=identity.job_id,
                generated_at=generated_at,
                cache_ttl_seconds=self.config.cache_ttl_seconds,
                observations=observations,
            )
        except DecisionFileWriteError as exc:
            with self._lock:
                self._last_decision_error = str(exc)
                self._decision_write_failures += 1
                self._state_version += 1
            self._publish_snapshot()
            logger.warning("Scheduler Exclusion decision publication failed: %s", exc)
            return

        with self._lock:
            self._last_decision_write = generated_at
            self._last_decision_error = None
            self._state_version += 1
        self._publish_snapshot()
        logger.info(
            "Scheduler Exclusion decision published path=%s tasks=%s nodes=%s observations=%s",
            writer.path,
            excluded_tasks,
            excluded_nodes,
            len(observations),
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            with self._refresh_condition:
                self._poll_in_progress = True
                self._refresh_requested = False
            try:
                self.poll_once()
            except Exception:
                logger.exception("Unexpected Scheduler Exclusion polling failure")
            finally:
                with self._refresh_condition:
                    self._poll_in_progress = False
            jitter = self.config.refresh_interval_seconds * self.config.jitter_fraction
            delay = self.config.refresh_interval_seconds + self._jitter(-jitter, jitter)
            with self._refresh_condition:
                self._refresh_condition.wait_for(
                    lambda: self._stop_event.is_set() or self._refresh_requested,
                    timeout=max(1.0, delay),
                )

    def _apply_records_locked(
        self,
        records: Mapping[str, NodeStateRecord],
        *,
        current_nodes: set[str],
        observed_at: float,
        array_task_generations_by_node: Mapping[str, tuple[ArrayTaskGeneration, ...]],
        retained_nodes: set[str],
    ) -> None:
        for node in current_nodes:
            record = records.get(node)
            if record is not None:
                self._excluded_nodes[node] = ExcludedNodeObservation(
                    state=record.state,
                    reason=record.reason,
                    observed_at=observed_at,
                    array_tasks=array_task_generations_by_node.get(node, ()),
                )
            elif node not in retained_nodes:
                self._excluded_nodes.pop(node, None)

    def _record_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message
            self._state_version += 1
        self._publish_snapshot()
        logger.warning("Scheduler Exclusion monitor: %s", message)

    def _evict_expired_locked(self, now: float) -> None:
        expired = [
            node
            for node, observation in self._excluded_nodes.items()
            if now - observation.observed_at > self.config.cache_ttl_seconds
        ]
        for node in expired:
            del self._excluded_nodes[node]


def _parse_squeue_output(
    output: str,
    *,
    is_array: bool,
) -> list[tuple[ArrayTaskGeneration | None, str, str]]:
    allocation_rows: list[tuple[ArrayTaskGeneration | None, str, str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split("|")]
        expected_parts = 4 if is_array else 2
        if len(parts) != expected_parts:
            raise MalformedSchedulerResponse(f"invalid squeue row: {line!r}")
        task_generation = None
        if is_array:
            array_task_id, restart_count_text = parts[:2]
            if not array_task_id or array_task_id.lower() in {"(null)", "n/a"}:
                raise MalformedSchedulerResponse(f"squeue row has no array task ID: {line!r}")
            try:
                restart_count = int(restart_count_text)
            except ValueError as exc:
                raise MalformedSchedulerResponse(
                    f"squeue row has invalid restart count: {line!r}"
                ) from exc
            if restart_count < 0:
                raise MalformedSchedulerResponse(f"squeue row has negative restart count: {line!r}")
            task_generation = ArrayTaskGeneration(array_task_id, restart_count)
        partition, nodelist = parts[-2:]
        if not partition:
            raise MalformedSchedulerResponse(f"squeue row has no partition: {line!r}")
        if not nodelist or nodelist.lower() in {"(null)", "n/a"}:
            raise MalformedSchedulerResponse(f"squeue row has no nodelist: {line!r}")
        allocation_rows.append((task_generation, partition, nodelist))
    return allocation_rows


def _array_task_generation_sort_key(
    task: ArrayTaskGeneration,
) -> tuple[int, int, str, int]:
    if task.task_id.isdigit():
        return (0, int(task.task_id), task.task_id, task.restart_count)
    return (1, 0, task.task_id, task.restart_count)


def _parse_sinfo_output(
    output: str,
    exclusion_states: frozenset[str],
) -> dict[str, NodeStateRecord]:
    records: dict[str, NodeStateRecord] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 2)
        if len(parts) < 2:
            raise MalformedSchedulerResponse(f"invalid sinfo row: {line!r}")
        node = parts[0].strip()
        raw_state = parts[1].strip()
        reason = parts[2].strip() if len(parts) == 3 else ""
        if not node:
            raise MalformedSchedulerResponse(f"sinfo row has no node: {line!r}")
        if not raw_state:
            raise MalformedSchedulerResponse(f"sinfo row has no state: {line!r}")
        normalized = "no_respond" if raw_state.endswith("*") else _normalize_state(raw_state)
        if normalized == "unknown":
            raise MalformedSchedulerResponse(f"sinfo row has unknown state: {line!r}")
        records[node] = NodeStateRecord(
            node=node,
            state=normalized.upper(),
            reason=reason,
            excluded=normalized in exclusion_states,
        )
    return records


def _normalize_state(raw_state: str) -> str:
    normalized = raw_state.strip().lower()
    normalized = re.sub(r"[\s-]+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
    return normalized or "unknown"


def _format_timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")
