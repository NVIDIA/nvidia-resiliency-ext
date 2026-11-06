# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler observation and exclusion publication using the deployed Bash protocol."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import subprocess  # nosec B404
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
_NODE = re.compile(r"[A-Za-z0-9._-]+")
_TASK = re.compile(r"[0-9]+")
_CSV = re.compile(r"[A-Za-z0-9._-]+(?:,[A-Za-z0-9._-]+)*")
_UNAVAILABLE_STATES = "drain,down,fail,no_respond"
_HOSTLIST_TIMEOUT = 5


def positive_integer(name: str, value: str) -> int:
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError(f"{name} must be a positive integer: {value}")
    return int(value)


@dataclass(frozen=True)
class PublisherConfig:
    job_id: str
    partition: str
    directory: Path
    stale_seconds: int = 1800
    command_timeout: int = 30

    def __post_init__(self) -> None:
        if not _TASK.fullmatch(self.job_id):
            raise ValueError("SLURM_ARRAY_JOB_ID must be numeric")
        if not self.partition:
            raise ValueError("set SLURM_JOB_PARTITION")
        if not self.directory.is_absolute():
            raise ValueError("NVRX_SEGMENT_HEALTH_CHECK_DIR must be absolute")
        positive_integer("NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS", str(self.stale_seconds))
        positive_integer("NVRX_SEGMENT_HEALTH_SLURM_CMD_TIMEOUT", str(self.command_timeout))

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> PublisherConfig:
        env = os.environ if env is None else env
        for name in ("SLURM_ARRAY_JOB_ID", "SLURM_JOB_PARTITION", "NVRX_SEGMENT_HEALTH_CHECK_DIR"):
            if not env.get(name):
                raise ValueError(f"set {name}")
        return cls(
            job_id=env["SLURM_ARRAY_JOB_ID"],
            partition=env["SLURM_JOB_PARTITION"],
            directory=Path(env["NVRX_SEGMENT_HEALTH_CHECK_DIR"]),
            stale_seconds=positive_integer(
                "NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS",
                env.get("NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS") or "1800",
            ),
            command_timeout=positive_integer(
                "NVRX_SEGMENT_HEALTH_SLURM_CMD_TIMEOUT",
                env.get("NVRX_SEGMENT_HEALTH_SLURM_CMD_TIMEOUT") or "30",
            ),
        )


class PollError(RuntimeError):
    """A scheduler snapshot is incomplete and must not be reconciled."""


def run_slurm(argv: Sequence[str], timeout: int) -> str:
    """Execute fixed Slurm argv via a resolved executable, without a shell."""
    executable = shutil.which(argv[0])
    if executable is None:
        raise PollError(f"{argv[0]} command not found")
    try:
        result = subprocess.run(  # nosec B603
            [executable, *argv[1:]], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as exc:
        raise PollError(f"{argv[0]} failed: {exc}") from exc
    if result.returncode:
        raise PollError(f"{argv[0]} failed: {result.stderr.strip()[:1024]}")
    return result.stdout


def _complete_lines(output: str) -> list[str]:
    if output and not output.endswith("\n"):
        raise PollError("scheduler returned an incomplete row")
    return [line.strip() for line in output.split("\n") if line.strip()]


def _nodes(output: str) -> set[str]:
    nodes = set(_complete_lines(output))
    if any(not _NODE.fullmatch(node) for node in nodes):
        raise PollError("scheduler returned an invalid node")
    return nodes


class SchedulerExclusionPublisher:
    """Reconcile one array's markers from complete Slurm snapshots.

    One process owns a job/directory pair. Persistent markers, not an in-memory
    node cache, are authoritative. Missing-task counters and the stale deadline
    are process-local, matching the deployed Bash producer.
    """

    def __init__(
        self,
        config: PublisherConfig,
        *,
        runner: Callable[[Sequence[str], int], str] = run_slurm,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._runner = runner
        self._clock = clock
        config.directory.mkdir(parents=True, exist_ok=True)
        self._last_complete_poll = clock()
        self._missing_tasks: set[str] = set()
        self._prefix = f"segment_health_check.{config.job_id}."
        self.history_path = config.directory / f"segment_health_check_history.{config.job_id}.log"

    def _path(self, task_id: str) -> Path:
        return self.config.directory / f"{self._prefix}{task_id}"

    def _snapshot(self) -> dict[str, str | None]:
        snapshot: dict[str, str | None] = {}
        for path in self.config.directory.iterdir():
            if not path.name.startswith(self._prefix):
                continue
            task_id = path.name[len(self._prefix) :]
            if not _TASK.fullmatch(task_id):
                continue
            try:
                # Never block on a FIFO or other unexpected artifact type.
                if not path.is_file():
                    raise OSError("not a regular file")
                value = path.read_bytes().decode("ascii")
                snapshot[task_id] = value if not value or _CSV.fullmatch(value) else None
            except (OSError, UnicodeError):
                snapshot[task_id] = None
            if snapshot[task_id] is None:
                logger.warning("Unreadable or noncanonical marker task=%s", task_id)
        return snapshot

    def _desired(self, prior: Mapping[str, str | None]) -> dict[str, str]:
        unavailable = _nodes(
            self._runner(
                [
                    "sinfo",
                    f"--partition={self.config.partition}",
                    f"--states={_UNAVAILABLE_STATES}",
                    "--Node",
                    "--noheader",
                    "--format=%N",
                ],
                self.config.command_timeout,
            )
        )
        if not unavailable and not prior:
            return {}
        taskmap = self._runner(
            [
                "squeue",
                f"--jobs={self.config.job_id}",
                "--states=RUNNING",
                "--array",
                "--noheader",
                "--format=%K|%N",
            ],
            self.config.command_timeout,
        )
        desired = {}
        for row in _complete_lines(taskmap):
            fields = [field.strip() for field in row.split("|")]
            if len(fields) != 2:
                raise PollError("invalid squeue task row")
            task_id, hostlist = fields
            if (
                not _TASK.fullmatch(task_id)
                or task_id in desired
                or not hostlist
                or "{" in hostlist
            ):
                raise PollError("invalid or duplicate squeue task row")
            nodes = _nodes(
                self._runner(["scontrol", "show", "hostnames", hostlist], _HOSTLIST_TIMEOUT)
            )
            if not nodes:
                raise PollError(f"empty hostlist expansion for task={task_id}")
            desired[task_id] = ",".join(sorted(unavailable & nodes))
        return desired

    def _write_current(self, task_id: str, value: str) -> None:
        path = self._path(task_id)
        # Keep an existing exclusion nonempty throughout replacement. The consumer
        # checks only size, so a transient partial CSV is harmless; truncation is not.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NONBLOCK, 0o666)
        with os.fdopen(fd, "wb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise OSError(f"not a regular file: {path}")
            stream.write(value.encode("ascii"))
            stream.flush()
            stream.truncate()

    def _publish(self, task_id: str, value: str) -> bool:
        try:
            self._write_current(task_id, value)
            return True
        except OSError:
            logger.warning("Could not publish task=%s; reconciling its actual value", task_id)
            try:
                path = self._path(task_id)
                return path.is_file() and path.read_bytes() == value.encode("ascii")
            except OSError:
                return False

    @staticmethod
    def _timestamp() -> str | None:
        try:
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (OSError, OverflowError, ValueError):
            logger.warning("History timestamp unavailable")
            return None

    def _audit(
        self,
        event: str,
        task_id: str,
        observed_at: str | None,
        *,
        nodes: str = "",
        reason: str = "",
    ) -> None:
        record = {"event": event, "task_id": int(task_id), "observed_at": observed_at}
        if nodes:
            record["nodes"] = nodes
        elif reason:
            record["reason"] = reason
        try:
            with self.history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError:
            logger.warning("Could not append history event task=%s event=%s", task_id, event)

    def _reconcile(self, prior: Mapping[str, str | None], desired: Mapping[str, str]) -> bool:
        observed_at = self._timestamp()
        success = True
        for task_id in sorted(desired, key=int):
            value = desired[task_id]
            if task_id in prior and prior[task_id] == value:
                continue
            if not value and task_id not in prior:
                continue
            if not self._publish(task_id, value):
                success = False
                continue
            if not value:
                if prior[task_id]:
                    self._audit("cleared", task_id, observed_at)
            else:
                event = "updated" if task_id in prior and prior[task_id] != "" else "excluded"
                self._audit(event, task_id, observed_at, nodes=value)

        missing_next = set()
        for task_id in sorted(prior.keys() - desired.keys(), key=int):
            if task_id not in self._missing_tasks:
                missing_next.add(task_id)
                logger.info(
                    "Task absent from one complete squeue result; deferring task=%s", task_id
                )
                continue
            path = self._path(task_id)
            try:
                if path.exists() or path.is_symlink():
                    path.replace(path.with_name(path.name + ".inactive"))
            except OSError:
                logger.warning("Could not move task artifact inactive task=%s", task_id)
                success = False
                missing_next.add(task_id)
                continue
            if prior[task_id]:
                self._audit("cleared", task_id, observed_at, reason="task_inactive")
        self._missing_tasks = missing_next
        return success

    def poll_once(self) -> bool:
        """Preserve prior decisions on incomplete reads, expiring them after the stale deadline."""
        try:
            prior = self._snapshot()
        except OSError:
            logger.warning("Artifact directory is unavailable; preserving previous decisions")
            return False
        try:
            desired = self._desired(prior)
        except PollError as exc:
            stale_seconds = self._clock() - self._last_complete_poll
            logger.warning("Scheduler poll failed (stale %.1fs): %s", stale_seconds, exc)
            if stale_seconds >= self.config.stale_seconds:
                observed_at = self._timestamp()
                for task_id in sorted(prior, key=int):
                    if prior[task_id] != "" and self._publish(task_id, ""):
                        self._audit(
                            "cleared", task_id, observed_at, reason="stale_decision_expired"
                        )
            return False
        self._last_complete_poll = self._clock()
        return self._reconcile(prior, desired)
