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

"""Scheduler abstraction. Only chain reconciliation needs it; restart-anomaly
detection reads cycle-info files and works anywhere ``ft_launcher`` runs."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from .types import LIVE_STATES, ChainGeneration, TaskInfo

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


class PlatformError(RuntimeError):
    """The scheduler could not be observed. Never means 'nothing is running'."""


@dataclass(frozen=True)
class JobDescription:
    """What a scheduler can tell us about a job id: enough to bootstrap the watcher's
    identity (name + owner) without the operator spelling them out. The run's directories
    come from the batch script (see ``batch_script``), not from here."""

    job_name: str = ""
    user: str = ""


class Platform(Protocol):
    name: str

    def describe_job(self, job_id: str) -> "JobDescription | None":
        """Resolve a job id to its name and owner, or None if unknown."""

    def batch_script(self, job_id: str) -> str | None:
        """The job's submitted batch script, or None if unavailable (unknown id, or no
        permission to read another owner's script)."""

    def batch_script_path(self, job_id: str) -> str | None:
        """The on-disk path of the job's batch script (its Command=), or None. Needed to
        resolve sbatches that locate their sourced libraries relative to their own path."""

    def list_generations(self, job_name: str) -> list[ChainGeneration]:
        """Generations with at least one task in the queue. Raises PlatformError if blind."""

    def terminal_info(self, gen_id: str, task: int) -> TaskInfo | None:
        """Accounting record for a task no longer in the queue, or None if unknown."""

    def cancel_pending(self, gen_id: str) -> bool:
        """Release queued-but-not-started tasks of one generation."""

    def recent_endings(self, job_name: str, since_seconds: float) -> list[tuple[str, TaskInfo]]:
        """(generation id, task 0 record) for generations that ended within the window."""


class NullPlatform:
    """Cycle-info-only mode: no scheduler, no chain reconciliation."""

    name = "none"

    def describe_job(self, job_id: str) -> JobDescription | None:
        return None

    def batch_script(self, job_id: str) -> str | None:
        return None

    def batch_script_path(self, job_id: str) -> str | None:
        return None

    def list_generations(self, job_name: str) -> list[ChainGeneration]:
        raise PlatformError("no platform configured")

    def terminal_info(self, gen_id: str, task: int) -> TaskInfo | None:
        return None

    def cancel_pending(self, gen_id: str) -> bool:
        return False

    def recent_endings(self, job_name: str, since_seconds: float) -> list[tuple[str, TaskInfo]]:
        return []


def _parse_slurm_time(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value or value in ("Unknown", "None", "N/A"):
        return None
    try:
        # sacct emits local-time ISO without a zone: 2026-07-30T11:58:00
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").astimezone().astimezone(timezone.utc)
    except ValueError:
        return None


def _parse_exit_code(value: str) -> int | None:
    # sacct ExitCode is '<code>:<signal>'; the code is what task0_exit branched on.
    head = (value or "").split(":")[0].strip()
    try:
        return int(head)
    except ValueError:
        return None


class SlurmPlatform:
    """Slurm implementation, via squeue and sacct.

    Every call is timeout-wrapped and its failure is raised as PlatformError. A hung
    squeue is exactly the silent death the dead-man heartbeat exists to catch, but not
    hanging is cheaper than being caught.
    """

    name = "slurm"

    def __init__(self, timeout: float = 30.0, user: str | None = None) -> None:
        self._timeout = timeout
        self._user = user

    def _run(self, argv: list[str]) -> str:
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PlatformError(f"{argv[0]} failed: {exc}") from exc
        if completed.returncode != 0:
            raise PlatformError(
                f"{argv[0]} exited {completed.returncode}: {completed.stderr.strip()[:200]}"
            )
        return completed.stdout

    def _user_args(self) -> list[str]:
        return ["-u", self._user] if self._user else []

    def describe_job(self, job_id: str) -> JobDescription | None:
        # scontrol carries the fields we need for a running/recent job: JobName and the owner
        # (UserId=name(uid)). Once a job ages out of scontrol, fall back to sacct, which
        # retains it for the accounting window, so an old id still resolves name and owner.
        try:
            out = self._run(["scontrol", "show", "job", job_id])
        except PlatformError:
            out = ""
        if out.strip():
            fields: dict[str, str] = {}
            for token in out.split():
                key, sep, value = token.partition("=")
                if sep and key not in fields:
                    fields[key] = value
            return JobDescription(
                job_name=fields.get("JobName", ""),
                user=fields.get("UserId", "").split("(")[0],
            )
        try:
            out = self._run(["sacct", "-j", job_id, "-X", "-n", "-P", "-o", "JobName,User"])
        except PlatformError:
            return None
        line = next((ln for ln in out.splitlines() if ln.strip()), "")
        if not line:
            return None
        parts = line.split("|")
        return JobDescription(
            job_name=parts[0].strip() if parts else "",
            user=parts[1].strip() if len(parts) > 1 else "",
        )

    def batch_script_path(self, job_id: str) -> str | None:
        # The script's path from scontrol (Command=) -- a plain read, unlike
        # `scontrol write batch_script`, which needs owner/operator rights. Assumes the
        # caller can read the owner's sbatch (the SRE model).
        try:
            out = self._run(["scontrol", "show", "job", job_id])
        except PlatformError:
            return None
        match = re.search(r"Command=(\S+)", out)
        return match.group(1) if match else None

    def batch_script(self, job_id: str) -> str | None:
        # Read the script file at its Command= path directly; None if missing or unreadable
        # (and the caller falls back).
        path = self.batch_script_path(job_id)
        if not path:
            return None
        try:
            with open(path) as fh:
                return fh.read()
        except OSError:
            return None

    def list_generations(self, job_name: str) -> list[ChainGeneration]:
        # %F is the ArrayJobID, NOT %A. %A returns the element's own JobID, which SLURM
        # reassigns as tasks start -- grouping by it splits one array into several
        # phantom generations.
        out = self._run(
            ["squeue", "-h", "-r", "-n", job_name, *self._user_args(), "-o", "%F|%K|%T"]
        )
        by_gen: dict[str, list[TaskInfo]] = {}
        for line in out.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 3 or not parts[0]:
                continue
            gen_id, task_raw, state = parts[0], parts[1], parts[2]
            try:
                # %K is empty for a non-array job and may be 'N-M' for an unsplit array.
                task = int(task_raw.split("-")[0]) if task_raw.strip() else 0
            except ValueError:
                continue
            by_gen.setdefault(gen_id, []).append(TaskInfo(task=task, state=state.strip()))
        return [
            ChainGeneration(gen_id=gen_id, tasks=tuple(tasks)) for gen_id, tasks in by_gen.items()
        ]

    def terminal_info(self, gen_id: str, task: int) -> TaskInfo | None:
        out = self._run(
            ["sacct", "-j", f"{gen_id}_{task}", "-X", "-n", "-P", "-o", "State,End,ExitCode"]
        )
        line = next((line for line in reversed(out.splitlines()) if line.strip()), "")
        if not line:
            # Empty output is UNKNOWN, never "gone". Callers must not act on None.
            return None
        parts = line.split("|")
        state = parts[0].split()[0] if parts and parts[0].strip() else ""
        if not state:
            return None
        end = _parse_slurm_time(parts[1]) if len(parts) > 1 else None
        code = _parse_exit_code(parts[2]) if len(parts) > 2 else None
        return TaskInfo(task=task, state=state, exit_code=code, end_time=end)

    def cancel_pending(self, gen_id: str) -> bool:
        try:
            self._run(["scancel", "--state=PENDING", gen_id])
            return True
        except PlatformError:
            return False

    def recent_endings(self, job_name: str, since_seconds: float) -> list[tuple[str, TaskInfo]]:
        starttime = f"now-{int(since_seconds)}seconds"
        argv = ["sacct", "-n", "-P", "-X", "--name", job_name, "-S", starttime]
        argv += [*self._user_args(), "-o", "JobID,State,End,ExitCode"]
        out = self._run(argv)
        endings: dict[str, TaskInfo] = {}
        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) < 4:
                continue
            job_id, state_raw, end_raw, code_raw = (p.strip() for p in parts[:4])
            # sacct has no ArrayJobID column; the array id is embedded in JobID as
            # "<arrayjob>_<task>" (with -X there are no .batch/.extern sub-steps). A
            # still-pending array collapses to "<arrayjob>_[0-3%3]" -- skip that.
            array_job, _, task_id = job_id.partition("_")
            if not array_job or task_id.startswith("["):
                continue
            # One ending per generation, taken from its task 0: an array's other tasks
            # end whenever the pool drains and would multiply-count the same generation.
            if task_id not in ("0", ""):
                continue
            state = state_raw.split()[0] if state_raw else ""
            if not state or state in LIVE_STATES:
                continue
            end = _parse_slurm_time(end_raw)
            if end is not None:
                endings[array_job] = TaskInfo(
                    task=0, state=state, exit_code=_parse_exit_code(code_raw), end_time=end
                )
        return sorted(endings.items(), key=lambda item: item[1].end_time or _EPOCH)


def create(name: str, timeout: float = 30.0, user: str | None = None) -> Platform:
    if name == "slurm":
        return SlurmPlatform(timeout=timeout, user=user or None)
    if name in ("none", ""):
        return NullPlatform()
    raise ValueError(f"unknown platform: {name!r}")
