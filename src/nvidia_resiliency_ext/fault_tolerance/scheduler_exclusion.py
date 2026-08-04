# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consume Scheduler Exclusion decisions before joining rendezvous."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from nvidia_resiliency_ext.shared_utils.job_metadata import (
    slurm_array_job_id_from_env,
    slurm_array_task_id_from_env,
    slurm_restart_count_from_env,
)
from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

log = logging.getLogger(LogConfig.name)

_INITIAL_DECISION_READ_BYTES = 8 * 1024
_MAX_DECISION_LINE_BYTES = 1024 * 1024
_DECISION_MAX_AGE_SECONDS = 30 * 60
_COMPACT_DECISION_PATTERN = re.compile(rb'\[(?:"[0-9]+"(?:,"[0-9]+")*)?\]')


@dataclass(frozen=True)
class SlurmArrayTaskGeneration:
    """Identity of one allocation generation of a Slurm array task."""

    job_id: str
    task_id: str
    restart_count: int

    @property
    def replacement_group_id(self) -> str:
        """Return the opaque rendezvous replacement-group token."""
        return f"{self.task_id}:{self.restart_count}"


def current_slurm_array_task_generation(
    env: Optional[Mapping[str, str]] = None,
) -> Optional[SlurmArrayTaskGeneration]:
    """Return the current Slurm array-task generation, if applicable."""
    environ = os.environ if env is None else env
    job_id = slurm_array_job_id_from_env(environ)
    task_id = slurm_array_task_id_from_env(environ)
    if not job_id or not task_id:
        return None

    return SlurmArrayTaskGeneration(job_id, task_id, slurm_restart_count_from_env(environ))


def current_slurm_array_replacement_group_id(
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return a generation-aware replacement-group token for a Slurm array task."""
    environ = os.environ if env is None else env
    task_id = slurm_array_task_id_from_env(environ)
    if not task_id:
        return None
    return f"{task_id}:{slurm_restart_count_from_env(environ)}"


def is_current_array_task_node0(env: Optional[Mapping[str, str]] = None) -> bool:
    """Return whether this launcher is Node0 of its Slurm array task."""
    environ = os.environ if env is None else env
    if slurm_array_task_id_from_env(environ) is None:
        return False

    node_id = environ.get("SLURM_NODEID")
    if node_id is not None:
        return node_id.strip() == "0"

    process_id = environ.get("SLURM_PROCID")
    return process_id is not None and process_id.strip() == "0"


def scheduler_exclusion_segment_is_healthy(
    directory: str,
    *,
    env: Optional[Mapping[str, str]] = None,
    now: Optional[float] = None,
    round_id: Optional[int] = None,
) -> bool:
    """Evaluate the current task's decision, failing open on artifact errors."""
    environ = os.environ if env is None else env
    if not is_current_array_task_node0(environ):
        return True

    check_started = time.monotonic()
    generation: Optional[SlurmArrayTaskGeneration] = None
    outcome = "not_applicable"
    try:
        generation = current_slurm_array_task_generation(environ)
        if generation is None:
            return True
        decision_path = _decision_file_path(directory, generation.job_id)
        decision, modified_at = _read_first_control_record(decision_path)
        current_time = time.time() if now is None else now
        if modified_at + _DECISION_MAX_AGE_SECONDS <= current_time:
            outcome = "expired"
            return True
        if _COMPACT_DECISION_PATTERN.fullmatch(decision) is None:
            raise ValueError("first Scheduler Exclusion record must be a compact array of task IDs")

        # The first record contains only quoted task IDs, so a fixed token cannot
        # confuse task "7" with task "17" and avoids allocating a decoded list.
        task_token = b'"' + generation.task_id.encode("ascii") + b'"'
        if task_token in decision:
            outcome = "excluded"
            return False
        outcome = "not_excluded"
        return True
    except FileNotFoundError:
        outcome = "missing"
        log.debug("Scheduler Exclusion decision is not available in %s", directory)
    except OSError as exc:
        outcome = "io_error"
        log.warning("Ignoring Scheduler Exclusion decision: %s", exc)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        outcome = "invalid"
        log.warning("Ignoring Scheduler Exclusion decision: %s", exc)
    finally:
        elapsed_ms = (time.monotonic() - check_started) * 1000.0
        log.info(
            "Scheduler Exclusion FT check job_id=%s round=%s task_id=%s restart_count=%s "
            "outcome=%s elapsed_ms=%.3f",
            generation.job_id if generation is not None else "unknown",
            round_id if round_id is not None else "unknown",
            generation.task_id if generation is not None else "unknown",
            generation.restart_count if generation is not None else "unknown",
            outcome,
            elapsed_ms,
        )
    return True


def _decision_file_path(directory: str, job_id: str) -> Path:
    base = Path(directory)
    if not base.is_absolute():
        raise ValueError(f"scheduler exclusion directory must be absolute: {directory!r}")
    if Path(job_id).name != job_id:
        raise ValueError(f"invalid Slurm array job ID: {job_id!r}")
    return base / f"scheduler_exclusion.{job_id}.jsonl"


def _read_first_control_record(path: Path) -> tuple[bytes, float]:
    descriptor = os.open(path, os.O_RDONLY)
    data = bytearray()
    read_size = _INITIAL_DECISION_READ_BYTES
    try:
        modified_at = os.fstat(descriptor).st_mtime
        while len(data) <= _MAX_DECISION_LINE_BYTES:
            remaining = _MAX_DECISION_LINE_BYTES + 1 - len(data)
            chunk = os.read(descriptor, min(read_size, remaining))
            if not chunk:
                break

            newline = chunk.find(b"\n")
            if newline >= 0:
                data.extend(chunk[:newline])
                return bytes(data), modified_at

            data.extend(chunk)
            read_size = _MAX_DECISION_LINE_BYTES + 1 - len(data)
    finally:
        os.close(descriptor)

    if len(data) > _MAX_DECISION_LINE_BYTES:
        raise ValueError(f"first decision record exceeds {_MAX_DECISION_LINE_BYTES} bytes: {path}")
    raise ValueError(f"first decision record is not newline terminated: {path}")
