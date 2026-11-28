# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic shared-file publication for Scheduler Exclusion decisions."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


class DecisionFileWriteError(RuntimeError):
    """Raised when a decision artifact cannot be published atomically."""


def decision_file_path(directory: str | Path, job_id: str) -> Path:
    """Return the component-owned decision filename for one Slurm job."""
    clean_job_id = str(job_id).strip()
    if not clean_job_id or Path(clean_job_id).name != clean_job_id:
        raise ValueError(f"invalid Slurm job ID: {job_id!r}")
    return Path(directory) / f"segment_health_check.{clean_job_id}.state"


@dataclass(frozen=True)
class DecisionObservation:
    """One excluded-node observation included in a decision artifact."""

    node: str
    state: str
    reason: str
    observed_at: float
    array_tasks: tuple[tuple[str, int], ...]


def build_decision_records(
    *,
    job_id: str,
    generated_at: float,
    cache_ttl_seconds: float,
    observations: Sequence[DecisionObservation],
) -> tuple[dict[str, object], dict[str, object]]:
    """Build task- and node-scoped Scheduler Exclusion decisions."""
    task_valid_until: dict[tuple[str, int], float] = {}
    node_valid_until: dict[str, float] = {}
    for observation in observations:
        valid_until = observation.observed_at + cache_ttl_seconds
        node_valid_until[observation.node] = valid_until
        for task in observation.array_tasks:
            task_valid_until[task] = max(task_valid_until.get(task, 0.0), valid_until)

    excluded_tasks = [
        {
            "task_id": task_id,
            "restart_count": restart_count,
            "valid_until": _format_timestamp(valid_until),
        }
        for (task_id, restart_count), valid_until in sorted(
            task_valid_until.items(), key=lambda item: _task_sort_key(item[0])
        )
    ]
    excluded_nodes = [
        {
            "node": node,
            "valid_until": _format_timestamp(valid_until),
        }
        for node, valid_until in sorted(node_valid_until.items())
    ]
    common = {
        "type": "decision",
        "schema_version": 1,
        "job_id": job_id,
        "generated_at": _format_timestamp(generated_at),
    }
    return (
        {
            **common,
            "scope": "array_task",
            "excluded_array_tasks": excluded_tasks,
        },
        {
            **common,
            "scope": "node",
            "excluded_nodes": excluded_nodes,
        },
    )


def build_decision_response(
    *,
    job_id: str,
    generated_at: float,
    cache_ttl_seconds: float,
    observations: Sequence[DecisionObservation],
) -> dict[str, object]:
    """Build the combined HTTP representation of one decision snapshot."""
    task_decision, node_decision = build_decision_records(
        job_id=job_id,
        generated_at=generated_at,
        cache_ttl_seconds=cache_ttl_seconds,
        observations=observations,
    )
    return {
        "type": "decision",
        "schema_version": task_decision["schema_version"],
        "job_id": job_id,
        "generated_at": task_decision["generated_at"],
        "excluded_array_tasks": task_decision["excluded_array_tasks"],
        "excluded_nodes": node_decision["excluded_nodes"],
    }


class DecisionFileWriter:
    """Publish compact task exclusions plus detailed decision evidence as JSONL."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def publish(
        self,
        *,
        job_id: str,
        generated_at: float,
        cache_ttl_seconds: float,
        observations: Sequence[DecisionObservation],
    ) -> tuple[int, int]:
        """Atomically replace the artifact and return task and node counts."""
        task_decision, node_decision = build_decision_records(
            job_id=job_id,
            generated_at=generated_at,
            cache_ttl_seconds=cache_ttl_seconds,
            observations=observations,
        )
        excluded_task_ids = list(
            dict.fromkeys(entry["task_id"] for entry in task_decision["excluded_array_tasks"])
        )
        records: list[object] = [excluded_task_ids, task_decision, node_decision]
        for observation in sorted(observations, key=lambda item: item.node):
            records.append(
                {
                    "type": "observation",
                    "node": observation.node,
                    "state": observation.state,
                    "reason": observation.reason,
                    "observed_at": _format_timestamp(observation.observed_at),
                    "valid_until": _format_timestamp(observation.observed_at + cache_ttl_seconds),
                    "array_tasks": [
                        {
                            "task_id": task_id,
                            "restart_count": restart_count,
                        }
                        for task_id, restart_count in sorted(
                            observation.array_tasks, key=_task_sort_key
                        )
                    ],
                }
            )

        self._atomic_write(records)
        return (
            len(task_decision["excluded_array_tasks"]),
            len(node_decision["excluded_nodes"]),
        )

    def _atomic_write(self, records: Sequence[object]) -> None:
        temporary_path: Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, separators=(",", ":")))
                    stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.path)
        except OSError as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise DecisionFileWriteError(
                f"could not publish Scheduler Exclusion decision to {self.path}: {exc}"
            ) from exc


def _task_sort_key(task: tuple[str, int]) -> tuple[int, int, str, int]:
    task_id, restart_count = task
    if task_id.isdigit():
        return (0, int(task_id), task_id, restart_count)
    return (1, 0, task_id, restart_count)


def _format_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")
