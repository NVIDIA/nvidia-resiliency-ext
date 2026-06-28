# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slurm node-state queries for nvidia_resiliency_ext.services.nodestatesvc."""

from __future__ import annotations

import logging
import re
import shutil

# Fixed Slurm argv is invoked with shell=False.
import subprocess  # nosec B404
from collections.abc import Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_BAD_STATES = frozenset(
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


class SlurmQueryError(RuntimeError):
    """Raised when a Slurm node-state query cannot be completed."""


@dataclass(frozen=True)
class NodeStateRecord:
    """Scheduler-visible state for one node."""

    node: str
    state: str
    raw_state: str
    bad: bool
    reason: str = ""
    slurm_visible: bool = True

    def to_dict(self) -> dict:
        return {
            "node": self.node,
            "state": self.state,
            "raw_state": self.raw_state,
            "bad": self.bad,
            "reason": self.reason,
            "slurm_visible": self.slurm_visible,
        }


def _resolve_slurm_command(command: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(command)
    return resolved


def _run_slurm_command(
    cmd: list[str],
    *,
    timeout: int | float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def normalize_node_state(raw_state: str) -> str:
    """Normalize Slurm's decorated state text into a stable lowercase state."""
    normalized = raw_state.strip().lower()
    normalized = re.sub(r"[\s-]+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
    return normalized or "unknown"


class SlurmNodeStateClient:
    """Batched Slurm client for scheduler-visible node state."""

    def __init__(
        self,
        *,
        timeout: int | float = 30.0,
        batch_size: int = 512,
        bad_states: set[str] | frozenset[str] = DEFAULT_BAD_STATES,
    ):
        self.timeout = timeout
        self.batch_size = batch_size
        self.bad_states = frozenset(normalize_node_state(state) for state in bad_states)
        self._command_paths: dict[str, str] = {}

    def _command(self, command: str) -> str:
        path = self._command_paths.get(command)
        if path is None:
            path = _resolve_slurm_command(command)
            self._command_paths[command] = path
        return path

    def check_available(self) -> tuple[bool, str | None]:
        try:
            result = _run_slurm_command([self._command("sinfo"), "--version"], timeout=5)
        except FileNotFoundError:
            return False, "sinfo command not found"
        except subprocess.TimeoutExpired as exc:
            return False, f"sinfo --version timed out after {exc.timeout}s"
        except OSError as exc:
            return False, f"sinfo could not be started: {exc}"
        if result.returncode != 0:
            return False, (result.stderr or result.stdout or "sinfo --version failed").strip()
        return True, None

    def get_node_states(self, nodes: Sequence[str]) -> dict[str, NodeStateRecord]:
        clean_nodes = _dedupe(nodes)
        if not clean_nodes:
            return {}
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        records: dict[str, NodeStateRecord] = {}
        for start in range(0, len(clean_nodes), self.batch_size):
            batch = clean_nodes[start : start + self.batch_size]
            records.update(self._get_node_states_batch(batch))

        for node in clean_nodes:
            if node not in records:
                records[node] = NodeStateRecord(
                    node=node,
                    state="UNKNOWN",
                    raw_state="UNKNOWN",
                    bad=True,
                    reason="node was not returned by sinfo",
                    slurm_visible=False,
                )
        return records

    def _get_node_states_batch(self, nodes: Sequence[str]) -> dict[str, NodeStateRecord]:
        try:
            result = _run_slurm_command(
                [
                    self._command("sinfo"),
                    "--noheader",
                    "--Node",
                    "--nodes",
                    ",".join(nodes),
                    "--format=%N|%T|%E",
                ],
                timeout=self.timeout,
            )
        except FileNotFoundError as exc:
            raise SlurmQueryError("sinfo command not found") from exc
        except subprocess.TimeoutExpired as exc:
            raise SlurmQueryError(f"sinfo timed out after {exc.timeout}s") from exc
        except OSError as exc:
            raise SlurmQueryError(f"sinfo could not be started: {exc}") from exc

        if result.returncode != 0:
            message = (result.stderr or result.stdout or "sinfo failed").strip()
            raise SlurmQueryError(message)

        return _parse_sinfo_output(result.stdout, self.bad_states)


def _dedupe(nodes: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for node in nodes:
        clean = str(node).strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _parse_sinfo_output(
    output: str,
    bad_states: set[str] | frozenset[str],
) -> dict[str, NodeStateRecord]:
    records: dict[str, NodeStateRecord] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        node, raw_state, reason = _split_sinfo_line(line)
        state = normalize_node_state(raw_state)
        records[node] = NodeStateRecord(
            node=node,
            state=state.upper(),
            raw_state=raw_state,
            bad=state in bad_states,
            reason=reason,
        )
    return records


def _split_sinfo_line(line: str) -> tuple[str, str, str]:
    parts = line.split("|", 2)
    if len(parts) == 1:
        return parts[0].strip(), "UNKNOWN", ""
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip(), ""
    return parts[0].strip(), parts[1].strip(), parts[2].strip()
