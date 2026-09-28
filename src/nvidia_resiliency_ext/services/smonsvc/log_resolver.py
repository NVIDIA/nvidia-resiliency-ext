# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a SLURM stdout path to the application log that job actually wrote.

Some launchers point ``StdOut`` at a batch wrapper under
``<run_dir>/slurm_out/slurm-<jobid>_<task>.out`` while the training process
writes to ``<run_dir>/logs/<name>_<jobid>_date_..._cycle<N>.log``. The wrapper
holds a few KB of launcher banner and no training output, so attributing it
yields "no failure signature found" no matter what the job did.

When enabled, this maps the wrapper back to the newest cycle log for the same
SLURM job. It is **off by default**: it encodes a site layout convention, and a
deployment whose ``StdOut`` already is the training log must not have its paths
rewritten underneath it.

Environment:

``NVRX_SMONSVC_APP_LOG_RESOLUTION``
    Set to ``1``/``true`` to enable.
``NVRX_SMONSVC_APP_LOG_STDOUT_SUBDIR``
    Directory holding the wrapper, stripped to find the run directory
    (default ``slurm_out``).
``NVRX_SMONSVC_APP_LOG_SUBDIR``
    Directory holding application logs, relative to the run directory
    (default ``logs``). Searched beside the wrapper and one level below it,
    since some launchers write ``StdOut`` above the run directory.

Two naming conventions are handled: multi-cycle runs write
``..._<jobid>_date_..._cycle<N>.log`` and single-cycle runs write
``..._<jobid>_date_....log``. A cycle log always wins over a plain one, and
metadata sidecars such as ``.env.log`` and ``.tasks.log`` are never selected.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ENABLED_ENV = "NVRX_SMONSVC_APP_LOG_RESOLUTION"
STDOUT_SUBDIR_ENV = "NVRX_SMONSVC_APP_LOG_STDOUT_SUBDIR"
LOG_SUBDIR_ENV = "NVRX_SMONSVC_APP_LOG_SUBDIR"

DEFAULT_STDOUT_SUBDIR = "slurm_out"
DEFAULT_LOG_SUBDIR = "logs"

_TRUE_VALUES = ("1", "true", "yes", "on")

#: Launchers that emit a paths banner put it at the very top of the wrapper.
_LOGS_DIR_KEY = "LOGS_DIR="
_BANNER_MAX_LINES = 400
_CYCLE_RE = re.compile(r"_cycle(\d+)\.log$")

#: Metadata written alongside the training log; never the analysis target.
SIDECAR_SUFFIXES = (".env.log", ".tasks.log")


@dataclass(frozen=True)
class AppLogResolution:
    """Settings for mapping a SLURM stdout wrapper to the application log."""

    enabled: bool = False
    stdout_subdir: str = DEFAULT_STDOUT_SUBDIR
    log_subdir: str = DEFAULT_LOG_SUBDIR

    @classmethod
    def from_env(cls) -> "AppLogResolution":
        return cls(
            enabled=(os.environ.get(ENABLED_ENV, "") or "").strip().lower() in _TRUE_VALUES,
            stdout_subdir=(os.environ.get(STDOUT_SUBDIR_ENV) or DEFAULT_STDOUT_SUBDIR).strip(),
            log_subdir=(os.environ.get(LOG_SUBDIR_ENV) or DEFAULT_LOG_SUBDIR).strip(),
        )

    def describe(self) -> str:
        """One-line status suitable for a startup log."""
        if not self.enabled:
            return "disabled (submitting SLURM StdOut as-is)"
        return f"enabled ({self.stdout_subdir}/ -> {self.log_subdir}/*_cycle<N>.log)"


def base_job_id(job_id: str) -> str:
    """Strip array-task and het-job suffixes: ``123_4`` and ``123+1`` both give ``123``.

    Application logs embed the parent job ID, so every array task of a job maps
    to the same log.
    """
    text = str(job_id).strip()
    for separator in ("_", "+"):
        if separator in text:
            text = text.split(separator, 1)[0]
    return text


def _cycle_number(path: Path) -> int:
    """Cycle index, or ``-1`` for a single-cycle log that carries no suffix."""
    match = _CYCLE_RE.search(path.name)
    return int(match.group(1)) if match else -1


def _is_sidecar(path: Path) -> bool:
    return path.name.endswith(SIDECAR_SUFFIXES)


def _candidate_log_dirs(run_dir: Path, log_subdir: str) -> list[Path]:
    """Log directories to search: beside the wrapper, then one level down.

    Some launchers write ``StdOut`` above the run directory, leaving
    ``<parent>/slurm-<jobid>.out`` next to ``<parent>/<phase>/logs/``. Searching
    one level of subdirectory covers that without walking the tree. Matches stay
    anchored on the job ID, so a deeper hit is still unambiguously this job's log.
    """
    dirs = []
    direct = run_dir / log_subdir
    if direct.is_dir():
        dirs.append(direct)
    try:
        children = sorted(run_dir.iterdir())
    except OSError:
        return dirs
    for child in children:
        nested = child / log_subdir
        if nested.is_dir():
            dirs.append(nested)
    return dirs


def declared_log_dir(stdout_path: Path) -> Optional[Path]:
    """Read ``LOGS_DIR=`` from the wrapper's launcher banner, if it emits one.

    This is the launcher's own declaration, so it is authoritative where the
    directory layout is only an inference. Roughly half the wrappers observed
    emit it; the rest print no banner at all.
    """
    try:
        with stdout_path.open("r", errors="ignore") as handle:
            for _ in range(_BANNER_MAX_LINES):
                line = handle.readline()
                if not line:
                    break
                if line.startswith(_LOGS_DIR_KEY):
                    value = line[len(_LOGS_DIR_KEY) :].strip()
                    return Path(value) if value else None
    except OSError:
        return None
    return None


def _logs_for_job(log_dir: Path, base: str) -> list[Path]:
    """Non-empty, non-sidecar logs in ``log_dir`` belonging to ``base``."""
    found = []
    for path in log_dir.glob(f"*_{base}_*.log"):
        if not path.is_file() or _is_sidecar(path):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        # A launcher can open a log and die before writing to it. An empty file
        # is not evidence, so it must not outrank a populated one.
        if size > 0:
            found.append(path)
    return found


def resolve_app_log(
    stdout_path: str,
    job_id: str,
    config: AppLogResolution,
) -> Optional[str]:
    """Return the application log for ``job_id``, or ``None`` when there is none.

    ``None`` means resolution is disabled, the layout does not match, or the job
    produced no non-empty log. Callers decide what that means; the monitor skips
    the job rather than analyzing the wrapper.
    """
    if not config.enabled or not stdout_path:
        return None

    stub = Path(stdout_path)
    parent = stub.parent
    # Strip the wrapper directory when present; otherwise treat the wrapper's
    # own directory as the run directory.
    run_dir = parent.parent if parent.name == config.stdout_subdir else parent

    base = base_job_id(job_id)
    if not base:
        return None

    candidates = []
    for log_dir in _candidate_log_dirs(run_dir, config.log_subdir):
        candidates.extend(_logs_for_job(log_dir, base))

    if not candidates:
        # Layout inference failed. Some launchers declare the directory outright,
        # which reaches places structure cannot - a run submitted from one
        # directory can write its logs to an unrelated sibling.
        declared = declared_log_dir(stub)
        if declared is not None and declared.is_dir():
            candidates = _logs_for_job(declared, base)

    if not candidates:
        return None

    # The highest cycle is the attempt that produced the terminal outcome, and a
    # cycle log outranks a plain one. mtime only breaks ties within a rank.
    best = max(candidates, key=lambda p: (_cycle_number(p), p.stat().st_mtime))
    return str(best)
