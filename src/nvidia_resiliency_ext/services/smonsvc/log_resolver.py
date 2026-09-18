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
    (default ``logs``).
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
_CYCLE_RE = re.compile(r"_cycle(\d+)\.log$")


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
    match = _CYCLE_RE.search(path.name)
    return int(match.group(1)) if match else -1


def resolve_app_log(
    stdout_path: str,
    job_id: str,
    config: AppLogResolution,
) -> Optional[str]:
    """Return the application log for ``job_id``, or ``None`` to keep ``stdout_path``.

    Returns ``None`` whenever resolution is disabled, the layout does not match,
    or no cycle log exists for the job — callers fall back to the original path
    rather than dropping the job.
    """
    if not config.enabled or not stdout_path:
        return None

    stub = Path(stdout_path)
    parent = stub.parent
    # Strip the wrapper directory when present; otherwise treat the wrapper's
    # own directory as the run directory.
    run_dir = parent.parent if parent.name == config.stdout_subdir else parent
    log_dir = run_dir / config.log_subdir
    if not log_dir.is_dir():
        return None

    base = base_job_id(job_id)
    if not base:
        return None

    candidates = [p for p in log_dir.glob(f"*_{base}_*_cycle*.log") if p.is_file()]
    if not candidates:
        return None

    # The highest cycle is the attempt that produced the terminal outcome;
    # mtime only breaks ties between equally numbered cycles.
    best = max(candidates, key=lambda p: (_cycle_number(p), p.stat().st_mtime))
    return str(best)
