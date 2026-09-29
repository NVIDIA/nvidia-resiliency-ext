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

import glob
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

#: Shell forms evaluated statically when reading a submit script.
_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_VAR_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::[-=][^}]*)?\}|\$([A-Za-z_][A-Za-z0-9_]*)")
#: Values we refuse to evaluate: command substitution and array literals.
_DYNAMIC_MARKERS = ("$(", "`", "(")
_SCRIPT_MAX_LINES = 900
_LOGS_DIR_VAR = "LOGS_DIR"
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


def _is_dir(path: Path) -> bool:
    """``is_dir`` that treats an unreadable path as "not a directory".

    Shared run trees contain directories this account cannot stat, including
    ones created by unexpanded shell variables. Resolution must step over them
    rather than fail the poll.
    """
    try:
        return path.is_dir()
    except OSError:
        return False


def _candidate_log_dirs(run_dir: Path, log_subdir: str) -> list[Path]:
    """Log directories to search: beside the wrapper, then one level down.

    Some launchers write ``StdOut`` above the run directory, leaving
    ``<parent>/slurm-<jobid>.out`` next to ``<parent>/<phase>/logs/``. Searching
    one level of subdirectory covers that without walking the tree. Matches stay
    anchored on the job ID, so a deeper hit is still unambiguously this job's log.
    """
    dirs = []
    direct = run_dir / log_subdir
    if _is_dir(direct):
        dirs.append(direct)
    try:
        children = sorted(run_dir.iterdir())
    except OSError:
        return dirs
    for child in children:
        nested = child / log_subdir
        if _is_dir(nested):
            dirs.append(nested)
    return dirs


def _script_assignments(script: Path) -> dict:
    """Static ``NAME=value`` assignments from a submit script; first one wins."""
    try:
        lines = script.read_text(errors="ignore").splitlines()[:_SCRIPT_MAX_LINES]
    except OSError:
        return {}
    env = {}
    for line in lines:
        match = _ASSIGNMENT.match(line)
        if not match:
            continue
        name, raw = match.group(1), match.group(2).split(" #")[0].strip()
        if raw[:1] in ('"', "'") and raw[-1:] == raw[:1] and len(raw) > 1:
            raw = raw[1:-1]
        if any(marker in raw for marker in _DYNAMIC_MARKERS):
            continue
        env.setdefault(name, raw)
    return env


def script_log_dir_pattern(script: Path) -> Optional[str]:
    """``LOGS_DIR`` from a submit script, each unknown variable becoming one ``*``.

    Full resolution is not required and demanding it throws away most of the
    value. A script building ``${SMOKE_ROOT}/${TAG}/logs`` still pins everything
    above ``TAG``, which turns an unbounded search into a single-level glob.
    The variables left as wildcards are precisely those supplied by the
    submitting environment, which SLURM does not record.
    """
    env = _script_assignments(script)
    value = env.get(_LOGS_DIR_VAR)
    if not value:
        return None

    def resolve(text: str, seen: frozenset, depth: int = 0) -> str:
        if depth > 12:
            return "*"

        def substitute(match) -> str:
            name = match.group(1) or match.group(2)
            if name in seen or name not in env:
                return "*"
            return resolve(env[name], seen | {name}, depth + 1)

        return _VAR_REF.sub(substitute, text)

    pattern = resolve(value, frozenset())
    # A relative or wildcard-rooted pattern would glob somewhere unintended.
    return pattern if pattern.startswith("/") and not pattern.startswith("/*") else None


def logs_from_pattern(pattern: str, base: str) -> list:
    """Logs for ``base`` under a possibly wildcarded directory pattern."""
    found = []
    try:
        matches = glob.glob(f"{pattern}/*_{base}_*.log")
    except OSError:
        return found
    for candidate in matches:
        path = Path(candidate)
        if _is_sidecar(path):
            continue
        try:
            if path.is_file() and path.stat().st_size > 0:
                found.append(path)
        except OSError:
            continue
    return found


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


def logs_from_job_name(root: Path, job_name: str, base: str, max_depth: int = 5) -> list:
    """Find logs by reading the run path out of the SLURM job name.

    Job names mirror the run directory with ``/`` flattened to ``_``, so both the
    split points and the offset at which the name starts matching are ambiguous. Trying each split and keeping only directories
    that exist resolves it with a handful of stat calls. Unlike the submit
    script this is recorded on the job itself, so it stays correct for a job
    whose script has since been edited.
    """
    tokens = [t for t in job_name.split("_") if t]
    if not tokens:
        return []
    found: list = []
    seen: set = set()

    def walk(prefix: Path, remaining: list, depth: int) -> None:
        if depth > max_depth:
            return
        log_dir = prefix / "logs"
        key = str(log_dir)
        if key not in seen and _is_dir(log_dir):
            seen.add(key)
            found.extend(_logs_for_job(log_dir, base))
        if not remaining:
            return
        for take in range(1, len(remaining) + 1):
            nxt = prefix / "_".join(remaining[:take])
            if _is_dir(nxt):
                walk(nxt, remaining[take:], depth + 1)

    # The search root already consumes some leading path components, and the name
    # carries product tokens the path never had, so the matching suffix starts at
    # an unknown offset. Each offset that does not correspond to a real directory
    # terminates immediately, so trying all of them stays cheap.
    for start in range(len(tokens)):
        walk(root, tokens[start:], 0)
    return found


def resolve_app_log(
    stdout_path: str,
    job_id: str,
    config: AppLogResolution,
    *,
    script: Optional[str] = None,
    job_name: str = "",
    search_root: Optional[str] = None,
) -> Optional[str]:
    """Return the application log for ``job_id``, or ``None`` when there is none.

    Sources are tried cheapest first, and each is anchored on the parent job ID
    so a wrong directory yields a miss rather than another job's log:

    1. the directory layout around the wrapper;
    2. ``LOGS_DIR`` from the submit script, wildcarding what it cannot resolve;
    3. directories implied by the SLURM job name; and
    4. a ``LOGS_DIR=`` line in the wrapper's own banner.

    ``None`` means resolution is disabled, or the job produced no non-empty log.
    Callers decide what that means; the monitor skips the job rather than
    analyzing the wrapper.
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

    if not candidates and script:
        # The submit script states where logs go, and states it precisely enough
        # even when part of the path comes from the submitting environment.
        pattern = script_log_dir_pattern(Path(script))
        if pattern:
            candidates = logs_from_pattern(pattern, base)

    if not candidates and job_name:
        # Recorded on the job, so unaffected by later edits to the script.
        root = Path(search_root) if search_root else run_dir.parent
        candidates = logs_from_job_name(root, job_name, base)

    if not candidates:
        # Some launchers declare the directory outright in their banner.
        declared = declared_log_dir(stub)
        if declared is not None and _is_dir(declared):
            candidates = _logs_for_job(declared, base)

    if not candidates:
        return None

    # The highest cycle is the attempt that produced the terminal outcome, and a
    # cycle log outranks a plain one. mtime only breaks ties within a rank.
    best = max(candidates, key=lambda p: (_cycle_number(p), p.stat().st_mtime))
    return str(best)
