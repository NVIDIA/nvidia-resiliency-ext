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

"""Pure parsers: strings and timestamps in, values out.

Mostly no I/O -- kept apart from ``readers`` so ``detectors`` (which must do no I/O) can
import ``expand_nodelist`` without pulling a reader into its import graph, keeping
"detectors are pure" a structural fact. The one exception is batch-script resolution,
which follows ``source`` includes and so must read those files; it takes an injectable
``read_file`` (real filesystem by default) and stays read-only.
"""

from __future__ import annotations

import posixpath
import re
from datetime import datetime, timezone
from typing import Callable

_RANGE_RE = re.compile(r"^(.*?)\[([^\]]*)\](.*)$")


def parse_iso(value: str | None) -> datetime | None:
    """Parse NVRx's ISO 8601 UTC timestamps ('...Z'). Returns None on anything else."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def split_nodelist(nodelist: str) -> list[str]:
    """Split a Slurm nodelist on top-level commas, keeping bracketed ranges intact."""
    tokens: list[str] = []
    depth = 0
    current: list[str] = []
    for char in nodelist:
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            tokens.append("".join(current))
            current = []
            continue
        current.append(char)
    tokens.append("".join(current))
    return [t.strip() for t in tokens if t.strip()]


def expand_nodelist(nodelist: str) -> list[str]:
    """Expand 'node[001-003,007],other1' into individual hostnames.

    Re-implemented here rather than imported from nvidia_resiliency_ext: the watcher
    runs on a login node, outside the training container, where NVRx need not be
    installed. Stdlib only is a hard requirement for this package.
    """
    hosts: list[str] = []
    for token in split_nodelist(nodelist or ""):
        match = _RANGE_RE.match(token)
        if not match:
            hosts.append(token)
            continue
        prefix, body, suffix = match.groups()
        for part in body.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                low, _, high = part.partition("-")
                if not (low.isdigit() and high.isdigit()):
                    hosts.append(f"{prefix}{part}{suffix}")
                    continue
                width = len(low)
                for number in range(int(low), int(high) + 1):
                    hosts.append(f"{prefix}{number:0{width}d}{suffix}")
            else:
                hosts.append(f"{prefix}{part}{suffix}")
    return hosts


# ---------------------------------------------------------------------------------
# ft_launcher argument resolution from a batch script
# ---------------------------------------------------------------------------------
# nvrx-watch reads the cycle-info and checkpoint paths straight from the ft_launcher
# arguments in a job's batch script (retrieved by job id), so it works for any InJob
# sbatch, not just this example. Real production sbatches are seldom self-contained: they
# `source` common libraries that build the ft arguments and hold the path variables, and
# they root those paths with `$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)` relative to the
# script's own location. So resolution follows `source`/`.` includes (reading each file)
# and evaluates the handful of path-shaped command substitutions (dirname/basename/
# realpath, `cd ... && pwd`, and `scontrol ... Command=` -> the known script path), seeded
# with the script's path and the array/job id. The id becomes '*' so the cycle-info glob
# spans every generation. Anything outside these idioms (an arbitrary command, or an env
# passed via --export and never assigned) stays unresolved and the caller falls back.

_ASSIGN_RE = re.compile(r"^(?:export\s+)?([A-Za-z_]\w*)=(.*)$")
_SOURCE_RE = re.compile(r"^(?:source|\.)\s+(.+)$")
_VAR_RE = re.compile(r"\$\{([A-Za-z_]\w*)(?::[-=]([^{}]*))?\}|\$([A-Za-z_]\w*)")
_CMDSUB_RE = re.compile(r"\$\(([^()]*)\)")  # innermost $(...) -- no nested parens
_BACKTICK_RE = re.compile(r"`([^`]*)`")
_SENTINEL = "\x00"  # marks a variable we could not resolve
_MAX_SOURCE_DEPTH = 16
_MAX_SOURCE_BYTES = 1_000_000


def _default_read_file(path: str) -> str | None:
    """Read a sourced file, read-only and size-capped; None if it cannot be read."""
    try:
        with open(path) as handle:
            return handle.read(_MAX_SOURCE_BYTES)
    except OSError:
        return None


def _strip_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def _resolve_expr(expr: str, values: dict[str, str]) -> str | None:
    """Expand ${VAR}/$VAR/${VAR:-default} against ``values`` (innermost first). Returns
    None if anything is left unresolved (an unknown variable, or leftover $/${)."""
    for _ in range(200):
        match = _VAR_RE.search(expr)
        if not match:
            break
        name = match.group(1) or match.group(3)
        default = match.group(2)
        if name in values:
            rep = values[name]
        elif default is not None:
            rep = default
        else:
            rep = _SENTINEL
        expr = expr[: match.start()] + rep + expr[match.end() :]
    if _SENTINEL in expr or "$" in expr:
        return None
    return expr


def _eval_cmdsub(inner: str, values: dict[str, str], script_path: str | None) -> str | None:
    """Evaluate the path-shaped command substitutions real sbatches use to root paths at
    the script's location. Returns None for anything else -- we never run commands."""
    body = inner.strip()
    # `scontrol show job ... | awk .../Command=/...` -- the script's own path, which we know.
    if "scontrol" in body and "Command" in body:
        return script_path
    match = re.match(r"dirname\s+(.+)$", body)
    if match:
        arg = _resolve_expr(_strip_quotes(match.group(1)), values)
        return posixpath.dirname(arg) if arg is not None else None
    match = re.match(r"basename\s+(.+)$", body)
    if match:
        arg = _resolve_expr(_strip_quotes(match.group(1)), values)
        return posixpath.basename(arg) if arg is not None else None
    match = re.match(r"(?:realpath|readlink(?:\s+-\w+)?)\s+(.+)$", body)
    if match:
        arg = _resolve_expr(_strip_quotes(match.group(1)), values)
        return posixpath.normpath(arg) if arg is not None else None
    # `cd <path> [redirs] && pwd`  /  `cd <path> ; pwd`
    match = re.match(r"cd\s+(.+?)(?:\s+\d?>\S+)*\s*(?:&&|;)\s*pwd\s*$", body)
    if match:
        arg = _resolve_expr(_strip_quotes(match.group(1)), values)
        return posixpath.normpath(arg) if arg is not None else None
    return None


def _eval_rhs(raw: str, values: dict[str, str], script_path: str | None) -> str | None:
    """Resolve an assignment RHS or a source argument: strip quotes, evaluate any
    path-shaped command substitutions innermost-first, then expand variables. None if it
    cannot be fully resolved (unknown var, or a command substitution we do not emulate)."""
    expr = _strip_quotes(raw)
    for pattern in (_CMDSUB_RE, _BACKTICK_RE):
        for _ in range(50):
            match = pattern.search(expr)
            if not match:
                break
            value = _eval_cmdsub(match.group(1), values, script_path)
            if value is None:
                return None
            expr = expr[: match.start()] + value + expr[match.end() :]
    if "$(" in expr or "`" in expr:
        return None
    return _resolve_expr(expr, values)


def _find_ft_arg(script: str, arg: str) -> str | None:
    """The raw (unresolved) value of ``--<arg>=X`` or ``--<arg> X`` in the script."""
    match = re.search(r"--" + re.escape(arg) + r"[= ]([^\s\\'\"]+)", script)
    return match.group(1) if match else None


def _walk_script(
    text: str,
    script_path: str | None,
    values: dict[str, str],
    ft_raw: dict[str, str | None],
    read_file: Callable[[str], str | None],
    seen: set[str],
    depth: int,
) -> None:
    """Process one script's lines in order -- recording assignments and following
    ``source`` into included files -- so a variable is known before the line that uses it.
    Collects the ft-argument templates (last occurrence across the whole include tree)."""
    for name, patterns in (
        ("cyc", ("ft-cycle-info-dir", "ft_cycle_info_dir")),
        ("ckpt", ("ft-checkpoint-iteration-file", "ft_checkpoint_iteration_file")),
    ):
        for pattern in patterns:
            found = _find_ft_arg(text, pattern)
            if found:
                ft_raw[name] = found

    for line in text.splitlines():
        stripped = line.strip()
        assign = _ASSIGN_RE.match(stripped)
        if assign:
            resolved = _eval_rhs(assign.group(2), values, script_path)
            if resolved is not None:
                values[assign.group(1)] = resolved
            continue
        source = _SOURCE_RE.match(stripped)
        if source and depth < _MAX_SOURCE_DEPTH:
            target = _eval_rhs(source.group(1).split()[0], values, script_path)
            if target and "*" not in target and target not in seen:
                seen.add(target)
                included = read_file(target)
                if included is not None:
                    _walk_script(included, target, values, ft_raw, read_file, seen, depth + 1)


def resolve_ft_launcher_paths(
    script: str,
    *,
    script_path: str | None = None,
    read_file: Callable[[str], str | None] | None = None,
) -> tuple[str | None, str | None]:
    """Resolve --ft-cycle-info-dir and --ft-checkpoint-iteration-file from a batch script
    to concrete paths, with the array/job id replaced by '*' so the cycle-info glob spans
    generations. Follows ``source`` includes (via ``read_file``, the real filesystem by
    default) and roots paths at ``script_path`` (the job's Command=, needed by sbatches that
    locate their libraries relative to their own path). Returns (cycle_info_glob,
    checkpoint_file); either is None when the arg is absent or cannot be resolved."""
    values: dict[str, str] = {"SLURM_ARRAY_JOB_ID": "*", "SLURM_JOB_ID": "*"}
    if script_path:
        values["SCRIPT_PATH"] = script_path
        values["BASH_SOURCE"] = script_path
    ft_raw: dict[str, str | None] = {"cyc": None, "ckpt": None}
    _walk_script(script, script_path, values, ft_raw, read_file or _default_read_file, set(), 0)

    cyc = _resolve_expr(ft_raw["cyc"], values) if ft_raw["cyc"] else None
    cycle_info_glob = cyc.rstrip("/") + "/cycle_info.*" if cyc else None

    ckpt = _resolve_expr(ft_raw["ckpt"], values) if ft_raw["ckpt"] else None
    if ckpt and "*" in ckpt:  # a checkpoint file is a single path, not a per-generation glob
        ckpt = None
    return cycle_info_glob, ckpt
