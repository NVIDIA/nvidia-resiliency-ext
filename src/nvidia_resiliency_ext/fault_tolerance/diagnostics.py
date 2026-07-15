# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort native diagnostics for fault-tolerance worker processes."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import socket
import subprocess  # nosec B404
import tempfile
import time
from pathlib import Path
from typing import Optional

CORE_DUMP_CLAIM_FILE_ENV_VAR = "NVRX_FT_CORE_DUMP_CLAIM_FILE"


def core_dump_claim_file(launcher_pid: int) -> str:
    """Return the node-local claim file shared by a launcher and its rank monitors."""
    return os.path.join(tempfile.gettempdir(), f"_ft_launcher{launcher_pid}_core_dump.claim")


def reset_core_dump_claim(claim_file: Optional[str]) -> None:
    """Start a new worker termination wave."""
    if claim_file:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(claim_file)


def _claim_core_dump(claim_file: Optional[str], pid: int) -> bool:
    if not claim_file:
        return True
    try:
        fd = os.open(claim_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(f"pid={pid}\n")
    return True


def _release_core_dump_claim(claim_file: Optional[str]) -> None:
    if claim_file:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(claim_file)


def _dump_directory(configured_path: Optional[str]) -> Path:
    path = Path(configured_path or "nvrx_dumps").expanduser().resolve()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _dump_stem(kind: str, pid: int, rank: Optional[int], cycle: int) -> str:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rank_text = "unknown" if rank is None else str(rank)
    hostname = socket.gethostname().replace(os.sep, "_")
    return f"nvrx_{kind}_{hostname}_cycle{cycle}_rank{rank_text}_pid{pid}_{timestamp}"


def capture_full_core_dump_once(
    *,
    pid: int,
    rank: Optional[int],
    cycle: int,
    dump_dir: Optional[str],
    timeout: float,
    claim_file: Optional[str],
    logger: logging.Logger,
) -> Optional[str]:
    """Capture one full core per claim file, returning its path on success."""
    if not _claim_core_dump(claim_file, pid):
        logger.debug("A full core dump was already captured for this termination wave")
        return None

    output_path: Optional[Path] = None
    try:
        directory = _dump_directory(dump_dir)
        stem = _dump_stem("core", pid, rank, cycle)
        gcore = shutil.which("gcore")
        gdb = shutil.which("gdb")
        if gcore:
            prefix = directory / stem
            output_path = Path(f"{prefix}.{pid}")
            command = [gcore, "-o", str(prefix), str(pid)]
        elif gdb:
            output_path = directory / f"{stem}.core"
            command = [
                gdb,
                "--batch",
                "--nx",
                "--quiet",
                "-ex",
                "set pagination off",
                "-ex",
                f"generate-core-file {json.dumps(str(output_path))}",
                "-p",
                str(pid),
            ]
        else:
            logger.warning(
                "Cannot capture a full core for PID %s: neither gcore nor gdb is installed", pid
            )
            _release_core_dump_claim(claim_file)
            return None

        logger.warning("Capturing full core dump for rank %s (PID=%s) before SIGTERM", rank, pid)
        result = subprocess.run(  # nosec B603
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0 or not output_path.exists():
            logger.warning(
                "Full core capture failed for rank %s (PID=%s), exit code %s: %s",
                rank,
                pid,
                result.returncode,
                result.stdout.strip(),
            )
            with contextlib.suppress(OSError):
                output_path.unlink()
            _release_core_dump_claim(claim_file)
            return None

        logger.warning("Full core dump for rank %s (PID=%s) written to %s", rank, pid, output_path)
        return str(output_path)
    except subprocess.TimeoutExpired:
        logger.warning(
            "Full core capture for rank %s (PID=%s) timed out after %.1fs", rank, pid, timeout
        )
    except Exception:
        logger.warning("Full core capture failed for rank %s (PID=%s)", rank, pid, exc_info=True)

    if output_path is not None:
        with contextlib.suppress(OSError):
            output_path.unlink()
    _release_core_dump_claim(claim_file)
    return None


def capture_stack_trace(
    *,
    pid: int,
    rank: Optional[int],
    cycle: int,
    dump_dir: Optional[str],
    timeout: float,
    logger: logging.Logger,
) -> Optional[str]:
    """Capture all native thread stacks with gdb, returning the text path on success."""
    gdb = shutil.which("gdb")
    if not gdb:
        logger.warning("Cannot capture a stack trace for PID %s: gdb is not installed", pid)
        return None

    output_path: Optional[Path] = None
    try:
        directory = _dump_directory(dump_dir)
        output_path = directory / f"{_dump_stem('stack', pid, rank, cycle)}.txt"
        command = [
            gdb,
            "--batch",
            "--nx",
            "--quiet",
            "-ex",
            "set pagination off",
            "-ex",
            "set print thread-events off",
            "-ex",
            "thread apply all bt full",
            "-p",
            str(pid),
        ]
        with output_path.open("w", encoding="utf-8") as stream:
            result = subprocess.run(  # nosec B603
                command,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
        if result.returncode != 0:
            logger.warning(
                "Stack trace capture failed for rank %s (PID=%s), exit code %s; gdb output is in %s",
                rank,
                pid,
                result.returncode,
                output_path,
            )
            return None
        logger.warning("Stack trace for rank %s (PID=%s) written to %s", rank, pid, output_path)
        return str(output_path)
    except subprocess.TimeoutExpired:
        logger.warning(
            "Stack trace capture for rank %s (PID=%s) timed out after %.1fs", rank, pid, timeout
        )
    except Exception:
        logger.warning("Stack trace capture failed for rank %s (PID=%s)", rank, pid, exc_info=True)
    return None
