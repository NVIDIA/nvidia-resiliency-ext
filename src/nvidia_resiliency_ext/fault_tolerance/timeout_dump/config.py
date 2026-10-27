# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Timeout dump configuration parsing, validation, and paths."""

import logging
import math
import os
import shutil
from pathlib import Path

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

logger = logging.getLogger(LogConfig.name)


def parse_capture_sections(section_list: str) -> frozenset[str]:
    """Select exact workload section names; these are not timeout-type aliases."""
    if not isinstance(section_list, str):
        raise ValueError("timeout_dump_sections must be a comma-separated list of section names")
    sections = tuple(section.strip() for section in section_list.split(","))
    if not all(sections) or len(set(sections)) != len(sections):
        raise ValueError("timeout_dump_sections must contain nonempty, unique section names")
    return frozenset(sections)


def parse_capture_tools(tool_list: str) -> tuple[str, ...]:
    """Parse and validate tool names, preserving the configured execution order."""
    if not isinstance(tool_list, str):
        raise ValueError("timeout_dump must be an ordered comma-separated tool list")
    tools = tuple(tool.strip() for tool in tool_list.split(","))
    if any(tool not in ("pyspy", "gdb", "cuda-gdb") for tool in tools):
        raise ValueError("timeout_dump must contain only pyspy, gdb, or cuda-gdb")
    if len(set(tools)) != len(tools):
        raise ValueError("timeout_dump must not contain duplicate tools")
    return tools


def tool_executable(cfg, tool):
    return getattr(cfg, "timeout_dump_" + tool.replace("-", "_"))


def debugger_command_file(cfg, tool):
    """Return the configured path without filesystem access on the capture path."""
    configured = getattr(cfg, "timeout_dump_" + tool.replace("-", "_") + "_command_file")
    if not isinstance(configured, str) or not configured.strip():
        raise ValueError(
            f"Timeout dump {tool} command file is required: "
            f"set --ft-timeout-dump-{tool}-command-file to a readable file"
        )
    return Path(configured)


def validate_capture_dependencies(cfg, *, log_resolved_paths=False):
    """Validate selected executables/files and store resolved paths at launcher startup."""
    tools = parse_capture_tools(cfg.timeout_dump)
    for tool in tools:
        executable = tool_executable(cfg, tool)
        resolved = shutil.which(executable) if isinstance(executable, str) and executable else None
        if resolved is None:
            raise ValueError(
                f"Timeout dump tool {tool!r} is missing or not executable: {executable!r}. "
                "Install it, set --ft-timeout-dump-" + tool + ", or adjust --ft-timeout-dump."
            )
        setattr(cfg, "timeout_dump_" + tool.replace("-", "_"), resolved)
        if tool in ("gdb", "cuda-gdb"):
            configured_path = debugger_command_file(cfg, tool)
            path = configured_path.resolve()
            if tool == "cuda-gdb" and cfg.timeout_dump_cuda_gdb_mode == "stdin":
                # A source filename is a command line. Check both the configured
                # path and its symlink target before storing the resolved path.
                if any("\n" in str(p) or "\r" in str(p) for p in (configured_path, path)):
                    raise ValueError(
                        "Timeout dump cuda-gdb stdin command file path must be a single line"
                    )
            try:
                if not path.is_file():
                    raise OSError("not a regular file")
                with path.open("rb") as stream:
                    stream.read(1)
            except OSError as exc:
                raise ValueError(
                    f"Timeout dump {tool} command file is not readable: {path}"
                ) from exc
            setattr(cfg, "timeout_dump_" + tool.replace("-", "_") + "_command_file", str(path))
    if log_resolved_paths and logger.isEnabledFor(logging.DEBUG):
        tool_details = []
        command_files = []
        for tool in tools:
            detail = f"{tool}={tool_executable(cfg, tool)}"
            if tool == "cuda-gdb":
                detail += f" (mode={cfg.timeout_dump_cuda_gdb_mode})"
            tool_details.append(detail)
            if tool in ("gdb", "cuda-gdb"):
                command_files.append(f"{tool}={debugger_command_file(cfg, tool)}")
        logger.debug("Timeout dump tools: %s", ", ".join(tool_details))
        if command_files:
            logger.debug("Timeout dump command files: %s", ", ".join(command_files))


def validate_config(cfg):
    if not isinstance(cfg.timeout_dump_enabled, bool):
        raise ValueError("timeout_dump_enabled must be a boolean")
    if not cfg.timeout_dump_enabled:
        return
    parse_capture_tools(cfg.timeout_dump)
    parse_capture_sections(cfg.timeout_dump_sections)
    # Programmatic configuration bypasses argparse choices, and Literal
    # annotations are not enforced at runtime. Keep this a user-facing error.
    if cfg.timeout_dump_cuda_gdb_mode not in ("batch", "stdin"):
        raise ValueError("timeout_dump_cuda_gdb_mode must be batch or stdin")
    for name in (
        "capture_timeout",
        "poll_interval",
        "shutdown_grace",
    ):
        value = getattr(cfg, "timeout_dump_" + name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"timeout_dump_{name} must be finite and positive")
    for name in ("max_capture_bytes", "upload_attempts"):
        value = getattr(cfg, "timeout_dump_" + name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"timeout_dump_{name} must be a positive integer")
    if cfg.timeout_dump_max_capture_bytes < 1024:
        raise ValueError("timeout_dump_max_capture_bytes must be at least 1024 (including headers)")
    if not os.path.isabs(cfg.timeout_dump_local_root):
        raise ValueError("timeout_dump_local_root must be absolute")


def diagnostics_directory(launcher_log_file):
    """Use the existing job/array log namespace; never reconstruct Slurm IDs."""
    if not launcher_log_file:
        raise ValueError("--ft-timeout-dump-enabled=true requires --ft-nvrx-logfile")
    path = Path(launcher_log_file)
    if not path.is_absolute() or path.parent.name != "logs":
        raise ValueError("Timeout dumps require --ft-nvrx-logfile=<nvrx-root>/logs/<filename>")
    return path.parent.parent / "diags"


def staging_dir(cfg):
    return Path(cfg.timeout_dump_local_root) / "nvrx-diags"
