# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Timeout dump configuration: shared settings, per-tool entries, and paths.

argparse checks every option value as it parses, so a bad option fails the launch like any
other flag. Each option's dest is its config path (``timeout_dump.<field>`` or
``timeout_dump.<tool>.<field>``), which `from_args` reads back.
"""

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

from ..arg_types import absolute_path, names, positive_float, positive_int
from .capture import ToolCapture, options_given
from .cuda_gdb import CudaGdb
from .gdb import Gdb
from .pyspy import PySpy

# Every tool, in the default execution order: cheapest and least invasive first.
# Adding a tool means adding its module and one entry here.
TOOLS = {tool.name: tool for tool in (PySpy, Gdb, CudaGdb)}


def create_capture(cfg, worker_pid, tool_cfg) -> ToolCapture:
    """A capture of `worker_pid` by the tool that `tool_cfg` configures."""
    return TOOLS[tool_cfg.name](cfg, worker_pid, tool_cfg)


def _default_tools():
    return [tool.Config() for tool in TOOLS.values()]


def _tool_names(text):
    values = names(text)
    unknown = [name for name in values if name not in TOOLS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown tools {', '.join(unknown)}; choose from {', '.join(TOOLS)}"
        )
    return values


def _capture_bytes(text):
    value = positive_int(text)
    if value < 1024:
        raise argparse.ArgumentTypeError(f"must be at least 1024, including headers, got {text!r}")
    return value


@dataclass
class TimeoutDumpConfig:
    """Timeout dump settings, built from the checked command-line options.

    `tools` is the ordered list of tools to run, every tool by default.
    """

    enabled: bool = False
    tools: list[ToolCapture.Config] = field(default_factory=_default_tools)
    sections: list[str] = field(default_factory=lambda: ["step"])
    local_root: str = "/tmp"
    capture_timeout: float = 30.0  # total for all tools
    max_capture_bytes: int = 1024 * 1024  # per tool, including header and footer
    poll_interval: float = 30.0
    upload_attempts: int = 3
    shutdown_grace: float = 10.0

    def resolve(self) -> "TimeoutDumpConfig":
        """Resolve tool executables and command files once, at launcher startup."""
        return dataclasses.replace(self, tools=[tool.resolve() for tool in self.tools])

    @classmethod
    def add_arguments(cls, parser):
        defaults = cls()
        group = parser.add_argument_group("Timeout dump")
        group.add_argument(
            "--ft-timeout-dump-enabled",
            dest="timeout_dump.enabled",
            action=argparse.BooleanOptionalAction,
            help="Capture stacks when a selected section times out (default: off). "
            "--ft-timeout-dump-tools also enables capture.",
        )
        group.add_argument(
            "--ft-timeout-dump-tools",
            dest="timeout_dump.tools",
            type=_tool_names,
            metavar="TOOL,...",
            help=f"Tools to run, in order (default: {','.join(TOOLS)}).",
        )
        group.add_argument(
            "--ft-timeout-dump-sections",
            dest="timeout_dump.sections",
            type=names,
            metavar="SECTION,...",
            help="Workload sections whose timeout triggers capture "
            f"(default: {','.join(defaults.sections)}).",
        )
        group.add_argument(
            "--ft-timeout-dump-local-root",
            dest="timeout_dump.local_root",
            type=absolute_path,
            metavar="DIR",
            help=f"Node-local staging root (default: {defaults.local_root}).",
        )
        group.add_argument(
            "--ft-timeout-dump-capture-timeout",
            dest="timeout_dump.capture_timeout",
            type=positive_float,
            metavar="SECONDS",
            help=f"Total capture time for all tools (default: {defaults.capture_timeout:g}).",
        )
        group.add_argument(
            "--ft-timeout-dump-max-capture-bytes",
            dest="timeout_dump.max_capture_bytes",
            type=_capture_bytes,
            metavar="BYTES",
            help="Output cap per tool, including headers; at least 1024 "
            f"(default: {defaults.max_capture_bytes}).",
        )
        group.add_argument(
            "--ft-timeout-dump-poll-interval",
            dest="timeout_dump.poll_interval",
            type=positive_float,
            metavar="SECONDS",
            help=f"Collector upload interval (default: {defaults.poll_interval:g}).",
        )
        group.add_argument(
            "--ft-timeout-dump-upload-attempts",
            dest="timeout_dump.upload_attempts",
            type=positive_int,
            metavar="N",
            help=f"Attempts per upload batch (default: {defaults.upload_attempts}).",
        )
        group.add_argument(
            "--ft-timeout-dump-shutdown-grace",
            dest="timeout_dump.shutdown_grace",
            type=positive_float,
            metavar="SECONDS",
            help=f"Collector shutdown grace (default: {defaults.shutdown_grace:g}).",
        )
        for tool in TOOLS.values():
            tool.Config.add_arguments(parser.add_argument_group(f"Timeout dump ({tool.name})"))

    @classmethod
    def from_args(cls, args) -> "TimeoutDumpConfig":
        """Settings from the command line, whose values argparse has checked.

        Listing tools enables dumps unless --no-ft-timeout-dump-enabled is also given. Options
        for a tool that is not listed are unused.
        """
        settings = options_given(args, "timeout_dump.", cls)
        tool_names = settings.pop("tools", None)
        settings.setdefault("enabled", tool_names is not None)
        tools = [TOOLS[name].Config.from_args(args) for name in tool_names or TOOLS]
        return cls(tools=tools, **settings)


def diagnostics_directory(launcher_log_file):
    """Use the existing job/array log namespace; never reconstruct Slurm IDs."""
    if not launcher_log_file:
        raise ValueError("Timeout dumps require --ft-nvrx-logfile")
    path = Path(launcher_log_file)
    if not path.is_absolute() or path.parent.name != "logs":
        raise ValueError("Timeout dumps require --ft-nvrx-logfile=<nvrx-root>/logs/<filename>")
    return path.parent.parent / "diags"
