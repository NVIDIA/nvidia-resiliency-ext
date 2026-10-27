# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDB: native CPU state of the timed-out training worker."""

import dataclasses
from dataclasses import dataclass
from typing import ClassVar, Optional

from .capture import ToolCapture, resolve_command_file, resolve_executable


class Gdb(ToolCapture):
    @dataclass
    class Config(ToolCapture.Config):
        name: ClassVar[str] = "gdb"
        executable: str = "gdb"
        # Required when GDB is selected; checked by resolve() at launcher startup.
        command_file: Optional[str] = None

        def resolve(self):
            executable = resolve_executable(
                self.executable, tool="gdb", flag="--ft-timeout-dump-gdb"
            )
            command_file = resolve_command_file(
                self.command_file, tool="gdb", flag="--ft-timeout-dump-gdb-command-file"
            )
            return dataclasses.replace(self, executable=executable, command_file=command_file)

        def describe(self):
            return f"gdb={self.executable} (commands={self.command_file})"

        @classmethod
        def add_arguments(cls, group):
            group.add_argument(
                "--ft-timeout-dump-gdb",
                dest="timeout_dump.gdb.executable",
                metavar="PATH",
                help=f"GDB executable (default: {cls.executable}).",
            )
            group.add_argument(
                "--ft-timeout-dump-gdb-command-file",
                dest="timeout_dump.gdb.command_file",
                metavar="FILE",
                help="GDB command file; required when gdb is selected.",
            )

    def command(self):
        # Disable init scripts; command files are supplied by the deployment.
        command = [self.tool_cfg.executable, "--nx", "--batch"]
        # Disable downloads before attach, including during shared-library loading.
        command.extend(["-iex", "set debuginfod enabled off", "-ex", f"attach {self.worker_pid}"])
        command.extend(["-ex", "set pagination off", "-x", self.tool_cfg.command_file])
        command.extend(["-ex", "detach"])
        return command
