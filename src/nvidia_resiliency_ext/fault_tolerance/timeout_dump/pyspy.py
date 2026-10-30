# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""py-spy: Python stacks of the timed-out training worker."""

import dataclasses
from dataclasses import dataclass
from typing import ClassVar

from .capture import ToolCapture, resolve_executable


class PySpy(ToolCapture):
    @dataclass
    class Config(ToolCapture.Config):
        name: ClassVar[str] = "pyspy"
        executable: str = "py-spy"

        def resolve(self):
            executable = resolve_executable(
                self.executable, tool="pyspy", flag="--ft-timeout-dump-pyspy"
            )
            return dataclasses.replace(self, executable=executable)

        def describe(self):
            return f"pyspy={self.executable}"

        @classmethod
        def add_arguments(cls, group):
            group.add_argument(
                "--ft-timeout-dump-pyspy",
                dest="timeout_dump.pyspy.executable",
                metavar="PATH",
                help=f"py-spy executable (default: {cls.executable}).",
            )

    def command(self):
        return [self.tool_cfg.executable, "dump", "--pid", str(self.worker_pid)]
