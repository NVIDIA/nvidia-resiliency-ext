# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-GDB: CPU and CUDA GPU state of the timed-out training worker."""

import contextlib
import dataclasses
import enum
import os
import subprocess
from dataclasses import dataclass
from typing import ClassVar, Optional

from .capture import CaptureStatus, ToolCapture, resolve_command_file, resolve_executable


class Mode(enum.Enum):
    """How CUDA-GDB receives its commands."""

    BATCH = "batch"  # --batch with -ex/-x
    STDIN = "stdin"  # fed through the normal command reader


class CudaGdb(ToolCapture):
    @dataclass
    class Config(ToolCapture.Config):
        name: ClassVar[str] = "cuda-gdb"
        executable: str = "cuda-gdb"
        # Required when CUDA-GDB is selected; checked by resolve() at launcher startup.
        command_file: Optional[str] = None
        mode: Mode = Mode.BATCH

        def resolve(self):
            executable = resolve_executable(
                self.executable, tool="cuda-gdb", flag="--ft-timeout-dump-cuda-gdb"
            )
            command_file = resolve_command_file(
                self.command_file, tool="cuda-gdb", flag="--ft-timeout-dump-cuda-gdb-command-file"
            )
            # In stdin mode the path follows "source" on one command line. Check both the
            # configured path and its symlink target.
            if self.mode is Mode.STDIN and any(
                "\n" in path or "\r" in path for path in (self.command_file, command_file)
            ):
                raise ValueError(
                    "Timeout dump cuda-gdb stdin command file path must be a single line"
                )
            return dataclasses.replace(self, executable=executable, command_file=command_file)

        def describe(self):
            return (
                f"cuda-gdb={self.executable} (mode={self.mode.value}, commands={self.command_file})"
            )

        @classmethod
        def add_arguments(cls, group):
            group.add_argument(
                "--ft-timeout-dump-cuda-gdb",
                dest="timeout_dump.cuda-gdb.executable",
                metavar="PATH",
                help=f"CUDA-GDB executable (default: {cls.executable}).",
            )
            group.add_argument(
                "--ft-timeout-dump-cuda-gdb-command-file",
                dest="timeout_dump.cuda-gdb.command_file",
                metavar="FILE",
                help="CUDA-GDB command file; required when cuda-gdb is selected.",
            )
            group.add_argument(
                "--ft-timeout-dump-cuda-gdb-mode",
                dest="timeout_dump.cuda-gdb.mode",
                type=Mode,
                metavar="{" + ",".join(mode.value for mode in Mode) + "}",
                help=f"CUDA-GDB command delivery (default: {cls.mode.value}).",
            )

    def command(self):
        if self.tool_cfg.mode is Mode.STDIN:
            return [self.tool_cfg.executable, "--nx", "-q", "-iex", "set debuginfod enabled off"]
        # Disable init scripts; command files are supplied by the deployment.
        command = [self.tool_cfg.executable, "--nx", "--batch"]
        # Disable downloads before attach, including during shared-library loading.
        command.extend(["-iex", "set debuginfod enabled off", "-ex", f"attach {self.worker_pid}"])
        command.extend(["-ex", "set pagination off", "-x", self.tool_cfg.command_file])
        command.extend(["-ex", "detach"])
        return command

    def spawn(self):
        with contextlib.ExitStack() as stack:
            stdin = subprocess.DEVNULL
            if self.tool_cfg.mode is Mode.STDIN:
                # Prepare all commands on the launch thread. A regular in-memory file
                # avoids pipe writes or prompt handling on the monitor event loop; the
                # child owns its fd 0 after exec.
                stdin = stack.enter_context(
                    os.fdopen(os.memfd_create("nvrx-cuda-gdb-stdin"), "w+b")
                )
                stdin.write(os.fsencode(self._stdin_script()))
                stdin.seek(0)
            return self._popen(stdin=stdin, env=self._worker_environment())

    def _stdin_script(self):
        # GDB's source command consumes the rest of the line as the filename,
        # including spaces and quotes. Shell quoting would alter the filename.
        # Keep attach at top level so normal command-loop events can be serviced.
        return (
            "set confirm off\n"
            "set pagination off\n"
            "set width 0\n"
            "set height 0\n"
            f"attach {self.worker_pid}\n"
            f"source {self.tool_cfg.command_file}\n"
            "detach\n"
            "quit\n"
        )

    def _worker_environment(self):
        env = os.environ.copy()
        # The monitor may have a different CUDA_VISIBLE_DEVICES or TMPDIR.
        # Read only these settings, with a bounded read; never log the worker environment.
        with open(f"/proc/{self.worker_pid}/environ", "rb") as stream:
            entries = stream.read(65537)
        if len(entries) > 65536:
            raise OSError("Worker environment exceeds diagnostic read limit")
        values = dict(entry.split(b"=", 1) for entry in entries.split(b"\0") if b"=" in entry)
        for name in ("TMPDIR", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
            if name.encode() in values:
                env[name] = os.fsdecode(values[name.encode()])
            else:
                env.pop(name, None)
        return env

    def exit_status(self, returncode):
        # Stdin command errors do not necessarily set a nonzero exit code.
        # Completion does not certify useful GPU state.
        if returncode == 0 and self.tool_cfg.mode is Mode.STDIN:
            return CaptureStatus.COMPLETED
        return super().exit_status(returncode)

    def header_details(self):
        return f"mode={self.tool_cfg.mode.value} "
