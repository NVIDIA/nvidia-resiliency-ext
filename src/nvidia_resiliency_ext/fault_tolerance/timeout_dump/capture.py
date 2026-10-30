# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded capture of one diagnostic tool against one worker."""

import asyncio
import contextlib
import dataclasses
import os
import shutil
import signal
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar

# Staged capture framing: the header and footer are truncated to these sizes, and
# max_capture_bytes reserves room for both.
HEADER_LIMIT = 768
FOOTER_LIMIT = 256


def staging_dir(cfg):
    return Path(cfg.local_root) / "nvrx-diags"


def options_given(args, prefix, config_class):
    """Fields of `config_class` given on the command line, as options with dest prefix+field."""
    values = {
        field.name: getattr(args, prefix + field.name, None)
        for field in dataclasses.fields(config_class)
    }
    return {name: value for name, value in values.items() if value is not None}


def resolve_executable(executable, *, tool, flag):
    """Return the executable's absolute path; the launcher checks this once at startup."""
    resolved = shutil.which(executable)
    if resolved is None:
        raise ValueError(
            f"Timeout dump tool {tool!r} is missing or not executable: {executable!r}. "
            f"Install it, set {flag}, or adjust --ft-timeout-dump-tools."
        )
    return resolved


def resolve_command_file(path, *, tool, flag):
    """Return the command file's absolute path after checking that it can be read."""
    if not path or not path.strip():
        raise ValueError(
            f"Timeout dump {tool} command file is required: set {flag} to a readable file"
        )
    resolved = Path(path).resolve()
    try:
        if not resolved.is_file():
            raise OSError("not a regular file")
        with resolved.open("rb") as stream:
            stream.read(1)
    except OSError as exc:
        raise ValueError(f"Timeout dump {tool} command file is not readable: {resolved}") from exc
    return str(resolved)


class CaptureStatus(Enum):
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    OUTPUT_LIMIT = "output_limit"
    SUCCESS = "success"
    # Exited 0 without certifying success, e.g. CUDA-GDB stdin mode, which does not
    # necessarily turn command errors into a nonzero exit.
    COMPLETED = "completed"
    EXITED = "exit"  # nonzero exit code
    ERROR = "error"  # could not start or read the tool


class ToolCapture(ABC):
    """One diagnostic tool run against one worker, with its settings in `Config`.

    The base class owns the lifecycle: startup off the event loop, deadlines, the
    output cap, process-group cleanup, and staging. Subclasses supply the command
    line and any tool-specific environment, stdin, and capture details.

    Collection is bounded and in memory; never wait indefinitely for a killed
    child. Output goes to a pipe, not the staging file, so a child stuck after
    SIGKILL cannot continue writing a published .ready file.
    """

    name: ClassVar[str]  # identifies the tool in --ft-timeout-dump-tools and capture files

    @dataclass
    class Config(ABC):
        """A tool's settings; `name` identifies the tool in --ft-timeout-dump-tools."""

        name: ClassVar[str]
        executable: str

        @abstractmethod
        def resolve(self) -> "ToolCapture.Config":
            """A copy with absolute paths, checked once at launcher startup."""

        @abstractmethod
        def describe(self) -> str:
            """One-line summary of the resolved settings for the startup log."""

        @classmethod
        @abstractmethod
        def add_arguments(cls, group) -> None:
            """Add this tool's options to its own argument group, with dest
            ``timeout_dump.<name>.<field>``."""

        @classmethod
        def from_args(cls, args) -> "ToolCapture.Config":
            """This tool's settings from the command line, whose values argparse has checked."""
            return cls(**options_given(args, f"timeout_dump.{cls.name}.", cls))

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Config.name is the one definition.
        cls.name = cls.Config.name

    def __init__(self, cfg, worker_pid, tool_cfg):
        self.cfg = cfg  # TimeoutDumpConfig
        self.tool_cfg = tool_cfg
        self.worker_pid = worker_pid
        self.process = None  # Diagnostic subprocess, not the training worker.
        self.output = bytearray()
        self.status = CaptureStatus.CANCELLED
        self.error = None  # message for CaptureStatus.ERROR
        self._launch_thread = None
        self._launch_error = None
        self._launch_abandoned = threading.Event()

    def is_running(self):
        """Include launches that have not yet returned a subprocess handle."""
        return (self._launch_thread is not None and self._launch_thread.is_alive()) or (
            self.process is not None and self.process.poll() is None
        )

    def _launch(self):
        """Start the tool on the launch thread and hand the result to `run()`.

        This thread only writes `process` or `_launch_error`; `run()` reads them after
        the thread exits. If `run()` stops waiting first, it sets `_launch_abandoned`,
        and this thread then kills and reaps a late child itself. Subclasses customize
        `spawn()`, not this method, so that hand-off stays in one place.
        """
        try:
            self.process = self.spawn()
        except Exception as exc:
            self._launch_error = exc
        finally:
            # Popen can return after run() has timed out or been cancelled.
            # Reap that late child here without holding up the monitor's loop.
            if self._launch_abandoned.is_set() and self.process is not None:
                self.kill()
                self.process.stdout.close()
                self.process.wait()

    def kill(self):
        if self.process is not None:
            # Also kill descendants holding the output pipe after the leader exits.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)

    @abstractmethod
    def command(self) -> list[str]:
        """The tool's argv; it must not touch shared storage (the tool reads its files)."""

    def spawn(self) -> subprocess.Popen:
        """Start the tool on the launch thread; override for tool-specific stdin or env."""
        return self._popen(stdin=subprocess.DEVNULL)

    def _popen(self, **kwargs) -> subprocess.Popen:
        # run() reads the pipe, and kill() signals the new session's process group.
        return subprocess.Popen(
            self.command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            **kwargs,
        )

    def exit_status(self, returncode) -> CaptureStatus:
        return CaptureStatus.SUCCESS if returncode == 0 else CaptureStatus.EXITED

    def status_text(self) -> str:
        """The status as written to the capture footer, e.g. ``timeout`` or ``exit_7``."""
        if self.status is CaptureStatus.EXITED:
            return f"exit_{self.process.returncode}"
        if self.status is CaptureStatus.ERROR:
            return f"error: {self.error}"
        return self.status.value

    def header_details(self) -> str:
        """Extra ``key=value `` fields for the capture header."""
        return ""

    async def run(self, timeout=None):
        """Capture for at most `timeout` seconds, then kill and (boundedly) reap the tool."""
        timeout = self.cfg.capture_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        # Reserve some of this tool's allocation for kill/reap before attaching
        # another debugger. Cleanup also stays within the shared overall deadline.
        capture_deadline = deadline - min(0.1, timeout / 10)
        try:
            if await self._await_launch(capture_deadline):
                await self._drain_output(capture_deadline)
        except OSError as exc:
            self.status = CaptureStatus.ERROR
            self.error = str(exc)
        finally:
            await self._stop(deadline)

    async def _await_launch(self, deadline):
        """Start the tool on a launch thread; False if it is still starting at `deadline`."""
        # Do not use asyncio's default executor: its shutdown waits for
        # threads. A stalled executable lookup must not block monitor exit.
        self._launch_thread = threading.Thread(target=self._launch, daemon=True)
        self._launch_thread.start()
        while self._launch_thread.is_alive():
            if time.monotonic() >= deadline:
                self.status = CaptureStatus.TIMEOUT
                return False
            await asyncio.sleep(0.01)
        if self._launch_error is not None:
            raise self._launch_error
        return True

    async def _drain_output(self, deadline):
        """Read the tool's output until it exits, fills the cap, or `deadline` passes."""
        limit = self.cfg.max_capture_bytes - HEADER_LIMIT - FOOTER_LIMIT
        stdout = self.process.stdout.fileno()
        os.set_blocking(stdout, False)
        while time.monotonic() < deadline:
            try:
                chunk = os.read(stdout, min(65536, limit + 1))
            except BlockingIOError:
                chunk = None
            if chunk:
                remaining = limit - len(self.output)
                self.output.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    self.status = CaptureStatus.OUTPUT_LIMIT
                    return
            elif chunk == b"" and self.process.poll() is not None:
                self.status = self.exit_status(self.process.returncode)
                return
            # Yield so the monitor can service termination timers and IPC events.
            # Delay polling only when no output is available.
            await asyncio.sleep(0 if chunk else 0.01)
        self.status = CaptureStatus.TIMEOUT

    async def _stop(self, deadline):
        """Kill the tool's process group and reap it without waiting past `deadline`."""
        self._launch_abandoned.set()
        self.kill()
        if self.process is not None:
            self.process.stdout.close()
            # WNOHANG polling yields to worker termination timers. If a
            # diagnostic child cannot exit, the monitor skips later tools.
            while self.process.poll() is None and time.monotonic() < deadline:
                await asyncio.sleep(min(0.01, max(0, deadline - time.monotonic())))

    def publish_local(self, rank_info, cycle, reason):
        directory = staging_dir(self.cfg) / f"cycle{cycle}"
        directory.mkdir(mode=0o700, exist_ok=True)
        name = f"rank{rank_info.local_rank}_pid{rank_info.pid}_{self.name}.txt"
        tmp = directory / (name + ".tmp")
        header = (
            f"=== BEGIN host={rank_info.host} cycle={cycle} global_rank={rank_info.global_rank} "
            f"local_rank={rank_info.local_rank} pid={rank_info.pid} tool={self.name} "
            f"{self.header_details()}reason={reason} time={time.time():.6f} ===\n"
        ).encode()[:HEADER_LIMIT]
        returncode = self.process.returncode if self.process is not None else None
        exit_code = f" exit_code={returncode}" if returncode is not None else ""
        footer = f"\n=== END status={self.status_text()[:128]}{exit_code} ===\n"
        footer = footer.encode()[:FOOTER_LIMIT]
        with tmp.open("xb") as stream:
            stream.write(header)
            stream.write(self.output)
            stream.write(footer)
        ready = directory / (name + ".ready")
        os.replace(tmp, ready)
        return ready
