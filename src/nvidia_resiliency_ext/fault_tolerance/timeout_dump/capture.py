# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded capture of one diagnostic tool against one worker."""

import asyncio
import contextlib
import os
import signal
import subprocess
import threading
import time

from .config import debugger_command_file, staging_dir, tool_executable


class ToolCapture:
    """Bounded in-memory collection; never wait indefinitely for a killed child.

    Output goes to a pipe, not the staging file. A child stuck after SIGKILL
    therefore cannot continue writing a published .ready file.
    """

    def __init__(self, cfg, worker_pid, tool):
        self.cfg = cfg
        self.worker_pid = worker_pid
        self.tool = tool
        self.process = None  # Diagnostic subprocess, not the training worker.
        self.output = bytearray()
        self.status = "cancelled"
        self._launch_thread = None
        self._launch_error = None
        self._launch_abandoned = threading.Event()

    def is_running(self):
        """Include launches that have not yet returned a subprocess handle."""
        return (self._launch_thread is not None and self._launch_thread.is_alive()) or (
            self.process is not None and self.process.poll() is None
        )

    def _launch(self):
        try:
            with contextlib.ExitStack() as stack:
                stdin = subprocess.DEVNULL
                if self._uses_stdin():
                    # Prepare all commands on this startup thread. A regular
                    # in-memory file avoids pipe writes or prompt handling on
                    # the monitor event loop; the child owns its fd 0 after exec.
                    stdin = stack.enter_context(
                        os.fdopen(os.memfd_create("nvrx-cuda-gdb-stdin"), "w+b")
                    )
                    stdin.write(os.fsencode(self.stdin_commands()))
                    stdin.seek(0)
                self.process = subprocess.Popen(
                    self.command(),
                    stdin=stdin,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                    env=self.environment(),
                )
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

    def command(self):
        executable = tool_executable(self.cfg, self.tool)
        if self.tool == "pyspy":
            return [executable, "dump", "--pid", str(self.worker_pid)]
        if self._uses_stdin():
            return [executable, "--nx", "-q", "-iex", "set debuginfod enabled off"]
        # Disable init scripts; command files are supplied by the deployment.
        command = [executable, "--nx", "--batch"]
        # Disable downloads before attach, including during shared-library loading.
        command.extend(["-iex", "set debuginfod enabled off", "-ex", f"attach {self.worker_pid}"])
        command.extend(
            ["-ex", "set pagination off", "-x", str(debugger_command_file(self.cfg, self.tool))]
        )
        command.extend(["-ex", "detach"])
        return command

    def _uses_stdin(self):
        return self.tool == "cuda-gdb" and self.cfg.timeout_dump_cuda_gdb_mode == "stdin"

    def stdin_commands(self):
        # GDB's source command consumes the rest of the line as the filename,
        # including spaces and quotes. Shell quoting would alter the filename.
        # Keep attach at top level so normal command-loop events can be serviced.
        return (
            "set confirm off\n"
            "set pagination off\n"
            "set width 0\n"
            "set height 0\n"
            f"attach {self.worker_pid}\n"
            f"source {debugger_command_file(self.cfg, self.tool)}\n"
            "detach\n"
            "quit\n"
        )

    def environment(self):
        env = os.environ.copy()
        if self.tool == "cuda-gdb":
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

    async def run(self, timeout=None):
        timeout = self.cfg.timeout_dump_capture_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        # Reserve some of this tool's allocation for kill/reap before attaching
        # another debugger. Cleanup also stays within the shared overall deadline.
        capture_deadline = deadline - min(0.1, timeout / 10)
        limit = self.cfg.timeout_dump_max_capture_bytes - 1024  # header/footer allowance
        try:
            # Do not use asyncio's default executor: its shutdown waits for
            # threads. A stalled executable lookup must not block monitor exit.
            self._launch_thread = threading.Thread(target=self._launch, daemon=True)
            self._launch_thread.start()
            while self._launch_thread.is_alive():
                if time.monotonic() >= capture_deadline:
                    self.status = "timeout"
                    return self
                await asyncio.sleep(0.01)
            if self._launch_error is not None:
                raise self._launch_error
            os.set_blocking(self.process.stdout.fileno(), False)
            while True:
                if time.monotonic() >= capture_deadline:
                    self.status = "timeout"
                    break
                try:
                    chunk = os.read(self.process.stdout.fileno(), min(65536, limit + 1))
                except BlockingIOError:
                    chunk = None
                if chunk:
                    remaining = limit - len(self.output)
                    self.output.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        self.status = "output_limit"
                        break
                elif chunk == b"" and self.process.poll() is not None:
                    if self.process.returncode == 0:
                        # Stdin command errors do not necessarily set a nonzero
                        # exit code. Completion does not certify useful GPU state.
                        self.status = "completed" if self._uses_stdin() else "success"
                    else:
                        self.status = f"exit_{self.process.returncode}"
                    break
                # Yield so the monitor can service termination timers and IPC events.
                # Delay polling only when no output is available.
                await asyncio.sleep(0 if chunk else 0.01)
        except OSError as exc:
            self.status = f"error: {exc}"
        finally:
            self._launch_abandoned.set()
            self.kill()
            if self.process is not None:
                self.process.stdout.close()
                # WNOHANG polling yields to worker termination timers. If a
                # diagnostic child cannot exit, the monitor skips later tools.
                while self.process.poll() is None and time.monotonic() < deadline:
                    await asyncio.sleep(min(0.01, max(0, deadline - time.monotonic())))
        return self

    def publish_local(self, rank_info, cycle, reason):
        directory = staging_dir(self.cfg) / f"cycle{cycle}"
        directory.mkdir(mode=0o700, exist_ok=True)
        name = f"rank{rank_info.local_rank}_pid{rank_info.pid}_{self.tool}.txt"
        tmp = directory / (name + ".tmp")
        mode = f"mode={self.cfg.timeout_dump_cuda_gdb_mode} " if self.tool == "cuda-gdb" else ""
        header = (
            f"=== BEGIN host={rank_info.host} cycle={cycle} global_rank={rank_info.global_rank} "
            f"local_rank={rank_info.local_rank} pid={rank_info.pid} tool={self.tool} "
            f"{mode}reason={reason} time={time.time():.6f} ===\n"
        ).encode()[:768]
        exit_code = (
            f" exit_code={self.process.returncode}"
            if self.tool == "cuda-gdb" and self.process is not None
            else ""
        )
        footer = f"\n=== END status={self.status[:128]}{exit_code} ===\n".encode()[:256]
        with tmp.open("xb") as stream:
            stream.write(header)
            stream.write(self.output)
            stream.write(footer)
        ready = directory / (name + ".ready")
        os.replace(tmp, ready)
        return ready
