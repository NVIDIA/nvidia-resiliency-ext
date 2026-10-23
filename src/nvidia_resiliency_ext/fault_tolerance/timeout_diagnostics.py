# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Best-effort timeout captures and a node-local, single-writer collector.

Capture output is staged locally; only the collector publishes it to shared
storage. Debugger command files may be read from shared storage by the tools.
Completed captures are the queue; no socket protocol or metadata sidecar is required.
"""

import asyncio
import contextlib
import ctypes
import fcntl
import logging
import math
import multiprocessing
import os
import random
import re
import shutil
import signal
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

logger = logging.getLogger(LogConfig.name)
CAPTURE_NAME = re.compile(
    r"rank(?P<local_rank>\d+)_pid\d+_(?P<tool>pyspy|gdb|cuda-gdb)\.txt\.ready\Z"
)


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


def authorize_monitor_ptrace(sock):
    """Allow this socket's monitor, and its descendants, under Linux Yama mode 1."""
    pid, uid, _ = struct.unpack(
        "iII", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("iII"))
    )
    if pid <= 0 or uid != os.geteuid():
        raise PermissionError("rank monitor has an unexpected PID or UID")
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_PTRACER changes this worker's exception, not the node's ptrace policy.
    if libc.prctl(0x59616D61, ctypes.c_ulong(pid), 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def staging_dir(cfg):
    return Path(cfg.timeout_dump_local_root) / "nvrx-diags"


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


@dataclass
class UploadBatch:
    paths: tuple
    attempts: int = 0
    published: bool = False
    exhausted: bool = False


class DiagnosticCollector:
    def __init__(self, cfg, destination, node_name=None, *, local_world_size=1):
        self.cfg = cfg
        self.local_world_size = local_world_size
        self.node_name = node_name or socket.gethostname()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.node_name):
            raise ValueError("invalid diagnostic node name")
        self.root = staging_dir(cfg)
        self.destination = Path(destination)
        self.batches = {}
        self.exhausted_paths = set()

    def discover(self):
        groups = {}
        for directory in self.root.glob("cycle*"):
            if not re.fullmatch(r"cycle\d+", directory.name) or directory.is_symlink():
                continue
            for path in directory.glob("*.ready"):
                match = CAPTURE_NAME.fullmatch(path.name)
                if match and path not in self.exhausted_paths and not path.is_symlink():
                    groups.setdefault((directory.name, match["tool"]), []).append(path)
        return {key: tuple(sorted(paths)) for key, paths in groups.items()}

    def publish(self, key, batch):
        cycle, tool = key
        directory = self.destination / cycle
        directory.mkdir(parents=True, exist_ok=True)
        final = directory / f"{self.node_name}_{tool}.txt"
        partial = directory / f"{self.node_name}_{tool}.txt.partial"
        if not batch.published:
            # Every retry rewrites the whole snapshot. Never append to final.
            with partial.open("wb") as target:
                try:
                    previous = final.open("rb")
                except FileNotFoundError:
                    previous = None
                if previous is not None:
                    with previous:
                        shutil.copyfileobj(previous, target, length=65536)
                for path in batch.paths:
                    with path.open("rb") as source:
                        shutil.copyfileobj(source, target, length=65536)
                target.flush()
                os.fsync(target.fileno())
            os.replace(partial, final)
            # If directory sync or local cleanup fails, retry that phase only.
            batch.published = True
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        for path in batch.paths:
            path.unlink(missing_ok=True)
        logger.info("Uploaded timeout diagnostics: %s (%d captures)", final, len(batch.paths))

    def tick(self, *, flush=False):
        # Defer a newly discovered incomplete group once. On the next poll,
        # refresh its inputs and publish even if some ranks are still missing.
        deferred = set()
        for key, paths in self.discover().items():
            if key not in self.batches or self.batches[key].exhausted:
                self.batches[key] = UploadBatch(paths)
                ranks = {int(CAPTURE_NAME.fullmatch(path.name)["local_rank"]) for path in paths}
                if not flush and len(ranks) < self.local_world_size:
                    deferred.add(key)
            elif self.batches[key].attempts == 0:
                self.batches[key].paths = paths
            # After the first upload attempt, preserve fixed inputs for retries;
            # new arrivals form a later batch after it succeeds or is exhausted.
        for key, batch in list(self.batches.items()):
            if batch.exhausted or key in deferred:
                continue
            batch.attempts += 1
            try:
                self.publish(key, batch)
            except OSError as exc:
                batch.exhausted = batch.attempts >= self.cfg.timeout_dump_upload_attempts
                if batch.exhausted:
                    # Retain failed evidence without consuming the retry budget
                    # of captures that arrive later for the same cycle/tool.
                    self.exhausted_paths.update(batch.paths)
                logger.warning(
                    "Diagnostic upload %s attempt %d failed%s: %s",
                    key,
                    batch.attempts,
                    " (exhausted; local files retained)" if batch.exhausted else "",
                    exc,
                )
            else:
                del self.batches[key]
                # Keep empty cycle directories: a producer may have completed
                # mkdir() and be about to open its temporary capture file.
            # Uploads are serial, with at most one attempt per batch per poll.


def _collector_main(cfg, destination, lock_fd, inherited_fds, launcher_pid, local_world_size=1):
    os.setsid()  # separate from worker process groups and their cleanup
    for fd in inherited_fds:
        if fd > 2 and fd != lock_fd:
            with contextlib.suppress(OSError):
                os.close(fd)
    # Keep stderr: it is the launcher's existing log aggregation pipe.
    from nvidia_resiliency_ext.shared_utils.log_manager import setup_logger

    setup_logger(force_reset=True, node_local_tmp_prefix="diags")
    stopping = False
    sleeping = False

    class WakeForShutdown(Exception):
        pass

    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True
        if sleeping:
            # Python restarts time.sleep after a handler that returns normally.
            # Interrupt only the idle wait, never an in-progress publication.
            raise WakeForShutdown

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    collector = DiagnosticCollector(cfg, destination, local_world_size=local_world_size)

    def wait_for_poll(delay):
        nonlocal sleeping
        try:
            sleeping = True
            time.sleep(delay)
        except WakeForShutdown:
            pass
        finally:
            sleeping = False

    try:
        # Desynchronize nodes once at startup. Subsequent collection and retry
        # polls use the same interval; there is no separate upload jitter knob.
        wait_for_poll(random.uniform(0, cfg.timeout_dump_poll_interval))
        while not stopping and os.getppid() == launcher_pid:
            collector.tick()
            if not stopping:
                wait_for_poll(cfg.timeout_dump_poll_interval)
        # Shutdown bypasses polling and incomplete-group deferral for one final
        # pass. The launcher independently enforces the shutdown grace period.
        collector.tick(flush=True)
    finally:
        os.close(lock_fd)


class DiagnosticCollectorProcess:
    """Launcher-owned fork child. No queues, events, or additional sockets."""

    def __init__(self, cfg, destination, *, local_world_size=1):
        validate_config(cfg)
        root = staging_dir(cfg)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.is_symlink() or root.stat().st_uid != os.geteuid():
            raise PermissionError(f"Diagnostic staging directory is not owned by this user: {root}")
        lock_fd = os.open(root / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Never upload stale files into a new job or delete unpublished evidence.
            if any(root.glob("cycle*/*")):
                raise RuntimeError(
                    f"Stale captures in {root}; collect them or choose another local root"
                )
            inherited = []
            for entry in os.listdir("/proc/self/fd"):
                fd = int(entry)
                try:
                    os.fstat(fd)
                except OSError:
                    continue  # directory FD used by listdir has already closed
                inherited.append(fd)
            self.cfg = cfg
            self._reported_exit = False
            self.process = multiprocessing.get_context("fork").Process(
                target=_collector_main,
                args=(cfg, destination, lock_fd, inherited, os.getpid(), local_world_size),
                name="nvrx-diag-collector",
                daemon=True,
            )
            self.process.start()
        finally:
            os.close(lock_fd)  # the child retains ownership, even if its launcher exits

    def check_health(self):
        if not self.process.is_alive() and not self._reported_exit:
            self._reported_exit = True
            logger.warning(
                "Diagnostic collector exited with code %s; unpublished files remain in %s. "
                "The collector will not be automatically restarted.",
                self.process.exitcode,
                staging_dir(self.cfg),
            )

    def shutdown(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=self.cfg.timeout_dump_shutdown_grace)
        if self.process.is_alive():
            logger.warning("Diagnostic collector exceeded shutdown grace; sending SIGKILL")
            self.process.kill()
            self.process.join(timeout=0.1)
        if self.process.is_alive():
            # multiprocessing's interpreter-exit handler otherwise joins even
            # daemon children without a deadline. An uninterruptible filesystem
            # wait must not hang launcher exit. Do not replace this live writer.
            multiprocessing.process._children.discard(self.process)
        else:
            self.process.join(timeout=0)
            self.process.close()
