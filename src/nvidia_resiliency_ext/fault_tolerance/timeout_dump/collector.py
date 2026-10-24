# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Node-local, single-writer collector that publishes staged captures."""

import contextlib
import fcntl
import logging
import multiprocessing
import os
import random
import re
import shutil
import signal
import socket
import time
from dataclasses import dataclass
from pathlib import Path

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

from .config import staging_dir, validate_config

logger = logging.getLogger(LogConfig.name)
CAPTURE_NAME = re.compile(
    r"rank(?P<local_rank>\d+)_pid\d+_(?P<tool>pyspy|gdb|cuda-gdb)\.txt\.ready\Z"
)


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
