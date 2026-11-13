# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Polling, refresh signals, and parent lifetime for an isolated publisher worker."""

from __future__ import annotations

import logging
import os
import select
import signal
import socket
import threading
from contextlib import contextmanager

from .publisher import SchedulerExclusionPublisher, positive_integer

logger = logging.getLogger(__name__)


@contextmanager
def parent_lifetime(parent_pid: int | None):
    """Stop this worker if its configured parent exits."""
    stopped = threading.Event()

    def watch_parent():
        while not stopped.is_set():
            if os.getppid() != parent_pid:
                os.kill(os.getpid(), signal.SIGTERM)
                return
            stopped.wait(1.0)

    if parent_pid is not None:
        threading.Thread(target=watch_parent, name="publisher-parent-watch", daemon=True).start()
    try:
        yield
    finally:
        stopped.set()


def serve(publisher: SchedulerExclusionPublisher, interval: int) -> None:
    """Poll immediately and periodically; SIGUSR1 requests another poll."""
    positive_integer("SEGMENT_HEALTH_POLL_INTERVAL", str(interval))
    pidfile = publisher.config.directory / f"segment_health_producer.{publisher.config.job_id}.pid"
    pid = str(os.getpid())

    def refresh(signum, frame):
        # set_wakeup_fd interrupts the wait without taking locks in a signal handler.
        pass

    def stop(signum, frame):
        # Unwind subprocess.run too, so a blocked Slurm child is killed and reaped.
        raise SystemExit(0)

    handlers = {signal.SIGUSR1: refresh, signal.SIGTERM: stop, signal.SIGINT: stop}
    previous = {sig: signal.getsignal(sig) for sig in handlers}
    reader, writer = socket.socketpair()
    with reader, writer:
        reader.setblocking(False)
        writer.setblocking(False)
        previous_wakeup = signal.set_wakeup_fd(writer.fileno(), warn_on_full_buffer=False)
        try:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            try:
                pidfile.write_text(f"{socket.gethostname()} {pid}\n", encoding="utf-8")
            except OSError:
                logger.warning("Could not write pidfile %s", pidfile)
            logger.info(
                "Serving array=%s interval=%ss (SIGUSR1 = poll now)",
                publisher.config.job_id,
                interval,
            )
            while True:
                try:
                    reader.recv(65536)
                except BlockingIOError:
                    pass
                if not publisher.poll_once():
                    logger.warning("Poll failed; continuing periodic refresh")
                # Requests during a poll leave a byte queued, making this wait immediate.
                select.select([reader], [], [], interval)
        finally:
            try:
                fields = pidfile.read_text(encoding="utf-8").split()
                if len(fields) >= 2 and fields[1] == pid:
                    pidfile.unlink()
            except (OSError, UnicodeError):
                pass
            signal.set_wakeup_fd(previous_wakeup)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
