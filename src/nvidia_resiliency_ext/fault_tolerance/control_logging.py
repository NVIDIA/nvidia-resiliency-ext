# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture colocated process logs independently of the gRPC data path."""

import contextlib
import os
import queue
import selectors
import socket
import stat
import sys
import threading
import time


def agent_log_path(nvrx_logfile):
    """Derive the shared node-agent log without changing the control-log path."""
    if not nvrx_logfile:
        return None
    stem = nvrx_logfile[:-4] if nvrx_logfile.endswith(".log") else nvrx_logfile
    return stem + "_agent.log"


class ControlLogCapture:
    """One pipe-draining thread and one file-writing thread, owned by the supervisor.

    File I/O never runs in the supervision loop or pipe reader. The implementation
    limits buffered chunks (up to 16 MiB of payload) and uses original console stderr on overflow or
    write failure, rather than blocking control or growing memory indefinitely.
    This is best-effort process logging, not a durable outcome journal.
    """

    _CHUNK = 8192

    def __init__(self, path, *, buffer_chunks=2048):
        self.path = path
        self.stdout_fd = os.dup(1)
        self.stderr_fd = os.dup(2)
        # Reopen the original stderr as an independent, nonblocking description.
        # Setting O_NONBLOCK on dup(stderr) would also alter Slurm's inherited FD.
        self._fallback_fd = None
        self._fallback_nonblocking = False
        with contextlib.suppress(OSError):
            self._fallback_fd = os.open("/proc/self/fd/2", os.O_WRONLY | os.O_NONBLOCK)
            mode = os.fstat(self._fallback_fd).st_mode
            self._fallback_nonblocking = stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)
        try:
            if path:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                self._file = open(path, "ab", buffering=0)
            else:
                self._file = os.fdopen(os.dup(self.stdout_fd), "wb", buffering=0)
        except BaseException:
            os.close(self.stdout_fd)
            os.close(self.stderr_fd)
            if self._fallback_fd is not None:
                os.close(self._fallback_fd)
            raise
        self._queue = queue.Queue(maxsize=buffer_chunks)
        self._pending = queue.SimpleQueue()
        self._wake_read, self._wake_write = socket.socketpair()
        self._wake_read.setblocking(False)
        self._wake_write.setblocking(False)
        self._stop = threading.Event()
        self._deadline = float("inf")
        self._redirected = False
        self._closed = False
        self._fallback_reported = False
        self._reader = threading.Thread(target=self._read, name="ControlLogCapture", daemon=True)
        self._writer = threading.Thread(target=self._write, name="ControlLogWriter", daemon=True)
        self._writer.start()
        self._reader.start()

    def _fallback(self, data):
        # Use the original stream, never our own captured stderr (which would recurse).
        # If the console is also blocked, drop the chunk: logging cannot stop control.
        if self._fallback_fd is None:
            return
        # O_NONBLOCK does not protect regular-file I/O (e.g. Slurm stderr on
        # Lustre). Only the writer may attempt that; readers drop overflow there.
        if not self._fallback_nonblocking and threading.current_thread() is not self._writer:
            return
        with contextlib.suppress(OSError):
            if not self._fallback_reported:
                os.write(
                    self._fallback_fd, b"NVRx control logging unavailable; using original stderr.\n"
                )
                self._fallback_reported = True
            remaining = memoryview(data)
            while remaining:
                size = os.write(self._fallback_fd, remaining)
                if size == 0:
                    break
                remaining = remaining[size:]

    def _enqueue(self, data):
        for start in range(0, len(data), self._CHUNK):
            part = data[start : start + self._CHUNK]
            try:
                self._queue.put_nowait(part)
            except queue.Full:
                self._fallback(part)

    def attach(self, stream):
        if self._closed:
            raise RuntimeError("control log capture is closed")
        self._pending.put(stream)
        with contextlib.suppress(BlockingIOError):
            self._wake_write.send(b"\0")

    def redirect_supervisor(self):
        """Include Python logging and raw stdout/stderr from the supervisor."""
        sys.stdout.flush()
        sys.stderr.flush()
        read_fd, write_fd = os.pipe()
        try:
            self.attach(os.fdopen(read_fd, "rb", buffering=0))
            os.dup2(write_fd, 1)
            os.dup2(write_fd, 2)
        finally:
            os.close(write_fd)
        self._redirected = True

    def _read(self):
        streams = {}
        with selectors.DefaultSelector() as selector:
            selector.register(self._wake_read, selectors.EVENT_READ, None)
            try:
                while True:
                    while not self._pending.empty():
                        stream = self._pending.get()
                        os.set_blocking(stream.fileno(), False)
                        streams[stream.fileno()] = [stream, b""]
                        selector.register(stream, selectors.EVENT_READ, stream.fileno())
                    if self._stop.is_set() and (not streams or time.monotonic() >= self._deadline):
                        break
                    wait = min(0.1, max(0.0, self._deadline - time.monotonic()))
                    for key, _ in selector.select(timeout=wait):
                        if key.data is None:
                            with contextlib.suppress(BlockingIOError):
                                self._wake_read.recv(4096)
                            continue
                        stream, tail = streams[key.data]
                        try:
                            data = os.read(key.data, self._CHUNK)
                        except BlockingIOError:
                            continue
                        if not data:
                            self._enqueue(tail)
                            selector.unregister(stream)
                            del streams[key.data]
                            stream.close()
                            continue
                        lines = (tail + data).split(b"\n")
                        for line in lines[:-1]:
                            self._enqueue(line + b"\n")
                        tail = lines[-1]
                        if len(tail) >= 65536:
                            self._enqueue(tail)
                            tail = b""
                        streams[key.data][1] = tail
            finally:
                for stream, tail in streams.values():
                    self._enqueue(tail)
                    stream.close()
                self._wake_read.close()
                self._wake_write.close()
                # Signal completion only after the last tail has been enqueued.
                # A stalled writer must not keep the reader alive at shutdown.
                while True:
                    try:
                        self._queue.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        if time.monotonic() >= self._deadline:
                            with contextlib.suppress(queue.Empty):
                                self._fallback(self._queue.get_nowait())

    def _write(self):
        failed = False
        try:
            while True:
                data = self._queue.get()
                if data is None:
                    break
                if failed:
                    self._fallback(data)
                else:
                    try:
                        remaining = memoryview(data)
                        while remaining:
                            size = self._file.write(remaining)
                            if not size:
                                raise OSError("control log write made no progress")
                            remaining = remaining[size:]
                    except OSError:
                        failed = True
                        self._fallback(data)
        finally:
            with contextlib.suppress(OSError):
                self._file.close()

    def close(self, timeout=10.0):
        if self._closed:
            return
        self._closed = True
        if self._redirected:
            with contextlib.suppress(OSError, ValueError):
                sys.stdout.flush()
                sys.stderr.flush()
            os.dup2(self.stdout_fd, 1)
            os.dup2(self.stderr_fd, 2)
        finish_by = time.monotonic() + timeout
        # Reserve some of the shutdown budget for flushing final partial lines.
        self._deadline = time.monotonic() + timeout * 0.8
        self._stop.set()
        self._reader.join(timeout)
        self._writer.join(max(0.0, finish_by - time.monotonic()))
        if self._writer.is_alive():
            self._fallback(b"Control log writer did not drain before shutdown deadline.\n")
        # A stalled daemon writer may still need its fallback FD. Avoid closing/reusing
        # it underneath that thread; the OS will reclaim it when this process exits.
        if not self._reader.is_alive() and not self._writer.is_alive():
            os.close(self.stdout_fd)
            os.close(self.stderr_fd)
            if self._fallback_fd is not None:
                os.close(self._fallback_fd)
