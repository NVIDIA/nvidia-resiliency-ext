#!/usr/bin/env python3

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Custom LogsSpecs implementation for consolidated per-cycle logging.

This module provides PipeBasedLogsSpecs:
   - Uses pipes (like srun --output) to prevent buffer loss
   - Parent ft_launcher reads from pipes and writes to file
   - Line-buffered pipes ensure data isn't lost on worker termination (SIGTERM)
   - Rank prefixes added in parent (works for C++ logs too)
   - Solves the stack trace loss problem

Architecture:
   Worker → pipe (line-buffered by kernel) → parent reader thread → file

Why not use PyTorch's built-in redirect?
--------------------------------------
PyTorch's torch.distributed.elastic.multiprocessing.redirects.redirect() opens files with
mode="w+b" which truncates the file. This is fine for per-rank files, but causes data loss
when multiple ranks write to the same file.
"""

import contextlib
import errno
import logging
import os
import queue
import random
import re
import select
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

# gRPC imports for log aggregation
import grpc
from torch.distributed.elastic.multiprocessing import LogsDest, LogsSpecs, Std
from torch.distributed.elastic.multiprocessing.subprocess_handler import SubprocessHandler

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig
from nvidia_resiliency_ext.shared_utils.proto import log_aggregation_pb2, log_aggregation_pb2_grpc

# Special marker string to signal pipe-based logging
# We can't use subprocess.PIPE directly because PyTorch expects strings
_PIPE_MARKER = "__SUBPROCESS_PIPE__"

# Writer thread batch size and flush interval
# Larger batches improve Lustre performance (reduces metadata operations)
# The reader thread drains 64KB pipes quickly into queue, then writer batches from queue
_WRITE_BATCH_SIZE = 256 * 1024  # 256 KB batches
_FLUSH_INTERVAL_SECONDS = 1.0  # Flush at least every 1 second

# gRPC configuration for distributed log aggregation
_GRPC_MAX_CHUNK_SIZE = 256 * 1024  # 256 KB chunks for gRPC transmission
_GRPC_RECONNECT_INTERVAL = 5.0  # Retry connection every 5 seconds

# Module-level flag to track if we've patched get_subprocess_handler
_SUBPROCESS_HANDLER_PATCHED = False

# Pre-compiled regex and constants for log filtering (module-level for performance)
# These patterns match Nvidia driver /proc/devices dumps that appear on process exit
_DEVICE_SECTION_HEADERS = frozenset(["Character devices:", "Block devices:"])
# Pattern matches: "252 device-mapper", "1 mem", "195 nvidia" (after stripping)
_DEVICE_ENTRY_PATTERN = re.compile(r'^\d{1,3}\s+[a-zA-Z0-9_\-/]+$')


def _should_filter_line(line: str) -> bool:
    """
    Filter out noisy lines that clutter logs but provide no useful information.

    Currently filters:
    - Nvidia driver /proc/devices dumps on process exit (Character/Block devices listings)

    Optimized for minimal overhead:
    - Pre-compiled regex at module level
    - Fast early-exit checks before expensive operations
    - Defers strip() until actually needed (avoids string allocation on fast path)

    Args:
        line: Log line to check (without rank prefix)

    Returns:
        True if line should be filtered out (dropped), False if it should be kept
    """
    # Fast path #1: Most log lines are much longer than device dumps
    # Device dump lines are typically < 50 chars: "NNN device-name" or "Character devices:"
    # Check original line length (no strip needed yet)
    # ~99% of lines exit here in typical workloads
    if len(line) > 65:  # Extra buffer for trailing newlines/spaces
        return False

    # Fast path #2: Empty lines - keep them (they're intentional formatting)
    if not line:
        return False

    # Now we know it's a short line that's not empty - strip for pattern matching
    # Only ~1% of lines reach this point in typical workloads
    stripped = line.strip()

    # Whitespace-only lines (e.g., "   \n", "\t\n") - keep them too
    # These happen when workers output blank lines for readability
    if not stripped:
        return False

    # Check for section headers (very fast frozenset lookup)
    if stripped in _DEVICE_SECTION_HEADERS:
        return True

    # Check for device number entries: "NNN device-name"
    # Pattern: 1-3 digits, space(s), device name
    # Examples: "252 device-mapper", "1 mem", "195 nvidia"
    # After stripping, device entries always start with a digit (fast pre-check before regex)
    first_char = stripped[0]
    if first_char.isdigit() and _DEVICE_ENTRY_PATTERN.match(stripped):
        return True

    return False


def _patch_subprocess_handler_once():
    """
    Monkey-patch get_subprocess_handler to use PipeSubprocessHandler.

    We need to patch it in the api module where it's actually used, not just in
    the handlers module where it's defined, because api.py has its own import reference.

    This is a module-level operation that happens once. After patching,
    all subsequent subprocess handler creation will use PipeSubprocessHandler.

    This is safe because:
    1. ft_launcher only uses one LogsSpecs type at a time
    2. Once using PipeBasedLogsSpecs, it stays that way for the process lifetime
    3. The patch should persist - no need to unpatch
    """
    global _SUBPROCESS_HANDLER_PATCHED

    if _SUBPROCESS_HANDLER_PATCHED:
        return  # Already patched

    # Import the api module where get_subprocess_handler is actually called
    from torch.distributed.elastic.multiprocessing import api

    # Create patched version that returns our custom handler
    def patched_get_subprocess_handler(**kwargs):
        return PipeSubprocessHandler(**kwargs)

    # Patch it in the api module where SubprocessContext uses it
    api.get_subprocess_handler = patched_get_subprocess_handler
    _SUBPROCESS_HANDLER_PATCHED = True


class PipeSubprocessHandler(SubprocessHandler):
    """
    Custom SubprocessHandler that supports pipe-based logging via a special marker string.

    Extends PyTorch's SubprocessHandler to detect our _PIPE_MARKER string and convert it
    to subprocess.PIPE. We can't pass subprocess.PIPE directly because PyTorch's LogsDest
    type signature expects strings (file paths), not integers.

    Merges stderr into stdout pipe (like default srun --output= behavior where both
    streams go to the same file).
    """

    def __init__(
        self,
        entrypoint: str,
        args: tuple,
        env: dict[str, str],
        stdout: Optional[str],  # Can be path, _PIPE_MARKER, or None
        stderr: Optional[str],  # Can be path, _PIPE_MARKER, or None
        local_rank_id: int,
        numa_options: Optional[Any] = None,  # Accept but ignore NUMA options
    ):
        # Override parent to handle our pipe marker
        # Don't call super().__init__() because it would try to open() the marker
        # Note: numa_options is accepted for compatibility but not used with pipe-based logging

        # Handle stdout
        if stdout == _PIPE_MARKER:
            self._stdout = subprocess.PIPE
        elif stdout:
            self._stdout = open(stdout, "w")
        else:
            self._stdout = None

        # Handle stderr - merge into stdout if both are pipes (like srun default)
        if stderr == _PIPE_MARKER and stdout == _PIPE_MARKER:
            # Merge stderr into stdout pipe (like srun --output= default behavior)
            self._stderr = subprocess.STDOUT
        elif stderr == _PIPE_MARKER:
            self._stderr = subprocess.PIPE
        elif stderr:
            self._stderr = open(stderr, "w")
        else:
            self._stderr = None

        # Inherit parent environment vars
        env_vars = os.environ.copy()
        env_vars.update(env)

        args_str = (entrypoint, *[str(e) for e in args])
        self.local_rank_id = local_rank_id
        self.proc: subprocess.Popen = self._popen(args_str, env_vars)

    def close(self, death_sig: Optional[Any] = None) -> None:
        """Override close to handle subprocess.PIPE properly."""
        # Import here to avoid circular dependency
        import signal

        if not death_sig:
            death_sig = signal.SIGTERM

        # Send signal to process group
        try:
            os.killpg(self.proc.pid, death_sig)
        except (OSError, AttributeError):
            # Fallback if killpg fails or on Windows
            try:
                self.proc.send_signal(death_sig)
            except OSError:
                pass

        # Close file handles for non-pipe cases (when we opened actual files)
        if self._stdout and self._stdout not in (subprocess.PIPE, subprocess.STDOUT):
            try:
                self._stdout.close()
            except Exception:
                pass

        if self._stderr and self._stderr not in (subprocess.PIPE, subprocess.STDOUT):
            try:
                self._stderr.close()
            except Exception:
                pass

        # For pipe cases, close the parent's read-end of the pipes
        # This is defense-in-depth: PyTorch calls this for alive processes,
        # but _stop_workers() in launcher.py also does explicit cleanup for ALL
        # processes (alive or crashed) to handle the case where PyTorch skips
        # this call. We keep this code for: 1) earlier cleanup timing, 2) safety
        # if called outside launcher context, 3) code clarity.
        for stream in (self.proc.stdout, self.proc.stderr):
            if stream:
                with contextlib.suppress(Exception):
                    stream.close()


class GrpcWriterThread(threading.Thread):
    """
    Writer thread that sends logs to gRPC server instead of writing to local file.

    This replaces the local file writer when using centralized log aggregation.
    All Lustre O_APPEND multi-writer problems are eliminated because only the
    gRPC server writes to the Lustre file (single writer).

    Benefits:
    - No Lustre lock contention (single writer at server)
    - No line corruption/interleaving at Lustre level
    - Deterministic ordering (controlled by server)
    - Can scale to thousands of nodes

    Trade-offs:
    - Network overhead (sends logs to server)
    - Single point of failure (server must be reliable)
    - Latency (network + server processing)
    """

    def __init__(
        self,
        write_queue: queue.Queue,
        grpc_server_address: str,
        node_id: Union[int, str],
        logger: logging.Logger,
    ):
        """
        Initialize gRPC writer thread.

        Args:
            write_queue: Queue of (log_file_path, data) tuples from reader thread
            grpc_server_address: Address of gRPC server (e.g., "localhost:50051")
            node_id: Unique node identifier (int for rank, or string for hostname/UUID)
            logger: Logger instance
        """
        super().__init__(daemon=True, name="GrpcLogWriter")

        self.write_queue = write_queue
        self.grpc_server_address = grpc_server_address
        # Convert node_id to string for protobuf (allows int ranks or string UUIDs)
        self.node_id = str(node_id)
        self.logger = logger
        self.shutdown_requested = False

        # Statistics
        self.total_bytes_sent = 0
        self.total_chunks_sent = 0
        self.connection_errors = 0

    def _wait_for_server_ready(self) -> Optional['grpc.Channel']:
        """
        Wait for gRPC server to be ready using health checks.

        This method performs lightweight health check probes instead of heavy
        connection attempts. It prevents thundering herd when 1000+ clients
        start before the server is ready.

        Returns:
            gRPC channel if server is ready (caller must close it)
            None if shutdown was requested
        """
        attempt = 0
        logged_waiting = False

        while not self.shutdown_requested:
            try:
                # Create channel with streaming-ready settings (will be reused)
                # This avoids double TCP handshake (health check + streaming)
                channel = grpc.insecure_channel(
                    self.grpc_server_address,
                    options=[
                        ('grpc.max_send_message_length', 10 * 1024 * 1024),  # 10MB
                        ('grpc.max_receive_message_length', 10 * 1024 * 1024),
                        ('grpc.keepalive_time_ms', 10000),
                        ('grpc.keepalive_timeout_ms', 5000),
                    ],
                )
                stub = log_aggregation_pb2_grpc.LogAggregationServiceStub(channel)

                # Lightweight health check (not a full streaming connection)
                response = stub.HealthCheck(log_aggregation_pb2.HealthRequest(), timeout=1.0)

                if response.healthy:
                    self.logger.info(
                        f"gRPC server ready ({response.connected_clients} clients connected)"
                    )
                    # Return channel for reuse (avoids second TCP handshake!)
                    return channel

            except grpc.RpcError:
                # Server not ready yet - expected, will retry
                attempt += 1

                # Close failed channel before retrying
                if 'channel' in locals():
                    channel.close()

                # Log only the first time and periodically to reduce spam
                if attempt == 1:
                    self.logger.info(
                        f"Waiting for gRPC server at {self.grpc_server_address} "
                        f"(node_id={self.node_id})..."
                    )
                    logged_waiting = True
                elif attempt % 30 == 0:  # Every 30 seconds
                    self.logger.info(
                        f"Still waiting for gRPC server (node_id={self.node_id}, "
                        f"attempt {attempt})..."
                    )

                # Sleep with small random jitter to prevent thundering herd
                # Base interval: 1 second, jitter: 0-0.5 seconds
                time.sleep(1.0 + random.uniform(0, 0.5))

            except Exception as e:
                # Non-RPC errors: AttributeError (protobuf issue), ImportError, etc.
                # These are NOT retryable - indicate a bug or misconfiguration
                if 'channel' in locals():
                    channel.close()
                self.logger.error(
                    f"Fatal error during health check (node_id={self.node_id}): {e}", exc_info=True
                )
                return None  # Abort, don't retry

        if logged_waiting:
            self.logger.info(
                f"Shutdown requested while waiting for server (node_id={self.node_id})"
            )
        return None

    def run(self):
        """
        Main loop: wait for server, then consume queue and stream to gRPC server.

        Uses health check-based startup synchronization to avoid thundering herd
        when many clients start before the server is ready.
        """
        # Add initial random delay (0-3 seconds) to stagger connection attempts
        # This prevents all 1500 clients from hitting the server simultaneously
        initial_delay = random.uniform(0, 3.0)
        self.logger.debug(
            f"Initial connection delay: {initial_delay:.2f}s (node_id={self.node_id})"
        )
        time.sleep(initial_delay)

        # Server is ready, now stream logs
        channel = None
        while not self.shutdown_requested:
            # Get channel (either from initial health check or after reconnect)
            channel = self._wait_for_server_ready()
            if channel is None:
                # Shutdown requested during health check
                break

            try:
                # Try to stream logs using the channel
                self._stream_logs(channel)
            except grpc.RpcError as e:
                self.connection_errors += 1

                # If shutdown was requested and streaming failed, logs may be lost
                if self.shutdown_requested:
                    queue_size = self.write_queue.qsize()
                    self.logger.error(
                        f"gRPC connection lost during shutdown (node_id={self.node_id}): {e}. "
                        f"Queue still has {queue_size} items. Logs may be lost!"
                    )
                else:
                    self.logger.warning(
                        f"gRPC connection lost (node_id={self.node_id}): {e}. "
                        f"Waiting for server to come back..."
                    )
            except Exception as e:
                self.logger.error(
                    f"Unexpected error in gRPC writer (node_id={self.node_id}): {e}",
                    exc_info=True,
                )
                time.sleep(_GRPC_RECONNECT_INTERVAL)
            finally:
                # Always close channel after streaming (success or error)
                # Next iteration will create a fresh channel
                if channel:
                    channel.close()
                    channel = None

        # When shutdown is requested, the generator exits immediately without draining.
        # Any remaining items in the queue are not sent - during shutdown, the server
        # may already be closing, so delivery cannot be guaranteed anyway.

        self.logger.debug(f"gRPC writer thread exiting (node_id={self.node_id})")

    def _stream_logs(self, channel: 'grpc.Channel'):
        """
        Stream logs to server using provided gRPC channel.

        The generator consumes the queue until shutdown is requested, then exits immediately.
        No drain phase is performed because:
        1. During shutdown, the server may already be closing/closed
        2. Draining adds delay that can exceed the reader's shutdown timeout
        3. Any logs still in queue at shutdown are not guaranteed to be delivered anyway

        Args:
            channel: Existing gRPC channel (caller manages lifecycle)
        """
        stub = log_aggregation_pb2_grpc.LogAggregationServiceStub(channel)

        def log_generator():
            """Generator that yields LogChunk messages from queue until shutdown."""
            while not self.shutdown_requested:
                try:
                    log_file_path, data = self.write_queue.get(timeout=0.1)

                    # Encode string to bytes (done once here, not in reader thread)
                    data_bytes = data.encode('utf-8')

                    # Create LogChunk message
                    chunk = log_aggregation_pb2.LogChunk(
                        node_id=self.node_id, data=data_bytes, file_path=log_file_path
                    )

                    # Update statistics
                    self.total_chunks_sent += 1
                    self.total_bytes_sent += len(data_bytes)

                    yield chunk

                except queue.Empty:
                    # Timeout - normal, just check shutdown flag
                    continue
                except UnicodeEncodeError as e:
                    # Log data encoding issue (rare - indicates non-UTF8 data)
                    self.logger.error(
                        f"Encoding error preparing log chunk (node_id={self.node_id}): {e}",
                        exc_info=True,
                    )
                    return
                except Exception as e:
                    # Unexpected errors (should not happen)
                    self.logger.error(
                        f"Unexpected error preparing log chunk (node_id={self.node_id}): {e}",
                        exc_info=True,
                    )
                    return

        try:
            # Stream logs to server until shutdown is requested
            # No timeout on the RPC - let it run as long as needed during normal operation
            # The reader thread's join(timeout=30s) provides protection during shutdown
            response = stub.StreamLogs(log_generator())
            self.logger.debug(
                f"Stream completed: {response.status}, "
                f"server received {response.bytes_received} bytes"
            )
        except grpc.RpcError as e:
            # Connection lost or server error - will be caught by caller for retry
            raise

    def shutdown(self):
        """Request graceful shutdown of writer thread."""
        self.logger.info(
            f"Shutting down gRPC writer. Sent {self.total_chunks_sent} chunks, "
            f"{self.total_bytes_sent / 1024 / 1024:.2f} MB"
        )
        self.shutdown_requested = True


@dataclass
class ReaderConfig:
    """Configuration for the reader thread - all state needed for one cycle."""

    pipes: dict[int, int]
    log_file_path: str
    world_size: Optional[int]
    local_to_global_rank: dict[int, int]
    launcher_pipe_fd: Optional[int] = None  # Special pipe for launcher logs
    launcher_log_file_path: Optional[str] = None  # Separate file for launcher logs


class MultiplexingReaderThread(threading.Thread):
    """
    Reader thread that multiplexes multiple worker pipes into log files.

    Architecture:
    - Reader thread: Polls pipes, reads data, adds rank prefixes, queues writes
    - Writer thread: Consumes queue, writes to Lustre in batches

    This decouples pipe draining from Lustre I/O to prevent:
    - Main thread blocking on pipe writes during slow Lustre I/O
    - GIL contention between main thread and I/O operations

    Uses select.poll() to efficiently monitor multiple pipes without busy-waiting.

    Line Buffering Strategy:
        To prevent line merging when multiple ranks write simultaneously, we buffer
        incomplete lines (those not ending with '\\n') until the next read completes them.

        Tradeoff: If a worker crashes mid-line, the incomplete line is still written
        when the pipe closes (with an added '\\n'), ensuring no data loss for debugging
        crash scenarios.
    """

    def __init__(
        self,
        pipes: dict[int, int],
        log_file_path: str,
        world_size: Optional[int] = None,
        local_to_global_rank: Optional[dict[int, int]] = None,
        launcher_pipe_fd: Optional[int] = None,
        launcher_log_file_path: Optional[str] = None,
        grpc_config: Optional[tuple[str, int]] = None,  # (server_address, node_id)
    ):
        """
        Initialize reader thread.

        Args:
            pipes: {local_rank: pipe_fd} mapping for all local workers
                   Each pipe carries both stdout and stderr (merged via subprocess.STDOUT)
            log_file_path: Path to consolidated log file to write to
            world_size: Total number of ranks across all nodes (for rank padding like srun -l)
            local_to_global_rank: Optional mapping of local_rank -> global_rank for prefixes.
                                 If provided, uses global ranks in prefixes (like srun -l).
                                 If None, uses local ranks.
            launcher_pipe_fd: Optional FD for launcher logs (writes without prefix)
            launcher_log_file_path: Optional separate file for launcher logs
            grpc_config: Optional (server_address, node_id) for gRPC-based logging.
                        If provided, logs are sent to gRPC server instead of local file.
        """
        super().__init__(daemon=True)

        # Current configuration
        self._current_config = ReaderConfig(
            pipes=pipes.copy(),
            log_file_path=log_file_path,
            world_size=world_size,
            local_to_global_rank=local_to_global_rank or {},
            launcher_pipe_fd=launcher_pipe_fd,
            launcher_log_file_path=launcher_log_file_path,
        )

        # Logger and control
        self.logger = logging.getLogger(LogConfig.name)
        self._shutdown_requested = False

        # Synchronization events for FD unregistration (prevents FD reuse bugs)
        # Used by clear_all_pipes() to block launcher until all FDs are unregistered
        self._clear_pipes_event = threading.Event()
        self._pipes_cleared_event = threading.Event()

        # Write queue for decoupling pipe reading from writer I/O
        # Queue items: (log_file_path, data_to_write)
        self._write_queue = queue.Queue()
        self._writer_shutdown = False
        self._grpc_config = grpc_config

        # Start writer thread immediately (must be before reader starts polling)
        # Use gRPC writer if configured, otherwise use local file writer
        if grpc_config:
            server_address, node_id = grpc_config
            self._writer_thread = GrpcWriterThread(
                write_queue=self._write_queue,
                grpc_server_address=server_address,
                node_id=node_id,
                logger=self.logger,
            )
            self.logger.info(f"gRPC writer: {server_address}")
        else:
            self._writer_thread = threading.Thread(
                target=self._writer_thread_loop, daemon=True, name="LogWriter"
            )
            self.logger.info("Using local file writer")

        self._writer_thread.start()

        # Pre-initialize poller and fd mapping
        self.poller = select.poll()
        self.fd_to_local_rank = {}

        # Initialize all other state from config
        self._initialize_state_from_config(self._current_config)

    def _writer_thread_loop(self):
        """
        Writer thread loop: consumes write queue and writes to Lustre in batches.

        This thread isolates Lustre I/O from pipe reading, preventing:
        - Main thread blocking when pipes fill during slow Lustre writes
        - Reader thread blocking on Lustre (can't drain other pipes)

        Batching strategy:
        - Accumulates writes up to _WRITE_BATCH_SIZE (256KB by default)
        - Flushes at least every _FLUSH_INTERVAL_SECONDS (1s)
        - Larger batches improve Lustre performance (reduces metadata ops)
        """
        # Track open files by path (lazy open)
        open_files = {}  # {path: file_object}

        # Batch accumulators per file path
        batches = {}  # {path: list of strings}
        batch_sizes = {}  # {path: total bytes}
        last_flush_time = time.time()

        def flush_batches():
            """Helper to flush all pending batches and update timestamp."""
            nonlocal last_flush_time
            for path, batch in batches.items():
                if batch and path in open_files:
                    open_files[path].write(''.join(batch))
                    open_files[path].flush()  # GIL released during flush syscall
                    batch.clear()
                    batch_sizes[path] = 0
            last_flush_time = time.time()

        def add_to_batch(log_file_path, data):
            """Helper to add data to batch, lazily opening file if needed."""
            # Lazy open files as needed
            if log_file_path not in open_files:
                # Match buffer size to batch size for optimal Lustre performance
                # This ensures 256KB batches are written in 1-2 large syscalls instead of ~32 small ones
                open_files[log_file_path] = open(log_file_path, 'a', buffering=_WRITE_BATCH_SIZE)
                batches[log_file_path] = []
                batch_sizes[log_file_path] = 0

            # Add to appropriate batch
            batches[log_file_path].append(data)
            batch_sizes[log_file_path] += len(data)

        try:
            while not self._writer_shutdown:
                try:
                    # Get next write request with short timeout for responsive shutdown
                    log_file_path, data = self._write_queue.get(timeout=0.1)

                    add_to_batch(log_file_path, data)

                    # Flush if any batch is large enough or time interval reached
                    current_time = time.time()
                    should_flush = (
                        any(size >= _WRITE_BATCH_SIZE for size in batch_sizes.values())
                        or current_time - last_flush_time >= _FLUSH_INTERVAL_SECONDS
                    )

                    if should_flush:
                        flush_batches()

                except queue.Empty:
                    # Timeout - check if flush interval reached (don't flush on every timeout!)
                    current_time = time.time()
                    if current_time - last_flush_time >= _FLUSH_INTERVAL_SECONDS:
                        flush_batches()

            # Shutdown: first drain any remaining queue items into batches
            while not self._write_queue.empty():
                try:
                    log_file_path, data = self._write_queue.get_nowait()
                    add_to_batch(log_file_path, data)
                except queue.Empty:
                    break

            # Then flush all batches (both accumulated and freshly drained)
            flush_batches()

        except Exception as e:
            # CRITICAL: Unexpected exception in writer thread
            # This indicates a bug in our code or system failure (disk full, etc.)
            # Log and exit thread - launcher continues, but logging may be incomplete
            self.logger.error(
                f"Writer thread crashed with unexpected exception: {e}\n"
                f"Logging may be incomplete. This indicates a bug or system failure.",
                exc_info=True,
            )

        finally:
            # Close any remaining open files
            for file_obj in open_files.values():
                with contextlib.suppress(Exception):
                    file_obj.close()

    def _initialize_state_from_config(self, config: ReaderConfig):
        """
        Initialize working state from a configuration.

        Args:
            config: Configuration to initialize from
        """
        # Set configuration values
        self.pipes = config.pipes.copy()
        self.log_file_path = config.log_file_path
        self.world_size = config.world_size
        self.local_to_global_rank = config.local_to_global_rank.copy()
        self.launcher_pipe_fd = config.launcher_pipe_fd
        self.launcher_log_file_path = config.launcher_log_file_path

        # Calculate rank prefix padding width (like srun -l)
        # Determine padding width based on world_size (max rank is world_size - 1)
        self._rank_width = (
            len(str(self.world_size - 1))
            if (self.world_size is not None and self.world_size > 1)
            else 0
        )

        # Buffer for incomplete lines (lines without trailing '\n')
        # Maps rank -> accumulated incomplete line text
        # This prevents line merging when os.read() returns partial data
        self._line_buffers = {}

        # Register all worker pipes with poller and increase buffer size
        for local_rank, pipe_fd in config.pipes.items():

            self.poller.register(pipe_fd, select.POLLIN | select.POLLHUP)
            self.fd_to_local_rank[pipe_fd] = local_rank

        # Register launcher pipe with special rank -1 (no prefix needed)
        # IMPORTANT: Check if already registered (happens during cycle transitions)
        # since _cleanup_resources() preserves the launcher pipe across cycles
        if config.launcher_pipe_fd is not None:
            # Only register if not already in fd_to_local_rank (first time or after unregister)
            if config.launcher_pipe_fd not in self.fd_to_local_rank:
                self.poller.register(config.launcher_pipe_fd, select.POLLIN | select.POLLHUP)
                self.fd_to_local_rank[config.launcher_pipe_fd] = -1  # Special rank for launcher

    def _cleanup_resources(self):
        """
        Clean up all resources: poller registrations and state.

        This centralizes all cleanup logic used by:
        - _apply_config() when switching cycles
        - run() finally block for shutdown

        IMPORTANT: Does NOT close FDs! File objects (handler.proc.stdout) own the FDs.
        They are closed by _stop_workers() in launcher.py to avoid FD ownership races.

        NOTE: Log file writing is handled by the writer thread, not here.
        """
        # 1. Queue any incomplete lines for writing
        if self._line_buffers:
            for local_rank, incomplete_data in list(self._line_buffers.items()):
                self._flush_incomplete_line(local_rank, incomplete_data)

            self._line_buffers.clear()

        # 3. Unregister all FDs from poller EXCEPT launcher pipe (it persists across cycles)
        for fd in list(self.fd_to_local_rank.keys()):
            # Skip launcher pipe - it should persist across training cycles
            if fd == self.launcher_pipe_fd:
                continue

            with contextlib.suppress(Exception):
                self.poller.unregister(fd)

            # Remove from tracking dict
            self.fd_to_local_rank.pop(fd, None)

        # 4. Clear worker pipes state (but keep launcher pipe in fd_to_local_rank)
        self.pipes.clear()
        self._line_buffers.clear()

    def _queue_lines_for_write(self, local_rank: int, lines: list[str]) -> None:
        """
        Queue complete lines for writing with rank prefix (unless launcher).

        Uses line-aligned chunking to ensure each queued item contains only complete
        lines and doesn't exceed the maximum chunk size. This is important for:
        1. gRPC transmission efficiency (avoids partial lines across chunks)
        2. Lustre write atomicity (if using direct file writing)
        3. Network efficiency (reasonable chunk sizes)

        Args:
            local_rank: Local rank number (-1 for launcher, 0+ for workers)
            lines: List of lines to write (should all end with '\\n')
        """
        # Filter out noise lines for worker logs (nvidia driver dumps, etc.)
        # Don't filter launcher logs - they won't have this noise
        if local_rank != -1:
            lines = [line for line in lines if not _should_filter_line(line)]
            # If all lines were filtered, nothing to do
            if not lines:
                return

        # Determine target file
        log_file_path = self.launcher_log_file_path if local_rank == -1 else self.log_file_path

        # Prepare lines with prefix
        if local_rank == -1:
            # Launcher logs - no prefix
            prefixed_lines = lines
        else:
            # Worker logs - add rank prefix to each line
            rank = self.local_to_global_rank.get(local_rank, local_rank)
            prefix = f'{rank:>{self._rank_width}}: ' if self._rank_width > 0 else f'{rank}: '
            prefixed_lines = [prefix + line for line in lines]

        # Chunk into line-aligned groups that don't exceed max chunk size
        # This ensures each write/transmission contains only complete lines
        current_chunk = []
        current_size = 0
        max_chunk_size = _GRPC_MAX_CHUNK_SIZE  # Used for both gRPC and direct file writing

        for line in prefixed_lines:
            line_size = len(line.encode('utf-8'))  # Byte size matters for atomicity

            # If adding this line would exceed max size, flush current chunk
            if current_size > 0 and current_size + line_size > max_chunk_size:
                # Queue complete chunk as string (writer thread will encode)
                self._write_queue.put((log_file_path, ''.join(current_chunk)))
                current_chunk = []
                current_size = 0

            current_chunk.append(line)
            current_size += line_size

        # Queue remaining lines as string
        if current_chunk:
            self._write_queue.put((log_file_path, ''.join(current_chunk)))

    def _flush_incomplete_line(self, local_rank: int, incomplete_data: str) -> None:
        """
        Queue an incomplete line (no trailing newline) for writing, adding newline.

        Helper to avoid code duplication between _cleanup_resources() and POLLHUP handling.

        Args:
            local_rank: Local rank number (-1 for launcher, 0+ for workers)
            incomplete_data: Incomplete line data (no trailing '\\n')
        """
        if not incomplete_data:
            return

        # Determine target file
        log_file_path = self.launcher_log_file_path if local_rank == -1 else self.log_file_path

        # Add prefix if worker rank
        if local_rank != -1:
            rank = self.local_to_global_rank.get(local_rank, local_rank)
            prefix = f'{rank:>{self._rank_width}}: ' if self._rank_width > 0 else f'{rank}: '
            incomplete_data = prefix + incomplete_data

        # Queue the incomplete line with newline as string
        self._write_queue.put((log_file_path, incomplete_data + '\n'))

    def _final_launcher_pipe_cleanup(self) -> None:
        """
        Final flush and cleanup of launcher pipe before shutdown.

        WHY THIS IS NEEDED:
        During shutdown, the launcher may still be writing final logs (e.g., "Workers stopped",
        "Checkpoint saved") to its pipe while this reader thread is shutting down. Without this
        final drain step, those logs would be lost or cause gRPC "client aborted" errors.

        WHAT IT DOES:
        1. Non-blocking poll to read any final data from launcher pipe (last chance capture)
        2. Wait for write queue to drain (up to 2s) - ensures buffered logs reach writer
        3. Unregister launcher pipe from poller (prevents further events)
        4. Remove from tracking dict (cleanup state)

        WHAT HAPPENS WITHOUT IT:
        - gRPC mode: Writer shuts down while queue has data → "client aborted" errors
        - File mode: Final launcher logs may not be flushed → incomplete log files
        - Both: Race condition where launcher writes after reader stops polling
        """
        if not self.launcher_pipe_fd:
            return

        # Read any remaining data from launcher pipe (non-blocking)
        try:
            final_events = self.poller.poll(0)  # Non-blocking poll
            if final_events:
                self._process_pipe_events(final_events)
        except OSError:
            pass

        # Wait for write queue to drain (give writer time to send queued logs)
        # This prevents "client aborted" by ensuring queue is empty before shutdown
        max_wait = 2.0  # seconds
        wait_start = time.time()
        while time.time() - wait_start < max_wait:
            if self._write_queue.qsize() == 0:
                break
            time.sleep(0.1)

        # NOW unregister launcher pipe - queue is drained
        with contextlib.suppress(Exception):
            self.poller.unregister(self.launcher_pipe_fd)
        self.fd_to_local_rank.pop(self.launcher_pipe_fd, None)

    def _process_pipe_events(self, events):
        """
        Process pipe events from poll().

        Helper method to avoid code duplication between main loop and final drain.

        Args:
            events: List of (fd, event) tuples from poller.poll()
        """
        for fd, event in events:
            local_rank = self.fd_to_local_rank.get(fd)
            if local_rank is None:
                # Defensive check - should be impossible since poller only returns
                # events for registered FDs, and we maintain the invariant that
                # all registered FDs are in fd_to_local_rank
                self.logger.warning(
                    f"Received event for unknown FD {fd} - this should not happen! "
                    f"This indicates a bug in the reader thread logic."
                )
                continue

            if event & (select.POLLIN | select.POLLHUP):
                try:
                    # Read available data
                    data = os.read(fd, 65536)
                except OSError as e:
                    # EINTR: system call interrupted by signal, retry
                    if e.errno == errno.EINTR:
                        continue
                    # Any other OSError means FD was closed externally - expected during
                    # worker termination. This is a race condition: launcher closes pipes
                    # in _stop_workers() while reader thread still has FD registered.
                    # The OS may reuse the FD number for another operation, causing various
                    # errors (EBADF, EISDIR, ESPIPE, etc.) depending on the reuse.
                    # Treat as pipe closure (data=None triggers cleanup below).
                    data = None

                if not data:
                    # Pipe closed - queue any buffered incomplete data
                    if local_rank in self._line_buffers:
                        self._flush_incomplete_line(local_rank, self._line_buffers[local_rank])
                        del self._line_buffers[local_rank]

                    # Unregister FD from poller (FD ownership: file objects own FDs)
                    rank_for_log = (
                        self.local_to_global_rank.get(local_rank, local_rank)
                        if local_rank != -1
                        else -1
                    )

                    try:
                        self.poller.unregister(fd)
                    except Exception:
                        # This is unexpected! FD should be registered (we just got an event from it)
                        # Log with full exception details for debugging
                        self.logger.warning(
                            f"UNEXPECTED: Failed to unregister FD {fd} for rank {rank_for_log} after pipe closure",
                            exc_info=True,
                        )

                    # Clean up tracking dicts
                    self.pipes.pop(local_rank, None)
                    self.fd_to_local_rank.pop(fd, None)
                else:
                    # Process data from this rank's pipe
                    # Decode with error replacement to handle any encoding issues
                    text = data.decode('utf-8', errors='replace')

                    # Clean NULL bytes to prevent log file from being detected as binary
                    # (rare, but can happen during worker crashes with binary data on stdout/stderr)
                    text = text.replace('\x00', '<NUL>')

                    # Split into lines, keeping line endings
                    # Important: splitlines(keepends=True) on "abc\ndef" returns ['abc\n', 'def']
                    # The last element 'def' has NO newline - it's an incomplete line!
                    lines = text.splitlines(keepends=True)

                    if not lines:
                        # Empty read (shouldn't happen, but be defensive)
                        continue

                    # First, complete any previously buffered incomplete line
                    if local_rank in self._line_buffers:
                        # Prepend buffered text to first line of this read
                        lines[0] = self._line_buffers[local_rank] + lines[0]
                        del self._line_buffers[local_rank]

                    # Check if last line is incomplete and buffer it
                    if not lines[-1].endswith('\n'):
                        # Last line incomplete - buffer it for next read
                        self._line_buffers[local_rank] = lines.pop()

                    # Queue any remaining complete lines for writing
                    if lines:
                        self._queue_lines_for_write(local_rank, lines)

    def run(self):
        """
        Main loop: poll pipes, read data, add rank prefixes, queue for writing.

        Runs until shutdown() is called or exception occurs.

        Thread lifecycle:
            - Created once for first training cycle
            - Reused across cycles via update_pipes()
            - Must call shutdown() + join() for graceful cleanup (flushes buffers!)
            - Daemon=True only as safety fallback if shutdown() not called

        Implementation notes:
            - Buffers incomplete lines (no trailing '\\n') to prevent line merging
            - Queues writes to separate writer thread (decouples Lustre I/O from pipe reading)
            - Handles partial reads from os.read() correctly
            - Supports thread reuse: updates config when paths change
            - Manages two log files: worker logs (with prefix) and launcher logs (no prefix)
        """
        active_config = None  # Track which config we're currently using

        try:
            active_config = self._current_config  # Remember what we've applied

            while not self._shutdown_requested:
                # Check for immediate pipe clear request (FD reuse prevention)
                # This must happen before config changes and poll() to ensure
                # FDs are unregistered ASAP after launcher closes pipes
                if self._clear_pipes_event.is_set():
                    self._cleanup_resources()  # Unregister all FDs immediately
                    self._clear_pipes_event.clear()
                    self._pipes_cleared_event.set()  # Signal completion to launcher
                    continue  # Skip to next iteration

                # Check if configuration changed
                current_config = self._current_config

                if current_config is not active_config:
                    # Configuration changed - apply it
                    # Flushes buffers, closes log files, unregisters FDs, opens new log files
                    # (FD ownership: see _cleanup_resources() docstring)
                    self._apply_config(current_config)
                    active_config = current_config

                # Note: Even if self.pipes is empty, we don't break!
                # poller.poll() returns [] (empty) when no FDs registered,
                # event loop doesn't execute, and we loop back to check for
                # config changes (update_pipes() adding new pipes for next cycle).
                # This is how thread reuse works - thread stays alive between cycles!

                try:
                    # Wait for any pipe to have data (timeout 100ms for quick shutdown response)
                    # Note: Shorter timeout = faster shutdown response when shutdown() is called
                    # Trade-off: More frequent loop iterations (negligible CPU overhead)
                    events = self.poller.poll(100)  # timeout is positional arg in milliseconds
                except OSError as e:
                    # Handle EINTR (interrupted system call) - can happen if process receives signal
                    if e.errno == errno.EINTR:
                        continue
                    else:
                        # Other OSError - will be caught and logged by outer exception handler
                        raise

                # Process events
                self._process_pipe_events(events)

        except KeyboardInterrupt:
            raise
        except (IOError, OSError) as e:
            self.logger.error(
                f"CRITICAL: MultiplexingReaderThread I/O failure (log file: {self.log_file_path})",
                exc_info=True,
            )
            raise
        except Exception as e:
            self.logger.error(
                f"CRITICAL: MultiplexingReaderThread crashed unexpectedly (log file: {self.log_file_path})",
                exc_info=True,
            )
            raise
        finally:
            # Clean up all resources (queue incomplete lines for writing)
            # Note: _cleanup_resources() preserves launcher pipe during cycle changes
            self._cleanup_resources()

            # CRITICAL: Final flush of launcher logs before shutdown
            # This ensures all launcher logs are captured without causing "client aborted"
            self._final_launcher_pipe_cleanup()

            # Shut down writer thread (queue should be empty or nearly empty now)
            if isinstance(self._writer_thread, GrpcWriterThread):
                # gRPC writer has its own shutdown method
                self._writer_thread.shutdown()
            else:
                # Local file writer uses flag
                self._writer_shutdown = True

            if self._writer_thread and self._writer_thread.is_alive():
                self._writer_thread.join(timeout=30.0)  # Wait for writer to complete disconnect
                if self._writer_thread.is_alive():
                    self.logger.warning(
                        "Writer thread did not exit within 30s timeout during reader shutdown. "
                        "Connection may be aborted."
                    )

    def update_pipes(
        self,
        new_pipes: dict[int, int],
        new_log_file: str,
        new_world_size: Optional[int] = None,
        new_local_to_global_rank: Optional[dict[int, int]] = None,
        new_launcher_pipe_fd: Optional[int] = None,
        new_launcher_log_file: Optional[str] = None,
    ):
        """
        Update the pipes being monitored by this thread (for new training cycle).

        Args:
            new_pipes: {local_rank: pipe_fd} for new cycle's workers
            new_log_file: Path to new cycle's log file
            new_world_size: Total ranks (for rank padding)
            new_local_to_global_rank: Rank mapping for new cycle
            new_launcher_pipe_fd: Optional FD for launcher logs (persists across cycles)
            new_launcher_log_file: Optional separate file for launcher logs

        How it works:
        1. Create new ReaderConfig with all new state
        2. Atomically swap the reference (Python GIL ensures atomicity)
        3. Thread detects change at start of next loop iteration
        4. Thread applies all new state at once
        5. No locks, no queues, just atomic reference assignment!
        """
        new_config = ReaderConfig(
            pipes=new_pipes.copy(),
            log_file_path=new_log_file,
            world_size=new_world_size,
            local_to_global_rank=new_local_to_global_rank or {},
            launcher_pipe_fd=new_launcher_pipe_fd,
            launcher_log_file_path=new_launcher_log_file,
        )

        # Atomic swap - thread will detect and apply this change
        self._current_config = new_config

    def clear_all_pipes(self):
        """
        Request immediate unregistration of all pipes. Blocks until complete.

        This is called by launcher after closing pipes to prevent FD reuse bugs.
        When the OS reuses FD numbers for new cycle workers, we must ensure old
        FD registrations in the poller are removed first. Otherwise, the poller
        will deliver events for the NEW pipe using the OLD FD number, causing
        data from cycle N to be written to cycle N-1's log file.

        This method blocks until the reader thread confirms all FDs are unregistered,
        guaranteeing it's safe for the launcher to spawn new workers.
        """
        self._pipes_cleared_event.clear()
        self._clear_pipes_event.set()

        # Block until reader confirms all FDs unregistered (5s timeout for safety)
        if not self._pipes_cleared_event.wait(timeout=5.0):
            self.logger.warning(
                "Reader thread did not clear pipes within 5s. "
                "This may indicate a deadlock or slow reader thread."
            )

    def _apply_config(self, new_config: ReaderConfig):
        """
        Apply new configuration.

        This is called from run() to switch to new cycle configuration.
        Uses centralized cleanup and initialization methods.

        Args:
            new_config: New configuration to apply
        """

        # Step 1: Clean up old resources (queues incomplete lines for writing)
        self._cleanup_resources()

        # Step 2: Initialize new state from config (sets all attributes, registers FDs, increases pipe buffers)
        self._initialize_state_from_config(new_config)

        # Note: No need to open/close log files - writer thread handles that

    def shutdown(self):
        """
        Request graceful shutdown of the reader thread.

        IMPORTANT: Must call this before ft_launcher exits to ensure:
        - Incomplete buffered lines are flushed to disk
        - Write queue is drained (all pending writes complete)
        - Log files are properly closed

        After calling shutdown(), use join(timeout=5.0) to wait for thread to exit.
        Daemon=True is only a safety fallback if shutdown is not called.

        Usage:
            if self._reader_thread:
                self._reader_thread.shutdown()
                self._reader_thread.join(timeout=5.0)
        """
        # First, give reader thread a moment to process any final pipe events
        # (especially POLLHUP after pipes are closed)
        # The poll timeout is 100ms, so 0.15s ensures at least one poll cycle
        time.sleep(0.15)

        self._shutdown_requested = True


class PipeBasedLogsSpecs(LogsSpecs):
    """
    LogsSpecs using pipes + reader thread (like srun --output) for consolidated per-cycle logging.

    This implementation solves buffer loss problems by using a pipe-based architecture where
    workers write to pipes and a parent thread reads from all pipes and writes to the consolidated
    log file.

    Architecture:
        1. Each worker's stdout/stderr is redirected to a pipe (via subprocess.PIPE)
        2. Parent ft_launcher runs a MultiplexingReaderThread that reads from all pipes
        3. Reader thread adds rank prefixes and writes to consolidated log file
        4. When workers exit, pipes close; call cleanup() before exit to flush buffers

    Benefits:
        - Line-buffered pipes prevent log loss on worker termination (SIGTERM)
        - Rank prefixes work for C++ logs (added in parent, not child)
        - Reduced Lustre lock contention (one writer per node vs N workers)
        - No need for sitecustomize.py injection
        - Simpler worker startup (no PYTHONPATH manipulation)

    How it works (mimics srun --output):
        Worker → pipe (line-buffered by kernel) → parent reader thread → file

        Even if worker dies (SIGTERM), data in pipe is not lost because parent
        continues reading until pipe is fully drained.

    Implementation note:
        Uses monkey-patching to inject PipeSubprocessHandler into PyTorch's
        get_subprocess_handler() function so that subprocess.PIPE markers
        are handled correctly.

    Args:
        base_log_file: Base path for log files (e.g., "/lustre/logs/job.log").
                      Creates: job_cycle0.log, job_cycle1.log, etc.
    """

    logger = logging.getLogger(LogConfig.name)

    def __init__(
        self,
        base_log_file: str,
        launcher_pipe_fd: Optional[int] = None,
        launcher_log_file: Optional[str] = None,
        grpc_server_address: Optional[str] = None,
        node_id: Optional[Union[int, str]] = None,
    ) -> None:
        """
        Initialize PipeBasedLogsSpecs.

        Args:
            base_log_file: Base path for log files (e.g., "/lustre/logs/job.log")
            launcher_pipe_fd: Optional FD for launcher logs
            launcher_log_file: Optional separate log file for launcher logs.
                              If not provided and launcher_pipe_fd is set, uses base_log_file.
            grpc_server_address: Optional gRPC server address (e.g., "localhost:50051").
                                If provided, logs are sent to gRPC server instead of
                                writing directly to Lustre file.
            node_id: Node identifier (int for rank, or string for hostname/UUID).
                    Required if grpc_server_address is provided.
        """
        if not base_log_file:
            raise ValueError("base_log_file is required for PipeBasedLogsSpecs")

        # Validate gRPC configuration
        if grpc_server_address and node_id is None:
            raise ValueError("node_id is required when grpc_server_address is provided")

        # Store the base log file path (already validated as absolute by launcher)
        # We use abspath() as a defensive measure in case this class is used standalone
        self._base_log_file = os.path.abspath(base_log_file)

        # Extract directory and ensure it exists
        log_dir = os.path.dirname(self._base_log_file) or "."
        os.makedirs(log_dir, exist_ok=True)

        # Store configuration
        self._launcher_pipe_fd = launcher_pipe_fd
        self._launcher_log_file = launcher_log_file  # Store launcher log file separately
        self._grpc_server_address = grpc_server_address
        self._node_id = node_id

        # Validate launcher pipe configuration
        if launcher_pipe_fd is not None and launcher_log_file is None:
            raise ValueError(
                "launcher_log_file must be provided when launcher_pipe_fd is set. "
                "This indicates a programming error in launcher.py."
            )

        # Always create reader thread in constructor (single construction point)
        # Thread starts immediately to prevent launcher pipe buffer from filling up
        # Worker pipes will be added per-cycle via update_pipes()
        grpc_config = (grpc_server_address, node_id) if grpc_server_address else None

        self._reader_thread = MultiplexingReaderThread(
            pipes={},  # No worker pipes yet (added per-cycle)
            log_file_path="/dev/null",  # Dummy, updated per-cycle
            world_size=None,  # Updated per-cycle
            local_to_global_rank={},  # Updated per-cycle
            launcher_pipe_fd=launcher_pipe_fd,  # May be None
            launcher_log_file_path=launcher_log_file,  # May be None
            grpc_config=grpc_config,
        )
        self._reader_thread.start()

        super().__init__(
            log_dir=log_dir,
            redirects=Std.ALL,
            tee=Std.NONE,
            local_ranks_filter=None,
        )

        self._error_dir = None
        # self._reader_thread already initialized above
        self._current_cycle_log = None
        self._world_size = None  # Will be set in reify()

    @property
    def root_log_dir(self) -> str:
        return str(self._root_log_dir)

    def reify(
        self,
        envs: dict[int, dict[str, str]],
    ) -> LogsDest:
        """
        Returns LogsDest with _PIPE_MARKER strings for stdout/stderr.

        We use a special marker string instead of subprocess.PIPE because PyTorch's
        LogsDest type signature expects strings (file paths). Our custom
        PipeSubprocessHandler detects this marker and converts to subprocess.PIPE.

        Pipes are created here; reading is configured per-cycle after workers spawn.

        Also installs monkey-patch to use PipeSubprocessHandler (once, module-level).

        Creates:
        - /tmp/torchelastic_errors_<run_id>/cycle_<N>/<rank>/error.json (per rank)
        """
        # Install monkey-patch before workers are created (idempotent, module-level)
        _patch_subprocess_handler_once()

        nprocs = len(envs)

        # For standby nodes (hot spares) with 0 workers, return early
        # _current_cycle_log will remain None until the node becomes active and reify() is called again
        if nprocs == 0:
            self.logger.info(
                "No workers to spawn (likely a standby/hot spare node). "
                "Log paths will be set up when node becomes active."
            )
            return LogsDest({}, {}, {}, {}, {})

        # Get restart count from environment
        global_env = envs[0]
        run_id = global_env.get("TORCHELASTIC_RUN_ID", "test_run_id")
        restart_count = global_env.get("TORCHELASTIC_RESTART_COUNT", "0")

        # Get world_size for rank prefix padding (like srun -l)
        world_size_str = global_env.get("WORLD_SIZE")
        world_size = int(world_size_str) if world_size_str else None

        # Build local_rank -> global_rank mapping for log prefixing
        # The 'envs' dict is keyed by local_rank and contains RANK (global_rank) env var
        local_to_global_rank = {
            local_rank: int(rank_env["RANK"]) for local_rank, rank_env in envs.items()
        }

        # Store cycle info for reader thread configuration
        self._restart_count = restart_count
        self._nprocs = nprocs
        self._world_size = world_size
        self._local_to_global_rank = local_to_global_rank

        # Calculate cycle log file path
        self._current_cycle_log = self.get_cycle_log_file(int(restart_count))

        # Return pipe marker strings to signal pipe-based redirection
        # We use a special string marker instead of subprocess.PIPE because
        # PyTorch's LogsDest expects strings, not integers
        # Our PipeSubprocessHandler will detect this marker and convert to subprocess.PIPE
        stdouts = {rank: _PIPE_MARKER for rank in range(nprocs)}
        stderrs = {rank: _PIPE_MARKER for rank in range(nprocs)}

        # Create error directory
        if not self._error_dir:
            self._error_dir = tempfile.mkdtemp(prefix=f"torchelastic_errors_{run_id}_")

        error_base_dir = os.path.join(self._error_dir, f"cycle_{restart_count}")
        os.makedirs(error_base_dir, exist_ok=True)

        # No tee
        tee_stdouts: dict[int, str] = {}
        tee_stderrs: dict[int, str] = {}

        # Create error files (per rank, required by PyTorch)
        error_files = {}
        for local_rank in range(nprocs):
            error_dir = os.path.join(error_base_dir, str(local_rank))
            os.makedirs(error_dir, exist_ok=True)
            error_file = os.path.join(error_dir, "error.json")
            error_files[local_rank] = error_file
            envs[local_rank]["TORCHELASTIC_ERROR_FILE"] = error_file

        return LogsDest(stdouts, stderrs, tee_stdouts, tee_stderrs, error_files)

    def update_reader_for_cycle(self, subprocess_handlers: dict) -> None:
        """
        Update the reader thread with new pipes for the current training cycle.

        Called from launcher.py after workers are spawned. Updates the existing reader
        thread (created in __init__) with pipes from the new cycle's workers.

        Thread Lifecycle:
            The reader thread is created once in __init__() and lives for the entire
            ft_launcher process lifetime. This method updates its configuration when
            workers restart, avoiding FD reuse issues and thread creation overhead.

        Args:
            subprocess_handlers: {rank: SubprocessHandler} for all local workers
        """
        # Check if log file path is set up
        if self._current_cycle_log is None:
            self.logger.debug(
                "update_reader_for_cycle() called but _current_cycle_log is None "
                "(normal for hot spare nodes - will be set up when node becomes active)"
            )
            return

        if not os.path.exists(self._current_cycle_log):
            open(self._current_cycle_log, 'a').close()

        # Collect pipes from all local workers
        # stdout carries both stdout and stderr (merged via subprocess.STDOUT)
        # This matches srun --output= default behavior
        pipes = {}
        for rank, handler in subprocess_handlers.items():
            # Only collect stdout pipe since stderr is redirected to it
            if handler.proc.stdout:
                pipes[rank] = handler.proc.stdout.fileno()

        if not pipes:
            # No pipes available (hot spare node with no workers)
            # Update thread with empty pipes to properly close old log file and clean up state
            self.logger.debug(
                "No pipes available from subprocess handlers (hot spare node with no workers)"
            )
            self._reader_thread.update_pipes(
                new_pipes={},  # Empty - no workers
                new_log_file=self._current_cycle_log,
                new_world_size=self._world_size,
                new_local_to_global_rank=self._local_to_global_rank,
                new_launcher_pipe_fd=self._launcher_pipe_fd,
                new_launcher_log_file=self._launcher_log_file,
            )
            return

        # Prepare gRPC config if enabled
        grpc_config = (
            (self._grpc_server_address, self._node_id) if self._grpc_server_address else None
        )

        # Always update the existing thread (created in __init__)
        # This eliminates the FD reuse catastrophe problem entirely
        self._reader_thread.update_pipes(
            new_pipes=pipes,
            new_log_file=self._current_cycle_log,
            new_world_size=self._world_size,
            new_local_to_global_rank=self._local_to_global_rank,
            new_launcher_pipe_fd=self._launcher_pipe_fd,
            new_launcher_log_file=self._launcher_log_file,
        )
        mode = "gRPC" if grpc_config else "local file"
        self.logger.info(
            f"Updated reader thread with {len(pipes)} workers for cycle, "
            f"writing to {self._current_cycle_log} ({mode} mode)"
        )

    def clear_all_pipes_from_reader(self):
        """
        Tell reader thread to unregister all pipes immediately. Blocks until complete.

        This is called by launcher._stop_workers() after closing pipes to prevent
        FD reuse bugs. When the OS reuses FD numbers for new cycle workers, we must
        ensure old FD registrations in the poller are removed first.

        This method blocks until the reader confirms all FDs are unregistered,
        guaranteeing it's safe for the launcher to spawn new workers without
        FD reuse issues.
        """
        self._reader_thread.clear_all_pipes()

    def get_cycle_log_file(self, cycle_index: int) -> str:
        """
        Instance helper to build cycle logfile for this spec's base path.
        """
        base_without_ext = os.path.splitext(self._base_log_file)[0]
        ext = os.path.splitext(self._base_log_file)[1] or ".log"
        return f"{base_without_ext}_cycle{cycle_index}{ext}"

    def cleanup(self):
        """
        Gracefully shut down the reader thread.

        IMPORTANT: Call this before ft_launcher exits to ensure:
        - All buffered log data is flushed to disk
        - Log file is properly closed
        - FDs are properly closed

        This must be called explicitly - daemon thread alone won't flush buffers!
        """
        self._reader_thread.shutdown()
        # Give reader thread enough time to:
        # 1. Complete current poll cycle (0.15s sleep in shutdown())
        # 2. Clean up resources and flush buffers
        # 3. Shut down writer thread (which may take up to 5s to drain queue)
        # Total: 0.15s + 5s writer timeout + 2s gRPC drain + buffer = 8s
        self._reader_thread.join(timeout=8.0)
        if self._reader_thread.is_alive():
            self.logger.warning(
                "Reader thread did not exit within 8s timeout. Logs may be incomplete."
            )
        else:
            self.logger.debug("Reader thread shutdown completed successfully.")

    def __repr__(self) -> str:
        return f"PipeBasedLogsSpecs(base_log_file={self._base_log_file})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PipeBasedLogsSpecs):
            return False
        return self._base_log_file == other._base_log_file
