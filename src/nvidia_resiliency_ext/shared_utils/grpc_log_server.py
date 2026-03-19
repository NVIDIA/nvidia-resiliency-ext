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
gRPC-based Multi-Node Log Server

This module provides a gRPC server for cluster-wide log aggregation, collecting logs from
all nodes in a distributed training job and writing them to a single centralized file.

**Key Features:**
- **Network-based**: Receives log streams from all nodes via gRPC
- **Multi-node scope**: Aggregates logs across entire cluster (1000+ nodes)
- **Single writer**: Writes to one centralized Lustre file (eliminates O_APPEND contention)
- **High performance**: Large batch writes (1MB) optimized for Lustre
- **Automatic deployment**: Launcher automatically spawns server on rank 0 node

**Comparison with log_aggregator.py:**

+--------------------+-------------------------+---------------------------+
|                    | log_aggregator.py       | grpc_log_server.py       |
+--------------------+-------------------------+---------------------------+
| Scope              | Single node             | Multi-node (cluster)      |
| Transport          | File-based (temp dir)   | Network (gRPC)            |
| Input              | All processes on 1 node | All nodes in cluster      |
| Output             | Per-node log files      | Single centralized file   |
| Lustre writers     | N (one per node)        | 1 (optimal)               |
| Deployment         | Manual (sbatch)         | Automatic (launcher)      |
+--------------------+-------------------------+---------------------------+

**When to use:**
- Use `log_aggregator.py` for node-local aggregation (multiple processes → one node file)
- Use `grpc_log_server.py` for cluster-wide centralization (multiple nodes → one global file)

**Two-level aggregation (ft_launcher, ``--ft-log-aggregator-count`` > 1):**
- **Root** (this module): listens on ``P+N``; only the root writes to Lustre; use
  ``max_workers >= N + 10`` (N = number of leaves, each holds one long-lived upstream stream).
- **Leaves**: ``grpc_log_leaf_server.py`` on ports ``P .. P+N-1``; forward to root.
  Raise ``ulimit -n`` on the store host (many TCP clients + N root streams). Leaves use a
  bounded queue (``--ft-log-leaf-max-queue-chunks``, auto-scaled by default when N>1)
  so slow Lustre applies backpressure to clients; handlers block (lossless) until a slot exists.

**Example Usage:**

Manually for testing:

    python -m nvidia_resiliency_ext.shared_utils.grpc_log_server \\
        --host 0.0.0.0 \\
        --port 50051 \\
        --output-file /lustre/logs/training.log
"""

import argparse
import contextlib
import logging
import os
import signal
import sys
import threading
import time
import urllib.parse
from concurrent import futures
from typing import Any, Dict, List, Optional, Tuple

import grpc

# Import generated protobuf code
# Note: Proto files are in shared_utils/proto/ and compiled during build (build.py)
try:
    from nvidia_resiliency_ext.shared_utils.proto import (
        log_aggregation_pb2,
        log_aggregation_pb2_grpc,
    )
except ImportError:
    # Fallback for running as script
    import os
    import sys

    # Add shared_utils/proto to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared_utils', 'proto'))
    import log_aggregation_pb2
    import log_aggregation_pb2_grpc


def _format_grpc_peer(raw: Optional[str]) -> str:
    """Decode URL-encoded characters in ``context.peer()`` for readable log lines."""
    if not raw:
        return "unknown_peer"
    return urllib.parse.unquote(raw)


class LogAggregationServicer(log_aggregation_pb2_grpc.LogAggregationServiceServicer):
    """
    gRPC servicer that receives log streams from multiple clients.

    Architecture:
    - Each client (node) opens a bidirectional stream
    - Server buffers incoming log chunks in memory
    - Writes to Lustre in large batches (1MB) for performance
    - Single writer eliminates all Lustre O_APPEND contention issues
    """

    def __init__(
        self,
        max_buffer_size: int = 1024 * 1024,
        graceful_shutdown_timeout: float = 60.0,
        flush_interval: float = 1.0,
    ):
        """
        Initialize log aggregation server.

        Args:
            max_buffer_size: Target buffer size per file used by periodic flush (default 1MB).
                             Buffers are not flushed synchronously when receiving chunks from clients.
                             The periodic flush thread flushes buffers every 1 second.
            graceful_shutdown_timeout: Maximum seconds to wait for clients during graceful
                                       shutdown (default 60.0). Server will wait for all
                                       clients to disconnect or this timeout to elapse,
                                       whichever comes first.
            flush_interval: Interval in seconds between periodic buffer flushes (default 1.0).
        """
        self.max_buffer_size = max_buffer_size
        self.graceful_shutdown_timeout = graceful_shutdown_timeout

        # Per-file state: file_path -> {file_handle, buffer, buffer_size}
        self.files: Dict[str, Dict[str, Any]] = {}
        self.files_lock = threading.Lock()

        # Track connected clients
        self.connected_clients = 0
        self.clients_lock = threading.Lock()

        # Graceful shutdown state
        self.graceful_shutdown_initiated = threading.Event()

        # Statistics
        self.total_bytes_received = 0
        self.total_chunks_received = 0

        # Logging
        self.logger = logging.getLogger("LogAggregationServer")
        self.logger.setLevel(logging.INFO)

        # Background flush thread (flushes buffer periodically)
        self.flush_interval = flush_interval
        self.shutdown_event = threading.Event()
        self.flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
        self.flush_thread.start()

    def _get_or_create_file_state(self, file_path: str) -> Dict[str, Any]:
        """
        Get or create file state for a given path.

        Args:
            file_path: Path to log file (must be non-empty)

        Returns:
            Dictionary with file_handle, buffer, buffer_size, buffer_lock, and io_lock
        """
        if not file_path:
            raise ValueError("file_path cannot be empty - clients must specify target file")

        with self.files_lock:
            if file_path not in self.files:
                # Create new file state
                os.makedirs(os.path.dirname(os.path.abspath(file_path)) or ".", exist_ok=True)
                file_handle = open(file_path, 'ab', buffering=self.max_buffer_size)

                self.files[file_path] = {
                    'file_handle': file_handle,
                    'buffer': [],
                    'buffer_size': 0,
                    'buffer_lock': threading.Lock(),  # Protects buffer operations (fast)
                    'io_lock': threading.Lock(),  # Protects file_handle I/O (slow)
                }
                self.logger.info(f"Opened new log file: {file_path}")

            return self.files[file_path]

    def _pending_buffer_totals(self) -> Tuple[int, int, List[Tuple[str, int, int]]]:
        """Sum in-memory bytes not yet flushed to disk; per-file (path, chunks, bytes)."""
        total_bytes = 0
        nonempty: List[Tuple[str, int, int]] = []
        with self.files_lock:
            snapshot = list(self.files.items())
        for fp, fs in snapshot:
            with fs['buffer_lock']:
                sz = fs['buffer_size']
                nchunks = len(fs['buffer'])
            total_bytes += sz
            if nchunks or sz:
                nonempty.append((fp, nchunks, sz))
        return total_bytes, len(nonempty), nonempty

    def _log_shutdown_snapshot(self, phase: str, *, extra: str = "") -> None:
        """Log queues / buffers that may still hold data not yet on Lustre."""
        with self.clients_lock:
            clients = self.connected_clients
        pending_bytes, n_files, details = self._pending_buffer_totals()
        line = (
            f"Shutdown snapshot [{phase}]: connected_streams={clients}, "
            f"in_memory_pending_bytes={pending_bytes}, files_with_pending_buffers={n_files}, "
            f"lifetime_chunks_received={self.total_chunks_received}, "
            f"lifetime_bytes_received={self.total_bytes_received}"
        )
        if extra:
            line += f", {extra}"
        self.logger.info(line)
        max_detail = 15
        for fp, nc, sz in details[:max_detail]:
            self.logger.info(f"  buffered path={fp!r} chunks={nc} bytes={sz}")
        if len(details) > max_detail:
            self.logger.info(f"  ... and {len(details) - max_detail} more paths with buffered data")

    def StreamLogs(self, request_iterator, context):
        """
        Handle streaming logs from one upstream ``StreamLogs`` RPC.

        Each RPC is one direct training node (single-level) or one leaf aggregator
        (two-level). Logging uses neutral ``peer`` + ``source_node_id`` (``node_id`` on
        the first chunk, from ft_launcher clients always set) so the same messages
        apply to both topologies.

        Args:
            request_iterator: Iterator of ``LogChunk`` messages from the upstream client.
            context: gRPC ServicerContext for this RPC.

        Returns:
            ``StreamResponse`` with status and cumulative bytes received on this stream.
        """
        with self.clients_lock:
            self.connected_clients += 1

        peer = _format_grpc_peer(context.peer())
        client_bytes = 0
        client_chunks = 0
        source_node_id: Optional[str] = None

        try:
            for chunk in request_iterator:
                if self.shutdown_event.is_set():
                    break
                if source_node_id is None:
                    source_node_id = chunk.node_id
                    self.logger.info(
                        f"StreamLogs stream opened peer={peer} source_node_id={source_node_id}"
                    )

                # Get data and target file
                data = chunk.data
                file_path = chunk.file_path

                if not file_path:
                    raise ValueError(
                        f"peer={peer} node_id={chunk.node_id!r} sent chunk without file_path"
                    )

                # Get file state for this path
                file_state = self._get_or_create_file_state(file_path)

                # Add to buffer for this specific file
                # Use buffer_lock (not io_lock) - fast memory operations only
                with file_state['buffer_lock']:
                    file_state['buffer'].append(data)
                    file_state['buffer_size'] += len(data)

                    # Update global statistics
                    self.total_bytes_received += len(data)
                    self.total_chunks_received += 1
                    client_bytes += len(data)
                    client_chunks += 1

            if source_node_id is not None:
                end_src = f"source_node_id={source_node_id}"
            else:
                end_src = "source_node_id=n/a"
            self.logger.info(
                f"StreamLogs stream ended peer={peer} {end_src} "
                f"chunks={client_chunks} received_KiB={client_bytes / 1024:.1f}"
            )

            # No flush-on-disconnect: periodic flush thread handles flushing (every 1s)
            # and graceful_shutdown() does a final flush before server exits.
            # This avoids severe lock contention when many clients (e.g., 768 clients
            # writing to 2 files) disconnect simultaneously during shutdown.

            return log_aggregation_pb2.StreamResponse(status="OK", bytes_received=client_bytes)

        except (OSError, IOError) as e:
            src = (
                f"source_node_id={source_node_id}"
                if source_node_id is not None
                else "source_node_id=n/a"
            )
            stream_info = f"peer={peer} {src}"
            self.logger.error(f"I/O error in StreamLogs ({stream_info}): {e}", exc_info=True)
            raise
        except ValueError as e:
            src = (
                f"source_node_id={source_node_id}"
                if source_node_id is not None
                else "source_node_id=n/a"
            )
            stream_info = f"peer={peer} {src}"
            self.logger.error(f"Protocol error in StreamLogs ({stream_info}): {e}")
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except grpc.RpcError as e:
            src = (
                f"source_node_id={source_node_id}"
                if source_node_id is not None
                else "source_node_id=n/a"
            )
            stream_info = f"peer={peer} {src}"
            if self.shutdown_event.is_set():
                self.logger.debug(f"StreamLogs aborted during shutdown ({stream_info}): {e}")
            else:
                self.logger.info(f"StreamLogs aborted ({stream_info}): {e}")
            raise
        except Exception as e:
            src = (
                f"source_node_id={source_node_id}"
                if source_node_id is not None
                else "source_node_id=n/a"
            )
            stream_info = f"peer={peer} {src}"
            self.logger.error(f"Unexpected error in StreamLogs ({stream_info}): {e}", exc_info=True)
            raise
        finally:
            with self.clients_lock:
                self.connected_clients -= 1

    def HealthCheck(self, request, context):
        """Health check endpoint."""
        with self.clients_lock:
            clients = self.connected_clients

        return log_aggregation_pb2.HealthResponse(healthy=True, connected_clients=clients)

    def _flush_file_buffer_locked(self, file_path: str, file_state: Dict[str, Any]):
        """
        Flush buffer for a specific file to Lustre using dual-lock pattern.

        This method uses two locks to minimize contention:
        1. buffer_lock: held briefly to copy and clear buffer (doesn't block client appends during I/O)
        2. io_lock: held during file I/O (serializes writes to file_handle)

        Args:
            file_path: Path to the file being flushed
            file_state: File state dictionary containing buffer, buffer_lock, io_lock, and file_handle
        """
        # Step 1: Copy buffer with buffer_lock (fast - only memory operations)
        with file_state['buffer_lock']:
            if not file_state['buffer']:
                return  # Nothing to flush

            buffer_copy = file_state['buffer'].copy()
            buffer_size = file_state['buffer_size']

            # Clear buffer immediately - clients can append new chunks while we do I/O
            file_state['buffer'].clear()
            file_state['buffer_size'] = 0

        # Step 2: Write to disk with io_lock (slow - but doesn't block buffer appends)
        with file_state['io_lock']:
            file_handle = file_state.get('file_handle')
            if not file_handle:
                return  # File was closed during shutdown

            try:
                # Write all buffered data directly as bytes
                for data in buffer_copy:
                    file_handle.write(data)

                # Flush to OS (kernel will aggregate for Lustre RPC)
                file_handle.flush()

                # Log statistics
                self.logger.debug(
                    f"Flushed {len(buffer_copy)} chunks, "
                    f"{buffer_size / 1024:.1f} KB to {file_path}"
                )

            except (OSError, IOError) as e:
                # File I/O errors: disk full, permission denied, Lustre unavailable, etc.
                self.logger.error(f"I/O error flushing to {file_path}: {e}", exc_info=True)
                raise
            except Exception as e:
                # Unexpected errors
                self.logger.error(f"Unexpected error flushing to {file_path}: {e}", exc_info=True)
                raise

    def _periodic_flush(self):
        """Background thread that flushes all file buffers periodically."""
        while True:
            # Use wait() with timeout to be responsive to shutdown
            if self.shutdown_event.wait(timeout=self.flush_interval):
                # Event was set - exit immediately
                break

            try:
                # Get snapshot of files without holding lock during flush
                with self.files_lock:
                    files_snapshot = list(self.files.items())

                # Flush each file (uses dual-lock pattern internally)
                for file_path, file_state in files_snapshot:
                    if file_state['buffer']:  # Quick check without lock
                        self._flush_file_buffer_locked(file_path, file_state)
            except (OSError, IOError) as e:
                # I/O errors during periodic flush - log but keep thread alive
                self.logger.error(
                    f"I/O error during periodic flush, will retry: {e}", exc_info=True
                )
            except Exception as e:
                # Unexpected errors - log but keep thread alive
                self.logger.error(
                    f"Unexpected error during periodic flush, will retry: {e}", exc_info=True
                )

    def graceful_shutdown(self):
        """
        Initiate graceful shutdown: keep server running to accept logs from remaining clients.

        This method will:
        1. Mark shutdown as initiated (but keep accepting connections)
        2. Wait for all clients to disconnect OR timeout to elapse
        3. Then perform final cleanup

        This ensures that when the store host is exiting, other ranks can still send their
        final logs before the server completely shuts down.
        """
        self.logger.info(
            f"Graceful shutdown initiated. Will wait up to {self.graceful_shutdown_timeout}s "
            f"for clients to finish..."
        )
        self._log_shutdown_snapshot(
            "graceful_start",
            extra="(in-memory buffers may still flush via periodic flush until final shutdown)",
        )

        # Mark graceful shutdown as initiated (but don't stop accepting connections yet)
        self.graceful_shutdown_initiated.set()

        # Wait for clients to disconnect or timeout
        start_time = time.time()
        check_interval = 0.1  # Check every 100ms for responsive shutdown

        while True:
            elapsed = time.time() - start_time

            with self.clients_lock:
                active_clients = self.connected_clients

            # Exit if no active clients
            if active_clients == 0:
                self.logger.info(
                    f"All clients disconnected after {elapsed:.1f}s. Proceeding with shutdown."
                )
                self._log_shutdown_snapshot(
                    "graceful_all_streams_closed", extra=f"waited_s={elapsed:.1f}"
                )
                break

            # Exit if timeout reached
            if elapsed >= self.graceful_shutdown_timeout:
                self.logger.warning(
                    f"Graceful shutdown timeout ({self.graceful_shutdown_timeout}s) reached "
                    f"with {active_clients} client(s) still connected. Proceeding with shutdown."
                )
                self._log_shutdown_snapshot(
                    "graceful_timeout",
                    extra=(
                        f"still_connected_streams={active_clients} — "
                        "remaining streams may be cut off by server.stop(); data may be lost"
                    ),
                )
                break

            # Log periodic status
            remaining = self.graceful_shutdown_timeout - elapsed
            if int(elapsed / 5) != int((elapsed - check_interval) / 5):  # Log every 5s
                pend_b, pend_nf, _ = self._pending_buffer_totals()
                self.logger.info(
                    f"Waiting for {active_clients} client(s) to finish... "
                    f"({remaining:.1f}s remaining) "
                    f"in_memory_pending_bytes={pend_b} files_with_pending_buffers={pend_nf}"
                )

            time.sleep(check_interval)

        # Now perform the actual shutdown
        self.shutdown()

    def shutdown(self):
        """Perform final shutdown: flush buffers, close files, stop threads."""
        self.logger.info("Shutting down log aggregation server...")
        self._log_shutdown_snapshot(
            "final_before_stop_flush_thread",
            extra="stopping periodic flush; remaining buffers will be written in final flush pass",
        )

        # Stop flush thread
        self.shutdown_event.set()
        if self.flush_thread.is_alive():
            self.flush_thread.join(timeout=0.5)

        # Flush any remaining data and close files
        with self.files_lock:
            files_snapshot = list(self.files.items())

        for file_path, file_state in files_snapshot:
            # Flush remaining data using dual-lock pattern
            if file_state['buffer']:  # Quick check without lock
                try:
                    self._flush_file_buffer_locked(file_path, file_state)
                except Exception as e:
                    # Log but don't crash shutdown
                    self.logger.error(f"Error during final flush in shutdown: {e}", exc_info=True)

            # Close file handle with io_lock
            with file_state['io_lock']:
                if file_state['file_handle']:
                    with contextlib.suppress(Exception):
                        file_state['file_handle'].close()
                    file_state['file_handle'] = None
                    self.logger.info(f"Closed log file: {file_path}")

            # Clear buffer with buffer_lock
            with file_state['buffer_lock']:
                file_state['buffer'].clear()
                file_state['buffer_size'] = 0

        leftover_b, leftover_nf, leftover_detail = self._pending_buffer_totals()
        if leftover_b or leftover_nf:
            self.logger.error(
                f"After final flush, unexpected non-empty buffers: bytes={leftover_b}, "
                f"files={leftover_nf}, detail={leftover_detail[:5]!r}"
            )
        else:
            self.logger.info("Final flush: all per-file in-memory buffers are empty.")

        self.logger.info(
            f"Server shutdown complete. "
            f"Total: {self.total_chunks_received} chunks, "
            f"{self.total_bytes_received / 1024 / 1024:.1f} MB"
        )


def serve(host: str, port: int, max_workers: int = 100, graceful_shutdown_timeout: float = 60.0):
    """
    Start gRPC log aggregation server.

    The server accepts log chunks from clients, each specifying a target file_path.
    Files are opened lazily as clients send data to them.

    Args:
        host: Host to bind to (e.g., '0.0.0.0' for all interfaces)
        port: Port to listen on
        max_workers: Maximum number of worker threads for gRPC
        graceful_shutdown_timeout: Maximum seconds to wait for clients during shutdown
    """
    # Set up logging (using aegis-style format)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] [%(process)5s] %(filename)s:%(lineno)d %(message)s',
    )
    logger = logging.getLogger("LogAggregationServer")

    servicer = LogAggregationServicer(graceful_shutdown_timeout=graceful_shutdown_timeout)

    # Create gRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', 10 * 1024 * 1024),  # 10MB
            ('grpc.max_receive_message_length', 10 * 1024 * 1024),  # 10MB
            # Keepalive enforcement (data center optimized - clients send keepalive every 60s)
            # These settings allow clients to send keepalive pings as frequently as every 5s
            ('grpc.http2.min_time_between_pings_ms', 5000),  # Min 5s between consecutive pings
            (
                'grpc.http2.min_ping_interval_without_data_ms',
                5000,
            ),  # Min 5s between pings when idle
            (
                'grpc.http2.max_pings_without_data',
                0,
            ),  # Allow unlimited pings for long-lived streams
            ('grpc.keepalive_permit_without_calls', 1),  # Permit keepalive even with no active RPCs
        ],
    )

    # Add servicer to server
    log_aggregation_pb2_grpc.add_LogAggregationServiceServicer_to_server(servicer, server)

    # Bind to address
    server_address = f'{host}:{port}'
    server.add_insecure_port(server_address)

    # Start server
    server.start()
    logger.info(f"Log aggregation server started on {server_address}")
    logger.info(f"Max workers: {max_workers}")
    logger.info(f"Graceful shutdown timeout: {graceful_shutdown_timeout}s")

    stop_grace_seconds = 5.0
    termination_wait_timeout_seconds = stop_grace_seconds + 60.0

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        # Use graceful shutdown to keep server running for remaining clients
        servicer.graceful_shutdown()
        server.stop(grace=stop_grace_seconds)
        # stop() returns before internal server thread exits; wait so the executor is not
        # torn down while still scheduling RPCs (avoids RuntimeError on shutdown).
        # wait_for_termination returns True iff the wait timed out (per grpc API).
        if server.wait_for_termination(timeout=termination_wait_timeout_seconds):
            logger.warning(
                "Timed out waiting for gRPC log server to finish termination after stop() "
                f"(>{termination_wait_timeout_seconds}s)"
            )
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait for server termination
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, initiating graceful shutdown...")
        servicer.graceful_shutdown()
        server.stop(grace=stop_grace_seconds)
        if server.wait_for_termination(timeout=termination_wait_timeout_seconds):
            logger.warning(
                "Timed out waiting for gRPC log server after keyboard interrupt "
                f"(>{termination_wait_timeout_seconds}s)"
            )


def main():
    """Main entry point for log aggregation server."""
    parser = argparse.ArgumentParser(
        description="gRPC Log Aggregation Server for Multi-Node Training. "
        "Clients specify target file paths in each log chunk."
    )
    parser.add_argument(
        '--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port', type=int, default=50051, help='Port to listen on (default: 50051)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=100,
        help='Maximum gRPC worker threads (default: 100). '
        'IMPORTANT: Should be >= number of expected client nodes. '
        'Each streaming client (node) occupies one worker thread for its entire duration. '
        'Insufficient workers cause clients to queue and block. '
        'Threads are I/O-bound (blocked 99%% of time), so 2048 threads ≈ 20MB memory. '
        'Recommended: max(100, num_nodes + 10). Safe to use 2048 for most clusters.',
    )
    parser.add_argument(
        '--graceful-shutdown-timeout',
        type=float,
        default=60.0,
        help='Maximum seconds to wait for clients during graceful shutdown (default: 60.0). '
        'When store host exits, server will continue accepting logs from other ranks '
        'for up to this timeout or until all clients disconnect, whichever comes first.',
    )

    args = parser.parse_args()

    serve(
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
        graceful_shutdown_timeout=args.graceful_shutdown_timeout,
    )


if __name__ == '__main__':
    main()
