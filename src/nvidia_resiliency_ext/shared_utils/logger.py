# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Log Manager for Large-Scale LLM Training

This module provides a simple and efficient log manager that supports both
regular logging and distributed logging for large-scale training with thousands
of GPUs. The design automatically adapts based on environment configuration.

Key Design Principles:
- Environment-driven behavior: NVRX_DIST_LOG_DIR controls distributed vs regular logging
- Per-node aggregation: When distributed logging is enabled, local rank 0 aggregates logs
- All ranks log to stderr: Ensures immediate visibility in all cases
- Simple and reliable: No complex failover mechanisms
- Scalable: Works with 3K+ GPUs without overwhelming logging infrastructure

Features:
- Dual mode operation: Regular logging (stderr/stdout) or distributed logging (file aggregation)
- Per-node log files: When distributed logging is enabled (e.g., node_hostname.log)
- Automatic rank and node identification in log messages
- Thread-safe logging with proper synchronization
- Environment variable configuration for easy deployment

Environment Variables:
    NVRX_DIST_LOG_DIR: Directory for log files. If set, enables distributed logging with aggregation.
                       If not set, logs go directly to stderr/stdout.
    NVRX_LOG_DEBUG: Set to "1", "true", "yes", or "on" to enable DEBUG level logging (default: INFO)
    NVRX_LOG_TO_STDOUT: Set to "1" to log to stdout instead of stderr

Usage:
    from nvidia_resiliency_ext.shared_utils.log import nvrx_logger as logger
    logger.info("Training started")
    logger.debug("Debug information")
    logger.error("Error occurred")
    logger.warning("Warning message")
    logger.critical("Critical error")
"""

import logging
import os
import socket
import sys
import threading
import time
from datetime import datetime
from typing import Optional


class LogMessage:
    """Represents a log message with metadata."""

    def __init__(
        self, level: int, message: str, rank: int, local_rank: int, hostname: str, timestamp: float
    ):
        self.level = level
        self.message = message
        self.rank = rank
        self.local_rank = local_rank
        self.hostname = hostname
        self.timestamp = timestamp

    def __str__(self):
        level_name = logging.getLevelName(self.level)
        timestamp_str = datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return f"[{timestamp_str}] [{level_name}] [{self.hostname}:{self.rank}({self.local_rank})] {self.message}"


class LogManager:
    """
    Log manager for large-scale LLM training.

    Supports both regular logging and distributed logging. When distributed logging
    is enabled (NVRX_DIST_LOG_DIR is set), each node logs independently to avoid
    overwhelming centralized logging systems. Local rank 0 acts as the node aggregator,
    collecting logs from all ranks on the same node and writing them to a per-node log file.
    """

    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize the distributed log manager.

        Args:
            log_dir: Directory for log files. If None, uses NVRX_DIST_LOG_DIR env var
        """
        self._lock = threading.Lock()
        self._log_queue = []
        self._aggregator_thread = None
        self._stop_event = threading.Event()

        # Get hostname for node identification
        self._hostname = socket.gethostname()

        # Use hostname as node identifier
        self.node_id = self._hostname

        # Get distributed info once during initialization
        self._rank = get_rank()
        self._local_rank = get_local_rank()
        self._is_aggregator = self._local_rank == 0

        # Use NVRX_LOG_DEBUG environment variable to determine log level
        debug_enabled = os.environ.get("NVRX_LOG_DEBUG", "").lower() in ("1", "true", "yes", "on")
        self.log_level = logging.DEBUG if debug_enabled else logging.INFO

        # Determine if distributed logging is enabled
        self._log_dir = log_dir or os.environ.get("NVRX_DIST_LOG_DIR", None)
        self.distributed_logging_enabled = self._log_dir is not None

        # Use shared temporary directory for pending messages (avoid Lustre for temp files)
        # Use /tmp with node_id to ensure all ranks on the same node use the same directory
        self._temp_dir = os.path.join("/tmp", f"nvrx_log_{self.node_id}")
        os.makedirs(self._temp_dir, exist_ok=True)

        # Track file positions for each rank to avoid re-reading
        self._file_positions = {}

        # Create logger
        self._logger = self._setup_logger()

        # Start aggregator if distributed logging is enabled and this is the aggregator
        if self.distributed_logging_enabled and self._is_aggregator:
            self._start_aggregator()

    @property
    def rank(self) -> int:
        """Get the global rank."""
        return self._rank

    @property
    def local_rank(self) -> int:
        """Get the local rank."""
        return self._local_rank

    @property
    def is_aggregator(self) -> bool:
        """Check if this rank should be the node aggregator."""
        return self._is_aggregator

    def _setup_logger(self) -> logging.Logger:
        """Setup the logger with appropriate handlers."""
        # Use a single logger name since rank is already in the log message
        logger = logging.getLogger("nvrx.dist")
        logger.setLevel(self.log_level)

        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        if self.distributed_logging_enabled:
            # Add custom handler that sends messages to aggregator
            handler = DistributedLogHandler(self)
            handler.setLevel(self.log_level)

            # Use standard formatter with distributed info
            formatter = logging.Formatter(
                fmt=f"%(asctime)s [%(levelname)s] [{self._hostname}:{self._rank}({self._local_rank})] %(filename)s:%(lineno)d %(message)s"
            )

            handler.setFormatter(formatter)
            logger.addHandler(handler)
        else:
            # Simple logging to stderr or file
            if os.environ.get("NVRX_LOG_TO_STDOUT", "").lower() in ("1", "true", "yes", "on"):
                handler = logging.StreamHandler(sys.stdout)
            else:
                handler = logging.StreamHandler(sys.stderr)

            handler.setLevel(self.log_level)

            # Use standard formatter with distributed info
            formatter = logging.Formatter(
                fmt=f"%(asctime)s [%(levelname)s] [{self._hostname}:{self._rank}({self._local_rank})] %(filename)s:%(lineno)d %(message)s"
            )

            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.propagate = False

        return logger

    def _start_aggregator(self):
        """Start the log aggregator thread."""
        if self._aggregator_thread is not None:
            return

        self._aggregator_thread = threading.Thread(
            target=self._aggregator_loop, daemon=True, name=f"LogAggregator-{self._rank}"
        )
        self._aggregator_thread.start()

    def _aggregator_loop(self):
        """Main loop for the log aggregator."""
        # Create log directory if it doesn't exist
        os.makedirs(self._log_dir, exist_ok=True)

        # Setup per-node log file
        log_file = os.path.join(self._log_dir, f"node_{self.node_id}.log")
        output = open(log_file, 'a', buffering=1)  # Line buffered

        try:
            while not self._stop_event.is_set():
                # Check for pending messages from other ranks
                self._check_pending_messages()

                # Process queued messages
                with self._lock:
                    messages = self._log_queue.copy()
                    self._log_queue.clear()

                # Write messages to output
                for msg in messages:
                    try:
                        output.write(str(msg) + '\n')
                        output.flush()
                    except Exception as e:
                        # Fallback to stderr if output fails
                        sys.stderr.write(f"Log output error: {e}\n")
                        sys.stderr.flush()

                # Sleep briefly to avoid busy waiting
                time.sleep(0.01)

        finally:
            output.close()

    def _queue_message(self, level: int, message: str):
        """Queue a log message for aggregation."""
        if not self.distributed_logging_enabled:
            return

        # Create log message
        log_msg = LogMessage(
            level=level,
            message=message,
            rank=self._rank,
            local_rank=self._local_rank,
            hostname=self._hostname,
            timestamp=time.time(),
        )

        # If this is the aggregator, queue directly
        if self._is_aggregator:
            with self._lock:
                self._log_queue.append(log_msg)
        else:
            # For non-aggregator ranks, we need to send to aggregator
            # Since we don't have inter-rank communication in this simple design,
            # we'll use a file-based message passing system
            self._send_message_to_aggregator(log_msg)

    def _send_message_to_aggregator(self, log_msg: LogMessage):
        """Send a message to the aggregator using file-based communication."""
        # Create message directory in temp dir
        msg_dir = os.path.join(self._temp_dir, "pending_messages")
        os.makedirs(msg_dir, exist_ok=True)

        # Use a single message file per rank
        msg_file = os.path.join(msg_dir, f"rank_{self._local_rank}.msg")

        try:
            # Append message to the rank's message file
            with open(msg_file, 'a') as f:
                f.write(
                    f"{log_msg.level}\t{log_msg.message}\t{log_msg.rank}\t{log_msg.local_rank}\t{log_msg.hostname}\t{log_msg.timestamp}\n"
                )
                f.flush()  # Ensure message is written immediately

        except Exception as e:
            # If file-based communication fails, just log to stderr
            sys.stderr.write(f"Failed to send message to aggregator: {e}\n")
            sys.stderr.flush()

    def _check_pending_messages(self):
        """Check for pending messages from other ranks (aggregator only)."""
        if not self._is_aggregator:
            return

        msg_dir = os.path.join(self._temp_dir, "pending_messages")
        if not os.path.exists(msg_dir):
            return

        try:
            # Look for message files from other ranks
            for filename in os.listdir(msg_dir):
                if filename.startswith('rank_') and filename.endswith('.msg'):
                    msg_file = os.path.join(msg_dir, filename)
                    try:
                        # Get current file size
                        try:
                            file_size = os.path.getsize(msg_file)
                        except (IOError, OSError):
                            # File doesn't exist or can't be accessed, skip
                            continue

                        # Get the last known position for this file
                        last_position = self._file_positions.get(msg_file, 0)

                        # If file hasn't grown, skip it
                        if file_size <= last_position:
                            continue

                        # Try to read new content from the file
                        try:
                            with open(msg_file, 'r') as f:
                                f.seek(last_position)
                                lines = f.readlines()
                        except (IOError, OSError):
                            # File is being written by another process, skip this cycle
                            continue

                        # Process each line
                        for line in lines:
                            line = line.strip()
                            if line:
                                parts = line.split('\t')
                                if len(parts) >= 6:
                                    level, message, rank, local_rank, hostname, timestamp = parts[
                                        :6
                                    ]

                                    log_msg = LogMessage(
                                        level=int(level),
                                        message=message,
                                        rank=int(rank),
                                        local_rank=int(local_rank),
                                        hostname=hostname,
                                        timestamp=float(timestamp),
                                    )

                                    with self._lock:
                                        self._log_queue.append(log_msg)

                        # Update the position for this file
                        self._file_positions[msg_file] = file_size

                    except (OSError, IOError) as e:
                        # If we can't read the file, skip it this cycle
                        continue

        except (OSError, IOError):
            # If we can't access the message directory, just continue
            pass

    @property
    def logger(self) -> logging.Logger:
        """Get the distributed logger instance.

        This property provides direct access to the underlying logger,
        allowing users to use all standard logging methods:
        - logger.debug(message)
        - logger.info(message)
        - logger.warning(message)
        - logger.error(message)
        - logger.critical(message)
        """
        return self._logger

    def shutdown(self):
        """Shutdown the log manager."""
        if self.distributed_logging_enabled and self._is_aggregator:
            self._stop_event.set()
            if self._aggregator_thread:
                self._aggregator_thread.join(timeout=5.0)

        # Clean up temporary directory (only if this is the last rank on the node)
        try:
            import shutil

            if hasattr(self, '_temp_dir') and os.path.exists(self._temp_dir):
                # Only clean up if this is the last rank (rank 0) to avoid conflicts
                if self._is_aggregator:
                    shutil.rmtree(self._temp_dir)
        except (OSError, IOError):
            # Ignore cleanup errors
            pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


class DistributedLogHandler(logging.Handler):
    """Custom log handler that sends messages to the log manager."""

    def __init__(self, log_manager: LogManager):
        super().__init__()
        self.log_manager = log_manager

    def emit(self, record: logging.LogRecord):
        """Emit a log record."""
        try:
            # Add distributed info to the record
            record.hostname = self.log_manager._hostname
            record.rank = self.log_manager._rank
            record.local_rank = self.log_manager._local_rank

            # Format the message
            msg = self.format(record)

            # Queue the message for aggregation (only if distributed logging is enabled)
            self.log_manager._queue_message(level=record.levelno, message=msg)
        except (OSError, IOError, RuntimeError):
            # Fallback to stderr if logging fails
            sys.stderr.write(f"Log handler error: {record.getMessage()}\n")
            sys.stderr.flush()


# Distributed logging utilities
def get_rank() -> int:
    """Get the global rank from environment variables.

    Supports both Slurm (SLURM_PROCID) and torchrun (RANK) conventions.
    Environment variables must be set for distributed logging to work.
    """
    # Try torchrun convention first
    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank)

    # Try Slurm convention
    rank = os.environ.get("SLURM_PROCID")
    if rank is not None:
        return int(rank)

    # Environment variables must be set for distributed logging
    raise RuntimeError(
        "Distributed logging requires environment variables to be set. "
        "Please set either RANK (torchrun) or SLURM_PROCID (Slurm) environment variable."
    )


def get_local_rank() -> int:
    """Get the local rank from environment variables.

    Supports both Slurm (SLURM_LOCALID) and torchrun (LOCAL_RANK) conventions.
    """
    # Try torchrun convention first
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank)

    # Try Slurm convention
    local_rank = os.environ.get("SLURM_LOCALID")
    if local_rank is not None:
        return int(local_rank)

    return 0


# Create and export a single shared logger instance
_log_manager_instances = {}


def setup_logger(log_dir=None) -> logging.Logger:
    """
    Setup a logger using LogManager.

    The logger automatically adapts to distributed or regular mode based on
    whether NVRX_DIST_LOG_DIR is set. If set, enables distributed logging
    with aggregation. If not set, logs go directly to stderr/stdout.

    This function is a singleton per process - each process gets its own logger instance.

    Args:
        log_dir: Optional directory path for log files. If None, uses NVRX_DIST_LOG_DIR env var.

    Returns:
        logging.Logger: Configured logger instance from LogManager
    """
    global _log_manager_instances

    # Use process ID to create process-specific logger instances
    process_id = os.getpid()

    # Return existing logger if already configured for this process
    if process_id in _log_manager_instances:
        return _log_manager_instances[process_id].logger

    # Create a LogManager instance with the specified log directory
    _log_manager_instances[process_id] = LogManager(log_dir=log_dir)

    # Return the logger from the log manager
    return _log_manager_instances[process_id].logger


# Create and export a single shared logger instance
log = setup_logger()
