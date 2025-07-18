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
- Fork-safe: All ranks use file-based messaging to ensure child processes can log

Features:
- Dual mode operation: Regular logging (stderr/stdout) or distributed logging (file aggregation)
- Per-node log files: When distributed logging is enabled (e.g., node_hostname.log)
- Automatic rank and node identification in log messages
- Thread-safe logging with proper synchronization
- Environment variable configuration for easy deployment
- Fork-safe design with file-based message passing for all ranks

Environment Variables:
    NVRX_DIST_LOG_DIR: Directory for log files. If set, enables distributed logging with aggregation.
                       If not set, logs go directly to stderr/stdout.
    NVRX_LOG_DEBUG: Set to "1", "true", "yes", or "on" to enable DEBUG level logging (default: INFO)
    NVRX_LOG_TO_STDOUT: Set to "1" to log to stdout instead of stderr

Usage:
    # In main script (launcher.py)
    from nvidia_resiliency_ext.shared_utils.log import setup_logger
    logger = setup_logger()  # Call once at startup
    
    # In other modules
    import logging
    logger = logging.getLogger("nvrx")
    logger.info("Training started")
    logger.debug("Debug information")
    logger.error("Error occurred")
    logger.warning("Warning message")
    logger.critical("Critical error")

Forking Support:
    The logger is designed to work safely with process forking. When using fork():
    
    # In parent process
    from nvidia_resiliency_ext.shared_utils.log import setup_logger
    logger = setup_logger()  # Setup before forking
    logger.info("Parent process logging")
    
    # Fork child process
    pid = os.fork()
    if pid == 0:
        # In child process - logger will work normally
        import logging
        logger = logging.getLogger("nvrx")
        logger.info("Child process logging")
    else:
        # Parent continues normally
        logger.info("Parent continues")
    
    All ranks use file-based message passing, ensuring child processes can log
    even when they don't inherit the aggregator thread from the parent.
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
    
    Fork-safe: Child processes automatically disable aggregation to avoid conflicts.
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
        # Configure the standard "nvrx" logger
        logger = logging.getLogger("nvrx")
        logger.setLevel(self.log_level)

        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        if self.distributed_logging_enabled:
            # Add custom handler that sends messages to aggregator
            handler = DistributedLogHandler(self)
            handler.setLevel(self.log_level)

            # Use dynamic formatter with static hostname and dynamic rank info
            formatter = DynamicLogFormatter(
                fmt=f"%(asctime)s [%(levelname)s] [{self._hostname}:%(rank)s(%(local_rank)s)] %(filename)s:%(lineno)d %(message)s"
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

            # Use dynamic formatter with static hostname and dynamic rank info
            formatter = DynamicLogFormatter(
                fmt=f"%(asctime)s [%(levelname)s] [{self._hostname}:%(rank)s(%(local_rank)s)] %(filename)s:%(lineno)d %(message)s"
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

        # All ranks (including aggregator) use file-based message passing
        # This ensures child processes can log even when they don't have the aggregator thread
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
            # Look for message files from all ranks (including this aggregator rank)
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
                self._aggregator_thread.join()

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
            # Format the message using the formatter (which handles rank info dynamically)
            msg = self.format(record)

            # Queue the message for aggregation (only if distributed logging is enabled)
            self.log_manager._queue_message(level=record.levelno, message=msg)
        except (OSError, IOError, RuntimeError):
            # Fallback to stderr if logging fails
            sys.stderr.write(f"Log handler error: {record.getMessage()}\n")
            sys.stderr.flush()


class DynamicLogFormatter(logging.Formatter):
    """Dynamic formatter that reads rank information at runtime with lazy initialization."""
    
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        # Cache for rank information - initialized on first use
        self._rank = None
        self._local_rank = None
    
    def format(self, record):
        # Initialize rank cache on first use
        if self._rank is None:
            self._update_rank_cache()
        
        # Add cached rank info to the record
        record.rank = self._rank
        record.local_rank = self._local_rank
        
        # Use the parent's format method
        return super().format(record)
    
    def _update_rank_cache(self):
        """Initialize the cached rank information."""
        try:
            self._rank = get_rank()
            self._local_rank = get_local_rank()
        except (RuntimeError, OSError):
            # Fallback if rank info is not available
            self._rank = 0
            self._local_rank = 0


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


def setup_logger(log_dir=None, force_reset=False) -> logging.Logger:
    """
    Setup the distributed logger.

    This function configures the standard Python logger "nvrx" with appropriate
    handlers for distributed logging. It's safe to call multiple times - if the
    logger is already configured, it won't be reconfigured unless force_reset=True.

    The logger automatically adapts to distributed or regular mode based on
    whether NVRX_DIST_LOG_DIR is set. If set, enables distributed logging
    with aggregation. If not set, logs go directly to stderr/stdout.

    The logger is fork-safe: all ranks use file-based message passing to ensure
    child processes can log even when they don't inherit the aggregator thread.

    Args:
        log_dir: Optional directory path for log files. If None, uses NVRX_DIST_LOG_DIR env var.
        force_reset: If True, force reconfiguration even if logger is already configured.
                    Useful for subprocesses that need fresh logger setup.

    Returns:
        logging.Logger: Configured logger instance

    Example:
        # In main script (launcher.py) or training subprocess
        from nvidia_resiliency_ext.shared_utils.logger import setup_logger
        logger = setup_logger()
        
        # In subprocesses that need fresh logger setup
        logger = setup_logger(force_reset=True)
        
        # In other modules
        import logging
        logger = logging.getLogger("nvrx")
        logger.info("Some message")
    """
    # Check if the nvrx logger is already configured
    logger = logging.getLogger("nvrx")
    
    # If force_reset is True or the logger has no handlers, configure it
    if force_reset or not logger.handlers:
        # Clear existing handlers if force_reset is True
        if force_reset:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            # Clear any stored log manager to force fresh creation
            if hasattr(setup_logger, '_log_manager'):
                delattr(setup_logger, '_log_manager')
        
        # Create a LogManager instance to handle the configuration
        log_manager = LogManager(log_dir=log_dir)
        
        # Get the configured logger from the log manager
        logger = log_manager.logger
        
        # Store the log manager instance to prevent garbage collection
        # This ensures the aggregator thread keeps running
        setup_logger._log_manager = log_manager
    else:
        # Logger is already configured, just return the existing logger
        logger = logging.getLogger("nvrx")
    
    return logger