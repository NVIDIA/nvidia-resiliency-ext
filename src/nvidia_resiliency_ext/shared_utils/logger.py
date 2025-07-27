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
NVRx Logger for Large-Scale LLM Training

This module provides a simple and efficient log manager that supports both
regular logging and distributed logging for large-scale training with thousands
of GPUs. The design automatically adapts based on environment configuration.

Key Design Principles:
- Environment-driven behavior: NVRX_DIST_LOG_DIR controls distributed vs regular logging
- Per-node aggregation: When distributed logging is enabled, local rank 0 aggregates logs
- Dynamic rank detection: Automatically reads rank info from environment variables
- Scalable: Works with 3K+ GPUs without overwhelming logging infrastructure
- Fork-safe: All ranks use file-based messaging to ensure child processes can log
- Subprocess-safe: Supports force_reset=True for fresh logger setup in subprocesses
- Service-based aggregation: Aggregator can run as a separate service for reliable log collection

Features:
- Dual mode operation: Regular logging (stderr/stdout) or distributed logging (file aggregation)
- Per-node log files: When distributed logging is enabled (e.g., node_hostname.log)
- Automatic rank and node identification in log messages
- Thread-safe logging with proper synchronization
- Environment variable configuration for easy deployment
- Fork-safe design with file-based message passing for all ranks
- Separate aggregator service: Can run independently of training processes
- Configurable temp directory: Customizable location for pending message files

Environment Variables:
    NVRX_LOG_DIR: Directory for log files. If set, enables distributed logging with aggregation.
                   If not set, logs go directly to stderr/stdout.
    NVRX_LOG_DEBUG: Set to "1", "true", "yes", or "on" to enable DEBUG level logging (default: INFO)
    NVRX_LOG_TO_STDOUT: Set to "1" to log to stdout instead of stderr
    NVRX_LOG_TEMP_DIR: Directory for temporary log files (default: /tmp)
    NVRX_LOG_AGGREGATOR: Set to "1" to run aggregator as a separate service
    NVRX_LOG_MAX_FILE_SIZE_MB: Maximum size of temporary message files in MB before rotation (default: 10)
    NVRX_LOG_MAX_BACKUP_FILES: Maximum number of backup files to keep per rank (default: 5)
    
Note: File rotation is designed to be safe for the aggregator service. When files are rotated,
the aggregator will automatically read from both current and backup files to ensure no messages are lost.

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

Separate Aggregator Service:
    The aggregator can run as a separate service for reliable log collection:
    
    # Start aggregator service (step 0 in slurm)
    python -m nvidia_resiliency_ext.shared_utils.log_aggregator \
        --log-dir /path/to/logs \
        --temp-dir /path/to/temp \
        --wait-file /path/to/shutdown.signal
    
    # In training processes (step 1 in slurm)
    export NVRX_LOG_DIR=/path/to/logs
    export NVRX_LOG_TEMP_DIR=/path/to/temp
    export NVRX_LOG_AGGREGATOR=1
    ft_launcher ... your_training_script.py
"""

import re
import heapq
import logging
import os
import socket
import sys
import threading
import time
import queue
from datetime import datetime
from typing import Optional, Dict, List


log_pattern = re.compile(
    r"(?P<asctime>[\d-]+\s[\d:,]+) \[(?P<levelname>\w+)\] \[(?P<hostname>[\w.-]+)\] "
    r"\[workload:(?P<workload_rank>\d+)\((?P<workload_local_rank>\d+)\) infra:(?P<infra_rank>\d+)\((?P<infra_local_rank>\d+)\)\] "
    r"(?P<filename>[\w.]+):(?P<lineno>\d+) (?P<message>.+)"
)


class LogMessage:
    """Represents a log message."""

    def __init__(self, full_message: str):
        self.full_message = full_message
        self.hash_table = {}
        match = log_pattern.match(full_message)
        if match:
            log_fields = match.groupdict()
            for key, value in log_fields.items():
                if key == 'asctime':
                    # Convert asctime to a datetime object, then to a Unix timestamp
                    dt = datetime.strptime(value, '%Y-%m-%d %H:%M:%S,%f')
                    timestamp = int(dt.timestamp())
                    self.hash_table[key] = value
                else:
                    self.hash_table[key] = value

        if 'asctime' not in self.hash_table:
            current_datetime = datetime.now()
            self.hash_table['asctime'] = int(current_datetime.timestamp())

    def getts(self):
        return self.hash_table['asctime']

    def __str__(self):
        return self.full_message


class LogManager:
    """
    Log manager for large-scale LLM training.

    Supports both regular logging and distributed logging. When distributed logging
    is enabled (NVRX_DIST_LOG_DIR is set), each node logs independently to avoid
    overwhelming centralized logging systems. Local rank 0 acts as the node aggregator,
    collecting logs from all ranks on the same node and writing them to a per-node log file.

    Fork-safe: Child processes automatically disable aggregation to avoid conflicts.
    Service-based: Aggregator can run as a separate service for reliable log collection.
    """

    def __init__(self, log_dir: Optional[str] = None, temp_dir: Optional[str] = None):
        """
        Initialize the distributed log manager.

        Args:
            log_dir: Directory for log files. If None, uses NVRX_DIST_LOG_DIR env var
            temp_dir: Directory for temporary files. If None, uses NVRX_TEMP_DIR env var or /tmp
        """
        self._lock = threading.Lock()
        self._log_dict_queue = {}
        self._aggregator_thread = None
        self._stop_event = threading.Event()

        # Get hostname for node identification
        self._hostname = socket.gethostname()

        # Use hostname as node identifier
        self.node_id = self._hostname

        # Get distributed info once during initialization
        self._workload_rank = int(os.environ.get("RANK", "0")) if os.environ.get("RANK") else None
        self._workload_local_rank = (
            int(os.environ.get("LOCAL_RANK", "0")) if os.environ.get("LOCAL_RANK") else None
        )
        self._infra_rank = (
            int(os.environ.get("SLURM_PROCID", "0")) if os.environ.get("SLURM_PROCID") else None
        )
        self._infra_local_rank = (
            int(os.environ.get("SLURM_LOCALID", "0")) if os.environ.get("SLURM_LOCALID") else None
        )

        # Use NVRX_LOG_DEBUG environment variable to determine log level
        debug_enabled = os.environ.get("NVRX_LOG_DEBUG", "").lower() in ("1", "true", "yes", "on")
        self.log_level = logging.DEBUG if debug_enabled else logging.INFO

        # Set log directory
        self._log_dir = log_dir or os.environ.get("NVRX_LOG_DIR", None)

        # Check if running as aggregator service
        self._is_aggregator_service = os.environ.get("NVRX_LOG_AGGREGATOR", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        # Use configurable temporary directory for pending messages
        self._temp_dir = temp_dir or os.environ.get("NVRX_LOG_TEMP_DIR", "/tmp")
        # Use node_id to ensure all ranks on the same node use the same directory
        self._temp_dir = os.path.join(self._temp_dir, f"nvrx_log_{self.node_id}")
        os.makedirs(self._temp_dir, exist_ok=True)

        # Track file positions for each rank to avoid re-reading
        self._file_positions = {}

        # File rotation settings (in bytes)
        max_file_size_mb = int(os.environ.get("NVRX_LOG_MAX_FILE_SIZE_MB", "10"))
        self._max_msg_file_size = max_file_size_mb * 1024 * 1024  # Convert MB to bytes
        self._max_backup_files = int(
            os.environ.get("NVRX_LOG_MAX_BACKUP_FILES", "5")
        )  # Keep at most 5 backup files per rank

        # Create logger
        self._logger = self._setup_logger()

        # Validate configuration for log aggregator service
        if self._is_aggregator_service:
            if self._log_dir is None:
                raise RuntimeError("Log directory must be set for log aggregator service")
            self._start_aggregator()

    @property
    def distributed_logging_enabled(self) -> bool:
        """Check if distributed logging is enabled."""
        return self._log_dir is not None

    @property
    def workload_rank(self) -> Optional[int]:
        """Get the workload rank (from RANK env var)."""
        return self._workload_rank

    @property
    def workload_local_rank(self) -> Optional[int]:
        """Get the workload local rank (from LOCAL_RANK env var)."""
        return self._workload_local_rank

    @property
    def infra_rank(self) -> Optional[int]:
        """Get the infrastructure rank (from SLURM_PROCID env var)."""
        return self._infra_rank

    @property
    def infra_local_rank(self) -> Optional[int]:
        """Get the infrastructure local rank (from SLURM_LOCALID env var)."""
        return self._infra_local_rank

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

            # Use dynamic formatter with static hostname and dynamic rank info
            formatter = DynamicLogFormatter(
                self,
                fmt=f"%(asctime)s [%(levelname)s] [{self._hostname}] [workload:%(workload_rank)s(%(workload_local_rank)s) infra:%(infra_rank)s(%(infra_local_rank)s)] %(filename)s:%(lineno)d %(message)s",
            )
        else:
            # Simple logging to stderr or stdout
            if os.environ.get("NVRX_LOG_TO_STDOUT", "").lower() in ("1", "true", "yes", "on"):
                handler = logging.StreamHandler(sys.stdout)
            else:
                handler = logging.StreamHandler(sys.stderr)

            # Use dynamic formatter with static hostname and dynamic rank info
            formatter = DynamicLogFormatter(
                self,
                fmt=f"%(asctime)s [%(levelname)s] [{self._hostname}] [workload:%(workload_rank)s(%(workload_local_rank)s) infra:%(infra_rank)s(%(infra_local_rank)s)] %(filename)s:%(lineno)d %(message)s",
            )

        handler.setLevel(self.log_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

        return logger

    def _start_aggregator(self):
        """Start the log aggregator thread."""
        if self._aggregator_thread is not None:
            return

        self._aggregator_thread = threading.Thread(
            target=self._aggregator_loop, daemon=True, name=f"LogAggregator-{self._workload_rank}"
        )
        self._aggregator_thread.start()

    def _write_messages_to_file(self, messages, output):
        # Write messages to output
        for msg in messages:
            try:
                # The message is already formatted by the formatter, just write it
                output.write(msg.full_message + '\n')
                output.flush()
            except Exception as e:
                # Fallback to stderr if output fails
                sys.stderr.write(f"Log output error: {e}\n")
                sys.stderr.flush()

    def _merge_sort_streaming_lists(
        self, msg_dict: Dict[str, queue.SimpleQueue], heap: List
    ) -> list:
        # Initialize heap with the first log of each list
        heap_keys = {}
        i = 0
        for _, key, lm in heap:
            heap_keys[key] = i
            i += 1

        for key, msg_q in msg_dict.items():
            # print(f"msg_q {type(msg_q)}")
            if msg_q and msg_q.qsize() > 0:
                if key not in heap_keys:
                    lm = msg_q.get()
                    # push <ts, q, entry, log>
                    heapq.heappush(heap, (lm.getts(), key, lm))

        sorted_msgs = []
        while heap:
            ts, key, log_entry = heapq.heappop(heap)
            sorted_msgs.append(log_entry)
            msg_q = msg_dict[key]
            if msg_q.qsize() > 0:
                next_log = msg_q.get()
                heapq.heappush(heap, (next_log.getts(), key, next_log))
            else:
                break

        return sorted_msgs

    def _process_messages(self, output):
        # Check for pending messages from other ranks
        keep_processing = 1
        msg_dict = {}
        heap = []

        while keep_processing:
            if self._stop_event.is_set():
                keep_processing = 0
            # Check for pending messages from other ranks
            self._check_pending_messages()

            # Process queued messages
            with self._lock:
                messages = self._log_dict_queue.copy()
                for key, lm_q in self._log_dict_queue.items():
                    if key in msg_dict:
                        curr_q = msg_dict[key]
                        while not lm_q.empty():
                            curr_q.put(lm_q.get())
                    else:
                        msg_dict[key] = lm_q
                self._log_dict_queue.clear()

            sorted_msgs = self._merge_sort_streaming_lists(msg_dict, heap)
            self._write_messages_to_file(sorted_msgs, output)
            # Sleep briefly to avoid busy waiting
            time.sleep(0.025)

    def _aggregator_loop(self):
        """Main loop for the log aggregator."""
        # Create log directory if it doesn't exist
        os.makedirs(self._log_dir, exist_ok=True)

        # Setup per-node log file
        log_file = os.path.join(self._log_dir, f"node_{self.node_id}.log")
        output = open(log_file, 'a', buffering=1)  # Line buffered
        try:
            self._process_messages(output)
        finally:
            output.close()

    def _queue_message(self, message: str):
        """Queue a log message for aggregation."""

        # All ranks use file-based message passing
        # This ensures child processes can log even when they don't have the aggregator thread
        self._send_message_to_aggregator(message)

    def _send_message_to_aggregator(self, message: str):
        """Send a message to the aggregator using file-based communication."""
        # Create message directory in temp dir
        msg_dir = os.path.join(self._temp_dir, "pending_messages")
        os.makedirs(msg_dir, exist_ok=True)

        # Use a single message file per rank
        msg_file = os.path.join(msg_dir, f"rank_{self._workload_local_rank}.msg")

        # Check if file needs rotation
        if os.path.exists(msg_file):
            try:
                file_size = os.path.getsize(msg_file)
                if file_size > self._max_msg_file_size:
                    # Rotate the file with atomic operation
                    backup_file = f"{msg_file}.{int(time.time())}"
                    os.rename(msg_file, backup_file)
                    # Clean up old backup files
                    self._cleanup_old_backup_files(msg_dir, f"rank_{self._workload_local_rank}.msg")
            except (OSError, IOError) as e:
                # File might be being read by aggregator, skip rotation for now
                # Log this as it might indicate a real problem
                sys.stderr.write(f"File rotation error for {msg_file}: {e}\n")
                sys.stderr.flush()

        # Append message to the rank's message file
        with open(msg_file, 'a') as f:
            f.write(f"{message}\n")
            f.flush()  # Ensure message is written immediately

    def _check_pending_messages(self):
        """Check for pending messages from other ranks (aggregator only)."""
        if not self._is_aggregator_service:
            return

        msg_dir = os.path.join(self._temp_dir, "pending_messages")
        if not os.path.exists(msg_dir):
            return

        # Check if we can access the directory
        if not os.access(msg_dir, os.R_OK):
            return

        # Look for message files from all ranks (including this aggregator rank)
        for filename in os.listdir(msg_dir):
            if not filename.startswith('rank_') or not filename.endswith('.msg'):
                continue

            msg_file = os.path.join(msg_dir, filename)

            # Check for backup files for this rank
            backup_files = []
            base_filename = filename
            for backup_filename in os.listdir(msg_dir):
                if backup_filename.startswith(f"{base_filename}."):
                    backup_files.append(os.path.join(msg_dir, backup_filename))

            # Sort backup files by timestamp (oldest first)
            backup_files.sort(key=lambda f: os.path.getmtime(f))

            # Process backup files first (if any)
            for backup_file in backup_files:
                self._process_message_file(backup_file)

            # Process current file
            self._process_message_file(msg_file)

    def _process_message_file(self, msg_file: str):
        """Process a single message file (current or backup)."""
        try:
            file_size = os.path.getsize(msg_file)
        except FileNotFoundError as e:
            # File was deleted/renamed between discovery and processing
            # This can happen due to race conditions, but should be logged for debugging
            sys.stderr.write(f"File not found during processing {msg_file}: {e}\n")
            sys.stderr.flush()
            return
        except (IOError, OSError) as e:
            # Unexpected: Permission issues, disk problems, etc.
            # Log this as it might indicate a real problem
            sys.stderr.write(f"Unexpected error accessing {msg_file}: {e}\n")
            sys.stderr.flush()
            return

        # Get the last known position for this file
        last_position = self._file_positions.get(msg_file, 0)

        # If file hasn't grown, skip it
        if file_size <= last_position:
            return

        # Read new content from the file
        try:
            with open(msg_file, 'r') as f:
                f.seek(last_position)
                lines = f.readlines()
                file_size = f.tell()

        except FileNotFoundError as e:
            # File was deleted between size check and read
            sys.stderr.write(f"File not found during read {msg_file}: {e}\n")
            sys.stderr.flush()
            return
        except (IOError, OSError) as e:
            # File is being written by another process or other I/O error
            # Log this as it might indicate a real problem
            sys.stderr.write(f"IO error reading {msg_file}: {e}\n")
            sys.stderr.flush()
            return

        # Process each line
        log_msg_q = queue.SimpleQueue()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            log_msg = LogMessage(line)
            log_msg_q.put(log_msg)

        with self._lock:
            self._log_dict_queue[msg_file] = log_msg_q

        # Update the position for this file
        self._file_positions[msg_file] = file_size

    def _cleanup_old_backup_files(self, msg_dir: str, base_filename: str):
        """Clean up old backup files, keeping only the most recent ones."""
        # Find all backup files for this rank
        backup_files = []
        for filename in os.listdir(msg_dir):
            if filename.startswith(f"{base_filename}."):
                backup_files.append(os.path.join(msg_dir, filename))

        # Sort by modification time (oldest first)
        backup_files.sort(key=lambda f: os.path.getmtime(f))

        # Remove oldest files if we have too many
        if len(backup_files) > self._max_backup_files:
            for old_file in backup_files[: -self._max_backup_files]:
                try:
                    os.remove(old_file)
                except (OSError, IOError) as e:
                    # Log the error but don't fail the entire operation
                    sys.stderr.write(f"Failed to remove backup file {old_file}: {e}\n")
                    sys.stderr.flush()

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
        if self._is_aggregator_service:
            self._stop_event.set()
            if self._aggregator_thread:
                self._aggregator_thread.join()

        # Clean up temporary directory (only if this is the aggregator service)
        try:
            import shutil

            if self._is_aggregator_service:
                shutil.rmtree(self._temp_dir)
        except (OSError, IOError) as e:
            # Log cleanup errors for debugging
            sys.stderr.write(f"Cleanup error during shutdown: {e}\n")
            sys.stderr.flush()

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
            self.log_manager._queue_message(message=msg)
        except (OSError, IOError, RuntimeError):
            # Fallback to stderr if logging fails
            sys.stderr.write(f"Log handler error: {record.getMessage()}\n")
            sys.stderr.flush()


class DynamicLogFormatter(logging.Formatter):
    """Dynamic formatter that reads rank information from LogManager."""

    def __init__(self, log_manager, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.log_manager = log_manager

    def format(self, record):
        # Get rank info from LogManager, with fallback to "?" for None values
        record.workload_rank = (
            self.log_manager.workload_rank if self.log_manager.workload_rank is not None else "?"
        )
        record.workload_local_rank = (
            self.log_manager.workload_local_rank
            if self.log_manager.workload_local_rank is not None
            else "?"
        )
        record.infra_rank = (
            self.log_manager.infra_rank if self.log_manager.infra_rank is not None else "?"
        )
        record.infra_local_rank = (
            self.log_manager.infra_local_rank
            if self.log_manager.infra_local_rank is not None
            else "?"
        )

        # Use the parent's format method
        return super().format(record)


def setup_logger(log_dir=None, temp_dir=None, force_reset=False) -> logging.Logger:
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
        temp_dir: Optional directory path for temporary files. If None, uses NVRX_TEMP_DIR env var or /tmp.
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
        log_manager = LogManager(log_dir=log_dir, temp_dir=temp_dir)

        # Get the configured logger from the log manager
        logger = log_manager.logger

        # Store the log manager instance to prevent garbage collection
        # This ensures the aggregator thread keeps running
        setup_logger._log_manager = log_manager
    else:
        # Logger is already configured, just return the existing logger
        logger = logging.getLogger("nvrx")

    return logger
