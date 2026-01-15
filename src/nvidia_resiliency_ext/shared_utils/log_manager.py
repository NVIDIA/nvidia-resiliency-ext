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
- Environment-driven behavior: NVRX_NODE_LOCAL_TMPDIR controls distributed vs regular logging
- Per-node aggregation: When distributed logging is enabled, a separate aggregator service does log aggregation
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
    NVRX_LOG_DEBUG: Set to "1", "true", "yes", or "on" to enable DEBUG level logging (default: INFO)
    NVRX_LOG_TO_STDOUT: Set to "1" to log to stdout instead of stderr
    NVRX_NODE_LOCAL_TMPDIR: Directory for temporary log files
    NVRX_LOG_MAX_FILE_SIZE_KB: Maximum size of temporary message files in KB before rotation (default: 10240 KB = 10 MB)
    NVRX_LOG_MAX_LOG_FILES: Maximum number of log files to keep per rank (default: 4)
    
Note: File rotation is designed to be safe for the aggregator service. When files are rotated,
the aggregator will automatically read from both current and backup files to ensure no messages are lost.

Usage:
    # In main script (launcher.py)
    from nvidia_resiliency_ext.shared_utils.log_manager import setup_logger
    logger = setup_logger()  # Call once at startup
    
    # In other modules
    import logging
    logger = logging.getLogger(LogConfig.name)
    logger.info("Training started")
    logger.debug("Debug information")
    logger.error("Error occurred")
    logger.warning("Warning message")
    logger.critical("Critical error")

Forking Support:
    The logger is designed to work safely with process forking. When using fork():
    
    # In parent process
    from nvidia_resiliency_ext.shared_utils.log_manager import setup_logger
    logger = setup_logger()  # Setup before forking
    logger.info("Parent process logging")
    
    # Fork child process
    pid = os.fork()
    if pid == 0:
        # In child process - logger will work normally
        import logging
        logger = logging.getLogger(LogConfig.name)
        logger.info("Child process logging")
    else:
        # Parent continues normally
        logger.info("Parent continues")
    
    All ranks use file-based message passing, ensuring child processes can log
    even when they don't inherit the aggregator thread from the parent.

Separate Aggregator Service, see log_aggregator.py for details.
"""

import logging
import os
import socket
import sys
from typing import Optional

from nvidia_resiliency_ext.shared_utils.log_node_local_tmp import (
    DynamicLogFormatter,
    NodeLocalTmpLogHandler,
)
from nvidia_resiliency_ext.shared_utils.os_utils import validate_directory


class LogConfig:
    """Utility class for log configuration."""

    name = "nvrx"
    _aggr_file_prefix = "nvrx_log_"

    @classmethod
    def get_node_id(cls):
        # Use hostname as node identifier
        return socket.gethostname()

    @classmethod
    def get_log_file(cls):
        return f"{cls._aggr_file_prefix}{cls.get_node_id()}.log"

    @classmethod
    def get_node_local_tmp_dir(cls, node_local_tmp_dir=None):
        # Use configurable temporary directory for pending messages
        return node_local_tmp_dir or os.environ.get("NVRX_NODE_LOCAL_TMPDIR")

    @classmethod
    def get_max_file_size(cls, file_size_kb=None) -> int:
        if file_size_kb is None:
            file_size_kb = int(os.environ.get("NVRX_LOG_MAX_FILE_SIZE_KB", "10240"))
        return file_size_kb * 1024  # Convert KB to bytes

    @classmethod
    def get_max_log_files(cls) -> int:
        return int(
            os.environ.get("NVRX_LOG_MAX_LOG_FILES", "4")
        )  # Keep default 4 backup files per rank

    @classmethod
    def get_workload_rank(cls):
        return int(os.environ.get("RANK", "0")) if os.environ.get("RANK") else None

    @classmethod
    def get_workload_local_rank(cls):
        return int(os.environ.get("LOCAL_RANK", "0")) if os.environ.get("LOCAL_RANK") else None

    @classmethod
    def get_infra_rank(cls):
        """
        Get infrastructure rank with SLURM job array support.

        For SLURM job arrays, calculates rank as:
            array_task_id * nnodes_per_array_task + slurm_procid

        Returns:
            Infrastructure rank or None if not available
        """
        # Check if we're in a SLURM job array
        slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        slurm_procid = os.environ.get("SLURM_PROCID")

        if slurm_array_task_id is not None and slurm_procid is not None:
            # SLURM job array deployment
            array_task_id = int(slurm_array_task_id)
            proc_id = int(slurm_procid)

            # Get number of nodes per array task
            nnodes_per_array = os.environ.get("SLURM_NNODES", os.environ.get("SLURM_JOB_NUM_NODES"))
            if nnodes_per_array is not None:
                nnodes = int(nnodes_per_array)
                return array_task_id * nnodes + proc_id
            # If we can't get nnodes, fall through to regular SLURM_PROCID

        # Regular SLURM deployment or fallback
        return int(slurm_procid) if slurm_procid else None

    @classmethod
    def get_infra_local_rank(cls):
        return (
            int(os.environ.get("SLURM_LOCALID", "0")) if os.environ.get("SLURM_LOCALID") else None
        )

    @classmethod
    def get_log_level(cls):
        # Use NVRX_LOG_DEBUG environment variable to determine log level
        debug_enabled = os.environ.get("NVRX_LOG_DEBUG", "").lower() in ("1", "true", "yes", "on")
        return logging.DEBUG if debug_enabled else logging.INFO

    @classmethod
    def get_process_name(cls, proc_name: Optional[str] = None) -> str:
        return proc_name if proc_name is not None else str(os.getpid())

    @classmethod
    def get_log_to_stdout_cfg(cls) -> bool:
        return os.environ.get("NVRX_LOG_TO_STDOUT", "").lower() in ("1", "true", "yes", "on")


class LogManager:
    """
    Log manager for large-scale LLM training.

    Supports both regular logging and node local temporary logging. When node local temporary logging
    is enabled (NVRX_NODE_LOCAL_TMPDIR is set), each node logs independently to avoid
    overwhelming centralized logging systems. Local rank 0 acts as the node aggregator,
    collecting logs from all ranks on the same node and writing them to a per-node log file.

    Fork-safe: Child processes automatically disable aggregation to avoid conflicts.
    Service-based: Aggregator can run as a separate service for reliable log collection.
    """

    def __init__(
        self,
        node_local_tmp_dir: Optional[str] = None,
        node_local_tmp_prefix: str = None,
        log_file: Optional[str] = None,
    ):
        self._node_local_tmp_prefix = LogConfig.get_process_name(node_local_tmp_prefix)
        # Get distributed info once during initialization
        self._workload_rank = LogConfig.get_workload_rank()
        self._workload_local_rank = LogConfig.get_workload_local_rank()
        self._infra_rank = LogConfig.get_infra_rank()
        self._infra_local_rank = LogConfig.get_infra_local_rank()

        # Use NVRX_LOG_DEBUG environment variable to determine log level
        self.log_level = LogConfig.get_log_level()

        # Use configurable temporary directory for pending messages
        self._node_local_tmp_dir = LogConfig.get_node_local_tmp_dir(node_local_tmp_dir)

        # File rotation settings (in bytes)
        self._max_msg_file_size = LogConfig.get_max_file_size()
        self._max_backup_files = LogConfig.get_max_log_files()
        self._log_to_stdout = LogConfig.get_log_to_stdout_cfg()

        # Optional log file (if specified, logs go to file instead of console)
        self._log_file = log_file

        # Create logger
        self._logger = self._setup_logger()

    @property
    def node_local_tmp_logging_enabled(self) -> bool:
        """Check if node local temporary logging is enabled."""
        return self._node_local_tmp_dir is not None

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
        logger = logging.getLogger(LogConfig.name)
        logger.setLevel(self.log_level)

        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Determine base format (same for all handlers)
        base_fmt = (
            "%(asctime)s [%(levelname)s] "
            "[{node}] "
            "[workload:%(workload_rank)s(%(workload_local_rank)s) "
            "infra:%(infra_rank)s(%(infra_local_rank)s)] "
            "%(filename)s:%(lineno)d %(message)s"
        ).format(node=LogConfig.get_node_id())

        # For file handler, prepend rank prefix (like srun -l)
        if self._log_file:
            infra_rank = self.infra_rank if self.infra_rank is not None else '?'
            world_size = os.environ.get('WORLD_SIZE')
            if world_size and str(infra_rank).isdigit():
                width = len(str(int(world_size) - 1))
                rank_prefix = f'{str(infra_rank):>{width}}: '
            else:
                rank_prefix = f'{infra_rank}: '
            fmt = rank_prefix + base_fmt
        else:
            fmt = base_fmt

        # Create handler: file, node-local-tmp, or console (mutually exclusive)
        if self._log_file:
            # File handler
            handler = logging.FileHandler(self._log_file, mode='a')
        elif self.node_local_tmp_logging_enabled:
            # Node-local temporary logging
            os.makedirs(self._node_local_tmp_dir, exist_ok=True)
            validate_directory(self._node_local_tmp_dir)
            handler = NodeLocalTmpLogHandler(
                self.workload_local_rank,
                self._node_local_tmp_dir,
                self._max_msg_file_size,
                self._max_backup_files,
                self._node_local_tmp_prefix,
            )
        else:
            # Console handler (stderr or stdout)
            if self._log_to_stdout:
                handler = logging.StreamHandler(sys.stdout)
            else:
                handler = logging.StreamHandler(sys.stderr)

        # Create formatter and apply to handler
        formatter = DynamicLogFormatter(
            self.workload_rank,
            self.workload_local_rank,
            self.infra_rank,
            self.infra_local_rank,
            fmt=fmt,
        )
        handler.setFormatter(formatter)
        handler.setLevel(self.log_level)
        logger.addHandler(handler)
        logger.propagate = False

        return logger

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


def setup_logger(
    node_local_tmp_dir=None,
    force_reset=False,
    node_local_tmp_prefix: str = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Setup the distributed logger.

    This function configures the standard Python logger "nvrx" with appropriate
    handlers for distributed logging. It's safe to call multiple times - if the
    logger is already configured, it won't be reconfigured unless force_reset=True.

    The expectation is that this function is called once at the start of the program,
    and then the logger is used throughout the program i.e. its a singleton.

    The logger automatically adapts to distributed or regular mode based on
    whether NVRX_NODE_LOCAL_TMPDIR is set. If set, enables distributed logging
    with aggregation. If not set, logs go directly to stderr/stdout.

    The logger is fork-safe: all ranks use file-based message passing to ensure
    child processes can log even when they don't inherit the aggregator thread.

    Args:
        node_local_tmp_dir: Optional directory path for temporary files. If None, uses NVRX_NODE_LOCAL_TMPDIR env var.
        force_reset: If True, force reconfiguration even if logger is already configured.
                    Useful for subprocesses that need fresh logger setup.
        node_local_tmp_prefix: Optional prefix for log files (e.g. "ftlauncher").
        log_file: Optional path to log file. When specified, logs are written to this file
                 with rank prefixes (like srun -l) instead of console. All processes write
                 to the same file using append mode for safe concurrent writes.

    Returns:
        logging.Logger: Configured logger instance

    Example:
        # In main script (launcher.py) or training subprocess
        from nvidia_resiliency_ext.shared_utils.log_manager import setup_logger
        logger = setup_logger()

        # With log file for consolidated logging across all ranks/nodes
        logger = setup_logger(log_file="/path/to/base.log")

        # In subprocesses that need fresh logger setup
        logger = setup_logger(force_reset=True)

        # In other modules
        import logging
        logger = logging.getLogger(LogConfig.name)
        logger.info("Some message")
    """
    # Check if the nvrx logger is already configured
    logger = logging.getLogger(LogConfig.name)

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
        log_manager = LogManager(
            node_local_tmp_dir=node_local_tmp_dir,
            node_local_tmp_prefix=node_local_tmp_prefix,
            log_file=log_file,
        )

        # Get the configured logger from the log manager
        logger = log_manager.logger

        # Store the log manager instance to prevent garbage collection
        # This ensures the aggregator thread keeps running
        setup_logger._log_manager = log_manager
    else:
        # Logger is already configured, just return the existing logger
        logger = logging.getLogger(LogConfig.name)

    return logger
