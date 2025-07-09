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
NVRx Singleton Logger Module

This module provides a singleton logger for the NVRx package with configurable
output destinations and log levels.

Environment Variables:
    NVRX_DEBUG: Set to "1", "true", "yes", or "on" to enable DEBUG level logging
    NVRX_NULL_HANDLER: Set to "1", "true", "yes", or "on" to disable logging 
                       (allows applications to configure their own handlers)
    NVRX_LOGFILE: Path to log file for file-based logging

Usage:
    from nvidia_resiliency_ext.shared_utils.log import logger
    
    logger.info("Application started")
    logger.debug("Debug information")
    logger.error("Error occurred")

File Logging:
    export NVRX_LOGFILE="/path/to/logfile.log"    
"""

import logging
import os
import socket
import sys


def get_log_level() -> int:
    """
    Determine the log level based on NVRX_DEBUG environment variable.
    Returns logging.DEBUG if NVRX_DEBUG is set to a truthy value,
    otherwise returns logging.INFO
    """
    debug_env = os.environ.get("NVRX_DEBUG", "").lower()
    return logging.DEBUG if debug_env in ("1", "true", "yes", "on") else logging.INFO


def setup_logger(logfile=None) -> logging.Logger:
    """
    Setup a single shared logger for the nvrx package.
    If NVRX_NULL_HANDLER is set to a truthy value, only adds a NullHandler,
    allowing applications to configure their own handlers.
    Otherwise, configures logger to a file or stdout/stderr.

    Args:
        logfile: Optional file path for logging. If None, checks NVRX_LOGFILE env var.

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create our logger with distinct namespace
    logger = logging.getLogger("nvrx")

    # Return existing logger if already configured
    if logger.handlers:
        return logger

    # Prevent logs from propagating to parent loggers
    logger.propagate = False

    # Check if NullHandler should be used
    null_handler_env = os.environ.get("NVRX_NULL_HANDLER", "").lower()
    if null_handler_env in ("1", "true", "yes", "on"):
        logger.addHandler(logging.NullHandler())
        return logger

    # Set log level based on environment variable
    log_level = get_log_level()
    logger.setLevel(log_level)

    # Determine logfile from parameter or environment variable
    logfile = logfile or os.environ.get("NVRX_LOGFILE", None)

    if logfile:
        handler = logging.FileHandler(filename=logfile)
    else:
        handler = logging.StreamHandler() # Defaults to stderr.

    handler.setLevel(log_level)

    hostname = socket.gethostname()
    # Set format with process ID, filename, and line number
    formatter = logging.Formatter(
        fmt=f"%(asctime)s [%(levelname)s] [{hostname}:%(process)5s] %(filename)s:%(lineno)d %(message)s"
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger

# Create and export a single shared logger instance
logger = setup_logger()
