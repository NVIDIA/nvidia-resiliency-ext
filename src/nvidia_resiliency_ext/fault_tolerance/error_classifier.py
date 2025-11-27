# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Utility to check error files for non-retryable exceptions.

This is used by the rendezvous logic to determine if a node should be marked as unhealthy
based on previous exception patterns.
"""

import json
import logging
from typing import List

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

logger = logging.getLogger(LogConfig.name)


def load_non_retryable_exception_patterns(exception_file: str) -> List[str]:
    """
    Load non-retryable exception patterns from a file.

    Args:
        exception_file: Path to file containing exception patterns (one per line)

    Returns:
        List of exception string patterns to match
    """
    # Let exceptions propagate - if the user configured a file path, it should be valid
    with open(exception_file, 'r') as f:
        # Read lines, strip whitespace, skip empty lines and comments
        patterns = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                patterns.append(line)
        logger.info(
            f"Loaded {len(patterns)} non-retryable exception patterns from {exception_file}"
        )
        return patterns


def is_non_retryable_exception(
    exc_type_name: str, exc_message: str, traceback_str: str, patterns: List[str]
) -> bool:
    """
    Check if an exception matches any non-retryable patterns.

    Args:
        exc_type_name: Exception type name (e.g., "RuntimeError")
        exc_message: Exception message
        traceback_str: Full traceback string
        patterns: List of patterns to match against

    Returns:
        True if exception matches any non-retryable pattern
    """
    # Combine all exception info for matching
    full_exception_text = f"{exc_type_name}: {exc_message}\n{traceback_str}"

    # Check if any pattern matches
    for pattern in patterns:
        if pattern in full_exception_text:
            logger.warning(
                f"Exception matches non-retryable pattern '{pattern}': "
                f"{exc_type_name}: {exc_message}"
            )
            return True

    return False


def check_error_file_for_non_retryable_exception(error_file_path: str, patterns: List[str]) -> bool:
    """
    Check if an error file contains a non-retryable exception.

    This is called by the launcher to check error files from failed workers.
    Caller is expected to verify file exists and patterns are non-empty.

    Args:
        error_file_path: Path to the error file to check (must exist)
        patterns: List of non-retryable exception patterns to match against (non-empty)

    Returns:
        True if the error file contains a non-retryable exception, False otherwise
        (also returns False if the error file is malformed or unreadable)
    """
    # Defensive: handle corrupted/malformed worker error files gracefully
    try:
        with open(error_file_path, 'r') as f:
            error_data = json.load(f)

        # Extract exception information
        message = error_data.get('message', {})
        if isinstance(message, dict):
            exc_message = message.get('message', '')
            extra_info = message.get('extraInfo', {})
            traceback_str = extra_info.get('py_callstack', '')

            # Extract exception type from message (format: "ExceptionType: message")
            exc_type_name = exc_message.split(':', 1)[0] if ':' in exc_message else ''

            return is_non_retryable_exception(exc_type_name, exc_message, traceback_str, patterns)
    except Exception as e:
        logger.error(f"Failed to check error file {error_file_path}: {e}")
        return False
