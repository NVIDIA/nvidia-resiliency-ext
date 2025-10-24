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
Simple utility to install a global exception handler that writes exceptions to error files.

This is training framework agnostic and works with any application using RankMonitorClient.
The error file format is compatible with torch.distributed.elastic.
"""

import json
import logging
import os
import sys
import time
import traceback

logger = logging.getLogger(__name__)


def install_exception_handler():
    """
    Install a global exception handler that writes uncaught exceptions to error files.

    This function sets up sys.excepthook to capture any uncaught exception and write it
    to the error file specified by TORCHELASTIC_ERROR_FILE environment variable.

    The error file format matches what torch.distributed.elastic expects, so the ft_launcher
    can read and display full exception tracebacks.

    Returns:
        bool: True if handler was installed, False otherwise
    """
    error_file = os.environ.get('TORCHELASTIC_ERROR_FILE')

    if not error_file:
        logger.debug(
            "TORCHELASTIC_ERROR_FILE not set. Exception handler not installed. "
            "This is expected if not running under ft_launcher."
        )
        return False

    # Save the original excepthook
    original_excepthook = sys.excepthook

    def error_file_excepthook(exc_type, exc_value, exc_traceback):
        """Custom exception hook that writes to error file."""
        # Format the traceback
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_string = ''.join(tb_lines)

        exc_type_name = exc_type.__name__
        exc_message = str(exc_value)

        # Create error data in the format expected by torch.distributed.elastic
        error_data = {
            "message": {
                "message": f"{exc_type_name}: {exc_message}",
                "extraInfo": {
                    "py_callstack": tb_string,
                    "timestamp": str(int(time.time())),
                },
            }
        }

        # Write to error file
        try:
            with open(error_file, 'w') as f:
                json.dump(error_data, f)
            logger.debug(f"Wrote exception to error file: {error_file}")
        except Exception as e:
            logger.error(f"Failed to write exception to error file {error_file}: {e}")

        # Call the original excepthook to maintain normal error handling
        original_excepthook(exc_type, exc_value, exc_traceback)

    # Install our custom excepthook
    sys.excepthook = error_file_excepthook

    logger.info(f"Installed global exception handler. Exceptions will be written to: {error_file}")

    return True
