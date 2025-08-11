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

import sys


class RestartError(Exception):
    r'''
    Base :py:exc:`Exception` for exceptions raised by
    :py:class:`inprocess.Wrapper`.
    '''

    pass


def _restart_abort_excepthook(exc_type, exc_value, exc_traceback):
    """
    Custom exception hook to handle RestartAbort automatically.

    When RestartAbort reaches the top level, this hook ensures it exits
    with the correct exit code (130) so that auto_restart.py recognizes
    it as a clean exit and doesn't restart the process.
    """
    if exc_type is RestartAbort:
        # Exit with the clean abort code to prevent auto-restart
        sys.exit(exc_value.exit_code)
    else:
        # Call the original excepthook for other exceptions
        sys.__excepthook__(exc_type, exc_value, exc_traceback)


class RestartAbort(BaseException):
    r'''
    A terminal Python :py:exc:`BaseException` indicating that the
    :py:class:`inprocess.Wrapper` should be aborted immediately, bypassing any
    further restart attempts.

    This exception uses exit code 130 (SIGINT-like) to signal a clean abort
    that should not trigger process restart in auto_restart.py.

    The exception hook automatically handles exit code 130 when this exception
    reaches the top level, so no manual handling is required in main programs.
    '''

    CLEAN_EXIT_CODE = 130  # SIGINT-like exit code for clean abort

    def __init__(self, message: str):
        super().__init__(message)
        self.exit_code = self.CLEAN_EXIT_CODE


# Set the custom exception hook to automatically handle RestartAbort
# This ensures that when RestartAbort reaches the top level, it exits
# with code 130 instead of the default exit code 1
sys.excepthook = _restart_abort_excepthook


class HealthCheckError(RestartError):
    r'''
    :py:exc:`RestartError` exception to indicate that
    :py:class:`inprocess.health_check.HealthCheck` raised errors, and execution
    shouldn't be restarted on this distributed rank.
    '''

    pass


class InternalError(RestartError):
    r'''
    :py:class:`inprocess.Wrapper` internal error.
    '''

    pass


class TimeoutError(RestartError):
    r'''
    :py:class:`inprocess.Wrapper` timeout error.
    '''

    pass
