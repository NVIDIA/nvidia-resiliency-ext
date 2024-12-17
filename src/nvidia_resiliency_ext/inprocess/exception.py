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


class RestartError(Exception):
    r'''
    Base :py:exc:`Exception` for exceptions raised by
    :py:class:`inprocess.Wrapper`.
    '''

    pass


class RestartAbort(BaseException):
    r'''
    A terminal Python :py:exc:`BaseException` indicating that the
    :py:class:`inprocess.Wrapper` should be aborted immediately, bypassing any
    further restart attempts.
    '''

    pass


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
