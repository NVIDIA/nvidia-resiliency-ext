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

import contextlib
import functools
import inspect
import logging
import os
import time
import warnings

import packaging.version
import torch


def torch_older_than(version):
    torch_version = packaging.version.Version(torch.__version__)
    return torch_version < packaging.version.Version(version)


def format_exc(exc: BaseException):
    excs = [repr(exc)]
    while (exc := exc.__cause__) is not None:
        excs.append(repr(exc))
    return ' <- '.join(excs)


def format_exc_chain(exc: BaseException):
    """
    Format exception chain showing full stack trace for the first exception,
    and just the current exception name for chained exceptions.
    """
    # If this exception has no cause, it's the first exception - show full stack trace
    if exc.__cause__ is None:
        return repr(exc)

    # Otherwise, it's a chained exception - show only the current exception name
    return type(exc).__name__


def log_exc(rank_or_state, exc, name):
    """
    Log exception with clean chain formatting to reduce log volume.

    Args:
        rank_or_state: Rank number or state object
        exc: Exception to log
        name: Name identifier for the exception

    Returns:
        str: Formatted exception message
    """
    return log_exc_controlled(rank_or_state, exc, name)


def log_exc_controlled(rank_or_state, exc, name):
    """
    Controlled exception logging with clean chain formatting.

    Args:
        rank_or_state: Rank number or state object
        exc: Exception to log
        name: Name identifier for the exception

    Returns:
        str: Formatted exception message
    """
    if isinstance(rank_or_state, int):
        rank = rank_or_state
    else:
        rank = rank_or_state.rank

    formatted_exc = format_exc_chain(exc)
    return f'{rank=} {name}: {formatted_exc}'


@contextlib.contextmanager
def _log_exec(target, offset=3):
    stack = inspect.stack()
    caller_frame = stack[offset]
    caller_modulename = inspect.getmodulename(caller_frame.filename)

    log = logging.getLogger(caller_modulename)
    rank = int(os.getenv('RANK', '0'))

    if callable(target):
        name = f'{target.__module__}.{target.__qualname__}'
    else:
        name = target

    log.debug(f'{rank=} starts execution: {name}')
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        log.debug(f'{rank=} ends execution: {name} [{elapsed=:.4e}]')


def log_exec(target):
    if callable(target):

        @functools.wraps(target)
        def wrapper(*args, **kwargs):
            with _log_exec(target):
                return target(*args, **kwargs)

        return wrapper
    else:
        return _log_exec(target)


def find_nearest_handler(logger, handler_cls):
    current_logger = logger
    while current_logger:
        for handler in current_logger.handlers:
            if isinstance(handler, handler_cls):
                return handler
        if not current_logger.propagate or not current_logger.parent:
            break
        current_logger = current_logger.parent
    return None


class Logging:

    @classmethod
    def initialize(cls):
        parent_module_name = cls.__module__.split('.')[-2]
        logger = logging.getLogger(parent_module_name)
        logger.propagate = False

        parent = logger.parent
        while parent is not None:
            stream_handlers = [
                handler for handler in parent.handlers if type(handler) is logging.StreamHandler
            ]
            if stream_handlers:
                break
            parent = parent.parent
        else:
            stream_handlers = []

        if stream_handlers:
            for handler in stream_handlers:
                logger.addHandler(handler)
        else:
            warnings.warn('logging not initialized, logs are disabled')
            logger.addHandler(logging.NullHandler())

        level = logger.getEffectiveLevel()
        handlers = logger.handlers
        logger.debug(f'logging initialized {level=} {handlers=}')

    @classmethod
    def deinitialize(cls):
        parent_module_name = cls.__module__.split('.')[-2]
        logger = logging.getLogger(parent_module_name)
        logger.debug('deinitialize logging')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
