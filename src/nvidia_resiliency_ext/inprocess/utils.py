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
    return repr(exc)


def format_rank_set_verbose(ranks):
    """
    Format a set of ranks for logging using range compression (e.g., "1-3, 5, 7-9").

    Args:
        ranks: Set or list of rank numbers

    Returns:
        str: Formatted rank set string with ranges
    """
    if not ranks:
        return "{}"

    # Convert to sorted list of unique ranks
    sorted_ranks = sorted(set(ranks))

    # Compress consecutive ranks into ranges
    ranges = []
    if sorted_ranks:
        start = end = sorted_ranks[0]

        for n in sorted_ranks[1:]:
            if n == end + 1:
                end = n
            else:
                ranges.append(f"{start}-{end}" if start != end else str(start))
                start = end = n

        ranges.append(f"{start}-{end}" if start != end else str(start))

    result = ", ".join(ranges)
    return f"{{{result}}}"


def format_rank_set_brief(ranks, max_show=8):
    """
    Format a set of ranks for logging, showing partial ranks and total count for large sets.

    Args:
        ranks: Set or list of rank numbers
        max_show: Maximum number of ranks to show before truncating

    Returns:
        str: Formatted rank set string
    """
    if not ranks:
        return "{}"

    rank_count = len(ranks)
    sorted_ranks = sorted(ranks)

    # Show all ranks if count is small enough
    if rank_count <= max_show:
        return f"{{{', '.join(map(str, sorted_ranks))}}}"

    # For large sets, show first few and last few with total count
    first_ranks = sorted_ranks[:4]
    last_ranks = sorted_ranks[-4:]
    return f"{{{', '.join(map(str, first_ranks))}...{', '.join(map(str, last_ranks))} (total: {rank_count})}}"


def format_rank_set(ranks):
    """
    Format a set of ranks for logging using either range compression or partial display.

    The formatting method is controlled by the environment variable
    'NVRX_LOG_RANK_FORMAT_VERBOSE':
    - True: Use range compression (e.g., "1-3, 5, 7-9")
    - False/not set: Use partial display with total count for large sets

    Args:
        ranks: Set or list of rank numbers

    Returns:
        str: Formatted rank set string

    Environment Variables:
        NVRX_LOG_RANK_FORMAT_VERBOSE: Controls the formatting method
            (True for verbose ranges, False/not set for partial display)
    """
    # Get verbose mode from environment variable, default to False (partial display)
    verbose_mode = os.getenv('NVRX_LOG_RANK_FORMAT_VERBOSE', 'false').lower() in (
        'true',
        '1',
        'yes',
        'on',
    )

    if verbose_mode:
        return format_rank_set_verbose(ranks)
    else:
        # Default to brief display mode
        return format_rank_set_brief(ranks)


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
    if isinstance(rank_or_state, int):
        rank = rank_or_state
    else:
        rank = rank_or_state.rank

    formatted_exc = format_exc(exc)
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
