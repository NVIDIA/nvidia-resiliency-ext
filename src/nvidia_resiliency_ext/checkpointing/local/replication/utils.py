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


import logging
from contextlib import contextmanager
from time import time
from typing import Optional


def zip_strict(*args):
    """
    Alternative to Python's builtin zip(..., strict=True) (available in 3.10+).
    Apart from providing functionality in earlier versions of Python is also more verbose.
    (Python's zip does not print lengths, only which iterable has finished earlier)
    """
    args = [list(a) for a in args]
    lens = [len(a) for a in args]
    assert len(set(lens)) <= 1, f"Tried to zip iterables of unequal lengths: {lens}!"
    return zip(*args)

fallback_logger = logging.getLogger(__name__)
__LOGGER_NAME_STACK = []
__LOGGER_STACK = []

@contextmanager
def logger_stack(name: Optional[str] = None, current_logger: Optional[logging.Logger] = None):
    if name:
        __LOGGER_NAME_STACK.append(name)
    if current_logger:
        __LOGGER_STACK.append(current_logger)
        last_logger = current_logger
    elif __LOGGER_STACK:
        last_logger = __LOGGER_STACK[-1]
    else:
        last_logger = fallback_logger
    try:
        yield ".".join(__LOGGER_NAME_STACK), last_logger
    finally:
        if name and __LOGGER_NAME_STACK:
            __LOGGER_NAME_STACK.pop(-1)
        if current_logger and __LOGGER_STACK:
            __LOGGER_STACK.pop(-1)

@contextmanager
def debug_time(name: str, logger: Optional[logging.Logger] = None, threshold: float = float("-inf"), level = None):
    """Simple context manager for timing functions/code blocks.

    Args:
        name: Label for what we're measuring the execution time of
        logger: What logger should be used to print the debug message.
                Note that not specifying logger means using the lowest specified logger in the execution stack.
        threshold: Do not print debug message if took less than `threshold` seconds.
        level: What debugging level to use. Default: DEBUG if `threshold` not specified, WARNING otherwise.
    """
    with logger_stack(name, logger) as (stacked_name, last_logger):
        start = time()
        try:
            yield
        finally:
            result = time() - start
            if result < threshold:
                return
            if level is None:
                level = (logging.DEBUG if threshold == float("-inf") else logging.WARNING)
            last_logger.log(level, f"{stacked_name} took {result:.4f}s")

def debug_msg(msg: str):
    with logger_stack(None, None) as (stacked_name, last_logger):
        last_logger.debug(f"{stacked_name} {msg}")
