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

import gc
import logging
from contextlib import contextmanager
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch

U = TypeVar("U")
V = TypeVar("V")


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
def debug_time(
    name: str, logger: Optional[logging.Logger] = None, threshold: float = float("-inf"), level=None
):
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
                level = logging.DEBUG if threshold == float("-inf") else logging.WARNING
            last_logger.log(level, f"{stacked_name} took {result:.4f}s")


def debug_msg(msg: str):
    with logger_stack(None, None) as (stacked_name, last_logger):
        last_logger.debug(f"{stacked_name} {msg}")


def preload_tensors(state_dict: Dict, non_blocking=True):
    """Preload tensors in state_dict to host memory through CPU memory
    Args:
        state_dict (Dict): state dictionary to checkpoint with torch.Tensors
        non_blocking (bool, optional): knob to enable pinned D2H memcpy. Default is True.
    """

    def preload_tensor(in_var):
        if isinstance(in_var, torch.Tensor):
            return in_var.detach().to("cpu", non_blocking=non_blocking)
        else:
            return in_var

    state_dict = dict_list_map_outplace(preload_tensor, state_dict)
    return state_dict


@contextmanager
def _disable_gc():
    """Temporarily disables GC."""
    gc_enabled = gc.isenabled()
    try:
        if gc_enabled:
            gc.disable()
        yield
    finally:
        if gc_enabled:
            gc.enable()


def wrap_for_async(fn):
    def wrapped(state_dict, *args, **kwargs):
        with _disable_gc():
            fn(state_dict, *args, **kwargs)

    return wrapped


# Dict utils copied from megatron.core.dist_checkpointing.dict_utils.py
def diff(x1: Any, x2: Any, prefix: Tuple = ()) -> Tuple[list, list, list]:
    """Recursive diff of dicts.

    Args:
        x1 (object): left dict
        x2 (object): right dict
        prefix (tuple): tracks recursive calls. Used for reporting differing keys.

    Returns:
        Tuple[list, list, list]: tuple of:
            - only_left: Prefixes present only in left dict
            - only_right: Prefixes present only in right dict
            - mismatch: values present in both dicts but not equal across dicts.
                For tensors equality of all elems is checked.
                Each element is a tuple (prefix, type of left value, type of right value).
    """
    mismatch = []
    if isinstance(x1, dict) and isinstance(x2, dict):
        only_left = [prefix + (k,) for k in x1.keys() - x2.keys()]
        only_right = [prefix + (k,) for k in x2.keys() - x1.keys()]
        for k in x2.keys() & x1.keys():
            _left, _right, _mismatch = diff(x1[k], x2[k], prefix + (k,))
            only_left.extend(_left)
            only_right.extend(_right)
            mismatch.extend(_mismatch)
    elif isinstance(x1, list) or isinstance(x1, tuple) or isinstance(x1, np.ndarray):
        assert isinstance(x1, type(x2))
        only_left = list(range(len(x1) - 1, len(x2) - 1, -1))
        only_right = list(range(len(x1) - 1, len(x2) - 1, -1))
        for i, (v1, v2) in enumerate(zip(x1, x2)):
            _left, _right, _mismatch = diff(v1, v2, prefix + (i,))
            only_left.extend(_left)
            only_right.extend(_right)
            mismatch.extend(_mismatch)
    else:
        only_left = []
        only_right = []
        if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
            if x1.device != x2.device:
                _is_mismatch = not torch.all(x1.cpu() == x2.cpu())
            else:
                _is_mismatch = not torch.all(x1 == x2)
        # TODO: change with concrete type that has both replica_id and data attrs
        elif hasattr(x1, 'replica_id') and hasattr(x2, 'replica_id'):
            assert isinstance(x1, type(x2))
            only_left, only_right, mismatch = diff(
                x1.data, x2.data, prefix + (type(x1),)
            )  # type: ignore
            _is_mismatch = False
        else:
            try:
                _is_mismatch = bool(x1 != x2)
            except RuntimeError:
                _is_mismatch = True

        if _is_mismatch:
            mismatch.append((prefix, type(x1), type(x2)))

    return only_left, only_right, mismatch


def dict_list_map_outplace(f: Callable[[U], V], x: Union[Dict, List, U]) -> Union[Dict, List, V]:
    """Maps dicts and lists *out-of-place* with a given function."""
    if isinstance(x, dict):
        return {k: dict_list_map_outplace(f, v) for k, v in x.items()}
    elif isinstance(x, list):
        return [dict_list_map_outplace(f, v) for v in x]
    else:
        return f(x)
