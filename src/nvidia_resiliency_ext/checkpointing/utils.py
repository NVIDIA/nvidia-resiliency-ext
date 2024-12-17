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
import torch
import gc 

from typing import Callable, List, Optional, Tuple, Dict, TypeVar, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)

U = TypeVar("U")
V = TypeVar("V")


def dict_list_map_outplace(f: Callable[[U], V], x: Union[Dict, List, U]) -> Union[Dict, List, V]:
    """Maps dicts and lists *out-of-place* with a given function."""
    if isinstance(x, dict):
        return {k: dict_list_map_outplace(f, v) for k, v in x.items()}
    elif isinstance(x, list):
        return [dict_list_map_outplace(f, v) for v in x]
    else:
        return f(x)

def preload_tensors(state_dict: Dict, non_blocking=True):
    """ Preload tensors in state_dict to host memory through CPU memory
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
