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
BasicTensorAwareStateDict provides a simple implementation of the TensorAwareStateDict interface,
which is used to manage state dictionaries within a CheckpointManager.

This class requires that all tensors in the user-provided state_dict are located on CUDA devices
and are easily accessible (i.e., they can only be nested within dictionaries or lists).

This implementation covers the most common use cases
for state dict management in distributed training scenarios.
"""

from typing import Union

import torch

from .base_state_dict import TensorAwareStateDict


def nested_values(x: Union[dict, list]):
    """Returns iterator over (nested) values of a given dict or list."""
    x_iter = x.values() if isinstance(x, dict) else x
    for v in x_iter:
        if isinstance(v, (dict, list)):
            yield from nested_values(v)
        else:
            yield v


def dict_list_map_inplace(f, x):
    """Maps dicts and lists *in-place* with a given function."""
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = dict_list_map_inplace(f, v)
    elif isinstance(x, list):
        x[:] = (dict_list_map_inplace(f, v) for v in x)
    else:
        return f(x)
    return x


class TensorPlaceholder:
    """
    A placeholder class that stores the device, shape, and dtype of tensor.
    This can be used to instantiate new tensors with the same properties (device, shape, and dtype)
    at a later time.
    """

    def __init__(self, ten):
        self._device = ten.device
        self._shape = ten.shape
        self._dtype = ten.dtype

    def init_tensor(self):
        """
        Creates a new tensor with using the saved properties (device, shape, and dtype).
        """
        return torch.empty(self._shape, dtype=self._dtype, device=self._device)


class BasicTensorAwareStateDict(TensorAwareStateDict):
    """
    The most basic implemention of TensorAwareStateDict
    defining the interface between the user code and checkpoint manager.

    This class requires that all tensors in the user state_dict are
    on cuda and are easily accessible (can be only nested in dicts or lists)
    """

    def __init__(self, state_dict):
        self.state_dict = state_dict
        for state_dict_value in nested_values(self.state_dict):
            # This simplifies and optimizes copy_tensors_to_cpu while streamlining device management
            if isinstance(state_dict_value, torch.Tensor):
                assert state_dict_value.is_cuda
        self._is_hollow = False

    def pop_tensors(self):
        """
        Extracts the tensor data from the state dict, preserving metadata.

        Removes the tensor data while retaining metadata (e.g., shape, dtype, device)
        needed to recreate empty tensors. After this operation, the state dictionary is "hollow",
        containing no tensor data.
        Further calls to `pop_tensor` will raise an error.

        Returns:
            List of extracted tensors
        """
        assert not self.is_hollow
        result = list(self.tensors)
        dict_list_map_inplace(
            lambda x: TensorPlaceholder(x) if isinstance(x, torch.Tensor) else x, self.state_dict
        )
        self._is_hollow = True
        return result

    @property
    def tensors(self):
        """
        Get the tensor data from the state dict.
        """
        assert not self.is_hollow
        for state_dict_value in nested_values(self.state_dict):
            if isinstance(state_dict_value, torch.Tensor):
                yield state_dict_value

    @property
    def is_hollow(self):
        """
        True iff tensors had been extracted and have not been inserted back yet.
        """
        return self._is_hollow

    def insert_tensors(self, tensor_data):
        """
        Reverse of `pop_tensors`. Replace tensor placeholders with actual values.
        The value of `self` is considered to be the same after:

        .. code-block:: python

            self.insert_tensors(self.pop_tensors())

        Args:
            tensor_data : An iterable containing the tensor data to be inserted
        """
        assert self.is_hollow
        tensor_stack = list(reversed(tensor_data))
        dict_list_map_inplace(
            lambda x: tensor_stack.pop() if isinstance(x, TensorPlaceholder) else x, self.state_dict
        )
        self._is_hollow = False

    def init_tensors(self):
        """
        Initializes empty tensors with the same properties as the original tensors.

        This function should only be called after the original tensors have been popped.
        It ensures that the newly created empty tensors match the shape,
        dtype, and device of the originals, but contain no data.
        """
        assert self.is_hollow
        dict_list_map_inplace(
            lambda x: x.init_tensor() if isinstance(x, TensorPlaceholder) else x, self.state_dict
        )
        self._is_hollow = False

    def copy_tensors_to_cpu(self, non_blocking=False):
        """
        Stores CPU copies of tensors in the state_dict, replacing the originals,
        but without destroying them.

        Args:
            non_blocking (bool): if set to True allows for asynchronous copying.
        """
        assert not self.is_hollow
        dict_list_map_inplace(
            lambda x: x.to("cpu", non_blocking=non_blocking) if isinstance(x, torch.Tensor) else x,
            self.state_dict,
        )

    def restore_tensor_device(self, non_blocking=True):
        """
        Restores all tensors to their original CUDA devices if a move is required.

        Args:
            non_blocking (bool): if set to True allows for asynchronous copying.
        """
        assert not self.is_hollow
        dict_list_map_inplace(
            lambda x: x.to("cuda", non_blocking=non_blocking) if isinstance(x, torch.Tensor) else x,
            self.state_dict,
        )
