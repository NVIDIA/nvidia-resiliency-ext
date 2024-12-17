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


import torch


def get_default_device_from_type(device_type: str) -> torch.device:
    """Returns the default PyTorch device based on the specified device type.

    This function maps a device type string to the corresponding PyTorch device.
    It supports both "cpu" and "cuda" types, raising an error for unsupported types.

    Args:
        device_type (str): The type of device to retrieve. Should be either "cpu" or "cuda".

    Returns:
        torch.device: The default device corresponding to the provided device type.

    Raises:
        ValueError: If the provided device type is unsupported.
    """
    if device_type == "cpu":
        return torch.device("cpu")
    elif device_type == "cuda":
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        raise ValueError(f"Device type {device_type} unsupported!")


class TensorPlaceholder:
    """A placeholder for a tensor that helps manage device-specific tensor operations.

    This class creates a "hollow" tensor that does not allocate memory on the original device.
    It provides methods to create empty tensors and restore them back to the original device.

    Attributes:
        hollow_tensor (torch.Tensor): A tensor initialized to an empty state with the "meta" device.
        orig_device_type (str): The original device type of the provided tensor.
    """

    def __init__(self, tensor: torch.Tensor):
        self.hollow_tensor = torch.empty_like(tensor, device="meta")
        self.orig_device_type = tensor.device.type

    def empty_like(self, device=None):
        """Creates an empty tensor with the same shape as the hollow tensor.

        Args:
            device (Optional[str]): The device on which to create the empty tensor.
                                    If None, uses the original device type.

        Returns:
            torch.Tensor: An empty tensor of the same shape as the hollow tensor.
        """
        if device is None:
            device = self.device
        return torch.empty_like(self.hollow_tensor, device=device)

    @property
    def device(self):
        """Returns the default device based on the original device type.

        Returns:
            torch.device: The device corresponding to the original device type.
        """
        return get_default_device_from_type(self.orig_device_type)

    def restore(self, data=None) -> torch.Tensor:
        """Restores the original tensor from the placeholder.

        Returns an empty tensor if no data is provided.

        Args:
            data (Optional[torch.Tensor]): Tensor data to restore; if None, returns an empty tensor.

        Returns:
            torch.Tensor: Restored tensor on the original device.

        Raises:
            AssertionError: If shape or dtype of data does not match the hollow tensor.
        """
        if data is None:
            return torch.empty_like(self.hollow_tensor, device=self.device)
        assert self.hollow_tensor.shape == data.shape
        assert self.hollow_tensor.dtype == data.dtype
        return data.to(self.device)
