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
TensorAwareStateDict defines an interface for managing various state dicts within CheckpointManager.
The primary feature of this class is its ability to distinguish tensor objects from other elements.
Additionally, it can be converted to and from original state_dicts.
"""

from abc import ABC, abstractmethod
from typing import Any, Iterable, Sequence, ValuesView

import torch


class TensorAwareStateDict(ABC):
    """
    Base class that defines the interface between the user state dict and the checkpoint manager.

    The primary goal is to differentiate tensor content from non-tensor content, enabling efficient
    migration of the state dict during checkpoint save and load.
    """

    @abstractmethod
    def pop_tensors(self) -> Sequence[torch.Tensor]:
        """
        Extracts the tensor data from the wrapped state dict, preserving metadata.

        Removes the tensor data while retaining metadata (e.g., shape, dtype, device)
        needed to recreate empty tensors. After this operation, the state dictionary is "hollow",
        containing no tensor data.
        Further calls to `pop_tensor` will raise an error.

        Returns:
            List of extracted tensors
        """

    @property
    @abstractmethod
    def tensors(self) -> Iterable[torch.Tensor]:
        """
        Get the tensor data from the wrapped state dict.
        """
        pass

    @property
    @abstractmethod
    def is_hollow(self) -> bool:
        """
        True iff tensors had been extracted and have not been inserted back yet.
        """
        pass

    @abstractmethod
    def insert_tensors(self, tensor_data: Iterable[torch.Tensor]):
        """
        Reverse of `pop_tensors`. Replace tensor placeholders with actual values.
        The value of `self` is considered to be the same after:

        .. code-block:: python

            self.insert_tensors(self.pop_tensors())

        Args:
            tensor_data : An iterable containing the tensor data to be inserted.
        """
        pass

    @abstractmethod
    def init_tensors(self):
        """
        Initializes empty tensors with the same properties as the original tensors.

        This function should only be called after the original tensors have been popped.
        It ensures that the newly created empty tensors match the shape,
        dtype, and device of the originals, but contain no data.
        """
        pass

    @abstractmethod
    def copy_tensors_to_cpu(self, non_blocking=False):
        """
        Stores CPU copies of tensors in the state_dict, replacing the originals,
        but without destroying them.
        The original devices are remembered for restoration with restore_tensor_device().

        Args:
            non_blocking (bool): if set to True allows for asynchronous copying.
        """
        pass

    @abstractmethod
    def restore_tensor_device(self, non_blocking=True):
        """
        Restores all tensors to their original devices, if a move is required.

        Args:
            non_blocking (bool): if set to True allows for asynchronous copying.
        """
        pass

    def values(self) -> ValuesView[Any]:
        """
        Returns:
            The values from the state dictionary.
        """
        return vars(self).values()
