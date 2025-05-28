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

import itertools
from typing import Dict, List

from .dist_utils import all_gather_object, is_all_true


class NameMapper:
    """
    A class to manage the mapping of kernel and section names to integers for optimized synchronization
    across multiple ranks in a distributed setting.

    Attributes:
        pg: Process group for communication.
        kernel_name_to_id (Dict[str, int]): Dictionary mapping kernel names to integer IDs.
        id_to_kernel_name (Dict[int, str]): Dictionary mapping integer IDs to kernel names.
        section_name_to_id (Dict[str, int]): Dictionary mapping section names to integer IDs.
        id_to_section_name (Dict[int, str]): Dictionary mapping integer IDs to section names.
        kernel_counter (int): Counter for assigning new integer IDs to kernel names.
        section_counter (int): Counter for assigning new integer IDs to section names.
    """

    def __init__(self, pg=None):
        self.group = pg

        self.kernel_name_to_id: Dict[str, int] = {}
        self.id_to_kernel_name: Dict[int, str] = {}
        self.section_name_to_id: Dict[str, int] = {}
        self.id_to_section_name: Dict[int, str] = {}

        self.kernel_counter: int = 0
        self.section_counter: int = 0

    def _check_if_has_all_names(self, kernel_names: List[str], section_names: List[str]) -> bool:
        """Returns True if all provided kernel and section names have assigned IDs."""
        if any(n not in self.kernel_name_to_id for n in kernel_names):
            return False
        if any(n not in self.section_name_to_id for n in section_names):
            return False
        return True

    def gather_and_assign_ids(self, kernel_names: List[str], section_names: List[str]) -> None:
        """
        Gathers kernel and section names from all ranks and assigns unique integer IDs to them.
        It's quaranteed that:
        - IDs are conseqentive integers starting from 0.
        - If a kernel/section name got an ID, it wont be changed.
        - A name is mapped to same ID across all ranks.

        Args:
            kernel_names (List[str]): List of kernel names recorded in the current rank.
            section_names (List[str]): List of section names recorded in the current rank.
        """
        has_all_names = self._check_if_has_all_names(kernel_names, section_names)
        need_gather = not is_all_true(has_all_names, self.group)
        if need_gather:
            gathered_names = all_gather_object((section_names, kernel_names), self.group)
            gathered_section_lists = (sections for sections, _ in gathered_names)
            gathered_kernel_lists = (kernels for _, kernels in gathered_names)

            # all ranks have identical lists of names at this point
            # we can go one by one and assign IDs if there are not seen entries

            for section in itertools.chain.from_iterable(gathered_section_lists):
                self._assign_section_id(section_name=section)
            for kernel in itertools.chain.from_iterable(gathered_kernel_lists):
                self._assign_kernel_id(kernel_name=kernel)

    def _assign_kernel_id(self, kernel_name: str) -> int:
        """
        Assigns a unique integer ID to a kernel name.

        Args:
            kernel_name (str): The name of the kernel to add.

        Returns:
            int: The integer ID assigned to the kernel name.
        """
        if kernel_name not in self.kernel_name_to_id:
            self.kernel_name_to_id[kernel_name] = self.kernel_counter
            self.id_to_kernel_name[self.kernel_counter] = kernel_name
            self.kernel_counter += 1
        return self.kernel_name_to_id[kernel_name]

    def get_kernel_name(self, kernel_id: int) -> str:
        """
        Retrieves the kernel name associated with a given integer ID.

        Args:
            kernel_id (int): The integer ID of the kernel.

        Returns:
            str: The name of the kernel.
        """
        return self.id_to_kernel_name[kernel_id]

    def get_kernel_id(self, kernel_name: str) -> int:
        """
        Retrieves the integer ID associated with a given kernel name.

        Args:
            kernel_name (str): The name of the kernel.

        Returns:
            int: The integer ID of the kernel.
        """
        return self.kernel_name_to_id[kernel_name]

    def _assign_section_id(self, section_name: str) -> int:
        """
        Assigns a unique integer ID to a section name.

        Args:
            section_name (str): The name of the section to add.

        Returns:
            int: The integer ID assigned to the section name.
        """
        if section_name not in self.section_name_to_id:
            self.section_name_to_id[section_name] = self.section_counter
            self.id_to_section_name[self.section_counter] = section_name
            self.section_counter += 1
        return self.section_name_to_id[section_name]

    def get_section_name(self, section_id: int) -> str:
        """
        Retrieves the section name associated with a given integer ID.

        Args:
            section_id (int): The integer ID of the section.

        Returns:
            str: The name of the section if found.
        """
        return self.id_to_section_name[section_id]

    def get_section_id(self, section_name: str) -> int:
        """
        Retrieves the integer ID associated with a given section name.

        Args:
            section_name (str): The name of the section.

        Returns:
            int: The integer ID of the section if found.
        """
        return self.section_name_to_id[section_name]
