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

import enum


class Statistic(enum.Enum):
    """Enumeration of constants representing common statistical measures that
    are used for performance analysis and reporting."""

    MIN = enum.auto()
    MAX = enum.auto()
    MED = enum.auto()
    AVG = enum.auto()
    STD = enum.auto()
    NUM = enum.auto()

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"
