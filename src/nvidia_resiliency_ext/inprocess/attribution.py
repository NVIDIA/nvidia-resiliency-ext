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

import dataclasses
import enum
import itertools
import re


class Interruption(enum.Enum):
    EXCEPTION = enum.auto()
    BASE_EXCEPTION = enum.auto()
    SOFT_TIMEOUT = enum.auto()
    HARD_TIMEOUT = enum.auto()
    TERMINATED = enum.auto()
    UNRESPONSIVE = enum.auto()
    MONITOR_PROCESS_EXCEPTION = enum.auto()


@dataclasses.dataclass(frozen=True)
class InterruptionRecord:
    rank: int
    interruption: Interruption

    @classmethod
    def from_str(cls, string: str):
        rank_match = re.search(r'rank=(\d+)', string)
        interruption_match = re.search(r'Interruption\.(\w+)', string)

        if not rank_match or not interruption_match:
            raise ValueError("Invalid State string format")

        rank = int(rank_match.group(1))
        interruption_name = interruption_match.group(1)
        interruption = Interruption[interruption_name]

        return cls(rank=rank, interruption=interruption)


def format_interruption_records(records):
    msg = ', '.join(
        (
            f'{interruption} on {ranks=}'
            for interruption, group in itertools.groupby(records, key=lambda r: r.interruption)
            for ranks in [set([elem.rank for elem in group])]
        )
    )
    return msg
