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

import time
from typing import NamedTuple

import torch
import torch.distributed as dist


class TimeoutsCalcError(Exception):
    pass


class _Timeouts(NamedTuple):
    """
    Represents calculated timeouts.
    Instances of this class are not supposed to be explicitly created in the client's code.
    """

    initial: float
    subsequent: float


class TimeoutsCalc:
    """
    This class implements logic for calculating timeouts:
        - initial_rank_heartbeat_timeout
        - rank_heartbeat_timeout
    At least 2 heartbeats are required to calculate the timeouts.
    `.update()` method should be called on every heartbeat.
    `.synchronize_all()` can be called to synchronize results from all ranks.
    NOTE: `synchronize_all` is the only `TimeoutsCalc` method that uses `torch.distributed`.
    Timeout for intial heartbeat is computed from `start_time` which is set to the current time in the constructor.
    If needed, `start_time` can be reset to the current time with `.reset_start_time`.
    """

    def __init__(self, start_time=None, safety_factor=5.0):
        self.start_time = time.monotonic() if start_time is None else start_time
        self.prev_hb_time = None
        self.count = 0
        self.initial_max_time = -1
        self.subsequent_max_time = -1
        self.safety_factor = safety_factor

    def reset_start_time(self):
        self.start_time = time.monotonic()

    @property
    def _device(self):
        device = None
        backend = dist.get_backend()
        if backend == dist.Backend.NCCL:
            device = torch.device('cuda')
        elif backend == dist.Backend.GLOO:
            device = torch.device('cpu')
        else:
            raise RuntimeError('Unsupported distributed backend')
        return device

    def synchronize_all(self):
        """
        Synchronize results from all ranks, by taking the max of all measured times.
        """
        if not (dist.is_available() and dist.is_initialized()):
            raise TimeoutsCalcError(".synchronize_all() requires initialized process group.")
        as_tensor = torch.tensor(
            [self.initial_max_time, self.subsequent_max_time],
            dtype=torch.float32,
            device=self._device,
        )
        dist.all_reduce(as_tensor, op=dist.ReduceOp.MAX)
        self.initial_max_time = float(as_tensor[0].item())
        self.subsequent_max_time = float(as_tensor[1].item())

    def update(self, hb_time=None):
        """
        Update the calculator with the new heartbeat.
        Update `initial_max_time` and `subsequent_max_time` accordingly.
        """
        hb_time = time.monotonic() if hb_time is None else hb_time
        if self.count == 0:
            elapsed = hb_time - self.start_time
            self.initial_max_time = max(self.initial_max_time, elapsed)
        else:
            elapsed = hb_time - self.prev_hb_time
            self.subsequent_max_time = max(self.subsequent_max_time, elapsed)
        self.count += 1
        self.prev_hb_time = hb_time

    def can_get_timeouts(self) -> bool:
        return self.initial_max_time > 0 and self.subsequent_max_time > 0

    def get_timeouts(self) -> _Timeouts:
        """
        Return the calculated timeouts.
        Timeouts are calculated by multiplying the max measured times by the "safety factor".
        """
        if not self.can_get_timeouts():
            raise TimeoutsCalcError("Not enough data to return the timeouts.")
        initial_timeout = self.safety_factor * self.initial_max_time
        subsequent_timeout = self.safety_factor * self.subsequent_max_time
        return _Timeouts(initial_timeout, subsequent_timeout)
