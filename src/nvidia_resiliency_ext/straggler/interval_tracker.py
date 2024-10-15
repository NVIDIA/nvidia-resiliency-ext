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

import dataclasses
import time
from typing import Optional, Sequence

import torch


@dataclasses.dataclass
class ReportIntervalTracker:
    """
    `ReportIntervalTracker` is used to calculate the reporting intervals based on a specified time interval.

    Attributes:
        INTERVAL_ESTIMATION_ITERS (int): Number of iterations used for estimating the report interval.
        time_interval (float): Target time interval for reporting.
        current_iter (int): Counter for the current iteration.
        iter_interval (int, optional): Computed iteration interval based on the target time interval.
    """

    INTERVAL_ESTIMATION_ITERS: int = 16
    time_interval: float = 60.0
    current_iter: int = 0
    iter_interval: Optional[int] = None
    prev_iter_start_time: Optional[float] = None
    step_times: Sequence[float] = dataclasses.field(default_factory=list)
    profiling_interval: int = 1

    def _gather_report_interval(self):
        """
        Gathers the report interval across all distributed processes and sets the maximum interval.
        """
        assert self.iter_interval is None, "Report iteration interval has already been gathered."

        step_times = torch.tensor(self.step_times, dtype=torch.float32)
        median_step_time = torch.median(step_times)

        gathered_interval = (self.time_interval / median_step_time).to(torch.cuda.current_device())
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(gathered_interval, op=torch.distributed.ReduceOp.MAX)
        # it makes no sense to report more frequently than the profiling interval
        self.iter_interval = int(max(gathered_interval.item(), self.profiling_interval))

    def iter_increase(self):
        """
        Increases the iteration counter and gathers the report interval if the estimation phase is completed.
        """
        self.current_iter += 1

        if self.iter_interval is None:
            if self.prev_iter_start_time is not None:
                step_time = time.monotonic() - self.prev_iter_start_time
                self.step_times.append(step_time)
                if len(self.step_times) == self.INTERVAL_ESTIMATION_ITERS:
                    self._gather_report_interval()
                    self.step_times.clear()
                    assert self.iter_interval is not None
            self.prev_iter_start_time = time.monotonic()

    def is_interval_elapsed(self):
        """
        Checks if the current iteration is a reporting interval.

        Returns:
            bool: True if the interval has elapsed, False otherwise.
        """
        return (self.iter_interval is not None) and (self.current_iter % self.iter_interval == 0)
