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
from typing import Collection, Mapping, Optional

import torch
import torch.distributed as dist

from nvidia_resiliency_ext.fault_tolerance.data import (
    HeartbeatTimeouts,
    SectionAction,
    SectionTimeouts,
)


class TimeoutsCalcError(Exception):
    pass


class TimeoutsCalc:
    """
    This class implements logic for calculating timeouts.
    `.update_on_heartbeat()` method should be called on every heartbeat.
    `.update_on_section_event()` method should be called on every section event.
    `.synchronize_all()` can be called to synchronize results from all ranks.
    NOTE: `synchronize_all` is the only `TimeoutsCalc` method that uses `torch.distributed`.
    Timeout for intial heartbeat is computed from `start_time` which is set to the current time in the constructor.
    If needed, `start_time` can be reset to the current time with `.reset_start_time`.
    """

    def __init__(self, sections=None, start_time=None, safety_factor=5.0):
        self.start_time = time.monotonic() if start_time is None else start_time
        self.prev_hb_time = None
        self.count = 0
        self.initial_max_time = -1
        self.subsequent_max_time = -1
        self.safety_factor = safety_factor
        self.open_sections: Mapping[str, float] = {}  # section name -> start time
        self.last_section_close_time = self.start_time
        self.out_of_section_max_time = -1
        if sections is not None:
            self.section_to_max_time = {name: -1 for name in sections}
        else:
            self.section_to_max_time = {}

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
            [self.initial_max_time, self.subsequent_max_time, self.out_of_section_max_time]
            + list(self.section_to_max_time.values()),
            dtype=torch.float32,
            device=self._device,
        )
        dist.all_reduce(as_tensor, op=dist.ReduceOp.MAX)
        self.initial_max_time = float(as_tensor[0].item())
        self.subsequent_max_time = float(as_tensor[1].item())
        self.out_of_section_max_time = float(as_tensor[2].item())
        for idx, section in enumerate(self.section_to_max_time):
            self.section_to_max_time[section] = float(as_tensor[3 + idx].item())

    def update_on_heartbeat(self, hb_time=None):
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

    def _open_section(self, section, event_time):
        if section in self.open_sections:
            raise TimeoutsCalcError(
                f"Tried to open {section} that is already open."
                + " Please ensure that each section open is followed by a close."
            )
        self.maybe_bump_oos_time(curr_time=event_time)
        self.open_sections[section] = event_time

    def _close_section(self, section, event_time):
        if section not in self.open_sections:
            raise TimeoutsCalcError(f"Tried to close {section} that is not open.")
        elapsed = event_time - self.open_sections[section]
        curent_max = self.section_to_max_time[section]
        self.section_to_max_time[section] = max(curent_max, elapsed)
        del self.open_sections[section]
        if not self.open_sections:
            self.last_section_close_time = event_time

    def update_on_section_event(
        self, section: Optional[str], action: SectionAction, event_time=None
    ):
        """
        Update the calculator due to a section event.
        """
        if section is not None and section not in self.section_to_max_time:
            raise TimeoutsCalcError(
                f"Unknown section: {section}. Does it have an entry in the config?"
            )
        event_time = time.monotonic() if event_time is None else event_time
        if action is SectionAction.OPEN:
            self._open_section(section, event_time)
        elif action is SectionAction.CLOSE:
            self._close_section(section, event_time)
        elif action is SectionAction.CLOSE_ALL:
            for section in list(self.open_sections.keys()):
                self._close_section(section, event_time)

    @staticmethod
    def _ema(x: float, curr: float, alpha: float = 0.5):
        return alpha * x + (1.0 - alpha) * curr

    def can_get_hb_timeouts(self) -> bool:
        """Return True if there is enough data to calculate heartbeat timeouts."""
        return self.initial_max_time > 0 and self.subsequent_max_time > 0

    def get_hb_timeouts(self, current: Optional[HeartbeatTimeouts] = None) -> HeartbeatTimeouts:
        """
        Return the calculated heartbeat timeouts.
        Timeouts are calculated by multiplying the max measured times by the "safety factor".
        """
        if not self.can_get_hb_timeouts():
            raise TimeoutsCalcError("Not enough data to return the timeouts.")
        initial = self.safety_factor * self.initial_max_time
        subsequent = self.safety_factor * self.subsequent_max_time
        if current is not None and current.were_calculated:
            # if current timeouts were also calculated, use EMA to merge the values
            assert current.are_valid
            initial = TimeoutsCalc._ema(initial, current.initial)
            subsequent = TimeoutsCalc._ema(subsequent, current.subsequent)
        new_timeouts = HeartbeatTimeouts(
            initial=initial, subsequent=subsequent, were_calculated=True
        )
        return new_timeouts

    def can_get_section_timeouts(
        self, selected_sections: Optional[Collection[str]] = None, calc_out_of_section: bool = True
    ) -> bool:
        """Return True if there is enough data to calculate section timeouts."""
        selected = selected_sections if selected_sections else self.section_to_max_time.keys()
        can1 = all(self.section_to_max_time[s] > 0 for s in selected)
        can2 = self.out_of_section_max_time > 0 if calc_out_of_section else True
        return can1 and can2

    @staticmethod
    def _get_merged_section_timeouts(
        new: SectionTimeouts, current: Optional[SectionTimeouts]
    ) -> SectionTimeouts:
        # This is used to merge currently used timeouts with newly calculated.
        # Logic is as follows:
        # - If the current timeout was also calculated, we use EMA to merge the values.
        #  Max value was used formerly, but it could lead to overestimated timeouts.
        #  With EMA outliers impact should be reduced.
        # - If the current timeout was not calculated, the newly calculated value is used.
        # - If the new timeout was not calculated, the current timeout value is kept.
        if current is None:
            return new
        section_timeout = {}
        for se in new.section:
            if se in current.calculated_sections and se in new.calculated_sections:
                section_timeout[se] = TimeoutsCalc._ema(current.section[se], new.section[se])
            elif se in new.calculated_sections:
                section_timeout[se] = new.section[se]
            else:
                section_timeout[se] = current.section[se]
        calculated_sections = set(current.calculated_sections) | set(new.calculated_sections)

        oos_timeout = None
        if current.is_out_of_section_calculated and new.is_out_of_section_calculated:
            oos_timeout = TimeoutsCalc._ema(current.out_of_section, new.out_of_section)
        elif new.is_out_of_section_calculated:
            oos_timeout = new.out_of_section
        else:
            oos_timeout = current.out_of_section
        is_oos_calculated = current.is_out_of_section_calculated or new.is_out_of_section_calculated

        res = SectionTimeouts(
            section=section_timeout,
            out_of_section=oos_timeout,
            calculated_sections=calculated_sections,
            is_out_of_section_calculated=is_oos_calculated,
        )
        return res

    def get_section_timeouts(
        self,
        selected_sections: Optional[Collection[str]] = None,
        calc_out_of_section: bool = True,
        current: Optional[SectionTimeouts] = None,
    ) -> SectionTimeouts:
        """
        Return calculated section timeouts and out-of-section timeout.
        Timeouts are calculated by multiplying the max measured times by the "safety factor".

        Timeouts not selected for calculation are set to None.
        If `current` timeout is provided, newly calculated timeouts are merged with the current timeouts.

        Args:
            selected_sections (Optional[Collection[str]]): A collection of section names to calculate timeouts for. If None, timeouts for all sections are calculated.
            calc_out_of_section (bool): Whether to include the out-of-section timeout in the calculation. Defaults to True.
            current (Optional[SectionTimeouts]): An optional current SectionTimeouts object to merge with the new calculated timeouts.

        Returns:
            SectionTimeouts: The calculated section timeouts.

        Raises:
            TimeoutsCalcError: If there is not enough data to return the timeouts.
        """
        if not self.can_get_section_timeouts(
            selected_sections=selected_sections, calc_out_of_section=calc_out_of_section
        ):
            raise TimeoutsCalcError("Not enough data to return the timeouts.")
        selected_sections = (
            selected_sections if selected_sections is not None else self.section_to_max_time.keys()
        )
        section_timeout = {se: None for se in self.section_to_max_time}
        for se in selected_sections:
            val = self.section_to_max_time[se]
            section_timeout[se] = self.safety_factor * val
            assert section_timeout[se] > 0

        oos_timeout = None
        if calc_out_of_section:
            oos_timeout = self.safety_factor * self.out_of_section_max_time
            assert oos_timeout > 0

        new = SectionTimeouts(
            section=section_timeout,
            out_of_section=oos_timeout,
            calculated_sections=selected_sections,
            is_out_of_section_calculated=calc_out_of_section,
        )
        merged = TimeoutsCalc._get_merged_section_timeouts(new, current)
        return merged

    @property
    def is_out_of_section(self):
        return not self.open_sections

    def maybe_bump_oos_time(self, curr_time=None):
        if self.is_out_of_section:
            curr_time = time.monotonic() if curr_time is None else curr_time
            elapsed_oos = curr_time - self.last_section_close_time
            self.out_of_section_max_time = max(self.out_of_section_max_time, elapsed_oos)
