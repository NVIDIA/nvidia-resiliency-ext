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

from nvidia_resiliency_ext.attribution.straggler import interval_tracker


def test_estimate():

    tracker = interval_tracker.ReportIntervalTracker()
    tracker.time_interval = 0.5

    assert tracker.iter_interval is None

    for i in range(120):
        tracker.iter_increase()
        time.sleep(0.01)
        if tracker.current_iter <= tracker.INTERVAL_ESTIMATION_ITERS:
            # estimate is available after INTERVAL_ESTIMATION_ITERS iterations
            assert tracker.iter_interval is None
        else:
            assert tracker.iter_interval is not None
            assert tracker.is_interval_elapsed() == (
                (tracker.current_iter % tracker.iter_interval) == 0
            )
        # a few longer initial steps should not affect the estimate
        if i < tracker.INTERVAL_ESTIMATION_ITERS // 2:
            time.sleep(0.04)

    # step times re not needed after the estimate is computed
    assert not tracker.step_times
    # iter time 0.01 and time interval 0.5sec should give estimate of ~50 iterations
    assert abs(tracker.iter_interval - 50) < 5
