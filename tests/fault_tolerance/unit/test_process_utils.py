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

import multiprocessing as mp
import time

import pytest

from nvidia_resiliency_ext.fault_tolerance.utils import (
    is_process_alive,
    wait_until_process_terminated,
)


def _sleeping_process(time_to_sleep):
    time.sleep(time_to_sleep)


def test_is_process_alive():
    proc_obj = mp.Process(target=_sleeping_process, args=(2,))
    proc_obj.start()
    assert is_process_alive(proc_obj.pid)
    wait_until_process_terminated(proc_obj.pid, timeout=10)
    assert not is_process_alive(proc_obj.pid)


def test_wait_until_process_terminated():
    proc_obj = mp.Process(target=_sleeping_process, args=(3,))
    proc_obj.start()
    with pytest.raises(Exception):
        wait_until_process_terminated(proc_obj.pid, timeout=0.1)
