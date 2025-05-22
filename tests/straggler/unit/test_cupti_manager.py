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

import pytest
import torch

from nvidia_resiliency_ext.attribution.straggler.cupti import CuptiManager


def test_cupti_manager_start_stop():
    cupti_mgr = CuptiManager()
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    torch.cuda.synchronize()
    cupti_mgr.initialize()
    with pytest.raises(Exception):
        cupti_mgr.stop_profiling()
    # start profiling 2 times
    cupti_mgr.start_profiling()
    cupti_mgr.start_profiling()
    # stop once, should be still profiling
    cupti_mgr.stop_profiling()
    # do the matmul, that should be captured
    torch.matmul(a, b)
    torch.cuda.synchronize()
    # stop again, profiling should be stopped
    cupti_mgr.stop_profiling()
    # do the matmul, should not be captured
    torch.matmul(a, b)
    torch.cuda.synchronize()
    # ensure that just one matmul was captured
    stats = cupti_mgr.get_results()
    cupti_mgr.shutdown()
    assert len(stats) == 1
    mm_kernel_name, mm_kernel_stats = list(stats.items())[0]
    assert mm_kernel_stats.num_calls == 1


def test_cupti_manager_captures_all_started_kernels():
    cupti_mgr = CuptiManager()
    cupti_mgr.initialize()
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    # do not not capture randn
    torch.cuda.synchronize()
    # some CUDA activity that should NOT be captured, as profiling is not started
    for _ in range(100):
        _ = torch.matmul(a, b)
    # now start capturing
    cupti_mgr.start_profiling()
    for _ in range(50):
        _ = torch.matmul(a, b)
    # another nested start
    cupti_mgr.start_profiling()
    # stop, but should still be capturing, as there were 2 starts
    cupti_mgr.stop_profiling()
    for _ in range(50):
        _ = torch.matmul(a, b)
    # second stop, all capturing should be stopped
    cupti_mgr.stop_profiling()
    for _ in range(100):
        _ = torch.matmul(a, b)
    # ensure that just matmul OP was captured
    torch.cuda.synchronize()
    stats = cupti_mgr.get_results()
    cupti_mgr.shutdown()
    assert len(stats) == 1
    mm_kernel_name, mm_kernel_stats = list(stats.items())[0]
    assert mm_kernel_stats.num_calls == 100
