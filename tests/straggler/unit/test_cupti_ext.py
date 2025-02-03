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

from nvidia_resiliency_ext.common.device_utils import get_current_device
import pytest
import torch
import torch.nn as nn

from nvidia_resiliency_ext.straggler import cupti_module


def test_basic_kernel_tracking():
    cupti_ext = cupti_module.CuptiProfiler()
    a = torch.randn(1000, 1000, device=get_current_device())
    b = torch.randn(1000, 1000, device=get_current_device())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cupti_ext.initialize()
    # start profiling
    cupti_ext.start()
    torch.matmul(a, b)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # should capture the matmul,
    # NOTE: `get_stats`` invokes CUPTI buffers flushing
    stats = cupti_ext.get_stats()
    assert len(stats) == 1
    mm_kernel_name, mm_kernel_stats = list(stats.items())[0]
    assert mm_kernel_stats.num_calls == 1
    cupti_ext.stop()
    # results should be still there, even after we stop the profiling
    stats = cupti_ext.get_stats()
    assert len(stats) == 1
    mm_kernel_name, mm_kernel_stats = list(stats.items())[0]
    assert mm_kernel_stats.num_calls == 1
    # explicit results reset
    cupti_ext.reset()
    stats = cupti_ext.get_stats()
    assert not stats


def test_tracking_start_stop():
    cupti_ext = cupti_module.CuptiProfiler()
    a = torch.randn(1000, 1000, device=get_current_device())
    b = torch.randn(1000, 1000, device=get_current_device())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cupti_ext.initialize()
    # start profiling
    cupti_ext.start()
    torch.matmul(a, b)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # CUPTI should capture the matmul
    cupti_ext.stop()
    # should not be captured as the profiler is stopped
    torch.matmul(a, b)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # restart profiling
    cupti_ext.start()
    torch.matmul(a, b)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # CUPTI should capture second matmul
    cupti_ext.stop()
    # there should be 2 matmuls captured,
    # `get_stats` flushes the CUPTI buffers
    stats = cupti_ext.get_stats()
    assert len(stats) == 1
    mm_kernel_name, mm_kernel_stats = list(stats.items())[0]
    assert mm_kernel_stats.num_calls == 2


def test_reset():
    cupti_ext = cupti_module.CuptiProfiler()
    a = torch.randn(1000, 1000, device=get_current_device())
    b = torch.randn(1000, 1000, device=get_current_device())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cupti_ext.initialize()
    # start profiling
    cupti_ext.start()
    torch.matmul(a, b)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # CUPTI should capture the matmul
    cupti_ext.stop()
    # explicit results reset
    cupti_ext.reset()
    # results should be empty
    stats = cupti_ext.get_stats()
    assert not stats


def test_max_stats_per_kernel():
    MAX_STATS_LEN_PER_KERNEL = 7
    cupti_ext = cupti_module.CuptiProfiler(statsMaxLenPerKernel=MAX_STATS_LEN_PER_KERNEL)
    a = torch.randn(1000, 1000, device=get_current_device())
    b = torch.randn(1000, 1000, device=get_current_device())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cupti_ext.initialize()
    # start profiling
    cupti_ext.start()
    for _ in range(3 * MAX_STATS_LEN_PER_KERNEL):
        _ = torch.matmul(a, b)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # CUPTI should capture the matmul
    cupti_ext.stop()
    # explicit results reset
    stats = cupti_ext.get_stats()
    assert len(stats) == 1
    _, mm_kernel_stats = list(stats.items())[0]
    assert mm_kernel_stats.num_calls == MAX_STATS_LEN_PER_KERNEL


def test_profiler_is_singleton():
    _ = cupti_module.CuptiProfiler()
    with pytest.raises(RuntimeError):
        _ = cupti_module.CuptiProfiler()


def test_with_cuda_graph():

    # Check if profiling of a CUDA graph yields the same results
    # as profiling individual kernels run sequentially.

    model = nn.Sequential(
        nn.Linear(256, 256, bias=False),
        nn.ReLU(),
        nn.Linear(256, 64, bias=False),
        nn.Sigmoid(),
    ).to(get_current_device(), torch.float32)

    x = torch.randn(256, 256, device=get_current_device())

    # capture fwd pass with graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _ = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    cupti_ext = cupti_module.CuptiProfiler()

    # start profiling
    cupti_ext.initialize()
    cupti_ext.start()

    # replay graph
    g.replay()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    stats_with_graph = cupti_ext.get_stats()
    cupti_ext.reset()

    # run fwd pass without a graph
    _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    stats_no_graph = cupti_ext.get_stats()
    cupti_ext.reset()

    # stop profiling
    cupti_ext.stop()
    cupti_ext.shutdown()

    # ensure that each kernel captured during sequential run
    # was also captured when graph was replayed
    for k, s in stats_no_graph.items():
        assert k in stats_with_graph
        assert s.num_calls == stats_with_graph[k].num_calls
