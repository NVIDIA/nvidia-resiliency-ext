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

import inspect

import pytest
import torch

from nvidia_resiliency_ext.attribution import straggler


@pytest.fixture
def _straggler_init_shutdown():
    straggler.Detector.initialize()
    try:
        yield
    finally:
        straggler.Detector.shutdown()


@pytest.fixture
def _straggler_shutdown_at_exit():
    try:
        yield
    finally:
        straggler.Detector.shutdown()


def test_fail_if_not_initialized():
    with pytest.raises(RuntimeError):
        with straggler.Detector.detection_section("section00"):
            pass


@pytest.mark.skip(reason="does not work with Cython")
def test_unique_name_is_enforced(_straggler_init_shutdown):
    with straggler.Detector.detection_section("section00"):
        pass
    with pytest.raises(ValueError):
        with straggler.Detector.detection_section("section00"):
            pass


@pytest.mark.skip(reason="does not work with Cython")
def test_default_names_are_unique(_straggler_init_shutdown):
    with straggler.Detector.detection_section():
        pass
    with straggler.Detector.detection_section():
        pass
    assert len(straggler.Detector.custom_sections) == 2
    sections = list(straggler.Detector.custom_sections.values())
    assert sections[0].name != sections[1].name


@pytest.mark.skip(reason="does not work with Cython")
def test_with_block_location_is_used(_straggler_init_shutdown):
    # ensure that the location of the "with" block is correctly identified
    section = straggler.Detector.detection_section()
    _ = 1
    _ = 2
    with section:
        pass
    sections = list(straggler.Detector.custom_sections.values())
    frameinfo = inspect.getframeinfo(inspect.currentframe())
    # NOTE assumption: "with" block is 3 lines before the `getframeinfo`
    expected_postfix = f"{frameinfo.filename}:{frameinfo.lineno-3}"
    assert sections[0].location.endswith(expected_postfix)


def test_periodic_capture(_straggler_shutdown_at_exit):
    # check that fraction of section entries is monitored with profiling_interval>1
    straggler.Detector.initialize(profiling_interval=2)
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    torch.cuda.synchronize()
    for _ in range(4):
        with straggler.Detector.detection_section(name="one"):
            _ = torch.matmul(a, b)
    # we should have 2 out of 4 section "one" runs monitored
    report = straggler.Detector.generate_report()
    # check kernel stats
    assert len(report.local_kernel_summaries) == 1  # one kernel on rank0
    mm_kernel_stats = list(report.local_kernel_summaries.values())[0]
    assert mm_kernel_stats[straggler.Statistic.NUM] == 2  # 2/4 matmuls were profiled
    # check stats for section "one"
    assert len(report.local_section_summaries) == 1  # one section on rank0
    section_stats = report.local_section_summaries["one"]
    assert section_stats[straggler.Statistic.NUM] == 2  # 2/4 section runs were benchmarked


def test_cuda_profiling_disabled(_straggler_shutdown_at_exit):
    # check that CUDA kernels are not captured when profile_cuda=False
    straggler.Detector.initialize(profiling_interval=1)
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    torch.cuda.synchronize()
    for _ in range(4):
        with straggler.Detector.detection_section(name="one", profile_cuda=False):
            _ = torch.matmul(a, b)
    # we should have 2 out of 4 section "one" runs monitored
    report = straggler.Detector.generate_report()
    # check kernel stats
    assert len(report.local_kernel_summaries) == 0  # no kernels on rank0
    # check stats for section "one"
    assert len(report.local_section_summaries) == 1  # one section on rank0
    section_stats = report.local_section_summaries["one"]
    assert section_stats[straggler.Statistic.NUM] == 4  # 4/4 section runs were benchmarked


def test_can_handle_empty_elapseds(_straggler_init_shutdown):
    # verify that it does not crash when there are no CPU and GPU summaries

    with straggler.Detector.detection_section(
        name="one",
        profile_cuda=True,
    ):
        pass
    _ = straggler.Detector.generate_report()
    # all elapsed for section "one" has been cleaned up during report generation
    _ = straggler.Detector.generate_report()
    assert True  # ok if the subsequent generate_report not crashed
