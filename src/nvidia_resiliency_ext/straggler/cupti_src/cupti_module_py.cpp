/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include <pybind11/pybind11.h>
#include "CuptiProfiler.h"

namespace py = pybind11;


static py::dict get_stats_py(CuptiProfiler* profiler_inst) {
    auto stats = profiler_inst->getStats();
    py::dict dict;
    for (auto& [key, value] : stats) {
        dict[py::cast(key)] = py::cast(value);
    }
    return dict;
}

PYBIND11_MODULE(nvrx_cupti_module, m) {
    py::class_<KernelStats>(m, "KernelStats")
        .def(py::init<>())
        .def_readwrite("min", &KernelStats::min)
        .def_readwrite("max", &KernelStats::max)
        .def_readwrite("median", &KernelStats::median)
        .def_readwrite("avg", &KernelStats::avg)
        .def_readwrite("stddev", &KernelStats::stddev)
        .def_readwrite("num_calls", &KernelStats::num_calls)
        .def("__str__", &KernelStats::toString);

    py::class_<CuptiProfiler>(m, "CuptiProfiler")
        .def(py::init<size_t, size_t, size_t>(),
             py::arg("bufferSize") = 1024 * 1024 * 8,
             py::arg("numBuffers") = 8,
             py::arg("statsMaxLenPerKernel") = 1024)
        .def("start", &CuptiProfiler::startProfiling, "Start profiling.")
        .def("stop", &CuptiProfiler::stopProfiling, "Stop profiling.")
        .def("initialize", &CuptiProfiler::initializeProfiling, "Initialize CUPTI.")
        .def("shutdown", &CuptiProfiler::shutdownProfiling, "Shutdown CUPTI.")
        .def("get_stats", get_stats_py, "Retrieve kernel execution statistics.")
        .def("reset", &CuptiProfiler::reset, "Reset statistics.");
}
