/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cpython.h"

#include <pybind11/pybind11.h>

#include <cstddef>
#include <string_view>

using namespace nvrx::checkpoint;
using namespace nvrx::checkpoint::detail;

py::object detail::instance_dict(PyObject* obj) {
    PyObject* const r = PyObject_GenericGetDict(obj, nullptr);
    if (!r) {
        throw py::error_already_set();
    }
    const auto d = py::reinterpret_steal<py::object>(r);
    if (!PyDict_CheckExact(d.ptr())) {
        throw py::type_error("expected a __dict__");
    }
    return d;
}

PyObject* detail::dict_get(PyObject* dict, PyObject* key) {
    PyObject* const r = PyDict_GetItemWithError(dict, key);
    if (!r && PyErr_Occurred()) {
        throw py::error_already_set();
    }
    return r;
}

std::string_view detail::utf8(PyObject* s) {
    if (!PyUnicode_CheckExact(s)) {
        throw py::type_error("expected str");
    }
    Py_ssize_t n;
    const char* const p = PyUnicode_AsUTF8AndSize(s, &n);
    if (!p) {
        throw py::error_already_set();
    }
    return {p, static_cast<size_t>(n)};
}
