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

auto detail::instance_dict(PyObject* obj) -> py::object {
    PyObject* const r = PyObject_GenericGetDict(obj, nullptr);
    if (r == nullptr) {
        throw py::error_already_set();
    }
    auto const d = py::reinterpret_steal<py::object>(r);
    if (!PyDict_CheckExact(d.ptr())) {
        throw py::type_error("expected a __dict__");
    }
    return d;
}

auto detail::dict_get(PyObject* dict, PyObject* key) -> PyObject* {
    PyObject* const r = PyDict_GetItemWithError(dict, key);
    if ((r == nullptr) && (PyErr_Occurred() != nullptr)) {
        throw py::error_already_set();
    }
    return r;
}

auto detail::utf8(PyObject* s) -> std::string_view {
    if (!PyUnicode_CheckExact(s)) {
        throw py::type_error("expected str");
    }
    Py_ssize_t n = 0;
    char const* const p = PyUnicode_AsUTF8AndSize(s, &n);
    if (p == nullptr) {
        throw py::error_already_set();
    }
    return {p, static_cast<size_t>(n)};
}
