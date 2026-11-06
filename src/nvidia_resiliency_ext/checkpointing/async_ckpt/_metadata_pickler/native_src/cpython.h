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

#pragma once

#include <pybind11/pybind11.h>

#include <cstddef>
#include <string_view>

namespace nvrx::checkpoint::detail {

namespace py = pybind11;

// Return a new reference to obj.__dict__, the state pickle writes for the metadata classes. Only
// objects of exactly those classes get here, whose __dict__ is the generic one, so it is fetched
// directly rather than looked up as an attribute.
[[nodiscard]] auto instance_dict(PyObject* obj) -> py::object;

// Borrowed reference to dict[key], or nullptr if key is missing.
[[nodiscard]] auto dict_get(PyObject* dict, PyObject* key) -> PyObject*;

// UTF-8 view of a str's characters, valid while s is alive.
[[nodiscard]] auto utf8(PyObject* s) -> std::string_view;

// Return the bytes of a bytes object, valid while it is alive.
[[nodiscard]] inline auto bytes_view(PyObject* b) -> std::string_view {
    return {PyBytes_AS_STRING(b), static_cast<size_t>(PyBytes_GET_SIZE(b))};
}

}  // namespace nvrx::checkpoint::detail
