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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "cpython.h"

namespace nvrx::checkpoint {

namespace py = pybind11;

// Kinds of state_dict_metadata entries, numbered as in state_dict_table.py.
enum class Kind : uint8_t { Bytes = 0, Tensor = 1, Other = 2 };
constexpr std::string_view STATE_DICT_METADATA = "state_dict_metadata";
constexpr std::string_view STORAGE_DATA = "storage_data";

// The element types of a state dict table's arrays.
template <typename T>
concept TableValue = std::same_as<T, uint8_t> || std::same_as<T, int64_t>;

template <TableValue T>
using Array = py::array_t<T, py::array::c_style>;

// A StateDictTable, checked as StateDictTable._check checks it, so that the writer can read its
// arrays without further checks.
class StateDictTableView {
   public:
    // A field of the Metadata: its name, and its opcodes (None for the fields written from the
    // table or the write results).
    struct Field {
        std::string name;
        py::object opcodes;
    };

    explicit StateDictTableView(py::handle table);

    [[nodiscard]] auto entries() const -> size_t { return kind_.size(); }
    // Return the fingerprint of the table's fqn list, read once, on first use: only index tables
    // need it.
    [[nodiscard]] auto fqns_fingerprint() const -> int64_t;
    [[nodiscard]] auto fields() const -> std::vector<Field> const& { return fields_; }
    [[nodiscard]] auto fqn(size_t i) const -> PyObject* { return PyTuple_GET_ITEM(fqns_.ptr(), i); }
    [[nodiscard]] auto kind(size_t i) const -> Kind { return static_cast<Kind>(kind_[i]); }
    [[nodiscard]] auto properties(size_t i) const -> std::string_view {
        return detail::bytes_view(PyTuple_GET_ITEM(properties_.ptr(), ref_[i]));
    }
    [[nodiscard]] auto other(size_t i) const -> std::string_view {
        return detail::bytes_view(PyTuple_GET_ITEM(others_.ptr(), ref_[i]));
    }
    [[nodiscard]] auto ndim(size_t i) const -> size_t { return static_cast<size_t>(ndim_[i]); }
    [[nodiscard]] auto size(size_t i) const -> std::span<int64_t const> {
        return dims_.subspan(static_cast<size_t>(dims_start_[i]), ndim(i));
    }
    [[nodiscard]] auto chunks(size_t i) const -> size_t {
        return static_cast<size_t>(chunk_start_[i + 1] - chunk_start_[i]);
    }
    // Return chunk k of entry i: its offsets, then its sizes, after the entry's size.
    [[nodiscard]] auto chunk(size_t i, size_t k) const -> std::span<int64_t const> {
        size_t const width = 2 * ndim(i);
        return dims_.subspan(static_cast<size_t>(dims_start_[i]) + ndim(i) + (width * k), width);
    }
    [[nodiscard]] auto dims_size() const -> size_t { return dims_.size(); }

   private:
    // Check that the arrays describe entries() entries consistently; see StateDictTable._check.
    void check() const;

    py::object table_;
    mutable std::optional<int64_t> fqns_fingerprint_;
    py::tuple fqns_, properties_, others_;
    Array<uint8_t> kind_array_;
    Array<int64_t> ref_array_, ndim_array_, chunk_start_array_, dims_start_array_, dims_array_;
    std::span<uint8_t const> kind_;
    std::span<int64_t const> ref_, ndim_, chunk_start_, dims_start_, dims_;
    std::vector<Field> fields_;
};

// Convert metadata, without its storage_data, to a table_cls (a StateDictTable).
[[nodiscard]] auto from_metadata(py::handle metadata,
                                 py::object small_pickle,
                                 py::object const& table_cls) -> py::object;

}  // namespace nvrx::checkpoint
