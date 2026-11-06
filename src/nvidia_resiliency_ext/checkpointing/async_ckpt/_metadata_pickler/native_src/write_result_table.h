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

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

namespace nvrx::checkpoint {

namespace py = pybind11;

// How a table names fqns: by string id, or by entry in the coordinator's fqn list.
enum class FqnMode : int64_t { Strings = 0, Indices = 1 };
enum class Column : uint8_t { Fqn, Index, OffsetNdim, Path, Offset, Length, Flags };
constexpr size_t COLUMNS = 7;
constexpr int64_t FLAG_INDEX = 0x1, FLAG_OFFSET = 0x2, FLAG_OFFSET_SIZE = 0x4;
// The gathered tables: one zero-padded table per row.
using Rows = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

namespace detail {

// Reject a malformed table, before anything reads past its row.
[[noreturn]] void bad_table(std::string_view what);

}  // namespace detail

// A run of little-endian int64s in a span of bytes, which need not be aligned.
class Int64s {
   public:
    // The int64s in bytes, whose size is a multiple of 8.
    explicit Int64s(std::span<std::byte const> bytes) : bytes_(bytes) {}

    // The number of int64s.
    [[nodiscard]] auto size() const -> size_t { return bytes_.size() / sizeof(int64_t); }

    // The i-th int64, i < size(). bit_cast needs an object to read, and there is none at an
    // arbitrary offset of the row, so the bytes are copied into one; this compiles to one load.
    [[nodiscard]] auto operator[](size_t i) const -> int64_t {
        std::array<std::byte, sizeof(int64_t)> v{};
        std::ranges::copy(bytes_.subspan(sizeof(int64_t) * i).first<sizeof(int64_t)>(), v.begin());
        return std::bit_cast<int64_t>(v);
    }

    // The field of a record (a header or an entry) that an enum names.
    template <typename Field>
        requires std::is_enum_v<Field>
    [[nodiscard]] auto operator[](Field field) const -> int64_t {
        return (*this)[static_cast<size_t>(field)];
    }

    // The count int64s from pos, pos + count <= size().
    [[nodiscard]] auto subspan(size_t pos, size_t count) const -> Int64s {
        return Int64s(bytes_.subspan(sizeof(int64_t) * pos, sizeof(int64_t) * count));
    }

   private:
    std::span<std::byte const> bytes_;
};

// A write-result table within its row of the gathered buffer. parse checks that its sections lie
// within the row; the accessors check what its entries refer to.
class WriteResultTable {
   public:
    // The table in a row.
    [[nodiscard]] static auto parse(std::span<std::byte const> row) -> WriteResultTable;

    // The number of entries.
    [[nodiscard]] auto entries() const -> size_t { return entries_.size() / COLUMNS; }

    // Entry r < entries(): its COLUMNS int64s, indexed by Column.
    [[nodiscard]] auto entry(size_t r) const -> Int64s {
        return entries_.subspan(COLUMNS * r, COLUMNS);
    }

    // Replace out's contents with the strings, as views into the row. Their bytes are not checked
    // to be UTF-8: tables come from this job's ranks, which encoded them (see
    // write_result_table.py).
    void split_strings(std::vector<std::string_view>& out) const;

    // The ndim offset dims from position pos of the offsets, checked to lie within them.
    [[nodiscard]] auto offset_dims(size_t pos, int64_t ndim) const -> Int64s;

    // Return how the table names fqns.
    [[nodiscard]] auto fqn_mode() const -> FqnMode { return fqn_mode_; }

    // Return the fingerprint of the fqn list an index table was encoded against.
    [[nodiscard]] auto fingerprint() const -> int64_t { return fingerprint_; }

    // Return the number of strings.
    [[nodiscard]] auto strings() const -> size_t { return string_len_.size(); }

    // Check that the entries' offsets, used up to position pos, are all of them.
    void check_offsets_used(size_t pos) const;

   private:
    // The sections parse found.
    WriteResultTable(Int64s string_len,
                     Int64s entries,
                     Int64s offsets,
                     std::string_view blob,
                     FqnMode fqn_mode,
                     int64_t fingerprint);

    Int64s string_len_;      // the byte length of each string
    Int64s entries_;         // entries() rows of COLUMNS
    Int64s offsets_;         // the entries' offset dims, concatenated
    std::string_view blob_;  // the strings, concatenated
    FqnMode fqn_mode_;
    int64_t fingerprint_;
};

// The tables in the rows of a 2-D uint8 array.
[[nodiscard]] auto parse_tables(Rows const& rows) -> std::vector<WriteResultTable>;

}  // namespace nvrx::checkpoint
