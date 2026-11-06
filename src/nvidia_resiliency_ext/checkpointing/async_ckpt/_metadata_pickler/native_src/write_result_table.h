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
    explicit Int64s(std::span<const std::byte> bytes) : bytes_(bytes) {}

    // The number of int64s.
    [[nodiscard]] size_t size() const { return bytes_.size() / sizeof(int64_t); }

    // The i-th int64, i < size(). bit_cast needs an object to read, and there is none at an
    // arbitrary offset of the row, so the bytes are copied into one; this compiles to one load.
    [[nodiscard]] int64_t operator[](size_t i) const {
        std::array<std::byte, sizeof(int64_t)> v;
        std::ranges::copy(bytes_.subspan(sizeof(int64_t) * i).first<sizeof(int64_t)>(), v.begin());
        return std::bit_cast<int64_t>(v);
    }

    // The field of a record (a header or an entry) that an enum names.
    template <typename Field>
        requires std::is_enum_v<Field>
    [[nodiscard]] int64_t operator[](Field field) const {
        return (*this)[static_cast<size_t>(field)];
    }

    // The count int64s from pos, pos + count <= size().
    [[nodiscard]] Int64s subspan(size_t pos, size_t count) const {
        return Int64s(bytes_.subspan(sizeof(int64_t) * pos, sizeof(int64_t) * count));
    }

   private:
    std::span<const std::byte> bytes_;
};

// A write-result table within its row of the gathered buffer. parse checks that its sections lie
// within the row; the accessors check what its entries refer to.
class WriteResultTable {
   public:
    // The table in a row.
    [[nodiscard]] static WriteResultTable parse(std::span<const std::byte> row);

    // The number of entries.
    [[nodiscard]] size_t entries() const { return entries_.size() / COLUMNS; }

    // Entry r < entries(): its COLUMNS int64s, indexed by Column.
    [[nodiscard]] Int64s entry(size_t r) const { return entries_.subspan(COLUMNS * r, COLUMNS); }

    // Replace out's contents with the strings, as views into the row. Their bytes are not checked
    // to be UTF-8: tables come from this job's ranks, which encoded them (see
    // write_result_table.py).
    void split_strings(std::vector<std::string_view>& out) const;

    // The ndim offset dims from position pos of the offsets, checked to lie within them.
    [[nodiscard]] Int64s offset_dims(size_t pos, int64_t ndim) const;

    // Return how the table names fqns.
    [[nodiscard]] FqnMode fqn_mode() const { return fqn_mode_; }

    // Return the fingerprint of the fqn list an index table was encoded against.
    [[nodiscard]] int64_t fingerprint() const { return fingerprint_; }

    // Return the number of strings.
    [[nodiscard]] size_t strings() const { return string_len_.size(); }

    // Check that the entries' offsets, used up to position pos, are all of them.
    void check_offsets_used(size_t pos) const;

   private:
    // The sections parse found.
    WriteResultTable(Int64s string_len, Int64s entries, Int64s offsets, std::string_view blob,
                     FqnMode fqn_mode, int64_t fingerprint);

    Int64s string_len_;  // the byte length of each string
    Int64s entries_;     // entries() rows of COLUMNS
    Int64s offsets_;     // the entries' offset dims, concatenated
    std::string_view blob_;  // the strings, concatenated
    FqnMode fqn_mode_;
    int64_t fingerprint_;
};

// The tables in the rows of a 2-D uint8 array.
[[nodiscard]] std::vector<WriteResultTable> parse_tables(const Rows& rows);

}  // namespace nvrx::checkpoint
