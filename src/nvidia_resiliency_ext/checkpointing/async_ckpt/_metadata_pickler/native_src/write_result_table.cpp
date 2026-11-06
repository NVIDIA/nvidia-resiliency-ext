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

#include "write_result_table.h"

#include <pybind11/pybind11.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace nvrx::checkpoint;
using namespace nvrx::checkpoint::detail;

namespace {

// Write-result tables, as write_result_table.py encodes them: little-endian int64s, header first.
static_assert(std::endian::native == std::endian::little, "write-result tables are little-endian");
constexpr int64_t TABLE_MAGIC = 0x314C42545852564E;  // the bytes "NVRXTBL1", read little-endian
constexpr int64_t TABLE_VERSION = 2;
enum class Header : uint8_t {
    Magic,
    Version,
    Entries,
    Strings,
    StringBytes,
    Offsets,
    FqnMode,
    Fingerprint
};
constexpr size_t HEADER = 8;  // int64s

}  // namespace

void detail::bad_table(std::string_view what) {
    throw py::value_error("write-result table: " + std::string(what));
}

WriteResultTable WriteResultTable::parse(std::span<const std::byte> row) {
    if (row.size() < sizeof(int64_t) * HEADER) {
        bad_table("truncated");
    }
    const Int64s header(row.first(sizeof(int64_t) * HEADER));
    if (header[Header::Magic] != TABLE_MAGIC || header[Header::Version] != TABLE_VERSION) {
        bad_table("not a table of this version");
    }
    std::span<const std::byte> rest = row.subspan(sizeof(int64_t) * HEADER);
    // The next section: as many items of item_size bytes as the header's count field says,
    // checked to fit in the rest of the row (comparing counts, so sizes cannot overflow).
    const auto take = [&](Header count, size_t item_size) {
        const int64_t n = header[count];
        if (n < 0 || static_cast<uint64_t>(n) > rest.size() / item_size) {
            bad_table("truncated");
        }
        const auto section = rest.first(static_cast<size_t>(n) * item_size);
        rest = rest.subspan(section.size());
        return section;
    };
    const Int64s string_len(take(Header::Strings, sizeof(int64_t)));
    const Int64s entries(take(Header::Entries, sizeof(int64_t) * COLUMNS));
    const Int64s offsets(take(Header::Offsets, sizeof(int64_t)));
    const std::span<const std::byte> blob = take(Header::StringBytes, 1);
    const int64_t fqn_mode = header[Header::FqnMode];
    if (fqn_mode != static_cast<int64_t>(FqnMode::Strings) &&
        fqn_mode != static_cast<int64_t>(FqnMode::Indices)) {
        bad_table("unknown fqn mode");
    }
    return WriteResultTable(string_len, entries, offsets,
                            {reinterpret_cast<const char*>(blob.data()), blob.size()},
                            static_cast<FqnMode>(fqn_mode), header[Header::Fingerprint]);
}

void WriteResultTable::split_strings(std::vector<std::string_view>& out) const {
    out.clear();
    std::string_view rest = blob_;
    for (size_t k = 0; k < string_len_.size(); ++k) {
        const int64_t len = string_len_[k];
        if (len < 0 || static_cast<uint64_t>(len) > rest.size()) {
            bad_table("string lengths exceed the string bytes");
        }
        out.push_back(rest.substr(0, static_cast<size_t>(len)));
        rest.remove_prefix(static_cast<size_t>(len));
    }
    if (!rest.empty()) {
        bad_table("string lengths don't match the string bytes");
    }
}

Int64s WriteResultTable::offset_dims(size_t pos, int64_t ndim) const {
    if (ndim < 0 || pos > offsets_.size() ||
        static_cast<uint64_t>(ndim) > offsets_.size() - pos) {
        bad_table("offsets out of range");
    }
    return offsets_.subspan(pos, static_cast<size_t>(ndim));
}

void WriteResultTable::check_offsets_used(size_t pos) const {
    if (pos != offsets_.size()) {
        bad_table("offset dims don't match the offsets");
    }
}

WriteResultTable::WriteResultTable(Int64s string_len, Int64s entries, Int64s offsets,
                                   std::string_view blob, FqnMode fqn_mode, int64_t fingerprint)
    : string_len_(string_len),
      entries_(entries),
      offsets_(offsets),
      blob_(blob),
      fqn_mode_(fqn_mode),
      fingerprint_(fingerprint) {}

std::vector<WriteResultTable> nvrx::checkpoint::parse_tables(const Rows& rows) {
    if (rows.ndim() != 2) {
        throw py::value_error("expected a 2-D array of write-result tables");
    }
    const size_t width = static_cast<size_t>(rows.shape(1));
    const std::span<const std::byte> bytes =
        std::as_bytes(std::span(rows.data(), static_cast<size_t>(rows.size())));
    std::vector<WriteResultTable> tables;
    tables.reserve(rows.shape(0));
    for (size_t r = 0; r < static_cast<size_t>(rows.shape(0)); ++r) {
        tables.push_back(WriteResultTable::parse(bytes.subspan(width * r, width)));
    }
    return tables;
}
