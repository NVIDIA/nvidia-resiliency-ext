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

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace nvrx::checkpoint::detail {

// Transparent hash, so a map keyed by std::string can be looked up by string_view without building
// a std::string per lookup. std::hash<std::string> is not transparent; this hashes both key types
// the same.
struct StringHash {
    using is_transparent = void;  // NOLINT(readability-identifier-naming): the name std looks up
    [[nodiscard]] auto operator()(std::string_view s) const noexcept -> size_t {
        return std::hash<std::string_view>{}(s);
    }
};

// A map keyed by strings that can be looked up by string_view.
template <typename T>
using StringMap = std::unordered_map<std::string, T, StringHash, std::equal_to<>>;

}  // namespace nvrx::checkpoint::detail
