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

#pragma once

#include <vector>
#include <cstdint>
#include <mutex>

class BufferPool {
public:
    BufferPool(size_t bufferSize = 1024 * 1024 * 4, int numBuffers = 20);
    ~BufferPool();

    uint8_t* getBuffer();
    void releaseBuffer(uint8_t* buffer);
    size_t getBufferSize() const;

private:
    std::vector<uint8_t*> freeBuffers;
    size_t bufferSize;
    int numBuffers;
    std::mutex mutex;
};
