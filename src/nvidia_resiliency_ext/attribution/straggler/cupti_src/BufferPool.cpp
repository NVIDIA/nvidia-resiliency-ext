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

#include "BufferPool.h"
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <mutex>


BufferPool::BufferPool(size_t bufferSize, int numBuffers)
    : bufferSize(bufferSize), numBuffers(numBuffers) {
    for (int i = 0; i < numBuffers; ++i) {
        uint8_t* newBuffer = (uint8_t*)malloc(bufferSize);
        if (!newBuffer) {
            throw std::bad_alloc();
        }
        freeBuffers.push_back(newBuffer);
    }
}
    
BufferPool::~BufferPool() {
    while (!freeBuffers.empty()) {
        free(freeBuffers.back());
        freeBuffers.pop_back();
    }
}

uint8_t* BufferPool::getBuffer() {
    std::lock_guard<std::mutex> lock(mutex);
    if (freeBuffers.empty()) {
        return nullptr;  // prob better to allocate a new buffer 
    }
    uint8_t* buffer = freeBuffers.back();
    freeBuffers.pop_back();
    return buffer;
}

void BufferPool::releaseBuffer(uint8_t* buffer) {
    std::lock_guard<std::mutex> lock(mutex);
    freeBuffers.push_back(buffer);
}

size_t BufferPool::getBufferSize() const {
    return bufferSize;
}
