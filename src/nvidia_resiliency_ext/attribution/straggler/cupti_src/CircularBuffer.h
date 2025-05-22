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

template <typename T>
class CircularBuffer {
private:
    std::vector<T> _buffer;
    size_t _head;
    size_t _tail;
    size_t _size;
    size_t _capacity;

public:
    explicit CircularBuffer(size_t capacity=32) : 
        _buffer(capacity), _head(0), _tail(0), _size(0), _capacity(capacity) {}

    ~CircularBuffer() = default;

    bool empty() const {
        return _size == 0;
    }

    bool full() const {
        return _size == _capacity;
    }

    size_t size() const {
        return _size;
    }

    size_t capacity() const {
        return _capacity;
    }

    void push_back(const T& value) {
        _buffer[_tail] = value;
        _tail = (_tail + 1) % _capacity;
        if (full()) {
            _head = (_head + 1) % _capacity;
        } else {
            _size++;
        }
    }

    std::vector<T> linearize() const {
        std::vector<T> res(_size);
        for (size_t linearIndex = 0; linearIndex < _size; linearIndex++) {
            res[linearIndex] = _buffer[(_head + linearIndex) % _capacity];
        }
        return res;
    }
};
