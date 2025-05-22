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

#include <map>
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <cmath>

#include <pybind11/pybind11.h>
#include <cupti.h>
#include <cuda_runtime_api.h>
#include <numeric>
#include <mutex>


#include "BufferPool.h"
#include "CircularBuffer.h"

namespace py = pybind11;

struct KernelStats {
    KernelStats() : num_calls(0), min(NAN), max(NAN), median(NAN), avg(NAN), stddev(NAN) {
    }
    int num_calls;
    float min, max, median, avg, stddev;
    std::string toString() const;
};

class CuptiProfiler {
public:
    CuptiProfiler(size_t cuptiBufferSize = 1024 * 1024 * 8, 
                  size_t cuptiBuffersNum = 8,
                  size_t statsMaxLenPerKernel = 1024);
    ~CuptiProfiler();

    void initializeProfiling();
    void shutdownProfiling();

    void startProfiling();
    void stopProfiling();

    void reset();
    
    std::map<std::string, KernelStats> getStats();

private:
    static CuptiProfiler* instance; 
    BufferPool _bufferPool;
    std::unordered_map<std::string, CircularBuffer<float>> _kernelDurations;
    std::mutex _kernelDurationsMutex;
    size_t _statsMaxLenPerKernel;
    bool _isInitialized {false};
    bool _isStarted {false};

    void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
    void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);

    static void CUPTIAPI bufferRequestedTrampoline(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
    static void CUPTIAPI bufferCompletedTrampoline(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);
};
