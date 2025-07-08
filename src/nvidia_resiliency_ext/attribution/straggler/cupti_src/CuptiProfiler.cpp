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

#include "CuptiProfiler.h"
#include <pybind11/pybind11.h>
#include <cupti.h>
#include <cuda_runtime_api.h>
#include <sstream>
#include <string>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace py = pybind11;

#define CUPTI_CALL(call) do { \
    CUptiResult _status = call; \
    if (_status != CUPTI_SUCCESS) { \
        const char *errstr; \
        cuptiGetResultString(_status, &errstr); \
        std::cerr << errstr << std::endl; \
        throw std::runtime_error(errstr); \
    } \
} while (0)


// global, single instance of the profiler, which is a static member of CuptiProfiler
CuptiProfiler* CuptiProfiler::instance = nullptr;

static KernelStats computeStats(const std::vector<float>& data) {
    KernelStats stats;
    if (data.empty()) {
        return stats;
    }
    std::vector<float> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    // Calculate min and max
    stats.min = sorted_data.front();
    stats.max = sorted_data.back();

    size_t n = sorted_data.size();
    if (n % 2 == 0) {
        stats.median = (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2;
    } else {
        stats.median = sorted_data[n / 2];
    }

    stats.avg = std::accumulate(sorted_data.begin(), sorted_data.end(), 0.0f) / n;
    float sq_sum = std::accumulate(sorted_data.begin(), sorted_data.end(), 0.0f, 
        [avg=stats.avg](float acc, float x) {
            return acc + (x - avg) * (x - avg);
        }
    );
    stats.stddev = std::sqrt(sq_sum / n);

    stats.num_calls = n;

    return stats;
}

std::string KernelStats::toString() const {
    std::stringstream ss;
    ss << " num calls: " <<  num_calls << ", min: " << min << ", max: " << max << ", median: " << median
    << ", avg: " << avg << ", stddev: " << stddev;
    return ss.str();
}

CuptiProfiler::CuptiProfiler(size_t cuptiBufferSize, size_t cuptiBuffersNum, size_t statsMaxLenPerKernel) 
    : _bufferPool(cuptiBufferSize, cuptiBuffersNum), _statsMaxLenPerKernel(statsMaxLenPerKernel)
{
    if (CuptiProfiler::instance) {
        throw std::runtime_error("Only one CuptiProfiler instance is allowed.");
    }
    CuptiProfiler::instance = this;
}

CuptiProfiler::~CuptiProfiler() {
    CuptiProfiler::instance = nullptr;
}

void CuptiProfiler::initializeProfiling() {
    if (!_isInitialized) {
        CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequestedTrampoline, bufferCompletedTrampoline));
        _isInitialized = true;
    }
    else {
        std::cerr << "CuptiProfiler::initializeProfiling subsequent call." << std::endl;
    }
}

void CuptiProfiler::shutdownProfiling() {
    if (_isInitialized) {
        CUPTI_CALL(cuptiFinalize());
        _isInitialized = false;
    }
    else {
        std::cerr << "CuptiProfiler::shutdownProfiling called while not initialized." << std::endl;
    }
}

void CuptiProfiler::startProfiling() {
    if (!_isStarted) {
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        _isStarted = true;
    }
    else {
        std::cerr << "CuptiProfiler::startProfiling subsequent call." << std::endl;
    }
}

void CuptiProfiler::stopProfiling() {
    if (_isStarted) {
        CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        _isStarted = false;
    }
    else {
        std::cerr << "CuptiProfiler::stopProfiling called while not profiling." << std::endl;
    }
}

std::map<std::string, KernelStats> CuptiProfiler::getStats() {    
    std::map<std::string, KernelStats> result;
    CUPTI_CALL(cuptiActivityFlushAll(0)); // flush CUPTI, note: without the lock or it can deadlock
    std::scoped_lock lk(_kernelDurationsMutex);
    for (const auto& [kernel_name, exec_data] : _kernelDurations) {
        auto last_measurements = exec_data.linearize();
        KernelStats stats = computeStats(last_measurements);
        result[kernel_name] = stats;
    }
    return result;
}

void CuptiProfiler::reset() {
    CUPTI_CALL(cuptiActivityFlushAll(0)); // flush CUPTI, note: without the lock or it can deadlock
    std::scoped_lock lk(_kernelDurationsMutex);
    _kernelDurations.clear();
}

void CUPTIAPI CuptiProfiler::bufferRequestedTrampoline(uint8_t **buffer, size_t *size,size_t *maxNumRecords) {
    CuptiProfiler::instance->bufferRequested(buffer, size, maxNumRecords);
}

void CuptiProfiler::bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    *buffer = _bufferPool.getBuffer();
    *size = _bufferPool.getBufferSize();
    *maxNumRecords = 0;
}

void CUPTIAPI CuptiProfiler::bufferCompletedTrampoline(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
    CuptiProfiler::instance->bufferCompleted(ctx, streamId, buffer, size, validSize);
}

void CuptiProfiler::CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
    
    CUpti_Activity *record = nullptr;
    CUptiResult status = CUPTI_SUCCESS;
    constexpr int KERNEL_NAME_BUF_LEN = 4096;
    char kernel_name_and_dims[KERNEL_NAME_BUF_LEN] = {0};
    
    std::scoped_lock lk(_kernelDurationsMutex);
    
    do {
        status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status == CUPTI_SUCCESS && record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            const auto *kernel = (const CUpti_ActivityKernel4 *)record;
            
            snprintf(kernel_name_and_dims, KERNEL_NAME_BUF_LEN, "%s_blk_%d_%d_%d_grid_%d_%d_%d", 
                kernel->name, 
                kernel->blockX, kernel->blockY, kernel->blockZ,
                kernel->gridX, kernel->gridY, kernel->gridZ);

            const float duration = (kernel->end - kernel->start) / 1000.0f; // Convert to us

            auto kernel_entry_iter = _kernelDurations.find(kernel_name_and_dims);
            if (kernel_entry_iter != _kernelDurations.end()) {
                kernel_entry_iter->second.push_back(duration);
            }
            else {
                auto new_entry = std::make_pair(kernel_name_and_dims, 
                                                CircularBuffer<float>(_statsMaxLenPerKernel));
                new_entry.second.push_back(duration);
                _kernelDurations.emplace(std::move(new_entry));
            }
        }
    } while (status == CUPTI_SUCCESS);

    _bufferPool.releaseBuffer(buffer);
}
