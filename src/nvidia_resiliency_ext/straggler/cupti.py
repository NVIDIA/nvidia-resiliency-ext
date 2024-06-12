# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading


class CuptiManager:
    """Provide thread safe access to the CUPTI extension.

    Implements simple usage counter, to track active profiling runs.
    """

    def __init__(self, bufferSize=1_000_000, numBuffers=8, statsMaxLenPerKernel=4096):
        """
        Args:
            bufferSize (int, optional): CUPTI buffer size. Defaults to 1MB.
            numBuffers (int, optional): Num of CUPTI buffers in a pool . Defaults to 8.
            statsMaxLenPerKernel (int, optional): Max number of timing entries per kernel.
                (when this limit is rached, oldest timing entries are discarded). Defaults to 4096.
        """

        # lazy load the extension module, to avoid circular import
        import nvrx_cupti_module as cupti_module  # type: ignore

        self.cupti_ext = cupti_module.CuptiProfiler(
            bufferSize=bufferSize,
            numBuffers=numBuffers,
            statsMaxLenPerKernel=statsMaxLenPerKernel,
        )
        self.is_initialized = False
        self.started_cnt = 0
        self.lock = threading.Lock()

    def _ensure_initialized(self):
        """Check for CuptiProfiler initialization."""
        if not self.is_initialized:
            raise RuntimeError("CuptiManager was not initialized")

    def initialize(self):
        """Call CuptiProfiler initialization method, registering CUPTI
        callbacks for profiling."""
        with self.lock:
            self.cupti_ext.initialize()
            self.is_initialized = True

    def shutdown(self):
        """Finalize CUPTI."""
        with self.lock:
            self.cupti_ext.shutdown()
            self.is_initialized = False
            self.started_cnt = 0

    def start_profiling(self):
        """Enable CUDA kernels activity tracking."""
        with self.lock:
            self._ensure_initialized()
            if self.started_cnt == 0:
                self.cupti_ext.start()
            self.started_cnt += 1

    def stop_profiling(self):
        """Disable CUDA kernels activity tracking."""
        with self.lock:
            self._ensure_initialized()
            if self.started_cnt > 0:
                self.started_cnt -= 1
                if self.started_cnt == 0:
                    self.cupti_ext.stop()
            else:
                raise RuntimeError("No active profiling run.")

    def get_results(self):
        """Calculate kernel execution timing statistics."""
        with self.lock:
            self._ensure_initialized()
            stats = self.cupti_ext.get_stats()
            return stats.copy()

    def reset_results(self):
        """Reset kernel execution timing records."""
        with self.lock:
            self._ensure_initialized()
            self.cupti_ext.reset()
