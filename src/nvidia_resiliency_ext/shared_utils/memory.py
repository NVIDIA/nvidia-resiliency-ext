# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import os

from .health_check import PynvmlMixin, with_pynvml_lock
from .log_manager import LogConfig


class GPUMemoryLogger(PynvmlMixin):
    """
    Utility class for logging GPU memory information using pynvml.
    Implemented as a singleton to avoid repeated pynvml availability checks.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPUMemoryLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if hasattr(self, '_initialized'):
            return
        super().__init__()
        self.log = logging.getLogger(LogConfig.name)
        self._pynvml_available = self.check_pynvml_availability()
        self._initialized = True

    @with_pynvml_lock
    def get_gpu_memory(self, device_index: int = None, log_memory: bool = False, context: str = ""):
        """
        Get GPU memory usage (used, free, total) for the specified device using pynvml.

        Uses the following device detection pattern:
        1. Use device_index if provided
        2. Otherwise, check LOCAL_RANK environment variable
        3. Raise RuntimeError if neither is available

        Args:
            device_index (int): Optional GPU device index. If None, uses LOCAL_RANK env variable.
            log_memory (bool): If True, log the memory information. Default: False.
            context (str): Optional context string to include in the log message (only used if log_memory=True).

        Returns:
            dict: Dictionary with keys 'total_mb', 'used_mb', 'free_mb' or None if pynvml not available
        """
        if not self._pynvml_available:
            return None

        try:
            # Determine device: use device_index if provided, otherwise use LOCAL_RANK
            if device_index is not None:
                device_id = device_index
            else:
                local_rank = os.getenv('LOCAL_RANK')
                if local_rank is None:
                    raise RuntimeError("Neither device_index provided nor LOCAL_RANK set")
                device_id = int(local_rank)

            self.pynvml.nvmlInit()

            # Get handle for the determined device
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_id)

            # Get memory information (v2 separates reserved memory from free memory)
            memory_info = self.pynvml.nvmlDeviceGetMemoryInfo(
                handle, version=self.pynvml.nvmlMemory_v2
            )

            # Convert bytes to MB for readability
            total_mb = memory_info.total / (1024 * 1024)
            used_mb = memory_info.used / (1024 * 1024)
            free_mb = memory_info.free / (1024 * 1024)

            # Log if requested
            if log_memory:
                context_str = f" ({context})" if context else ""
                self.log.info(
                    f"GPU {device_id} Memory{context_str} - Total: {total_mb:.2f} MB, "
                    f"Used: {used_mb:.2f} MB, Free: {free_mb:.2f} MB"
                )

            return {
                'total_mb': total_mb,
                'used_mb': used_mb,
                'free_mb': free_mb,
            }

        except Exception as e:
            # Include device_index in error for clarity
            device_str = f" (device_index={device_index})" if device_index is not None else ""
            self.log.error(f"Failed to get GPU memory information{device_str}: {e}")
            return None
        finally:
            try:
                self.pynvml.nvmlShutdown()
            except Exception as e:
                self.log.warning(f"Error during NVML shutdown: {e}")
