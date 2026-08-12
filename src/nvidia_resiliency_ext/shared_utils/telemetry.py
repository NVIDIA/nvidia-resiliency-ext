# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

try:
    from nemo.lens import NemoLensConfig as _NemoLensConfig
    from nemo.lens import managed_span
    from nemo.lens import setup_telemetry as _setup_telemetry

    def setup_telemetry(rank: int, world_size: int):
        return _setup_telemetry(_NemoLensConfig.from_env(), rank, world_size)

except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        yield None

    class _NoOpHandle:
        def shutdown(self):
            pass

    def setup_telemetry(rank: int, world_size: int):
        return _NoOpHandle()
