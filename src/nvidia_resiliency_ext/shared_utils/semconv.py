# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Telemetry names NVRx shares with the framework that drives it.

Only names a second repository has to spell identically for a query to join.
Two hardcoded copies of one string is a rename waiting to break silently.

Imports nothing, so a caller need not reason about NVRx's optional dependencies.
"""

#: Identifies one async checkpoint save. NVRx assigns it and returns it from
#: schedule/finalize; the framework stamps what it received on its active span.
#: The three spans cannot share a trace, so this relates them, not parentage.
CKPT_CALL_IDX = "nvrx.call_idx"

#: The training iteration a checkpoint belongs to. The framework puts it in
#: Baggage; NVRx only reads it. Absent Baggage, the attribute is simply absent.
ITERATION = "nvrx.iteration"
