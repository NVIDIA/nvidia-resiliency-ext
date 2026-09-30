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
"""

SPAN_GROUP_STARTUP = "nv.nvrx.ftl.python"
"""Process startup and import spans."""

SPAN_GROUP_FT = "nv.nvrx.ftl"
"""Fault-tolerance operation spans."""

SPAN_GROUP_CKPT = "nv.nvrx.ckpt"
"""Checkpoint scheduling, worker request, and finalization spans."""

SPAN_GROUP_CKPT_PHASES = "nv.nvrx.ckpt.save"
"""Checkpoint stage and completion synchronization spans."""

CKPT_CALL_IDX = "nv.nvrx.ckpt.call_idx"
"""Attribute key for the queue-local index shared by schedule, request, and finalize spans."""
