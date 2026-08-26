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

Only the names that cross the boundary. NVRx's own span and attribute names are
literals at their call sites; a name is promoted here when a *second repository*
has to spell it identically for a query to join, at which point two hardcoded
copies of one string is a rename waiting to break silently.

This module imports nothing. It is safe to import with nemo-lens absent, with
telemetry disabled, and from a process that never initializes telemetry -- which
is the point, since the caller on the other side is a training framework that
should not have to reason about NVRx's optional dependencies to spell a key.
"""

#: Identifies one async checkpoint save, across the three spans that see it.
#:
#: **NVRx assigns this value; the framework propagates it.** It is minted by
#: ``AsyncCallsQueue.schedule_async_request`` and returned to the caller, and
#: returned again as the finalized list from ``maybe_finalize_async_calls``.
#: NVRx stamps it on the trainer's schedule span, the worker's request span and
#: the finalize span; the framework is expected to stamp whichever value it
#: received on whatever span is active when it receives it.
#:
#: The three spans cannot share a trace -- the worker is a different process and
#: the finalize happens iterations later -- so this attribute, not parentage, is
#: what relates them.
CKPT_CALL_IDX = "nvrx.call_idx"

#: The training iteration a checkpoint belongs to, read from Baggage.
#:
#: **The framework sets this; NVRx only reads it.** NVRx takes it out of Baggage
#: at enqueue and sets it on the schedule, request and finalize spans, so a
#: checkpoint can be attributed to the step that asked for it without NVRx
#: knowing anything about training. Absent Baggage, the attribute is absent and
#: nothing else changes.
ITERATION = "nvrx.iteration"
