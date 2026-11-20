# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cycle records and their transport to the CycleAnalyzer.

The store host publishes how each cycle started and the launchers publish how it ended
for them; the CycleAnalyzer reads the records and writes the job-wide cycle info. The
records are the messages of nvrx_interface.proto; messages.py builds and checks them.
"""

from .messages import build_cycle_info, build_node_cycle_end_report, ended_cycle_info, now_ms
from .transport import CycleInfoPublisher, CycleInfoReader

__all__ = [
    "CycleInfoPublisher",
    "CycleInfoReader",
    "build_cycle_info",
    "build_node_cycle_end_report",
    "ended_cycle_info",
    "now_ms",
]
