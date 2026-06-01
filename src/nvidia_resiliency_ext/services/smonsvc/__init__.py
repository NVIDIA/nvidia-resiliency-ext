# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SLURM Job Monitor with Attribution Service Integration.

A standalone program that monitors SLURM jobs and integrates with the
nvrx-attrsvc attribution service for log analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import JobState, MonitorState, SlurmJob

if TYPE_CHECKING:
    from .monitor import SlurmJobMonitor

__all__ = [
    "SlurmJobMonitor",
    "SlurmJob",
    "JobState",
    "MonitorState",
]


def __getattr__(name: str):
    if name == "SlurmJobMonitor":
        from .monitor import SlurmJobMonitor

        return SlurmJobMonitor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
