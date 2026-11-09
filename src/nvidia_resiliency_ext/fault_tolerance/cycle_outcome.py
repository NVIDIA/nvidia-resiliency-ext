# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Moved to `cycle_info.records`; kept only for `cycle_info_writer` until the CycleAnalyzer
takes it over. Do not import from here in new code."""

from .cycle_info.records import CycleEndReason, LocalCycleOutcome

__all__ = ["CycleEndReason", "LocalCycleOutcome"]
