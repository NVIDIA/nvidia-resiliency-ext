# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for cycle-scoped log and evidence artifact paths."""

from __future__ import annotations

import os


def insert_suffix_before_ext(path: str, suffix: str) -> str:
    """Insert ``suffix`` before the file extension of ``path``."""
    base_without_ext, ext = os.path.splitext(path)
    return f"{base_without_ext}{suffix}{ext}"


def get_source_cycle_log_file(path_prefix: str, source_name: str, cycle_index: int) -> str:
    """Build a source-specific cycle logfile path from ``path_prefix``."""
    return insert_suffix_before_ext(path_prefix, f"_{source_name}_cycle{cycle_index}")
