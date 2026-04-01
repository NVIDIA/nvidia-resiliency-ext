# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small path helpers shared across attribution (log validation, FR discovery, etc.)."""

from __future__ import annotations

import os


def path_is_under_allowed_root(candidate: str, allowed_root: str) -> bool:
    """Return True if ``candidate`` resolves under ``allowed_root`` (same rule as log path validation).

    Unlike validating a log file path, this does not require ``candidate`` to exist or be a regular file,
    so it can validate dump path prefixes and inferred directories before reading them.
    """
    if not os.path.isabs(candidate) or not os.path.isabs(allowed_root):
        return False
    try:
        real = os.path.realpath(candidate)
        allowed = os.path.realpath(allowed_root)
        common = os.path.commonpath([real, allowed])
    except (ValueError, OSError):
        return False
    return common == allowed
