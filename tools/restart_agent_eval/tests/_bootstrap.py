# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared import-path setup for pytest, unittest discovery, and direct test runs."""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


def configure_test_imports() -> None:
    """Make the harness source tree importable without installing the package."""
    source = str(SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)
