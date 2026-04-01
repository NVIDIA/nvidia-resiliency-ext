# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test/diagnostic helpers to capture logger output or stdout (used by FR / trace tooling)."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from io import StringIO
from typing import Optional


@contextmanager
def capture_logs(logger_name: Optional[str] = None):
    logger = logging.getLogger(logger_name)
    original_handlers = logger.handlers.copy()
    log_capture = StringIO()
    capture_handler = logging.StreamHandler(log_capture)
    logger.handlers = [capture_handler]

    try:
        yield log_capture
    finally:
        logger.handlers = original_handlers


@contextmanager
def capture_stdout(logger_name: Optional[str] = None):
    del logger_name  # unused; kept for backward-compatible signature
    output = StringIO()
    original_stdout = sys.stdout
    sys.stdout = output

    try:
        yield output
    finally:
        sys.stdout = original_stdout
