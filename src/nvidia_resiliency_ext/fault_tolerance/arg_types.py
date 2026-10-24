# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""argparse types for fault tolerance options.

Values are checked once, when the command line is parsed, so the config dataclasses trust
them.
"""

import argparse
import math
import os


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be finite and positive, got {text!r}")
    return value


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text!r}")
    return value


def absolute_path(text):
    if not os.path.isabs(text):
        raise argparse.ArgumentTypeError(f"must be an absolute path, got {text!r}")
    return text


def names(text):
    """A comma-separated list of unique, nonempty names."""
    values = [name.strip() for name in text.split(",")]
    if not all(values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(
            f"must be a comma-separated list of unique, nonempty names, got {text!r}"
        )
    return values
