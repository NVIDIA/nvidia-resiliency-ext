# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable assertions for structured harness artifacts and payloads."""

from __future__ import annotations

import json
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def assert_mapping_fields(
    case: unittest.TestCase,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Assert selected mapping fields while preserving per-field diagnostics."""
    for field, expected_value in expected.items():
        with case.subTest(field=field):
            case.assertIn(field, actual)
            case.assertEqual(actual[field], expected_value)


def assert_paths_exist(case: unittest.TestCase, paths: Sequence[Path]) -> None:
    """Assert every expected artifact path exists with path-specific diagnostics."""
    for path in paths:
        with case.subTest(path=path):
            case.assertTrue(path.is_file(), f"expected artifact file: {path}")


def assert_json_file(
    case: unittest.TestCase,
    path: Path,
    *,
    required_fields: Sequence[str] = (),
) -> dict[str, Any]:
    """Assert a file contains a JSON object with the requested top-level fields."""
    case.assertTrue(path.is_file(), f"expected JSON artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        case.fail(f"unable to read valid JSON from {path}: {error}")
    case.assertIsInstance(payload, dict)
    for field in required_fields:
        with case.subTest(path=path, field=field):
            case.assertIn(field, payload)
    return payload
