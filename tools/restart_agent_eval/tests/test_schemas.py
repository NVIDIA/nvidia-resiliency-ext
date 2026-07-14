# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval import schemas  # noqa: E402


class SchemaContractTest(unittest.TestCase):
    def test_accepts_exact_schema_version(self) -> None:
        actual = schemas.require_schema(
            {"schema_version": "artifact.v1"},
            "artifact.v1",
            artifact="result.json",
        )

        self.assertIsNone(actual)

    def test_rejects_missing_and_unexpected_schema_versions(self) -> None:
        for payload in ({}, {"schema_version": "artifact.v2"}):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    schemas.require_schema(payload, "artifact.v1", artifact="result.json")


if __name__ == "__main__":
    unittest.main()
