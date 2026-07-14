# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval import artifact_io  # noqa: E402


class ArtifactIoTest(unittest.TestCase):
    def test_json_round_trip_is_published_without_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "result.json"

            artifact_io.write_json(path, {"decision": "RESTART"})
            actual = artifact_io.read_json(path)

            self.assertEqual(actual, {"decision": "RESTART"})
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_missing_and_malformed_json_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            malformed = Path(tmp) / "malformed.json"
            malformed.write_text("{", encoding="utf-8")

            missing_result = artifact_io.read_json(missing)
            malformed_result = artifact_io.read_json(malformed)

            self.assertEqual(missing_result, {})
            self.assertEqual(malformed_result, {})

    def test_failed_atomic_replace_removes_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.txt"
            with mock.patch.object(artifact_io.os, "replace", side_effect=OSError("replace")):
                with self.assertRaises(OSError):
                    artifact_io.write_text_atomic(path, "payload")

            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
