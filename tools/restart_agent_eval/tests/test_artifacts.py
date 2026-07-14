# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact layout validation, explicit overrides, and root-boundary tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval.artifacts import resolve_artifact_layout  # noqa: E402


class ArtifactLayoutTest(unittest.TestCase):
    def test_explicit_run_dir_allows_single_log_without_managed_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "input.log"
            run_dir = root / "explicit-run"

            layout = resolve_artifact_layout(
                log_path=log_path,
                log_root=None,
                gold_root=None,
                run_root=None,
                run_dir=run_dir,
                gold_label=None,
                run_id="run-1",
            )

        self.assertEqual(layout.relative_log_path, Path("input.log"))
        self.assertEqual(layout.run_dir, run_dir.resolve())
        self.assertEqual(layout.run_id, "run-1")
        self.assertIsNone(layout.gold_path)

    def test_explicit_gold_label_overrides_mirrored_gold_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            explicit_gold = root / "labels" / "case.json"

            layout = resolve_artifact_layout(
                log_path=root / "logs" / "case.log",
                log_root=root / "logs",
                gold_root=root / "gold",
                run_root=root / "runs",
                run_dir=None,
                gold_label=explicit_gold,
                run_id="run-1",
            )

        self.assertEqual(layout.gold_path, explicit_gold.resolve())

    def test_managed_layout_requires_run_and_gold_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = {
                "log_path": root / "logs" / "case.log",
                "log_root": root / "logs",
                "run_dir": None,
                "gold_label": None,
            }
            for missing_field in ("run_root", "gold_root"):
                values = {
                    **common,
                    "run_root": root / "runs",
                    "gold_root": root / "gold",
                }
                values[missing_field] = None
                with self.subTest(missing_field=missing_field):
                    with self.assertRaises(SystemExit):
                        resolve_artifact_layout(**values)

    def test_gold_root_without_log_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(SystemExit):
                resolve_artifact_layout(
                    log_path=root / "case.log",
                    log_root=None,
                    gold_root=root / "gold",
                    run_root=root / "runs",
                    run_dir=root / "explicit-run",
                    gold_label=None,
                )

    def test_equal_and_ancestor_roots_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for gold_root in (root / "logs", root / "logs" / "gold"):
                with self.subTest(gold_root=gold_root):
                    with self.assertRaises(SystemExit):
                        resolve_artifact_layout(
                            log_path=root / "logs" / "case.log",
                            log_root=root / "logs",
                            gold_root=gold_root,
                            run_root=root / "runs",
                            run_dir=None,
                            gold_label=None,
                        )


if __name__ == "__main__":
    unittest.main()
