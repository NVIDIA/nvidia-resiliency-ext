# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Review run layout, manifest publication, and normalized context tests."""

from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _bootstrap import configure_test_imports

configure_test_imports()

from _assertions import assert_json_file, assert_mapping_fields  # noqa: E402
from restart_agent_eval import review as review_log  # noqa: E402
from restart_agent_eval.artifacts import resolve_artifact_layout  # noqa: E402
from restart_agent_eval.review_context import ReviewContext  # noqa: E402


class ReviewArtifactTest(unittest.TestCase):
    def test_run_id_uses_injected_clock(self) -> None:
        class _Clock:
            def now_utc(self):
                return dt.datetime(2026, 7, 18, 5, 7, 26, 238429, tzinfo=dt.timezone.utc)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            layout = resolve_artifact_layout(
                log_path=root / "logs" / "case.log",
                log_root=root / "logs",
                gold_root=root / "gold",
                run_root=root / "runs",
                run_dir=None,
                gold_label=None,
                clock=_Clock(),
            )

        self.assertEqual(layout.run_id, "20260718T050726238429Z")

    def test_mirrors_relative_log_path_across_three_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_root = root / "logs"
            gold_root = root / "restart_agent_gold"
            run_root = root / "restart_agent_runs"
            log_path = log_root / "checkpoint_logs" / "training.log"

            layout = resolve_artifact_layout(
                log_path=log_path,
                log_root=log_root,
                gold_root=gold_root,
                run_root=run_root,
                run_dir=None,
                gold_label=None,
            )

            self.assertEqual(
                layout.relative_log_path,
                Path("checkpoint_logs/training.log"),
            )
            self.assertEqual(
                layout.gold_path,
                (gold_root / "checkpoint_logs" / "training.log" / "gold.json").resolve(),
            )
            self.assertEqual(
                layout.run_dir.parent,
                (run_root / "checkpoint_logs" / "training.log").resolve(),
            )
            self.assertRegex(layout.run_id, r"^\d{8}T\d{12}Z$")

    def test_review_context_normalizes_loaded_payloads_without_files(self) -> None:
        context = ReviewContext.from_payloads(
            {"decision": "RESTART"},
            {
                "schema_version": "restart_agent_cli_trace.v1",
                "request": {},
                "analyzer_trace": {},
                "l0_bundle": {},
                "analysis_result": {"decision": "STOP"},
            },
        )

        self.assertEqual(context.result["decision"], "RESTART")
        self.assertEqual(context.analysis["decision"], "STOP")

    def test_rejects_log_outside_declared_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(SystemExit):
                resolve_artifact_layout(
                    log_path=root / "other" / "training.log",
                    log_root=root / "logs",
                    gold_root=root / "gold",
                    run_root=root / "runs",
                    run_dir=None,
                    gold_label=None,
                )

    def test_rejects_overlapping_artifact_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(SystemExit):
                resolve_artifact_layout(
                    log_path=root / "logs" / "training.log",
                    log_root=root / "logs",
                    gold_root=root / "gold",
                    run_root=root / "logs" / "generated-runs",
                    run_dir=None,
                    gold_label=None,
                )

    def test_run_manifest_records_source_and_repository_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_root = root / "logs"
            log_path = log_root / "checkpoint_logs" / "training.log"
            log_path.parent.mkdir(parents=True)
            log_path.write_text("line one\n", encoding="utf-8")
            layout = resolve_artifact_layout(
                log_path=log_path,
                log_root=log_root,
                gold_root=root / "gold",
                run_root=root / "runs",
                run_dir=None,
                gold_label=None,
            )
            layout.run_dir.mkdir(parents=True)
            with mock.patch.object(
                review_log,
                "_git_identity",
                return_value={"path": "/repo", "commit": "abc123", "dirty": False},
            ):
                review_log.write_review_index(
                    layout.run_dir,
                    log_path,
                    root,
                    [],
                    layout=layout,
                )

            manifest = assert_json_file(
                self,
                layout.run_dir / "run_manifest.json",
                required_fields=("source", "repositories", "gold_attached"),
            )
            assert_mapping_fields(
                self,
                manifest["source"],
                {
                    "relative_path": "checkpoint_logs/training.log",
                    "byte_size": 9,
                },
            )
            self.assertEqual(len(manifest["source"]["sha256"]), 64)
            assert_mapping_fields(
                self,
                manifest["repositories"]["product"],
                {"commit": "abc123", "dirty": False},
            )
            self.assertFalse(manifest["gold_attached"])


if __name__ == "__main__":
    unittest.main()
