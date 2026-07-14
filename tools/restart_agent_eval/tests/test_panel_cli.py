# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Panel CLI discovery, schema validation, output overrides, and empty-input tests."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval import panel  # noqa: E402
from restart_agent_eval.schemas import REVIEW_SUMMARY_SCHEMA_VERSION  # noqa: E402


def _review(target="deterministic"):
    return {
        "schema_version": REVIEW_SUMMARY_SCHEMA_VERSION,
        "target": target,
        "run_label": target,
        "analysis": {},
        "artifacts": {},
    }


class PanelCliTest(unittest.TestCase):
    def test_missing_directory_and_empty_run_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path in (root / "missing", root):
                with self.subTest(path=path):
                    with self.assertRaises(SystemExit):
                        panel.main([str(path)])

    def test_cli_writes_custom_outputs_and_quiet_suppresses_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "deterministic.review.json").write_text(
                json.dumps(_review()), encoding="utf-8"
            )
            json_out = run_dir / "custom" / "panel.json"
            md_out = run_dir / "custom" / "panel.md"
            json_out.parent.mkdir()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = panel.main(
                    [
                        str(run_dir),
                        "--json-out",
                        str(json_out),
                        "--md-out",
                        str(md_out),
                        "--quiet",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(output.getvalue(), "")
            self.assertTrue(json_out.is_file())
            self.assertTrue(md_out.is_file())
            self.assertTrue((run_dir / "panel_diagnostics.md").is_file())

    def test_cli_rejects_review_with_wrong_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "bad.review.json").write_text(
                json.dumps({**_review(), "schema_version": "review.v99"}),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                panel.main([str(run_dir)])

    def test_index_and_files_are_deduplicated_and_sorted_by_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            index_summary = _review("gpt")
            (run_dir / "review_index.json").write_text(
                json.dumps({"runs": [index_summary, "ignored"]}),
                encoding="utf-8",
            )
            (run_dir / "duplicate.review.json").write_text(
                json.dumps(index_summary), encoding="utf-8"
            )
            (run_dir / "deterministic.review.json").write_text(
                json.dumps(_review("deterministic")), encoding="utf-8"
            )

            panel.main([str(run_dir), "--quiet"])
            payload = json.loads((run_dir / "panel_summary.json").read_text(encoding="utf-8"))

        self.assertEqual([row["target"] for row in payload["rows"]], ["deterministic", "gpt"])


if __name__ == "__main__":
    unittest.main()
