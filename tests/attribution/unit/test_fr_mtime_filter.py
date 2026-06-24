# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from nvidia_resiliency_ext.attribution.base import NVRxAttribution
from nvidia_resiliency_ext.attribution.trace_analyzer import fr_attribution
from nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution import (
    CollectiveAnalyzer,
    _filter_files_by_min_mtime,
)


def test_collective_analyzer_filters_dump_files_older_than_fr_min_mtime(
    tmp_path,
    monkeypatch,
):
    old_dump = tmp_path / "_dump_0"
    fresh_dump = tmp_path / "_dump_1"
    old_dump.write_text("old\n", encoding="utf-8")
    fresh_dump.write_text("fresh\n", encoding="utf-8")
    os.utime(old_dump, (100.0, 100.0))
    os.utime(fresh_dump, (200.0, 200.0))

    analyzer = CollectiveAnalyzer(
        {
            "fr_path": str(tmp_path),
            "pattern": "_dump_*",
            "verbose": False,
            "health_check": False,
            "llm_analyze": False,
            "threshold": None,
            "fr_min_mtime": 150.0,
        }
    )
    processed: list[str] = []

    monkeypatch.setattr(
        analyzer,
        "process_file",
        lambda filepath: processed.append(os.path.basename(filepath)) or True,
    )
    monkeypatch.setattr(analyzer, "group_collectives_by_windows", lambda: {})
    monkeypatch.setattr(analyzer, "analyze_matches", lambda verbose=False: ({}, {}))

    try:
        analyzer._loop.run_until_complete(analyzer.preprocess_FR_dumps())
    finally:
        NVRxAttribution.reset_thread_event_loop()

    assert processed == ["_dump_1"]


def test_fr_min_mtime_filter_reports_stale_and_unavailable_mtime_separately(
    tmp_path,
    caplog,
):
    old_dump = tmp_path / "_dump_0"
    fresh_dump = tmp_path / "_dump_1"
    missing_dump = tmp_path / "_dump_2"
    old_dump.write_text("old\n", encoding="utf-8")
    fresh_dump.write_text("fresh\n", encoding="utf-8")
    os.utime(old_dump, (100.0, 100.0))
    os.utime(fresh_dump, (200.0, 200.0))

    caplog.set_level(logging.INFO, logger=fr_attribution.__name__)

    kept = _filter_files_by_min_mtime(
        [str(old_dump), str(fresh_dump), str(missing_dump)],
        150.0,
    )

    assert kept == [str(fresh_dump)]
    assert "filtered_stale=1" in caplog.text
    assert "skipped_unavailable=1" in caplog.text
