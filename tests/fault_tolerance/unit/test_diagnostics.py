# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from nvidia_resiliency_ext.fault_tolerance.diagnostics import (
    capture_full_core_dump_once,
    capture_stack_trace,
)


def test_full_core_dump_is_captured_only_once_per_claim(tmp_path):
    claim_file = tmp_path / "core.claim"
    logger = MagicMock()

    def fake_run(command, **kwargs):
        prefix = Path(command[command.index("-o") + 1])
        Path(f"{prefix}.123").touch()
        return subprocess.CompletedProcess(command, 0, stdout="saved")

    with (
        patch(
            "nvidia_resiliency_ext.fault_tolerance.diagnostics.shutil.which",
            side_effect=lambda name: f"/usr/bin/{name}",
        ),
        patch(
            "nvidia_resiliency_ext.fault_tolerance.diagnostics.subprocess.run",
            side_effect=fake_run,
        ) as run,
    ):
        first = capture_full_core_dump_once(
            pid=123,
            rank=4,
            cycle=2,
            dump_dir=str(tmp_path),
            timeout=30,
            claim_file=str(claim_file),
            logger=logger,
        )
        second = capture_full_core_dump_once(
            pid=456,
            rank=5,
            cycle=2,
            dump_dir=str(tmp_path),
            timeout=30,
            claim_file=str(claim_file),
            logger=logger,
        )

    assert first is not None
    assert Path(first).exists()
    assert second is None
    run.assert_called_once()


def test_failed_full_core_releases_claim_for_fallback_rank(tmp_path):
    claim_file = tmp_path / "core.claim"
    logger = MagicMock()

    with (
        patch(
            "nvidia_resiliency_ext.fault_tolerance.diagnostics.shutil.which",
            side_effect=lambda name: f"/usr/bin/{name}",
        ),
        patch(
            "nvidia_resiliency_ext.fault_tolerance.diagnostics.subprocess.run",
            return_value=subprocess.CompletedProcess([], 1, stdout="attach denied"),
        ),
    ):
        result = capture_full_core_dump_once(
            pid=123,
            rank=4,
            cycle=2,
            dump_dir=str(tmp_path),
            timeout=30,
            claim_file=str(claim_file),
            logger=logger,
        )

    assert result is None
    assert not claim_file.exists()


def test_stack_trace_writes_gdb_output(tmp_path):
    logger = MagicMock()

    def fake_run(command, **kwargs):
        kwargs["stdout"].write("Thread 1\n#0 checkpoint_save()\n")
        return subprocess.CompletedProcess(command, 0)

    with (
        patch(
            "nvidia_resiliency_ext.fault_tolerance.diagnostics.shutil.which",
            return_value="/usr/bin/gdb",
        ),
        patch(
            "nvidia_resiliency_ext.fault_tolerance.diagnostics.subprocess.run",
            side_effect=fake_run,
        ),
    ):
        result = capture_stack_trace(
            pid=123,
            rank=4,
            cycle=2,
            dump_dir=str(tmp_path),
            timeout=30,
            logger=logger,
        )

    assert result is not None
    assert "checkpoint_save" in Path(result).read_text(encoding="utf-8")
