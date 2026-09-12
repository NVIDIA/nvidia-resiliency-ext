# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path


def test_builds_minimal_standalone_zipapp(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    builder = repo_root / "services" / "scheduler_exclusions" / "build_zipapp.py"
    artifact = tmp_path / "scheduler-exclusion.pyz"

    subprocess.run(  # nosec B603
        [sys.executable, str(builder), "--output", str(artifact)],
        check=True,
        cwd=tmp_path,
    )

    assert artifact.stat().st_mode & stat.S_IXUSR
    with zipfile.ZipFile(artifact) as archive:
        names = set(archive.namelist())
    assert "__main__.py" in names
    assert "nvidia_resiliency_ext/services/scheduler_exclusions/decision_file.py" in names
    assert "nvidia_resiliency_ext/services/scheduler_exclusions/monitor.py" in names
    assert not any("fault_tolerance" in name for name in names)

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(  # nosec B603
        [str(artifact), "--help"],
        check=True,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert "NVRx Scheduler Exclusion Service" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--scheduler-exclusion-dir" not in result.stdout

    env["NVRX_SCHEDULER_EXCLUSION_ARTIFACT"] = str(artifact)
    env["NVRX_SCHEDULER_EXCLUSION_PYTHON"] = sys.executable
    launcher = repo_root / "services" / "scheduler_exclusions" / "deploy" / "run_service.sh"
    result = subprocess.run(  # nosec B603
        [str(launcher), "--help"],
        check=True,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert "NVRx Scheduler Exclusion Service" in result.stdout
