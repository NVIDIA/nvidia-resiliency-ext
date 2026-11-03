#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the NVRx Scheduler Exclusion Service as a standalone Python zipapp."""

from __future__ import annotations

import argparse
import shutil
import stat
import subprocess
import sys
import tempfile
import zipapp
from pathlib import Path

_ARTIFACT_NAME = "nvrx-scheduler-exclusion-service.pyz"
_ENTRY_POINT = "nvidia_resiliency_ext.services.scheduler_exclusions.__main__:main"
_RUNTIME_FILES = (
    "__init__.py",
    "__main__.py",
    "config.py",
    "decision_file.py",
    "monitor.py",
    "server.py",
)


def _parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the standalone NVRx Scheduler Exclusion Service"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "dist" / _ARTIFACT_NAME,
        help="Output .pyz path (default: %(default)s)",
    )
    return parser


def build_zipapp(repo_root: Path, output: Path) -> Path:
    """Build and validate a zipapp containing only the service runtime."""
    source_root = repo_root / "src" / "nvidia_resiliency_ext"
    service_source = source_root / "services" / "scheduler_exclusions"
    missing = [name for name in _RUNTIME_FILES if not (service_source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing Scheduler Exclusion sources: {missing}")

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nvrx-scheduler-exclusion-") as temp_dir:
        staging = Path(temp_dir)
        package_root = staging / "nvidia_resiliency_ext"
        service_target = package_root / "services" / "scheduler_exclusions"
        service_target.mkdir(parents=True)
        shutil.copy2(source_root / "__init__.py", package_root / "__init__.py")
        shutil.copy2(
            source_root / "services" / "__init__.py",
            package_root / "services" / "__init__.py",
        )
        for name in _RUNTIME_FILES:
            shutil.copy2(service_source / name, service_target / name)

        zipapp.create_archive(
            staging,
            target=output,
            interpreter="/usr/bin/env python3",
            main=_ENTRY_POINT,
            compressed=True,
        )

    output.chmod(output.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    subprocess.run(  # nosec B603 - validates the artifact built above
        [sys.executable, str(output), "--help"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return output


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    args = _parser(repo_root).parse_args()
    artifact = build_zipapp(repo_root, args.output)
    print(artifact)


if __name__ == "__main__":
    main()
