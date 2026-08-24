# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared JSON and atomic text I/O for generated harness artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol


class ArtifactStore(Protocol):
    """Read and publish generated harness artifacts."""

    def read_json(self, path: Path) -> Any: ...

    def write_json(self, path: Path, payload: Any) -> None: ...

    def write_text(self, path: Path, text: str) -> None: ...


class LocalArtifactStore:
    """Filesystem artifact store using atomic publication."""

    def read_json(self, path: Path) -> Any:
        return read_json(path)

    def write_json(self, path: Path, payload: Any) -> None:
        write_json(path, payload)

    def write_text(self, path: Path, text: str) -> None:
        write_text_atomic(path, text)


def read_json(path: Path) -> Any:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: Any) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", errors="replace") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


LOCAL_ARTIFACT_STORE = LocalArtifactStore()
