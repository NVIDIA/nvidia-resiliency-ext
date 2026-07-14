"""Repository provenance behind the harness process boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .product_process import ProcessExecutor, SubprocessExecutor


def git_identity(
    repo: Path | None,
    *,
    process_executor: ProcessExecutor | None = None,
) -> dict[str, Any] | None:
    if repo is None:
        return None
    executor = process_executor or SubprocessExecutor()
    try:
        commit = executor.run(["git", "rev-parse", "HEAD"], cwd=repo)
        dirty = executor.run(["git", "status", "--porcelain"], cwd=repo)
    except OSError:
        return {"path": str(repo), "commit": None, "dirty": None}
    return {
        "path": str(repo),
        "commit": (commit.stdout.strip() or None) if commit.returncode == 0 else None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }
