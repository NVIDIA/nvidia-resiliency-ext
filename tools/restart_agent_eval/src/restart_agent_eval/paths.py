"""Stable filesystem locations and environment resolution for the harness."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent
TOOL_ROOT = SOURCE_ROOT.parent
REPO_ROOT = TOOL_ROOT.parents[1]


def path_from_env(
    name: str,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    resolved_environment = os.environ if environment is None else environment
    value = resolved_environment.get(name)
    return Path(value).expanduser() if value else None


def product_repo_from_env(environment: Mapping[str, str] | None = None) -> Path:
    return path_from_env("NVRX_RESTART_AGENT_PRODUCT_REPO", environment) or REPO_ROOT
