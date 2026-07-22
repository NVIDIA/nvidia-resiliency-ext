"""Harness-owned machine-readable artifact contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REVIEW_SUMMARY_SCHEMA_VERSION = "restart_agent_review.v1"
PANEL_SUMMARY_SCHEMA_VERSION = "restart_agent_panel.v1"


def require_schema(
    payload: Mapping[str, Any],
    expected: str,
    *,
    artifact: str,
) -> None:
    """Reject missing or unexpected harness artifact versions."""

    actual = payload.get("schema_version")
    if actual != expected:
        raise ValueError(f"{artifact} schema_version must be {expected!r}, got {actual!r}")
