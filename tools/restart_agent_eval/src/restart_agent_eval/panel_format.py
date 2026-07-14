# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small display-only value formatters for panel Markdown."""

from __future__ import annotations

from typing import Any


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _yes_no(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return _md(value)


def _short_identity(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.rsplit(":", 1)[-1][:12]


def _affected_entity_label(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    kind = value.get("kind")
    identity = value.get("identity")
    return f"{kind}:{identity}" if kind and identity else None


def _primary_label(row: dict[str, Any], stage: str) -> str:
    failure_class = row.get(f"{stage}_primary_class")
    line = row.get(f"{stage}_primary_line")
    if failure_class is None and line is None:
        return "-"
    return f"{_md(failure_class)}@{_md(line)}"


def _md(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")
