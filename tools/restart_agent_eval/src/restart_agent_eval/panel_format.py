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


def _primary_label(row: dict[str, Any], stage: str) -> str:
    fine_class = row.get(f"{stage}_primary_class")
    line = row.get(f"{stage}_primary_line")
    if fine_class is None and line is None:
        return "-"
    return f"{_md(fine_class)}@{_md(line)}"


def _md(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")
