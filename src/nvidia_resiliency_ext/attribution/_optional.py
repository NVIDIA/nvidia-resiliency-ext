# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for attribution optional dependency messaging."""

from __future__ import annotations

_OPTIONAL_ATTRIBUTION_MODULES = frozenset(
    {
        "langchain_openai",
        "logsage",
        "mcp",
        "setproctitle",
        "fastapi",
        "uvicorn",
        "pydantic",
        "pydantic_settings",
        "slowapi",
        "slack_bolt",
        "slack_sdk",
    }
)


def reraise_if_missing_attribution_dependency(exc: ModuleNotFoundError, *, feature: str) -> None:
    """Raise a clearer error when attribution extras are not installed."""
    if exc.name not in _OPTIONAL_ATTRIBUTION_MODULES:
        return

    raise ModuleNotFoundError(
        f"{feature} requires optional attribution dependencies that are not installed. "
        "Install them with `pip install nvidia-resiliency-ext[attribution]` "
        f"(missing module: {exc.name})."
    ) from exc
