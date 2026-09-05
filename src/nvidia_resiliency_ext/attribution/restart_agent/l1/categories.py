# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static failure-category catalog used by the L1 category-selection field.

The 38 category entries live in ``categories.json`` next to this module.
This module loads and validates the JSON at import time, exposing a frozen
``CATEGORIES`` tuple and the ``category_by_id`` accessor. Keeping the data
in JSON lets non-engineers extend the taxonomy without touching Python and
makes the catalog easier to diff, review, and reuse from other tools.

Loading is done via ``importlib.resources`` so the file is discoverable
both from a source checkout and from an installed wheel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

CATEGORIES_SCHEMA_VERSION = "restart_agent_categories.v1"
_ALLOWED_DECISIONS = frozenset({"STOP", "RESTART", "EXCLUDED"})
_REQUIRED_FIELDS = ("id", "name", "description", "decision", "failure_domain", "retry_outlook")


@dataclass(frozen=True)
class CategoryDef:
    """One curated failure-category definition."""

    id: int
    name: str
    description: str
    decision: str  # STOP | RESTART | EXCLUDED
    failure_domain: str  # workload | infrastructure | unknown | "" | "-"
    retry_outlook: str  # may_recover | cannot_recover | "" | "-"


def _load_categories() -> tuple[CategoryDef, ...]:
    """Read and validate categories.json. Raises on any schema violation."""

    with resources.files(__package__).joinpath("categories.json").open("r") as f:
        payload: Any = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("categories.json must be a JSON object")
    if payload.get("schema_version") != CATEGORIES_SCHEMA_VERSION:
        raise ValueError(f"categories.json schema_version must be {CATEGORIES_SCHEMA_VERSION!r}")
    raw = payload.get("categories")
    if not isinstance(raw, list) or not raw:
        raise ValueError("categories.json 'categories' must be a non-empty array")

    entries: list[CategoryDef] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"categories[{index}] must be an object")
        missing = [k for k in _REQUIRED_FIELDS if k not in item]
        if missing:
            raise ValueError(f"categories[{index}] missing required fields: {', '.join(missing)}")
        cid = item["id"]
        if not isinstance(cid, int) or isinstance(cid, bool) or cid < 1:
            raise ValueError(f"categories[{index}].id must be a positive integer")
        if cid in seen_ids:
            raise ValueError(f"categories[{index}].id={cid} is a duplicate")
        expected_id = index + 1
        if cid != expected_id:
            raise ValueError(
                f"categories[{index}].id must be {expected_id} (ids are 1-based sequential)"
            )
        name = item["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"categories[{index}].name must be a non-empty string")
        if name in seen_names:
            raise ValueError(f"categories[{index}].name={name!r} is a duplicate")
        description = item["description"]
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"categories[{index}].description must be a non-empty string")
        decision = item["decision"]
        if decision not in _ALLOWED_DECISIONS:
            raise ValueError(
                f"categories[{index}].decision={decision!r} must be one of "
                f"{sorted(_ALLOWED_DECISIONS)}"
            )
        failure_domain = item["failure_domain"]
        if not isinstance(failure_domain, str):
            raise ValueError(f"categories[{index}].failure_domain must be a string")
        retry_outlook = item["retry_outlook"]
        if not isinstance(retry_outlook, str):
            raise ValueError(f"categories[{index}].retry_outlook must be a string")
        entries.append(
            CategoryDef(
                id=cid,
                name=name,
                description=description,
                decision=decision,
                failure_domain=failure_domain,
                retry_outlook=retry_outlook,
            )
        )
        seen_ids.add(cid)
        seen_names.add(name)
    return tuple(entries)


CATEGORIES: tuple[CategoryDef, ...] = _load_categories()
_BY_ID: dict[int, CategoryDef] = {entry.id: entry for entry in CATEGORIES}


def category_by_id(cid: int) -> CategoryDef | None:
    """Return the curated category for ``cid`` if known, else ``None``."""

    if not isinstance(cid, int) or isinstance(cid, bool):
        return None
    return _BY_ID.get(cid)


__all__ = ["CATEGORIES", "CATEGORIES_SCHEMA_VERSION", "CategoryDef", "category_by_id"]
