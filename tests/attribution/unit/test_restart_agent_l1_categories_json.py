# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for JSON-backed L1 category taxonomy loading and schema validation.

The taxonomy lives in ``l1/categories.json`` and is loaded + validated at
import time by ``l1/categories.py``. These tests verify the on-disk schema
holds and that the loader rejects malformed payloads.
"""

import json
from importlib import resources

import pytest

from nvidia_resiliency_ext.attribution.restart_agent.l1 import categories as categories_module
from nvidia_resiliency_ext.attribution.restart_agent.l1.categories import (
    CATEGORIES,
    CATEGORIES_SCHEMA_VERSION,
    CategoryDef,
    category_by_id,
)

# ---------------------------------------------------------------------------
# On-disk JSON shape
# ---------------------------------------------------------------------------


def _load_raw_json() -> dict:
    path = resources.files("nvidia_resiliency_ext.attribution.restart_agent.l1").joinpath(
        "categories.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_categories_json_uses_declared_schema_version():
    payload = _load_raw_json()
    assert payload["schema_version"] == CATEGORIES_SCHEMA_VERSION


def test_categories_json_has_at_least_one_entry():
    payload = _load_raw_json()
    assert isinstance(payload["categories"], list)
    assert len(payload["categories"]) >= 1


# ---------------------------------------------------------------------------
# CATEGORIES tuple properties
# ---------------------------------------------------------------------------


def test_categories_ids_are_sequential_1_based():
    ids = [c.id for c in CATEGORIES]
    assert ids == list(range(1, len(CATEGORIES) + 1))


def test_categories_names_are_unique():
    names = [c.name for c in CATEGORIES]
    assert len(names) == len(set(names))


def test_categories_decisions_are_within_allowed_set():
    allowed = {"STOP", "RESTART", "EXCLUDED"}
    for c in CATEGORIES:
        assert c.decision in allowed, f"{c.id} ({c.name}) has invalid decision {c.decision!r}"


def test_categories_json_and_module_agree_on_size():
    payload = _load_raw_json()
    assert len(payload["categories"]) == len(CATEGORIES)


def test_category_by_id_returns_expected_entry():
    entry = category_by_id(1)
    assert isinstance(entry, CategoryDef)
    assert entry.id == 1


def test_category_by_id_returns_none_on_placeholder_and_out_of_range():
    assert category_by_id(0) is None
    assert category_by_id(-1) is None
    assert category_by_id(len(CATEGORIES) + 1) is None
    assert category_by_id(True) is None  # noqa: E712 -- bools rejected


# ---------------------------------------------------------------------------
# Loader rejects malformed payloads
# ---------------------------------------------------------------------------


def _run_loader_on(payload) -> None:
    """Invoke the private loader with a monkeypatched resources.open()."""
    # Directly call the validator by simulating a broken payload through the
    # same code path. We reuse the module's constants for the required-fields
    # sanity checks.
    import json as _json
    from importlib import resources as _resources

    orig = _resources.files

    class _Files:
        def __init__(self, target):
            self._target = target

        def joinpath(self, name):
            return self

        def open(self, *args, **kwargs):
            import io

            return io.StringIO(_json.dumps(payload))

    def fake_files(pkg):
        return _Files(pkg)

    _resources.files = fake_files
    try:
        categories_module._load_categories()
    finally:
        _resources.files = orig


def test_loader_rejects_wrong_schema_version():
    payload = {"schema_version": "not_the_right_version", "categories": []}
    with pytest.raises(ValueError, match="schema_version"):
        _run_loader_on(payload)


def test_loader_rejects_empty_categories():
    payload = {"schema_version": CATEGORIES_SCHEMA_VERSION, "categories": []}
    with pytest.raises(ValueError, match="non-empty array"):
        _run_loader_on(payload)


def test_loader_rejects_duplicate_id():
    entry = {
        "id": 1,
        "name": "one",
        "description": "d",
        "decision": "RESTART",
        "failure_domain": "workload",
        "retry_outlook": "may_recover",
    }
    payload = {
        "schema_version": CATEGORIES_SCHEMA_VERSION,
        "categories": [entry, dict(entry, name="two")],
    }
    with pytest.raises(ValueError, match="duplicate"):
        _run_loader_on(payload)


def test_loader_rejects_non_sequential_id():
    payload = {
        "schema_version": CATEGORIES_SCHEMA_VERSION,
        "categories": [
            {
                "id": 5,
                "name": "gap",
                "description": "d",
                "decision": "RESTART",
                "failure_domain": "workload",
                "retry_outlook": "may_recover",
            }
        ],
    }
    with pytest.raises(ValueError, match="1-based sequential"):
        _run_loader_on(payload)


def test_loader_rejects_duplicate_name():
    entry = {
        "name": "same",
        "description": "d",
        "decision": "RESTART",
        "failure_domain": "workload",
        "retry_outlook": "may_recover",
    }
    payload = {
        "schema_version": CATEGORIES_SCHEMA_VERSION,
        "categories": [dict(entry, id=1), dict(entry, id=2)],
    }
    with pytest.raises(ValueError, match="duplicate"):
        _run_loader_on(payload)


def test_loader_rejects_invalid_decision():
    payload = {
        "schema_version": CATEGORIES_SCHEMA_VERSION,
        "categories": [
            {
                "id": 1,
                "name": "bad",
                "description": "d",
                "decision": "MAYBE",
                "failure_domain": "workload",
                "retry_outlook": "may_recover",
            }
        ],
    }
    with pytest.raises(ValueError, match="decision"):
        _run_loader_on(payload)


def test_loader_rejects_missing_field():
    payload = {
        "schema_version": CATEGORIES_SCHEMA_VERSION,
        "categories": [
            {
                "id": 1,
                "name": "bad",
                "description": "d",
                "decision": "RESTART",
                # missing failure_domain
                "retry_outlook": "may_recover",
            }
        ],
    }
    with pytest.raises(ValueError, match="missing required fields"):
        _run_loader_on(payload)
