# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A compiler that never ran is not a compiler that produced a bad artifact.

Both appear as "the JIT failed on a few ranks". One leaves an empty or
unusable artifact behind (cat 39); the other never produces one, and the
compiler's own diagnostics are not in the log, so what went wrong is not
recoverable from it (cat 40).
"""

from nvidia_resiliency_ext.attribution.restart_agent.l1.categories import CATEGORIES


def _by_id(category_id: int) -> dict:
    """Read the packaged catalog, not a path relative to the caller's cwd."""
    category = next(c for c in CATEGORIES if c.id == category_id)
    return {
        "id": category.id,
        "name": category.name,
        "description": category.description,
        "decision": category.decision,
    }


def test_an_undiagnosable_invocation_failure_restarts():
    """Following cat 33 and cat 34: undecidable from one log resolves to retry."""
    assert _by_id(40)["decision"] == "RESTART"


def test_it_records_that_recurrence_is_not_determinable():
    """The description states a property of the failure. Why that property
    leads to a retry is the decision field's business, not the prose's."""
    description = _by_id(40)["description"]

    assert "not determinable from this log" in description


def test_the_rank_fraction_rules_out_the_deterministic_reading():
    description = _by_id(40)["description"]

    assert "required" in description
    assert "every rank that attempts the same build" in description
    assert "observed rank count" in description


def test_it_records_that_the_compiler_diagnostics_are_missing():
    """The reason this cannot be decided is itself part of the category."""
    description = _by_id(40)["description"]

    assert "never recorded" in description


def test_the_two_jit_categories_point_at_each_other():
    assert "cat 40" in _by_id(39)["description"]
    assert "cat 39" in _by_id(40)["description"]


def test_it_separates_itself_from_the_universal_toolchain_fault():
    assert "cat 29" in _by_id(40)["description"]


def test_category_ids_stay_unique_and_sequential():
    ids = [c.id for c in CATEGORIES]

    assert ids == list(range(1, len(ids) + 1))
