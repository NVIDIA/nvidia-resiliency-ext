# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A compile failure on a few ranks is not a broken toolchain.

Category 29 stops the job because the compilation is deterministic
in-container. That reasoning requires the failure to be universal. When two
ranks of four thousand fail to read a cache artifact and the rest compile the
same kernel, the artifact was bad, not the toolchain, and the next attempt
rewrites it.
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


def test_a_transient_cache_artifact_restarts():
    assert _by_id(39)["decision"] == "RESTART"


def test_a_universally_reproducible_toolchain_fault_still_stops():
    assert _by_id(29)["decision"] == "STOP"


def test_the_rank_fraction_is_required_not_advisory():
    description = _by_id(39)["description"]

    assert "required" in description
    assert "observed rank count" in description


def test_each_category_sends_the_other_case_away():
    """Neither may silently absorb the other's cases."""
    assert "cat 39" in _by_id(29)["description"]
    assert "cat 29" in _by_id(39)["description"]


def test_category_29_names_the_signatures_it_requires():
    description = _by_id(29)["description"]

    # Narrowed: it had become a general "something in Triton" bucket.
    assert "Unknown key: cubin" in description
    assert "StaticCudaLauncher" in description
    assert "Requires one of those two" in description


def test_progress_does_not_rule_out_a_cache_artifact_failure():
    """Compilation recurs whenever a new shape is traced, not only at warmup."""
    description = _by_id(39)["description"]

    assert "Zero completed iterations does not argue against this" in description


def test_category_ids_stay_unique():
    ids = [c.id for c in CATEGORIES]

    assert len(ids) == len(set(ids))


def test_the_discriminator_does_not_hardcode_a_job_size():
    """Rank counts differ per job; an absolute count from one incident
    would be measured against jobs of a completely different size."""
    import re

    for category_id in (29, 39, 40):
        description = _by_id(category_id)["description"]
        literals = [
            n
            for n in re.findall(r"\b\d{2,}\b", description)
            # cross-references to other categories are not rank counts
            if f"cat {n}" not in description
        ]
        assert not literals, f"cat {category_id} hardcodes {literals}"
