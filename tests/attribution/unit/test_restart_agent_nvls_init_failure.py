# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A node that cannot acquire a fabric resource at init is not a peer failure.

One node fails to bind NVLS multicast memory; its ranks die with an unhandled
cuda error, and every surviving rank then reports "remote process exited".
The peer reports are the cascade. Attributing them as the cause describes the
wrong failure and routes a deterministic, node-bound condition to restart.
"""

from nvidia_resiliency_ext.attribution.restart_agent.identity import extract_node
from nvidia_resiliency_ext.attribution.restart_agent.l1.categories import CATEGORIES

_NVLS_LINE = (
    "6976: [2026-10-28 18:18:18] host-a01:1251976:1251976 [0] "
    "transport/nvls.cc:376 (nvlsAllocateMem) NCCL WARN Failed to bind "
    "NVLink SHARP (NVLS) Multicast memory of size 2097152"
)


def _category(category_id: int):
    return next(c for c in CATEGORIES if c.id == category_id)


def test_the_nccl_line_prefix_identifies_the_failing_node():
    # "<host>:<pid>:<tid>" is the NCCL prefix; without this the node that went
    # first is invisible and repeated failures on one host cannot be seen.
    assert extract_node(_NVLS_LINE) == "host-a01"


def test_a_source_reference_is_not_mistaken_for_a_node():
    # "nvls.cc:376" and a bare timestamp both carry colons.
    assert extract_node("rank 5 at transport/nvls.cc:376 (nvlsAllocateMem)") is None
    assert extract_node("[2026-10-28 18:18:18] starting up") is None


def test_the_explicit_node_form_still_works():
    assert extract_node("detected failure on node=gpu-0123") == "gpu-0123"


def test_fabric_allocation_failure_is_excluded():
    """The workload is healthy and schedulable elsewhere; isolating the
    hardware belongs to Health Check and FACT, not to this analysis."""
    category = _category(42)

    assert category.decision == "EXCLUDED"


def test_the_category_requires_zero_completed_iterations():
    """The allocation happens during setup, so progress rules this category out."""
    category = _category(42)

    assert "Zero completed iterations is required" in category.description


def test_the_category_separates_itself_from_the_peer_cascade():
    category = _category(42)

    # Says explicitly which neighbouring categories it must not be confused with.
    assert "cat 13" in category.description
    assert "cat 7" in category.description


def test_category_ids_stay_unique():
    ids = [c.id for c in CATEGORIES]

    assert len(ids) == len(set(ids))


def test_the_description_describes_the_failure_not_the_policy():
    """The decision field carries the disposition. The prose is what the
    model reads to decide whether a log matches, so policy reasoning in it
    is noise at best."""
    description = _category(42).description

    for policy_word in (
        "Health Check",
        "FACT",
        "progress guard",
        "retry budget",
        "policy layer",
        "EXCLUDED",
        "excluded",
    ):
        assert policy_word not in description, f"policy wording leaked: {policy_word}"
