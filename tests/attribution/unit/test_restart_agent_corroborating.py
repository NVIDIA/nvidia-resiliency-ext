# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The seam that asks sources sitting beside the log rather than inside it.

A run can end for a reason the application never records. The deciding fact is
usually written down somewhere else under the same root, and a source that
cannot answer must cost nothing.
"""

from __future__ import annotations

from nvidia_resiliency_ext.services.attrsvc.corroborating import (
    DEFAULT_PROBES,
    CorroboratedFinding,
    first_finding,
)


def test_a_source_that_raises_is_not_an_answer():
    """These read shared storage that can be absent or momentarily unreadable.
    None of that is grounds for losing a verdict."""

    def boom(*_args):
        raise OSError("lustre hiccup")

    def answers(*_args):
        return CorroboratedFinding(
            source="second", category_id=1, decision="RESTART", basis="b", summary="s"
        )

    found = first_finding([boom, answers], log_path="/x", job_id="1", cycle_id=0, allowed_root=None)

    assert found is not None and found.source == "second"


def test_nothing_found_is_the_normal_case():
    assert (
        first_finding([lambda *_a: None], log_path="/x", job_id="1", cycle_id=0, allowed_root=None)
        is None
    )


def test_the_registered_sources_are_asserted_not_assumed():
    """Sources are added one per change, so the registry is pinned rather than
    left to grow silently."""

    assert [p.__name__ for p in DEFAULT_PROBES] == [
        "probe_section_timeout",
        "probe_scheduler_cancellation",
    ]
