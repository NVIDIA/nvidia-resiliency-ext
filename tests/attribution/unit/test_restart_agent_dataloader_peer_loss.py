# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A worker that vanished is not a workload bug and not a worker I/O error.

The feeder thread's view is all the log keeps: the worker is gone, and why it
went is not recorded. Read as a workload fault it stops a job that would have
run; read as a worker I/O failure it claims an error the log never shows.
"""

from nvidia_resiliency_ext.attribution.restart_agent.l1.categories import CATEGORIES


def _category(category_id: int):
    return next(c for c in CATEGORIES if c.id == category_id)


def test_a_vanished_worker_restarts():
    assert _category(41).decision == "RESTART"


def test_it_names_the_ipc_frames_the_log_actually_shows():
    """These frames are visible to L1 whether or not L0 resolved the
    exception, which is what lets the category work on its own."""
    description = _category(41).description

    for frame in (
        "_feed_queue",
        "queue.get",
        "rebuild_storage_fd",
        "resource_sharer.detach",
        "recv_bytes",
    ):
        assert frame in description, f"missing frame: {frame}"


def test_it_covers_the_message_less_exception():
    """EOFError raised without arguments is the common terminator here."""
    description = _category(41).description

    assert "EOFError" in description
    assert "no message at all" in description


def test_it_refuses_the_workload_code_reading():
    """The failure that this category exists to stop being mistaken for."""
    description = _category(41).description

    assert "Not cat 28" in description
    assert "no workload code raised here" in description


def test_it_defers_to_a_visible_worker_io_failure():
    """cat 15 owns the case where the worker reports its own I/O error."""
    description = _category(41).description

    assert "Not cat 15" in description
    for signal in ("Bus error", "/dev/shm"):
        assert signal in description


def test_it_claims_more_than_the_off_file_answer_without_denying_it():
    """cat 3 is not wrong here, only less specific."""
    description = _category(41).description

    assert "cat 3" in description
    assert "both are true" in description


def test_the_description_carries_no_policy_reasoning():
    """The decision field holds the disposition; the prose is for matching."""
    description = _category(41).description

    for policy_word in (
        "progress guard",
        "retry budget",
        "policy layer",
        "bounded choice",
        "RESTART",
    ):
        assert policy_word not in description, f"policy wording: {policy_word}"


def test_category_ids_stay_unique_and_sequential():
    ids = [c.id for c in CATEGORIES]

    assert ids == list(range(1, len(ids) + 1))
