# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A traceback that ends on a message-less exception still has an identity.

``raise EOFError`` closes a traceback with a bare class name and no colon.
When that line is not recognised as the exception summary, the episode falls
back to its own traceback header and L0 publishes a root identity that names
no exception -- every such failure, in every job, collapsing onto one
fingerprint.
"""

from nvidia_resiliency_ext.attribution.restart_agent.l0 import build_l0_bundle


def _interleaved_log() -> str:
    """Two ranks failing together, as one shared file records them.

    Rank 9 raises a message-less ``EOFError``; rank 24 raises a
    ``ConnectionResetError`` carrying a message. Their frames interleave,
    which is the normal shape when every rank writes to one file.
    """
    return (
        "0: iteration 4530/60223 | consumed samples: 10437120\n"
        "24: Traceback (most recent call last):\n"
        '24:   File "/app/loader.py", line 48, in _feed_queue\n'
        "9: Traceback (most recent call last):\n"
        '9:   File "/app/loader.py", line 48, in _feed_queue\n'
        "24:     res = queue.get(timeout=1)\n"
        "9:     res = queue.get(timeout=1)\n"
        "9:     raise EOFError\n"
        "9: EOFError\n"
        "24: ConnectionResetError: [Errno 104] Connection reset by peer\n"
    )


def test_message_less_exception_closes_its_episode(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(_interleaved_log(), encoding="utf-8")

    bundle = build_l0_bundle(log_path)
    by_rank = {episode.exception_rank: episode for episode in bundle.failure_episodes}

    rank9 = by_rank.get("9")
    assert rank9 is not None, "rank 9 raised and must produce an episode"
    # Resolved on the EOFError line, not on the traceback header above it.
    assert rank9.terminal_exception_line != rank9.first_exception_line
    assert "EOFError" in (rank9.terminal_exception_quote or "")
    assert "Traceback" not in (rank9.terminal_exception_quote or "")


def test_the_published_identity_names_an_exception(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text(_interleaved_log(), encoding="utf-8")

    primary = build_l0_bundle(log_path).deterministic_primary_candidate

    assert primary is not None
    # "traceback_most_recent_call_last" is the word "Traceback": it identifies
    # no failure, and every truncated-looking episode would share it.
    assert "traceback_most_recent_call_last" not in (primary.root_fingerprint or "")
    assert "Traceback (most recent call last)" not in (primary.signature or "")


def test_a_frame_line_is_not_mistaken_for_the_summary(tmp_path):
    """``raise EOFError`` sits one line above the summary and must not win."""
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "0: iteration 10/100 | consumed samples: 40\n"
        "9: Traceback (most recent call last):\n"
        '9:   File "/app/loader.py", line 48, in _feed_queue\n'
        "9:     raise EOFError\n"
        "9: EOFError\n",
        encoding="utf-8",
    )

    episodes = build_l0_bundle(log_path).failure_episodes
    rank9 = next(e for e in episodes if e.exception_rank == "9")

    assert "raise" not in (rank9.terminal_exception_quote or "")
    assert (rank9.terminal_exception_quote or "").strip().endswith("EOFError")
