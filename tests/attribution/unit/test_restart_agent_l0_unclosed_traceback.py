# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A traceback that never closes must not outrank one that did.

Episodes are ordered by where their identity sits in the file. An episode whose
traceback never reached an exception anchors on its own header - and a header
is necessarily earlier than the exception line of any episode that did close.
So in a purely positional order, failing to close wins, and the analysis
reports "Traceback (most recent call last):" while another rank's real error
sits a few lines below.
"""

from nvidia_resiliency_ext.attribution.restart_agent.l0 import build_l0_bundle


def _interleaved(tmp_path, *, closed_first: bool) -> str:
    """Two ranks fail; one closes on an exception, one is cut off mid-traceback."""
    closed = [
        "2048: [rank2048]: Traceback (most recent call last):",
        '2048: [rank2048]:   File "/app/train.py", line 1, in <module>',
        "2048: [rank2048]: RuntimeError: (TP, PP) mismatch after resume",
    ]
    unclosed = [
        "54: [rank54]: Traceback (most recent call last):",
        '54: [rank54]:   File "/app/train.py", line 1, in <module>',
    ]
    first, second = (closed, unclosed) if closed_first else (unclosed, closed)
    log = tmp_path / "run_2200100_date_x_cycle0.log"
    log.write_text("0: iteration 10/1000 | loss 2.0\n" + "\n".join(first + second) + "\n")
    return str(log)


def test_the_closed_traceback_wins_even_when_it_comes_second(tmp_path):
    primary = build_l0_bundle(_interleaved(tmp_path, closed_first=False))
    quote = primary.deterministic_primary_candidate.quote or ""

    assert "TP, PP) mismatch" in quote, "the rank that named its failure describes it better"
    assert "Traceback (most recent call last)" not in quote


def test_source_order_still_decides_between_two_closed_episodes(tmp_path):
    """The rank ordering is unchanged where both episodes actually closed."""
    log = tmp_path / "run_2200100_date_x_cycle0.log"
    log.write_text(
        "0: iteration 10/1000 | loss 2.0\n"
        "11: [rank11]: Traceback (most recent call last):\n"
        "11: [rank11]: ValueError: first failure\n"
        "12: [rank12]: Traceback (most recent call last):\n"
        "12: [rank12]: ValueError: second failure\n"
    )

    quote = build_l0_bundle(str(log)).deterministic_primary_candidate.quote or ""

    assert "first failure" in quote


def test_an_all_unclosed_log_still_reports_something(tmp_path):
    """When nothing closed there is no better anchor, so the header stands."""
    log = tmp_path / "run_2200100_date_x_cycle0.log"
    log.write_text(
        "0: iteration 10/1000 | loss 2.0\n"
        "54: [rank54]: Traceback (most recent call last):\n"
        '54: [rank54]:   File "/app/train.py", line 1, in <module>\n'
    )

    primary = build_l0_bundle(str(log)).deterministic_primary_candidate

    assert primary is not None
