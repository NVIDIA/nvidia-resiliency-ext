# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failures that never raise still have to name themselves.

A launcher rejected for a bad flag and a fabric error reported in driver prose
both reach the end of a log with no traceback and no exception class anywhere
in it. The terminal-episode ladder recognized only typed surfaces, so these
produced no episode and therefore no primary at all - the analysis reported
nothing rather than reporting the obvious line.
"""

import pytest

from nvidia_resiliency_ext.attribution.restart_agent.l0 import build_l0_bundle
from nvidia_resiliency_ext.attribution.restart_agent.l0.assembly import _terminal_episode_priority


@pytest.mark.parametrize(
    "line",
    [
        "pretrain.py: error: unrecognized arguments: --save-main-grads-interval 1",
        "trainer.py: error: argument --micro-batch-size: invalid int value: 'x'",
        "trainer.py: error: the following arguments are required: --tokenizer",
        "what():  CUDA error: uncorrectable NVLink error detected during the execution",
        "Warning: CUDA warning: uncorrectable NVLink error detected during the execution",
    ],
)
def test_a_failure_that_never_raised_can_still_anchor(line):
    assert _terminal_episode_priority(line) > 0


@pytest.mark.parametrize(
    "line",
    [
        # Bare "error:" opens a great deal of ordinary output. Only the argparse
        # phrases qualify, or every log would anchor on its first log line.
        "error: something went wrong but training continued",
        "INFO: error: in a sentence about errors",
        "NVLink bandwidth measured at 300 GB/s",
        "iteration 42/1000 | loss 1.5",
    ],
)
def test_ordinary_output_still_scores_zero(line):
    assert _terminal_episode_priority(line) == 0


def _primary_quote(tmp_path, lines: list[str]) -> str:
    log = tmp_path / "run_2200100_date_x_cycle0.log"
    log.write_text("".join(lines))
    primary = build_l0_bundle(str(log)).deterministic_primary_candidate
    assert primary is not None
    return primary.quote or ""


_ARGPARSE = "0: pretrain.py: error: unrecognized arguments: --save-main-grads-interval 1\n"
_EXCEPTION = "0: RuntimeError: boom\n"


@pytest.mark.parametrize(
    "lines, expected",
    [
        pytest.param([_ARGPARSE, _EXCEPTION], "RuntimeError", id="argparse_then_exception"),
        pytest.param([_EXCEPTION, _ARGPARSE], "RuntimeError", id="exception_then_argparse"),
    ],
)
def test_a_typed_exception_outranks_them_inside_one_episode(tmp_path, lines, expected):
    """Within a single episode the typed surface is the better anchor, whichever
    order the two lines appear in.

    Asserted through build_l0_bundle rather than by comparing priorities.
    ``_terminal_episode_priority`` is a gate: callers only ever test it against
    zero, so two scores being ordered says nothing about which line is chosen.
    """

    assert expected in _primary_quote(tmp_path, lines)


def test_across_episodes_the_earlier_failure_anchors_instead(tmp_path):
    """The preference above does not reach across the log.

    Separated by more than UNRANKED_TRACEBACK_SCAN_LINES the two stop being one
    episode, and primary selection orders episodes by source position rather
    than by priority. A launcher rejected at startup is then the anchor and the
    exception that follows is downstream of it - which is the right answer
    here, and the reason the preference is scoped to one episode rather than
    stated as a global rule.
    """

    distant = [_ARGPARSE, *[f"0: iteration {i}/1000 | loss 1.5\n" for i in range(130)], _EXCEPTION]

    assert "unrecognized arguments" in _primary_quote(tmp_path, distant)


def test_an_argparse_rejection_produces_a_primary(tmp_path):
    log = tmp_path / "run_2200100_date_x_cycle0.log"
    log.write_text(
        "0: usage: pretrain.py [-h] [--num-layers NUM_LAYERS]\n"
        "0: pretrain.py: error: unrecognized arguments: --save-main-grads-interval 1\n"
    )

    primary = build_l0_bundle(str(log)).deterministic_primary_candidate

    assert primary is not None, "a rejected launcher must not analyze to nothing"
    assert "unrecognized arguments" in (primary.quote or "")


def test_an_nvlink_error_produces_a_primary(tmp_path):
    log = tmp_path / "run_2200100_date_x_cycle0.log"
    log.write_text(
        "0: iteration 10/1000 | loss 2.0\n"
        "0: what():  CUDA error: uncorrectable NVLink error detected during the execution\n"
    )

    primary = build_l0_bundle(str(log)).deterministic_primary_candidate

    assert primary is not None
    assert "NVLink error detected" in (primary.quote or "")
