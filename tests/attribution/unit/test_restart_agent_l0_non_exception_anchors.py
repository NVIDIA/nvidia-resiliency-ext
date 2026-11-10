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


def test_a_typed_exception_still_outranks_them(tmp_path):
    """These sit below the typed surfaces: an exception elsewhere in the log is
    the better anchor whenever there is one."""
    assert _terminal_episode_priority("RuntimeError: boom") > _terminal_episode_priority(
        "train.py: error: unrecognized arguments: --x"
    )


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
