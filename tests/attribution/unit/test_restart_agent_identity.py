# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable failure-identity normalization behavior."""

import pytest

from nvidia_resiliency_ext.attribution.restart_agent.identity import (
    normalize_token,
    normalized_pattern,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        (
            "E207 12:34:56.123 rank=7 cuda:3 RuntimeError: boom",
            "runtimeerror_boom",
        ),
        (
            "rank7:pid42 cuda:3 GPU_9 iteration=15 bucket#2 failure",
            "failure",
        ),
        (
            "node=worker-a17 device 4 retry 2 attempt 3 line 99",
            "",
        ),
        (
            "2026-07-25T12:34:56Z loss=inf at 0xdeadbeef",
            "loss_inf_at",
        ),
        (
            "allocated 16 GiB after 250ms",
            "allocated_after",
        ),
    ),
)
def test_normalize_token_removes_volatile_routing_fields(text, expected):
    assert normalize_token(text) == expected


def test_normalized_pattern_retains_stable_words_and_replaces_free_numbers():
    text = "RuntimeError code 17 on tensor shard 23"

    assert normalized_pattern(text) == "runtimeerror_code_n_on_tensor_shard_n"
