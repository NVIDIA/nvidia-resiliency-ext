# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for neutral L0 failure-signature classification."""

import pytest

from nvidia_resiliency_ext.attribution.restart_agent.l0.registry import (
    MVP_SIGNATURES,
    match_registry,
    root_fingerprint,
)


@pytest.mark.parametrize(
    ("line", "expected_registry_ids"),
    (
        (
            "ValueError: invalid config option tensor_model_parallel_size",
            {"observed_exception", "configuration_validation_failure"},
        ),
        (
            "configuration loader failed: missing key model.hidden_size",
            {"configuration_validation_failure"},
        ),
        (
            "RuntimeError: invalid argument",
            {"observed_exception", "argument_validation_failure"},
        ),
        (
            "FileNotFoundError: [Errno 2] No such file or directory",
            {"observed_exception", "artifact_or_path_not_found"},
        ),
        (
            "checkpoint metadata version mismatch",
            {"checkpoint_compatibility_mismatch"},
        ),
        (
            "loaded tensor has a shape mismatch",
            {"shape_mismatch"},
        ),
    ),
)
def test_registry_emits_observed_mechanisms_without_user_attribution(
    line,
    expected_registry_ids,
):
    rows = match_registry(line)
    registry_ids = {row.registry_id for row in rows}

    assert expected_registry_ids <= registry_ids
    assert "user_config_error" not in registry_ids


def test_neutral_structural_signatures_have_stable_family_fingerprints():
    lines_by_registry_id = {
        "configuration_validation_failure": "invalid config option hidden_size",
        "argument_validation_failure": "invalid argument",
        "artifact_or_path_not_found": "No such file or directory",
        "checkpoint_compatibility_mismatch": "checkpoint metadata mismatch",
        "shape_mismatch": "shape mismatch",
    }
    rows_by_registry_id = {row.registry_id: row for row in MVP_SIGNATURES}

    for registry_id, line in lines_by_registry_id.items():
        fingerprint = root_fingerprint(rows_by_registry_id[registry_id], line)

        assert fingerprint is not None
        assert fingerprint.startswith(f"{registry_id}:")


@pytest.mark.parametrize(
    "line",
    (
        "INFO:hypercorn.error: server started",
        "INFO checkpoint metadata loaded",
        "information about gradient health",
    ),
)
def test_nonfinite_routing_does_not_index_on_inf_inside_words(line):
    rows = match_registry(line)

    assert all(row.registry_id != "nan_or_inf" for row in rows)


@pytest.mark.parametrize(
    "line",
    (
        "loss: inf",
        "gradient = NaN",
        "non-finite activation detected",
    ),
)
def test_nonfinite_routing_preserves_real_numeric_instability(line):
    rows = match_registry(line)

    assert "nan_or_inf" in {row.registry_id for row in rows}


@pytest.mark.parametrize(
    ("line", "expected"),
    (
        ("slurmstepd: error: JOB CANCELLED DUE TO TIME LIMIT", True),
        ("scheduler terminated allocation because wall-time limit expired", True),
        ("configured wall-time limit is 04:00:00", False),
        ("training has 30 minutes remaining before the time limit", False),
    ),
)
def test_time_limit_registry_requires_an_observed_termination_event(line, expected):
    rows = match_registry(line)

    assert ("time_limit" in {row.registry_id for row in rows}) is expected
