# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for neutral L0 failure-signature classification."""

import pytest

from nvidia_resiliency_ext.attribution.restart_agent.l0.registry import (
    MVP_SIGNATURES,
    failure_signal_classifiers,
    match_registry,
    parse_nccl_rdma_port_lifecycle_event,
    root_fingerprint,
)
from nvidia_resiliency_ext.attribution.restart_agent.models import FailureClassifier


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
    "line",
    (
        "RuntimeError: CUDA out of memory",
        "torch.AcceleratorError: CUDA error: out of memory",
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB",
        "RuntimeError: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate",
    ),
)
def test_cuda_oom_variants_share_typed_registry_identity(line):
    rows = match_registry(line)

    assert "cuda_oom" in {row.registry_id for row in rows}
    assert FailureClassifier.CUDA_OOM.value in failure_signal_classifiers(line)


def test_cuda_memory_diagnostic_reference_is_not_an_oom_signal():
    line = "Search for cudaErrorMemoryAllocation in the CUDA Runtime API documentation"

    assert "cuda_oom" not in {row.registry_id for row in match_registry(line)}
    assert FailureClassifier.CUDA_OOM.value not in failure_signal_classifiers(line)


@pytest.mark.parametrize(
    "line",
    (
        "OSError: [Errno 98] Address already in use",
        "zmq.error.ZMQError: Address already in use (addr='tcp://0.0.0.0:12346')",
        "RuntimeError: bind failed with EADDRINUSE",
    ),
)
def test_port_bind_conflict_variants_share_typed_classifier(line):
    assert FailureClassifier.PORT_BIND_CONFLICT.value in failure_signal_classifiers(line)


def test_generic_connection_failure_is_not_a_port_bind_conflict():
    assert FailureClassifier.PORT_BIND_CONFLICT.value not in failure_signal_classifiers(
        "RuntimeError: connection refused"
    )


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


def test_rejected_nonfinite_iteration_has_policy_neutral_classifiers():
    classifiers = failure_signal_classifiers(
        "180: [rank180]: RuntimeError: iteration 670314: Unexpected result inf"
    )

    assert classifiers == (
        FailureClassifier.NAN_OR_INF.value,
        FailureClassifier.REJECTED_NONFINITE_ITERATION.value,
    )


@pytest.mark.parametrize(
    "line",
    (
        "INFO number of nan iterations: 0",
        "gradient = inf",
        "RuntimeError: unrelated failure",
    ),
)
def test_rejected_nonfinite_classifier_excludes_ordinary_nonfinite_mentions(line):
    assert failure_signal_classifiers(line) == ()


@pytest.mark.parametrize(
    ("event_text", "event_type", "event_code"),
    (
        ("port error(10)", "port_error", 10),
        ("client reregistration(17)", "client_reregistration", 17),
        ("port active(9)", "port_active", 9),
    ),
)
def test_nccl_rdma_port_lifecycle_parser_preserves_explicit_event_semantics(
    event_text,
    event_type,
    event_code,
):
    line = (
        "348: host-a:123:456 [0] transport/net_ib.cc:253 "
        "NCCL WARN NET/IB : mlx5_1:1 Got non-fatal async event: "
        f"{event_text}"
    )

    parsed = parse_nccl_rdma_port_lifecycle_event(line)

    assert parsed is not None
    assert parsed.event_type == event_type
    assert parsed.device == "mlx5_1"
    assert parsed.port == "1"
    assert parsed.event_code == event_code
    assert parsed.node == "host-a"
    assert parsed.source_dialect == "nccl_net_ib"
    assert parsed.network_protocol is None
    registry_ids = {row.registry_id for row in match_registry(line)}
    assert ("nccl_rdma_port_error_event" in registry_ids) is (event_type == "port_error")


@pytest.mark.parametrize(
    "line",
    (
        "NCCL INFO NET/IB : Using mlx5_1:1 for communication",
        "NCCL WARN NET/RoCE : mlx5_1:1 Got non-fatal async event: port error(10)",
        "configured IB port mlx5_1:1 is active",
        "diagnostic: previous text mentioned port error(10)",
        "mlx5_1 link state is unknown",
    ),
)
def test_nccl_rdma_port_lifecycle_parser_rejects_generic_rdma_mentions(line):
    assert parse_nccl_rdma_port_lifecycle_event(line) is None
    assert "nccl_rdma_port_error_event" not in {row.registry_id for row in match_registry(line)}
