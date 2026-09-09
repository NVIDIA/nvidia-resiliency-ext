# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral coverage for L0 NCCL RDMA port lifecycle episodes."""

from nvidia_resiliency_ext.attribution.restart_agent.infrastructure.log_source import (
    SOURCE_READ_MODE_SINGLE_SNAPSHOT,
    ChunkedLogReader,
)
from nvidia_resiliency_ext.attribution.restart_agent.l0 import (
    ProgressiveL0Accumulator,
    build_l0_bundle,
    canonical_l0a_payload,
)
from nvidia_resiliency_ext.attribution.restart_agent.l0.codec import read_l0_bundle, write_l0_bundle
from nvidia_resiliency_ext.attribution.restart_agent.l0.decision import build_decision_evidence
from nvidia_resiliency_ext.attribution.restart_agent.l0.projection import build_l0_model_facing_view


def _progress(iteration: int) -> str:
    return (
        f"0: [2026-02-11 16:00:00] iteration {iteration}/1000 | "
        f"consumed samples: {iteration * 10} |"
    )


def _rdma_event(rank: int, event: str, *, timestamp: str = "16:02:37") -> str:
    return (
        f"{rank}: [2026-02-11 {timestamp}] nvl72012-T09:293550:294720 [0] "
        "transport/net_ib.cc:253 NCCL WARN NET/IB : mlx5_1:1 "
        f"Got non-fatal async event: {event}"
    )


def _write_log(tmp_path, lines):
    log_path = tmp_path / "job.log"
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


def test_unrecovered_rdma_port_error_anchors_following_nccl_timeout(tmp_path):
    log_path = _write_log(
        tmp_path,
        [
            _progress(100),
            *(_rdma_event(rank, "port error(10)") for rank in range(4)),
            (
                "9: [rank9]:[E211 16:12:38.000000000 ProcessGroupNCCL.cpp:697] "
                "Watchdog caught collective operation timeout: "
                "WorkNCCL(SeqNum=20, OpType=ALLGATHER, NumelIn=10, NumelOut=20, "
                "Timeout(ms)=600000) ran for 600000 milliseconds before timing out."
            ),
        ],
    )

    bundle = build_l0_bundle(str(log_path))

    assert bundle.deterministic_primary_candidate is not None
    assert bundle.deterministic_primary_candidate.registry_id == "nccl_rdma_port_error_event"
    assert bundle.deterministic_primary_candidate.line == 2
    episode = next(item for item in bundle.failure_episodes if item.lifecycle_family == "rdma_port")
    assert episode.status == "terminal"
    assert episode.lifecycle_fault_lines == (2, 3, 4, 5)
    assert episode.recovery_attempt_lines == ()
    assert episode.recovery_confirmation_lines == ()
    assert episode.lifecycle_source_dialects == ("nccl_net_ib",)
    assert episode.terminal_exception_line == 6
    assert episode.identity_anchor_line == 2


def test_matching_port_active_closes_recovered_episode_before_later_cuda_failure(tmp_path):
    log_path = _write_log(
        tmp_path,
        [
            _progress(100),
            *(_rdma_event(rank, "port error(10)") for rank in range(4)),
            *(
                _rdma_event(rank, "client reregistration(17)", timestamp="16:03:30")
                for rank in range(4)
            ),
            *(_rdma_event(rank, "port active(9)", timestamp="16:03:30") for rank in range(4)),
            _progress(110),
            "7: [rank7]: RuntimeError: CUDA error: unspecified launch failure",
        ],
    )

    bundle = build_l0_bundle(str(log_path))

    rdma_episode = next(
        item for item in bundle.failure_episodes if item.lifecycle_family == "rdma_port"
    )
    assert rdma_episode.status == "recovered"
    assert rdma_episode.lifecycle_fault_lines == (2, 3, 4, 5)
    assert rdma_episode.recovery_attempt_lines == (6, 7, 8, 9)
    assert rdma_episode.recovery_confirmation_lines == (10, 11, 12, 13)
    assert rdma_episode.terminal_exception_line is None
    assert rdma_episode.first_progress_after is not None
    assert rdma_episode.first_progress_after.line == 14
    assert bundle.deterministic_primary_candidate is not None
    assert bundle.deterministic_primary_candidate.line == 15
    assert bundle.deterministic_primary_candidate.registry_id != "nccl_rdma_port_error_event"
    assert bundle.run_progress_summary.progress_after_failure_episode is False


def test_later_progress_without_port_active_does_not_claim_component_recovery(tmp_path):
    log_path = _write_log(
        tmp_path,
        [
            _progress(100),
            _rdma_event(4, "port error(10)"),
            _rdma_event(4, "client reregistration(17)", timestamp="16:03:30"),
            _progress(110),
            "7: [rank7]: RuntimeError: later workload failure",
        ],
    )

    bundle = build_l0_bundle(str(log_path))

    rdma_episode = next(
        item for item in bundle.failure_episodes if item.lifecycle_family == "rdma_port"
    )
    assert rdma_episode.status == "progressed_after"
    assert rdma_episode.recovery_attempt_lines == (3,)
    assert rdma_episode.recovery_confirmation_lines == ()
    assert rdma_episode.first_progress_after is not None
    assert bundle.deterministic_primary_candidate is not None
    assert bundle.deterministic_primary_candidate.line == 5


def test_repeated_recovered_port_errors_remain_separate_episodes(tmp_path):
    log_path = _write_log(
        tmp_path,
        [
            _progress(100),
            _rdma_event(4, "port error(10)"),
            _rdma_event(4, "client reregistration(17)", timestamp="16:03:30"),
            _rdma_event(4, "port active(9)", timestamp="16:03:30"),
            _progress(110),
            _rdma_event(4, "port error(10)", timestamp="17:02:05"),
            _rdma_event(4, "client reregistration(17)", timestamp="17:02:25"),
            _rdma_event(4, "port active(9)", timestamp="17:02:25"),
            _progress(120),
            "8: [rank8]: RuntimeError: iteration 121: Unexpected result inf",
        ],
    )

    bundle = build_l0_bundle(str(log_path))

    rdma_episodes = [
        item for item in bundle.failure_episodes if item.lifecycle_family == "rdma_port"
    ]
    assert [item.status for item in rdma_episodes] == ["recovered", "recovered"]
    assert [item.lifecycle_fault_lines for item in rdma_episodes] == [(2,), (6,)]
    assert all(item.terminal_exception_line is None for item in rdma_episodes)
    assert bundle.deterministic_primary_candidate is not None
    assert bundle.deterministic_primary_candidate.line == 10
    assert bundle.run_progress_summary.progress_after_failure_episode is False


def test_port_lifecycle_fanout_is_compact_in_l0b(tmp_path):
    log_path = _write_log(
        tmp_path,
        [
            _progress(100),
            *(_rdma_event(rank, "port error(10)") for rank in range(4)),
            *(
                _rdma_event(rank, "client reregistration(17)", timestamp="16:03:30")
                for rank in range(4)
            ),
            *(_rdma_event(rank, "port active(9)", timestamp="16:03:30") for rank in range(4)),
            _progress(110),
            "7: [rank7]: RuntimeError: later workload failure",
        ],
    )

    bundle = build_l0_bundle(str(log_path))
    groups = {group.registry_id: group for group in bundle.occurrence_groups}
    prompt = build_l0_model_facing_view(
        bundle,
        build_decision_evidence(bundle),
    ).evidence_bundle
    prompt_episode = next(
        item for item in prompt["failure_episodes"] if item["lifecycle_family"] == "rdma_port"
    )

    assert groups["nccl_rdma_port_error_event"].count == 4
    assert groups["nccl_rdma_client_reregistration_event.v1"].count == 4
    assert groups["nccl_rdma_port_active_event.v1"].count == 4
    assert prompt_episode["status"] == "recovered"
    assert prompt_episode["lifecycle_entities"] == ["nvl72012-T09/mlx5_1:1"]
    assert prompt_episode["lifecycle_source_dialects"] == ["nccl_net_ib"]
    assert prompt_episode["recovery_confirmation_lines"] == [10, 11, 12, 13]


def test_port_active_without_prior_error_is_context_only(tmp_path):
    log_path = _write_log(
        tmp_path,
        [
            _progress(100),
            _rdma_event(4, "port active(9)"),
            _progress(110),
        ],
    )

    bundle = build_l0_bundle(str(log_path))

    assert all(item.lifecycle_family != "rdma_port" for item in bundle.failure_episodes)
    assert bundle.deterministic_primary_candidate is None
    assert any(
        group.registry_id == "nccl_rdma_port_active_event.v1" for group in bundle.occurrence_groups
    )


def test_rdma_port_lifecycle_round_trips_through_l0_replay_artifact(tmp_path):
    log_path = _write_log(
        tmp_path,
        [
            _progress(100),
            _rdma_event(4, "port error(10)"),
            _rdma_event(4, "client reregistration(17)", timestamp="16:03:30"),
            _rdma_event(4, "port active(9)", timestamp="16:03:30"),
            _progress(110),
        ],
    )
    bundle_path = tmp_path / "l0_bundle.json"
    bundle = build_l0_bundle(str(log_path))

    write_l0_bundle(bundle_path, bundle)
    replayed = read_l0_bundle(bundle_path, expected_log_path=str(log_path))

    assert replayed == bundle


def test_rdma_port_lifecycle_is_identical_across_chunked_and_snapshot_reads(tmp_path):
    log_path = _write_log(
        tmp_path,
        [
            _progress(100),
            _rdma_event(4, "port error(10)"),
            _rdma_event(4, "client reregistration(17)", timestamp="16:03:30"),
            _rdma_event(4, "port active(9)", timestamp="16:03:30"),
            _progress(110),
            "7: [rank7]: RuntimeError: CUDA error: unspecified launch failure",
        ],
    )

    chunked = ProgressiveL0Accumulator(
        str(log_path),
        reader=ChunkedLogReader(chunk_bytes=7),
    ).finalize()
    snapshot = ProgressiveL0Accumulator(
        str(log_path),
        reader=ChunkedLogReader(read_mode=SOURCE_READ_MODE_SINGLE_SNAPSHOT),
    ).finalize()

    assert canonical_l0a_payload(
        chunked.bundle,
        chunked.decision_evidence,
    ) == canonical_l0a_payload(
        snapshot.bundle,
        snapshot.decision_evidence,
    )
