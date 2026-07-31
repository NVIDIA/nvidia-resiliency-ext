# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral boundaries for distributed collective-timeout incidents."""

import pytest

from nvidia_resiliency_ext.attribution.restart_agent.infrastructure.log_source import (
    SOURCE_READ_MODE_SINGLE_SNAPSHOT,
    ChunkedLogReader,
)
from nvidia_resiliency_ext.attribution.restart_agent.l0 import (
    ProgressiveL0Accumulator,
    build_l0_bundle,
    canonical_l0a_payload,
)


def _timeout(
    rank: int,
    timestamp: str | None,
    sequence: int,
    *,
    timeout_ms: int = 600_000,
) -> str:
    timestamp_prefix = f"[E224 {timestamp} ProcessGroupNCCL.cpp:697] " if timestamp else ""
    return (
        f"{rank}: [rank{rank}]:{timestamp_prefix}"
        "Watchdog caught collective operation timeout: "
        f"WorkNCCL(SeqNum={sequence}, OpType=ALLGATHER, Timeout(ms)={timeout_ms}) "
        f"ran for {timeout_ms + 1} milliseconds before timing out."
    )


def test_timeout_wave_matches_any_prior_event_across_midnight(tmp_path):
    # Arrange
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                _timeout(0, "00:00:00.000000000", 1),
                _timeout(1, "00:01:00.000000000", 2),
                _timeout(2, "23:59:50.000000000", 3),
            ]
        ),
        encoding="utf-8",
    )

    # Act
    bundle = build_l0_bundle(log_path)

    # Assert
    assert len(bundle.distributed_failure_incidents) == 1
    incident = bundle.distributed_failure_incidents[0]
    assert incident.event_count == 3
    assert incident.member_event_lines == (1, 2, 3)


def test_progress_between_timeout_events_starts_a_new_wave(tmp_path):
    # Arrange
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                _timeout(0, "19:02:08.000000000", 1),
                ("0: [2026-02-24 19:02:08.500000] iteration 2/100 | " "consumed samples: 20 |"),
                _timeout(1, "19:02:09.000000000", 2),
            ]
        ),
        encoding="utf-8",
    )

    # Act
    bundle = build_l0_bundle(log_path)

    # Assert
    assert [incident.member_event_lines for incident in bundle.distributed_failure_incidents] == [
        (1,),
        (3,),
    ]


def test_configured_timeout_change_starts_a_new_wave(tmp_path):
    # Arrange
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                _timeout(0, "19:02:08.000000000", 1, timeout_ms=600_000),
                _timeout(1, "19:02:09.000000000", 2, timeout_ms=300_000),
            ]
        ),
        encoding="utf-8",
    )

    # Act
    bundle = build_l0_bundle(log_path)

    # Assert
    assert [incident.member_event_lines for incident in bundle.distributed_failure_incidents] == [
        (1,),
        (2,),
    ]


@pytest.mark.parametrize(
    ("filler_line_count", "expected_incident_count"),
    ((999, 1), (1000, 2)),
)
def test_timeout_without_timestamp_uses_line_distance_boundary(
    tmp_path,
    filler_line_count,
    expected_incident_count,
):
    # Arrange
    log_path = tmp_path / "job.log"
    lines = [_timeout(0, None, 1)]
    lines.extend(f"ordinary output {index}" for index in range(filler_line_count))
    lines.append(_timeout(1, None, 2))
    log_path.write_text("\n".join(lines), encoding="utf-8")

    # Act
    bundle = build_l0_bundle(log_path)

    # Assert
    assert len(bundle.distributed_failure_incidents) == expected_incident_count


def test_collective_timeout_l0a_is_identical_across_read_modes(tmp_path):
    # Arrange
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(_timeout(rank, f"19:02:{rank:02d}.000000000", rank + 1) for rank in range(8)),
        encoding="utf-8",
    )

    # Act
    chunked = ProgressiveL0Accumulator(
        str(log_path),
        reader=ChunkedLogReader(chunk_bytes=17),
    ).finalize()
    single_snapshot = ProgressiveL0Accumulator(
        str(log_path),
        reader=ChunkedLogReader(read_mode=SOURCE_READ_MODE_SINGLE_SNAPSHOT),
    ).finalize()

    # Assert
    assert canonical_l0a_payload(
        chunked.bundle,
        chunked.decision_evidence,
    ) == canonical_l0a_payload(
        single_snapshot.bundle,
        single_snapshot.decision_evidence,
    )
    assert chunked.canonical_hash == single_snapshot.canonical_hash
