# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared byte-ingestion behavior for terminal and progressive L0A."""

import pytest

from nvidia_resiliency_ext.attribution.restart_agent.infrastructure.log_source import (
    SOURCE_READ_MODE_SINGLE_SNAPSHOT,
    ChunkedLogReader,
    IncrementalLineDecoder,
)
from nvidia_resiliency_ext.attribution.restart_agent.l0 import (
    ProgressiveL0Accumulator,
    canonical_l0a_payload,
)
from nvidia_resiliency_ext.attribution.restart_agent.models import RestartAgentRequest
from nvidia_resiliency_ext.attribution.restart_agent.pipeline import RestartAgent


def test_chunk_boundaries_preserve_linux_physical_lines_and_unterminated_tail(tmp_path):
    # Arrange
    log_path = tmp_path / "mixed-newlines.log"
    log_path.write_bytes("alpha\r\nbeta\rgamma\nsnowman \N{SNOWMAN}\nlast\r".encode())

    # Act
    one_byte = ChunkedLogReader(chunk_bytes=1).snapshot(log_path)
    seven_bytes = ChunkedLogReader(chunk_bytes=7).snapshot(log_path)
    single_snapshot = ChunkedLogReader(
        chunk_bytes=7,
        read_mode=SOURCE_READ_MODE_SINGLE_SNAPSHOT,
    ).snapshot(log_path)

    # Assert
    expected = ("alpha", "beta\rgamma", "snowman \N{SNOWMAN}", "last\r")
    assert one_byte.lines == expected
    assert seven_bytes.lines == expected
    assert single_snapshot.lines == expected
    assert one_byte.line_count == len(expected)
    assert one_byte.context_before(4, limit=2) == ("beta\rgamma", "snowman \N{SNOWMAN}")


def test_incremental_decoder_retains_partial_line_and_split_crlf():
    # Arrange
    decoder = IncrementalLineDecoder()

    # Act and assert
    assert decoder.feed(b"first\r") == ()
    assert decoder.pending_bytes == len(b"first\r")
    assert decoder.feed(b"\nRuntimeError: CUDA out of") == ("first",)
    assert decoder.feed(b" memory\nlast") == ("RuntimeError: CUDA out of memory",)
    assert decoder.feed(b"", final=True) == ("last",)


def test_incremental_decoder_preserves_bare_cr_content_and_source_offsets():
    # Arrange
    decoder = IncrementalLineDecoder()

    # Act
    complete = decoder.feed_records(b"a\rb\r\nc\n")
    tail = decoder.feed_records(b"last\r", final=True)

    # Assert
    assert [
        (record.text, record.start_offset, record.end_offset, record.next_offset)
        for record in complete
    ] == [
        ("a\rb", 0, 3, 5),
        ("c", 5, 6, 7),
    ]
    assert [
        (record.text, record.start_offset, record.end_offset, record.next_offset) for record in tail
    ] == [("last\r", 7, 12, 12)]


def test_chunked_reader_replaces_invalid_utf8_without_replaying_boundary(tmp_path):
    # Arrange
    log_path = tmp_path / "malformed-utf8.log"
    log_path.write_bytes(b"valid\ninvalid \xde byte\n")

    # Act
    snapshot = ChunkedLogReader(chunk_bytes=1).snapshot(log_path)

    # Assert
    assert snapshot.encoding == "utf-8"
    assert snapshot.lines == ("valid", "invalid \N{REPLACEMENT CHARACTER} byte")


@pytest.mark.parametrize("terminal_chunk_bytes", [1, 2, 3, 7, 11, 64 * 1024])
def test_terminal_progressive_and_single_snapshot_have_identical_l0a(
    tmp_path,
    terminal_chunk_bytes,
):
    # Arrange
    log_path = tmp_path / "train.log"
    prefix = "[2026-01-01 00:00:00] iteration 7 / 100 | consumed samples: 64 |\n"
    suffix = "RuntimeError: CUDA out of memory\n"
    log_path.write_text(prefix, encoding="utf-8")
    progressive = ProgressiveL0Accumulator(
        str(log_path),
        reader=ChunkedLogReader(chunk_bytes=3),
    )

    # Act
    assert progressive.refresh() is True
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(suffix)
    assert progressive.refresh() is True
    progressive_final = progressive.finalize()
    chunked_terminal = ProgressiveL0Accumulator(
        str(log_path),
        reader=ChunkedLogReader(chunk_bytes=terminal_chunk_bytes),
    ).finalize()
    single_snapshot = ProgressiveL0Accumulator(
        str(log_path),
        reader=ChunkedLogReader(read_mode=SOURCE_READ_MODE_SINGLE_SNAPSHOT),
    ).finalize()

    # Assert
    assert progressive_final.precomputed is True
    assert chunked_terminal.precomputed is False
    payloads = [
        canonical_l0a_payload(result.bundle, result.decision_evidence)
        for result in (progressive_final, chunked_terminal, single_snapshot)
    ]
    assert payloads[0] == payloads[1] == payloads[2]
    assert {
        progressive_final.canonical_hash,
        chunked_terminal.canonical_hash,
        single_snapshot.canonical_hash,
    } == {progressive_final.canonical_hash}


def test_partial_failure_line_is_not_observed_until_completed(tmp_path):
    # Arrange
    log_path = tmp_path / "train.log"
    log_path.write_bytes(b"RuntimeError: CUDA out")
    accumulator = ProgressiveL0Accumulator(
        str(log_path),
        reader=ChunkedLogReader(chunk_bytes=2),
    )

    # Act
    accumulator.refresh()
    partial_state = accumulator.state()
    with log_path.open("ab") as handle:
        handle.write(b" of memory\n")
    accumulator.refresh()
    finalized = accumulator.finalize()

    # Assert
    assert partial_state.line_count == 0
    assert partial_state.pending_line_bytes == len(b"RuntimeError: CUDA out")
    assert finalized.source_log.lines == ("RuntimeError: CUDA out of memory",)
    assert finalized.bundle.line_count == 1
    assert finalized.bundle.deterministic_primary_candidate is not None
    assert finalized.bundle.deterministic_primary_candidate.line == 1
    assert finalized.source_log.storage_mode == "indexed_file"


def test_finalization_can_exclude_an_actively_written_partial_tail(tmp_path):
    # Arrange
    log_path = tmp_path / "train.log"
    log_path.write_bytes(b"iteration 1 completed\nRuntimeError: still being written")
    accumulator = ProgressiveL0Accumulator(str(log_path))
    accumulator.refresh(precompute=False)

    # Act
    finalized = accumulator.finalize(include_incomplete_tail=False)

    # Assert
    assert finalized.source_log.lines == ("iteration 1 completed",)
    assert finalized.bundle.line_count == 1
    assert finalized.progressive_metrics["discarded_incomplete_tail_bytes"] == len(
        b"RuntimeError: still being written"
    )


def test_late_invalid_utf8_is_replaced_without_replaying_complete_source(tmp_path):
    # Arrange
    log_path = tmp_path / "late-malformed-utf8.log"
    first = "snowman \N{SNOWMAN}\n".encode("utf-8")
    log_path.write_bytes(first)
    progressive = ProgressiveL0Accumulator(
        str(log_path),
        reader=ChunkedLogReader(chunk_bytes=2),
    )
    progressive.refresh()

    # Act
    with log_path.open("ab") as handle:
        handle.write(b"invalid \xde byte\n")
    progressive.refresh()
    progressive_final = progressive.finalize()
    terminal_final = ProgressiveL0Accumulator(str(log_path)).finalize()

    # Assert
    assert progressive_final.source_log.encoding == "utf-8"
    assert progressive_final.source_log.lines == terminal_final.source_log.lines
    assert progressive_final.canonical_hash == terminal_final.canonical_hash
    assert progressive_final.progressive_metrics["decode_replacement_count"] == 1
    assert progressive_final.progressive_metrics["decode_replacement_line_count"] == 1
    assert progressive_final.progressive_metrics["bytes_reread"] == 0


def test_finalize_rebuilds_precomputed_l0a_after_unobserved_late_append(tmp_path):
    # Arrange
    log_path = tmp_path / "late-terminal-output.log"
    log_path.write_text("iteration 7 completed\n", encoding="utf-8")
    progressive = ProgressiveL0Accumulator(str(log_path))
    assert progressive.refresh(precompute=True) is True
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("RuntimeError: CUDA out of memory\n")

    # Act
    progressive_final = progressive.finalize()
    terminal_final = ProgressiveL0Accumulator(str(log_path)).finalize()

    # Assert
    assert progressive_final.precomputed is False
    assert progressive_final.progressive_metrics["l0a_build_count"] == 2
    assert progressive_final.source_log.lines == terminal_final.source_log.lines
    assert progressive_final.canonical_hash == terminal_final.canonical_hash
    assert progressive_final.bundle.deterministic_primary_candidate is not None
    assert progressive_final.bundle.deterministic_primary_candidate.line == 2


def test_unchanged_refresh_builds_checkpoint_for_exact_current_boundary(tmp_path):
    # Arrange
    log_path = tmp_path / "stable-terminal-output.log"
    log_path.write_text(
        "iteration 7 completed\nRuntimeError: CUDA out of memory\n",
        encoding="utf-8",
    )
    accumulator = ProgressiveL0Accumulator(str(log_path))
    assert accumulator.refresh(precompute=False) is True
    assert accumulator.state().l0a_build_count == 0

    # Act
    assert accumulator.refresh(precompute=True) is False
    finalized = accumulator.finalize()

    # Assert
    assert finalized.precomputed is True
    assert finalized.progressive_metrics["l0a_build_count"] == 1
    assert finalized.source_boundary == finalized.source_log.source_boundary


def test_repeated_large_reductions_match_fresh_terminal_reduction(tmp_path):
    # Arrange
    log_path = tmp_path / "many-high-signal-lines.log"
    initial_lines = [f"ERROR: repeated failure rendering {index}" for index in range(4_100)]
    log_path.write_text("\n".join(initial_lines) + "\n", encoding="utf-8")
    progressive = ProgressiveL0Accumulator(str(log_path))
    assert progressive.refresh(precompute=True) is True
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("RuntimeError: terminal failure\n")

    # Act
    progressive_final = progressive.finalize()
    terminal_final = ProgressiveL0Accumulator(str(log_path)).finalize()

    # Assert
    assert progressive_final.progressive_metrics["l0a_build_count"] == 2
    assert progressive_final.canonical_hash == terminal_final.canonical_hash


def test_invalid_utf8_in_unterminated_tail_is_replaced_at_finalization(tmp_path):
    # Arrange
    log_path = tmp_path / "unterminated-malformed-utf8.log"
    log_path.write_bytes(b"first\ninvalid \xde")
    accumulator = ProgressiveL0Accumulator(
        str(log_path),
        reader=ChunkedLogReader(chunk_bytes=1),
    )
    accumulator.refresh()

    # Act
    finalized = accumulator.finalize()

    # Assert
    assert finalized.source_log.encoding == "utf-8"
    assert finalized.source_log.lines == ("first", "invalid \N{REPLACEMENT CHARACTER}")
    assert finalized.progressive_metrics["decode_replacement_count"] == 1
    assert finalized.progressive_metrics["decode_replacement_line_count"] == 1
    assert finalized.progressive_metrics["bytes_reread"] == 0


def test_unchanged_poll_is_metadata_only_and_same_size_rewrite_resets(tmp_path):
    # Arrange
    log_path = tmp_path / "train.log"
    log_path.write_text("RuntimeError: first failure\n", encoding="utf-8")
    accumulator = ProgressiveL0Accumulator(
        str(log_path),
        reader=ChunkedLogReader(chunk_bytes=2),
    )

    # Act
    assert accumulator.refresh() is True
    assert accumulator.refresh() is False
    log_path.write_text("RuntimeError: other failure\n", encoding="utf-8")
    assert accumulator.refresh() is True

    # Assert
    state = accumulator.state()
    assert state.poll_count == 3
    assert state.unchanged_poll_count == 1
    assert state.reset_count == 1
    assert state.l0a_build_count == 2


def test_source_replacement_resets_progressive_state(tmp_path):
    # Arrange
    log_path = tmp_path / "train.log"
    log_path.write_text("RuntimeError: first failure\n", encoding="utf-8")
    accumulator = ProgressiveL0Accumulator(str(log_path))
    accumulator.refresh()
    replacement = tmp_path / "replacement.log"
    replacement.write_text("RuntimeError: replacement\n", encoding="utf-8")

    # Act
    replacement.replace(log_path)
    accumulator.refresh()
    finalized = accumulator.finalize()

    # Assert
    assert finalized.source_log.lines == ("RuntimeError: replacement",)
    assert finalized.progressive_metrics["reset_count"] == 1


def test_finalized_snapshot_hides_later_appends(tmp_path):
    # Arrange
    log_path = tmp_path / "train.log"
    log_path.write_text("first\nsecond\n", encoding="utf-8")
    finalized = ProgressiveL0Accumulator(str(log_path)).finalize()

    # Act
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("late\n")

    # Assert
    assert finalized.source_log.lines == ("first", "second")
    assert finalized.source_log.line(3) is None


def test_prepared_runtime_does_not_require_a_second_source_read(tmp_path):
    # Arrange
    log_path = tmp_path / "train.log"
    log_path.write_text("RuntimeError: CUDA out of memory\n", encoding="utf-8")
    finalized = ProgressiveL0Accumulator(str(log_path)).finalize()
    log_path.unlink()

    # Act
    run = RestartAgent().run_prepared(
        RestartAgentRequest(
            log_path=str(log_path),
            job_id="job-1",
            cycle_id=1,
        ),
        finalized,
    )

    # Assert
    assert run.bundle == finalized.bundle
    assert run.trace["l0_source"]["canonical_hash"] == finalized.canonical_hash
    assert run.trace["l0_source"]["storage_mode"] == "indexed_file"
