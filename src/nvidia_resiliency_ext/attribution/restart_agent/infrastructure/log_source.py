# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded byte and logical-line access for restart-agent analysis."""

from __future__ import annotations

import os
from array import array
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import BinaryIO, Callable, Iterator, Protocol, Sequence

from ..models import LogLine

DEFAULT_LOG_READ_CHUNK_BYTES = 1024 * 1024
SOURCE_READ_MODE_CHUNKED = "chunked"
SOURCE_READ_MODE_SINGLE_SNAPSHOT = "single_snapshot"
SOURCE_READ_MODES = frozenset(
    {
        SOURCE_READ_MODE_CHUNKED,
        SOURCE_READ_MODE_SINGLE_SNAPSHOT,
    }
)


@dataclass(frozen=True)
class SourceBoundary:
    """Immutable source identity and byte boundary for one analysis snapshot."""

    device: int
    inode: int
    byte_size: int
    mtime_ns: int

    def same_file(self, other: "SourceBoundary") -> bool:
        return self.device == other.device and self.inode == other.inode

    def to_payload(self) -> dict[str, int]:
        return {
            "device": self.device,
            "inode": self.inode,
            "byte_size": self.byte_size,
            "mtime_ns": self.mtime_ns,
        }


@dataclass(frozen=True)
class DecodedLine:
    """One decoded physical line and its byte offsets in the source."""

    text: str
    start_offset: int
    end_offset: int
    next_offset: int
    decode_replacement_count: int = 0


class IncrementalLineDecoder:
    """Turn arbitrary byte chunks into exactly-once LF-delimited physical lines."""

    def __init__(self, *, encoding: str = "utf-8") -> None:
        if encoding not in {"utf-8", "latin-1"}:
            raise ValueError("encoding must be 'utf-8' or 'latin-1'")
        self._encoding = encoding
        self._pending = b""
        self._bytes_received = 0
        self._finalized = False
        self._decode_replacement_count = 0
        self._decode_replacement_line_count = 0

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def pending_bytes(self) -> int:
        return len(self._pending)

    @property
    def decode_replacement_count(self) -> int:
        return self._decode_replacement_count

    @property
    def decode_replacement_line_count(self) -> int:
        return self._decode_replacement_line_count

    def feed(self, chunk: bytes, *, final: bool = False) -> tuple[str, ...]:
        """Decode complete lines, retaining an incomplete tail until later."""

        return tuple(record.text for record in self.feed_records(chunk, final=final))

    def feed_records(self, chunk: bytes, *, final: bool = False) -> tuple[DecodedLine, ...]:
        """Decode complete lines with stable source-byte offsets."""

        if self._finalized:
            raise RuntimeError("line decoder is already finalized")
        if not isinstance(chunk, bytes):
            raise TypeError("chunk must be bytes")

        data_start = self._bytes_received - len(self._pending)
        self._bytes_received += len(chunk)
        data = self._pending + chunk
        lines: list[DecodedLine] = []
        record_start = 0
        while True:
            lf_index = data.find(b"\n", record_start)
            if lf_index < 0:
                break
            text_end = (
                lf_index - 1
                if lf_index > record_start and data[lf_index - 1] == ord("\r")
                else lf_index
            )
            text, replacement_count = _decode_bytes(
                data[record_start:text_end],
                self._encoding,
            )
            lines.append(
                DecodedLine(
                    text=text,
                    start_offset=data_start + record_start,
                    end_offset=data_start + text_end,
                    next_offset=data_start + lf_index + 1,
                    decode_replacement_count=replacement_count,
                )
            )
            self._record_decode_replacements(replacement_count)
            record_start = lf_index + 1

        self._pending = data[record_start:]
        if final:
            if self._pending:
                text, replacement_count = _decode_bytes(self._pending, self._encoding)
                lines.append(
                    DecodedLine(
                        text=text,
                        start_offset=data_start + record_start,
                        end_offset=data_start + len(data),
                        next_offset=data_start + len(data),
                        decode_replacement_count=replacement_count,
                    )
                )
                self._record_decode_replacements(replacement_count)
            self._pending = b""
            self._finalized = True
        return tuple(lines)

    def _record_decode_replacements(self, count: int) -> None:
        if count:
            self._decode_replacement_count += count
            self._decode_replacement_line_count += 1

    def discard_pending(self) -> int:
        """Finalize without treating an actively written tail as a complete line."""

        if self._finalized:
            raise RuntimeError("line decoder is already finalized")
        discarded = len(self._pending)
        self._pending = b""
        self._finalized = True
        return discarded


class _LineStore(Protocol):
    @property
    def line_count(self) -> int: ...

    @property
    def storage_mode(self) -> str: ...

    def line(self, line: int) -> str | None: ...

    def log_lines(
        self,
        *,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> Iterator[LogLine]: ...


class InMemoryLineStore:
    """Physical lines retained directly for explicit snapshots and small tests."""

    def __init__(self, lines: Sequence[str] = ()) -> None:
        self._lines = list(lines)

    @property
    def line_count(self) -> int:
        return len(self._lines)

    @property
    def storage_mode(self) -> str:
        return "memory"

    def append(self, text: str) -> None:
        self._lines.append(text)

    def line(self, line: int) -> str | None:
        if line < 1 or line > self.line_count:
            return None
        return self._lines[line - 1]

    def log_lines(
        self,
        *,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> Iterator[LogLine]:
        start = max(1, start_line)
        end = self.line_count if end_line is None else min(self.line_count, end_line)
        for line_no in range(start, end + 1):
            yield LogLine(line=line_no, text=self._lines[line_no - 1])


class IndexedFileLineStore:
    """Compact logical-line index over one captured local-file boundary."""

    def __init__(
        self,
        path: str | Path,
        *,
        boundary: SourceBoundary,
        encoding: str,
        line_starts: Sequence[int] = (),
        indexed_end_offset: int = 0,
        read_chunk_bytes: int = DEFAULT_LOG_READ_CHUNK_BYTES,
    ) -> None:
        self._path = str(path)
        self._boundary = boundary
        self._encoding = encoding
        self._line_starts = array("Q", line_starts)
        self._indexed_end_offset = indexed_end_offset
        self._read_chunk_bytes = read_chunk_bytes
        self._lock = RLock()
        self._handle = Path(path).open("rb")
        ChunkedLogReader._validate_open_source(self._handle, boundary)

    @property
    def line_count(self) -> int:
        return len(self._line_starts)

    @property
    def storage_mode(self) -> str:
        return "indexed_file"

    @property
    def boundary(self) -> SourceBoundary:
        return self._boundary

    @property
    def encoding(self) -> str:
        return self._encoding

    def update_boundary(self, boundary: SourceBoundary) -> None:
        with self._lock:
            if not self._boundary.same_file(boundary):
                raise OSError("indexed source was replaced")
            if boundary.byte_size < self._boundary.byte_size:
                raise OSError("indexed source was truncated")
            self._boundary = boundary

    def append_record(self, record: DecodedLine) -> None:
        with self._lock:
            self._line_starts.append(record.start_offset)
            self._indexed_end_offset = record.next_offset

    def line(self, line: int) -> str | None:
        if line < 1 or line > self.line_count:
            return None
        with self._lock:
            start = self._line_starts[line - 1]
            end = self._line_starts[line] if line < self.line_count else self._indexed_end_offset
            raw = os.pread(self._handle.fileno(), end - start, start)
        return _decode_record(raw, self._encoding)

    def log_lines(
        self,
        *,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> Iterator[LogLine]:
        start = max(1, start_line)
        end = self.line_count if end_line is None else min(self.line_count, end_line)
        if start > end:
            return
        with self._lock:
            start_offset = self._line_starts[start - 1]
            end_offset = (
                self._line_starts[end] if end < self.line_count else self._indexed_end_offset
            )
            fd = self._handle.fileno()
            encoding = self._encoding
            chunk_bytes = self._read_chunk_bytes

        decoder = IncrementalLineDecoder(encoding=encoding)
        offset = start_offset
        line_no = start
        while offset < end_offset:
            read_size = min(chunk_bytes, end_offset - offset)
            chunk = os.pread(fd, read_size, offset)
            if not chunk:
                raise OSError("indexed source ended before its captured line boundary")
            offset += len(chunk)
            for text in decoder.feed(chunk):
                yield LogLine(line=line_no, text=text)
                line_no += 1
        for text in decoder.feed(b"", final=True):
            yield LogLine(line=line_no, text=text)
            line_no += 1
        if line_no != end + 1:
            raise OSError("indexed source line count does not match its byte index")

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class ChunkedLogReader:
    """Read a captured source boundary as fixed chunks or one test snapshot."""

    def __init__(
        self,
        *,
        chunk_bytes: int = DEFAULT_LOG_READ_CHUNK_BYTES,
        read_mode: str = SOURCE_READ_MODE_CHUNKED,
    ) -> None:
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int):
            raise TypeError("chunk_bytes must be an integer")
        if chunk_bytes < 1:
            raise ValueError("chunk_bytes must be greater than zero")
        if read_mode not in SOURCE_READ_MODES:
            raise ValueError("read_mode must be one of: " + ", ".join(sorted(SOURCE_READ_MODES)))
        self._chunk_bytes = chunk_bytes
        self._read_mode = read_mode

    @property
    def chunk_bytes(self) -> int:
        return self._chunk_bytes

    @property
    def read_mode(self) -> str:
        return self._read_mode

    def boundary(self, log_path: str | Path) -> SourceBoundary:
        path = Path(log_path)
        with path.open("rb") as handle:
            return _source_boundary(os.fstat(handle.fileno()))

    def chunks(
        self,
        log_path: str | Path,
        *,
        boundary: SourceBoundary,
        start_offset: int = 0,
    ) -> Iterator[bytes]:
        """Yield exactly the bytes in ``[start_offset, boundary.byte_size)``."""

        if start_offset < 0 or start_offset > boundary.byte_size:
            raise ValueError("start_offset is outside the captured source boundary")
        with Path(log_path).open("rb") as handle:
            self._validate_open_source(handle, boundary)
            handle.seek(start_offset)
            remaining = boundary.byte_size - start_offset
            while remaining:
                read_size = (
                    remaining
                    if self._read_mode == SOURCE_READ_MODE_SINGLE_SNAPSHOT
                    else min(self._chunk_bytes, remaining)
                )
                chunk = handle.read(read_size)
                if not chunk:
                    raise OSError(
                        "log file ended before captured boundary: "
                        f"{boundary.byte_size - remaining}/{boundary.byte_size}"
                    )
                remaining -= len(chunk)
                yield chunk

    def snapshot(self, log_path: str | Path) -> "LogSnapshot":
        """Capture one immutable boundary through the shared chunk decoder."""

        path = str(log_path)
        boundary = self.boundary(path)
        if self._read_mode == SOURCE_READ_MODE_SINGLE_SNAPSHOT:
            lines, encoding = self._decode_boundary(path, boundary)
            return LogSnapshot(
                path=path,
                lines=lines,
                byte_size=boundary.byte_size,
                source_boundary=boundary,
                encoding=encoding,
                read_mode=self._read_mode,
            )
        store, encoding = self._index_boundary(path, boundary)
        return LogSnapshot.from_line_store(
            path=path,
            line_store=store,
            byte_size=boundary.byte_size,
            source_boundary=boundary,
            encoding=encoding,
            read_mode=self._read_mode,
        )

    def _index_boundary(
        self,
        log_path: str,
        boundary: SourceBoundary,
    ) -> tuple[IndexedFileLineStore, str]:
        return self._index_boundary_with_encoding(log_path, boundary, "utf-8"), "utf-8"

    def _index_boundary_with_encoding(
        self,
        log_path: str,
        boundary: SourceBoundary,
        encoding: str,
    ) -> IndexedFileLineStore:
        decoder = IncrementalLineDecoder(encoding=encoding)
        line_starts = array("Q")
        indexed_end_offset = 0
        for chunk in self.chunks(log_path, boundary=boundary):
            for record in decoder.feed_records(chunk):
                line_starts.append(record.start_offset)
                indexed_end_offset = record.next_offset
        for record in decoder.feed_records(b"", final=True):
            line_starts.append(record.start_offset)
            indexed_end_offset = record.next_offset
        return IndexedFileLineStore(
            log_path,
            boundary=boundary,
            encoding=encoding,
            line_starts=line_starts,
            indexed_end_offset=indexed_end_offset,
            read_chunk_bytes=self._chunk_bytes,
        )

    def _decode_boundary(
        self,
        log_path: str,
        boundary: SourceBoundary,
    ) -> tuple[tuple[str, ...], str]:
        return self._decode_boundary_with_encoding(log_path, boundary, "utf-8"), "utf-8"

    def _decode_boundary_with_encoding(
        self,
        log_path: str,
        boundary: SourceBoundary,
        encoding: str,
    ) -> tuple[str, ...]:
        decoder = IncrementalLineDecoder(encoding=encoding)
        lines: list[str] = []
        for chunk in self.chunks(log_path, boundary=boundary):
            lines.extend(decoder.feed(chunk))
        lines.extend(decoder.feed(b"", final=True))
        return tuple(lines)

    @staticmethod
    def _validate_open_source(handle: BinaryIO, boundary: SourceBoundary) -> None:
        current = _source_boundary(os.fstat(handle.fileno()))
        if not boundary.same_file(current):
            raise OSError("log file was replaced after boundary capture")
        if current.byte_size < boundary.byte_size:
            raise OSError("log file was truncated after boundary capture")
        if current.byte_size == boundary.byte_size and current.mtime_ns != boundary.mtime_ns:
            raise OSError("log file was modified after boundary capture")


class LogSnapshot:
    """One immutable source boundary shared by all downstream stages."""

    def __init__(
        self,
        *,
        path: str,
        lines: Sequence[str],
        byte_size: int,
        source_boundary: SourceBoundary | None = None,
        encoding: str = "utf-8",
        read_mode: str = "provided_snapshot",
    ) -> None:
        self.path = path
        self.byte_size = byte_size
        self.source_boundary = source_boundary
        self.encoding = encoding
        self.read_mode = read_mode
        self._line_store: _LineStore = InMemoryLineStore(lines)

    @classmethod
    def from_line_store(
        cls,
        *,
        path: str,
        line_store: _LineStore,
        byte_size: int,
        source_boundary: SourceBoundary | None = None,
        encoding: str = "utf-8",
        read_mode: str = "provided_snapshot",
    ) -> "LogSnapshot":
        snapshot = cls.__new__(cls)
        snapshot.path = path
        snapshot.byte_size = byte_size
        snapshot.source_boundary = source_boundary
        snapshot.encoding = encoding
        snapshot.read_mode = read_mode
        snapshot._line_store = line_store
        return snapshot

    @classmethod
    def read(
        cls,
        log_path: str | Path,
        *,
        chunk_bytes: int = DEFAULT_LOG_READ_CHUNK_BYTES,
        read_mode: str = SOURCE_READ_MODE_CHUNKED,
    ) -> "LogSnapshot":
        return ChunkedLogReader(
            chunk_bytes=chunk_bytes,
            read_mode=read_mode,
        ).snapshot(log_path)

    @property
    def line_count(self) -> int:
        return self._line_store.line_count

    @property
    def lines(self) -> tuple[str, ...]:
        """Materialize all lines only for explicit compatibility/test callers."""

        return tuple(item.text for item in self.log_lines())

    @property
    def storage_mode(self) -> str:
        return self._line_store.storage_mode

    def log_lines(
        self,
        *,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> Iterator[LogLine]:
        yield from self._line_store.log_lines(
            start_line=start_line,
            end_line=end_line,
        )

    def line(self, line: int) -> str | None:
        return self._line_store.line(line)

    def context_before(self, line: int, *, limit: int) -> tuple[str, ...]:
        end_line = max(0, min(line - 1, self.line_count))
        start_line = max(1, end_line - limit + 1)
        return tuple(
            item.text
            for item in self.log_lines(
                start_line=start_line,
                end_line=end_line,
            )
        )

    def validate_source(self) -> None:
        """Memory snapshots are immutable and need no later source validation."""


class LogSource(Protocol):
    """Storage-neutral source for one immutable analysis snapshot."""

    @property
    def path(self) -> str: ...

    def unavailable_reason(self) -> str | None: ...

    def snapshot(self) -> LogSnapshot: ...


LogReaderFactory = Callable[[], ChunkedLogReader]


class LocalLogSource:
    """Read one analysis source from the local filesystem."""

    def __init__(
        self,
        log_path: str | Path,
        *,
        chunk_bytes: int = DEFAULT_LOG_READ_CHUNK_BYTES,
        read_mode: str = SOURCE_READ_MODE_CHUNKED,
    ) -> None:
        self._path = Path(log_path)
        self._reader = ChunkedLogReader(
            chunk_bytes=chunk_bytes,
            read_mode=read_mode,
        )

    @property
    def path(self) -> str:
        return str(self._path)

    @property
    def chunk_reader(self) -> ChunkedLogReader:
        return self._reader

    def unavailable_reason(self) -> str | None:
        if not self._path.exists():
            return f"log path is missing: {self._path}"
        if not self._path.is_file():
            return f"log path is not a file: {self._path}"
        try:
            if self._path.stat().st_size == 0:
                return f"log path is empty: {self._path}"
            with self._path.open("rb") as handle:
                handle.read(1)
        except OSError as exc:
            return f"log path is not readable: {exc}"
        return None

    def snapshot(self) -> LogSnapshot:
        return self._reader.snapshot(self._path)


def read_log_text_lines(log_path: str | Path) -> list[str]:
    """Read logs as UTF-8 while replacing malformed source bytes."""

    return list(LogSnapshot.read(log_path).lines)


def read_log_lines(log_path: str | Path) -> list[LogLine]:
    return list(LogSnapshot.read(log_path).log_lines())


def read_log_line(log_path: str | Path, line: int) -> str | None:
    return LogSnapshot.read(log_path).line(line)


def _source_boundary(stat_result: os.stat_result) -> SourceBoundary:
    return SourceBoundary(
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
        byte_size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
    )


def _decode_record(raw: bytes, encoding: str) -> str:
    if raw.endswith(b"\n"):
        raw = raw[:-1]
        if raw.endswith(b"\r"):
            raw = raw[:-1]
    return _decode_bytes(raw, encoding)[0]


def _decode_bytes(raw: bytes, encoding: str) -> tuple[str, int]:
    try:
        return raw.decode(encoding, errors="strict"), 0
    except UnicodeDecodeError:
        text = raw.decode(encoding, errors="replace")
        return text, max(1, text.count("\N{REPLACEMENT CHARACTER}"))
