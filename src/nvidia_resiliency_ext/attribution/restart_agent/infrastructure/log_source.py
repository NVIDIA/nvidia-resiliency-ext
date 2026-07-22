# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Log file read helpers shared by restart agent stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

from ..models import LogLine


@dataclass(frozen=True)
class LogSnapshot:
    """One immutable source-log snapshot shared by a pipeline invocation."""

    path: str
    lines: tuple[str, ...]
    byte_size: int

    @classmethod
    def read(cls, log_path: str | Path) -> "LogSnapshot":
        path = Path(log_path)
        return cls(
            path=str(log_path),
            lines=tuple(read_log_text_lines(path)),
            byte_size=path.stat().st_size,
        )

    def log_lines(
        self,
        *,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> Iterator[LogLine]:
        start = max(1, start_line)
        end = len(self.lines) if end_line is None else min(len(self.lines), end_line)
        for line_no in range(start, end + 1):
            yield LogLine(line=line_no, text=self.lines[line_no - 1])

    def line(self, line: int) -> str | None:
        if line < 1 or line > len(self.lines):
            return None
        return self.lines[line - 1]

    def context_before(self, line: int, *, limit: int) -> tuple[str, ...]:
        end_index = max(0, min(line - 1, len(self.lines)))
        return self.lines[max(0, end_index - limit) : end_index]


class LogSource(Protocol):
    """Storage-neutral source for one immutable analysis snapshot."""

    @property
    def path(self) -> str: ...

    def unavailable_reason(self) -> str | None: ...

    def snapshot(self) -> LogSnapshot: ...


class LocalLogSource:
    """Read one analysis source from the local filesystem."""

    def __init__(self, log_path: str | Path) -> None:
        self._path = Path(log_path)

    @property
    def path(self) -> str:
        return str(self._path)

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
        return LogSnapshot.read(self._path)


def read_log_text_lines(log_path: str | Path) -> list[str]:
    """Read logs with the same UTF-8 then Latin-1 fallback used by LogSage."""

    path = Path(log_path)
    try:
        return _read_text_lines(path, encoding="utf-8")
    except UnicodeDecodeError:
        # Latin-1 maps every byte to a code point, so malformed/mixed logs do
        # not fail analysis after UTF-8 decoding fails.
        return _read_text_lines(path, encoding="latin-1")


def read_log_lines(log_path: str | Path) -> list[LogLine]:
    return [
        LogLine(line=line_no, text=text)
        for line_no, text in enumerate(read_log_text_lines(log_path), start=1)
    ]


def read_log_line(log_path: str | Path, line: int) -> str | None:
    if line < 1:
        return None
    for line_no, text in enumerate(read_log_text_lines(log_path), start=1):
        if line_no == line:
            return text
    return None


def _read_text_lines(path: Path, *, encoding: str) -> list[str]:
    with path.open("r", encoding=encoding) as handle:
        return [line.rstrip("\n") for line in handle]
