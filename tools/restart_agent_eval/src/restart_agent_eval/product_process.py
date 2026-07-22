"""Process boundary for invoking the Restart Agent product CLI."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence, TextIO

PRODUCT_CLI_MODULE = "nvidia_resiliency_ext.attribution.restart_agent.cli"


def product_cli_command(python: str, log_path: str | Path) -> list[str]:
    return [python, "-m", PRODUCT_CLI_MODULE, str(log_path)]


@dataclass(frozen=True)
class ProcessResult:
    """Harness-owned result for one completed external command."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class ProcessTimeoutError(TimeoutError):
    """An external process did not stop within the requested interval."""


class RunningProcess(Protocol):
    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class ProcessExecutor(Protocol):
    """Replaceable operating-system process adapter."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult: ...

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdout: TextIO,
        stderr: TextIO,
    ) -> RunningProcess: ...


class SubprocessExecutor:
    """Production process adapter backed by :mod:`subprocess`."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        normalized_command = tuple(command)
        completed = subprocess.run(
            list(normalized_command),
            cwd=cwd,
            env=dict(env) if env is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return ProcessResult(
            command=normalized_command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdout: TextIO,
        stderr: TextIO,
    ) -> RunningProcess:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(env),
            text=True,
            stdout=stdout,
            stderr=stderr,
        )
        return _SubprocessHandle(process)


class _SubprocessHandle:
    """Translate subprocess-specific timeout errors at the adapter boundary."""

    def __init__(self, process: subprocess.Popen[str]) -> None:
        self._process = process

    def poll(self) -> int | None:
        return self._process.poll()

    def wait(self, timeout: float | None = None) -> int:
        try:
            return self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise ProcessTimeoutError(str(exc)) from exc

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self._process.kill()
