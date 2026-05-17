# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launcher-side lifecycle management for nvrx-fact-agent."""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

from nvidia_resiliency_ext.attribution.fact.rpc import notify_fact_agent
from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

logger = logging.getLogger(LogConfig.name)

DEFAULT_FACT_AGENT_STARTUP_TIMEOUT = 5.0
_FACT_AGENT_STOP_TIMEOUT = 5.0
_FACT_AGENT_READY_POLL_INTERVAL = 0.1


@dataclass(frozen=True)
class FactAgentEndpoint:
    socket_path: str


class FactAgentManager:
    """Start and stop one local nvrx-fact-agent process for this launcher."""

    def __init__(
        self,
        *,
        fact_url: Optional[str],
        socket_path: Optional[str] = None,
        rpc_timeout_s: float = 2.0,
        startup_timeout_s: float = DEFAULT_FACT_AGENT_STARTUP_TIMEOUT,
        log_file: Optional[str] = None,
    ) -> None:
        self.fact_url = str(fact_url).strip() if fact_url else None
        self.socket_path = socket_path or _managed_fact_agent_socket_path()
        self.rpc_timeout_s = max(0.1, float(rpc_timeout_s))
        self.startup_timeout_s = max(0.1, float(startup_timeout_s))
        self.log_file = log_file or _managed_fact_agent_log_path()
        self.process: Optional[subprocess.Popen] = None

    @property
    def is_enabled(self) -> bool:
        return self.fact_url is not None

    def start_if_needed(self) -> Optional[FactAgentEndpoint]:
        if not self.is_enabled:
            return None

        assert self.fact_url is not None
        cmd = _fact_agent_command() + [
            "--fact-url",
            self.fact_url,
            "--socket-path",
            self.socket_path,
        ]
        logger.info(
            "Starting local nvrx-fact-agent (socket_path=%s, log_file=%s)",
            self.socket_path,
            self.log_file,
        )

        os.makedirs(os.path.dirname(self.socket_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
        log_fd = open(self.log_file, "w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(  # nosec B603
                cmd,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                shell=False,
            )
        finally:
            log_fd.close()

        try:
            self._wait_until_ready()
        except Exception:
            self.stop()
            raise

        logger.info(
            "nvrx-fact-agent is ready: PID=%s socket_path=%s",
            self.process.pid if self.process else None,
            self.socket_path,
        )
        return FactAgentEndpoint(socket_path=self.socket_path)

    def stop(self) -> None:
        proc = self.process
        if proc is None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.socket_path)
            return
        if proc.poll() is not None:
            logger.info(
                "nvrx-fact-agent PID=%s already exited with returncode=%s",
                proc.pid,
                proc.returncode,
            )
            self.process = None
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.socket_path)
            return

        logger.info("Sending SIGTERM to nvrx-fact-agent PID=%s", proc.pid)
        with contextlib.suppress(Exception):
            proc.terminate()
        try:
            proc.wait(timeout=_FACT_AGENT_STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.warning(
                "nvrx-fact-agent PID=%s did not exit within %.0fs; killing",
                proc.pid,
                _FACT_AGENT_STOP_TIMEOUT,
            )
            with contextlib.suppress(Exception):
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait()
        logger.info("nvrx-fact-agent PID=%s finished with returncode=%s", proc.pid, proc.returncode)
        self.process = None
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.socket_path)

    def _wait_until_ready(self) -> None:
        assert self.process is not None
        deadline = time.monotonic() + self.startup_timeout_s
        last_error = "not probed"
        while time.monotonic() < deadline:
            rc = self.process.poll()
            if rc is not None:
                raise RuntimeError(
                    f"nvrx-fact-agent exited before becoming ready "
                    f"(returncode={rc}, log_file={self.log_file})"
                )
            try:
                ack = notify_fact_agent(
                    socket_path=self.socket_path,
                    payload={"event": "ping"},
                    timeout_s=self.rpc_timeout_s,
                )
                if ack.get("accepted"):
                    return
                last_error = f"ping rejected: {ack}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(_FACT_AGENT_READY_POLL_INTERVAL)

        raise TimeoutError(
            f"nvrx-fact-agent did not become ready within {self.startup_timeout_s:.1f}s "
            f"at {self.socket_path} (last_error={last_error}, log_file={self.log_file})"
        )


def _fact_agent_command() -> list[str]:
    exe = shutil.which("nvrx-fact-agent")
    if exe:
        return [exe]
    return [sys.executable, "-m", "nvidia_resiliency_ext.attribution.fact.agent"]


def _managed_fact_agent_socket_path() -> str:
    return os.path.join(
        tempfile.gettempdir(),
        f"nvrx-fact-agent-{os.getuid()}-{os.getpid()}.sock",
    )


def _managed_fact_agent_log_path() -> str:
    return os.path.join(
        tempfile.gettempdir(),
        f"nvrx-fact-agent-{os.getuid()}-{os.getpid()}.log",
    )
