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
_FACT_AGENT_STOP_TIMEOUT = 10.0
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
        run_id: Optional[str] = None,
        rdzv_endpoint: Optional[str] = None,
        store_timeout_s: Optional[float] = None,
        local_node: Optional[str] = None,
        is_store_host: bool = False,
        job_id: Optional[str] = None,
        ranks_per_node: Optional[int] = None,
        username: Optional[str] = None,
        cluster: Optional[str] = None,
        health_log_prefix: Optional[str] = None,
        dmesg_artifact_enabled: bool = False,
        result_artifact_enabled: bool = False,
        grpc_server_address: Optional[str] = None,
        grpc_node_id: Optional[str] = None,
        fact_history_es_url: Optional[str] = None,
        fact_history_es_auth_file: Optional[str] = None,
        fact_history_lookback: Optional[str] = None,
        fact_history_index: Optional[str] = None,
        fact_history_max_candidate_nodes: Optional[int] = None,
        fact_history_query_timeout_s: Optional[float] = None,
        fact_min_repeat_count_for_avoid: Optional[int] = None,
        fact_max_attribution_avoids_per_cycle: Optional[int] = None,
    ) -> None:
        self.fact_url = str(fact_url).strip() if fact_url else None
        self.socket_path = socket_path or _managed_fact_agent_socket_path()
        self.rpc_timeout_s = max(0.1, float(rpc_timeout_s))
        self.startup_timeout_s = max(0.1, float(startup_timeout_s))
        self.log_file = log_file or _managed_fact_agent_log_path()
        self.run_id = str(run_id).strip() if run_id else None
        self.rdzv_endpoint = str(rdzv_endpoint).strip() if rdzv_endpoint else None
        self.store_timeout_s = float(store_timeout_s) if store_timeout_s is not None else None
        self.local_node = str(local_node).strip() if local_node else None
        self.is_store_host = bool(is_store_host)
        self.job_id = str(job_id).strip() if job_id else None
        self.ranks_per_node = max(1, int(ranks_per_node)) if ranks_per_node is not None else None
        self.username = str(username).strip() if username else None
        self.cluster = str(cluster).strip() if cluster else None
        self.health_log_prefix = str(health_log_prefix).strip() if health_log_prefix else None
        self.dmesg_artifact_enabled = bool(dmesg_artifact_enabled)
        self.result_artifact_enabled = bool(result_artifact_enabled)
        self.grpc_server_address = str(grpc_server_address).strip() if grpc_server_address else None
        self.grpc_node_id = str(grpc_node_id).strip() if grpc_node_id else None
        self.fact_history_es_url = str(fact_history_es_url).strip() if fact_history_es_url else None
        self.fact_history_es_auth_file = (
            str(fact_history_es_auth_file).strip() if fact_history_es_auth_file else None
        )
        self.fact_history_lookback = (
            str(fact_history_lookback).strip() if fact_history_lookback else None
        )
        self.fact_history_index = str(fact_history_index).strip() if fact_history_index else None
        self.fact_history_max_candidate_nodes = fact_history_max_candidate_nodes
        self.fact_history_query_timeout_s = fact_history_query_timeout_s
        self.fact_min_repeat_count_for_avoid = fact_min_repeat_count_for_avoid
        self.fact_max_attribution_avoids_per_cycle = fact_max_attribution_avoids_per_cycle
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
        cmd.extend(self._session_args())
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

        try:
            notify_fact_agent(
                socket_path=self.socket_path,
                payload={"event": "shutdown"},
                timeout_s=self.rpc_timeout_s,
            )
            logger.info("Requested graceful nvrx-fact-agent shutdown for PID=%s", proc.pid)
        except Exception as exc:
            logger.info(
                "Graceful nvrx-fact-agent shutdown request failed for PID=%s: %s",
                proc.pid,
                exc,
            )
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

    def _session_args(self) -> list[str]:
        args: list[str] = []
        if self.run_id:
            args.extend(["--run-id", self.run_id])
        if self.rdzv_endpoint:
            args.extend(["--rdzv-endpoint", self.rdzv_endpoint])
        if self.store_timeout_s is not None:
            args.extend(["--store-timeout", str(self.store_timeout_s)])
        if self.local_node:
            args.extend(["--local-node", self.local_node])
        if self.is_store_host:
            args.append("--is-store-host")
        if self.job_id:
            args.extend(["--job-id", self.job_id])
        if self.ranks_per_node is not None:
            args.extend(["--ranks-per-node", str(self.ranks_per_node)])
        if self.username:
            args.extend(["--username", self.username])
        if self.cluster:
            args.extend(["--cluster", self.cluster])
        if self.health_log_prefix:
            args.extend(["--health-log-prefix", self.health_log_prefix])
        if self.dmesg_artifact_enabled:
            args.append("--dmesg-artifact-enabled")
        if self.result_artifact_enabled:
            args.append("--result-artifact-enabled")
        if self.grpc_server_address:
            args.extend(["--grpc-server-address", self.grpc_server_address])
        if self.grpc_node_id:
            args.extend(["--grpc-node-id", self.grpc_node_id])
        if self.fact_history_es_url:
            args.extend(["--fact-history-es-url", self.fact_history_es_url])
        if self.fact_history_es_auth_file:
            args.extend(["--fact-history-es-auth-file", self.fact_history_es_auth_file])
        if self.fact_history_lookback:
            args.extend(["--fact-history-lookback", self.fact_history_lookback])
        if self.fact_history_index:
            args.extend(["--fact-history-index", self.fact_history_index])
        if self.fact_history_max_candidate_nodes is not None:
            args.extend(
                [
                    "--fact-history-max-candidate-nodes",
                    str(self.fact_history_max_candidate_nodes),
                ]
            )
        if self.fact_history_query_timeout_s is not None:
            args.extend(["--fact-history-query-timeout", str(self.fact_history_query_timeout_s)])
        if self.fact_min_repeat_count_for_avoid is not None:
            args.extend(
                [
                    "--fact-min-repeat-count-for-avoid",
                    str(self.fact_min_repeat_count_for_avoid),
                ]
            )
        if self.fact_max_attribution_avoids_per_cycle is not None:
            args.extend(
                [
                    "--fact-max-attribution-avoids-per-cycle",
                    str(self.fact_max_attribution_avoids_per_cycle),
                ]
            )
        return args


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
