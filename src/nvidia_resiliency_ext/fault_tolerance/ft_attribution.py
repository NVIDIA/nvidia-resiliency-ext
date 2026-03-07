# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fault-tolerance integration with attribution.

Attribution has multiple analyzer backends; this module integrates the LogAnalysis
analyzer (nvidia_resiliency_ext.attribution.log_analyzer) with the FT launcher.

Provides LogAnalysisConfig, LogAnalysisClient, and AttributionServiceClient for
invoking log analysis on the Restart & progress path (lib, mcp, or url mode).
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional
from urllib.parse import quote_plus

import httpx

from nvidia_resiliency_ext.attribution.log_analyzer.utils import attribution_no_restart
from nvidia_resiliency_ext.fault_tolerance.config import SlackConfig
from nvidia_resiliency_ext.fault_tolerance.utils import job_id_from_env, job_user_from_env
from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

logger = logging.getLogger(LogConfig.name)

# Re-export for launcher (parse attribution result → restart decision)
__all__ = [
    "LogAnalysisConfig",
    "LogAnalysisClient",
    "LogAnalysisMode",
    "AttributionServiceClient",
    "SlackConfig",
    "attribution_no_restart",
]

# --- Config ---
LogAnalysisMode = Literal["lib", "mcp", "url"]


def _validate_attribution_url(url: str) -> str:
    """Validate attribution URL and return normalized form (with scheme if missing)."""
    if not url or not url.strip():
        raise ValueError("--ft-attribution-loganalysis URL must be non-empty")
    s = url.strip()
    if "://" in s:
        if not s.startswith(("http://", "https://")):
            raise ValueError(f"--ft-attribution-loganalysis: expected http(s) URL, got: {url!r}")
        return s
    if ":" in s:
        return f"http://{s}"
    raise ValueError(
        f"--ft-attribution-loganalysis: expected host:port or http(s)://host:port, got: {url!r}"
    )


@dataclass(frozen=True)
class LogAnalysisConfig:
    """Configuration for log analysis invocation on the Restart & progress path.

    Use mode ``lib`` (in-process), ``mcp`` (MCP subprocess), or ``url`` (HTTP service).
    When mode is ``url``, attribution_service_url must be set (e.g. http://host:8000).
    user and job_id are read from env by LogAnalysisClient (SLURM_JOB_USER, SLURM_*_JOB_ID).
    slack: SlackConfig for lib/mcp alerts; reuses FaultToleranceConfig.slack when provided.
    dataflow_index: Elasticsearch index for lib/mcp posting; reuses FaultToleranceConfig.dataflow_index.
    """

    mode: LogAnalysisMode
    attribution_service_url: Optional[str] = None
    timeout_seconds: int = 60
    slack: Optional[SlackConfig] = None
    dataflow_index: Optional[str] = None

    @property
    def use_lib(self) -> bool:
        return self.mode == "lib"

    @property
    def use_mcp(self) -> bool:
        return self.mode == "mcp"

    @property
    def use_url(self) -> bool:
        return self.mode == "url"

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "mode": self.mode,
            "attribution_service_url": self.attribution_service_url,
            "timeout_seconds": self.timeout_seconds,
        }
        if self.slack is not None:
            d["slack"] = self.slack.to_dict()
        if self.dataflow_index is not None:
            d["dataflow_index"] = self.dataflow_index
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LogAnalysisConfig":
        if not d:
            return cls(mode="lib", timeout_seconds=60)
        return cls(
            mode=d.get("mode", "lib"),
            attribution_service_url=d.get("attribution_service_url"),
            timeout_seconds=int(d.get("timeout_seconds", 60)),
            slack=SlackConfig.from_dict(d.get("slack")),
            dataflow_index=d.get("dataflow_index"),
        )

    @classmethod
    def from_ft_cli_value(
        cls,
        val: str,
        timeout_seconds: int = 60,
        slack: Optional[SlackConfig] = None,
        dataflow_index: Optional[str] = None,
    ) -> "LogAnalysisConfig":
        """Build from CLI string (lib, mcp, or URL). user/job_id read by LogAnalysisClient from env."""
        v = val.strip().lower()
        if v == "lib":
            return cls(
                mode="lib",
                timeout_seconds=timeout_seconds,
                slack=slack,
                dataflow_index=dataflow_index,
            )
        if v == "mcp":
            return cls(
                mode="mcp",
                timeout_seconds=timeout_seconds,
                slack=slack,
                dataflow_index=dataflow_index,
            )
        url = _validate_attribution_url(v)
        return cls(
            mode="url",
            attribution_service_url=url,
            timeout_seconds=timeout_seconds,
            slack=slack,
            dataflow_index=dataflow_index,
        )


# --- HTTP client (URL mode) ---
class AttributionServiceClient:
    """
    HTTP client for the attribution service (URL mode).
    Talks to nvrx_attrsvc AttributionService over HTTP.
    """

    def __init__(self, base_url: str, timeout_seconds: float = 60.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = max(1.0, float(timeout_seconds))

    def path_notify(self, log_path: str) -> None:
        """Notify path before workers start (fire-and-forget POST)."""
        threading.Thread(
            target=self._do_submit_log,
            args=(log_path,),
            daemon=True,
        ).start()

    def _do_submit_log(self, log_path: str) -> None:
        try:
            with httpx.Client(timeout=10.0) as client:
                url = f"{self._base_url}/logs"
                logger.debug("AttributionServiceClient POST: %s (log_path=%s)", url, log_path)
                client.post(
                    url,
                    json={"log_path": log_path},
                    headers={"accept": "application/json"},
                )
        except Exception as e:
            logger.warning(
                "AttributionServiceClient POST %s failed: %s: %s", log_path, type(e).__name__, e
            )

    def get_result_sync(self, log_path: str) -> Optional[Dict[str, Any]]:
        """Get analysis results via GET (blocking). Uses client timeout."""
        if not log_path:
            return None
        try:
            with httpx.Client(timeout=self._timeout) as client:
                q_path = quote_plus(log_path)
                url = f"{self._base_url}/logs?log_path={q_path}"
                logger.debug("AttributionServiceClient GET: %s (log_path=%s)", url, log_path)
                resp = client.get(url, headers={"accept": "application/json"})
                if resp.status_code == 200:
                    payload = resp.json() if resp.text else {}
                    result = payload.get("result", payload)
                    if isinstance(result, dict):
                        return result
                    return {"result": result} if result is not None else None
                logger.warning(
                    "AttributionServiceClient GET for %s returned %d", log_path, resp.status_code
                )
                return None
        except Exception as e:
            logger.warning(
                "AttributionServiceClient GET %s failed: %s: %s", log_path, type(e).__name__, e
            )
            return None


# --- Client (selects lib / mcp / url backend) ---
class LogAnalysisClient:
    """Client for log analysis attribution. Chooses backend from config."""

    def __init__(self, config: LogAnalysisConfig) -> None:
        self._config = config
        self._timeout = max(1, config.timeout_seconds)
        self._user = job_user_from_env()
        self._job_id = job_id_from_env()
        self._fetch_result: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None
        self._path_notify: Optional[Callable[[str], None]] = None
        self._init_backend()

    def _init_backend(self) -> None:
        if self._config.use_lib or self._config.use_mcp:
            from nvidia_resiliency_ext.attribution.log_analyzer.runner import (
                ensure_analyzer_ready,
                run_log_analysis_sync,
            )
            from nvidia_resiliency_ext.attribution.postprocessing import (
                configure_postprocessing_resolved,
            )

            # Postprocessing: centralized config (Slack from token file or env; dataflow; cluster from SLURM)
            slack_cfg = self._config.slack
            slack_token = slack_cfg.bot_token if slack_cfg else None
            slack_channel = slack_cfg.channel if slack_cfg else None
            dataflow_index = (self._config.dataflow_index or "").strip()

            configure_postprocessing_resolved(
                cluster_name="",
                dataflow_index=dataflow_index,
                slack_token=slack_token,
                slack_channel=slack_channel,
                cluster_name_env="SLURM_CLUSTER_NAME",
                create_dataflow_poster_if_needed=True,
            )

            if not ensure_analyzer_ready(
                timeout_seconds=self._timeout, use_lib_log_analysis=self._config.use_lib
            ):
                self._fetch_result = None
                return

            user = self._user
            job_id = self._job_id

            def fetch(log_path: str) -> Optional[Dict[str, Any]]:
                return run_log_analysis_sync(log_path, user=user, job_id=job_id)

            self._fetch_result = fetch
        elif self._config.use_url and self._config.attribution_service_url:
            attr_svc = AttributionServiceClient(
                base_url=self._config.attribution_service_url,
                timeout_seconds=self._timeout,
            )

            def fetch(log_path: str) -> Optional[Dict[str, Any]]:
                return attr_svc.get_result_sync(log_path)

            self._fetch_result = fetch
            self._path_notify = attr_svc.path_notify

    def fetch_result(self, log_path: str) -> Optional[Dict[str, Any]]:
        """Run log analysis and return result. None on skip/timeout/error.
        Timeout from config (set at init)."""
        if self._fetch_result is None:
            return None
        return self._fetch_result(log_path)

    def should_stop(self, log_path: str) -> bool:
        """Return True if attribution recommends stop (no restart), False to restart.
        Wraps fetch_result + attribution_no_restart."""
        attr_result = self.fetch_result(log_path)
        return attribution_no_restart(attr_result)

    @property
    def path_notify(self) -> Optional[Callable[[str], None]]:
        """Notify path before workers start (URL mode only; None for lib/mcp)."""
        return self._path_notify
