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

Integrates :mod:`nvidia_resiliency_ext.attribution.log_analyzer` with ``ft_launcher``.
Supports **multiple backends** per run: e.g. ``mcp`` (LogSage + FR via MCP) and one or more
HTTP URLs (third-party or attrsvc). Restart/stop uses **any** backend that reports
``STOP`` / no-restart (:func:`attribution_no_restart`).
"""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import httpx

from nvidia_resiliency_ext.attribution.log_analyzer.llm_output import attribution_no_restart
from nvidia_resiliency_ext.fault_tolerance.config import SlackConfig
from nvidia_resiliency_ext.fault_tolerance.utils import job_id_from_env, job_user_from_env
from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

logger = logging.getLogger(LogConfig.name)

__all__ = [
    "AttributionRunConfig",
    "LogAnalysisClient",
    "AttributionServiceClient",
    "SlackConfig",
    "attribution_no_restart",
    "dedupe_attribution_backends",
    "validate_backend_entry",
]


def _validate_attribution_url(url: str) -> str:
    """Validate attribution URL and return normalized form (with scheme if missing)."""
    if not url or not url.strip():
        raise ValueError("attribution backend URL must be non-empty")
    s = url.strip()
    if "://" in s:
        if not s.startswith(("http://", "https://")):
            raise ValueError(f"attribution backend: expected http(s) URL, got: {url!r}")
        return s
    if ":" in s:
        return f"http://{s}"
    raise ValueError(
        f"attribution backend: expected host:port or http(s)://host:port, got: {url!r}"
    )


def validate_backend_entry(entry: str) -> str:
    """Normalize one backend: ``mcp`` or a validated HTTP URL string."""
    v = entry.strip()
    if not v:
        raise ValueError("empty attribution backend entry")
    low = v.lower()
    if low == "lib":
        raise ValueError("attribution from ft_launcher requires 'mcp' or an HTTP URL")
    if low == "mcp":
        return "mcp"
    return _validate_attribution_url(v)


def dedupe_attribution_backends(backends: List[str]) -> List[str]:
    """Deduplicate backend list: one ``mcp``, unique URLs (order preserved)."""
    out: List[str] = []
    seen_mcp = False
    seen_urls: set = set()
    for raw in backends:
        b = validate_backend_entry(raw)
        if b == "mcp":
            if not seen_mcp:
                seen_mcp = True
                out.append("mcp")
            else:
                logger.debug("duplicate mcp backend entry ignored")
        else:
            if b not in seen_urls:
                seen_urls.add(b)
                out.append(b)
    return out


@dataclass(frozen=True)
class AttributionRunConfig:
    """FT attribution: one or more backends (``mcp``, HTTP URL(s)).

    Each backend is either the literal ``mcp`` (LogSage + optional FR via MCP) or an HTTP
    base URL for a log/attribution service. :class:`LogAnalysisClient` queries backends in
    order; :meth:`~LogAnalysisClient.should_stop` is True if **any** backend says do not restart.

    ``llm_api_key_file`` (optional): if set, :class:`LogAnalysisClient` sets ``LLM_API_KEY_FILE`` before MCP init.
    """

    backends: Tuple[str, ...]
    timeout_seconds: int = 60
    slack: Optional[SlackConfig] = None
    dataflow_index: Optional[str] = None
    llm_api_key_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "backends": list(self.backends),
            "timeout_seconds": self.timeout_seconds,
        }
        if self.slack is not None:
            d["slack"] = self.slack.to_dict(include_secrets=True)
        if self.dataflow_index is not None:
            d["dataflow_index"] = self.dataflow_index
        if self.llm_api_key_file is not None:
            d["llm_api_key_file"] = self.llm_api_key_file
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AttributionRunConfig":
        """Load from serialized dict. Requires a non-empty ``backends`` list (same shape as :meth:`to_dict`)."""
        raw = d.get("backends") if d else None
        if not raw:
            raise ValueError(
                "attribution_config requires non-empty 'backends' (e.g. ['mcp'] or HTTP URL strings)"
            )
        backends = tuple(validate_backend_entry(x) for x in raw)
        return cls(
            backends=backends,
            timeout_seconds=int(d.get("timeout_seconds", 60)),
            slack=SlackConfig.from_dict(d.get("slack")),
            dataflow_index=d.get("dataflow_index"),
            llm_api_key_file=d.get("llm_api_key_file"),
        )

    @classmethod
    def from_backend_strings(
        cls,
        entries: List[str],
        timeout_seconds: int = 60,
        slack: Optional[SlackConfig] = None,
        dataflow_index: Optional[str] = None,
        llm_api_key_file: Optional[str] = None,
    ) -> "AttributionRunConfig":
        """Build from CLI/YAML string list (``mcp`` and/or URLs)."""
        norm = dedupe_attribution_backends(entries)
        if not norm:
            raise ValueError("at least one attribution backend is required")
        return cls(
            backends=tuple(norm),
            timeout_seconds=timeout_seconds,
            slack=slack,
            dataflow_index=dataflow_index,
            llm_api_key_file=llm_api_key_file,
        )


# --- HTTP client (URL backend) ---
class AttributionServiceClient:
    """
    HTTP client for an attribution service (URL backend).
    Talks to HTTP APIs that expose log analysis results (e.g. nvrx_attrsvc).
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


class LogAnalysisClient:
    """Run cycle attribution across one or more backends (MCP + HTTP URL(s))."""

    def __init__(self, config: AttributionRunConfig) -> None:
        self._config = config
        self._timeout = max(1, config.timeout_seconds)
        self._user = job_user_from_env()
        self._job_id = job_id_from_env()
        self._fetchers: List[Callable[[str], Optional[Dict[str, Any]]]] = []
        self._path_notify_fns: List[Callable[[str], None]] = []
        self._init_backends()

    def _init_backends(self) -> None:
        from nvidia_resiliency_ext.attribution.log_analyzer.runner import (
            ensure_analyzer_ready,
            notify_log_path_sync,
            run_log_analysis_sync,
        )
        from nvidia_resiliency_ext.attribution.postprocessing import config as pp_config
        from nvidia_resiliency_ext.attribution.postprocessing import configure_from_env

        slack_cfg = self._config.slack
        if slack_cfg is None:
            slack_token_arg: Optional[str] = None
            slack_channel_arg: Optional[str] = None
        else:
            slack_token_arg = slack_cfg.bot_token
            if slack_token_arg is not None:
                slack_token_arg = slack_token_arg.strip()
            slack_channel_arg = slack_cfg.channel
            if slack_channel_arg is not None:
                slack_channel_arg = slack_channel_arg.strip()
        df_idx = (self._config.dataflow_index or "").strip()
        cluster = (os.getenv("SLURM_CLUSTER_NAME") or "").strip()
        configure_from_env(
            slack_token=slack_token_arg,
            slack_channel=slack_channel_arg,
            dataflow_index=df_idx,
            cluster_name=cluster,
        )
        llm_key_file = (self._config.llm_api_key_file or "").strip()
        if llm_key_file:
            os.environ["LLM_API_KEY_FILE"] = llm_key_file
            logger.debug("FT attribution: set LLM_API_KEY_FILE from fault_tolerance config")
        if pp_config.slack_bot_token:
            logger.info(
                "Slack notifications enabled for FT attribution (channel=%s)",
                pp_config.slack_channel or "(none)",
            )
        if df_idx:
            logger.info(
                "FT attribution: dataflow posting enabled (index=%s, cluster=%s)",
                df_idx,
                cluster or "(unset)",
            )

        mcp_initialized = False
        for b in self._config.backends:
            if b == "mcp":
                if mcp_initialized:
                    continue
                if not ensure_analyzer_ready(
                    timeout_seconds=float(self._timeout), use_lib_log_analysis=False
                ):
                    logger.warning("FT attribution: MCP analyzer not ready; skipping mcp backend")
                    continue
                mcp_initialized = True
                logger.info(
                    "FT attribution: MCP backend — nvrx-mcp-analysis (log + FR when discoverable)"
                )
                user = self._user
                job_id = self._job_id

                def _fetch_mcp(log_path: str, u=user, j=job_id) -> Optional[Dict[str, Any]]:
                    return run_log_analysis_sync(
                        log_path,
                        user=u,
                        job_id=j,
                        timeout_seconds=float(self._timeout),
                        use_lib_log_analysis=False,
                    )

                self._fetchers.append(_fetch_mcp)

                def _path_notify_mcp(
                    log_path: str, u=user, j=job_id, to=float(self._timeout)
                ) -> None:
                    def _run() -> None:
                        try:
                            notify_log_path_sync(
                                log_path,
                                user=u,
                                job_id=j,
                                timeout_seconds=to,
                                use_lib_log_analysis=False,
                            )
                        except Exception as e:
                            logger.warning(
                                "FT attribution: MCP path_notify failed: %s: %s",
                                type(e).__name__,
                                e,
                            )

                    threading.Thread(
                        target=_run,
                        daemon=True,
                        name="ft-attr-mcp-path-notify",
                    ).start()

                self._path_notify_fns.append(_path_notify_mcp)
            else:
                svc = AttributionServiceClient(base_url=b, timeout_seconds=float(self._timeout))

                def _fetch_url(
                    log_path: str, client: AttributionServiceClient = svc
                ) -> Optional[Dict[str, Any]]:
                    return client.get_result_sync(log_path)

                self._fetchers.append(_fetch_url)
                self._path_notify_fns.append(svc.path_notify)

        if not self._fetchers:
            logger.warning("FT attribution: no usable backends; attribution disabled")

    def fetch_result(self, log_path: str) -> Optional[Dict[str, Any]]:
        """Return the first non-None result across backends (for debugging); prefer :meth:`should_stop`."""
        for fetch in self._fetchers:
            r = fetch(log_path)
            if r is not None:
                return r
        return None

    def should_stop(self, log_path: str) -> bool:
        """True if **any** backend recommends do not restart.

        Backends are queried **in parallel** so wall-clock time is roughly the slowest
        fetch (each fetch already applies its own timeout), not the sum of all backends.
        """
        fetchers = self._fetchers
        if not fetchers:
            return False

        def _safe_fetch(
            fetch: Callable[[str], Optional[Dict[str, Any]]]
        ) -> Optional[Dict[str, Any]]:
            try:
                return fetch(log_path)
            except Exception as e:
                logger.debug(
                    "FT attribution: should_stop backend fetch failed: %s: %s",
                    type(e).__name__,
                    e,
                )
                return None

        if len(fetchers) == 1:
            return attribution_no_restart(_safe_fetch(fetchers[0]))

        ex = ThreadPoolExecutor(
            max_workers=len(fetchers),
            thread_name_prefix="ft-attr-should-stop",
        )
        try:
            futures = [ex.submit(_safe_fetch, f) for f in fetchers]
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                except Exception:
                    r = None
                if attribution_no_restart(r):
                    return True
            return False
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    @property
    def path_notify(self) -> Optional[Callable[[str], None]]:
        """Chain backends' early path notify (fire-and-forget): MCP submit-only, then HTTP POST /logs."""
        if not self._path_notify_fns:
            return None

        def _notify(log_path: str) -> None:
            for fn in self._path_notify_fns:
                fn(log_path)

        return _notify
